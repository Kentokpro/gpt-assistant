import logging
from fastapi import FastAPI, Request, Depends, HTTPException, APIRouter, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Any
import os
import re
import httpx
import uuid
import redis.asyncio as aioredis
import time
from backend.utils.passwords import generate_password
from sqlalchemy import select, update
from fastapi_users.password import PasswordHelper

print("DEBUG: Any imported:", Any)
print("DEBUG: MAIN.PY location:", __file__)
print("DEBUG: Working directory:", os.getcwd())

redis = aioredis.from_url("redis://localhost:6379", decode_responses=True)
password_helper = PasswordHelper()

async def send_to_telegram_bot(email, phone, password):
    async with httpx.AsyncClient() as client:
        await client.post(
            "http://tg-bot-endpoint/send-login",  # заменить на реальный endpoint!
            json={"email": email, "phone": phone, "password": password}
        )

def get_or_create_session_id(request: Request) -> str:
    session_id = request.cookies.get("sessionid")
    try:
        if session_id and uuid.UUID(session_id):
            return session_id
    except Exception:
        pass
    return str(uuid.uuid4())

def is_valid_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\+7\d{10}", phone))

def is_valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+", email))

def extract_code(text: str):
    match = re.search(r"\b\d{6}\b", text)
    return match.group() if match else None

from backend.auth import (
    fastapi_users, auth_backend, require_active_subscription, current_active_user_optional
)
from backend.config import (
    DEBUG, LOG_LEVEL, ADMIN_EMAIL, GA_MEASUREMENT_ID, METRIKA_ID, SUPPORT_EMAIL
)
from backend.email_utils import send_email
from backend.openai_utils import ask_openai
from backend.models import User, Message, Session as SessionModel
from backend.schemas import UserRead, UserCreate, ChatRequest, SupportRequest
from backend.database import SessionLocal
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI(
    title="Leadinc AI Assistant",
    description="AI SaaS Assistant (B2B)",
    debug=DEBUG,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

LOG_PATH = "/root/ai-assistant/backend/leadinc-backend.log"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("leadinc-backend")
logger.info("Leadinc backend стартовал успешно!")

ALLOWED_ORIGINS = [
    "https://leadinc.ru",
    "https://gpt.leadinc.ru",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_DIR = Path(__file__).parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_register_router(
        user_schema=UserRead,
        user_create_schema=UserCreate
    ),
    prefix="/auth",
    tags=["auth"],
)

ai_router = APIRouter(prefix="/ai", tags=["ai"])

MESSAGE_LIMITS = [15, 20, 20]   # Этапы 1-3
PROJECT_LIMIT_PER_DAY = 5
USER_LIMIT = 30                 # Этап 4: 30 сообщений за 10 минут

def current_timestamp():
    return int(time.time())

def ten_minutes_ago():
    return int(time.time()) - 600

@ai_router.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    user: User = Depends(current_active_user_optional),
    session: AsyncSession = Depends(lambda: SessionLocal()),
):
    content = payload.content
    msg_type = payload.type
    session_id = get_or_create_session_id(request)

    client_ip = request.client.host
    ip_sessions_key = f"sessions_per_ip:{client_ip}:{time.strftime('%Y-%m-%d')}"
    current_sessions = int(await redis.get(ip_sessions_key) or 0)

    session_db_obj = await session.execute(
        select(SessionModel).where(SessionModel.id == session_id)
    )
    session_obj = session_db_obj.scalar_one_or_none()
    if not session_obj:
        if current_sessions >= 5:
            return JSONResponse({
                "reply": "Лимит сессий превышен. Обратитесь в поддержку.",
                "meta": {"stage": 1, "reason": "ip_session_limit"}
            })
        await redis.incr(ip_sessions_key)
        await redis.expire(ip_sessions_key, 86400)
        session.add(SessionModel(id=session_id))
        await session.commit()

    lim_prefix = f"{user.id}" if user else session_id
    logger.info(f"SessionID from cookie: {request.cookies.get('sessionid')}, Using: {session_id}")
    stage_key = f"stage:{session_id}"

    try:
        stage = int(await redis.get(stage_key) or 1)
    except Exception:
        stage = 1

    msg_count_key = f"msg_count:{lim_prefix}:stage{stage}"
    msg_count = int(await redis.get(msg_count_key) or 0)

    ip_flood_key = f"code_flood:{client_ip}"
    flood_attempts = int(await redis.get(ip_flood_key) or 0)
    if flood_attempts > 15:  # максимум 15 попыток на IP за 10 минут
        return JSONResponse({
            "reply": "Слишком много попыток ввода кода с вашего IP. Подождите 10 минут и попробуйте снова.",
            "meta": {"stage": stage, "reason": "ip_flood"}
        })

    user_code = extract_code(content)
    if stage == 1 and user_code:
        await redis.incr(ip_flood_key)
        await redis.expire(ip_flood_key, 600)
        is_real_code = await redis.exists(f"real_code:{user_code}")
        is_used_code = await redis.exists(f"code_used:{user_code}")
        if not is_real_code:
            return JSONResponse({
                "reply": "Введённый код не найден. Пожалуйста, запросите новый код в Telegram-боте.",
                "meta": {"stage": 1}
            })
        if is_used_code:
            return JSONResponse({
                "reply": "Этот код уже был использован. Запросите новый код в Telegram-боте.",
                "meta": {"stage": 1}
            })
        await redis.set(stage_key, 2, ex=12*60*60)
        await redis.set(f"code_used:{user_code}", 1, ex=86400)
        await redis.delete(f"real_code:{user_code}")
        for n in range(1, 5):
            if n != 2:
                await redis.delete(f"msg_count:{lim_prefix}:stage{n}")
        return JSONResponse({
            "reply": "Код принят! Теперь расскажи о своём бизнесе и регионе (город/область), чтобы подобрать клиентов.",
            "meta": {"stage": 2}
        })

    # --- Переход с этапа 2 на 3 (ниша и город указаны) ---
    if stage == 2:
        has_niche = any(x in content.lower() for x in ["штукатурка", "курсы", "обучение", "натяжные потолки"])
        has_city = any(x in content.lower() for x in ["москва", "спб", "казань", "регион", "город"])
        if has_niche and has_city:
            await redis.set(stage_key, 3, ex=12*60*60)
            for n in range(1, 5):
                if n != 3:
                    await redis.delete(f"msg_count:{lim_prefix}:stage{n}")
            return JSONResponse({
                "reply": "Принято! Теперь для полного доступа введи номер телефона и e-mail (пример: +79998887766 example@mail.ru).",
                "meta": {"stage": 3}
            })

    # -- Проектные лимиты только для авторизованных --
    if user:
        today = time.strftime("%Y-%m-%d")
        project_limit_key = f"project_count:{user.id}:{today}"
        project_count = int(await redis.get(project_limit_key) or 0)
        if project_count >= PROJECT_LIMIT_PER_DAY:
            resp = JSONResponse({
                "reply": "На сегодня вы уже создали 5 проектов. Следующий можно будет создать завтра или по запросу через поддержку.",
                "meta": {"stage": 4, "reason": "project_limit"}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp

        # --- Лимит 30 сообщений за 10 минут (Redis ZSET) ---
        zset_key = f"msg_zset:{user.id}"
        now = current_timestamp()
        await redis.zremrangebyscore(zset_key, 0, ten_minutes_ago())
        import uuid
        await redis.zadd(zset_key, {str(uuid.uuid4()): now})
        msg_in_window = await redis.zcount(zset_key, ten_minutes_ago(), now)
        if msg_in_window > USER_LIMIT:
            resp = JSONResponse({
                "reply": "Вы слишком активно отправляете сообщения. Пожалуйста, сделайте паузу.",
                "meta": {"stage": 4, "reason": "msg_limit"}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp

    else:
        if stage == 1 and msg_count >= MESSAGE_LIMITS[0]:
            await redis.delete(msg_count_key)
            resp = JSONResponse({
                "reply": "Достигнут лимит сообщений. Для продолжения введите 6-значный код подтверждения из Telegram.",
                "meta": {"stage": 1}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp
        elif stage == 2 and msg_count >= sum(MESSAGE_LIMITS[:2]):
            await redis.delete(msg_count_key)
            resp = JSONResponse({
                "reply": "Достигнут лимит сообщений. Пожалуйста, завершите регистрацию.",
                "meta": {"stage": 2}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp
        elif stage == 3 and msg_count >= sum(MESSAGE_LIMITS):
            await redis.delete(msg_count_key)
            resp = JSONResponse({
                "reply": "Достигнут лимит сообщений. Для продолжения требуется регистрация.",
                "meta": {"stage": 3}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp

    # --- Инкремент счетчика сообщений и TTL (для этапов 1-3) ---
    if not user:
        await redis.incr(msg_count_key)
        await redis.expire(msg_count_key, 600)

    # --- Защита от флуд-атаки для гостей ---
    if not user:
        zset_key = f"guest_flood:{session_id}"
        now = current_timestamp()
        await redis.zremrangebyscore(zset_key, 0, ten_minutes_ago())
        await redis.zadd(zset_key, {str(now): now})
        guest_msgs = await redis.zcount(zset_key, ten_minutes_ago(), now)
        if guest_msgs > 20:
            resp = JSONResponse({
                "reply": "Слишком много сообщений за короткое время. Попробуйте позже.",
                "meta": {"stage": stage, "reason": "flood"}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp

    # --- РЕГИСТРАЦИЯ и АВТО-ЛОГИН (переход на этап 4) ---
    if stage == 3:
        parts = content.split()
        phone = next((p for p in parts if p.startswith("+7")), None)
        email = next((p for p in parts if "@" in p), None)
        valid_phone = is_valid_phone(phone) if phone else False
        valid_email = is_valid_email(email) if email else False

        attempts_key = f"register:{lim_prefix}:fail_count"
        attempts = int(await redis.get(attempts_key) or 0)
        if attempts >= 10:
            reg_response = JSONResponse({"reply": "Регистрация заблокирована. Слишком много ошибок. Попробуйте позже.", "meta": {}})
            reg_response.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return reg_response

        if valid_phone and valid_email:
            password = generate_password(8)
            password_hash = password_helper.hash(password)
            q = select(User).where((User.phone == phone) | (User.email == email))
            result = await session.execute(q)
            existing = result.scalar_one_or_none()
            if existing:
                reg_response = JSONResponse({"reply": "Этот телефон или почта уже зарегистрированы.", "meta": {}})
                reg_response.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
                return reg_response
            user_obj = User(
                email=email,
                phone=phone,
                hashed_password=password_hash,
                is_active=True,
                is_verified=True,
            )
            session.add(user_obj)
            await session.commit()

            try:
                await send_to_telegram_bot(email, phone, password)
                logger.info(f"Отправлено в ТГ-бот: {email}, {phone}, {password}")
            except Exception as e:
                logger.error(f"Ошибка отправки в ТГ-бот: {e}")

            await session.execute(
                update(Message)
                .where(Message.session_id == session_id, Message.user_id == None)
                .values(user_id=user_obj.id)
            )
            await session.commit()

            from fastapi_users.authentication import JWTStrategy
            jwt_strategy = fastapi_users.get_jwt_strategy()
            token = await jwt_strategy.write_token(user_obj)

            await redis.set(stage_key, 4, ex=12*60*60)
            for n in range(1, 5):
                if n != 4:
                    await redis.delete(f"msg_count:{lim_prefix}:stage{n}")

            await redis.delete(attempts_key)
            today = time.strftime("%Y-%m-%d")
            await redis.set(f"project_count:{user_obj.id}:{today}", 0, ex=86400)

            reg_response = JSONResponse({
                "reply": "Регистрация завершена! Вы автоматически авторизованы. Пароль отправлен в Telegram.",
                "token": token,
                "meta": {"stage": 4}
            })
            reg_response.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            logger.info(f"Generated password for {email}: {password}")
            return reg_response
        else:
            await redis.incr(attempts_key)
            await redis.expire(attempts_key, 3600)
            reg_response = JSONResponse({"reply": "Некорректный телефон или почта.", "meta": {}})
            reg_response.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return reg_response

    # --- Остальная логика ассистента: OpenAI, сохранение сообщений, инкремент проектов ---
    try:
        openai_result = await ask_openai(content, msg_type)
        reply = openai_result["text"]
        usage = openai_result["usage"]
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI ошибка")

    try:
        user_id = user.id if user else None
        user_msg = Message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            type=msg_type,
            status="ok",
            content=content,
            meta={},
        )
        session.add(user_msg)
        await session.commit()
        assistant_msg = Message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            type="text",
            status="ok",
            content=reply,
            meta=usage,
        )
        session.add(assistant_msg)
        await session.commit()

        if user and ("создать проект" in content.lower() or "новый проект" in content.lower()):
            today = time.strftime("%Y-%m-%d")
            project_limit_key = f"project_count:{user.id}:{today}"
            await redis.incr(project_limit_key)
            await redis.expire(project_limit_key, 86400)
    except Exception as e:
        logger.error(f"DB error: {str(e)}")

    response_data = {"reply": reply, "meta": usage, "stage": stage}

    def is_dashboard_query(content: str) -> bool:
        key_phrases = [
            "аналитика", "покажи аналитику", "какая аналитика",
            "статистика", "дай данные", "какая динамика", "dashboard",
            "есть инфа по", "есть информация по", "покажи спрос"
        ]
        lc = content.lower()
        return any(k in lc for k in key_phrases)

    if is_dashboard_query(content):
        response_data["reply"] = "Данные по вашему запросу отображены на дашборде."
        response_data["dashboard"] = {
            "table": [
                {"№": "1", "Ключевой запрос": "вездеход", "упоминаний": "8", "Категория": "Вездеходы"},
                {"№": "2", "Ключевой запрос": "профи", "упоминаний": "5", "Категория": "Снегоходы"},
                {"№": "3", "Ключевой запрос": "мтлб", "упоминаний": "4", "Категория": "Спецтехника"}
            ],
            "analytics": (
                "Самые популярные поисковые запросы — «вездеход» и «профи», что говорит о высоком спросе на универсальные машины и брендированные решения.\n"
                "По категориям лидируют классические вездеходы, болотоходы и спецтехника, заметна растущая доля электромоделей и квадроциклов.\n"
                "Среди брендов доминирует «профи», есть рост DIY-платформ (самодельные, гусеничные)."
            )
        }

    response = JSONResponse(response_data)
    # -- Гостю сетим UUID-сессию только если её не было --
    if not request.cookies.get("sessionid"):
        response.set_cookie(
            key="sessionid",
            value=session_id,
            max_age=12 * 60 * 60,
            httponly=True,
            secure=False,
            samesite="Lax"
        )
    return response

@ai_router.post("/support")
async def support_request(
    payload: SupportRequest,
    user=Depends(require_active_subscription)
):
    await send_email(
        to=SUPPORT_EMAIL,
        subject=f"[SUPPORT] {payload.subject}",
        body=payload.message,
        from_email=user.email,
    )
    return {"status": "sent"}

@app.middleware("http")
async def add_analytics_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-GA-Measurement-Id"] = GA_MEASUREMENT_ID or ""
    response.headers["X-Metrika-Id"] = METRIKA_ID or ""
    return response

app.include_router(ai_router)
