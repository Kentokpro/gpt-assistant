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
            "http://tg-bot-endpoint/send-login",
            json={"email": email, "phone": phone, "password": password}
        )

def get_or_create_session_id(request: Request) -> str:
    session_id = request.cookies.get("sessionid")
    if session_id:
        return session_id
    return str(uuid.uuid4())

def is_valid_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\+7\d{10}", phone))

def is_valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+", email))

from backend.auth import (
    fastapi_users, auth_backend, require_active_subscription, current_active_user_optional
)
from backend.config import (
    DEBUG, LOG_LEVEL, ADMIN_EMAIL, GA_MEASUREMENT_ID, METRIKA_ID, SUPPORT_EMAIL
)
from backend.email_utils import send_email
from backend.openai_utils import ask_openai
from backend.models import User, Message, Session
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

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("leadinc-backend")

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

def get_stage(msg_count):
    if msg_count < MESSAGE_LIMITS[0]:
        return 1
    elif msg_count < sum(MESSAGE_LIMITS[:2]):
        return 2
    elif msg_count < sum(MESSAGE_LIMITS):
        return 3
    else:
        return 4

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
    lim_prefix = f"{user.id}" if user else session_id

    msg_count_key = f"msg_count:{lim_prefix}"
    msg_count = int(await redis.get(msg_count_key) or 0)

    # --- Stage & лимиты по этапам ---
    stage = get_stage(msg_count)
    # -- Проектные лимиты только для авторизованных --
    if user:
        # Считаем проекты по user.id + дате
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
        # --- Этап 1 ---
        if stage == 1 and msg_count >= MESSAGE_LIMITS[0]:
            await redis.delete(msg_count_key)
            resp = JSONResponse({
                "reply": "Достигнут лимит сообщений. Для продолжения введите 6-значный код подтверждения из Telegram.",
                "meta": {"stage": 1}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp
        # --- Этап 2 ---
        elif stage == 2 and msg_count >= sum(MESSAGE_LIMITS[:2]):
            await redis.delete(msg_count_key)
            resp = JSONResponse({
                "reply": "Достигнут лимит сообщений. Пожалуйста, завершите регистрацию.",
                "meta": {"stage": 2}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp
        # --- Этап 3 ---
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

    # --- Защита от флуд-атаки (анализ временного окна для гостей) ---
    if not user:
        zset_key = f"guest_flood:{session_id}"
        now = current_timestamp()
        await redis.zremrangebyscore(zset_key, 0, ten_minutes_ago())
        await redis.zadd(zset_key, {str(now): now})
        guest_msgs = await redis.zcount(zset_key, ten_minutes_ago(), now)
        if guest_msgs > 20:  # максимум 20 сообщений за 10 минут у гостя
            resp = JSONResponse({
                "reply": "Слишком много сообщений за короткое время. Попробуйте позже.",
                "meta": {"stage": stage, "reason": "flood"}
            })
            resp.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            return resp

    # --- РЕГИСТРАЦИЯ и АВТО-ЛОГИН ---
    if "регистрация" in content.lower():
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
            
            # Привязка истории сообщений к user_id
            await session.execute(
                update(Message)
                .where(Message.session_id == session_id, Message.user_id == None)
                .values(user_id=user_obj.id)
            )
            await session.commit()
            
            # --- Авто-логин (выдача токена через fastapi-users) ---
            # Получить токен через fastapi-users и вернуть его клиенту (пример):
            from fastapi_users.authentication import JWTStrategy
            jwt_strategy = fastapi_users.get_jwt_strategy()
            token = await jwt_strategy.write_token(user_obj)
            
            await redis.delete(msg_count_key)
            await redis.delete(attempts_key)
            today = time.strftime("%Y-%m-%d")
            await redis.set(f"project_count:{user_obj.id}:{today}", 0, ex=86400)
            
            reg_response = JSONResponse({
                "reply": "Регистрация завершена! Вы автоматически авторизованы. Пароль отправлен в Telegram.",
                "token": token,
                "meta": {"stage": 4}
            })
            reg_response.set_cookie(key="sessionid", value=session_id, max_age=12 * 60 * 60, httponly=True, secure=False, samesite="Lax")
            # await send_to_telegram_bot(email, phone, password)
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

        # --- Если пользователь авторизован и запрос связан с созданием проекта ---
        if user and ("создать проект" in content.lower() or "новый проект" in content.lower()):
            today = time.strftime("%Y-%m-%d")
            project_limit_key = f"project_count:{user.id}:{today}"
            await redis.incr(project_limit_key)
            await redis.expire(project_limit_key, 86400)
    except Exception as e:
        logger.error(f"DB error: {str(e)}")

    response_data = {"reply": reply, "meta": usage, "stage": (get_stage(msg_count) if not user else 4)}

    response = JSONResponse(response_data)
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
