import logging
from fastapi import FastAPI, Request, Depends, HTTPException, APIRouter, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import Body
import uuid
import time
import re
import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

import redis.asyncio as aioredis

from backend.auth import get_user_manager
from backend.utils.passwords import generate_password
from fastapi_users.password import PasswordHelper
from backend.models import Session as SessionModel

from backend.auth import (
    fastapi_users, auth_backend, require_active_subscription, current_active_user_optional, get_jwt_strategy
)
from backend.config import (
    DEBUG, LOG_LEVEL, ADMIN_EMAIL, GA_MEASUREMENT_ID, METRIKA_ID, SUPPORT_EMAIL, SESSION_COOKIE_NAME
)
from backend.email_utils import send_email
from backend.openai_utils import ask_openai
from backend.models import User, Message, Session as SessionModel
from backend.schemas import UserRead, UserCreate, ChatRequest, SupportRequest
from backend.database import SessionLocal

async def get_db():
    async with SessionLocal() as session:
        yield session

async def clear_session_keys(sessionid: str):
    keys = [
        f"stage:{sessionid}",
        f"reg_phone:{sessionid}",
        f"reg_email:{sessionid}",
        f"reg_attempts:{sessionid}",
        f"msg_count:{sessionid}:stage1",
        f"msg_count:{sessionid}:stage2",
        f"msg_count:{sessionid}:stage3",
        f"guest_flood:{sessionid}",
    ]
    for key in keys:
        await redis.delete(key)

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

redis = aioredis.from_url("redis://localhost:6379", decode_responses=True)
password_helper = PasswordHelper()

engine = create_async_engine(
    f"postgresql+asyncpg://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@{os.environ.get('POSTGRES_HOST')}:{os.environ.get('POSTGRES_PORT')}/{os.environ.get('POSTGRES_DB')}",
    echo=True
)


def set_session_cookie(response: Response, session_id: str):
    response.set_cookie(
        key="sessionid",
        value=session_id,
        max_age=12 * 60 * 60,
        httponly=True,
        secure=not DEBUG,
        samesite="Strict" if not DEBUG else "Lax"
    )
    logger.info(f"Set sessionid cookie: {session_id}")
    print(f"Set sessionid cookie: {session_id}")
    return response

def get_or_create_session_id(request: Request) -> str:
    session_id = request.cookies.get("sessionid")
    try:
        if session_id and uuid.UUID(session_id):
            return session_id
    except Exception:
        pass
    return str(uuid.uuid4())

def current_timestamp():
    return int(time.time())

def ten_minutes_ago():
    return int(time.time()) - 600

def five_days():
    return 5 * 24 * 60 * 60


app = FastAPI(
    title="Leadinc AI Assistant",
    description="AI SaaS Assistant (B2B)",
    debug=DEBUG,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

@app.get("/auth/users/me", tags=["auth"])
async def get_current_user(user=Depends(current_active_user_optional)):
    logger.info(f"Проверка авторизации: user={user}")
    if user is None:
        logger.warning("User не найден — считаем неавторизованным.")
        return JSONResponse({"is_authenticated": False}, status_code=401)
    return {
        "is_authenticated": True,
        "id": str(user.id),
        "email": user.email,
        "phone": user.phone
    }

@app.post("/auth/jwt/login_custom", tags=["auth"])
async def login_custom(
    request: Request,
    username: str = Body(..., embed=True),  
    password: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    user_manager=Depends(get_user_manager)
):
    session_id = request.cookies.get("sessionid")
    if not session_id:
        session_id = str(uuid.uuid4())
    result = await db.execute(
        select(User).where((User.email == username) | (User.phone == username))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=400, detail="Пользователь не найден")

    password_helper = PasswordHelper()
    verified, updated_password = password_helper.verify_and_update(password, user.hashed_password)    
    if not verified:
        raise HTTPException(status_code=400, detail="Неверный пароль")

    jwt_strategy = get_jwt_strategy()
    token = await jwt_strategy.write_token(user)
    response = JSONResponse({"token": token, "email": user.email, "phone": user.phone})
    response = set_session_cookie(response, session_id)
    return response

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

MESSAGE_LIMITS = [50, 20, 20]
PROJECT_LIMIT_PER_DAY = 10
USER_LIMIT = 30


@ai_router.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    user: User = Depends(current_active_user_optional),
    db: AsyncSession = Depends(get_db),
):
    content = payload.content
    msg_type = payload.type
    session_id = get_or_create_session_id(request)

    logger.info(f"--- NEW CHAT REQ --- session={session_id} user={getattr(user, 'id', None)} content='{content[:40]}'")

    phone_redis = await redis.get(f"reg_phone:{session_id}")
    email_redis = await redis.get(f"reg_email:{session_id}")



    if user:
        stage = 4
        logger.info(f"User is authorized. Forcing stage=4 for user_id={user.id}, session={session_id}")
    else:
        stage_key = f"stage:{session_id}"
        raw_stage = await redis.get(stage_key)
        if raw_stage is None:
            stage = 1
            await redis.set(stage_key, stage, ex=12*60*60)
            logger.info(f"Stage for session {session_id} not found. Set to 1.")

            async with db.begin():
                q = select(SessionModel).where(SessionModel.id == session_id)
                result = await db.execute(q)
                existing = result.scalar_one_or_none()
                if not existing:
                    db.add(SessionModel(id=session_id))
        else:
            stage = int(raw_stage)
            await redis.expire(stage_key, 12*60*60)
            logger.info(f"Stage for session {session_id}: {stage}")


    lim_prefix = f"{user.id}" if user else session_id
    msg_count_key = f"msg_count:{lim_prefix}:stage{stage}"
    msg_count = int(await redis.get(msg_count_key) or 0)
    if not user:
        if stage == 1 and msg_count >= MESSAGE_LIMITS[0]:
            await redis.delete(msg_count_key)
            logger.warning(f"Stage 1: guest msg limit, session={session_id}")
            return set_session_cookie(JSONResponse({
                "reply": "Достигнут лимит сообщений. Для продолжения введите 6-значный код подтверждения из Telegram.",
                "meta": {"stage": 1, "reason": "guest_limit"}
            }), session_id)
        elif stage == 2 and msg_count >= sum(MESSAGE_LIMITS[:2]):
            await redis.delete(msg_count_key)
            logger.warning(f"Stage 2: guest msg limit, session={session_id}")
            return set_session_cookie(JSONResponse({
                "reply": "Достигнут лимит сообщений. Пожалуйста, завершите регистрацию.",
                "meta": {"stage": 2, "reason": "guest_limit"}
            }), session_id)
        elif stage == 3 and msg_count >= sum(MESSAGE_LIMITS):
            await redis.delete(msg_count_key)
            logger.warning(f"Stage 3: guest msg limit, session={session_id}")
            return set_session_cookie(JSONResponse({
                "reply": "Достигнут лимит сообщений. Для продолжения требуется регистрация.",
                "meta": {"stage": 3, "reason": "guest_limit"}
            }), session_id)
        zset_key = f"guest_flood:{session_id}"
        now = current_timestamp()
        await redis.zremrangebyscore(zset_key, 0, ten_minutes_ago())
        await redis.zadd(zset_key, {str(now): now})
        guest_msgs = await redis.zcount(zset_key, ten_minutes_ago(), now)
        if guest_msgs > 20:
            logger.warning(f"Flood protection: guest, session={session_id}")
            return set_session_cookie(JSONResponse({
                "reply": "Слишком много сообщений за короткое время. Попробуйте позже.",
                "meta": {"stage": stage, "reason": "flood"}
            }), session_id)
        await redis.incr(msg_count_key)
        await redis.expire(msg_count_key, 600)

    if user:
        today = time.strftime("%Y-%m-%d")
        project_limit_key = f"project_count:{user.id}:{today}"
        project_count = int(await redis.get(project_limit_key) or 0)
        if project_count >= PROJECT_LIMIT_PER_DAY:
            logger.warning(f"User project limit. user_id={user.id} session={session_id}")
            return set_session_cookie(JSONResponse({
                "reply": "На сегодня вы уже создали 5 проектов. Следующий можно будет создать завтра или по запросу через поддержку.",
                "meta": {"stage": 4, "reason": "project_limit"}
            }), session_id)
        zset_key = f"msg_zset:{user.id}"
        now = current_timestamp()
        await redis.zremrangebyscore(zset_key, 0, ten_minutes_ago())
        await redis.zadd(zset_key, {str(uuid.uuid4()): now})
        msg_in_window = await redis.zcount(zset_key, ten_minutes_ago(), now)
        if msg_in_window > USER_LIMIT:
            logger.warning(f"User msg limit. user_id={user.id} session={session_id}")
            return set_session_cookie(JSONResponse({
                "reply": "Вы слишком активно отправляете сообщения. Пожалуйста, сделайте паузу.",
                "meta": {"stage": 4, "reason": "msg_limit"}
            }), session_id)

    phone_redis = await redis.get(f"reg_phone:{session_id}")
    email_redis = await redis.get(f"reg_email:{session_id}")
    ai_response = await ask_openai(
        content=content,
        msg_type=msg_type,
        stage=stage,
        user_authenticated=bool(user),
        phone=phone_redis,
        email=email_redis
    )

    new_stage = ai_response.get('stage', stage)
    fields = ai_response.get('fields', {})
    logger.info(f"AI response: stage={new_stage} fields={fields}")

    updated = False
    if fields.get("phone") and fields["phone"] != phone_redis:
        await redis.set(f"reg_phone:{session_id}", fields["phone"], ex=3600)
        phone_redis = fields["phone"]
        updated = True
    if fields.get("email") and fields["email"] != email_redis:
        await redis.set(f"reg_email:{session_id}", fields["email"], ex=3600)
        email_redis = fields["email"]
        updated = True

    phone_final = fields.get("phone") or phone_redis
    email_final = fields.get("email") or email_redis

    if updated:
        logger.info(f"[PATCH] reg_phone:{session_id}={phone_redis}, reg_email:{session_id}={email_redis}")

    if not user and stage == 1 and new_stage == 2:
        user_code = fields.get("code")
        if not user_code:
            logger.warning(f"Код не найден в fields, stage=1→2, session={session_id}")
            return set_session_cookie(JSONResponse({
                "reply": "Код не распознан. Введите 6-значный код из Telegram.",
                "meta": {"stage": 1, "reason": "code_missing"}
            }), session_id)
        code_key = f"real_code:{user_code}"
        code_exists = await redis.exists(code_key)
        if not code_exists:
            logger.warning(f"Невалидный код: {user_code}, session={session_id}")
            return set_session_cookie(JSONResponse({
                "reply": "Введённый код неверен или устарел. Запросите новый код в Telegram-боте.",
                "meta": {"stage": 1, "reason": "code_invalid"}
            }), session_id)
        await redis.delete(code_key)
        logger.info(f"Код принят: {user_code}, session={session_id}")

    allow = False
    if user:
        if new_stage == 4:
            allow = True
        else:
            logger.warning(f"User is authenticated, but AI tried to set stage={new_stage}. Forcing stage=4")
            new_stage = 4
            allow = True
    else:
        if new_stage == stage or new_stage == stage + 1:
            allow = True
        else:
            logger.warning(f"Прыжок stage запрещён: {stage} → {new_stage}")
            return set_session_cookie(JSONResponse({
                "reply": "Ошибка этапа! Давай попробуем ещё раз по шагам.",
                "meta": {"stage": stage, "reason": "stage_jump"}
            }), session_id)

    stage_key = f"stage:{session_id}"
    if allow:
        await redis.set(stage_key, new_stage, ex=12*60*60)
        logger.info(f"Stage updated: {stage} → {new_stage} session={session_id}")

    if not user and stage == 3 and new_stage == 4 and phone_final and email_final:
        try:
            async with db.begin():
                q = select(User).where(
                    (User.phone == phone_final) | (User.email == email_final)
                )
                result = await db.execute(q)
                existing = result.scalar_one_or_none()
                if existing:
                    logger.info(f"Регистрация отклонена: телефон/почта заняты, session={session_id}")
                    return set_session_cookie(JSONResponse({
                        "reply": "Этот телефон или почта уже зарегистрированы.",
                        "meta": {"stage": 3}
                    }), session_id)
                password = generate_password(8)
                password_hash = password_helper.hash(password)
                user_obj = User(
                    email=email_final,
                    phone=phone_final,
                    hashed_password=password_hash,
                    is_active=True,
                    is_verified=True,
                )
                db.add(user_obj)
                await db.flush()                 
                q = select(SessionModel).where(SessionModel.id == session_id)
                res = await db.execute(q)
                session_db = res.scalar_one_or_none()
                if session_db and not session_db.user_id:
                    session_db.user_id = user_obj.id

            jwt_strategy = get_jwt_strategy()
            token = await jwt_strategy.write_token(user_obj)
            dev_block = (
                "\n\n------------------------\n"
                "[альфа тест]\n"
                "Вы зарегистрированы! Теперь вам доступен расширенный функционал Leadinc.\n"
                f"Ваши данные для входа:\n"
                f"Телефон: {phone_final}\n"
                f"Email: {email_final}\n"
                f"Пароль: {password}\n"
                "Вы автоматически авторизованы.\n"
                "------------------------"
            )
            ai_response["reply"] = (ai_response.get("reply") or "") + dev_block
            logger.info(f"Новый пользователь зарегистрирован: email={email_final}, phone={phone_final}")
            print(f"[DEBUG] Generated password for {email_final}: {password}")
            logger.info(f"Final AI reply: {ai_response['reply']}")
            await redis.delete(f"stage:{session_id}")
            await redis.delete(f"reg_phone:{session_id}")
            await redis.delete(f"reg_email:{session_id}")
            await redis.delete(f"reg_attempts:{session_id}")
            await redis.delete(f"msg_count:{session_id}:stage1")
            await redis.delete(f"msg_count:{session_id}:stage2")
            await redis.delete(f"msg_count:{session_id}:stage3")
            await redis.delete(f"guest_flood:{session_id}")
            await redis.expire(session_id, five_days())

            response_data = {
                "reply": ai_response["reply"],
                "meta": {
                    "stage": new_stage,
                    "usage": ai_response.get("usage", {}),
                    "fields": fields,
                    "token": token,
                    "login": email_final,
                    "password": password
                }
            }
            response = JSONResponse(response_data)
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_id,
                max_age=12 * 60 * 60,
                httponly=True,
                secure=not DEBUG,
                samesite="Strict" if not DEBUG else "Lax"
            )
            response.set_cookie(
                key="fastapiusersauth",
                value=token,
                max_age=12 * 60 * 60,
                httponly=True,
                secure=not DEBUG,
                samesite="Strict" if not DEBUG else "Lax"
            )
            return response
        except Exception as e:
            logger.error(f"Ошибка при регистрации: {str(e)}")
            return set_session_cookie(JSONResponse({
                "reply": "Техническая ошибка регистрации. Попробуйте снова.",
                "meta": {"stage": 3, "reason": "register_error"}
        }), session_id)

    try:
        user_id = user.id if user else None
        async with db.begin():
            user_msg = Message(
                session_id=session_id,
                user_id=user_id,
                role="user",
                type=msg_type,
                status="ok",
                content=content,
                meta={},
            )
            db.add(user_msg)
            assistant_msg = Message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                type="text",
                status="ok",
                content=ai_response["reply"],
                meta=ai_response.get("usage", {}),
            )
            db.add(assistant_msg)
    except Exception as e:
        logger.error(f"DB error while saving messages: {str(e)}")
        print(f"Return: DB error (messages), session={session_id}")

    response_data = {
        "reply": ai_response["reply"],
        "meta": {
            "stage": new_stage,
            "usage": ai_response.get("usage", {}),
            "fields": fields,
            "token": ai_response.get("token")
        }
    }

    logger.info(f"Final response: stage={new_stage}, session={session_id}, user={getattr(user, 'id', None)}")
    print(f"Return: Final, session={session_id}, stage={new_stage}, cookie={session_id}")

    return set_session_cookie(JSONResponse(response_data), session_id)

@app.post("/auth/jwt/logout", tags=["auth"])
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("sessionid")
    if session_id:
        await redis.delete(f"stage:{session_id}")
        await redis.delete(f"reg_phone:{session_id}")
        await redis.delete(f"reg_email:{session_id}")
        await redis.delete(f"reg_attempts:{session_id}")
        await redis.delete(f"msg_count:{session_id}:stage1")
        await redis.delete(f"msg_count:{session_id}:stage2")
        await redis.delete(f"msg_count:{session_id}:stage3")
        await redis.delete(f"guest_flood:{session_id}")
    response = JSONResponse({"detail": "Logout complete"})
    response.delete_cookie(key="sessionid")
    response.delete_cookie(key="fastapiusersauth")
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
