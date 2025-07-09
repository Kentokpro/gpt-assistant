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
from backend.openai_utils import ask_openai, get_embedding
from backend.models import User, Message, Session as SessionModel
from backend.schemas import UserRead, UserCreate, ChatRequest, SupportRequest
from backend.database import SessionLocal
from backend.chroma_utils import search_chunks_by_embedding, get_full_article

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
        secure=not DEBUG,  # На проде: True
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
    response.set_cookie(
        key="fastapiusersauth",
        value=token,
        max_age=12 * 60 * 60,
        httponly=True,
        secure=not DEBUG,
        samesite="Strict" if not DEBUG else "Lax" 
    )
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

MESSAGE_LIMITS = [20, 20, 20]
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
    async with db.begin():
        q = select(SessionModel).where(SessionModel.id == session_id)
        result = await db.execute(q)
        existing = result.scalar_one_or_none()
        if not existing:
            db.add(SessionModel(id=session_id))
            await db.flush()
        logger.info(f"SessionModel for {session_id}: {'created' if not existing else 'exists'}")    
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
        if not user:
            client_ip = request.client.host
            ip_ban_key = f"guest_ip_ban:{client_ip}"
            is_banned = await redis.exists(ip_ban_key)
            if is_banned:
                logger.warning(f"IP BAN active for guest: {client_ip}")
                return set_session_cookie(JSONResponse({
                    "reply": "Временная блокировка на 2 часа.",
                    "meta": {"stage": stage, "reason": "ip_ban"}
                }), session_id)
        zset_key = f"guest_flood:{session_id}"
        now = current_timestamp()
        await redis.zremrangebyscore(zset_key, 0, ten_minutes_ago())
        await redis.zadd(zset_key, {str(now): now})
        guest_msgs = await redis.zcount(zset_key, ten_minutes_ago(), now)
        if guest_msgs > 20:
            client_ip = request.client.host
            ip_ban_key = f"guest_ip_ban:{client_ip}"
            await redis.set(ip_ban_key, 1, ex=2*60*60)  # бан по ip на 2 часа
            logger.warning(f"Flood protection + IP BAN: guest, session={session_id}, ip={client_ip}")
            return set_session_cookie(JSONResponse({
                "reply": "Временная блокировка на 2 часа.",
                "meta": {"stage": stage, "reason": "ip_ban"}
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
                    "reply": "На сегодня вы уже создали 10 проектов. Следующий можно будет создать завтра или по запросу через поддержку.",
                    "meta": {"stage": 4, "reason": "project_limit"}
                }), session_id)
            block_key = f"user_block:{user.id}"
            is_blocked = await redis.exists(block_key)
            if is_blocked:
                logger.warning(f"User is blocked for spamming. user_id={user.id}")
                return set_session_cookie(JSONResponse({
                    "reply": "Вы временно заблокированы из-за превышения лимита сообщений. Пауза 2 часа.",
                    "meta": {"stage": 4, "reason": "user_blocked"}
                }), session_id)

            zset_key = f"msg_zset:{user.id}"
            now = current_timestamp()
            await redis.zremrangebyscore(zset_key, 0, ten_minutes_ago())
            await redis.zadd(zset_key, {str(uuid.uuid4()): now})
            msg_in_window = await redis.zcount(zset_key, ten_minutes_ago(), now)
            if msg_in_window > USER_LIMIT:
                await redis.set(block_key, 1, ex=7200)
                logger.warning(f"User msg limit EXCEEDED. user_id={user.id} session={session_id}")
                return set_session_cookie(JSONResponse({
                    "reply": "Вы превысили лимит сообщений. Сделайте перерыв и возвращайтесь позже.",
                    "meta": {"stage": 4, "reason": "msg_limit_exceeded"}
                }), session_id)
        context_chunks = []
        if user and stage == 4:
            try:
                query_emb = await get_embedding(content)
                result = await search_chunks_by_embedding(query_emb, n_results=5, collection_name="faq_leadinc")
                docs = result.get("documents", [[]])[0]
                metas = result.get("metadatas", [[]])[0]
                context_chunks = [
                    {
                        "article_id": str(meta.get("article_id", "unknown")),
                        "title": meta.get("title", ""),
                        "summary": doc
                    }
                    for doc, meta in zip(docs, metas)
                ]
                logger.info(f"[RAG] Найдено {len(context_chunks)} чанков для context. IDs: {[meta.get('article_id') for meta in metas]}")
            except Exception as e:
                logger.error(f"[RAG] Ошибка поиска в базе: {e}")
        SHORT_CONFIRM = {
            "да", "давай", "дальше", "ещё", "еще", "продолжи", "продолжай", "погнали", "Пожалуйста", "весь текст",
            "полностью", "ок", "да, расскажи", "давай полностью", "да, интересно", "ага", "расскажи", "подробнее", "расширенно", "go on", "yes", "sure",
            "expand", "all", "👍", "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿", "ok", "lets go", "let's go",
            "continue", "more", "next", "well", "of course", "конечно", "хочу больше", "расскажи полностью",
            "поясни", "поясни полностью", "больше", "ещё раз", "ещё чуть-чуть", "ещё инфы", "ещё информации",
            "полную", "разверни", "расширь", "развёрнуто", "show all", "give me all", "give all", "расскажи до конца",
            "поясни с нуля", "полный текст", "весь", "всю", "расширенный ответ", "эм", "эмм", "угу", "yes please", "да, расскажи", "да, хочу", "да, давай", "да, полностью", "да, интересно", "да, пожалуйста", "да, конечно", "окей", "окей, расскажи", "расширь", "ещё!", "ещё инфы", "поясни подробнее", "ещё раз!", "разверни", "расскажи все", "расскажи всё", "покажи всё", "мне интересно", "да, погнали", "go", "yes, tell me more", "all right", "alright", "more info", "show more", "расскажи дальше", "ещё расскажи"
        }
        user_input = content.strip().lower()
        pending_article_id = await redis.get(f"pending_full_article:{session_id}")
        logger.info(f"DEBUG: content='{content}', user_input='{user_input}', session_id='{session_id}', pending_article_id='{pending_article_id}'")
        logger.info(f"SHORT_CONFIRM: {SHORT_CONFIRM}")
        logger.info(f"user_input: {user_input!r}")
        logger.info(f"pending_article_id: {pending_article_id!r}")
        logger.info(f"any: {any(key in user_input for key in SHORT_CONFIRM)}")
        logger.info(f"session_id: {session_id!r}")

        q = select(SessionModel).where(SessionModel.id == session_id)
        result = await db.execute(q)
        existing = result.scalar_one_or_none()
        if not existing:
            db.add(SessionModel(id=session_id))
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
            db.add(user_msg)
        except Exception as e:
            logger.error(f"DB error while saving user message: {str(e)}")
            print(f"Return: DB error (user message), session={session_id}")
        if pending_article_id and any(key in user_input for key in SHORT_CONFIRM):
            try:
                article_text = await get_full_article(pending_article_id)
                logger.debug(f"article_text type: {type(article_text)} value: {article_text}")
                await redis.delete(f"pending_full_article:{session_id}")
                logger.info(f"[DEBUG] Ключ pending_full_article:{session_id} удалён после выдачи полной статьи")
                try:
                    assistant_msg = Message(
                        session_id=session_id,
                        user_id=user_id,
                        role="assistant",
                        type="text",
                        status="ok",
                        content=article_text,
                        meta={},
                    )
                    db.add(assistant_msg)
                except Exception as e:
                    logger.error(f"DB error while saving assistant message (full_article): {str(e)}")
                q = (
                    select(Message)
                    .where(Message.session_id == session_id)
                    .order_by(Message.created_at.desc())
                    .limit(10)
                )
                result = await db.execute(q)
                msgs_keep = result.scalars().all()[::-1]
                q_all = (
                    select(Message)
                    .where(Message.session_id == session_id)
                    .order_by(Message.created_at.desc())
                )
                result_all = await db.execute(q_all)
                all_msgs = result_all.scalars().all()
                if len(all_msgs) > 10:
                    ids_keep = set(msg.id for msg in msgs_keep)
                    ids_del = [msg.id for msg in all_msgs if msg.id not in ids_keep]
                    if ids_del:
                        await db.execute(
                            Message.__table__.delete().where(Message.id.in_(ids_del))
                        )
                await db.commit()
                return set_session_cookie(JSONResponse({
                    "reply": article_text,
                    "meta": {
                        "stage": stage,
                        "action": "full_article",
                        "article_id": pending_article_id,
                    }
                }), session_id)
            except Exception as e:
                logger.error(f"Ошибка выдачи полной статьи по article_id={pending_article_id}: {e}")
                await redis.delete(f"pending_full_article:{session_id}")
                logger.info(f"[DEBUG] Ключ удалён pending_full_article:{session_id}")
                await db.rollback()
                return set_session_cookie(JSONResponse({
                    "reply": "Техническая ошибка. Не удалось получить полный текст.",
                    "meta": {
                        "stage": stage,
                        "action": "full_article",
                        "article_id": pending_article_id,
                        "error": str(e)
                    }
                }), session_id)

        messages_for_gpt = []
        q = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(10)
        )
        result = await db.execute(q)
        msgs_keep = result.scalars().all()[::-1]
        q_all = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
        )
        result_all = await db.execute(q_all)
        all_msgs = result_all.scalars().all()
        if len(all_msgs) > 10:
            ids_keep = set(msg.id for msg in msgs_keep)
            ids_del = [msg.id for msg in all_msgs if msg.id not in ids_keep]
            if ids_del:
                await db.execute(
                    Message.__table__.delete().where(Message.id.in_(ids_del))
                )
        messages_for_gpt = [{"role": msg.role, "content": msg.content} for msg in msgs_keep]

        phone_redis = await redis.get(f"reg_phone:{session_id}")    
        email_redis = await redis.get(f"reg_email:{session_id}")  
        ai_response = await ask_openai(  
            content=content,  
            msg_type=msg_type,  
            stage=stage,  
            user_authenticated=bool(user),  
            phone=phone_redis,  
            email=email_redis,  
            context=context_chunks if context_chunks else [],
            messages=messages_for_gpt  
        )  
        logger.info(f"ai_response: {ai_response}")

        try:
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
            logger.error(f"DB error while saving assistant message (AI): {str(e)}")

        q = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(10)
        )
        result = await db.execute(q)
        msgs_keep = result.scalars().all()[::-1]
        q_all = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
        )
        result_all = await db.execute(q_all)
        all_msgs = result_all.scalars().all()
        if len(all_msgs) > 10:
            ids_keep = set(msg.id for msg in msgs_keep)
            ids_del = [msg.id for msg in all_msgs if msg.id not in ids_keep]
            if ids_del:
                await db.execute(
                    Message.__table__.delete().where(Message.id.in_(ids_del))
                )
        action = ai_response.get("action") or ai_response.get("fields", {}).get("action")
        article_id = ai_response.get("article_id") or ai_response.get("fields", {}).get("article_id")
        if action == "offer_full_article" and article_id:
            await redis.set(f"pending_full_article:{session_id}", article_id, ex=3600)

        if ai_response.get("action") == "full_article" and ai_response.get("article_id"):
            article_id = ai_response.get("article_id")
            try:
                article_text = await get_full_article(article_id)
                await redis.delete(f"pending_full_article:{session_id}")
                logger.info(f"[DEBUG] Сохраняем pending_full_article:{session_id} = {article_id}")
                await db.commit()
                return set_session_cookie(JSONResponse({
                    "reply": article_text,
                    "meta": {
                        "stage": stage,
                        "action": "full_article",
                        "article_id": article_id,
                    }
                }), session_id)
            except Exception as e:
                logger.error(f"Ошибка выдачи полной статьи по article_id={article_id}: {e}")
                await db.rollback()
                return set_session_cookie(JSONResponse({
                    "reply": "Техническая ошибка. Не удалось получить полный текст.",
                    "meta": {
                        "stage": stage,
                        "action": "full_article",
                        "article_id": article_id,
                        "error": str(e)
                    }
                }), session_id)

        new_stage = ai_response.get('stage', stage)
        fields = ai_response.get('fields', {})
        if "action" in ai_response:
            fields["action"] = ai_response["action"]
        if "article_id" in ai_response:
            fields["article_id"] = ai_response["article_id"]
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
                await db.commit()
                return response
            except Exception as e:
                logger.error(f"Ошибка при регистрации: {str(e)}")
                await db.rollback()
                return set_session_cookie(JSONResponse({
                    "reply": "Техническая ошибка регистрации. Попробуйте снова.",
                    "meta": {"stage": 3, "reason": "register_error"}
            }), session_id)

    reply = ai_response.get("reply", "")
    if not isinstance(reply, str):
        reply = json.dumps(reply, ensure_ascii=False)
    
    response_data = {
        "reply": reply,
        "meta": {
            "stage": new_stage,
            "usage": ai_response.get("usage", {}),
            "fields": fields,
            "token": ai_response.get("token")
        }
    }

    logger.info(f"Final response: stage={new_stage}, session={session_id}, user={getattr(user, 'id', None)}")
    print(f"Return: Final, session={session_id}, stage={new_stage}, cookie={session_id}")

    await db.commit()
    return set_session_cookie(JSONResponse(response_data), session_id)


@ai_router.post("/rag")
async def rag_search(
    payload: ChatRequest,
    request: Request,
    user: User = Depends(current_active_user_optional),
):
    if not user:
        return JSONResponse(
            {"reply": "Поиск по базе Leadinc доступен только авторизованным пользователям.", "meta": {"reason": "unauthorized"}},
            status_code=403
        )
    session_id = request.cookies.get("sessionid")
    stage_key = f"stage:{session_id}"
    stage = None
    if session_id:
        raw_stage = await redis.get(stage_key)
        if raw_stage is not None:
            try:
                stage = int(raw_stage)
            except Exception:
                stage = None
    if stage != 4:
        return JSONResponse(
            {"reply": "Доступ к базе открыт только после завершения регистрации и авторизации.", "meta": {"reason": "not_authorized_stage"}},
            status_code=403
        )
    try:
        query_emb = await get_embedding(payload.content)
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        return JSONResponse(
            {"reply": "Ошибка обработки embedding. Попробуйте позже.", "meta": {"reason": "embedding_error"}},
            status_code=500
        )
    try:
        result = await search_chunks_by_embedding(
            query_emb=query_emb,
            n_results=3,
            collection_name="faq_leadinc"
        )
    except Exception as e:
        logger.error(f"ChromaDB search error: {e}")
        return JSONResponse(
            {"reply": "Ошибка поиска по базе знаний. Попробуйте позже.", "meta": {"reason": "chroma_error"}},
            status_code=500
        )

    found_texts = result.get("documents", [[]])[0]
    found_metas = result.get("metadatas", [[]])[0]

    if not found_texts:
        return JSONResponse(
            {"reply": "По вашему запросу не найдено релевантной информации в базе Leadinc.", "meta": {"chunks": [], "found": 0}},
            status_code=200
        )
    joined_chunks = "\n\n".join(found_texts)
    meta_info = [meta.get("title", "") for meta in found_metas]

    return JSONResponse(
        {
            "reply": joined_chunks,
            "meta": {
                "chunks": meta_info,
                "found": len(found_texts)
            }
        },
        status_code=200
    )


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

    response.delete_cookie(
        key="sessionid",
        path="/",
        httponly=True,
        secure=True,      
        samesite="lax"    
    )
    response.delete_cookie(
        key="fastapiusersauth",
        path="/",
        httponly=True,
        secure=True,      
        samesite="strict" 
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
