"""
Leadinc AI Backend — полный рефакторинг (2024-07)
- Мягкая валидация через OpenAI-ассистента
- DEV ONLY: временный вывод логина/пароля через чат
- Управление этапами через ассистента, backend валидирует только переходы и защищает от скачков
- Подробное логирование всех ключевых событий
- Защита stage, нет дублей логики, правила валидации только в prompt
- После авторизации stage не может быть <4, код и регистрацию не спрашиваем
- Автоочистка неактуальных ключей Redis после регистрации (5 дней)
"""

import logging
from fastapi import FastAPI, Request, Depends, HTTPException, APIRouter, Response, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import Body
import json
import uuid
import time
import re
import os

from backend.tasks.stt import stt_task
from backend.tasks.tts import tts_task
from backend.utils.stt_utils import save_upload_file
from celery.result import AsyncResult
from backend.utils.audio_constants import SUPPORTED_TTS_FORMATS

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

import redis.asyncio as aioredis

from backend.auth import get_user_manager
from backend.utils.passwords import generate_password
from fastapi_users.password import PasswordHelper
from backend.models import Session as SessionModel
from backend.chroma_utils import filter_chunks

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

from starlette.concurrency import run_in_threadpool

# --- VOICE TRIGGERS ---
VOICE_TRIGGER_PHRASES = [
    "голос",
    "ответь голосом",
    "озвучь ответ",
    "надиктуй пожалуйста",
    "скажи вслух",
    "прочитай мне",
    "можешь проговорить",
    "покажи голосом",
    "дай аудио ответ",
    "скажи, не пиши",
    "голосом объясни",
    "озвучь это",
    "прочти",
    "читай",
    "говори",
    "скажи словами",
    "расскажи голосом",
    "прямо голосом ответь",
    "озвуч вариант",
    "устно ответь",
    "можешь вслух сказать",
    "в аудио скажи",
    "проговори ответ",
    "скажи это вслух",
    "не пиши, а говори",
    "озвуч инфу",
    "ответ голосовой",
    "зачитай пожалуйста",
    "голосовое сообщение",
    "надиктуй ответ",
    "прочитай вслух",
    "скажи в микрофон",
    "аудио пожалуйста",
]

TEXT_TRIGGER_PHRASES = [
    "ответ текстом",
    "напиши текстом",
    "выведи текст",
    "только текст",
    "без аудио",
    "без звука"
    "текст",
    "можно текстом?",
    "напиши ответ",
    "покажи в виде текста",
    "расшифруй в текст",
    "мне нужен текст",
    "переведи в текст",
    "отправь как текст",
    "сделай текстовый ответ",
    "напиши, что получилось",
    "напиши это",
    "что получилось в тексте?",
    "мне удобнее текст",
    "без озвучки",
    "не говори",
    "можно без звука?",
    "дай только текст",
    "ответь текстом",
    "хочу увидеть текст",
    "выведи только текст",
    "выведи без звука",
    "дублируй текстом",
    "распознай текст",
    "распиши текстом",
    "ответ напиши",
    "пиши",
]

# Миграция SessionLocal Depends
async def get_db():
    async with SessionLocal() as session:
        yield session

# Очистка всех временных ключей сессии
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

# === 1. Логгер и настройки ===
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

# === 2. SQLAlchemy, FastAPI, Directories ===
engine = create_async_engine(
    f"postgresql+asyncpg://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@{os.environ.get('POSTGRES_HOST')}:{os.environ.get('POSTGRES_PORT')}/{os.environ.get('POSTGRES_DB')}",
    echo=True
)

# === 3. Куки, сессии и утилиты ===

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

# === 4. FastAPI App и Middleware ===

app = FastAPI(
    title="Leadinc AI Assistant",
    description="AI SaaS Assistant (B2B)",
    debug=DEBUG,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Проверяет основные сервисы: Redis, Celery (наличие таски), OpenAI API KEY, ElevenLabs API KEY.
@app.get("/health", tags=["infra"])
async def health_check():
    import os
    from openai import AsyncOpenAI
    from elevenlabs.client import ElevenLabs

    health = {}

    # Redis check
    try:
        pong = await redis.ping()
        health["redis"] = "ok" if pong else "fail"
    except Exception as e:
        health["redis"] = f"fail: {e}"

    # Celery (наличие задачи)
    try:
        # Пытаемся просто импортировать задачу (worker может быть не поднят)
        tid = str(uuid.uuid4())
        dummy_task = tts_task.AsyncResult(tid)
        health["celery"] = "ok"
    except Exception as e:
        health["celery"] = f"fail: {e}"

    # OpenAI KEY
    try:
        openai_key = os.getenv("OPENAI_API_KEY") or "NO"
        health["openai_api_key"] = "ok" if openai_key and openai_key != "NO" else "not_set"
        # можно попытаться сделать dummy-запрос к OpenAI (по желанию)
    except Exception as e:
        health["openai_api_key"] = f"fail: {e}"

    # ElevenLabs KEY
    try:
        eleven_key = os.getenv("ELEVENLABS_API_KEY") or "NO"
        health["elevenlabs_api_key"] = "ok" if eleven_key and eleven_key != "NO" else "not_set"
    except Exception as e:
        health["elevenlabs_api_key"] = f"fail: {e}"
    return JSONResponse(health)

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

MEDIA_DIR = Path("/srv/leadinc-media/audio")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
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

# === 5. Ограничения и лимиты ===
MESSAGE_LIMITS = [20, 20, 20]
PROJECT_LIMIT_PER_DAY = 10
USER_LIMIT = 30

# === 6. Основной AI endpoint — ЛОГИКА СЦЕНАРИЯ/СТАДИЙ ===
@ai_router.post("/chat")
async def chat(
    request: Request,
    response: Response,
    user: User = Depends(current_active_user_optional),
    db: AsyncSession = Depends(get_db),
    audio: UploadFile = File(None),
    type_: str = Form(None),
):
    """
    Универсальный endpoint /ai/chat: поддержка JSON и FormData (text, voice)
    - Для текста: работает как раньше.
    - Для голоса: парсит файл, делает STT, приводит к единому payload.
    """
    logger.info(f"==== [UNIVERSAL CHAT ENDPOINT] START ====")

    session_id = get_or_create_session_id(request)
    stage_key = f"stage:{session_id}"
    raw_stage = await redis.get(stage_key)
    if raw_stage is None:
        stage = 1
        await redis.set(stage_key, stage, ex=12*60*60)
    else:
        stage = int(raw_stage)
        await redis.expire(stage_key, 12*60*60)
    phone_redis = None
    email_redis = None
    context_chunks = []
    messages_for_gpt = []
    answer_format = None
    msg_type = None
    content = None
    logger.info(f"[INIT] session_id={session_id}, stage={stage}")

    # Блок обработки аудио
    if audio and (type_ == "voice" or (audio.filename and audio.content_type in ["audio/mpeg", "audio/mp3", "audio/ogg", "audio/webm"])):
        # --- Обработка аудио ---
        logger.info(f"[VOICE] Detected audio input: {audio.filename}, type: {type_}")
        try:
            from backend.utils.stt_utils import save_upload_file
            audio_path = await save_upload_file(audio)
            logger.info(f"[VOICE] Saved audio file: {audio_path}")

            from backend.tasks.stt import stt_task
            stt_result = await run_in_threadpool(lambda: stt_task.apply_async(args=[audio_path, None, None]).get(timeout=60))
            logger.info(f"[VOICE] STT result: {stt_result}")

            if stt_result.get("status") != "ok" or not stt_result.get("transcript"):
                return JSONResponse(
                    {
                        "reply": "Ошибка распознавания речи. Попробуйте ещё раз.",
                        "meta": {"stage": 1, "reason": "stt_error"}
                    }, status_code=400
                )
            msg_type = "voice"

            # после STT (speech-to-text) результата:
            payload = {
                "content": stt_result["transcript"],
                "type": "voice",
                "audio_path": audio_path,
                "answer_format": answer_format,
                "audio_meta": {
                    "filename": audio.filename,
                    "content_type": audio.content_type,
                }
            }

            # Дальше payload всегда dict!
            content = payload.get("content", "")
            msg_type = payload.get("type", "text")
            answer_format = payload.get("answer_format")

        except Exception as e:
            logger.error(f"[VOICE] Error while processing audio: {e}")
            return JSONResponse(
                {
                    "reply": "Ошибка обработки голосового сообщения.",
                    "meta": {"stage": 1, "reason": "voice_processing_error", "details": str(e)}
                }, status_code=500
            )

    else:
        # --- Обработка обычного текста (JSON) ---
        logger.info(f"[TEXT] Detected text input (JSON or fallback)")
        try:
            body = await request.body()
            data = json.loads(body)
            payload = {
                "content": data.get("content", ""),
                "type": data.get("type", "text"),
                "answer_format": data.get("answer_format", "text"),
            }

        except Exception as e:
            logger.error(f"[TEXT] Ошибка парсинга JSON: {e}")
            return JSONResponse(
                {
                    "reply": "Ошибка в формате запроса. Проверьте поля!",
                    "meta": {"stage": 1, "reason": "json_parse_error", "details": str(e)}
                }, status_code=422
            )

    # --- Единая точка после веток: payload уже определён ---
    content = payload.get("content", "")
    msg_type = payload.get("type", "text")
    answer_format = payload.get("answer_format")  # может быть None

    content_lower = content.lower() if isinstance(content, str) else ""
    logger.warning(f"[DEBUG][PAYLOAD] content={content!r}, msg_type={msg_type}, answer_format={answer_format}, payload={payload}")

    # --- Определяем, просит ли пользователь текст/войс через триггеры ---
    is_voice_trigger = any(trigger in content_lower for trigger in VOICE_TRIGGER_PHRASES)
    is_text_trigger = any(trigger in content_lower for trigger in TEXT_TRIGGER_PHRASES)
    logger.warning(f"[DEBUG][TRIGGERS] is_voice_trigger={is_voice_trigger}, is_text_trigger={is_text_trigger}")

    # --- Логика выбора формата ответа строго по чек-листу ---
    if msg_type == "voice" and (answer_format == "text" or is_text_trigger):
        answer_format = "text"
        logger.warning(f"[DEBUG][BRANCH] msg_type=voice, but text trigger or answer_format=text =>  answer_format=text")
    elif msg_type == "voice":
        answer_format = "voice"
        logger.warning(f"[DEBUG][BRANCH] msg_type=voice, no text trigger => answer_format=voice")
    elif answer_format == "voice" or is_voice_trigger:
        answer_format = "voice"
        logger.warning(f"[DEBUG][BRANCH] answer_format=voice or voice trigger =>    answer_format=voice")
    else:
        answer_format = "text"
        logger.warning(f"[DEBUG][BRANCH] default fallback => answer_format=text")

    logger.info(f"[PAYLOAD] content={content!r}, msg_type={msg_type}, answer_format={answer_format}")


    session_id = get_or_create_session_id(request)

    logger.info(f"--- NEW CHAT REQ --- session={session_id} user={getattr(user, 'id', None)} content='{content[:40]}'")

    # ====== ВСЯ ЛОГИКА В ОДНОМ БЛОКЕ ТРАНЗАКЦИИ ======
    async with db.begin():
        # --- 0. Сессия пользователя (SessionModel) ---
        q = select(SessionModel).where(SessionModel.id == session_id)
        result = await db.execute(q)
        existing = result.scalar_one_or_none()
        if not existing:
            db.add(SessionModel(id=session_id))
            await db.flush()  # [OK]
        logger.info(f"SessionModel for {session_id}: {'created' if not existing else 'exists'}")

        # --- 1. Память для AI: phone/email в Redis ---
        phone_redis = await redis.get(f"reg_phone:{session_id}")
        email_redis = await redis.get(f"reg_email:{session_id}")

        # --- 2. Определение этапа (stage) ---
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

        # --- 3. Лимиты, спам, guest limits ---
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
        
        # Flood protection (гостям)
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
            await redis.set(ip_ban_key, 1, ex=2*60*60)
            logger.warning(f"Flood protection + IP BAN: guest, session={session_id}, ip={client_ip}")
            return set_session_cookie(JSONResponse({
                "reply": "Временная блокировка на 2 часа.",
                "meta": {"stage": stage, "reason": "ip_ban"}
            }), session_id)

        await redis.incr(msg_count_key)
        await redis.expire(msg_count_key, 600)

        # --- 4. Лимиты пользователей ---
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

        # ============ RAG-поиск (база знаний) ============
        context_chunks = []
        if user and stage == 4:
            try:
                query_emb = await get_embedding(content)
                context_chunks = await search_chunks_by_embedding(query_emb, n_results=5, collection_name="faq_leadinc")
                if context_chunks:
                    log_context = [
                        {
                            "article_id": chunk.get("article_id"),
                            "title": chunk.get("title"),
                            "summary": chunk.get("summary"),
                            "text_sample": (chunk.get("text") or "")[:100]
                        }
                        for chunk in context_chunks
                    ]
                    logger.debug(f"[DEBUG] Передаю ассистенту context_chunks: {json.dumps(log_context, ensure_ascii=False, indent=2)}")
                else:
                    logger.debug("[DEBUG] context_chunks пуст — ассистент не получит статьи из базы")
                logger.info(f"[RAG] Найдено {len(context_chunks)} чанков для context. IDs: {[chunk.get('article_id') for chunk in context_chunks]}")
            except Exception as e:
                logger.error(f"[RAG] Ошибка поиска в базе: {e}")

        # --- [БЛОК] Обработка коротких подтверждений пользователя для выдачи полной статьи ---
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

        # --- A. SessionModel ---
        q = select(SessionModel).where(SessionModel.id == session_id)
        result = await db.execute(q)
        existing = result.scalar_one_or_none()
        if not existing:
            db.add(SessionModel(id=session_id))

        # --- B. Сохраняем сообщение пользователя ---
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

        # --- [БЛОК] "Да/Ок/Подробнее" → full_article (короткий путь, без вызова OpenAI) ---
        if pending_article_id and any(key in user_input for key in SHORT_CONFIRM):
            try:
                # 1. Получаем все чанки по article_id (через filter_chunks)
                chunks = await filter_chunks(article_id=pending_article_id)
                full_text = chunks[0]["text"] if chunks else ""
                logger.info(f"[DEBUG full_article] pending_article_id={pending_article_id}, chunks_count={len(chunks)}")
                context = [{
                    "article_id": pending_article_id,
                    "title": chunks[0]["title"] if chunks else "",
                    "meta_tags": chunks[0]["meta_tags"] if chunks else "",
                    "tags": chunks[0]["tags"] if chunks else [],
                    "summary": chunks[0]["summary"] if chunks else "",
                    "text": full_text
                }]
                logger.info(f"[DEBUG full_article] context (for ask_openai): {json.dumps(context, ensure_ascii=False)[:500]}")

                if not full_text:
                    logger.warning(f"Статья с article_id={pending_article_id} не найдена или пуста.")
                    await redis.delete(f"pending_full_article:{session_id}")
                    return set_session_cookie(JSONResponse({
                        "reply": "Ошибка при поиске информации.",
                        "meta": {
                            "stage": stage,
                            "action": "full_article",
                            "article_id": pending_article_id
                        }
                    }), session_id)

                # 3. Удаляем pending_full_article
                await redis.delete(f"pending_full_article:{session_id}")
                logger.info(f"[DEBUG] Ключ pending_full_article:{session_id} удалён после выдачи полной статьи")

                q = (
                    select(Message)
                    .where(Message.session_id == session_id)
                    .order_by(Message.created_at.desc())
                    .limit(10)
                )
                result = await db.execute(q)
                msgs_keep = result.scalars().all()[::-1]
                messages_for_gpt = [{"role": msg.role, "content": msg.content} for msg in msgs_keep]

                ai_response = await ask_openai(
                    content=content,
                    msg_type="text",
                    answer_format=payload.get("answer_format"),
                    stage=stage,
                    user_authenticated=bool(user),
                    phone=phone_redis,
                    email=email_redis,
                    context=context,
                    messages=messages_for_gpt
                )

                if (
                    len(context) == 1
                    and context[0].get("text")
                    and len(context[0]["text"]) > 1000
                    and ai_response.get("action") != "full_article"
                ):
                    logger.warning(f"[CONTRACT ERROR] LLM не вернула action=full_article при context=full_article! Ответ: {ai_response}")
                    # Fallback — выдаём текст напрямую, чтобы не сломать UX
                    return set_session_cookie(JSONResponse({
                        "reply": context[0]["text"],
                        "meta": {
                            "stage": stage,
                            "action": "full_article",
                            "article_id": pending_article_id,
                            "contract_error": True
                        }
                    }), session_id)

                # 5. Сохраняем сообщение ассистента (выдача полной статьи)
                try:
                    assistant_msg = Message(
                        session_id=session_id,
                        user_id=user.id if user else None,
                        role="assistant",
                        type="text",
                        status="ok",
                        content=ai_response["reply"],
                        meta=ai_response.get("usage", {}),
                    )
                    db.add(assistant_msg)
                except Exception as e:
                    logger.error(f"DB error while saving assistant message (full_article): {str(e)}")

                # 6. Обрезаем историю до 10 последних сообщений
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

                # Возврат пользователю через session cookie
                return set_session_cookie(JSONResponse({
                    "reply": ai_response["reply"],
                    "meta": {
                        "stage": stage,
                        "action": "full_article",
                        "article_id": pending_article_id
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
        msgs_keep = result.scalars().all()[::-1]  # старые сообщения в начало

        # Для обрезки истории — все сообщения по сессии
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

        # --- 3. Блок генерации ответа от OpenAI с памятью/контекстом ---
        ai_response = await ask_openai(
            content=content,
            msg_type=msg_type,
            answer_format=answer_format,
            stage=stage,
            user_authenticated=bool(user),
            phone=phone_redis,
            email=email_redis,
            context=context_chunks if context_chunks else [],
            messages=messages_for_gpt
        )
        logger.info(f"ai_response: {ai_response}")

        # --- Сохраняем сообщение ассистента (AI) ---
        try:
            assistant_msg = Message(
                session_id=session_id,
                user_id=user_id if 'user_id' in locals() else (user.id if user else None),
                role="assistant",
                type=msg_type,
                status="ok",
                content=ai_response["reply"],
                meta=ai_response.get("usage", {}),
            )
            db.add(assistant_msg)
        except Exception as e:
            logger.error(f"DB error while saving assistant message (AI): {str(e)}")

        response_payload = None

        # --- После этого — снова обрезаем историю до 10 последних сообщений ---
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

        # --- 1. Если GPT просит "предложить полную статью" (короткий ответ, ждем подтверждения)
        action = ai_response.get("action") or ai_response.get("fields", {}).get("action")
        article_id = ai_response.get("article_id") or ai_response.get("fields", {}).get("article_id")
        if action == "offer_full_article" and article_id:
            await redis.set(f"pending_full_article:{session_id}", article_id, ex=3600)

        # --- ai_response: reply, stage, fields, [token], [dev_creds]
        new_stage = ai_response.get('stage', stage)
        fields = ai_response.get('fields', {})
        if "action" in ai_response:
            fields["action"] = ai_response["action"]
        if "article_id" in ai_response:
            fields["article_id"] = ai_response["article_id"]
        logger.info(f"AI response: stage={new_stage} fields={fields}")

        # --- B. Сохраняем phone/email в Redis всегда (контекст сохраняется) ---
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
        
	    # --- A. BACKEND-ВАЛИДАЦИЯ КОДА для перехода с 1 на 2 этап ---
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

        # --- C. Жёсткая защита stage для авторизованных ---
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

        # --- Фиксируем stage и поля в Redis ---
        stage_key = f"stage:{session_id}"
        if allow:
            await redis.set(stage_key, new_stage, ex=12*60*60)
            logger.info(f"Stage updated: {stage} → {new_stage} session={session_id}")

        # --- Регистрируем пользователя ---
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
                    "Вы автоматически авторизованы и зарегистрированы! Теперь вам доступен расширенный функционал Leadinc.\n"
                    f"Ваши данные для входа:\n"
                    f"Телефон: {phone_final}\n"
                    f"Email: {email_final}\n"
                    f"Пароль: {password}\n"
                    "------------------------"
                )
                ai_response["reply"] = (ai_response.get("reply") or "") + dev_block
                logger.info(f"Новый пользователь зарегистрирован: email={email_final}, phone={phone_final}")
                print(f"[DEBUG] Generated password for {email_final}: {password}")
                logger.info(f"Final AI reply: {ai_response['reply']}")
                
                # Очищаем временные ключи (и autoочистка на 5 дней)
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
                        "login": email_final,    # или phone_final
                        "password": password     # ТОЛЬКО ДЛЯ DEV!
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
                # Кука с JWT-токеном (имя куки = как в fastapi_users.config, обычно "fastapiusersauth")
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

        # --- Сбор финального ответа и поддержка voice-ответа ---
        reply = ai_response.get("reply", "")
        if not isinstance(reply, str):
            reply = json.dumps(reply, ensure_ascii=False)

        # --- Централизованный return ответа ассистента (строго по answer_format) ---
        if answer_format == "voice":
            logger.warning(f"[DEBUG][TTS ENTRY] answer_format=voice, about to call TTS! reply={reply[:60]}")
            # Генерируем войс через TTS (Celery)
            tts_format = payload.get("tts_format", "mp3")
            if tts_format not in SUPPORTED_TTS_FORMATS:
                logger.warning(f"[VOICE] Некорректный tts_format '{tts_format}', принудительно mp3")
                tts_format = "mp3"
            logger.info(f"[VOICE] Выбран формат TTS: {tts_format}")

            try:
                from backend.tasks.tts import tts_task
                tts_result = await run_in_threadpool(
                    lambda: tts_task.apply_async(
                        args=[reply, None, str(getattr(user, "id", None)), session_id, tts_format]
                    ).get(timeout=60)
                )
                logger.info(f"[DEBUG] tts_result: {tts_result}")
                if tts_result and tts_result.get("status") == "ok" and tts_result.get("audio_url"):
                    response_payload = {
                        "reply_type": "voice",
                        "audio_url": tts_result["audio_url"],
                        "meta": tts_result.get("meta", {}),
                    }
                else:
                    logger.warning(f"[DEBUG][TTS FAIL] tts_result error or audio_url missing: {tts_result}")
                    response_payload = {
                        "reply_type": "voice",
                        "audio_url": None,
                        "meta": {
                            "stage": new_stage,
                            "usage": ai_response.get("usage", {}),
                            "fields": fields,
                            "tts_error": (
                                tts_result.get("error", "unknown") if 'tts_result' in locals() and tts_result else str(e) if                'e' in locals() else "TTS error"
                            )
                        }
                    }
            except Exception as e:
                logger.error(f"TTS voice generation failed: {e}")
                response_payload = {
                    "reply_type": "voice",
                    "audio_url": None,
                    "meta": {
                        "stage": new_stage,
                        "usage": ai_response.get("usage", {}),
                        "fields": fields,
                        "tts_error": (
                                tts_result.get("error", "unknown") if 'tts_result' in locals() and tts_result else str(e) if                'e' in locals() else "TTS error"
                        )
                    }
                }
        else:
            logger.warning(f"[DEBUG][NO TTS] answer_format={answer_format}, TTS не вызывается, ответ текстом.")
            response_payload = {
                "reply_type": "text",
                "reply": reply,
                "meta": {
                    "stage": new_stage,
                    "usage": ai_response.get("usage", {}),
                    "fields": fields,
                }
            }
        logger.debug(f"[RESPONSE_PAYLOAD]: {json.dumps(response_payload, ensure_ascii=False, indent=2)}")
        await db.commit()
        return set_session_cookie(JSONResponse(response_payload), session_id)

# Блок RAG логики
@ai_router.post("/rag")
async def rag_search(
    payload: ChatRequest,
    request: Request,
    user: User = Depends(current_active_user_optional),
):
    # 1. Проверяем авторизацию
    if not user:
        return JSONResponse(
            {"reply": "Поиск по базе Leadinc доступен только авторизованным пользователям.", "meta": {"reason": "unauthorized"}},
            status_code=403
        )

    # 2. Проверяем этап авторизации (stage 4)
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

    # 3. Получаем embedding запроса
    try:
        query_emb = await get_embedding(payload.content)
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        return JSONResponse(
            {"reply": "Ошибка обработки embedding. Попробуйте позже.", "meta": {"reason": "embedding_error"}},
            status_code=500
        )

    # 4. Поиск по ChromaDB
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

    # Склеиваем результат для ассистента (можешь доработать, если нужна отдельная логика)
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

@ai_router.post("/voice_upload")
async def voice_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),  # Или получи через cookie/Depends
    user: User = Depends(current_active_user_optional)
):
    # 1. Сохраняем файл на диск асинхронно
    audio_path = await save_upload_file(file)
    user_id = str(user.id) if user else None

    # 2. Кидаем задачу в Celery (важно — sync вызов в async-функции!)
    task = stt_task.apply_async(args=[audio_path, user_id, session_id])

    return {"task_id": task.id, "status": "pending"}

@ai_router.post("/tts")
async def tts_generate(
    text: str = Form(...),
    voice_id: str = Form(None),
    output_format: str = Form("mp3"),
    session_id: str = Form(...),
    user: User = Depends(current_active_user_optional)
):
    user_id = str(user.id) if user else None
    try:
        task = tts_task.apply_async(args=[text, voice_id, user_id, session_id, output_format])
        if not task or not task.id:
            logger.error(f"TTS Celery: Не удалось отправить задачу в очередь (task=None) | user={user_id}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "failed",
                    "error": "Не удалось отправить задачу TTS в очередь.",
                    "meta": {"stage": 4, "user_id": user_id}
                }
            )
        logger.info(f"TTS Celery: задача отправлена | task_id={task.id} | user={user_id}")
        return {"task_id": task.id, "status": "pending"}
    except Exception as e:
        logger.error(f"TTS Celery: Ошибка отправки задания | user={user_id} | error={e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": f"TTS Celery: Ошибка отправки задания: {e}",
                "meta": {"stage": 4, "user_id": user_id}
            }
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
        secure=True,      # Только если у тебя прод/https!
        samesite="lax"    # Или "strict", если выставляешь так при login!
    )
    response.delete_cookie(
        key="fastapiusersauth",
        path="/",
        httponly=True,
        secure=True,      # Только если у тебя прод/https!
        samesite="strict" # Или "lax" — смотри как при установке!
    )
    return response

# --- Поддержка/почта ---
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

# --- Аналитика/метрики ---
@app.middleware("http")
async def add_analytics_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-GA-Measurement-Id"] = GA_MEASUREMENT_ID or ""
    response.headers["X-Metrika-Id"] = METRIKA_ID or ""
    return response

app.include_router(ai_router)
