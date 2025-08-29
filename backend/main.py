
"""
Leadinc AI Backend — полный рефакторинг (2025-08)
- Мягкая валидация через OpenAI-ассистента
- DEV ONLY: временный вывод логина/пароля через чат
- Управление этапами через ассистента, backend валидирует только переходы и защищает от скачков
- Подробное логирование всех ключевых событий
- Защита stage, нет дублей логики, правила валидации только в prompt
- После авторизации stage не может быть <4, код и регистрацию не спрашиваем
- Автоочистка неактуальных ключей Redis после регистрации (5 дней)
"""

import logging
from dotenv import load_dotenv
load_dotenv(dotenv_path="/root/ai-assistant/backend/.env.backend", override=True)
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
from backend.chroma_utils import filter_chunks, search_chunks_by_embedding

from backend.auth import (
    fastapi_users, auth_backend, require_active_subscription, current_active_user_optional, get_jwt_strategy
)
from backend.config import (
    DEBUG, LOG_LEVEL, ADMIN_EMAIL, GA_MEASUREMENT_ID, METRIKA_ID, SUPPORT_EMAIL, SESSION_COOKIE_NAME, FAQ_COLLECTION_NAME, ANALYTICS_COLLECTION_NAME
)

from backend.email_utils import send_email
from backend.openai_utils import ask_openai, get_embedding
from backend.models import User, Message, Session as SessionModel
from backend.schemas import UserRead, UserCreate, ChatRequest, SupportRequest
from backend.database import SessionLocal

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
    "без звука",
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

from backend.config import (
    DEBUG, LOG_LEVEL, ADMIN_EMAIL, GA_MEASUREMENT_ID, METRIKA_ID, SUPPORT_EMAIL, SESSION_COOKIE_NAME, FAQ_COLLECTION_NAME, ANALYTICS_COLLECTION_NAME
)

# Превратим LOG_LEVEL в числовой
LEVEL = logging.getLevelName(str(LOG_LEVEL).upper())
if not isinstance(LEVEL, int):
    LEVEL = logging.INFO  # дефолт на всякий случай

logging.basicConfig(
    level=LEVEL,
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
        # импорт задачи (worker может быть не поднят)
        tid = str(uuid.uuid4())
        dummy_task = tts_task.AsyncResult(tid)
        health["celery"] = "ok"
    except Exception as e:
        health["celery"] = f"fail: {e}"

    # OpenAI KEY
    try:
        openai_key = os.getenv("OPENAI_API_KEY") or "NO"
        health["openai_api_key"] = "ok" if openai_key and openai_key != "NO" else "not_set"
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
MESSAGE_LIMITS = [500, 500, 500]
PROJECT_LIMIT_PER_DAY = 10
USER_LIMIT = 500

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
    analytic_main = None
    fields = {}
    dashboard = None
    confirm_used = False

    logger.info(f"==== [UNIVERSAL CHAT ENDPOINT] START ====")

    session_id = get_or_create_session_id(request)
    logger.info(f"[INIT] session_id={session_id}")

    stage_key = f"stage:{session_id}"
    raw_stage = await redis.get(stage_key)
    if raw_stage is None:
        stage = 1
        await redis.set(stage_key, stage, ex=12*60*60)
    else:
        stage = int(raw_stage)
        await redis.expire(stage_key, 12*60*60)
    logger.info(f"[REG] stage={stage} (для REGISTRATION)")

    new_stage = stage
    stage_out = stage
    emit_stage = False

    phone_redis = None
    email_redis = None
    context_chunks = []
    messages_for_gpt = []
    answer_format = None
    msg_type = None
    content = None
    logger.info(f"[INIT] session_id={session_id}, stage={stage}")

    ai_response = {
        "scenario": "",     # нейтрально
        "stage": stage,     # текущий stage
        "action": "",       # пустой action
        "fields": {},       # пустые поля
        "reply": ""         # пустой reply
    }

    # Блок обработки аудио STT
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
    logger.info(
        f"[ПРИЁМ] Тип входа={msg_type}; запрошенный формат={answer_format}; "
        f"контент='{str(content)[:120]}'"
    )

    # --- Определяем, просит ли пользователь текст/войс через триггеры ---
    is_voice_trigger = any(trigger in content_lower for trigger in VOICE_TRIGGER_PHRASES)
    is_text_trigger = any(trigger in content_lower for trigger in TEXT_TRIGGER_PHRASES)
    logger.info(f"[ОТЛАДКА] Триггеры: голос={is_voice_trigger}, текст={is_text_trigger}")


    # --- Логика выбора формата ответа ---
    if msg_type == "voice" and (answer_format == "text" or is_text_trigger):
        answer_format = "text"
        logger.info("[ФОРМАТ ОТВЕТА] Вход был голосом, но запрошен текст — отвечаем текстом")
    elif msg_type == "voice":
        answer_format = "voice"
        logger.info("[ФОРМАТ ОТВЕТА] Вход был голосом — отвечаем голосом")
    elif answer_format == "voice" or is_voice_trigger:
        answer_format = "voice"
        logger.info("[ФОРМАТ ОТВЕТА] Обнаружен голосовой триггер — отвечаем голосом")
    else:
        answer_format = "text"
        logger.info("[ФОРМАТ ОТВЕТА] Отвечаем текстом (значение по умолчанию)")

    logger.info(f"[PAYLOAD] content={content!r}, msg_type={msg_type}, answer_format={answer_format}")

    logger.info(f"--- NEW CHAT REQ --- session={session_id} user={getattr(user, 'id', None)} content='{content[:40]}'")

    # ====== Вся логика в транзакции БД (история/лимиты/регистрация и т.п.) ======
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
            await redis.set(stage_key, stage, ex=12 * 60 * 60)
            logger.info(f"User is authorized. Forcing stage=4 for user_id={user.id}, session={session_id}")


        # --- 3. Лимиты, спам, guest limits ---
        lim_prefix = f"{user.id}" if user else session_id
        msg_count_key = f"msg_count:{lim_prefix}"
        msg_count = int(await redis.get(msg_count_key) or 0)

        # === Мотивация регистрации — только 1 раз за сессию на 10-м сообщении гостя ===
        if not user and msg_count == 10 and not await redis.get(f"motivation_shown:{session_id}"):
            await redis.set(f"motivation_shown:{session_id}", 1, ex=12*60*60)
            return set_session_cookie(JSONResponse({
                "reply": (
                    "Дарим подарки первым пользователям в честь запуска!"
                    "Зарегистрируйся сейчас и получи 10 бесплатных лидов!\n\n"
                ),
                "meta": {
                    "stage": stage,
                    "reason": "motivate_register",
                    "msg_count": msg_count
                }
            }), session_id)
        
        # Flood protection (гостям)
        if not user:
            client_ip = request.client.host
            ip_ban_key = f"guest_ip_ban:{client_ip}"
            is_banned = await redis.exists(ip_ban_key)
            if is_banned:
                logger.warning(f"[ANTISPAM] блокировка по IP для гостя: {client_ip}")
                return set_session_cookie(JSONResponse({
                    "reply": "Временная блокировка на 2 часа.",
                    "meta": {"stage": stage, "reason": "ip_ban"}
                }), session_id)

        zset_key = f"guest_flood:{session_id}"
        now = current_timestamp()
        await redis.zremrangebyscore(zset_key, 0, ten_minutes_ago())
        await redis.zadd(zset_key, {str(now): now})
        guest_msgs = await redis.zcount(zset_key, ten_minutes_ago(), now)
        if guest_msgs > 100:
            client_ip = request.client.host
            ip_ban_key = f"guest_ip_ban:{client_ip}"
            await redis.set(ip_ban_key, 1, ex=2*60*60)
            logger.warning(f"[ANTISPAM] Превышение сообщений — IP BAN: session={session_id}, ip={client_ip}")
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
                    "meta": {"reason": "project_limit"}
                }), session_id)
            block_key = f"user_block:{user.id}"
            is_blocked = await redis.exists(block_key)
            if is_blocked:
                logger.warning(f"User is blocked for spamming. user_id={user.id}")
                return set_session_cookie(JSONResponse({
                    "reply": "Вы временно заблокированы из-за превышения лимита сообщений. Пауза 2 часа.",
                    "meta": {"reason": "user_blocked"}
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
                    "meta": {"reason": "msg_limit_exceeded"}
                }), session_id)

        # --- [БЛОК] Обработка коротких подтверждений пользователя для выдачи полной статьи ---
        # 1. короткие подтверждения — только целые слова
        CONFIRM_WORDS = {
            "да","ок","окей","ага","угу","yes","sure", "давай", "дальше", "ещё", "еще", "продолжи", "продолжай", "погнали", "расскажи", "подробнее", "поясни", "больше", "полную", "полностью", "весь", "всю", "развёрнуто", "расширь", "давайте",
            "да, давай", "давай полностью", "расскажи дальше", "расскажи полностью", "полный текст", "покажи всё", "расскажи до конца", "да, интересно",  "ещё расскажи","расширенно", "расширенный ответ", "весь текст","больше","продолжай","продолжи", "да, интересно", "да, хочу", "хочу больше", "расскажи все", "расскажи всё",
        }
        
        def is_confirm_trigger(txt: str) -> bool:
            t = (txt or "").strip().lower()
            if t in CONFIRM_WORDS:
                return True
            # Допускаем частые формы подтверждения без строгого совпадения
            return t.startswith("да") or t.startswith("ок") or t in {"ок","окей","ага","угу","yes","sure", "давай", "дальше", "ещё", "еще", "продолжи", "продолжай", "погнали", "расскажи", "подробнее", "поясни", "больше"}

        # Хранилище последней статьи FAQ по сессии
        FAQ_LAST_AID_KEY = f"faq:last_article_id:{session_id}"
        FAQ_LAST_SCENARIO_KEY = f"last_ai_scenario:{session_id}"

        async def _faq_load_by_id(aid: str) -> tuple[str, str]:
            # Вернуть (title, full_text) для article_id или пустые строки, если не найдено.
            chunks = await filter_chunks(article_id=aid)
            if not chunks:
                logger.warning(f"[FAQ][ПОИСК] Не удалось найти статью по article_id={aid}")
                return "", ""
            title = chunks[0].get("title") or ""
            full_text = chunks[0].get("text") or ""
            logger.info(f"[FAQ][ПОИСК] Успешно найдено: article_id={aid}, заголовок='{title[:80]}'")
            return title, full_text

        user_input = (content or "")
        user_input_norm = user_input.strip().lower()
        confirm_hit = is_confirm_trigger(user_input_norm)

        # Читаем из Redis, что мы отдавали в прошлый раз
        last_aid = await redis.get(FAQ_LAST_AID_KEY) or ""
        last_scenario = await redis.get(FAQ_LAST_SCENARIO_KEY) or ""       
        logger.info(f"[FAQ] Прочитали состояние: last_article_id={last_aid!r}, last_scenario={last_scenario!r}")

        faq_context = None
        faq_article_id = None

        if confirm_hit and last_scenario and last_scenario.upper() == "FAQ":
            logger.info("[FAQ][CONFIRM] Получен confirm-триггер в сценарии FAQ")
            if last_aid:
                title, full_text = await _faq_load_by_id(last_aid)
                if full_text:
                    faq_article_id = last_aid
                    faq_context = {"faq_article": {"article_id": last_aid, "title": title, "full_text": full_text}}
                    confirm_used = True  # <— отмечаем, что confirm будет потреблён
                    logger.info(f"[FAQ][CONFIRM] Повтор ранее выданной статьи: article_id={last_aid}, title='{title[:60]}'")
                else:
                    logger.warning(f"[FAQ][CONFIRM] В Redis сохранён article_id={last_aid}, но текста нет — confirm пропущен")
            else:
                logger.info("[FAQ][CONFIRM] Подтверждение получено, но article_id не сохранён — попросим уточнить вопрос")
        
        if faq_context:
            logger.info(f"[FAQ] Готов контекст статьи для LLM: article_id={faq_article_id!r}")
        else:
            logger.info("[FAQ] Confirm-контекст отсутствует — решение за LLM (router/поиск через tools).")

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


# 1. Назначение: сократить летучую память истории для LLM.
# 2. Изменение: limit(3) вместо limit(10); чистка, если >3 (было >10).
# 3. Причина: снизить расход токенов и «прилипчивость» контекста.
        messages_for_gpt = []
        q = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(3)   # было .limit(10)
        )
        result = await db.execute(q)
        msgs_keep = result.scalars().all()[::-1]  # старые сообщения в начало

        messages_for_gpt = []
        for m in msgs_keep:
            payload = {"role": m.role, "content": m.content}
            if m.role == "assistant":
                try:
                    m_meta = m.meta or {}
                    # Если на шаге shortlist аналитики мы сохранили fields.list — отдадим LLM JSON: reply+fields
                    if isinstance(m_meta, dict) and m_meta.get("fields"):
                        payload["content"] = json.dumps(
                            {"reply": m.content, "fields": m_meta["fields"]},
                            ensure_ascii=False
                        )
                except Exception:
                    pass
            messages_for_gpt.append(payload)

        # Для обрезки истории — все сообщения по сессии
        q_all = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
        )
        result_all = await db.execute(q_all)
        all_msgs = result_all.scalars().all()
        if len(all_msgs) > 3:   # было > 10
            ids_keep = set(msg.id for msg in msgs_keep)
            ids_del = [msg.id for msg in all_msgs if msg.id not in ids_keep]
            if ids_del:
                await db.execute(
                    Message.__table__.delete().where(Message.id.in_(ids_del))
                )
        messages_for_gpt = [{"role": msg.role, "content": msg.content} for msg in msgs_keep]

        # Собираем контекст для confirm (если был)
        single_pass_context = {}
        if faq_context:
            single_pass_context.update(faq_context)
            logger.info(f"[LLM] В контекст передана статья FAQ (faq_article): {json.dumps(faq_context, ensure_ascii=False)[:160]}...")

        # Единственный вызов "мозга" — LLM решает сценарий и сама ходит в RAG через tools
        try:
            logger.info("[LLM] Единичный вызов ask_openai запущен")
            ai_response = await ask_openai(
                content=content,
                msg_type=msg_type,
                answer_format=answer_format,
                stage=stage,
                user_authenticated=bool(user),
                phone=phone_redis,
                email=email_redis,
                context=single_pass_context,
                messages=messages_for_gpt          # летучая память последних 10 сообщений
            )
            dashboard = ai_response.get("dashboard") if isinstance(ai_response, dict) else None    
        except Exception as e:
            logger.error(f"[LLM] Ошибка при вызове ask_openai: {e}")
            ai_response = {
                "scenario": "OFFTOPIC",
                "stage": stage,
                "action": "smalltalk",
                "fields": {},
                "reply": "Техническая заминка. Давайте попробуем ещё раз?"
            }


        scenario_lock = (ai_response.get("scenario") or "OFFTOPIC").upper()
        action = ai_response.get("action") or (ai_response.get("fields") or {}).get("action")
        
        raw_fields = (ai_response.get("fields") or {}).copy()
        ALLOWED_FIELDS = {
            "FAQ": {"action", "article_id"},
            "ANALYTICS": {"action", "query", "niche", "selection", "list"},
            "REGISTRATION": {"code", "phone", "email", "niche", "city"},
            "OFFTOPIC": set()
        }
        sanitized_fields = {k: v for k, v in raw_fields.items() if k in ALLOWED_FIELDS.get(scenario_lock, set())}
        if "action" in ai_response and ("action" in ALLOWED_FIELDS.get(scenario_lock, set())):
            sanitized_fields["action"] = ai_response["action"]

        # Только для FAQ разрешаем article_id
        if scenario_lock == "FAQ":
            aid = ai_response.get("article_id") or sanitized_fields.get("article_id")
            if aid is not None:
                aid_str = str(aid).strip()
                if aid_str and aid_str.lower() not in ("none", "null", "nan", "0"):
                    ai_response["article_id"] = aid_str
                    sanitized_fields["article_id"] = aid_str
            # Сохранение последней статьи (TTL=1h) — только когда LLM реально вернула article_id
            if ai_response.get("article_id"):
                try:
                    await redis.set(FAQ_LAST_AID_KEY, ai_response["article_id"], ex=3600)  # TTL 1 час
                    logger.info(f"[FAQ][STATE] Сохранён article_id={ai_response['article_id']} (TTL=1h)")
                except Exception as err:
                    logger.warning(f"[FAQ][STATE] Ошибка сохранения article_id в Redis: {err}")
        else:
            ai_response.pop("article_id", None)
            sanitized_fields.pop("article_id", None)

        ai_response["fields"] = sanitized_fields
        fields = sanitized_fields

        # Фиксируем last_ai_* ТОЛЬКО ПОСЛЕ ответа LLM (а не до него)
        try:
            await redis.set(f"last_ai_scenario:{session_id}", scenario_lock, ex=12*60*60)
            await redis.set(f"last_ai_action:{session_id}", action or "", ex=12*60*60)
            logger.info(f"[STATE] last_ai = {scenario_lock}/{action}")
        except Exception as e:
            logger.warning(f"[STATE] failed to store last_ai_*: {e}")

# STAGE единый верификатор
        # Работает только для REGISTRATION. Для остальных сценариев stage не меняем и наружу не отдаём.
        desired_stage = ai_response.get("stage", stage)   # что запросил ассистент
        if scenario_lock == "REGISTRATION":
            emit_stage = True
            # Авторизованный — сразу финальная стадия
            if user:
                desired_stage = 4
            # Разрешён только stay или +1
            if not (desired_stage == stage or desired_stage == stage + 1):
                logger.warning(f"Прыжок stage запрещён в REGISTRATION: {stage} → {desired_stage}")
                return set_session_cookie(JSONResponse({
                    "reply": "Неожиданная ошибка! Давай попробуем ещё раз.",
                    "meta": {"stage": stage, "reason": "stage_jump"}
                }), session_id)
            # Спец-проверка перехода 1→2: валидируем=т код прежде чем менять stage
            if (not user) and stage == 1 and desired_stage == 2:
                user_code = fields.get("code")
                if not user_code:
                    logger.warning(f"Код не найден в fields, stage=1→2, session={session_id}")
                    return set_session_cookie(JSONResponse({
                        "reply": "Код не распознан. Введите 6-значный код из Telegram.",
                        "meta": {"stage": 1, "reason": "code_missing"}
                    }), session_id)
                code_key = f"real_code:{user_code}"
                if not (await redis.exists(code_key)):
                    logger.warning(f"Невалидный код: {user_code}, session={session_id}")
                    return set_session_cookie(JSONResponse({
                        "reply": "Введённый код неверен или устарел. Запросите новый код в Telegram-боте.",
                        "meta": {"stage": 1, "reason": "code_invalid"}
                    }), session_id)
                await redis.delete(code_key)
                logger.info(f"Код принят: {user_code}, session={session_id}")
            # Переход прошёл проверки — фиксируем
            new_stage = desired_stage
            await redis.set(stage_key, new_stage, ex=12*60*60)
            stage_out = new_stage
            logger.info(f"[REG] Stage updated: {stage} → {new_stage} session={session_id}")
        else:
            # FAQ / ANALYTICS / OFFTOPIC — stage не трогаем и не отдаем
            logger.info(f"[NON-REG] Stage unchanged: {stage} (scenario={scenario_lock})")

        # --- Унифицированный выбор коллекции и второй вызов ассистента (FAQ ИЛИ ANALYTICS) ---
        action   = ai_response.get("action") or ai_response.get("fields", {}).get("action")
        logger.info(f"[ROUTER] scenario_lock={scenario_lock} action={action} stage={stage}")

        # --- Сохраняется глобальное сообщение ассистента в БД ---               
        try:
            _assistant_type = "text"
            if answer_format == "voice":
                _assistant_type = "text"

            _reply_db = ai_response.get("reply", "")
            if not isinstance(_reply_db, str):
                try:
                    _reply_db = json.dumps(_reply_db, ensure_ascii=False)
                except Exception:
                    _reply_db = str(_reply_db)
            # Доп. защита: не даём в БД пустую «пыль» (пробелы/пустота)
            _reply_db = (_reply_db or "")
            _meta_payload = {}
            
            try:
                if (scenario_lock == "ANALYTICS") and isinstance(ai_response, dict):
                    _fld = ai_response.get("fields") or {}
                    if isinstance(_fld, dict) and _fld.get("list"):
                        # сохраняется только то, что нужно модели на следующем шаге
                        _meta_payload = {
                            **(ai_response.get("usage", {}) or {}),
                            "fields": {"list": _fld["list"]}
                        }
            except Exception:
                _meta_payload = {}               

            assistant_msg = Message(
                session_id=session_id,
                user_id=user_id if 'user_id' in locals() else (user.id if user else None),
                role="assistant",
                type=_assistant_type,
                status="ok",
                content=_reply_db,
                meta=_meta_payload,
            )
            db.add(assistant_msg)
        except Exception as e:
            logger.error(f"DB error while saving assistant message (AI): {str(e)}")

        # --- После этого — снова обрезается история до 10 последних сообщений ---
        q = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(3) # было .limit(10)
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
        if len(all_msgs) > 3: # было > 10
            ids_keep = set(msg.id for msg in msgs_keep)
            ids_del = [msg.id for msg in all_msgs if msg.id not in ids_keep]
            if ids_del:
                await db.execute(
                    Message.__table__.delete().where(Message.id.in_(ids_del))
                )

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

        # --- Регистрируем пользователя ---
        if (not user) and stage == 3 and new_stage == 4 and phone_final and email_final:
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
                # 2. Генерируем пароль, хешируем
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

                # 3. Привязываем сессию к user_id
                q = select(SessionModel).where(SessionModel.id == session_id)
                res = await db.execute(q)
                session_db = res.scalar_one_or_none()
                if session_db and not session_db.user_id:
                    session_db.user_id = user_obj.id

                # 4. Генерируем JWT
                jwt_strategy = get_jwt_strategy()
                
                # 5. Генерируем уникальный промокод с порядковым номером (по IP)
                guest_ip = request.client.host  # получаем IP гостя
                promo_counter_key = f"promo_counter:{guest_ip}"
                promo_number = await redis.incr(promo_counter_key)
                promo_code = f"LEAD{promo_number:03d}"  # LEAD001, LEAD002 и т.д.
                
                # 6. Логируем выдачу промокода
                promo_log_key = f"promo_issued:{guest_ip}:{promo_number:03d}"
                promo_log_value = f"{email_final}|{phone_final}|{int(time.time())}"
                await redis.set(promo_log_key, promo_log_value)

                # 7. Формируем финальный текст для пользователя с промокодом
                promo_text = (
                    f"\n\n 🎁 Ваш промокод на 10 бесплатных телефонных номеров: {promo_code}\n"
                    f"Порядковый номер регистрации: {promo_number:03d}"
                )
                token = await jwt_strategy.write_token(user_obj)

                # 8. dev info
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
                ai_response["reply"] = (ai_response.get("reply") or "") + promo_text + dev_block
                logger.info(f"Новый пользователь зарегистрирован: email={email_final}, phone={phone_final}, promo={promo_code}, ip={guest_ip}, номер={promo_number:03d}")
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

                # Отправляем финальный ответ и куки
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
        # НОРМАЛИЗАЦИЯ reply ДЛЯ ВЫДАЧИ/ОЗВУЧКИ ===
        reply = ai_response.get("reply", "")
        if not isinstance(reply, str):
            try:
                reply = json.dumps(reply, ensure_ascii=False)
            except Exception:
                reply = str(reply)

        reply = (reply or "")
        if not reply.strip():
            reply = (
                "Упс, неожиданная ошибка 4. Давайте попробуем снова."
            )

        # --- Централизованный return ответа ассистента (строго по answer_format) ---
        meta_base = {
            "usage": ai_response.get("usage", {}),
            "fields": fields,
        }
        if emit_stage:  # только для REGISTRATION
            meta_base["stage"] = stage_out

        if answer_format == "voice":
            _preview = (reply or "")
            if not isinstance(_preview, str) or not _preview.strip():
                # Нет текста для озвучки — честный фолбэк в текст
                logger.warning("[TTS] Пропуск озвучки: пустой/невалидный текст → текстовый ответ")
                response_payload = {
                    "reply_type": "text",
                    "reply": "Что то горло болит, ответить смогу текстом.",
                    "meta": {**meta_base, "tts_skipped": "empty_reply"}
                }
            else:
                logger.info(f"[TTS] Генерация аудио-ответа. Превью текста: {reply[:60]!r}")
                # Генерируем voice через TTS (Celery)
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
                            "meta": meta_base,
                        }
                        try:
                            assistant_msg.type = "voice"
                        except Exception:
                            pass
                    else:
                        logger.warning(f"[TTS] Сбой генерации: нет audio_url или ошибка. Ответ воркера: {tts_result}")
                        err_meta = dict(meta_base)
                        err_meta["tts_error"] = (tts_result.get("error", "unknown") if tts_result else "TTS error")
                        response_payload = {
                            "reply_type": "text",
                            "reply": reply,
                            "meta": err_meta,
                        }

                except Exception as e:
                    logger.error(f"TTS voice generation failed: {e}")
                    err_meta = dict(meta_base)
                    err_meta["tts_error"] = str(e)
                    response_payload = {
                        "reply_type": "text",
                        "reply": reply,
                        "meta": err_meta,
                    }

        else:
            logger.info("[ВЫДАЧА] Ответ текстом (TTS не требуется)")
            response_payload = {
                "reply_type": "text",
                "reply": reply,
                "meta": meta_base,
            }
        if dashboard:
            logger.info(f"[DASHBOARD PAYLOAD] {json.dumps(dashboard, ensure_ascii=False, indent=2)}")
            response_payload["dashboard"] = dashboard

        try:
            safe_log = dict(response_payload)
            r = safe_log.get("reply")
            if isinstance(r, str) and len(r) > 500:
                safe_log["reply"] = r[:500] + "…"
            logger.warning(f"[RESPONSE TO FRONT]: {json.dumps(safe_log, ensure_ascii=False, indent=2)}")
        except Exception as e:
            logger.error(f"[LOGGING ERROR] Can't dump response_payload: {e}")

        logger.debug(f"[RESPONSE_PAYLOAD]: {json.dumps(response_payload, ensure_ascii=False, indent=2)}")
        await db.commit()
        
        try:
            reply_text = response_payload.get("reply") if isinstance(response_payload, dict) else ""
            is_error_reply = isinstance(reply_text, str) and reply_text.startswith("Случайная ошибка")
            effective_action = action or (fields.get("action") if isinstance(fields, dict) else "")
            if confirm_used and scenario_lock == "FAQ" and effective_action == "full_article" and not is_error_reply:
                await redis.delete(FAQ_LAST_AID_KEY)
                logger.info("[FAQ][CONFIRM] last_article_id очищен после успешной выдачи full_article")
        except Exception as e:
            logger.warning(f"[FAQ][CONFIRM] Ошибка отложенного удаления last_article_id: {e}")

        return set_session_cookie(JSONResponse(response_payload), session_id)


@ai_router.post("/voice_upload")
async def voice_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
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
                    "meta": {"user_id": user_id}
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
                "meta": {"user_id": user_id}
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
    resp = JSONResponse({"detail": "Logout complete"})

# патч с точным удалением куки после логаута.
    for cookie_name in ("sessionid", SESSION_COOKIE_NAME, "fastapiusersauth"):
        resp.delete_cookie(
            key=cookie_name,
            path="/",
            secure=True,      
            samesite="lax"    
        )

    resp.set_cookie(
        key="sessionid",
        value="",
        max_age=0,
        expires=0,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return resp


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
