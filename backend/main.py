import logging
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from auth import (
    fastapi_users, auth_backend, current_active_user, require_active_subscription
)
from config import (
    DEBUG, LOG_LEVEL, ADMIN_EMAIL, GA_MEASUREMENT_ID, METRIKA_ID, SUPPORT_EMAIL
)
from email_utils import send_email
from openai_utils import ask_openai
from models import Message, Session
from schemas import UserRead, UserCreate
from database import SessionLocal
from sqlalchemy.ext.asyncio import AsyncSession

# --- Логирование ---
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("leadinc-backend")

app = FastAPI(
    title="Leadinc AI Assistant",
    description="AI SaaS Assistant (B2B)",
    debug=DEBUG
)

# --- CORS ---
ALLOWED_ORIGINS = [
    "https://leadinc.ru",
    "https://gpt.leadinc.ru",
    "http://localhost:3000",      # Для разработки
    "http://127.0.0.1:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Статические файлы ---
MEDIA_DIR = Path(__file__).parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# --- Auth routers ---
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

# --- Endpoint для /chat ---
@app.post("/chat", tags=["ai"])
async def chat(
    request: Request,
    user=Depends(require_active_subscription),
    session: AsyncSession = Depends(SessionLocal),
):
    data = await request.json()
    content = data.get("content")
    msg_type = data.get("type", "text")
    meta = {}

    if msg_type != "text":
        raise HTTPException(status_code=400, detail="Мультимодальность пока не реализована.")

    try:
        openai_result = await ask_openai(content, msg_type)
        reply = openai_result["text"]
        usage = openai_result["usage"]
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI ошибка")

    try:
        user_msg = Message(
            session_id=None,
            user_id=user.id,
            role="user",
            type=msg_type,
            status="ok",
            content=content,
            meta={},
        )
        session.add(user_msg)
        await session.commit()
        assistant_msg = Message(
            session_id=None,
            user_id=user.id,
            role="assistant",
            type="text",
            status="ok",
            content=reply,
            meta=usage,
        )
        session.add(assistant_msg)
        await session.commit()
    except Exception as e:
        logger.error(f"DB error: {str(e)}")

    return JSONResponse(
        {
            "reply": reply,
            "meta": usage,
        }
    )

# --- Endpoint истории сообщений пользователя ---
@app.get("/history", tags=["ai"])
async def history(
    user=Depends(require_active_subscription),
    session: AsyncSession = Depends(SessionLocal),
    limit: int = 50,
):
    result = await session.execute(
        Message.__table__.select()
        .where(Message.user_id == user.id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = [dict(row) for row in result.fetchall()]
    return messages

# --- Healthcheck endpoint ---
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}

# --- Email рассылка (пример endpoint) ---
@app.post("/support", tags=["support"])
async def support_request(
    request: Request,
    user=Depends(require_active_subscription)
):
    data = await request.json()
    subject = data.get("subject", "Support request")
    message = data.get("message", "")
    await send_email(
        to=SUPPORT_EMAIL,
        subject=f"[SUPPORT] {subject}",
        body=message,
        from_email=user.email,
    )
    return {"status": "sent"}

# --- Google Analytics и Яндекс.Метрика (сквозная мета) ---
@app.middleware("http")
async def add_analytics_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-GA-Measurement-Id"] = GA_MEASUREMENT_ID or ""
    response.headers["X-Metrika-Id"] = METRIKA_ID or ""
    return response
