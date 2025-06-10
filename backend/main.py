import logging
from fastapi import FastAPI, Request, Depends, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Any
import os

print("DEBUG: Any imported:", Any)
print("DEBUG: MAIN.PY location:", __file__)
print("DEBUG: Working directory:", os.getcwd())

from backend.auth import (
    fastapi_users, auth_backend, require_active_subscription
)
from backend.config import (
    DEBUG, LOG_LEVEL, ADMIN_EMAIL, GA_MEASUREMENT_ID, METRIKA_ID, SUPPORT_EMAIL
)
from backend.email_utils import send_email
from backend.openai_utils import ask_openai
from backend.models import Message, Session
from backend.schemas import UserRead, UserCreate, ChatRequest, SupportRequest
from backend.database import SessionLocal
from sqlalchemy.ext.asyncio import AsyncSession

# ------------------ FastAPI app ------------------
app = FastAPI(
    title="Leadinc AI Assistant",
    description="AI SaaS Assistant (B2B)",
    debug=DEBUG
)

# ------------------ Логирование ------------------
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("leadinc-backend")

# ------------------ CORS ------------------------
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

# ------------------ Статические файлы ------------
MEDIA_DIR = Path(__file__).parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# ------------------ Auth routers -----------------
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

# =========== AI ROUTER: все ассистентные endpoint’ы через /ai ============
ai_router = APIRouter(prefix="/ai", tags=["ai"])

@ai_router.post("/chat")
async def chat(
    payload: ChatRequest,
    user=Depends(require_active_subscription),
    session: AsyncSession = Depends(lambda: SessionLocal()),
):
    content = payload.content
    msg_type = payload.type

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

    return JSONResponse({"reply": reply, "meta": usage})

@ai_router.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}

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

# ------------------ Сквозная аналитика ------------------
@app.middleware("http")
async def add_analytics_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-GA-Measurement-Id"] = GA_MEASUREMENT_ID or ""
    response.headers["X-Metrika-Id"] = METRIKA_ID or ""
    return response

# ------------------ Включаем ai_router ------------------
app.include_router(ai_router)

