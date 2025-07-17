"""
chat.py — Celery-задача для text/chat очереди
— Принимает текстовый запрос, вызывает OpenAI (через openai_utils.ask_openai)
— Сохраняет ответ в Message (type='text')
— SLA: логирует время обработки, ошибки — в ErrorLog
— Требует:
    - openai_utils.py
    - models.py (User, Message, ErrorLog, Session)
    - celeryconfig.py
    - celery_worker.py (инициализация celery)
    - .env.backend с OPENAI_API_KEY
    - Логгер leadinc-backend
"""

import logging
import time
import uuid
from datetime import datetime

from backend.openai_utils import ask_openai
from backend.database import SessionLocal
from backend.models import User, Message, Session, ErrorLog
from backend.config import LOG_LEVEL

logger = logging.getLogger("leadinc-backend")
logger.setLevel(LOG_LEVEL)

# ——— Celery импортируется из celery_worker.py (важно: не делать новый celery instance!)
from backend.celery_worker import celery_app

@celery_app.task(bind=True, name="chat.process_text", queue="text", soft_time_limit=45, time_limit=60)
def process_text(self, chat_payload: dict):
    """
    Основная задача text/chat: 
    — Получает chat_payload (dict: content, user_id, session_id, meta и пр)
    — Вызывает ask_openai, сохраняет ответ, логирует SLA, ошибки
    — ErrorLog при ошибках, SLA в meta
    """
    task_id = str(self.request.id)
    started_at = time.monotonic()
    utc_started = datetime.utcnow()
    logger.info(f"[chat] New chat-task {task_id}, user={chat_payload.get('user_id')}, session={chat_payload.get('session_id')}")
    try:
        # 1. Сохраняем user/message/session
        with SessionLocal() as db:
            user_id = chat_payload.get("user_id")
            session_id = chat_payload.get("session_id")
            user = db.query(User).filter_by(id=user_id).first() if user_id else None
            session = db.query(Session).filter_by(id=session_id).first() if session_id else None

            # 2. Сохраняем входящее сообщение
            message = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                role="user",
                type="text",
                status="ok",
                content=chat_payload.get("content"),
                meta=chat_payload.get("meta", {}),
                created_at=utc_started,
            )
            db.add(message)
            db.commit()

            # 3. Вызов ask_openai (ответ ассистента)
            ai_response = ask_openai(
                content=chat_payload.get("content"),
                msg_type="text",
                stage=chat_payload.get("stage", 4),
                user_authenticated=True if user_id else False,
                phone=chat_payload.get("phone"),
                email=chat_payload.get("email"),
                context=chat_payload.get("context"),
                messages=chat_payload.get("messages"),
            )
            # (sync — для Celery 5.x, если будет async нужен run_until_complete)

            elapsed = time.monotonic() - started_at
            utc_completed = datetime.utcnow()
            logger.info(f"[chat] Chat-task {task_id} complete, elapsed={elapsed:.2f}s, user={user_id}")

            # 4. Сохраняем ответ ассистента
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                type="text",
                status="ok",
                content=ai_response.get("reply", ""),
                meta={
                    "usage": ai_response.get("usage", {}),
                    "elapsed": elapsed,
                    "task_id": task_id,
                },
                created_at=utc_completed,
            )
            db.add(assistant_msg)
            db.commit()

            # 5. Возврат результата
            return {
                "task_id": task_id,
                "status": "done",
                "reply": ai_response.get("reply", ""),
                "meta": {
                    "elapsed": elapsed,
                    "task_id": task_id,
                    "usage": ai_response.get("usage", {}),
                },
            }
    except Exception as e:
        elapsed = time.monotonic() - started_at
        logger.error(f"[chat] Error in chat-task {task_id}: {e}")
        # ErrorLog пишем в БД
        with SessionLocal() as db:
            errorlog = ErrorLog(
                id=str(uuid.uuid4()),
                user_id=chat_payload.get("user_id"),
                error="chat_failed",
                details={
                    "task_id": task_id,
                    "error": str(e),
                    "elapsed": elapsed,
                    "stage": chat_payload.get("stage"),
                },
                created_at=datetime.utcnow(),
            )
            db.add(errorlog)
            db.commit()
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "elapsed": elapsed,
        }
