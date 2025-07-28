"""
backend/tasks/chat.py

Celery-задача для text/chat очереди:
— Получает текстовый запрос, вызывает OpenAI (через openai_utils.ask_openai)
— Сохраняет оба сообщения (user, assistant) в Message
— SLA: логирует время, все ошибки через ErrorLog
"""
"""
Основная задача text/chat:
— Получает chat_payload (dict: content, user_id, session_id, meta и пр.)
— Вызывает ask_openai (async), сохраняет оба сообщения, логирует SLA, ошибки
— Все ошибки через log_error_to_db
"""

import logging
import time
import uuid
from datetime import datetime
import asyncio

from backend.openai_utils import ask_openai
from backend.database import SessionLocal
from backend.models import User, Message, Session
from backend.utils.error_log_utils import log_error_to_db
from backend.celery_worker import celery_app
from backend.utils.audio_constants import TTS_HARD_TIME_LIMIT, STT_HARD_TIME_LIMIT

logger = logging.getLogger("leadinc-backend")

# ——— Лимит времени задачи (максимум из TTS/STT, если понадобится)
HARD_TIME_LIMIT = max(TTS_HARD_TIME_LIMIT, STT_HARD_TIME_LIMIT)  # 60

@celery_app.task(bind=True, name="chat.process_text", queue="text", soft_time_limit=HARD_TIME_LIMIT, time_limit=HARD_TIME_LIMIT+10)
def process_text(self, chat_payload: dict):
    task_id = str(self.request.id)
    started_at = time.monotonic()
    utc_started = datetime.utcnow()
    user_id = chat_payload.get("user_id")
    session_id = chat_payload.get("session_id")

    logger.info(f"[chat] New chat-task {task_id}, user={user_id}, session={session_id}")

    try:
        with SessionLocal() as db:
            # — 1. Сохраняем входящее сообщение пользователя
            message = Message(
                id=uuid.uuid4(),
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

            # — 2. Вызов OpenAI (ответ ассистента) через run_until_complete
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ai_response = loop.run_until_complete(
                    ask_openai(
                        content=chat_payload.get("content"),
                        msg_type="text",
                        stage=chat_payload.get("stage", 4),
                        user_authenticated=bool(user_id),
                        phone=chat_payload.get("phone"),
                        email=chat_payload.get("email"),
                        context=chat_payload.get("context"),
                        messages=chat_payload.get("messages"),
                    )
                )
            finally:
                loop.close()

            elapsed = time.monotonic() - started_at
            utc_completed = datetime.utcnow()
            logger.info(f"[chat] Chat-task {task_id} complete, elapsed={elapsed:.2f}s, user={user_id}")

            # — 3. Сохраняем ответ ассистента
            assistant_msg = Message(
                id=uuid.uuid4(),
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
        # Централизованно логируем ошибку (ErrorLog)
        log_error_to_db(
            user_id=user_id,
            error="chat_failed",
            details={
                "task_id": task_id,
                "error": str(e),
                "elapsed": elapsed,
                "stage": chat_payload.get("stage"),
            }
        )
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "elapsed": elapsed,
        }
