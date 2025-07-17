"""
tasks/__init__.py

Инициализация подпакета задач Celery для ai-assistant.
Обеспечивает:
- Общий логгер для всех тасков (tasks_logger)
- Импорт и экспорт задач: stt, tts, chat
- Экспорт функции log_error_to_db для централизованной записи ошибок в ErrorLog (используется всеми задачами)
"""

import logging

# Основной логгер для всех задач Celery
tasks_logger = logging.getLogger("leadinc-tasks")
tasks_logger.setLevel(logging.INFO)

# Импорт задач (фактическая реализация в отдельных файлах)
from .stt import stt_transcribe_voice_task  # noqa: F401
from .tts import tts_generate_voice_task    # noqa: F401
from .chat import chat_process_text_task    # noqa: F401

# Функция логирования ошибок в ErrorLog (используется всеми задачами)
# (Импортировать из backend.utils.error_log_utils если реализовано централизованно)
from backend.database import SessionLocal
from backend.models import ErrorLog
import uuid
from datetime import datetime

async def log_error_to_db(user_id, error, details=None):
    """
    Записывает ошибку в ErrorLog.
    Использовать во всех тасках при критических ошибках, SLA-превышениях, сбоях API и т.п.
    """
    async with SessionLocal() as db:
        try:
            error_log = ErrorLog(
                id=uuid.uuid4(),
                user_id=user_id,
                error=error,
                details=details,
                created_at=datetime.utcnow()
            )
            db.add(error_log)
            await db.commit()
        except Exception as e:
            tasks_logger.error(f"Ошибка при записи ErrorLog в БД: {e}")

# (Далее, если нужно, можно добавить дополнительные экспортируемые утилиты)


