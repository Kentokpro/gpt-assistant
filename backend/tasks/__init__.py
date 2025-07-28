"""
tasks/__init__.py

Инициализация подпакета задач Celery для ai-assistant.
Обеспечивает:
- Импорт и экспорт задач: stt, tts, chat
- Экспорт функции log_error_to_db для централизованной записи ошибок в ErrorLog (используется всеми задачами)
"""

import logging

# Глобальный логгер для задач Celery (используется всеми тасками)
tasks_logger = logging.getLogger("leadinc-tasks")
tasks_logger.setLevel(logging.INFO)

# Импорт Celery задач (реализация в отдельных файлах)
from .stt import stt_task         # noqa: F401
from .tts import tts_task         # noqa: F401
from .chat import process_text    # noqa: F401

# Импорт централизованного логгера ошибок (единая точка)
from backend.utils.error_log_utils import log_error_to_db


