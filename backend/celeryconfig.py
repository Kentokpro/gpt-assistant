"""
Celery configuration for Leadinc AI Assistant.
- Использует Redis как брокер/бэкенд (адрес из REDIS_URL)
- Конфигурирует две очереди: text (быстрые), audio (STT/TTS)
- Таймауты задач (SLA): soft 90s, hard 120s — см. требования
- Все переменные загружаются через backend.config
"""

import os
from backend.config import REDIS_URL
from kombu import Queue, Exchange

from backend.utils.audio_constants import (
    TTS_SOFT_TIME_LIMIT, TTS_HARD_TIME_LIMIT,
    STT_SOFT_TIME_LIMIT, STT_HARD_TIME_LIMIT
)

# Core Celery settings
broker_url = REDIS_URL
result_backend = REDIS_URL

# Task serializer
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]

# Queues (multi-stage best practice)
task_queues = [
    Queue("text", Exchange("text"), routing_key="text"),
    Queue("audio", Exchange("audio"), routing_key="audio"),
]

# Result expiration: храним статус задачи 15 минут (MVP)
result_expires = 900

# Retry policy (по дефолту, best practice)
task_default_retry_delay = 5      # секунд между попытками
task_max_retries = 3              # максимум 3 попытки

# Названия очередей по умолчанию
task_default_queue = "text"
task_routes = {
    "backend.tasks.chat.*": {"queue": "text"},
    "backend.tasks.stt.*": {"queue": "audio"},
    "backend.tasks.tts.*": {"queue": "audio"},
}

# Логирование Celery (только WARNING/ERROR, весь info — в основной логгер)
worker_hijack_root_logger = False
worker_redirect_stdouts = True
worker_log_color = True

# Имя воркера — для мониторинга (меняется при запуске, можно через env)
worker_hostname = os.getenv("CELERY_WORKER_NAME", "leadinc-ai-worker")

# (При необходимости добавить monitoring hooks — flower/prometheus, настройки тут)
