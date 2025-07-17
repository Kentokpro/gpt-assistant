"""
Celery worker for Leadinc AI Assistant.
- Запускает задачи: chat (text), stt (Whisper), tts (ElevenLabs)
- Можно запускать на отдельные очереди: text (chat) и audio (stt/tts)
- Использует celeryconfig.py для настроек брокера, таймингов, SLA
- Все события и ошибки — в leadinc-backend.log
"""

import logging
import os

from celery import Celery, signals

# Важно: путь "backend.celeryconfig" — именно так, чтобы Celery нашёл настройки
celery_app = Celery("leadinc-ai", config_source="backend.celeryconfig")

# Импортируем задачи, чтобы они регистрировались
from backend.tasks import chat, stt, tts  # noqa

# Настроим основной логгер (один на всё приложение)
logger = logging.getLogger("leadinc-backend")

@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    # Перенаправляем логи Celery в наш логгер (leadinc-backend.log)
    pass  # уже настроено в logging.basicConfig приложения

@signals.worker_ready.connect
def on_worker_ready(**kwargs):
    logger.info(f"[Celery] Worker ready (PID: {os.getpid()}) — queues: {celery_app.conf.task_queues}")

@signals.task_prerun.connect
def on_task_prerun(task_id=None, task=None, **kwargs):
    logger.info(f"[Celery] Task started: {task.name} | id={task_id}")

@signals.task_postrun.connect
def on_task_postrun(task_id=None, task=None, retval=None, state=None, **kwargs):
    logger.info(f"[Celery] Task finished: {task.name} | id={task_id} | state={state}")

@signals.task_failure.connect
def on_task_failure(task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **other):
    logger.error(f"[Celery] Task failed: {task_id} | error={exception}")

@signals.task_soft_time_limit.connect
def on_task_soft_time_limit(task_id=None, task=None, **kwargs):
    logger.warning(f"[Celery] Task soft_time_limit exceeded: {task.name} | id={task_id}")

@signals.task_time_limit.connect
def on_task_time_limit(task_id=None, task=None, **kwargs):
    logger.error(f"[Celery] Task HARD time_limit exceeded: {task.name} | id={task_id}")

if __name__ == "__main__":
    # Пример запуска worker через команду:
    # celery -A backend.celery_worker worker -Q audio -c 2 --loglevel=info
    logger.info("[Celery] Manual run — use 'celery worker' command instead")
