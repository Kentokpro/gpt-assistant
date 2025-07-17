"""
backend/tasks/stt.py

STT-задача для Celery: 
- Принимает аудиофайл (mp3/ogg), проверяет валидность
- При необходимости конвертирует в mp3
- Вызывает OpenAI Whisper API
- Сохраняет транскрипт, логи, SLA
- В случае ошибки — ErrorLog и возврат статуса "failed"

Требует:
- backend/utils/audio_utils.py (ffmpeg utils)
- backend/utils/stt_utils.py (STT helper)
- backend/models.py (Message, ErrorLog)
- celeryconfig.py (broker/result_backend)
"""

import os
import uuid
import time
import logging

from celery import Celery, Task
from backend.celeryconfig import celery_app
from backend.utils.audio_utils import (
    validate_audio_file, 
    convert_ogg_to_mp3, 
    convert_mp3_to_ogg, 
    delete_file
)
from backend.utils.stt_utils import stt_transcribe_whisper
from backend.models import ErrorLog, Message
from backend.config import MEDIA_DIR

logger = logging.getLogger("leadinc-backend")

STT_SLA_LIMIT = 45  # seconds for SLA
STT_SOFT_TIME_LIMIT = 90
STT_HARD_TIME_LIMIT = 120

@celery_app.task(bind=True, soft_time_limit=STT_SOFT_TIME_LIMIT, time_limit=STT_HARD_TIME_LIMIT)
def stt_task(self, audio_path: str, user_id: str = None, session_id: str = None):
    """
    Задача STT:
    - Проверяет файл, конвертирует ogg→mp3 если нужно
    - Отправляет в Whisper через openai api
    - Возвращает transcript, meta, статус, логирует всё

    Args:
        audio_path (str): путь к аудиофайлу (mp3 или ogg)
        user_id (str): id пользователя
        session_id (str): id сессии

    Returns:
        dict: {
            "status": "ok"/"failed",
            "transcript": str,
            "audio_url": str,
            "meta": {...}
        }
    """

    task_id = str(uuid.uuid4())
    t_start = time.time()
    status = "pending"
    transcript = ""
    error_details = {}
    audio_url = ""
    ext = os.path.splitext(audio_path)[1].lower()

    logger.info(f"[STT] [{task_id}] Получен файл: {audio_path}, ext={ext}, user_id={user_id}, session_id={session_id}")

    try:
        # 1. Валидация аудиофайла (размер, длительность)
        valid, reason, duration = validate_audio_file(audio_path)
        if not valid:
            status = "failed"
            error_details = {"reason": reason, "duration": duration}
            logger.warning(f"[STT] [{task_id}] Файл невалиден: {reason}")
            delete_file(audio_path)
            _log_error("STT invalid audio", user_id, session_id, error_details)
            return _make_result(status, transcript, audio_url, t_start, error_details)

        # 2. Если ogg → mp3 (Whisper требует mp3)
        if ext == ".ogg":
            mp3_path = convert_ogg_to_mp3(audio_path)
            logger.info(f"[STT] [{task_id}] OGG→MP3: {mp3_path}")
            delete_file(audio_path)
            audio_path = mp3_path
            ext = ".mp3"

        # 3. Отправка в Whisper API (OpenAI)
        transcript, stt_meta = stt_transcribe_whisper(audio_path)
        status = "ok"
        logger.info(f"[STT] [{task_id}] Whisper API результат: {transcript[:100]}..., meta={stt_meta}")

        # 4. Формируем URL для медиа (возвращаем фронту только если успех)
        rel_path = os.path.relpath(audio_path, MEDIA_DIR)
        audio_url = f"/media/{rel_path}"

        # 5. Сохраняем сообщение типа "voice" (Message) и транскрипцию в базе
        Message.save_voice_message(
            user_id=user_id,
            session_id=session_id,
            content=audio_url,
            meta={"transcript": transcript, "stt_meta": stt_meta, "elapsed": stt_meta.get("elapsed")},
        )

        # 6. SLA, ErrorLog при превышении лимита
        elapsed = time.time() - t_start
        if elapsed > STT_SLA_LIMIT:
            logger.error(f"[STT][SLA_TIMEOUT] [{task_id}] Превышено время обработки: {elapsed:.2f}s")
            _log_error("SLA timeout (STT)", user_id, session_id, {
                "elapsed": elapsed, "task_id": task_id, "audio_path": audio_path
            })

        # 7. Логируем успех, возвращаем результат
        return _make_result(status, transcript, audio_url, t_start, {"stt_meta": stt_meta})

    except Exception as e:
        status = "failed"
        error_details = {"exception": str(e)}
        logger.error(f"[STT][ERROR] [{task_id}] {e}")
        _log_error("STT failed", user_id, session_id, error_details)
        if os.path.exists(audio_path):
            delete_file(audio_path)
        return _make_result(status, transcript, audio_url, t_start, error_details)


# ============ Хелперы ============

def _log_error(error: str, user_id, session_id, details):
    try:
        ErrorLog.save(
            error=error,
            user_id=user_id,
            details=details,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"[STT][ErrorLog FAIL] {e}")

def _make_result(status, transcript, audio_url, t_start, meta):
    elapsed = time.time() - t_start
    return {
        "status": status,
        "transcript": transcript,
        "audio_url": audio_url,
        "meta": {
            "elapsed": elapsed,
            **(meta or {})
        }
    }
