"""
backend/tasks/tts.py

TTS-задача для Celery:
- Принимает текст и voice_id, генерирует аудиофайл через ElevenLabs API
- Сохраняет mp3, при необходимости — конвертирует в ogg
- Логирует событие, SLA, ошибки
- В случае ошибки — ErrorLog и fallback (ответ текстом)
- Для Telegram: отдаёт ogg, для web — mp3

Требует:
- backend/utils/tts_utils.py (ElevenLabs helper)
- backend/utils/audio_utils.py (ffmpeg, валидация)
- backend/models.py (Message, ErrorLog)
- celeryconfig.py
"""

import os
import uuid
import time
import logging

from celery import Celery
from backend.celeryconfig import celery_app
from backend.utils.tts_utils import tts_generate_elevenlabs
from backend.utils.audio_utils import (
    convert_mp3_to_ogg, 
    delete_file, 
    validate_audio_file
)
from backend.models import ErrorLog, Message
from backend.config import MEDIA_DIR

logger = logging.getLogger("leadinc-backend")

TTS_SLA_LIMIT = 45  # seconds
TTS_SOFT_TIME_LIMIT = 90
TTS_HARD_TIME_LIMIT = 120

@celery_app.task(bind=True, soft_time_limit=TTS_SOFT_TIME_LIMIT, time_limit=TTS_HARD_TIME_LIMIT)
def tts_task(self, text: str, voice_id: str, user_id: str = None, session_id: str = None, output_format: str = "mp3"):
    """
    Задача TTS:
    - Генерирует аудио по тексту через ElevenLabs
    - Сохраняет mp3/ogg, возвращает ссылку, логи, SLA

    Args:
        text (str): исходный текст для синтеза
        voice_id (str): идентификатор кастомного голоса
        user_id (str): id пользователя
        session_id (str): id сессии
        output_format (str): "mp3" (web) или "ogg" (Telegram)

    Returns:
        dict: {
            "status": "ok"/"failed",
            "audio_url": str,
            "text_transcript": str,
            "meta": {...}
        }
    """

    task_id = str(uuid.uuid4())
    t_start = time.time()
    status = "pending"
    audio_url = ""
    error_details = {}
    text_transcript = text

    logger.info(f"[TTS] [{task_id}] Старт синтеза: len={len(text)}, voice_id={voice_id}, user_id={user_id}, session_id={session_id}")

    try:
        # 1. Валидация текста (только не пустой)
        if not isinstance(text, str) or not text.strip():
            status = "failed"
            error_details = {"reason": "empty_text", "len": len(text) if isinstance(text, str) else None}
            logger.warning(f"[TTS] [{task_id}] Текст невалиден — пустой")
            _log_error("TTS invalid text", user_id, session_id, error_details)
            return _make_result(status, audio_url, text_transcript, t_start, error_details)

        # 2. Генерация аудио через ElevenLabs
        mp3_path, tts_meta = tts_generate_elevenlabs(text, voice_id)
        if not mp3_path or not os.path.exists(mp3_path):
            status = "failed"
            error_details = {"reason": "TTS generate failed", "meta": tts_meta}
            logger.error(f"[TTS] [{task_id}] Ошибка генерации ElevenLabs")
            _log_error("TTS failed", user_id, session_id, error_details)
            return _make_result(status, audio_url, text_transcript, t_start, error_details)

        # 3. По необходимости конвертировать mp3→ogg (для Telegram)
        audio_path = mp3_path
        if output_format == "ogg":
            ogg_path = convert_mp3_to_ogg(mp3_path)
            if ogg_path and os.path.exists(ogg_path):
                delete_file(mp3_path)
                audio_path = ogg_path
                logger.info(f"[TTS] [{task_id}] MP3→OGG: {ogg_path}")
            else:
                logger.warning(f"[TTS] [{task_id}] Не удалось сконвертировать в ogg")
                # fallback — оставляем mp3

        # 4. Валидация файла (длительность, размер)
        valid, reason, duration = validate_audio_file(audio_path)
        if not valid:
            status = "failed"
            error_details = {"reason": reason, "duration": duration}
            logger.warning(f"[TTS] [{task_id}] Аудиофайл невалиден: {reason}")
            delete_file(audio_path)
            _log_error("TTS invalid audio", user_id, session_id, error_details)
            return _make_result(status, audio_url, text_transcript, t_start, error_details)

        # 5. Формируем URL для медиа (frontend получает только путь, не download link)
        rel_path = os.path.relpath(audio_path, MEDIA_DIR)
        audio_url = f"/media/{rel_path}"

        # 6. Сохраняем сообщение типа "voice" в базе
        Message.save_voice_message(
            user_id=user_id,
            session_id=session_id,
            content=audio_url,
            meta={"tts_meta": tts_meta, "elapsed": tts_meta.get("elapsed"), "voice_id": voice_id, "format": output_format},
        )

        # 7. SLA: логируем превышение лимита
        elapsed = time.time() - t_start
        if elapsed > TTS_SLA_LIMIT:
            logger.error(f"[TTS][SLA_TIMEOUT] [{task_id}] Превышено время обработки: {elapsed:.2f}s")
            _log_error("SLA timeout (TTS)", user_id, session_id, {
                "elapsed": elapsed, "task_id": task_id, "audio_path": audio_path
            })

        # 8. Возврат результата
        return _make_result("ok", audio_url, text_transcript, t_start, {"tts_meta": tts_meta, "voice_id": voice_id})

    except Exception as e:
        status = "failed"
        error_details = {"exception": str(e)}
        logger.error(f"[TTS][ERROR] [{task_id}] {e}")
        _log_error("TTS failed", user_id, session_id, error_details)
        return _make_result(status, audio_url, text_transcript, t_start, error_details)


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
        logger.error(f"[TTS][ErrorLog FAIL] {e}")

def _make_result(status, audio_url, text_transcript, t_start, meta):
    elapsed = time.time() - t_start
    return {
        "status": status,
        "audio_url": audio_url,
        "text_transcript": text_transcript,
        "meta": {
            "elapsed": elapsed,
            **(meta or {})
        }
    }

