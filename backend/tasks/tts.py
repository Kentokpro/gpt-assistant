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
"""
backend/tasks/tts.py

Celery-задача для генерации голосового ответа (TTS) через ElevenLabs:
- Получает текст, voice_id, user_id, session_id, формат (mp3/ogg)
- Валидирует текст, вызывает generate_tts (асинхронно)
- Сохраняет аудиофайл, транскрипцию, метаданные
- Логирует ошибки через ErrorLog и log_error_to_db
- SLA-контроль, возврат результата (status, audio_url, transcript, meta)
"""

import os
import time
import logging
import asyncio

from celery import Task

from backend.celery_worker import celery_app
from backend.utils.tts_utils import generate_tts
from backend.models import Message, AudioEventLog
from backend.utils.audio_utils import MEDIA_AUDIO_DIR
from backend.utils.error_log_utils import log_error_to_db
from backend.database import SessionLocalSync
from backend.utils.audio_constants import TTS_SLA_LIMIT, TTS_SOFT_TIME_LIMIT, TTS_HARD_TIME_LIMIT

logger = logging.getLogger("leadinc-backend")

def _make_result(status, audio_url, text_transcript, started_at, meta):
    """
    Вспомогательная функция для финального формирования результата TTS.
    - Гарантирует только dict с сериализуемыми данными.
    """
    elapsed = time.monotonic() - started_at
    if meta is not None and isinstance(meta, dict):
        meta["final_elapsed_time"] = elapsed
        if isinstance(audio_url, str):
            meta["audio_url"] = audio_url
    result = {
        "status": status,
        "audio_url": audio_url if isinstance(audio_url, str) or audio_url is None else str(audio_url),
        "text_transcript": text_transcript if isinstance(text_transcript, str) else str(text_transcript),
        "meta": meta if isinstance(meta, dict) else {},
    }
    # Дополнительная проверка сериализации
    try:
        import json
        json.dumps(result)
    except Exception as e:
        logger.error(f"[TTS][ERROR] Result not serializable: {e}, result={result}")
        result = {
            "status": "failed",
            "audio_url": None,
            "text_transcript": text_transcript if isinstance(text_transcript, str) else "",
            "meta": {"error": "Result serialization failed", "exception": str(e)}
        }
    return result

@celery_app.task(bind=True, name="tts.tts_task", queue="audio", soft_time_limit=TTS_SOFT_TIME_LIMIT, time_limit=TTS_HARD_TIME_LIMIT)
def tts_task(self, text: str, voice_id: str = None, user_id: str = None, session_id: str = None, output_format: str = "mp3"):
    """
    Celery-задача для генерации речи через ElevenLabs TTS.
    Всегда возвращает только dict, пригодный для json-сериализации.
    """
    task_id = str(self.request.id)
    started_at = time.monotonic()
    status = "pending"
    audio_url = None
    text_transcript = text.strip() if isinstance(text, str) else ""
    meta = {
        "task_id": task_id,
        "voice_id": voice_id,
        "user_id": user_id,
        "session_id": session_id,
        "output_format": output_format,
        "sla_slow": False
    }

    logger.info(f"[TTS TASK] Start: text='{text_transcript[:50]}', voice_id='{voice_id}', user_id='{user_id}', session_id='{session_id}', output_format='{output_format}'")

    # Проверка валидности текста
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"[TTS TASK] Empty or invalid text, skipping TTS")
        status = "failed"
        meta["error"] = "Empty or invalid text"
        result = _make_result(status, None, text_transcript, started_at, meta)
        return result

    # Проверка и логирование ключей окружения
    import os
    env_api_key = os.environ.get("ELEVENLABS_API_KEY")
    env_voice_id = os.environ.get("ELEVENLABS_VOICE_ID")
    env_fallback_voice_id = os.environ.get("ELEVENLABS_FALLBACK_VOICE_ID")
    logger.info(f"[TTS TASK][ENV] API_KEY={'YES' if env_api_key else 'NO'}, VOICE_ID={env_voice_id}, FALLBACK={env_fallback_voice_id}")

    # Основной асинхронный вызов generate_tts (через event loop)
    try:
        # Внимание: запуск отдельного event loop для Celery Sync задачи!
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tts_result = loop.run_until_complete(
                generate_tts(
                    text=text_transcript,
                    voice_id=voice_id,
                    fallback=True,
                    audio_format=output_format,
                    telegram=(output_format == "ogg"),
                    user_id=user_id,
                    meta=meta
                )
            )
        finally:
            loop.close()

        logger.info(f"[TTS TASK] generate_tts result: {tts_result}")

        # Обработка результата — только dict!
        if tts_result is None:
            logger.error(f"[TTS TASK][ERROR] generate_tts вернул None!")
            status = "failed"
            meta["error"] = "generate_tts returned None"
            result = _make_result(status, None, text_transcript, started_at, meta)
            return result

        if not isinstance(tts_result, dict):
            logger.error(f"[TTS TASK][ERROR] generate_tts вернул НЕ dict: {tts_result}")
            status = "failed"
            meta["error"] = "generate_tts did not return dict"
            result = _make_result(status, None, text_transcript, started_at, meta)
            return result

        # Фолбэк на текст — если нет аудио (например, ошибка TTS)
        if tts_result.get("reply_type") == "text" or not tts_result.get("audio_url"):
            status = "failed"
            meta.update(tts_result.get("meta", {}))
            meta["error"] = meta.get("error", "TTS failed or fallback")
            log_error_to_db(user_id, "TTS failed", {"task_id": task_id, **meta})
            # Логируем событие AudioEventLog
            try:
                with SessionLocalSync() as db:
                    audio_event = AudioEventLog(
                        user_id=user_id,
                        session_id=session_id,
                        message_id=None,
                        event_type="audio_failed",
                        file_path=None,
                        status="failed",
                        details={
                            "tts_meta": meta,
                            "task_id": task_id,
                            "format": output_format,
                            "error": meta.get("error", "TTS failed or fallback"),
                            "elapsed": time.monotonic() - started_at
                        }
                    )
                    db.add(audio_event)
                    db.commit()
            except Exception as db_e:
                logger.error(f"[TTS TASK][DB ERROR] Failed to log AudioEventLog: {db_e}")
            result = _make_result(status, None, text_transcript, started_at, meta)
            return result

        # Всё ОК — аудиофайл успешно сгенерирован
        audio_url = tts_result.get("audio_url")
        if isinstance(audio_url, bytes):
            audio_url = audio_url.decode("utf-8")
        elif not isinstance(audio_url, str):
            audio_url = str(audio_url)

        # Логируем AudioEventLog (успех)
        try:
            msg = Message.save_voice_message_sync(
                user_id=user_id,
                session_id=session_id,
                content=audio_url,
                meta={
                    "tts_meta": tts_result.get("meta", {}),
                    "format": output_format,
                    "task_id": task_id,
                    "transcript": text_transcript
                }
            )
            with SessionLocalSync() as db:
                audio_event = AudioEventLog(
                    user_id=user_id,
                    session_id=session_id,
                    message_id=msg.id if msg else None,
                    event_type="audio_created",
                    file_path=audio_url,
                    status="ok",
                    details={
                        "tts_meta": tts_result.get("meta", {}),
                        "task_id": task_id,
                        "format": output_format,
                        "elapsed": time.monotonic() - started_at
                    }
                )
                db.add(audio_event)
                db.commit()
        except Exception as db_e:
            logger.error(f"[TTS TASK][DB ERROR] Failed to log success AudioEventLog: {db_e}")

        # Проверка времени выполнения (SLA)
        elapsed = time.monotonic() - started_at
        if elapsed > TTS_SLA_LIMIT:
            meta["sla_slow"] = True
            log_error_to_db(user_id, "TTS SLA timeout", {
                "task_id": task_id, "elapsed": elapsed, "audio_url": audio_url
            })

        status = "ok"
        meta.update({
            "elapsed_time": elapsed,
            "audio_url": audio_url,
            "format": output_format
        })

        logger.info(f"[TTS] [{task_id}] Успешно: audio_url={audio_url}, elapsed={elapsed:.2f}s, user={user_id}")

        result = _make_result(status, audio_url, text_transcript, started_at, meta)
        return result

    except Exception as e:
        # Глобальная обработка всех ошибок
        status = "failed"
        meta["error"] = str(e)
        logger.error(f"[TTS][ERROR] [{task_id}] {e}", exc_info=True)
        log_error_to_db(user_id, "TTS error", {"task_id": task_id, "error": str(e)})
        try:
            with SessionLocalSync() as db:
                audio_event = AudioEventLog(
                    user_id=user_id,
                    session_id=session_id,
                    message_id=None,
                    event_type="audio_failed",
                    file_path=None,
                    status="failed",
                    details={
                        "task_id": task_id,
                        "format": output_format,
                        "error": str(e),
                        "elapsed": time.monotonic() - started_at
                    }
                )
                db.add(audio_event)
                db.commit()
        except Exception as db_e:
            logger.error(f"[TTS][ERROR] Failed to log error AudioEventLog: {db_e}")
        result = _make_result(status, None, text_transcript, started_at, meta)
        return result
# Конец функции tts_task
