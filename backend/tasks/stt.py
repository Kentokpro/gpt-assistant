
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
import asyncio

from backend.utils.audio_constants import STT_SLA_LIMIT, STT_SOFT_TIME_LIMIT, STT_HARD_TIME_LIMIT
from backend.celery_worker import celery_app
from backend.utils.audio_utils import (
    MEDIA_AUDIO_DIR,
    audio_file_is_valid,
)
from backend.utils.stt_utils import (
    async_convert_to_mp3,
    transcribe_audio,
)
from backend.models import Message, ErrorLog, AudioEventLog
from backend.utils.error_log_utils import log_error_to_db

logger = logging.getLogger("leadinc-backend")
logger.setLevel(logging.DEBUG)

#    Celery-задача: аудиофайл → текст через OpenAI Whisper, логирование и SLA.
@celery_app.task(
    bind=True,
    name="stt.stt_task",
    queue="audio",
    soft_time_limit=STT_SOFT_TIME_LIMIT,
    time_limit=STT_HARD_TIME_LIMIT
)
def stt_task(self, audio_path: str, user_id: str = None, session_id: str = None):
    task_id = str(self.request.id)
    started_at = time.monotonic()
    status = "pending"
    transcript = ""
    audio_url = ""
    meta = {"task_id": task_id}
    ext = os.path.splitext(audio_path)[1].lower()

    logger.info(f"[STT] [{task_id}] Файл принят: {audio_path}, ext={ext}, user_id={user_id}, session_id={session_id}")

    # Для трассировки окружения
    logger.debug(f"[STT] [{task_id}] ENV: cwd={os.getcwd()}, pid={os.getpid()}, file_exists={os.path.exists(audio_path)}")

    # Старт отдельного event loop для синхронного Celery-задачи
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # === 1. Валидация аудиофайла ===
        valid, reason, duration = audio_file_is_valid(audio_path)
        logger.debug(f"[STT] [{task_id}] Валидация файла: valid={valid}, reason={reason}, duration={duration}")

        if not valid:
            status = "failed"
            meta.update({"error": reason, "duration": duration})
            logger.warning(f"[STT] [{task_id}] Ошибка валидации файла: {reason}, длина={duration}")
            # [БОНУС] — вывод первых 32 байт файла (если он читается)
            try:
                if os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        first_bytes = f.read(32)
                    logger.debug(f"[STT] [{task_id}] Первые 32 байта файла: {first_bytes}")
            except Exception as e:
                logger.debug(f"[STT] [{task_id}] Не удалось прочитать байты файла: {e}")
            log_error_to_db(user_id, "STT invalid audio", {"reason": reason, "duration": duration, "task_id": task_id})
            return _make_result(status, transcript, audio_url, started_at, meta)

        # === 2. Конвертация .ogg в .mp3 (если нужно) ===
        if ext == ".ogg":
            logger.info(f"[STT] [{task_id}] Начинаем конвертацию OGG→MP3: {audio_path}")
            try:
                mp3_path = loop.run_until_complete(async_convert_to_mp3(audio_path))
                logger.info(f"[STT] [{task_id}] Конвертация завершена: {mp3_path}")
                audio_path = mp3_path
                ext = ".mp3"
            except Exception as conv_e:
                status = "failed"
                meta["error"] = f"Ошибка конвертации: {conv_e}"
                logger.error(f"[STT][ERROR] [{task_id}] Ошибка конвертации .ogg→.mp3: {conv_e}")
                log_error_to_db(user_id, "STT ogg->mp3 conversion failed", {"error": str(conv_e), "task_id": task_id})
                return _make_result(status, transcript, audio_url, started_at, meta)

        # === 3. Асинхронная транскрипция (Whisper) ===
        try:
            logger.info(f"[STT] [{task_id}] Отправляем файл на транскрипцию через Whisper: {audio_path}")
            stt_result = loop.run_until_complete(transcribe_audio(audio_path, user_id=user_id, session_id=session_id))
            logger.info(f"[STT] [{task_id}] Результат транскрипции: {stt_result}")
        except Exception as stt_e:
            status = "failed"
            meta["error"] = f"STT failed: {stt_e}"
            logger.error(f"[STT][ERROR] [{task_id}] Ошибка Whisper: {stt_e}")
            log_error_to_db(user_id, "STT failed", {"exception": str(stt_e), "task_id": task_id})
            return _make_result(status, transcript, audio_url, started_at, meta)

        status = "ok" if stt_result.get("ok") else "failed"
        transcript = stt_result.get("transcript", "")
        meta.update({
            "elapsed": stt_result.get("elapsed"),
            "transcript": transcript,
            "stt_status": status,
            "stt_task_id": stt_result.get("task_id"),
        })

        if not stt_result.get("ok"):
            error_text = stt_result.get("error", "Unknown STT error")
            meta["error"] = error_text
            logger.warning(f"[STT][RESULT_FAIL] [{task_id}] Транскрипция не удалась: {error_text}")
            log_error_to_db(user_id, "STT failed", meta)
            return _make_result(status, transcript, audio_url, started_at, meta)

        logger.info(f"[STT][SUCCESS] [{task_id}] Распознавание завершено успешно. Текст: {transcript[:60]}...")

        # === 4. Формируем URL для медиа (возвращаем фронту только если успех) ===
        rel_path = os.path.relpath(audio_path, MEDIA_AUDIO_DIR.parent)
        audio_url = f"/media/audio/{os.path.basename(audio_path)}"

        # === 5. Сохраняем сообщение типа "voice" в Message (только assistant-side) ===
        Message.save_voice_message_sync(
            user_id=user_id,
            session_id=session_id,
            content=audio_url,
            meta={
                "transcript": transcript,
                "stt_meta": meta,
                "elapsed": meta.get("elapsed"),
            },
        )

        AudioEventLog.create_event(
            user_id=user_id,
            session_id=session_id,
            event_type="audio_created",
            file_path=audio_url,
            status="ok",
            details=meta
        )

        # === 6. SLA, ErrorLog при превышении лимита ===
        elapsed = time.monotonic() - started_at
        if elapsed > STT_SLA_LIMIT:
            logger.error(f"[STT][SLA_TIMEOUT] [{task_id}] Превышено время обработки: {elapsed:.2f}s")
            log_error_to_db(user_id, "SLA timeout (STT)", {
                "elapsed": elapsed,
                "task_id": task_id,
                "audio_path": audio_path
            })
        # Итоговое время только в result_elapsed
        meta["result_elapsed"] = elapsed

        logger.info(f"[STT] [{task_id}] Успех. {audio_url}, elapsed={elapsed:.2f}s, user={user_id}")

        return _make_result(status, transcript, audio_url, started_at, meta)

    except Exception as e:
        status = "failed"
        meta["error"] = str(e)
        logger.error(f"[STT][ERROR] [{task_id}] {e}")
        log_error_to_db(user_id, "STT failed", {"exception": str(e), "task_id": task_id})
        return _make_result(status, transcript, audio_url, started_at, meta)
    finally:
        loop.close()

# === Вспомогательная функция результата ===
def _make_result(status, transcript, audio_url, started_at, meta):
    elapsed = time.monotonic() - started_at
    if meta is not None:
        meta["result_elapsed"] = elapsed
    return {
        "status": status,
        "transcript": transcript,
        "audio_url": audio_url,
        "meta": meta,
    }
