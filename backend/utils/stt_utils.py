"""
stt_utils.py — Асинхронные утилиты для голосового стека (STT, Whisper)
- Сохраняет загруженные файлы в нужную директорию
- Валидирует аудиофайлы (через централизованный валидатор audio_utils.py)
- Асинхронно конвертирует ogg→mp3 через ffmpeg
- Асинхронно отправляет аудио в OpenAI Whisper через OpenAI API
- Логирует все ошибки и превышения SLA через log_error_to_db
- Используется в Celery-задачах, API, ручных обработчиках файлов
"""

import os
import io
import uuid
import logging
import aiofiles
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from backend.config import OPENAI_API_KEY
from backend.utils.audio_utils import (
    MEDIA_AUDIO_DIR,
    audio_file_is_valid,
    convert_to_mp3,
)
from backend.utils.audio_constants import ALLOWED_EXTENSIONS, MAX_AUDIO_SIZE_MB, MAX_AUDIO_DURATION_SEC, STT_SLA_LIMIT
from backend.utils.error_log_utils import log_error_to_db

logger = logging.getLogger("leadinc-backend")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# === 1. Сохранение загруженного файла ===

async def save_upload_file(upload_file, ext: str = "mp3") -> str:
    """
    Асинхронно сохраняет загруженный upload_file во временную директорию.
    Возвращает путь к сохранённому файлу.
    """
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = MEDIA_AUDIO_DIR / filename
    async with aiofiles.open(file_path, "wb") as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    logger.info(f"stt_utils: Audio file saved: {file_path}")
    return str(file_path)

# === 2. Асинхронная конвертация ogg → mp3 через ffmpeg ===

async def async_convert_to_mp3(input_path: str) -> str:
    output_path = str(MEDIA_AUDIO_DIR / (uuid.uuid4().hex + ".mp3"))
    cmd = [
        "ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "1",
        "-codec:a", "libmp3lame", "-b:a", "128k", output_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"stt_utils: ffmpeg завершился с ошибкой (код {proc.returncode}) при конвертации {input_path}")
        log_error_to_db(
            user_id=None,
            error="stt_ffmpeg_conversion_error",
            details={"src_path": input_path, "cmd": " ".join(cmd), "returncode": proc.returncode},
        )
        raise Exception(f"Ошибка конвертации аудио в mp3 (ffmpeg code {proc.returncode})")
    if not os.path.exists(output_path):
        logger.error(f"stt_utils: Ошибка конвертации {input_path} в mp3 (файл не создан)")
        log_error_to_db(
            user_id=None,
            error="stt_ffmpeg_conversion_error",
            details={"src_path": input_path, "cmd": " ".join(cmd)},
        )
        raise Exception("Ошибка конвертации аудио в mp3 (файл не создан)")
    logger.info(f"stt_utils: Audio converted to mp3: {output_path}")
    return output_path

# === 3. Асинхронная транскрипция через OpenAI Whisper ===

async def transcribe_audio(
    file_path: str,
    user_id: str = None,
    session_id: str = None,
) -> dict:
    """
    Асинхронно транскрибирует аудиофайл через OpenAI Whisper API.
    — Проверяет валидность файла
    — Отправляет на транскрипцию
    — Логирует SLA, ошибки, пишет ErrorLog при ошибках
    — Возвращает dict: {ok: bool, transcript: str, error: str, elapsed: float, ...}
    """
    result = {
        "ok": False,
        "transcript": None,
        "error": None,
        "elapsed": None,
        "status": None,
        "task_id": None,
        "file_path": str(file_path)
    }
    task_id = str(uuid.uuid4())
    t0 = asyncio.get_running_loop().time()
    try:
        # Валидируем файл
        valid, reason, duration = audio_file_is_valid(file_path)
        if not valid:
            logger.warning(f"stt_utils: Не валидный файл для STT: {file_path}, причина: {reason}")
            log_error_to_db(
                user_id=user_id,
                error="stt_audio_invalid",
                details={"file": file_path, "reason": reason, "duration": duration, "task_id": task_id}
            )
            result.update({
                "error": reason,
                "elapsed": 0.0,
                "status": "invalid",
                "task_id": task_id,
            })
            return result

        # Передача пути к аудио файлу
        transcript_resp = await client.audio.transcriptions.create(
            model="whisper-1",
            file=Path(file_path),
            response_format="text"
        )
        elapsed = asyncio.get_running_loop().time() - t0
        try:
            transcript = transcript_resp.strip()
        except Exception as e:
            logger.error(f"stt_utils: Whisper API вернул некорректный ответ: {transcript_resp}, ошибка: {e}")
            transcript = None
            result["error"] = f"Whisper API вернул некорректный ответ: {e}"
        result.update({
            "ok": True if transcript else False,
            "transcript": transcript,
            "elapsed": elapsed,
            "status": "ok" if transcript else "error",
            "task_id": task_id,
        })
        logger.info(f"stt_utils: STT complete | user={user_id} session={session_id} task={task_id} elapsed={elapsed:.2f}s")
        return result

    except Exception as e:
        elapsed = asyncio.get_running_loop().time() - t0
        error_msg = f"Ошибка транскрипции через OpenAI Whisper: {e}"
        logger.error(f"stt_utils: STT failed | user={user_id} session={session_id} task={task_id} | {error_msg}")
        log_error_to_db(
            user_id=user_id,
            error="stt_failed",
            details={
                "file": file_path,
                "session_id": session_id,
                "task_id": task_id,
                "elapsed": elapsed,
                "reason": str(e)
            }
        )
        result.update({
            "error": error_msg,
            "elapsed": elapsed,
            "status": "error",
            "task_id": task_id,
        })
        return result

# === END stt_utils.py ===
