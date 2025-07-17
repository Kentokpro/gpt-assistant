"""
Файл: backend/utils/stt_utils.py

Назначение:
- Вспомогательные функции для голосового стека (STT).
- Валидация аудиофайлов, асинхронная конвертация, транскрипция через OpenAI Whisper API.
- SLA-замеры времени, логирование ошибок, удаление файлов.
- Архитектура и стандарты — строго по проекту Leadinc (log-path, ErrorLog, переменные из .env).

Зависимости:
- ffmpeg должен быть установлен в системе.
- Требуются переменные OPENAI_API_KEY и путь к media/audio в .env.backend.
- Импортируется в tasks/stt.py, celery_worker.py.
"""

import os
import uuid
import logging
import aiofiles
import shutil
import asyncio
import subprocess
import time
from pathlib import Path

from openai import AsyncOpenAI
from backend.config import OPENAI_API_KEY
from backend.models import ErrorLog  # Для записи ошибок в БД
from backend.database import SessionLocal  # Для ErrorLog
from sqlalchemy.ext.asyncio import AsyncSession

# === 1. Настройки и переменные окружения ===

MEDIA_DIR = Path(__file__).parent.parent / "media" / "audio"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = "/root/ai-assistant/backend/leadinc-backend.log"
logger = logging.getLogger("leadinc-backend")

# Лимиты (можно вынести в конфиг)
MAX_AUDIO_SIZE_MB = 10
MAX_AUDIO_DURATION_SEC = 120   # 2 минуты
ALLOWED_EXTENSIONS = {"mp3", "ogg"}

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# === 2. Утилиты для аудиофайлов ===

async def save_upload_file(upload_file, ext="mp3") -> str:
    """
    Сохраняет загруженный файл во временную директорию.
    Возвращает путь к сохранённому файлу.
    """
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = MEDIA_DIR / filename
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    logger.info(f"Audio file saved: {file_path}")
    return str(file_path)

def validate_audio_file(file_path: str) -> dict:
    """
    Проверяет размер, длительность, расширение аудиофайла.
    Возвращает {ok: bool, error: str}.
    """
    result = {"ok": True, "error": None}

    # Проверяем размер файла
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_AUDIO_SIZE_MB:
        result["ok"] = False
        result["error"] = f"Превышен лимит размера аудио ({size_mb:.2f}MB > {MAX_AUDIO_SIZE_MB}MB)"
        return result

    # Проверяем расширение
    ext = Path(file_path).suffix[1:].lower()
    if ext not in ALLOWED_EXTENSIONS:
        result["ok"] = False
        result["error"] = f"Недопустимый формат файла ({ext})"
        return result

    # Проверяем длительность через ffmpeg
    try:
        import ffmpeg
        probe = ffmpeg.probe(file_path)
        duration = float(probe["format"]["duration"])
        if duration > MAX_AUDIO_DURATION_SEC:
            result["ok"] = False
            result["error"] = f"Превышен лимит длины аудио ({duration:.1f}s > {MAX_AUDIO_DURATION_SEC}s)"
        result["duration"] = duration
    except Exception as e:
        result["ok"] = False
        result["error"] = f"Ошибка проверки длительности: {e}"
    return result

async def convert_to_mp3(input_path: str) -> str:
    """
    Конвертирует ogg/opus аудио в mp3 через ffmpeg, возвращает путь к mp3.
    """
    output_path = str(MEDIA_DIR / (uuid.uuid4().hex + ".mp3"))
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
    if not os.path.exists(output_path):
        raise Exception("Ошибка конвертации аудио в mp3")
    logger.info(f"Audio converted to mp3: {output_path}")
    return output_path

async def delete_audio_file(file_path: str, reason="") -> None:
    """
    Удаляет файл, логирует событие.
    """
    try:
        os.remove(file_path)
        logger.info(f"Audio file deleted: {file_path} | Reason: {reason}")
    except Exception as e:
        logger.warning(f"Ошибка при удалении файла {file_path}: {e}")

# === 3. Основная функция транскрипции через OpenAI Whisper API ===

async def transcribe_audio(
    file_path: str, 
    user_id: str = None, 
    session_id: str = None, 
    db: AsyncSession = None
) -> dict:
    """
    Транскрибирует аудиофайл через OpenAI Whisper API.
    - Возвращает {ok: bool, transcript: str, error: str, elapsed: float, ...}
    - Логирует SLA (elapsed), ошибки, пишет ErrorLog в БД при ошибках.
    """
    result = {
        "ok": False, "transcript": None, "error": None,
        "elapsed": None, "status": None, "task_id": None
    }
    task_id = str(uuid.uuid4())
    t0 = time.time()
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            file_bytes = await f.read()
        # OpenAI expects a file-like object
        import io
        audio_file = io.BytesIO(file_bytes)
        audio_file.name = os.path.basename(file_path)
        # Отправляем на транскрипцию
        transcript_resp = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        elapsed = time.time() - t0
        transcript = transcript_resp.strip()
        result.update({
            "ok": True,
            "transcript": transcript,
            "elapsed": elapsed,
            "task_id": task_id,
            "status": "ok"
        })
        logger.info(f"STT complete | user={user_id} session={session_id} task={task_id} elapsed={elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - t0
        error_msg = f"Ошибка транскрипции через OpenAI Whisper: {e}"
        result.update({
            "error": error_msg,
            "elapsed": elapsed,
            "status": "error",
            "task_id": task_id
        })
        logger.error(f"STT failed | user={user_id} session={session_id} task={task_id} elapsed={elapsed:.2f}s | {error_msg}")
        # Логируем в БД при необходимости
        if db:
            try:
                error_log = ErrorLog(
                    user_id=user_id,
                    error="STT failed",
                    details={
                        "file": file_path,
                        "session_id": session_id,
                        "task_id": task_id,
                        "elapsed": elapsed,
                        "reason": str(e)
                    }
                )
                db.add(error_log)
                await db.commit()
            except Exception as log_e:
                logger.error(f"Ошибка при записи ErrorLog в БД: {log_e}")
    return result

# === 4. Вспомогательные функции для SLA, логирования, цепочки задач ===

def get_audio_info(file_path: str) -> dict:
    """
    Получить метаданные аудиофайла: размер, длительность, формат.
    """
    info = {}
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        info["size_mb"] = size_mb
        ext = Path(file_path).suffix[1:].lower()
        info["ext"] = ext
        # Получаем длительность через ffmpeg
        import ffmpeg
        probe = ffmpeg.probe(file_path)
        info["duration"] = float(probe["format"]["duration"])
    except Exception as e:
        info["error"] = f"Ошибка получения метаданных: {e}"
    return info


