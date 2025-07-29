"""
tts_utils.py — Универсальные утилиты для генерации речи (TTS) через ElevenLabs API.
- Для elevenlabs==2.7.1 используется только синхронный клиент (ElevenLabs)
- Лимиты, формат и логика — только через audio_constants, MEDIA_AUDIO_DIR.
- Асинхронная обёртка через run_in_executor для совместимости с FastAPI/Celery.
- Логирование всех ошибок.
- Фолбэк на текст при ошибках, логирование SLA.
"""

import os
import uuid
import time
import logging
import aiofiles
import asyncio
from pathlib import Path

from elevenlabs.client import ElevenLabs  # ✅ Исправлен импорт клиента

from backend.config import LOG_LEVEL
from backend.utils.audio_constants import (
    ALLOWED_EXTENSIONS,
    MAX_AUDIO_SIZE_MB,
    MAX_AUDIO_DURATION_SEC,
    SUPPORTED_TTS_FORMATS,
    TTS_SOFT_TIME_LIMIT,
    TTS_HARD_TIME_LIMIT,
    DEFAULT_TTS_FORMAT,
    TTS_FORMAT_MAP,
    TTS_RETRY_COUNT,
    AUDIO_FORMAT,
    OGG_FORMAT,
    DEFAULT_TTS_FORMAT,
)
from backend.utils.audio_utils import MEDIA_AUDIO_DIR
from backend.utils.error_log_utils import log_error_to_db

logger = logging.getLogger("leadinc-backend")
logger.setLevel(LOG_LEVEL)

# === ENV: ключи и ID голосов ===
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVEN_FALLBACK_VOICE_ID = os.getenv("ELEVENLABS_FALLBACK_VOICE_ID")

# ✅ Синхронный клиент ElevenLabs
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# === 1. Сохранение аудиофайла ===
#  Асинхронно сохраняет аудиофайл в MEDIA_AUDIO_DIR, возвращает путь.
async def _write_audio_file(data: bytes, filename: str) -> str:
    file_path = MEDIA_AUDIO_DIR / filename
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(data)
    logger.info(f"tts_utils: Saved audio file: {file_path}")
    return str(file_path)

# Генерирует уникальное имя файла."""
def _gen_audio_filename(extension="mp3") -> str:
    return f"{uuid.uuid4().hex}.{extension}"

# Асинхронная конвертация mp3 → ogg через ffmpeg-python.
async def _convert_mp3_to_ogg(mp3_path: str, ogg_path: str):
    import ffmpeg
    try:
        stream = ffmpeg.input(mp3_path)
        stream = ffmpeg.output(stream, ogg_path, acodec="libopus")
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        logger.info(f"tts_utils: Converted mp3 to ogg: {ogg_path}")
    except Exception as e:
        logger.error(f"tts_utils: Failed to convert mp3 to ogg: {e}")
        raise

# === 2. Синхронная функция генерации TTS ===
def generate_tts_sync(
    text: str,
    voice_id: str = None,
    audio_format: str = DEFAULT_TTS_FORMAT,
    timeout: int = TTS_HARD_TIME_LIMIT
) -> bytes:
    import logging
    logger = logging.getLogger("leadinc-backend")

    # Валидация входных
    if not isinstance(text, str) or not text.strip():
        logger.error("[TTS SYNC] Пустой текст для синтеза речи!")
        raise ValueError("Текст для TTS не может быть пустым")

    voice = voice_id or ELEVEN_VOICE_ID

    # Валидируем формат, fallback на mp3
    if audio_format not in SUPPORTED_TTS_FORMATS:
        logger.warning(f"[TTS SYNC] Некорректный формат '{audio_format}', fallback к {DEFAULT_TTS_FORMAT}")
        audio_format = DEFAULT_TTS_FORMAT
    el_format = TTS_FORMAT_MAP[audio_format]

    # Пробуем несколько раз (с фолбэком)
    last_error = None
    for attempt in range(1, TTS_RETRY_COUNT + 2):
        try:
            logger.info(f"[TTS SYNC] Попытка #{attempt}: text='{text[:48]}...' | voice='{voice}' | format='{el_format}'")
            # (опционально: логируем список доступных голосов)
            try:
                voices = tts_client.voices.list()
                logger.debug(f"[TTS SYNC] Voices available: {voices}")
            except Exception as ve:
                logger.debug(f"[TTS SYNC] Не удалось получить список голосов: {ve}")
            # --- Вызов ElevenLabs ---
            audio_data = tts_client.text_to_speech.convert(
                text=text,
                voice_id=voice,
                model_id=None,
                output_format=el_format,
                optimize_streaming_latency="4"
            )
            # Некоторые клиенты могут вернуть генератор — собираем в bytes
            if hasattr(audio_data, "__iter__") and not isinstance(audio_data, bytes):
                audio_data = b"".join(audio_data)
            if not audio_data or not isinstance(audio_data, bytes):
                raise RuntimeError("Пустой или некорректный ответ от ElevenLabs TTS")
            logger.info(f"[TTS SYNC] Успешно: формат={el_format}, размер={len(audio_data)} байт")
            return audio_data
        except Exception as e:
            last_error = e
            logger.error(f"[TTS SYNC] Ошибка при синтезе: {e} (попытка {attempt})", exc_info=True)
            # Логируем в error_log (если есть такая функция)
            try:
                from backend.utils.error_log_utils import log_error_to_db
                log_error_to_db(
                    user_id=None,
                    error="TTS API error",
                    details={
                        "tts_error": str(e),
                        "voice_id": voice,
                        "format": audio_format,
                        "attempt": attempt,
                        "text": text[:50]
                    }
                )
            except Exception as le:
                logger.warning(f"[TTS SYNC] Не удалось записать ошибку в лог: {le}")

            # Первый фейл — пробуем fallback voice (если указан)
            if attempt == 1 and ELEVEN_FALLBACK_VOICE_ID and voice != ELEVEN_FALLBACK_VOICE_ID:
                logger.warning(f"[TTS SYNC] Пробуем fallback voice: {ELEVEN_FALLBACK_VOICE_ID}")
                voice = ELEVEN_FALLBACK_VOICE_ID
                continue
            # Если это не первая попытка или fallback не помог — пробуем дальше (до max попыток)
    # Если ничего не сработало
    logger.critical(f"[TTS SYNC] Все попытки синтеза неудачны. Последняя ошибка: {last_error}")
    raise Exception(f"TTS failed after {TTS_RETRY_COUNT+1} attempts: {last_error}")




# === 3. Асинхронная обёртка для FastAPI/Celery ===
# Асинхронно генерирует голосовой ответ через ElevenLabs. Если ошибка — возвращает текст с reply_type: text.
async def generate_tts(
    text: str,
    voice_id: str = None,
    fallback: bool = True,
    audio_format: str = DEFAULT_TTS_FORMAT,   # <-- Добавить этот параметр!
    telegram: bool = False,
    user_id: str = None,
    meta: dict = None,
) -> dict:
    # Валидация формата
    if telegram:
        file_ext = "ogg"
    else:
        if audio_format not in SUPPORTED_TTS_FORMATS:
            audio_format = DEFAULT_TTS_FORMAT
        file_ext = audio_format

    filename = _gen_audio_filename(extension=file_ext)
    audio_path = MEDIA_AUDIO_DIR / filename
    audio_url = f"/media/audio/{filename}"
    transcript = text.strip()
    meta = meta or {}
    start_time = time.monotonic()

    try:
        loop = asyncio.get_running_loop()
        audio_data = await loop.run_in_executor(
            None,
            generate_tts_sync,
            text,
            voice_id,
            file_ext,  # <-- Вот тут передаём формат
            TTS_HARD_TIME_LIMIT
        )
        file_path = await _write_audio_file(audio_data, filename)
        if isinstance(file_path, Path):
            file_path = str(file_path)
        elapsed = round(time.monotonic() - start_time, 2)
        meta.update({
            "tts_status": "ok",
            "voice_id": voice_id or ELEVEN_VOICE_ID,
            "elapsed_time": elapsed,
        })
        logger.info(f"tts_utils: TTS OK: voice={voice_id} | file={audio_url} | elapsed={elapsed}s")
    except Exception as final_e:
        elapsed = round(time.monotonic() - start_time, 2)
        meta.update({
            "tts_status": "failed",
            "error": str(final_e),
            "elapsed_time": elapsed,
            "format": file_ext,
        })
        logger.error(f"tts_utils: TTS FINAL FAIL: {final_e}")
        return {
            "reply_type": "text",
            "audio_url": None,
            "text_transcript": transcript,
            "meta": meta,
        }

    # -- OGG для Telegram --
    if telegram:
        ogg_filename = _gen_audio_filename(extension="ogg")
        ogg_path = MEDIA_AUDIO_DIR / ogg_filename
        try:
            await _convert_mp3_to_ogg(str(audio_path), str(ogg_path))
            audio_url = f"/media/audio/{ogg_filename}"
            os.remove(audio_path)
            meta["format"] = "ogg"
            logger.info(f"tts_utils: Audio for Telegram, mp3 converted to ogg: {audio_url}")
        except Exception as conv_e:
            logger.error(f"tts_utils: OGG conversion error: {conv_e}")
            log_error_to_db(
                user_id,
                "OGG conversion error",
                {"tts_error": str(conv_e), "filename": str(audio_path)}
            )
            meta.update({
                "tts_status": "failed",
                "error": str(conv_e),
                "format": "ogg",
            })
            return {
                "reply_type": "text",
                "audio_url": None,
                "text_transcript": transcript,
                "meta": meta,
            }
    else:
        meta["format"] = file_ext

    # --- Финальное обновление meta ---
    elapsed = round(time.monotonic() - start_time, 2)
    meta.update({
        "tts_status": "ok",
        "voice_id": str(voice_id or ELEVEN_VOICE_ID),
        "audio_url": str(audio_url),
        "elapsed_time": elapsed,
        "format": meta.get("format", file_ext),
        "transcript": transcript
    })

    return {
        "reply_type": "voice",
        "audio_url": str(audio_url),
        "text_transcript": transcript,
        "meta": meta,
    }
