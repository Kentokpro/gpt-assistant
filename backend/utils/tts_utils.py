"""
tts_utils.py — Работа с ElevenLabs TTS API, обработка ошибок, SLA, конвертация и логирование.
Зависимости:
- elevenlabs (pip)
- ffmpeg-python (pip)
- aiofiles, asyncio, os, time, logging, uuid
- backend.models (ErrorLog)
- backend.config (переменные, MEDIA_DIR, LOG_LEVEL)
"""

import os
import uuid
import time
import logging
import aiofiles
import asyncio
from elevenlabs.client import AsyncElevenLabs
import ffmpeg
from pathlib import Path
from backend.models import ErrorLog
from backend.config import (
    LOG_LEVEL, 
    MEDIA_DIR,         # тип: Path
    DEBUG,
    SENTRY_DSN
)

# === 1. Логгер ===
logger = logging.getLogger("leadinc-backend")
logger.setLevel(LOG_LEVEL)

# === 2. ENV-переменные для TTS ===
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")   # кастомный voice_id (например, "Алексей Михалёв")
ELEVEN_FALLBACK_VOICE_ID = os.getenv("ELEVENLABS_FALLBACK_VOICE_ID")   # стандартный fallback
MEDIA_AUDIO_DIR = MEDIA_DIR / "audio"
MEDIA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# === 3. Основной TTS-клиент ===
tts_client = AsyncElevenLabs(
    api_key=ELEVEN_API_KEY
)

# === 4. Настройки SLA/лимитов ===
TTS_SOFT_LIMIT = 45   # сек — SLA для ответа пользователю
TTS_HARD_LIMIT = 120  # сек — максимальный лимит на задачу
TTS_RETRY_COUNT = 2   # сколько раз пробуем делать TTS (retry)
AUDIO_FORMAT = "mp3"  # основной формат хранения (для веба)
OGG_FORMAT = "ogg"    # для Telegram

# === 5. Вспомогательные функции ===

async def _write_audio_file(data: bytes, filename: str) -> str:
    """
    Сохраняет аудиофайл в MEDIA_AUDIO_DIR, возвращает путь.
    """
    file_path = MEDIA_AUDIO_DIR / filename
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(data)
    logger.info(f"Saved audio file: {file_path}")
    return str(file_path)

def _gen_audio_filename(extension="mp3") -> str:
    """
    Генерирует уникальное имя для аудиофайла.
    """
    return f"{uuid.uuid4().hex}.{extension}"

async def _convert_mp3_to_ogg(mp3_path: str, ogg_path: str) -> None:
    """
    Конвертация mp3 → ogg через ffmpeg-python.
    """
    try:
        stream = ffmpeg.input(mp3_path)
        stream = ffmpeg.output(stream, ogg_path, acodec="libopus")
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        logger.info(f"Converted mp3 to ogg: {ogg_path}")
    except Exception as e:
        logger.error(f"Failed to convert mp3 to ogg: {e}")
        raise

async def log_error_to_db(session, user_id, error, details=None):
    """
    Пишет ошибку в ErrorLog (БД).
    """
    try:
        err = ErrorLog(
            user_id=user_id,
            error=error,
            details=details or {},
        )
        session.add(err)
        await session.commit()
        logger.info(f"Error logged to DB: {error} | {details}")
    except Exception as e:
        logger.error(f"Failed to log error to DB: {e}")

# === 6. Основная функция TTS ===

async def generate_tts(
    text: str,
    voice_id: str = None,
    fallback: bool = True,
    telegram: bool = False,
    user_id: str = None,
    session=None,  # SQLAlchemy AsyncSession для ErrorLog
    meta: dict = None,
) -> dict:
    """
    Генерирует голосовой ответ через ElevenLabs.
    Возвращает dict: audio_url, text_transcript, meta (SLA, статус и пр.).
    Если ошибка — логирует и возвращает ошибку с fallback на текст.
    """
    voice = voice_id or ELEVEN_VOICE_ID
    file_ext = AUDIO_FORMAT if not telegram else OGG_FORMAT
    filename = _gen_audio_filename(extension=file_ext)
    audio_path = MEDIA_AUDIO_DIR / filename
    audio_url = f"/media/audio/{filename}"
    transcript = text.strip()
    meta = meta or {}
    start_time = time.monotonic()
    elapsed = None

    # --- 1. Генерация TTS через ElevenLabs ---
    try:
        for attempt in range(1, TTS_RETRY_COUNT + 2):
            try:
                logger.info(f"TTS gen (ElevenLabs): voice_id={voice} | try={attempt}")
                tts_response = await asyncio.wait_for(
                    tts_client.text_to_speech(
                        text=text,
                        voice_id=voice,
                        model_id=None,  # auto
                        output_format=AUDIO_FORMAT,
                        optimize_streaming_latency="4"
                    ),
                    timeout=TTS_HARD_LIMIT,
                )
                audio_data = await tts_response.aread()
                await _write_audio_file(audio_data, filename)
                elapsed = round(time.monotonic() - start_time, 2)
                meta.update({
                    "tts_status": "ok",
                    "voice_id": voice,
                    "attempt": attempt,
                    "elapsed_time": elapsed,
                })
                logger.info(f"TTS OK: voice={voice} | file={audio_url} | elapsed={elapsed}s")
                break
            except asyncio.TimeoutError:
                err = f"TTS SLA timeout (> {TTS_HARD_LIMIT}s)"
                logger.error(err)
                await log_error_to_db(session, user_id, err, {
                    "tts_error": err, "voice_id": voice, "attempt": attempt, "text": text[:50]
                })
                if fallback and attempt == 1 and ELEVEN_FALLBACK_VOICE_ID:
                    voice = ELEVEN_FALLBACK_VOICE_ID
                    continue  # пробуем fallback-голос
                else:
                    raise
            except Exception as e:
                logger.error(f"TTS API error: {e}")
                await log_error_to_db(session, user_id, "TTS API error", {
                    "tts_error": str(e), "voice_id": voice, "attempt": attempt, "text": text[:50]
                })
                if fallback and attempt == 1 and ELEVEN_FALLBACK_VOICE_ID:
                    voice = ELEVEN_FALLBACK_VOICE_ID
                    continue  # fallback
                else:
                    raise
        else:
            raise Exception("TTS failed after retries")
    except Exception as final_e:
        elapsed = round(time.monotonic() - start_time, 2)
        meta.update({
            "tts_status": "failed",
            "error": str(final_e),
            "elapsed_time": elapsed,
        })
        # Возвращаем fallback: только текст
        return {
            "reply_type": "text",
            "audio_url": None,
            "text_transcript": transcript,
            "meta": meta,
        }

    # --- 2. (Optional) Конвертация для Telegram ---
    if telegram:
        ogg_filename = _gen_audio_filename(extension=OGG_FORMAT)
        ogg_path = MEDIA_AUDIO_DIR / ogg_filename
        try:
            await _convert_mp3_to_ogg(str(audio_path), str(ogg_path))
            audio_url = f"/media/audio/{ogg_filename}"
            os.remove(audio_path)  # удаляем исходный mp3 после конвертации
            meta["format"] = "ogg"
            logger.info(f"Audio for Telegram, mp3 converted to ogg: {audio_url}")
        except Exception as conv_e:
            logger.error(f"OGG conversion error: {conv_e}")
            await log_error_to_db(session, user_id, "OGG conversion error", {
                "tts_error": str(conv_e), "filename": str(audio_path)
            })
            # Возвращаем fallback: только текст
            meta.update({
                "tts_status": "failed",
                "error": str(conv_e)
            })
            return {
                "reply_type": "text",
                "audio_url": None,
                "text_transcript": transcript,
                "meta": meta,
            }

    # --- 3. Возврат результата ---
    elapsed = round(time.monotonic() - start_time, 2)
    meta.update({
        "tts_status": "ok",
        "voice_id": voice,
        "audio_url": audio_url,
        "elapsed_time": elapsed,
        "format": OGG_FORMAT if telegram else AUDIO_FORMAT,
        "transcript": transcript
    })
    return {
        "reply_type": "voice",
        "audio_url": audio_url,
        "text_transcript": transcript,
        "meta": meta,
    }
