"""
audio_utils.py — Универсальные утилиты для работы с аудиофайлами
— Централизованная валидация формата, размера, длительности (только через audio_constants.py)
— Конвертация между mp3/ogg с помощью ffmpeg-python
— Очистка временных файлов, логирование удаления и ошибок
— Все пути и лимиты централизованы, интеграция с log_error_to_db

Используется во всех модулях STT/TTS/чате, НЕ дублируется!

"""

import os
import logging
import ffmpeg
import uuid
import time
from pathlib import Path
from pydub.utils import mediainfo

from backend.utils.audio_constants import (
    ALLOWED_EXTENSIONS,
    MAX_AUDIO_SIZE_MB,
    MAX_AUDIO_DURATION_SEC,
)
from backend.utils.error_log_utils import log_error_to_db

# === Глобальные директории ===
MEDIA_AUDIO_DIR = Path("/srv/leadinc-media/audio")
MEDIA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("leadinc-backend")

# === 1. Централизованный валидатор ===
# Проверяет валидность аудиофайла по формату, размеру, длительности.
def audio_file_is_valid(filepath: str) -> tuple[bool, str, float]:
    ext = Path(filepath).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, "Неподдерживаемый формат аудио", 0.0
    try:
        if not os.path.isfile(filepath):
            return False, "Файл не существует", 0.0
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > MAX_AUDIO_SIZE_MB:
            return False, "Файл превышает лимит размера", 0.0
        info = mediainfo(filepath)
        duration_str = info.get("duration")
        if not duration_str:
            return False, "Не удалось определить длительность файла", 0.0
        try:
            duration = float(duration_str)
        except Exception:
            return False, "Ошибка преобразования длительности", 0.0
        if duration > MAX_AUDIO_DURATION_SEC:
            return False, "Файл слишком длинный (лимит 2 мин)", duration
        return True, "", duration
    except Exception as e:
        logger.warning(f"audio_utils: Ошибка валидации файла {filepath}: {e}")
        log_error_to_db(
            user_id=None,
            error="audio_file_is_valid_error",
            details={"filepath": str(filepath), "reason": str(e)},
        )
        return False, f"Ошибка проверки файла: {e}", 0.0

# === 2. Конвертация аудио ===
def convert_to_mp3(src_path: str, out_dir: Path = MEDIA_AUDIO_DIR) -> str:
    mp3_name = f"{uuid.uuid4()}.mp3"
    dst_path = out_dir / mp3_name
    try:
        (
            ffmpeg
            .input(src_path)
            .output(str(dst_path), format="mp3", acodec="libmp3lame", audio_bitrate="128k")
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"audio_utils: Конвертация {src_path} -> {dst_path}")
        return str(dst_path)
    except Exception as e:
        logger.error(f"audio_utils: Ошибка конвертации {src_path} в mp3: {e}")
        log_error_to_db(
            user_id=None,
            error="convert_to_mp3_error",
            details={"src_path": str(src_path), "reason": str(e)},
        )
        raise
#     Конвертирует любой аудиофайл в ogg (для Telegram).
def convert_to_ogg(src_path: str, out_dir: Path = MEDIA_AUDIO_DIR) -> str:
    ogg_name = f"{uuid.uuid4()}.ogg"
    dst_path = out_dir / ogg_name
    try:
        (
            ffmpeg
            .input(src_path)
            .output(str(dst_path), format="ogg", acodec="libopus", audio_bitrate="64k")
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"audio_utils: Конвертация {src_path} -> {dst_path}")
        return str(dst_path)
    except Exception as e:
        logger.error(f"audio_utils: Ошибка конвертации {src_path} в ogg: {e}")
        log_error_to_db(
            user_id=None,
            error="convert_to_ogg_error",
            details={"src_path": str(src_path), "reason": str(e)},
        )
        raise

# === 3. Очистка файлов (удаление старых/битых) ===
def cleanup_audio_files(older_than_days: int = 5):
    now = int(time.time())
    deleted = 0
    for fname in os.listdir(MEDIA_AUDIO_DIR):
        fpath = MEDIA_AUDIO_DIR / fname
        try:
            if not fpath.is_file():
                continue
            file_age_days = (now - int(fpath.stat().st_mtime)) / 86400
            if file_age_days > older_than_days:
                os.remove(fpath)
                logger.info(f"audio_utils: Удалён аудиофайл {fpath} (возраст {file_age_days:.1f} дн.)")
                log_error_to_db(
                    user_id=None,
                    error="audio_file_deleted_expired",
                    details={"filepath": str(fpath), "reason": "expired", "age_days": file_age_days}
                )
                deleted += 1
        except Exception as e:
            logger.warning(f"audio_utils: Не удалось удалить {fpath}: {e}")
            log_error_to_db(
                user_id=None,
                error="audio_cleanup_delete_failed",
                details={"filepath": str(fpath), "reason": str(e)}
            )
    logger.info(f"audio_utils: Всего удалено файлов за cleanup: {deleted}")

def delete_file(filepath: str, reason: str = ""):
    try:
        os.remove(filepath)
        logger.info(f"audio_utils: Удалён файл {filepath} ({reason})")
        log_error_to_db(
            user_id=None,
            error="audio_file_deleted",
            details={"filepath": str(filepath), "reason": reason}
        )
    except Exception as e:
        logger.warning(f"audio_utils: Ошибка при удалении {filepath}: {e}")
        log_error_to_db(
            user_id=None,
            error="audio_file_delete_failed",
            details={"filepath": str(filepath), "reason": str(e)}
        )

def convert_to_m4a(src_path: str, out_dir: Path = MEDIA_AUDIO_DIR) -> str:
    m4a_name = f"{uuid.uuid4()}.m4a"
    dst_path = out_dir / m4a_name
    try:
        (
            ffmpeg
            .input(src_path)
            .output(str(dst_path), format="ipod", acodec="aac", audio_bitrate="128k")
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"audio_utils: Конвертация {src_path} -> {dst_path}")
        return str(dst_path)
    except Exception as e:
        logger.error(f"audio_utils: Ошибка конвертации {src_path} в m4a: {e}")
        log_error_to_db(
            user_id=None,
            error="convert_to_m4a_error",
            details={"src_path": str(src_path), "reason": str(e)},
        )
        raise


# === 4. Вспомогательные функции (если нужны) ===
def is_allowed_audio(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS



# === END audio_utils.py ===
