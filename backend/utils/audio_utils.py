"""
audio_utils.py — утилиты для работы с аудиофайлами
— Проверка и валидация аудио (mp3/ogg), ограничения по размеру/длине
— Конвертация между mp3/ogg с помощью ffmpeg-python
— Очистка временных файлов, логирование удаления
— Все лимиты и пути — только через env и config
"""

import os
import logging
import ffmpeg
import uuid
from pathlib import Path
from pydub.utils import mediainfo
from backend.config import BASE_DIR

logger = logging.getLogger("leadinc-backend")

# Настройки лимитов (секунды, мегабайты)
MAX_AUDIO_DURATION_SEC = 120    # 2 минуты
MAX_AUDIO_SIZE_MB = 10          # 10 мегабайт

MEDIA_AUDIO_DIR = BASE_DIR / "media" / "audio"
MEDIA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def is_allowed_audio(filename: str) -> bool:
    """Разрешённые форматы для входного аудио"""
    return filename.lower().endswith((".mp3", ".ogg"))

def audio_file_size_ok(filepath: str) -> bool:
    """Проверить, что файл не превышает лимит по размеру"""
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    return size_mb <= MAX_AUDIO_SIZE_MB

def audio_duration_ok(filepath: str) -> bool:
    """Проверить длительность аудиофайла"""
    try:
        info = mediainfo(filepath)
        duration = float(info["duration"])
        return duration <= MAX_AUDIO_DURATION_SEC
    except Exception as e:
        logger.warning(f"audio_utils: Не удалось получить длительность файла {filepath}: {e}")
        return False

def convert_to_mp3(src_path: str, out_dir: Path = MEDIA_AUDIO_DIR) -> str:
    """Конвертировать любой аудиофайл в mp3, вернуть путь к mp3"""
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
        raise

def convert_to_ogg(src_path: str, out_dir: Path = MEDIA_AUDIO_DIR) -> str:
    """Конвертировать любой аудиофайл в ogg (для Telegram)"""
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
        raise

def cleanup_audio_files(older_than_days: int = 5):
    """Удалить файлы в MEDIA_AUDIO_DIR старше N дней. Логировать все удаления"""
    now = int(os.path.getmtime(MEDIA_AUDIO_DIR))
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
                deleted += 1
        except Exception as e:
            logger.warning(f"audio_utils: Не удалось удалить {fpath}: {e}")
    logger.info(f"audio_utils: Всего удалено файлов за cleanup: {deleted}")

def validate_audio_file(filepath: str) -> (bool, str):
    """Валидация аудиофайла перед STT/TTS: формат, размер, длительность"""
    if not is_allowed_audio(filepath):
        return False, "Неподдерживаемый формат аудио"
    if not audio_file_size_ok(filepath):
        return False, "Файл превышает лимит размера"
    if not audio_duration_ok(filepath):
        return False, "Файл слишком длинный (лимит 2 мин)"
    return True, ""
