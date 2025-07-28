#!/usr/bin/env python3
"""
Чистит записи Message и AudioEventLog с отсутствующими аудиофайлами.
Запускается после bash-скрипта, чтобы база не "засорялась" битым мусором.
Логирует все действия.
"""
import os
import sys
from pathlib import Path
import logging
from backend.database import SessionLocal
from backend.models import Message, AudioEventLog
from backend.utils.audio_utils import MEDIA_AUDIO_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
LOG_FILE = Path(__file__).parent / "cleanup_audio_db.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("audio-cleanup")

def file_exists(audio_url):
    """Проверяет, существует ли файл по относительному пути (audio_url: '/media/audio/filename.mp3')."""
    if not audio_url:
        return False
    fname = os.path.basename(audio_url)
    path = MEDIA_AUDIO_DIR / fname
    return path.exists()

def cleanup():
    deleted_msgs, deleted_logs = 0, 0
    with SessionLocal() as db:
        # Чистим Message с type="voice"
        msgs = db.query(Message).filter(Message.type == "voice").all()
        for msg in msgs:
            if not file_exists(msg.content):
                logger.info(f"Удаляю Message: {msg.id} (file not found: {msg.content})")
                db.delete(msg)
                deleted_msgs += 1
        db.commit()
        # Чистим AudioEventLog (может ссылаться на audio_url или путь)
        logs = db.query(AudioEventLog).all()
        for log in logs:
            if log.file_path and not file_exists(log.file_path):
                logger.info(f"Удаляю AudioEventLog: {log.id} (file not found: {log.file_path})")
                db.delete(log)
                deleted_logs += 1
        db.commit()
    logger.info(f"Очистка завершена. Удалено: {deleted_msgs} Message, {deleted_logs} AudioEventLog")

if __name__ == "__main__":
    cleanup()
    print("DB audio cleanup done. Смотри лог:", LOG_FILE)
