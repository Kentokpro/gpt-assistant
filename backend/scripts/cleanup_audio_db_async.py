#!/usr/bin/env python3
"""
Асинхронная очистка БД от битых записей аудио.
- Удаляет из Message и AudioEventLog все записи, ссылающиеся на несуществующие аудиофайлы.
- Логирует все действия в отдельный log-файл.
"""

import os
import sys
from pathlib import Path
import logging
import asyncio

# Добавить корень проекта в sys.path для корректного импорта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal
from backend.models import Message, AudioEventLog
from backend.utils.audio_utils import MEDIA_AUDIO_DIR

LOG_FILE = Path(__file__).parent / "cleanup_audio_db_async.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("audio-cleanup")

def file_exists(audio_url):
    """Проверяет существование физ.файла (audio_url = '/media/audio/xxx.mp3')."""
    if not audio_url:
        return False
    fname = os.path.basename(audio_url)
    path = MEDIA_AUDIO_DIR / fname
    return path.exists()

async def cleanup():
    deleted_msgs, deleted_logs = 0, 0
    async with SessionLocal() as db:
        # Message (type="voice")
        msgs = await db.execute(
            Message.__table__.select().where(Message.type == "voice")
        )
        msgs = msgs.fetchall()
        for row in msgs:
            if not file_exists(row.content):
                logger.info(f"Удаляю Message: {row.id} (file not found: {row.content})")
                await db.execute(
                    Message.__table__.delete().where(Message.id == row.id)
                )
                deleted_msgs += 1
        await db.commit()
        # AudioEventLog
        logs = await db.execute(AudioEventLog.__table__.select())
        logs = logs.fetchall()
        for row in logs:
            if row.file_path and not file_exists(row.file_path):
                logger.info(f"Удаляю AudioEventLog: {row.id} (file not found: {row.file_path})")
                await db.execute(
                    AudioEventLog.__table__.delete().where(AudioEventLog.id == row.id)
                )
                deleted_logs += 1
        await db.commit()
    logger.info(f"Очистка завершена. Удалено: {deleted_msgs} Message, {deleted_logs} AudioEventLog")

if __name__ == "__main__":
    asyncio.run(cleanup())
    print(f"Async DB audio cleanup done. Смотри лог: {LOG_FILE}")
