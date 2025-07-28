import uuid
from datetime import datetime
from backend.database import SessionLocal
from backend.models import ErrorLog
import logging
from datetime import datetime
import json

# Указываем путь к своему лог-файлу (можно поменять на любой)
LOG_FILE = "/srv/leadinc-logs/error_log.log"

# Настраиваем логгер один раз при импорте
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("leadinc-errorlog")

def log_error_to_db(user_id, error, details=None):
    """
    Пишет ошибку в файл error_log.log.
    """
    record = {
        "user_id": str(user_id) if user_id else None,
        "error": error,
        "details": details,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.error(json.dumps(record, ensure_ascii=False))
