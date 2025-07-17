import os
import logging
import sentry_sdk
from config import SENTRY_DSN, ENVIRONMENT

def setup_sentry():
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=ENVIRONMENT,
            traces_sample_rate=0.5,
            send_default_pii=True
        )
        logging.getLogger("leadinc-backend").info("Sentry интегрирован")
    else:
        logging.getLogger("leadinc-backend").info("Sentry DSN не указан, мониторинг отключён.")
