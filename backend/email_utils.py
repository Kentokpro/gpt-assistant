import smtplib
import logging
from email.mime.text import MIMEText
from email.utils import formataddr

from backend.config import (
    EMAILS_FROM_EMAIL, EMAILS_FROM_NAME,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
)

logger = logging.getLogger("leadinc-backend")

async def send_email(to: str, subject: str, body: str, from_email: str = None) -> None:
    sender = formataddr((EMAILS_FROM_NAME, from_email or EMAILS_FROM_EMAIL))
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(sender, [to], msg.as_string())
        logger.info(f"✅ Email sent to {to} | Subject: {subject}")
    except Exception as e:
        logger.error(f"❌ Failed to send email to {to}: {e}")
        raise
