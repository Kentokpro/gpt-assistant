import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

env_path = BASE_DIR / ".env.backend"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()  # fallback: ищет .env везде

SECRET = os.getenv("SECRET")
SECRET_ADMIN_TOKEN = os.getenv("SECRET_ADMIN_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")

REDIS_URL = os.getenv("REDIS_URL")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
SECURE_COOKIES = os.getenv("SECURE_COOKIES", "false").lower() == "true"
CSRF_ENABLED = os.getenv("CSRF_ENABLED", "true").lower() == "true"
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "sessionid")

EMAILS_FROM_EMAIL = os.getenv("EMAILS_FROM_EMAIL")
EMAILS_FROM_NAME = os.getenv("EMAILS_FROM_NAME", "Leadinc Support")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_RESET_TOKEN_EXPIRE_HOURS = int(os.getenv("EMAIL_RESET_TOKEN_EXPIRE_HOURS", "48"))
EMAIL_TEMPLATES_DIR = os.getenv("EMAIL_TEMPLATES_DIR", "./email-templates")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", EMAILS_FROM_EMAIL)

SENTRY_DSN = os.getenv("SENTRY_DSN")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
GA_MEASUREMENT_ID = os.getenv("GA_MEASUREMENT_ID")
METRIKA_ID = os.getenv("METRIKA_ID")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
TIMEZONE = os.getenv("TIMEZONE", "Europe/Moscow")
