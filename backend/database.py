"""
database.py — создание асинхронных и синхронных движков, сессий для FastAPI и Celery.
- Используется SessionLocal для async FastAPI (asyncpg).
- Используется SessionLocalSync для sync задач Celery (psycopg2).
- Base — базовый класс для моделей SQLAlchemy.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from backend.config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
)

DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

DATABASE_URL_SYNC = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Асинхронный движок и sessionmaker для FastAPI (asyncpg)
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# Синхронный движок и sessionmaker для Celery, миграций, CLI-утилит (psycopg2)
engine_sync = create_engine(
    DATABASE_URL_SYNC,
    pool_pre_ping=True,
    pool_recycle=3600,
)
SessionLocalSync = sessionmaker(
    bind=engine_sync,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# Базовый класс моделей
Base = declarative_base()
