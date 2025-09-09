"""
models.py

Leadinc: SQLAlchemy ORM-модели для всех сущностей системы:
- Пользователи, сессии, подписки, сообщения, логи ошибок и событий
- Поддержка мультимодальных сообщений: text, voice
- Логирование ошибок (ErrorLog) и SLA событий (SLAEventLog)
- Логика хранения ссылок на аудиофайлы, транскрипций, параметров аудио
- Архитектура под асинхронный FastAPI (async SQLAlchemy)
"""

import uuid
from datetime import datetime
from typing import Optional
from datetime import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    Integer,
    ForeignKey,
    JSON,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from backend.database import Base, SessionLocal

# --- User, Subscription, Session --- #

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    login = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)
    last_activity = Column(DateTime, nullable=True)
    login_attempts = Column(Integer, default=0)
    referral_code = Column(String, nullable=True)
    tags = Column(JSON, nullable=True)
    phone = Column(String, unique=True, index=True, nullable=True)

    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    error_logs = relationship("ErrorLog", back_populates="user", cascade="all, delete-orphan")
    audio_events = relationship("AudioEventLog", back_populates="user", cascade="all, delete-orphan")

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    status = Column(String, default="inactive")  # active, inactive, cancelled
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="subscriptions")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    expired_at = Column(DateTime, nullable=True)

    user = relationship("User")

# --- Сообщения (text/voice) --- #

class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    role = Column(String, nullable=False)    # "user" / "assistant"
    type = Column(String, default="text")    # "text" / "voice"
    status = Column(String, default="ok")    # "ok" / "error"
    content = Column(String, nullable=False) # текст сообщения или ссылка на аудиофайл
    meta = Column(JSON, nullable=True)       # usage, transcript, audio params, elapsed, error_details, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")

    # --- Audio-specific helpers --- #
    @property
    def is_voice(self):
        return self.type == "voice"

    @property
    def audio_url(self):
        if self.is_voice and self.content:
            return self.content
        return None

    @property
    def text_transcript(self):
        if self.meta and "transcript" in self.meta:
            return self.meta["transcript"]
        return None

    @property
    def audio_meta(self):
        if self.meta and "audio" in self.meta:
            return self.meta["audio"]
        return {}

    @classmethod
    def save_voice_message(cls, user_id, session_id, content, meta, role="assistant"):
        """
        Сохраняет голосовое сообщение (voice) в БД.
        user_id: str — ID пользователя
        session_id: str — ID сессии
        content: str — ссылка на аудиофайл (audio_url)
        meta: dict — метаданные (transcript, параметры аудио, SLA, ошибки)
        """

        # Создаём sync-сессию (также работает из Celery)
        with SessionLocalSync() as db:
            msg = cls(
                id=uuid.uuid4(),
                session_id=session_id,
                user_id=user_id,
                role="assistant",    # или "user", если хочешь логировать пользовательский voice
                type="voice",
                status="ok",
                content=content,
                meta=meta,
                created_at=datetime.utcnow()
            )
            db.add(msg)
            db.commit()
        return msg   

    @classmethod
    def save_voice_message_sync(cls, user_id, session_id, content, meta, role="assistant"):
        from backend.database import SessionLocalSync  
        with SessionLocalSync() as db:
            msg = cls(
                id=uuid.uuid4(),
                session_id=session_id,
                user_id=user_id,
                role=role,
                type="voice",
                status="ok",
                content=content,
                meta=meta,
                created_at=datetime.utcnow()
            )
            db.add(msg)
            db.commit()
        return msg

# --- Логи ошибок (в т.ч. SLA/timeout/ошибки STT/TTS) --- #
class ErrorLog(Base):
    __tablename__ = "error_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    error = Column(String, nullable=False)   # "STT failed", "TTS error", "SLA timeout", etc.
    details = Column(JSON, nullable=True)    # {filename, task_id, elapsed, reason, ...}
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="error_logs")

# --- Логи событий аудио (для аудиостека и SLA monitoring) --- #
class AudioEventLog(Base):
    __tablename__ = "audio_event_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True)
    message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=True
    ) 
    event_type = Column(String, nullable=False)   # "audio_created", "audio_deleted", "audio_played", "audio_expired", "audio_failed"
    file_path = Column(String, nullable=True)     # относительный путь к аудиофайлу
    status = Column(String, default="ok")         # "ok", "deleted", "failed", "timeout"
    details = Column(JSON, nullable=True)         # meta info, SLA, usage, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audio_events")

    @classmethod
    def create_event(cls, user_id, session_id, event_type, file_path, status, details):
        from backend.database import SessionLocalSync
        with SessionLocalSync() as db:
            event = cls(
                id=uuid.uuid4(),
                user_id=user_id,
                session_id=session_id,
                event_type=event_type,
                file_path=file_path,
                status=status,
                details=details,
                created_at=datetime.utcnow(),
            )
            db.add(event)
            db.commit()
        return event

"""
--- Примечания ---
- Все поля meta/details допускают любые параметры для гибкости (usage, model, elapsed_time, task_id, etc.)
- Для голосовых сообщений (type="voice") content — всегда ссылка на аудиофайл, meta содержит transcript, параметры аудио (длительность, размер), usage и ошибки, если есть.
- Для логов ошибок и аудио-событий — связь с user/session/message по UUID.
- При удалении user/session/message — соответствующие логи и аудио-события удаляются (cascade).
"""
