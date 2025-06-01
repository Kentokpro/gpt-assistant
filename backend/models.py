from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey,
    JSON, ARRAY, UniqueConstraint, func, Index
)
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.orm import relationship
import uuid as uuid_lib  # избегаем конфликта имён с полями uuid
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid_lib.uuid4)
    username = Column(String, nullable=False)
    display_name = Column(String)
    email = Column(String)
    auth_provider = Column(String(32), nullable=False, default="local")
    provider_id = Column(String)
    roles = Column(ARRAY(String), nullable=False, default=lambda: ["user"])
    is_active = Column(Boolean, default=True)
    is_staff = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    is_flagged = Column(Boolean, default=False)
    invite_token = Column(String)
    referral_code = Column(String)
    refresh_token = Column(UUID(as_uuid=True), default=uuid_lib.uuid4)
    language = Column(String(8), default="ru")
    timezone = Column(String(64))
    last_login = Column(DateTime(timezone=True))
    last_activity = Column(DateTime(timezone=True))
    login_attempts = Column(Integer, default=0)
    login_history = Column(JSON)
    user_score = Column(Integer, default=0)
    custom_settings = Column(JSON)
    notifications_enabled = Column(Boolean, default=True)
    tags = Column(ARRAY(String))
    blocked_until = Column(DateTime(timezone=True))
    deleted_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        UniqueConstraint("username", "auth_provider", name="uq_username_provider"),
        UniqueConstraint("email", "auth_provider", name="uq_email_provider"),
        Index("idx_users_referral", "referral_code"),
        Index("idx_users_tags", "tags"),
    )

    subscriptions = relationship("Subscription", back_populates="user")
    sessions = relationship("Session", back_populates="user")
    messages = relationship("Message", back_populates="user")

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    status = Column(String(32), nullable=False)
    plan = Column(String(32), nullable=False)
    started_at = Column(DateTime(timezone=True), default=func.now())
    expires_at = Column(DateTime(timezone=True))
    cancelled_at = Column(DateTime(timezone=True))
    payment_id = Column(String)
    payment_method = Column(String(32))
    auto_renew = Column(Boolean, default=False)

    user = relationship("User", back_populates="subscriptions")

    __table_args__ = (
        Index("idx_subscriptions_user_id", "user_id"),
        Index("idx_subscriptions_status", "status"),
    )

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid_lib.uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    session_token = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid_lib.uuid4)
    ip_address = Column(INET)
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    last_active_at = Column(DateTime(timezone=True), default=func.now())
    expires_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session")

    __table_args__ = (
        Index("idx_sessions_token", "session_token"),
        Index("idx_sessions_user_id", "user_id"),
        Index("idx_sessions_active", "is_active"),
    )

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid_lib.uuid4)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    role = Column(String(20), nullable=False)   # 'user' или 'assistant'
    type = Column(String(16), default="text")   # 'text', 'voice', 'image'
    status = Column(String(16), default="ok")
    content = Column(Text)
    usage_tokens = Column(Integer)
    meta = Column(JSON)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=func.now())

    user = relationship("User", back_populates="messages")
    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        Index("idx_messages_session_id", "session_id"),
        Index("idx_messages_user_id", "user_id"),
        Index("idx_messages_type", "type"),
        Index("idx_messages_created_at", "created_at"),
    )

class ErrorLog(Base):
    __tablename__ = "error_logs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="SET NULL"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    error_code = Column(String(64))
    error_message = Column(Text)
    stacktrace = Column(Text)
    component = Column(String(32))
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        Index("idx_errorlogs_user_id", "user_id"),
        Index("idx_errorlogs_session_id", "session_id"),
    )
