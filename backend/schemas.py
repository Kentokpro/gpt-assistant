"""
Leadinc: Pydantic-схемы для всех API (text/voice), ErrorLog, SLA, транскрипций.
- Строго для Pydantic v2.11.7 (FastAPI 0.110+).
- Содержит расширенные схемы для чата (text/voice), логики аудиоответа, статуса Celery-задач и ошибок.
"""

from typing import Any, Optional, List, Dict, Literal, Union
from pydantic import BaseModel, EmailStr, UUID4, Field, ConfigDict
from datetime import datetime
from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate
from uuid import UUID
from pydantic import BaseModel, EmailStr, ConfigDict

# === 1. Текстовые и голосовые чаты (универсальный запрос) ===
class ChatRequest(BaseModel):
    content: Optional[str] = None
    type: Literal["text", "voice"] = "text"
    answer_format: Literal["text", "voice"] = "text"
    tts_format: Optional[Literal["mp3", "m4a", "ogg", "webm"]] = "mp3"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Расскажи как устроен Leadinc.",
                "type": "text",
                "answer_format": "voice"
            }
        }
    )

# === 2. Статус Celery задачи ===
class TaskStatus(BaseModel):
    task_id: str
    status: Literal["pending", "processing", "done", "failed", "timeout"]
    elapsed_time: Optional[float] = None
    reply_type: Optional[Literal["text", "voice"]] = None
    audio_url: Optional[str] = None
    text_transcript: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "7e5c6ed4-bd32-42d7-bba5-418f0b1e3fc0",
                "status": "done",
                "elapsed_time": 37.2,
                "reply_type": "voice",
                "audio_url": "/media/audio/7e5c6ed4-bd32-42d7-bba5-418f0b1e3fc0.mp3",
                "text_transcript": "Ваш заказ обработан. Leadinc рад помочь!",
                "result": None,
                "error": None,
                "meta": {
                    "voice_id": "abc123xyz",
                    "user_id": "uuid-...-...",
                    "sla_slow": False
                }
            }
        }
    )

# === 3. Голосовой ответ (API JSON) ===
class VoiceReply(BaseModel):
    reply_type: Literal["voice"] = "voice"
    audio_url: str
    text_transcript: str
    meta: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "reply_type": "voice",
                "audio_url": "/media/audio/6e3b1ff8-abe8-4f2f-a9e2-3f5379c82d2a.mp3",
                "text_transcript": "Ваш лид успешно найден. Leadinc поздравляет!",
                "meta": {
                    "task_id": "6e3b1ff8-abe8-4f2f-a9e2-3f5379c82d2a",
                    "voice_id": "xyz...",
                    "elapsed_time": 28.5,
                    "sla_slow": False
                }
            }
        }
    )

# === 4. Стандартный текстовый ответ ===
class TextReply(BaseModel):
    reply_type: Literal["text"] = "text"
    text: str
    meta: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reply_type": "text",
                "text": "Ваш вопрос принят! Ожидайте ответа.",
                "meta": {
                    "stage": 4,
                    "usage": {
                        "total_tokens": 243,
                        "model": "gpt-4o"
                    }
                }
            }
        }
    )

# === 5. ErrorLog ===
class ErrorLogRead(BaseModel):
    id: UUID
    user_id: Optional[UUID]
    error: str
    details: Optional[dict]
    created_at: Optional[datetime]

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "b98ae7b6-d6e5-11ec-9d64-0242ac120002",
                "user_id": "12345678-1234-1234-1234-123456789abc",
                "error": "STT failed",
                "details": {
                    "task_id": "9f0e3ab3-2f57-4c3b-bd0b-85405d6c5fa1",
                    "filename": "audio_7f7bde.mp3",
                    "reason": "Whisper API timeout",
                    "elapsed": 92.4,
                    "status": "timeout"
                },
                "created_at": "2025-07-16T09:23:44"
            }
        }
    )

# === 6. SupportRequest, LeadRequest, User/Subscription/Session/Message схемы ===

class SupportRequest(BaseModel):
    subject: Optional[str] = "Support request"
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "subject": "Проблема с подпиской",
                "message": "Не получается продлить подписку, выдает ошибку оплаты."
            }
        }
    )

class LeadRequest(BaseModel):
    user_id: str
    session_id: str
    subscription_id: str
    full_name: str
    email: EmailStr
    phone: str
    user_agent: str
    browser_lang: str
    submitted_at: datetime
    ip_address: str

    tags: Optional[dict] = None
    service_category: Optional[str] = None
    referral_code: Optional[str] = None
    telegram_id: Optional[str] = None
    from_bot: Optional[bool] = None
    device_type: Optional[str] = None
    app_version: Optional[str] = None
    subscription_type: Optional[str] = None
    trial_days_left: Optional[int] = None
    business_niche: Optional[str] = None
    client_type: Optional[str] = None
    form_errors: Optional[List[str]] = None
    input_time: Optional[str] = None
    is_duplicate: Optional[bool] = None
    original_id: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "12345",
                "session_id": "abcde-12345",
                "subscription_id": "sub-67890",
                "full_name": "Иван Иванов",
                "email": "ivan@example.com",
                "phone": "+79001112233",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "browser_lang": "ru",
                "submitted_at": "2025-06-09T12:34:56",
                "ip_address": "192.168.1.1",
                "tags": {"segment": "VIP", "region": "Москва", "utm": "yandex_direct"},
                "service_category": "Онлайн-курс",
                "referral_code": "FRIEND100",
                "telegram_id": "43434343",
                "from_bot": True,
                "device_type": "android",
                "app_version": "1.2.3",
                "subscription_type": "trial",
                "trial_days_left": 5,
                "business_niche": "строительство",
                "client_type": "b2c",
                "form_errors": ["email", "phone"],
                "input_time": "12:44:03",
                "is_duplicate": False,
                "original_id": "2fa6ee1"
            }
        }
    )

# --- User/Subscription/Session/Message ---

class UserRead(BaseModel):
    id: UUID
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

class UserCreate(BaseModel):
    display_name: Optional[str] = None
    phone: Optional[str] = None
    tags: Optional[Dict] = None
    referral_code: Optional[str] = None

class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    phone: Optional[str] = None
    tags: Optional[Dict] = None
    referral_code: Optional[str] = None

class SubscriptionBase(BaseModel):
    status: Optional[str] = "inactive"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class SubscriptionRead(SubscriptionBase):
    id: UUID
    user_id: UUID
    created_at: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

class SessionRead(BaseModel):
    id: UUID
    user_id: UUID
    created_at: Optional[datetime]
    expired_at: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

class MessageRead(BaseModel):
    id: UUID
    session_id: Optional[UUID]
    user_id: UUID
    role: str
    type: str
    status: str
    content: str
    meta: Optional[dict]
    created_at: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

# --- END schemas.py ---
