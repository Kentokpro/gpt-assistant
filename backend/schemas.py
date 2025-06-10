from typing import Any, Optional, List, Dict
from pydantic import BaseModel, EmailStr, UUID4
from datetime import datetime
from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate

# --- Pydantic-модель для chat endpoint ---
class ChatRequest(BaseModel):
    content: str
    type: str = "text"

    class Config:
        schema_extra = {
            "example": {
                "content": "Привет, ассистент!",
                "type": "text"
            }
        }

# --- Pydantic-модель для поддержки ---
class SupportRequest(BaseModel):
    subject: Optional[str] = "Support request"
    message: str

    class Config:
        schema_extra = {
            "example": {
                "subject": "Проблема с подпиской",
                "message": "Не получается продлить подписку, выдает ошибку оплаты."
            }
        }

# --- LeadRequest (новый класс для лида) ---
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

    # НЕобязательные
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

    class Config:
        schema_extra = {
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

# --- User schemas ---
class UserRead(BaseUser):
    id: UUID4
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class UserCreate(BaseUserCreate):
    display_name: Optional[str] = None
    phone: Optional[str] = None
    tags: Optional[Dict] = None
    referral_code: Optional[str] = None

class UserUpdate(BaseUserUpdate):
    display_name: Optional[str] = None
    phone: Optional[str] = None
    tags: Optional[Dict] = None
    referral_code: Optional[str] = None

# --- Subscription schemas ---
class SubscriptionBase(BaseModel):
    status: Optional[str] = "inactive"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class SubscriptionRead(SubscriptionBase):
    id: UUID4
    user_id: UUID4
    created_at: Optional[datetime]

    class Config:
        orm_mode = True

# --- Session schemas ---
class SessionRead(BaseModel):
    id: UUID4
    user_id: UUID4
    created_at: Optional[datetime]
    expired_at: Optional[datetime]

    class Config:
        orm_mode = True

# --- Message schemas ---
class MessageRead(BaseModel):
    id: UUID4
    session_id: Optional[UUID4]
    user_id: UUID4
    role: str
    type: str
    status: str
    content: str
    meta: Optional[dict]
    created_at: Optional[datetime]

    class Config:
        orm_mode = True

# --- Error log schemas ---
class ErrorLogRead(BaseModel):
    id: UUID4
    user_id: Optional[UUID4]
    error: str
    details: Optional[dict]
    created_at: Optional[datetime]

    class Config:
        orm_mode = True
