from typing import Any, Optional, List, Dict
from pydantic import BaseModel, EmailStr

class UserRead(BaseModel):
    id: int
    uuid: str
    username: str
    display_name: Optional[str]
    email: EmailStr
    auth_provider: Optional[str]
    roles: List[str]
    is_active: bool
    is_staff: bool
    is_admin: bool
    is_flagged: Optional[bool]
    invite_token: Optional[str]
    referral_code: Optional[str]
    language: Optional[str]
    timezone: Optional[str]
    last_login: Optional[str]
    last_activity: Optional[str]
    login_attempts: Optional[int]
    user_score: Optional[int]
    notifications_enabled: Optional[bool]
    tags: Optional[List[str]]
    blocked_until: Optional[str]
    deleted_at: Optional[str]
    created_at: Optional[str]

    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    # phone: Optional[str]  # если используешь

# Если есть другие Pydantic-схемы (например, SessionRead, MessageRead), сделай для них аналогично.
