from fastapi import Depends, HTTPException, status
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy, CookieTransport, AuthenticationBackend
from models import User, Subscription
import uuid
from config import SECRET_KEY, SESSION_COOKIE_NAME, SECURE_COOKIES
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import EmailStr
from typing import Optional
from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate
from sqlalchemy.future import select
from database import SessionLocal
from user_manager import get_user_manager  # Импортируем новый менеджер

# --- Pydantic схемы для пользователя ---
class UserRead(BaseUser):
    display_name: Optional[str] = None
    email: Optional[EmailStr] = None

class UserCreate(BaseUserCreate):
    display_name: Optional[str] = None
    email: Optional[EmailStr] = None

class UserUpdate(BaseUserUpdate):
    display_name: Optional[str] = None
    email: Optional[EmailStr] = None

def get_jwt_strategy():
    return JWTStrategy(secret=SECRET_KEY, lifetime_seconds=60 * 60 * 24 * 7, token_audience="fastapi-users")

cookie_transport = CookieTransport(
    cookie_name=SESSION_COOKIE_NAME,
    cookie_max_age=60 * 60 * 24 * 7,
    cookie_secure=SECURE_COOKIES,
    cookie_httponly=True,
    cookie_samesite="lax"
)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# ВАЖНО! Используем get_user_manager вместо get_user_db
fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
current_superuser = fastapi_users.current_user(superuser=True)

# --- Проверка JWT вручную (опционально) ---
def decode_jwt_token(token: str):
    from jose import jwt, JWTError
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Упс, для доступа к AI нужна подписка",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- Зависимость для проверки подписки ---
async def require_active_subscription(user=Depends(current_active_user)):
    """
    Разрешает доступ только если у пользователя есть активная подписка.
    """
    async with SessionLocal() as session:
        result = await session.execute(
            select(Subscription)
            .where(Subscription.user_id == user.id)
            .where(Subscription.status == "active")
        )
        subscription = result.scalar_one_or_none()
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Упс! Для доступа к ассистенту Leadinc нужна активная подписка 😇",
            )
    return user
