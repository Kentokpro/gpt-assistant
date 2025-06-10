from fastapi import Depends, HTTPException, status
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy, CookieTransport, AuthenticationBackend
from backend.models import User, Subscription
import uuid
from backend.config import SECRET_KEY, SESSION_COOKIE_NAME, SECURE_COOKIES
from sqlalchemy.future import select
from backend.database import SessionLocal
from backend.user_manager import get_user_manager

def get_jwt_strategy():
    return JWTStrategy(
        secret=SECRET_KEY,
        lifetime_seconds=60 * 60 * 24 * 7,  # 7 дней
        token_audience="fastapi-users"
    )

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

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)
current_superuser = fastapi_users.current_user(superuser=True)

# Проверка активной подписки — только для активных пользователей
async def require_active_subscription(user=Depends(current_active_user)):
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
