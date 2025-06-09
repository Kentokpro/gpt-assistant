from fastapi import Request, HTTPException, status
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy, CookieTransport, AuthenticationBackend
from backend.models import User, Subscription
import uuid
from backend.config import SECRET_KEY, SESSION_COOKIE_NAME, SECURE_COOKIES
from sqlalchemy.future import select
from backend.database import SessionLocal
from backend.user_manager import get_user_manager

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

# Конструктор FastAPIUsers без схем — ты их задаёшь в main.py при регистрации
fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

# Функция декодирования токена — используется если нужно вручную
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

# 🔥 Главное: безопасная проверка активной подписки БЕЗ Depends(current_user)
async def require_active_subscription(request: Request):
    user = await fastapi_users.current_user(active=True)(request)

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
