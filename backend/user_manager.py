from typing import Optional
from uuid import UUID

from fastapi_users import BaseUserManager, UUIDIDMixin
from fastapi_users.db import SQLAlchemyUserDatabase

from backend.models import User
from backend.config import SECRET_KEY
from backend.database import SessionLocal


class UserManager(UUIDIDMixin, BaseUserManager[User, UUID]):
    reset_password_token_secret = SECRET_KEY
    verification_token_secret = SECRET_KEY

    async def on_after_register(self, user: User, request: Optional[object] = None):
        print(f"✅ Зарегистрирован пользователь: {user.email}")

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[object] = None):
        print(f"🔐 Восстановление пароля: {user.email}, токен: {token}")

    async def on_after_request_verify(self, user: User, token: str, request: Optional[object] = None):
        print(f"📩 Запрос на верификацию: {user.email}, токен: {token}")


async def get_user_manager():
    async with SessionLocal() as session:
        user_db = SQLAlchemyUserDatabase(session, User)
        yield UserManager(user_db)
