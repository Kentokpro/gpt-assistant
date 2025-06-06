from fastapi_users import BaseUserManager, UUIDIDMixin
from typing import Optional, Union
from models import User
from config import SECRET_KEY
from database import SessionLocal

class UserManager(UUIDIDMixin, BaseUserManager[User, str]):
    reset_password_token_secret = SECRET_KEY
    verification_token_secret = SECRET_KEY

    async def on_after_register(self, user: User, request: Optional[object] = None):
        print(f"Пользователь зарегистрирован: {user.email}")

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[object] = None):
        print(f"Пользователь запросил восстановление пароля: {user.email}, токен: {token}")

    async def on_after_request_verify(self, user: User, token: str, request: Optional[object] = None):
        print(f"Пользователь запросил верификацию: {user.email}, токен: {token}")

async def get_user_manager(user_db=SessionLocal()):
    yield UserManager(user_db)
