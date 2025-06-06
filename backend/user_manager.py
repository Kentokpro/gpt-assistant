import uuid
from fastapi_users.manager import BaseUserManager, UUIDIDMixin
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from models import User
from config import SECRET

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    # Тут можно добавить кастомные методы-обработчики (например, on_after_register)

async def get_user_db():
    from database import SessionLocal  # импорт внутри функции, чтобы избежать циклических импортов
    async with SessionLocal() as session:
        yield SQLAlchemyUserDatabase(session, User)

async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)
