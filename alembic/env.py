import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

config = context.config

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.backend'))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend.models import Base, User, Session, Subscription, Message, ErrorLog
target_metadata = Base.metadata

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
if not SQLALCHEMY_DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set!")

if SQLALCHEMY_DATABASE_URL.startswith("postgresql+asyncpg"):
    SYNC_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("+asyncpg", "")
else:
    SYNC_DATABASE_URL = SQLALCHEMY_DATABASE_URL

print("[DEBUG] env.py is loaded and running.")
print("[DEBUG] SYNC_DATABASE_URL =", SYNC_DATABASE_URL)
print("[DEBUG] RUNNING run_migrations_online")
config.set_main_option('sqlalchemy.url', SYNC_DATABASE_URL)

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
