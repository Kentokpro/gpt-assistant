"""add login column to users

Revision ID: 7a206ae179aa
Revises: c1aa0ca10ffb
Create Date: 2025-09-08 19:42:32.947558
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "7a206ae179aa"
down_revision: Union[str, None] = "c1aa0ca10ffb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("users", sa.Column("login", sa.String(), nullable=True))
    op.create_index("ix_users_login", "users", ["login"], unique=True)

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_users_login", table_name="users")
    op.drop_column("users", "login")
