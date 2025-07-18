"""Add unique constraint for phone in users

Revision ID: fb3022bb57b5
Revises: ff4d53ce1be7
Create Date: 2025-06-30 17:50:09.282663

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fb3022bb57b5'
down_revision: Union[str, None] = 'ff4d53ce1be7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(op.f('ix_users_phone'), 'users', ['phone'], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_users_phone'), table_name='users')
    # ### end Alembic commands ###
