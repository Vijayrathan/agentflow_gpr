"""drop top_air_extra_m column from simulations

Revision ID: a3c1d4e7b912
Revises: 1f8e2f7d543b
Create Date: 2026-05-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a3c1d4e7b912'
down_revision: Union[str, Sequence[str], None] = '1f8e2f7d543b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('simulations', 'top_air_extra_m')


def downgrade() -> None:
    op.add_column(
        'simulations',
        sa.Column('top_air_extra_m', sa.Float(), nullable=True),
    )
