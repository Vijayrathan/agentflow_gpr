"""add derived_layers (per-layer eps/sigma) to simulations

Revision ID: c4a7f1b26d38
Revises: b7d2e9f4c815
Create Date: 2026-08-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = 'c4a7f1b26d38'
down_revision: Union[str, Sequence[str], None] = 'b7d2e9f4c815'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # One entry per `layers` entry, same order:
    # {name, eps_r_dry, eps_r_wet, sigma_dry, sigma_wet}. Nullable — existing rows
    # were written before the derive manifest was joined in and stay NULL.
    op.add_column(
        'simulations',
        sa.Column('derived_layers', JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('simulations', 'derived_layers')
