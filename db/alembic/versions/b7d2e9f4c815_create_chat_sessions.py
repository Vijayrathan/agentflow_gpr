"""create chat_sessions table (multi-chat persistence)

Revision ID: b7d2e9f4c815
Revises: a3c1d4e7b912
Create Date: 2026-07-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = 'b7d2e9f4c815'
down_revision: Union[str, Sequence[str], None] = 'a3c1d4e7b912'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), nullable=False, server_default='New chat'),
        sa.Column('thread_id', sa.Text(), nullable=False),
        sa.Column('complete', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('has_dataset', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('session_state', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index(
        'ix_chat_sessions_user_updated',
        'chat_sessions',
        ['user_id', sa.text('updated_at DESC')],
    )


def downgrade() -> None:
    op.drop_index('ix_chat_sessions_user_updated', table_name='chat_sessions')
    op.drop_table('chat_sessions')
