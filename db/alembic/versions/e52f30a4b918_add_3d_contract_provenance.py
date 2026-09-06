"""Add nullable 3D geometry and execution provenance without inventing history.

Revision ID: e52f30a4b918
Revises: b7d2e9f4c815
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "e52f30a4b918"
down_revision = "b7d2e9f4c815"
branch_labels = None
depends_on = None

FIELDS = {"domain_z": sa.Float, "dimensionality": sa.Text,
          "coordinate_frame": sa.Text, "contract_version": sa.Integer,
          "contract_digest": sa.Text, "input_sha256": sa.Text,
          "resolved_scene": postgresql.JSONB, "requested_sample": postgresql.JSONB,
          "executed_metadata": postgresql.JSONB, "qualification_status": sa.Text}


def upgrade():
    for name, kind in FIELDS.items():
        op.add_column("simulations", sa.Column(name, kind(), nullable=True))


def downgrade():
    for name in reversed(FIELDS):
        op.drop_column("simulations", name)
