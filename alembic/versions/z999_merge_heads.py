"""Merge two branches into one

Revision ID: z999_merge_heads
Revises: abc123, f9a8c7b6d5e4
Create Date: 2026-01-27

"""
from alembic import op

# revision identifiers
revision = 'z999_merge_heads'
down_revision = 'abc123'
branch_labels = None
depends_on = ['abc123', 'f9a8c7b6d5e4']


def upgrade():
    """Merge completed - no schema changes needed."""
    pass


def downgrade():
    """Split branches - not recommended."""
    pass