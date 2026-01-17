"""Add password_hash to users table

Revision ID: 736c812b73c2
Revises: 6245702b05ce
Create Date: 2026-01-14 15:30:39.970806

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '736c812b73c2'
down_revision = '6245702b05ce'
branch_labels = None
depends_on = None


def upgrade():
    # Добавляем password_hash как nullable=False (обязательное поле)
    op.add_column('users', sa.Column('password_hash', sa.String(length=255), nullable=False))


def downgrade():
    # Удаляем password_hash
    op.drop_column('users', 'password_hash')
