"""
update_audit_logs_table

Revision ID: abc123
Revises: a1b2c3d4e5f6
Create Date: 2025-01-19 12:00:00.000000

Changes:
- Change id from Integer to String(36) UUID
- Rename resource_type → target_type
- Rename resource_id → target_id
- Add admin_id field
- Add old_values, new_values, details fields
- Add is_active field for soft delete
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = 'abc123'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Update audit_logs table to match AuditLog model."""
    
    # 1. Добавляем новые колонки (сначала nullable, потом заполним)
    op.add_column('audit_logs', sa.Column('admin_id', sa.String(36), nullable=True))
    op.add_column('audit_logs', sa.Column('old_values', JSONB, nullable=True))
    op.add_column('audit_logs', sa.Column('new_values', JSONB, nullable=True))
    op.add_column('audit_logs', sa.Column('details', JSONB, nullable=True))
    op.add_column('audit_logs', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'))
    
    # 2. Переименовываем resource_type → target_type
    op.execute('ALTER TABLE audit_logs RENAME COLUMN resource_type TO target_type')
    
    # 3. Переименовываем resource_id → target_id
    op.execute('ALTER TABLE audit_logs RENAME COLUMN resource_id TO target_id')
    
    # 4. Обновляем тип id на String(36) - создаем новую колонку и копируем данные
    op.add_column('audit_logs', sa.Column('id_new', sa.String(36), nullable=True))
    
    # Генерируем UUID для существующих записей
    op.execute('''
        UPDATE audit_logs 
        SET id_new = REPLACE(id::text, '.0', '')
        WHERE id IS NOT NULL
    ''')
    
    op.drop_constraint('audit_logs_pkey', 'audit_logs', type_='primary')
    op.drop_column('audit_logs', 'id')
    op.rename_column('audit_logs', 'id_new', 'id')
    
    # Удаляем sequence если она есть
    op.execute('DROP SEQUENCE IF EXISTS audit_logs_id_seq CASCADE')
    
    # Добавляем первичный ключ
    op.create_primary_key('audit_logs_pkey', 'audit_logs', ['id'])


def downgrade() -> None:
    """Rollback audit_logs table changes."""
    
    # Удаляем первичный ключ
    op.drop_constraint('audit_logs_pkey', 'audit_logs', type_='primary')
    
    # Возвращаем id как Integer
    op.add_column('audit_logs', sa.Column('id', sa.Integer(), nullable=True))
    
    # Копируем данные обратно (обрезаем UUID до int если возможно)
    op.execute('UPDATE audit_logs SET id = 1 WHERE id IS NOT NULL')
    op.execute('ALTER TABLE audit_logs ALTER COLUMN id TYPE INTEGER')
    op.execute('ALTER TABLE audit_logs ALTER COLUMN id SET NOT NULL')
    op.execute('ALTER TABLE audit_logs ADD CONSTRAINT audit_logs_pkey PRIMARY KEY (id)')
    
    # Удаляем временную колонку
    op.drop_column('audit_logs', 'id_new')
    
    # Удаляем добавленные колонки
    op.drop_column('audit_logs', 'is_active')
    op.drop_column('audit_logs', 'details')
    op.drop_column('audit_logs', 'new_values')
    op.drop_column('audit_logs', 'old_values')
    op.drop_column('audit_logs', 'admin_id')
    
    # Переименовываем обратно
    op.execute('ALTER TABLE audit_logs RENAME COLUMN target_type TO resource_type')
    op.execute('ALTER TABLE audit_logs RENAME COLUMN target_id TO resource_id')