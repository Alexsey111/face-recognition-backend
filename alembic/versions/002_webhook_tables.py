# alembic/versions/002_webhook_tables.py
"""Webhooks + Security fix (PostgreSQL)

Revision ID: 002_webhook_tables
Revises: 001_initial_migration
Create Date: 2025-01-30 12:00:00.000000

CHANGES:
- Добавлены таблицы webhook_configs и webhook_logs
- 🔒 SECURITY FIX: Удалено поле 'embedding' (plaintext)
- Добавлены индексы для PostgreSQL
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '002_webhook_tables'
down_revision = '001_initial_migration'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Создание webhook таблиц + удаление plaintext embedding."""
    
    print("\n" + "="*70)
    print("🐘 PostgreSQL Migration: Webhooks + Security Fix")
    print("="*70)
    
    # ========================================================================
    # 🔒 SECURITY FIX: Удаляем plaintext embedding
    # ========================================================================
    
    print("\n🔒 Removing plaintext 'embedding' field from references...")
    
    try:
        # Проверяем существование колонки
        conn = op.get_bind()
        inspector = sa.inspect(conn)
        columns = [col['name'] for col in inspector.get_columns('references')]
        
        if 'embedding' in columns:
            op.drop_column('references', 'embedding')
            print("✅ Removed plaintext 'embedding' column")
        else:
            print("ℹ️  Column 'embedding' not found - skipping")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
    
    # ========================================================================
    # Создание enum типов для PostgreSQL
    # ========================================================================
    
    print("\n📋 Creating PostgreSQL enum types...")
    
    webhook_event_type_enum = sa.Enum(
        'verification.completed',
        'liveness.completed',
        'reference.created',
        'user.activity',
        'system.alert',
        'webhook.test',
        name='webhook_event_type_enum',
        create_type=False  # ✅ Не создавать явно
    )
    
    webhook_status_enum = sa.Enum(
        'pending',
        'sending',
        'success',
        'failed',
        'retry',
        'expired',
        name='webhook_status_enum',
        create_type=False  # ✅ Не создавать явно
    )
    
    # ✅ УДАЛЕНО: Явное создание enum типов
    # webhook_event_type_enum.create(op.get_bind(), checkfirst=True)
    # webhook_status_enum.create(op.get_bind(), checkfirst=True)
    
    print("✅ Enum types defined (will be created automatically with tables)")

    
    # ========================================================================
    # Создание таблицы webhook_configs
    # ========================================================================
    
    print("\n📋 Creating webhook_configs table...")
    
    op.create_table('webhook_configs',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('webhook_url', sa.Text(), nullable=False),
        sa.Column('secret', sa.String(255), nullable=False),
        sa.Column('event_types', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('timeout', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('retry_delay', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('total_sent', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('successful_sent', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_sent', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    print("✅ webhook_configs table created")
    
    # ========================================================================
    # Создание таблицы webhook_logs
    # ========================================================================
    
    print("\n📋 Creating webhook_logs table...")
    
    op.create_table('webhook_logs',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('webhook_config_id', sa.String(36), nullable=False),
        sa.Column('event_type', webhook_event_type_enum, nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('payload_hash', sa.String(64), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_attempt_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', webhook_status_enum, nullable=False, server_default='pending'),
        sa.Column('http_status', sa.Integer(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('signature', sa.String(128), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('request_headers', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('client_ip', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['webhook_config_id'], ['webhook_configs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    print("✅ webhook_logs table created")
    
    # ========================================================================
    # Создание индексов
    # ========================================================================
    
    print("\n📋 Creating indexes...")
    
    # webhook_configs indexes
    op.create_index('idx_webhook_config_user_id', 'webhook_configs', ['user_id'])
    op.create_index('idx_webhook_config_active', 'webhook_configs', ['is_active'])
    op.create_index('idx_webhook_configs_user_active', 'webhook_configs', ['user_id', 'is_active'])
    op.create_index('ix_webhook_configs_created_at', 'webhook_configs', ['created_at'])
    
    # webhook_logs indexes
    op.create_index('idx_webhook_log_config_id', 'webhook_logs', ['webhook_config_id'])
    op.create_index('idx_webhook_log_status', 'webhook_logs', ['status'])
    op.create_index('idx_webhook_log_event_type', 'webhook_logs', ['event_type'])
    op.create_index('idx_webhook_log_next_retry', 'webhook_logs', ['next_retry_at'])
    op.create_index('idx_webhook_log_created_at', 'webhook_logs', ['created_at'])
    op.create_index('idx_webhook_log_payload_hash', 'webhook_logs', ['payload_hash'])
    op.create_index('idx_webhook_logs_config_status', 'webhook_logs', ['webhook_config_id', 'status'])
    op.create_index('idx_webhook_logs_status_retry', 'webhook_logs', ['status', 'next_retry_at'])
    
    # ✅ PostgreSQL specific: GIN index для JSONB
    op.create_index(
        'idx_webhook_configs_event_types_gin',
        'webhook_configs',
        ['event_types'],
        postgresql_using='gin'
    )
    op.create_index(
        'idx_webhook_logs_payload_gin',
        'webhook_logs',
        ['payload'],
        postgresql_using='gin'
    )
    
    print("✅ Indexes created (including GIN for JSONB)")
    
    print("\n" + "="*70)
    print("✅ Migration completed successfully")
    print("="*70 + "\n")


def downgrade() -> None:
    """Откат миграции."""
    
    print("\n" + "="*70)
    print("⏪ Rolling back migration...")
    print("="*70)
    
    # Удаление таблиц
    op.drop_table('webhook_logs')
    op.drop_table('webhook_configs')
    
    # Удаление enum типов
    op.execute('DROP TYPE IF EXISTS webhook_status_enum CASCADE')
    op.execute('DROP TYPE IF EXISTS webhook_event_type_enum CASCADE')
    
    # ⚠️ Восстановление plaintext embedding (НЕ РЕКОМЕНДУЕТСЯ)
    print("\n⚠️  WARNING: Restoring plaintext 'embedding' field")
    op.add_column('references', sa.Column('embedding', sa.Text(), nullable=True))
    
    print("\n⏪ Rollback completed")
    print("="*70 + "\n")
