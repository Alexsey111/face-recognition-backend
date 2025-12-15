"""Инициализация базы данных - создание всех таблиц

Revision ID: 001_initial_migration
Revises: 
Create Date: 2024-01-01 00:00:00.000000

CHANGES:
- Removed username, password_hash, password_salt from users (not in models.py)
- Removed user_role from users (not in models.py)
- Added phone to users
- Added full_name to users
- Renamed metadata to user_metadata in users
- Added last_verified_at to users (instead of last_login)
- Added deleted_at to users (for soft delete)
- Added deleted_at to references (for soft delete)
- Renamed metadata to metadata_info in references
- Added success field to audit_logs
- Added error_message to audit_logs (was missing in old migration)
- Removed username index from users
- Updated all field names to match models.py exactly
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_migration'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Создание всех таблиц для Face Recognition Service."""
    # Проверяем тип БД
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == 'postgresql'
    
    # Создание enum типов только для PostgreSQL
    if is_postgresql:
        # ✅ UPDATED: Удален user_role_enum (не используется в models.py)
        session_type_enum = sa.Enum(
            'verification', 
            'liveness', 
            'enrollment', 
            'identification', 
            name='session_type_enum'
        )
        session_status_enum = sa.Enum(
            'pending', 
            'processing', 
            'completed', 
            'failed', 
            'expired', 
            'cancelled', 
            name='session_status_enum'
        )
        config_type_enum = sa.Enum(
            'string', 
            'integer', 
            'float', 
            'boolean', 
            'json', 
            name='config_type_enum'
        )
        
        session_type_enum.create(bind)
        session_status_enum.create(bind)
        config_type_enum.create(bind)
        
        session_type_col = session_type_enum
        status_type = session_status_enum
        config_type_col = config_type_enum
    else:
        # Для SQLite используем String
        session_type_col = sa.String(20)
        status_type = sa.String(20)
        config_type_col = sa.String(20)
    
    # ========================================================================
    # Создание таблицы users
    # ========================================================================
    op.create_table('users',
        sa.Column('id', sa.String(36), nullable=False),
        
        # ✅ UPDATED: Removed username, password_hash, password_salt
        # ✅ ADDED: phone, full_name
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('phone', sa.String(20), nullable=True),
        sa.Column('full_name', sa.String(255), nullable=True),
        
        # ✅ REMOVED: role (not in models.py)
        # ✅ REMOVED: first_name, last_name (replaced with full_name)
        
        # Статус
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false'),
        
        # Временные метки
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_verified_at', sa.DateTime(timezone=True), nullable=True),  # ✅ UPDATED: was last_login
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),  # ✅ ADDED: for soft delete
        
        # Метаданные
        sa.Column('user_metadata', sa.JSON(), nullable=True),  # ✅ UPDATED: was 'metadata'
        sa.Column('settings', sa.JSON(), nullable=True),
        
        # Статистика
        sa.Column('total_uploads', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_verifications', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('successful_verifications', sa.Integer(), nullable=False, server_default='0'),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # ========================================================================
    # Создание таблицы references
    # ========================================================================
    op.create_table('references',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('label', sa.String(100), nullable=True),
        
        # Файл
        sa.Column('file_url', sa.Text(), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('image_format', sa.String(20), nullable=True),
        sa.Column('image_dimensions', sa.JSON(), nullable=True),
        
        # Embedding
        sa.Column('embedding', sa.Text(), nullable=False),
        sa.Column('embedding_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('quality_score', sa.Float(), nullable=True),
        
        # Статус
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        
        # Временные метки
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),  # ✅ ADDED: for soft delete
        
        # Использование
        sa.Column('usage_count', sa.Integer(), nullable=False, server_default='0'),
        
        # Метаданные
        sa.Column('metadata_info', sa.JSON(), nullable=True),  # ✅ UPDATED: was 'metadata'
        sa.Column('original_filename', sa.String(255), nullable=True),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # ========================================================================
    # Создание таблицы verification_sessions
    # ========================================================================
    op.create_table('verification_sessions',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=True),
        sa.Column('reference_id', sa.String(36), nullable=True),
        
        # Тип и статус
        sa.Column('session_type', session_type_col, nullable=False),
        sa.Column('status', status_type, nullable=False, server_default='pending'),
        
        # Данные запроса/ответа
        sa.Column('request_data', sa.JSON(), nullable=True),
        sa.Column('response_data', sa.JSON(), nullable=True),
        
        # Временные метки
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        
        # Техническая информация
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        
        # Метаданные
        sa.Column('metadata_info', sa.JSON(), nullable=True),  # ✅ UPDATED: was 'metadata'
        
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reference_id'], ['references.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # ========================================================================
    # Создание таблицы audit_logs
    # ========================================================================
    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(36), nullable=True),
        
        # Информация о действии
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.String(36), nullable=True),
        
        # Детали действия
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('old_values', sa.JSON(), nullable=True),
        sa.Column('new_values', sa.JSON(), nullable=True),
        
        # Техническая информация
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        
        # ✅ ADDED: Статус операции
        sa.Column('success', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('error_message', sa.Text(), nullable=True),
        
        # Временная метка
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # ========================================================================
    # Создание таблицы system_config
    # ========================================================================
    op.create_table('system_config',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('key', sa.String(100), nullable=False),
        sa.Column('value', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('config_type', config_type_col, nullable=False, server_default='string'),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_readonly', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_by', sa.String(36), nullable=True),
        sa.ForeignKeyConstraint(['updated_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # ========================================================================
    # Создание таблицы api_keys
    # ========================================================================
    op.create_table('api_keys',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('key_hash', sa.String(64), nullable=False),
        sa.Column('key_prefix', sa.String(8), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('permissions', sa.JSON(), nullable=True),
        sa.Column('rate_limit_per_hour', sa.Integer(), nullable=False, server_default='1000'),
        sa.Column('rate_limit_per_day', sa.Integer(), nullable=False, server_default='10000'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_requests', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('successful_requests', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_requests', sa.Integer(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # ========================================================================
    # Создание индексов для оптимизации производительности
    # ========================================================================
    
    # Индексы для таблицы users
    op.create_index('idx_user_email', 'users', ['email'], unique=True)
    op.create_index('idx_user_phone', 'users', ['phone'], unique=True)
    op.create_index('idx_users_email_active', 'users', ['email', 'is_active'])
    op.create_index('ix_users_is_active', 'users', ['is_active'])
    op.create_index('ix_users_is_verified', 'users', ['is_verified'])
    op.create_index('ix_users_created_at', 'users', ['created_at'])
    op.create_index('ix_users_deleted_at', 'users', ['deleted_at'])  # ✅ ADDED
    
    # Индексы для таблицы references
    op.create_index('idx_reference_user', 'references', ['user_id'])
    op.create_index('idx_reference_version', 'references', ['user_id', 'embedding_version'])
    op.create_index('idx_reference_user_active', 'references', ['user_id', 'is_active'])
    op.create_index('idx_reference_quality', 'references', ['quality_score'])
    op.create_index('idx_reference_created', 'references', ['created_at'])
    op.create_index('ix_references_user_id', 'references', ['user_id'])
    op.create_index('ix_references_label', 'references', ['label'])
    op.create_index('ix_references_is_active', 'references', ['is_active'])
    op.create_index('ix_references_created_at', 'references', ['created_at'])
    op.create_index('ix_references_deleted_at', 'references', ['deleted_at'])  # ✅ ADDED
    
    # Уникальное ограничение для user_id + label
    if bind.dialect.name == 'sqlite':
        with op.batch_alter_table('references') as batch_op:
            batch_op.create_unique_constraint('uq_user_label', ['user_id', 'label'])
    else:
        op.create_unique_constraint('uq_user_label', 'references', ['user_id', 'label'])
    
    # Индексы для таблицы verification_sessions
    op.create_index('idx_verification_session_id', 'verification_sessions', ['id'], unique=True)
    op.create_index('idx_verification_user', 'verification_sessions', ['user_id'])
    op.create_index('idx_verification_created', 'verification_sessions', ['created_at'])
    op.create_index('idx_session_type_status', 'verification_sessions', ['session_type', 'status'])
    op.create_index('idx_session_user_type', 'verification_sessions', ['user_id', 'session_type'])
    op.create_index('idx_session_expires', 'verification_sessions', ['expires_at'])
    op.create_index('idx_sessions_user_created', 'verification_sessions', ['user_id', 'created_at'])
    op.create_index('ix_verification_sessions_user_id', 'verification_sessions', ['user_id'])
    op.create_index('ix_verification_sessions_reference_id', 'verification_sessions', ['reference_id'])
    op.create_index('ix_verification_sessions_session_type', 'verification_sessions', ['session_type'])
    op.create_index('ix_verification_sessions_status', 'verification_sessions', ['status'])
    op.create_index('ix_verification_sessions_created_at', 'verification_sessions', ['created_at'])
    
    # Индексы для таблицы audit_logs
    op.create_index('idx_audit_action', 'audit_logs', ['action'])
    op.create_index('idx_audit_resource', 'audit_logs', ['resource_type', 'resource_id'])
    op.create_index('idx_audit_user_action', 'audit_logs', ['user_id', 'action'])
    op.create_index('idx_audit_created', 'audit_logs', ['created_at'])
    op.create_index('idx_audit_logs_user_created', 'audit_logs', ['user_id', 'created_at'])
    op.create_index('idx_audit_success', 'audit_logs', ['success'])  # ✅ ADDED
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'])
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_logs_created_at', 'audit_logs', ['created_at'])
    
    # Индексы для таблицы system_config
    op.create_index('ix_system_config_key', 'system_config', ['key'], unique=True)
    op.create_index('ix_system_config_config_type', 'system_config', ['config_type'])
    op.create_index('ix_system_config_category', 'system_config', ['category'])
    op.create_index('ix_system_config_is_active', 'system_config', ['is_active'])
    
    # Индексы для таблицы api_keys
    op.create_index('ix_api_keys_user_id', 'api_keys', ['user_id'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)
    op.create_index('ix_api_keys_is_active', 'api_keys', ['is_active'])
    op.create_index('ix_api_keys_expires_at', 'api_keys', ['expires_at'])


def downgrade() -> None:
    """Удаление всех таблиц (откат миграции)."""
    
    # Проверяем тип БД для downgrade enum типов
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == 'postgresql'
    
    # Удаление таблиц в обратном порядке
    op.drop_table('api_keys')
    op.drop_table('system_config')
    op.drop_table('audit_logs')
    op.drop_table('verification_sessions')
    op.drop_table('references')
    op.drop_table('users')
    
    # Удаление enum типов только для PostgreSQL
    if is_postgresql:
        op.execute('DROP TYPE IF EXISTS config_type_enum')
        op.execute('DROP TYPE IF EXISTS session_status_enum')
        op.execute('DROP TYPE IF EXISTS session_type_enum')
        # ✅ REMOVED: user_role_enum (was deleted)
