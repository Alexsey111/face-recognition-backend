"""Инициализация базы данных - создание всех таблиц

Revision ID: 001_initial_migration
Revises: 
Create Date: 2024-01-01 00:00:00.000000

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
    
    # Для SQLite включаем batch режим
    if bind.dialect.name == 'sqlite':
        # SQLite требует batch для некоторых операций
        with op.batch_alter_table('users') as batch_op:
            pass  # Инициализация batch режима
    
    # Создание enum типов только для PostgreSQL
    if is_postgresql:
        user_role_enum = sa.Enum('user', 'admin', 'moderator', name='user_role_enum')
        session_type_enum = sa.Enum('verification', 'liveness', 'enrollment', 'identification', name='session_type_enum')
        session_status_enum = sa.Enum('pending', 'processing', 'completed', 'failed', 'expired', 'cancelled', name='session_status_enum')
        config_type_enum = sa.Enum('string', 'integer', 'float', 'boolean', 'json', name='config_type_enum')
        
        user_role_enum.create(bind)
        session_type_enum.create(bind)
        session_status_enum.create(bind)
        config_type_enum.create(bind)
        
        role_type = user_role_enum
        session_type_col = session_type_enum
        status_type = session_status_enum
        config_type_col = config_type_enum
    else:
        # Для SQLite используем String
        role_type = sa.String(20)
        session_type_col = sa.String(20)
        status_type = sa.String(20)
        config_type_col = sa.String(20)
    
    # Создание таблицы users
    op.create_table('users',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('password_salt', sa.String(255), nullable=False),
        sa.Column('first_name', sa.String(100), nullable=True),
        sa.Column('last_name', sa.String(100), nullable=True),
        sa.Column('role', role_type, nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('settings', sa.JSON(), nullable=True),
        sa.Column('total_uploads', sa.Integer(), nullable=False),
        sa.Column('total_verifications', sa.Integer(), nullable=False),
        sa.Column('successful_verifications', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создание таблицы references
    op.create_table('references',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('label', sa.String(100), nullable=True),
        sa.Column('file_url', sa.Text(), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('image_format', sa.String(20), nullable=True),
        sa.Column('image_dimensions', sa.JSON(), nullable=True),
        sa.Column('embedding', sa.Text(), nullable=False),
        sa.Column('embedding_version', sa.Integer(), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('original_filename', sa.String(255), nullable=True),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создание таблицы verification_sessions
    op.create_table('verification_sessions',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=True),
        sa.Column('reference_id', sa.String(36), nullable=True),
        sa.Column('session_type', session_type_col, nullable=False),
        sa.Column('status', status_type, nullable=False),
        sa.Column('request_data', sa.JSON(), nullable=True),
        sa.Column('response_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['reference_id'], ['references.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создание таблицы audit_logs
    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(36), nullable=True),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.String(36), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('old_values', sa.JSON(), nullable=True),
        sa.Column('new_values', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создание таблицы system_config
    op.create_table('system_config',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('key', sa.String(100), nullable=False),
        sa.Column('value', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('config_type', config_type_col, nullable=False),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_readonly', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_by', sa.String(36), nullable=True),
        sa.ForeignKeyConstraint(['updated_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создание таблицы api_keys
    op.create_table('api_keys',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('key_hash', sa.String(64), nullable=False),
        sa.Column('key_prefix', sa.String(8), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('permissions', sa.JSON(), nullable=True),
        sa.Column('rate_limit_per_hour', sa.Integer(), nullable=False),
        sa.Column('rate_limit_per_day', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_requests', sa.Integer(), nullable=False),
        sa.Column('successful_requests', sa.Integer(), nullable=False),
        sa.Column('failed_requests', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создание индексов для оптимизации производительности
    
    # Индексы для таблицы users
    op.create_index('idx_users_email_active', 'users', ['email', 'is_active'])
    op.create_index('idx_users_username_active', 'users', ['username', 'is_active'])
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_role', 'users', ['role'])
    op.create_index('ix_users_is_active', 'users', ['is_active'])
    op.create_index('ix_users_is_verified', 'users', ['is_verified'])
    op.create_index('ix_users_created_at', 'users', ['created_at'])
    
    # Индексы для таблицы references
    op.create_index('idx_reference_user_active', 'references', ['user_id', 'is_active'])
    op.create_index('idx_reference_quality', 'references', ['quality_score'])
    op.create_index('idx_reference_created', 'references', ['created_at'])
    op.create_index('ix_references_user_id', 'references', ['user_id'])
    op.create_index('ix_references_label', 'references', ['label'])
    op.create_index('ix_references_is_active', 'references', ['is_active'])
    op.create_index('ix_references_created_at', 'references', ['created_at'])
    
    # Уникальное ограничение для user_id + label
    op.create_unique_constraint('uq_user_label', 'references', ['user_id', 'label'])
    
    # Индексы для таблицы verification_sessions
    op.create_index('idx_session_type_status', 'verification_sessions', ['session_type', 'status'])
    op.create_index('idx_session_created', 'verification_sessions', ['created_at'])
    op.create_index('idx_session_user_type', 'verification_sessions', ['user_id', 'session_type'])
    op.create_index('idx_session_expires', 'verification_sessions', ['expires_at'])
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
        op.execute('DROP TYPE IF EXISTS user_role_enum')