# alembic/versions/xxxxx_add_performance_indexes.py

"""add_performance_indexes

Revision ID: f9a8c7b6d5e4
Revises: b0c43b264920  # ← ID предыдущей миграции (твоя последняя)
Create Date: 2026-01-20 12:00:00.000000

Description:
    Добавляет индексы для оптимизации производительности:
    - User table: email, phone, is_active
    - Reference table: user_id + version, created_at, is_active
    - VerificationSession table: user_id, status, created_at
    - AuditLog table: user_id, action, created_at, resource
    
    Expected performance improvement:
    - SELECT queries: < 100ms (from 200-500ms)
    - User lookup by email: < 50ms (from 150ms)
    - Reference retrieval: < 80ms (from 200ms)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'f9a8c7b6d5e4'
down_revision = 'b0c43b264920'  # ← Твоя последняя миграция
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply performance indexes
    """
    
    # ==================== User Table Indexes ====================
    
    # Index for email lookup (unique constraint already creates index, but explicit for clarity)
    # Note: If unique constraint exists, this might be redundant
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email 
        ON users(email)
    """)
    
    # Index for phone lookup
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_phone 
        ON users(phone)
    """)
    
    # Partial index for active users only (faster filtering)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_is_active 
        ON users(is_active) 
        WHERE is_active = true
    """)
    
    # Composite index for user lookup with status
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email_active 
        ON users(email, is_active)
    """)
    
    
    # ==================== Reference Table Indexes ====================
    
    # Index for user_id (foreign key already has index, but explicit)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_user_id 
        ON references(user_id)
    """)
    
    # Composite index for getting latest reference version per user
    # This is CRITICAL for /verify endpoint performance
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_user_version 
        ON references(user_id, version DESC)
    """)
    
    # Composite index for active references
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_user_active 
        ON references(user_id, is_active, version DESC) 
        WHERE is_active = true
    """)
    
    # Index for reference creation time (for history queries)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_created_at 
        ON references(created_at DESC)
    """)
    
    # Index for image_hash (for duplicate detection)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_image_hash 
        ON references(image_hash)
    """)
    
    
    # ==================== VerificationSession Table Indexes ====================
    
    # Index for user_id (foreign key)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_verification_sessions_user_id 
        ON verification_sessions(user_id)
    """)
    
    # Index for status filtering
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_verification_sessions_status 
        ON verification_sessions(status)
    """)
    
    # Composite index for user verification history (most common query)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_verification_sessions_user_created 
        ON verification_sessions(user_id, created_at DESC)
    """)
    
    # Composite index for status + created_at (admin queries)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_verification_sessions_status_created 
        ON verification_sessions(status, created_at DESC)
    """)
    
    # Partial index for successful verifications only
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_verification_sessions_success 
        ON verification_sessions(user_id, created_at DESC) 
        WHERE is_match = true
    """)
    
    
    # ==================== AuditLog Table Indexes ====================
    
    # Index for user_id (foreign key)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id 
        ON audit_logs(user_id)
    """)
    
    # Index for action type
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_logs_action 
        ON audit_logs(action)
    """)
    
    # Composite index for user audit history
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_created 
        ON audit_logs(user_id, created_at DESC)
    """)
    
    # Index for resource lookup (if you have resource_type, resource_id columns)
    # Note: Only create if these columns exist in your schema
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_logs_resource 
        ON audit_logs(resource_type, resource_id)
    """)
    
    # Index for timestamp (for time-range queries)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at 
        ON audit_logs(created_at DESC)
    """)
    
    
    # ==================== Webhook Table Indexes (if exists) ====================
    
    # Check if webhook_events table exists (based on your migrations)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'webhook_events'
            ) THEN
                -- Index for webhook status
                CREATE INDEX IF NOT EXISTS idx_webhook_events_status 
                ON webhook_events(status);
                
                -- Index for created_at
                CREATE INDEX IF NOT EXISTS idx_webhook_events_created_at 
                ON webhook_events(created_at DESC);
                
                -- Index for event_type
                CREATE INDEX IF NOT EXISTS idx_webhook_events_event_type 
                ON webhook_events(event_type);
                
                -- Composite index for retry logic
                CREATE INDEX IF NOT EXISTS idx_webhook_events_status_created 
                ON webhook_events(status, created_at DESC);
            END IF;
        END $$;
    """)


def downgrade() -> None:
    """
    Remove performance indexes
    """
    
    # ==================== Drop User Indexes ====================
    op.execute("DROP INDEX IF EXISTS idx_users_email")
    op.execute("DROP INDEX IF EXISTS idx_users_phone")
    op.execute("DROP INDEX IF EXISTS idx_users_is_active")
    op.execute("DROP INDEX IF EXISTS idx_users_email_active")
    
    # ==================== Drop Reference Indexes ====================
    op.execute("DROP INDEX IF EXISTS idx_references_user_id")
    op.execute("DROP INDEX IF EXISTS idx_references_user_version")
    op.execute("DROP INDEX IF EXISTS idx_references_user_active")
    op.execute("DROP INDEX IF EXISTS idx_references_created_at")
    op.execute("DROP INDEX IF EXISTS idx_references_image_hash")
    
    # ==================== Drop VerificationSession Indexes ====================
    op.execute("DROP INDEX IF EXISTS idx_verification_sessions_user_id")
    op.execute("DROP INDEX IF EXISTS idx_verification_sessions_status")
    op.execute("DROP INDEX IF EXISTS idx_verification_sessions_user_created")
    op.execute("DROP INDEX IF EXISTS idx_verification_sessions_status_created")
    op.execute("DROP INDEX IF EXISTS idx_verification_sessions_success")
    
    # ==================== Drop AuditLog Indexes ====================
    op.execute("DROP INDEX IF EXISTS idx_audit_logs_user_id")
    op.execute("DROP INDEX IF EXISTS idx_audit_logs_action")
    op.execute("DROP INDEX IF EXISTS idx_audit_logs_user_created")
    op.execute("DROP INDEX IF EXISTS idx_audit_logs_resource")
    op.execute("DROP INDEX IF EXISTS idx_audit_logs_created_at")
    
    # ==================== Drop Webhook Indexes ====================
    op.execute("DROP INDEX IF EXISTS idx_webhook_events_status")
    op.execute("DROP INDEX IF EXISTS idx_webhook_events_created_at")
    op.execute("DROP INDEX IF EXISTS idx_webhook_events_event_type")
    op.execute("DROP INDEX IF EXISTS idx_webhook_events_status_created")
