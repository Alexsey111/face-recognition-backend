"""
Add performance indexes for query optimization

Revision ID: f9a8c7b6d5e4
Revises: b0c43b264920
Create Date: 2025-01-20

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'f9a8c7b6d5e4'
down_revision = 'b0c43b264920'
branch_labels = None
depends_on = None


def upgrade():
    """Add performance indexes for common query patterns."""
    # Index for verification sessions
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_verification_sessions_user_id 
        ON verification_sessions (user_id);
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_verification_sessions_status 
        ON verification_sessions (status);
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_verification_sessions_created_at 
        ON verification_sessions (created_at DESC);
    """)
    
    # Index for references
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_user_id 
        ON references_table (user_id);
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_is_active 
        ON references_table (is_active);
    """)


def downgrade():
    """Remove performance indexes."""
    op.execute("DROP INDEX IF EXISTS idx_verification_sessions_user_id;")
    op.execute("DROP INDEX IF EXISTS idx_verification_sessions_status;")
    op.execute("DROP INDEX IF EXISTS idx_verification_sessions_created_at;")
    op.execute("DROP INDEX IF EXISTS idx_references_user_id;")
    op.execute("DROP INDEX IF EXISTS idx_references_is_active;")