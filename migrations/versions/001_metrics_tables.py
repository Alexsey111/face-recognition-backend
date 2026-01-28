"""Database migration for metrics tables.

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create verification_metrics table for FAR/FRR tracking
    op.create_table(
        'verification_metrics',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('verification_id', sa.String(255), nullable=False, unique=True),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('is_genuine', sa.Boolean(), nullable=False),
        sa.Column('is_accepted', sa.Boolean(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('threshold', sa.Float(), nullable=False),
        sa.Column('verification_type', sa.String(50), nullable=False, default='face_to_reference'),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('model_version', sa.String(100), nullable=True),
        sa.PrimaryKey('id'),
        sa.Index('ix_verification_metrics_timestamp', 'timestamp'),
        sa.Index('ix_verification_metrics_is_genuine', 'is_genuine'),
        sa.Index('ix_verification_metrics_is_accepted', 'is_accepted'),
        sa.Index('ix_verification_metrics_user_id', 'user_id'),
        sa.Index('ix_verification_metrics_threshold', 'threshold'),
    )

    # Create biometric_events table for 152-ФЗ compliance
    op.create_table(
        'biometric_events',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('event_id', sa.String(255), nullable=False, unique=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.PrimaryKey('id'),
        sa.Index('ix_biometric_events_timestamp', 'timestamp'),
        sa.Index('ix_biometric_events_event_type', 'event_type'),
        sa.Index('ix_biometric_events_user_id', 'user_id'),
    )

    # Create daily_metrics table for historical aggregation
    op.create_table(
        'daily_metrics',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False, unique=True),
        sa.Column('total_verifications', sa.Integer(), default=0),
        sa.Column('genuine_attempts', sa.Integer(), default=0),
        sa.Column('impostor_attempts', sa.Integer(), default=0),
        sa.Column('true_accepts', sa.Integer(), default=0),
        sa.Column('false_rejects', sa.Integer(), default=0),
        sa.Column('false_accepts', sa.Integer(), default=0),
        sa.Column('true_rejects', sa.Integer(), default=0),
        sa.Column('far_percent', sa.Float(), default=0.0),
        sa.Column('frr_percent', sa.Float(), default=0.0),
        sa.Column('eer_percent', sa.Float(), default=0.0),
        sa.Column('accuracy_percent', sa.Float(), default=0.0),
        sa.Column('avg_genuine_similarity', sa.Float(), default=0.0),
        sa.Column('avg_impostor_similarity', sa.Float(), default=0.0),
        sa.Column('avg_threshold', sa.Float(), default=0.0),
        sa.Column('compliance_far', sa.Boolean(), default=False),
        sa.Column('compliance_frr', sa.Boolean(), default=False),
        sa.Column('compliance_overall', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, default=sa.func.now()),
        sa.PrimaryKey('id'),
        sa.Index('ix_daily_metrics_date', 'date'),
    )

    # Create spoofing_attempts table
    op.create_table(
        'spoofing_attempts',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('attempt_id', sa.String(255), nullable=False, unique=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('attack_type', sa.String(50), nullable=False),
        sa.Column('detection_method', sa.String(100), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('blocked', sa.Boolean(), default=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.PrimaryKey('id'),
        sa.Index('ix_spoofing_attempts_timestamp', 'timestamp'),
        sa.Index('ix_spoofing_attempts_attack_type', 'attack_type'),
    )

    # Comments for documentation (152-ФЗ compliance)
    op.execute("""
        COMMENT ON TABLE verification_metrics IS 'Метрики верификации для расчета FAR/FRR';
        COMMENT ON COLUMN verification_metrics.is_genuine IS 'Ground truth: настоящий пользователь (True) или impostor (False)';
        COMMENT ON COLUMN verification_metrics.is_accepted IS 'Результат системы: принято (True) или отклонено (False)';
        COMMENT ON COLUMN verification_metrics.similarity_score IS 'Косинусная схожесть (0-1)';
        COMMENT ON TABLE biometric_events IS 'События биометрической обработки для аудита 152-ФЗ';
        COMMENT ON TABLE spoofing_attempts IS 'Заблокированные попытки спуфинга';
    """)


def downgrade() -> None:
    op.drop_table('spoofing_attempts')
    op.drop_table('daily_metrics')
    op.drop_table('biometric_events')
    op.drop_table('verification_metrics')