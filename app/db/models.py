import uuid
import enum
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Index, LargeBinary, JSON, Text, func
)
from sqlalchemy.orm import relationship, Mapped, mapped_column

# Импортируем Base из database модуля
from .database import Base

# ============================================================================
# Enums
# ============================================================================

class VerificationStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"

class ActionType(str, enum.Enum):
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    REFERENCE_CREATED = "reference_created"
    REFERENCE_UPDATED = "reference_updated"
    REFERENCE_DELETED = "reference_deleted"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"

# ============================================================================
# User Model
# ============================================================================

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("idx_user_email", "email", unique=True),
        Index("idx_user_phone", "phone", unique=True),
        Index("idx_user_is_active", "is_active"),
        # Убрали дублирующиеся индексы - они есть в миграции Alembic
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20), unique=True, nullable=True, index=True)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_verified_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True, index=True)
    
    # Additional fields from migration
    user_metadata = Column(JSON, nullable=True)
    settings = Column(JSON, nullable=True)
    total_uploads = Column(Integer, default=0, nullable=False)
    total_verifications = Column(Integer, default=0, nullable=False)
    successful_verifications = Column(Integer, default=0, nullable=False)
    
    # Relationships
    # Используем строковые имена классов, чтобы избежать ошибок порядка объявления
    references = relationship("Reference", back_populates="user", cascade="all, delete-orphan")
    verification_sessions = relationship("VerificationSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    system_configs = relationship("SystemConfig", back_populates="updated_by_user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"

# ============================================================================
# Reference Model (Face Embedding Storage)
# ============================================================================

class Reference(Base):
    __tablename__ = "references"
    __table_args__ = (
        # Убрали все индексы - они создаются в миграции Alembic
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    label = Column(String(100), nullable=True)
    
    # File information
    file_url = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=True)
    image_format = Column(String(20), nullable=True)
    image_dimensions = Column(JSON, nullable=True)
    
    # ML Data
    embedding = Column(Text, nullable=False)
    embedding_version = Column(Integer, default=1, index=True)
    quality_score = Column(Float, nullable=True)
    
    # Additional ML fields from models
    embedding_encrypted = Column(LargeBinary, nullable=False)
    embedding_hash = Column(String(255), nullable=False, index=True)
    face_landmarks = Column(JSON, nullable=True)
    
    # Versioning
    version = Column(Integer, default=1, index=True)
    previous_reference_id = Column(String(36), ForeignKey("references.id"), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_used = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True, index=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Additional metadata
    metadata_info = Column(JSON, nullable=True)
    original_filename = Column(String(255), nullable=True)
    checksum = Column(String(64), nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # Image metadata (from models)
    image_filename = Column(String(255), nullable=False)
    image_size_mb = Column(Float, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="references")
    verification_sessions = relationship("VerificationSession", back_populates="reference", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Reference(user_id={self.user_id}, version={self.version})>"

# ============================================================================
# Verification Session Model
# ============================================================================

class VerificationSession(Base):
    __tablename__ = "verification_sessions"
    __table_args__ = (
        # Убрали все индексы - они создаются в миграции Alembic
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), unique=True, nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    reference_id = Column(String(36), ForeignKey("references.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Type and status from migration
    session_type = Column(String(20), nullable=False, default="verification")
    status = Column(String(20), default=VerificationStatus.PENDING, index=True)
    
    # Request/response data
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=False)
    
    # Technical information
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    metadata_info = Column(JSON, nullable=True)
    
    # Additional fields from models
    is_match = Column(Boolean, nullable=True)
    similarity_score = Column(Float, nullable=True)  # 0.0 - 1.0
    confidence = Column(Float, nullable=True)
    
    # Face detection info
    face_detected = Column(Boolean, default=False)
    face_quality_score = Column(Float, nullable=True)
    
    # Liveness check results
    is_liveness_passed = Column(Boolean, nullable=True)
    liveness_score = Column(Float, nullable=True)
    liveness_method = Column(String(50), nullable=True)
    
    # Image data
    image_filename = Column(String(255), nullable=False)
    image_size_mb = Column(Float, nullable=False)
    
    # Error handling
    error_code = Column(String(50), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="verification_sessions")
    reference = relationship("Reference", back_populates="verification_sessions")
    
    def __repr__(self):
        return f"<VerificationSession(id={self.id}, status={self.status})>"

# ============================================================================
# Audit Log Model
# ============================================================================

class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        # Убрали все индексы - они создаются в миграции Alembic
    )
    
    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    
    # Action details
    action = Column(String(50), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(36), nullable=True)
    description = Column(Text, nullable=True)
    
    # User information
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Change tracking
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    
    # Request metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Status
    success = Column(Boolean, default=True, index=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    
    # Relationship
    user = relationship("User", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<AuditLog(action={self.action}, user_id={self.user_id})>"

# ============================================================================
# System Configuration Model
# ============================================================================

class SystemConfig(Base):
    __tablename__ = "system_config"
    __table_args__ = (
        # Убрали все индексы - они создаются в миграции Alembic
    )
    
    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    key = Column(String(100), nullable=False, unique=True)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    config_type = Column(String(20), default="string", nullable=False)
    category = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_readonly = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    updated_by = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Relationship
    updated_by_user = relationship("User", back_populates="system_configs")
    
    def __repr__(self):
        return f"<SystemConfig(key={self.key}, value={self.value})>"

# ============================================================================
# API Keys Model
# ============================================================================

class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        # Убрали все индексы - они создаются в миграции Alembic
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(64), nullable=False, unique=True)
    key_prefix = Column(String(8), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    permissions = Column(JSON, nullable=True)
    rate_limit_per_hour = Column(Integer, default=1000, nullable=False)
    rate_limit_per_day = Column(Integer, default=10000, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used = Column(DateTime, nullable=True)
    total_requests = Column(Integer, default=0, nullable=False)
    successful_requests = Column(Integer, default=0, nullable=False)
    failed_requests = Column(Integer, default=0, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<ApiKey(user_id={self.user_id}, name={self.name})>"