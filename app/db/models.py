# app/db/models.py
import uuid
import enum
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, 
    ForeignKey, Index, LargeBinary, JSON, Text, func
)
from sqlalchemy.types import DateTime
from sqlalchemy.orm import relationship, Mapped, mapped_column

# Импортируем Base из database модуля
from .database import Base

# ====================================================================
# Enums
# ====================================================================

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

# ====================================================================
# User Model
# ====================================================================

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        #  Все индексы создаются только в миграциях
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=True)  # ← ДОБАВИТЬ
    phone = Column(String(20), unique=True, nullable=True)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_verified_at = Column(DateTime(timezone=True), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional fields
    user_metadata = Column(JSON, nullable=True)
    settings = Column(JSON, nullable=True)
    total_uploads = Column(Integer, default=0, nullable=False)
    total_verifications = Column(Integer, default=0, nullable=False)
    successful_verifications = Column(Integer, default=0, nullable=False)
    
    # Relationships
    references = relationship("Reference", back_populates="user", cascade="all, delete-orphan")
    verification_sessions = relationship("VerificationSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    system_configs = relationship("SystemConfig", back_populates="updated_by_user", cascade="all, delete-orphan")
    webhook_configs = relationship("WebhookConfig", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"

# ====================================================================
# Reference Model (Face Embedding Storage)
# ====================================================================

class Reference(Base):
    __tablename__ = "references"
    __comment__ = "Face embedding storage with encrypted and fast-access variants"
    __table_args__ = (
        #  Все индексы создаются только в миграциях
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Embedding data
    embedding_encrypted = Column(LargeBinary, nullable=False)
    embedding = Column(JSON, nullable=True)
    embedding_version = Column(String(20), nullable=True)
    embedding_hash = Column(String(64), nullable=True)
    
    # Metadata
    label = Column(String(255), nullable=True)
    file_url = Column(Text, nullable=True)
    image_filename = Column(String(255), nullable=True)
    image_size_mb = Column(Float, nullable=True)
    image_format = Column(String(10), nullable=True)
    face_landmarks = Column(JSON, nullable=True)
    quality_score = Column(Float, nullable=True)
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    previous_reference_id = Column(String(36), ForeignKey("references.id", ondelete="SET NULL"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="references")
    verification_sessions = relationship(
        "VerificationSession",
        back_populates="reference",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Reference(id={self.id}, user_id={self.user_id}, label={self.label})>"
    
# ====================================================================
# Verification Session Model
# ====================================================================

class VerificationSession(Base):
    __tablename__ = "verification_sessions"
    __table_args__ = (
        #  Все индексы создаются только в миграциях
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), unique=True, nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reference_id = Column(String(36), ForeignKey("references.id", ondelete="SET NULL"), nullable=True)
    
    # Type and status
    session_type = Column(String(20), nullable=False, default="verification")
    status = Column(String(20), default=VerificationStatus.PENDING)
    
    # Request/response data
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Technical information
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    metadata_info = Column(JSON, nullable=True)
    
    # Verification results
    is_match = Column(Boolean, nullable=True)
    similarity_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    threshold_used = Column(Float, nullable=True)
    
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

# ====================================================================
# Audit Log Model
# ====================================================================

class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        #  Все индексы создаются только в миграциях
    )
    
    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    
    # Action details
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(36), nullable=True)
    description = Column(Text, nullable=True)
    
    # User information
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Change tracking
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    
    # Request metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<AuditLog(action={self.action}, user_id={self.user_id})>"

# ====================================================================
# System Configuration Model
# ====================================================================

class SystemConfig(Base):
    __tablename__ = "system_config"
    __table_args__ = (
        # Все индексы создаются только в миграциях
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

# ====================================================================
# API Keys Model
# ====================================================================

class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        #  Все индексы создаются только в миграциях
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
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

# ====================================================================
# Webhook Models
# ====================================================================

class WebhookEventType(str, enum.Enum):
    """Типы событий для webhook"""
    VERIFICATION_COMPLETED = "verification.completed"
    LIVENESS_COMPLETED = "liveness.completed"
    REFERENCE_CREATED = "reference.created"
    USER_ACTIVITY = "user.activity"
    SYSTEM_ALERT = "system.alert"
    WEBHOOK_TEST = "webhook.test"


class WebhookStatus(str, enum.Enum):
    """Статусы отправки webhook"""
    PENDING = "pending"
    SENDING = "sending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    EXPIRED = "expired"


class WebhookConfig(Base):
    __tablename__ = "webhook_configs"
    __table_args__ = (
        # ✅ ИСПРАВЛЕНО: Все индексы создаются только в миграциях
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    webhook_url = Column(Text, nullable=False)
    secret = Column(String(255), nullable=False)
    event_types = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    timeout = Column(Integer, default=10, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)
    retry_delay = Column(Integer, default=1, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Statistics
    total_sent = Column(Integer, default=0, nullable=False)
    successful_sent = Column(Integer, default=0, nullable=False)
    failed_sent = Column(Integer, default=0, nullable=False)
    last_sent_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="webhook_configs")
    webhook_logs = relationship("WebhookLog", back_populates="webhook_config", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<WebhookConfig(user_id={self.user_id}, url={self.webhook_url[:50]}...)>"


class WebhookLog(Base):
    __tablename__ = "webhook_logs"
    __table_args__ = (
        # ✅ ИСПРАВЛЕНО: Все индексы создаются только в миграциях
    )
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    webhook_config_id = Column(String(36), ForeignKey("webhook_configs.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(String(50), nullable=False)
    payload = Column(JSON, nullable=False)
    payload_hash = Column(String(64), nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    last_attempt_at = Column(DateTime, nullable=True)
    next_retry_at = Column(DateTime, nullable=True)
    status = Column(String(20), default=WebhookStatus.PENDING, nullable=False)
    http_status = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    signature = Column(String(128), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_attempt_at = Column(DateTime(timezone=True), nullable=True)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional metadata
    processing_time = Column(Float, nullable=True)
    request_headers = Column(JSON, nullable=True)
    client_ip = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Relationships
    webhook_config = relationship("WebhookConfig", back_populates="webhook_logs")
    
    def __repr__(self):
        return f"<WebhookLog(config_id={self.webhook_config_id}, event={self.event_type}, status={self.status})>"
