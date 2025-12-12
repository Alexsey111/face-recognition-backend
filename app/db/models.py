"""
SQLAlchemy модели базы данных.
Определение всех таблиц и их структуры.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    JSON,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base


def generate_uuid() -> str:
    """Генерация UUID для primary key."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Получение текущего времени в UTC."""
    return datetime.now(timezone.utc)


class User(Base):
    """Модель пользователя системы."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)

    # Безопасность
    password_hash = Column(String(255), nullable=False)
    password_salt = Column(String(255), nullable=True)  # Опционально для legacy

    # Личная информация
    first_name = Column(String(100))
    last_name = Column(String(100))

    # Роли и статус
    role = Column(String(20), default="user", nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_verified = Column(Boolean, default=False, nullable=False, index=True)

    # Временные метки
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    last_login = Column(DateTime(timezone=True))

    # Метаданные
    metadata_info = Column("metadata", JSON)
    settings = Column(JSON)

    # Статистика
    total_uploads = Column(Integer, default=0, nullable=False)
    total_verifications = Column(Integer, default=0, nullable=False)
    successful_verifications = Column(Integer, default=0, nullable=False)

    # Relationships
    references = relationship(
        "Reference",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    verification_sessions = relationship(
        "VerificationSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    audit_logs = relationship(
        "AuditLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="noload",  # Не загружать автоматически (много записей)
    )
    api_keys = relationship(
        "ApiKey", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )
    system_configs_updated = relationship(
        "SystemConfig", back_populates="updater", lazy="noload"
    )

    # Индексы создаются в миграции

    def __repr__(self) -> str:
        return f"<User(id='{self.id}', username='{self.username}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "role": self.role,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata_info,
            "settings": self.settings,
            "total_uploads": self.total_uploads,
            "total_verifications": self.total_verifications,
            "successful_verifications": self.successful_verifications,
        }


class Reference(Base):
    """Модель эталонного изображения."""

    __tablename__ = "references"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    label = Column(String(100), index=True)
    file_url = Column(Text, nullable=False)
    file_size = Column(Integer)
    image_format = Column(String(20))
    image_dimensions = Column(JSON)
    embedding = Column(Text, nullable=False)
    embedding_version = Column(Integer, default=1, nullable=False)
    quality_score = Column(Float, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0, nullable=False)
    metadata_info = Column("metadata", JSON)
    original_filename = Column(String(255))
    checksum = Column(String(64))
    processing_time = Column(Float)

    # Relationships
    user = relationship("User", back_populates="references")
    verification_sessions = relationship(
        "VerificationSession", back_populates="reference", lazy="noload"
    )

    # Constraints (индексы в миграции)
    __table_args__ = (UniqueConstraint("user_id", "label", name="uq_user_label"),)

    def __repr__(self) -> str:
        return f"<Reference(id='{self.id}', label='{self.label}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "label": self.label,
            "file_url": self.file_url,
            "file_size": self.file_size,
            "image_format": self.image_format,
            "image_dimensions": self.image_dimensions,
            "embedding_version": self.embedding_version,
            "quality_score": self.quality_score,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "metadata": self.metadata_info,
            "original_filename": self.original_filename,
            "checksum": self.checksum,
            "processing_time": self.processing_time,
        }


class VerificationSession(Base):
    """
    Модель сессии верификации.
    """

    __tablename__ = "verification_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), index=True)
    reference_id = Column(String(36), ForeignKey("references.id"), index=True)

    # Тип и статус сессии
    session_type = Column(
        String(20), nullable=False, index=True
    )  # verification, liveness, etc.
    status = Column(String(20), default="pending", nullable=False, index=True)

    # Данные запроса и ответа
    request_data = Column(JSON)
    response_data = Column(JSON)

    # Временные метки
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Техническая информация
    ip_address = Column(String(45))  # IPv6 поддержка
    user_agent = Column(Text)
    processing_time = Column(Float)
    error_message = Column(Text)

    # Метаданные
    metadata_info = Column("metadata", JSON)

    # Связи
    user = relationship("User", back_populates="verification_sessions")
    reference = relationship("Reference", back_populates="verification_sessions")

    # Индексы
    __table_args__ = (
        Index("idx_session_type_status", "session_type", "status"),
        Index("idx_session_created", "created_at"),
        Index("idx_session_user_type", "user_id", "session_type"),
        Index("idx_session_expires", "expires_at"),
        Index("idx_sessions_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<VerificationSession(id='{self.id}', type='{self.session_type}', status='{self.status}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "reference_id": self.reference_id,
            "session_type": self.session_type,
            "status": self.status,
            "request_data": self.request_data,
            "response_data": self.response_data,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "expires_at": self.expires_at,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "metadata": self.metadata_info,
        }


class AuditLog(Base):
    """
    Модель журнала аудита.
    """

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), index=True)

    # Информация о действии
    action = Column(String(50), nullable=False, index=True)
    resource_type = Column(String(50), index=True)
    resource_id = Column(String(36), index=True)

    # Детали действия
    description = Column(Text)
    old_values = Column(JSON)
    new_values = Column(JSON)

    # Техническая информация
    ip_address = Column(String(45))
    user_agent = Column(Text)

    # Временная метка
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Связи
    user = relationship("User", back_populates="audit_logs")

    # Индексы
    __table_args__ = (
        Index("idx_audit_action", "action"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_user_action", "user_id", "action"),
        Index("idx_audit_created", "created_at"),
        Index("idx_audit_logs_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action='{self.action}', resource='{self.resource_type}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "description": self.description,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at,
        }


class SystemConfig(Base):
    """
    Модель системных настроек.
    """

    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text)
    description = Column(Text)

    # Тип и категория настройки
    config_type = Column(String(20), default="string", index=True)
    category = Column(String(50), index=True)

    # Статус
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_readonly = Column(Boolean, default=False, nullable=False)

    # Временные метки
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    updated_by = Column(String(36), ForeignKey("users.id"))

    # Связи
    updater = relationship("User", back_populates="system_configs_updated")

    def __repr__(self) -> str:
        return f"<SystemConfig(key='{self.key}', type='{self.config_type}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "description": self.description,
            "config_type": self.config_type,
            "category": self.category,
            "is_active": self.is_active,
            "is_readonly": self.is_readonly,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
        }


class ApiKey(Base):
    """
    Модель API ключей.
    """

    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # Информация о ключе
    name = Column(String(100), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    key_prefix = Column(String(8), nullable=False)

    # Статус и права доступа
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    permissions = Column(JSON)

    # Ограничения
    rate_limit_per_hour = Column(Integer, default=1000)
    rate_limit_per_day = Column(Integer, default=10000)

    # Временные метки
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at = Column(DateTime(timezone=True), index=True)
    last_used = Column(DateTime(timezone=True))

    # Статистика использования
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)

    # Связи
    user = relationship("User", back_populates="api_keys")

    def __repr__(self) -> str:
        return (
            f"<ApiKey(id='{self.id}', name='{self.name}', prefix='{self.key_prefix}')>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "is_active": self.is_active,
            "permissions": self.permissions,
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "rate_limit_per_day": self.rate_limit_per_day,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used": self.last_used,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
        }
