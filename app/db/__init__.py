"""
Database package initialization.
Экспортируем всё необходимое для удобной работы с БД.
"""

# Основные утилиты
from . import database as _database

# Экспортируем Base и get_db напрямую, но db_manager делаем прокси, чтобы
# замена _database.db_manager во время runtime (например, в тестах) работала
# для всех импортов, которые обращаются к `app.db.db_manager`.
Base = _database.Base
get_db = _database.get_db


class _DBManagerProxy:
    def __getattr__(self, item):
        return getattr(_database.db_manager, item)

db_manager = _DBManagerProxy()

# Все модели
from .models import (
    User, Reference, VerificationSession,
    AuditLog, SystemConfig, ApiKey,
    WebhookConfig, WebhookLog,
    VerificationStatus, ActionType,
    WebhookEventType, WebhookStatus
)

# CRUD-классы (самые часто используемые)
from .crud import (
    UserCRUD,
    ReferenceCRUD,
    VerificationSessionCRUD,
    # Если будешь часто использовать — добавь сюда и другие
    # AuditLogCRUD,
)

__all__ = [
    # Утилиты
    "Base",
    "db_manager",
    "get_db",

    # Модели
    "User",
    "Reference",
    "VerificationSession",
    "AuditLog",
    "SystemConfig",
    "ApiKey",
    "WebhookConfig",
    "WebhookLog",
    "VerificationStatus",
    "ActionType",
    "WebhookEventType",
    "WebhookStatus",

    # CRUD
    "UserCRUD",
    "ReferenceCRUD",
    "VerificationSessionCRUD",
]
