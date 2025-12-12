"""
Константы приложения.
Определение всех констант, используемых в системе.
"""

from typing import Dict, List, Tuple
import logging
from datetime import timezone

# =============================================================================
# IMAGE CONSTANTS
# =============================================================================

# Поддерживаемые форматы изображений
IMAGE_FORMATS = ["JPEG", "JPG", "PNG", "WEBP", "HEIC", "HEIF"]

# Размеры изображений
IMAGE_DIMENSIONS = {
    "min_width": 224,
    "min_height": 224,
    "max_width": 4096,
    "max_height": 4096,
    "optimal_width": 512,
    "optimal_height": 512,
}

# Качество изображений
IMAGE_QUALITY = {
    "excellent_threshold": 0.9,
    "good_threshold": 0.7,
    "fair_threshold": 0.5,
    "poor_threshold": 0.3,
}

# =============================================================================
# FILE CONSTANTS
# =============================================================================

FILE_LIMITS = {
    "max_image_size": 10 * 1024 * 1024,  # 10MB
    "max_batch_size": 50,  # Максимум файлов в батче
    "max_filename_length": 255,
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".webp"],
}

# MIME типы
MIME_TYPES = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

# =============================================================================
# API CONSTANTS
# =============================================================================

# Лимиты API
API_LIMITS = {
    "default_requests_per_minute": 60,
    "default_burst": 10,
    "upload_requests_per_minute": 30,
    "verify_requests_per_minute": 100,
    "admin_requests_per_minute": 20,
    "health_requests_per_minute": 1000,
}

# Заголовки ответов
RESPONSE_HEADERS = {
    "request_id": "X-Request-ID",
    "rate_limit_limit": "X-RateLimit-Limit",
    "rate_limit_remaining": "X-RateLimit-Remaining",
    "rate_limit_reset": "X-RateLimit-Reset",
    "new_access_token": "X-New-Access-Token",
}

# =============================================================================
# CACHE CONSTANTS
# =============================================================================

# TTL для кэша (в секундах)
CACHE_TTL = {
    "user_session": 1800,  # 30 минут
    "verification_session": 1800,  # 30 минут
    "access_token": 1800,  # 30 минут
    "refresh_token": 604800,  # 7 дней
    "embedding": 3600,  # 1 час
    "rate_limit": 60,  # 1 минута
    "user_stats": 300,  # 5 минут
    "system_config": 3600,  # 1 час
    "maintenance_mode": 86400,  # 24 часа
}

# Ключи кэша
CACHE_KEYS = {
    "user_session": "session:user:{user_id}",
    "verification_session": "verification_session:{session_id}",
    "access_token": "access_token:{user_id}:{token}",
    "refresh_token": "refresh_token:{user_id}:{token}",
    "embedding": "embedding:{image_hash}",
    "rate_limit": "rate_limit:{identifier}",
    "maintenance_mode": "system:maintenance_mode",
    "revoked_token": "revoked_token:{jti}",
}

# =============================================================================
# DATABASE CONSTANTS
# =============================================================================

# Размеры страниц для пагинации
PAGINATION = {"default_page_size": 20, "max_page_size": 100, "min_page_size": 1}

# Время жизни сессий в БД
SESSION_LIFETIME = {
    "verification_session": 1800,  # 30 минут
    "user_session": 86400,  # 24 часа
    "api_key": 2592000,  # 30 дней
}

# =============================================================================
# ML CONSTANTS
# =============================================================================

# Параметры ML моделей
ML_CONFIG = {
    "embedding_model": {
        "name": "face_recognition_model",
        "version": "v1.0",
        "input_size": 512,
        "output_size": 512,
    },
    "liveness_model": {
        "name": "liveness_detection_model",
        "version": "v1.0",
        "input_size": 224,
        "confidence_threshold": 0.8,
    },
    "face_detection": {"min_face_size": 30, "scale_factor": 1.1, "min_neighbors": 5},
}

# Пороговые значения для ML
THRESHOLDS = {
    "verification": {
        "default": 0.8,
        "strict": 0.9,
        "lenient": 0.7,
        "min": 0.5,
        "max": 0.95,
    },
    "liveness": {"default": 0.8, "strict": 0.9, "lenient": 0.7},
    "image_quality": {"excellent": 0.9, "good": 0.7, "acceptable": 0.5, "poor": 0.3},
}

# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

# Параметры безопасности
SECURITY = {
    "jwt_secret_key_length": 32,
    "password_min_length": 8,
    "password_max_length": 128,
    "max_login_attempts": 5,
    "lockout_duration": 900,  # 15 минут
    "session_timeout": 1800,  # 30 минут
    "api_key_length": 32,
}

# Алгоритмы шифрования
ENCRYPTION_ALGORITHMS = {
    "primary": "AES-256-GCM",
    "fallback": "AES-256-CBC",
    "hash": "SHA-256",
    "hmac": "HMAC-SHA256",
}

# =============================================================================
# TIME CONSTANTS
# =============================================================================

# Временные константы (в секундах)
TIME_PERIODS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,  # 30 дней
    "year": 31536000,  # 365 дней
}

# Форматы времени
TIME_FORMATS = {
    "ISO8601": "%Y-%m-%dT%H:%M:%SZ",
    "SIMPLE": "%Y-%m-%d %H:%M:%S",
    "DETAILED": "%Y-%m-%d %H:%M:%S.%f",
}

# =============================================================================
# USER CONSTANTS
# =============================================================================

# Роли пользователей
USER_ROLES = {
    "USER": "user",
    "ADMIN": "admin",
    "MODERATOR": "moderator",
    "SERVICE": "service",
}

# Статусы пользователей
USER_STATUS = {
    "ACTIVE": True,
    "INACTIVE": False,
    "PENDING": "pending",
    "SUSPENDED": "suspended",
    "DELETED": "deleted",
}

# =============================================================================
# SYSTEM CONSTANTS
# =============================================================================

# Статусы системы
SYSTEM_STATUS = {
    "HEALTHY": "healthy",
    "DEGRADED": "degraded",
    "UNHEALTHY": "unhealthy",
    "MAINTENANCE": "maintenance",
}

# Статусы сессий верификации
SESSION_STATUS = {
    "PENDING": "pending",
    "PROCESSING": "processing",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "EXPIRED": "expired",
    "CANCELLED": "cancelled",
}

# Типы сессий
SESSION_TYPES = {
    "VERIFICATION": "verification",
    "LIVENESS": "liveness",
    "ENROLLMENT": "enrollment",
    "IDENTIFICATION": "identification",
}

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Правила валидации
VALIDATION_RULES = {
    "username": {"min_length": 3, "max_length": 50, "pattern": r"^[a-zA-Z0-9_-]+$"},
    "email": {
        "max_length": 255,
        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    },
    "password": {
        "min_length": 8,
        "max_length": 128,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_digit": True,
        "require_special": True,
    },
    "phone": {"pattern": r"^\+?[\d\s\-()]+$", "min_length": 10, "max_length": 15},
}

# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

# Уровни логирования
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Категории событий для аудита
AUDIT_EVENTS = {
    "USER_LOGIN": "user.login",
    "USER_LOGOUT": "user.logout",
    "USER_CREATE": "user.create",
    "USER_UPDATE": "user.update",
    "USER_DELETE": "user.delete",
    "REFERENCE_CREATE": "reference.create",
    "REFERENCE_UPDATE": "reference.update",
    "REFERENCE_DELETE": "reference.delete",
    "VERIFICATION_REQUEST": "verification.request",
    "VERIFICATION_RESULT": "verification.result",
    "SYSTEM_ERROR": "system.error",
    "SECURITY_EVENT": "security.event",
}

# =============================================================================
# EXTERNAL SERVICES CONSTANTS
# =============================================================================

# Таймауты для внешних сервисов (в секундах)
SERVICE_TIMEOUTS = {
    "ml_service": 30,
    "storage_service": 60,
    "webhook_service": 10,
    "database": 30,
    "cache": 5,
}

# Статусы ответов внешних сервисов
SERVICE_STATUS = {
    "SUCCESS": "success",
    "FAILED": "failed",
    "TIMEOUT": "timeout",
    "UNAVAILABLE": "unavailable",
}

# =============================================================================
# WEBHOOK CONSTANTS
# =============================================================================

# Типы webhook событий
WEBHOOK_EVENTS = {
    "USER_REGISTERED": "user.registered",
    "USER_LOGIN": "user.login",
    "VERIFICATION_COMPLETED": "verification.completed",
    "LIVENESS_COMPLETED": "liveness.completed",
    "REFERENCE_CREATED": "reference.created",
    "SYSTEM_ALERT": "system.alert",
}

# Стандартные webhook поля
WEBHOOK_FIELDS = {
    "event_type": "event_type",
    "event_id": "event_id",
    "timestamp": "timestamp",
    "user_id": "user_id",
    "data": "data",
}

# =============================================================================
# ERROR CONSTANTS
# =============================================================================

# Коды ошибок
ERROR_CODES = {
    "VALIDATION_ERROR": "VALIDATION_ERROR",
    "PROCESSING_ERROR": "PROCESSING_ERROR",
    "DATABASE_ERROR": "DATABASE_ERROR",
    "NOT_FOUND": "NOT_FOUND",
    "UNAUTHORIZED": "UNAUTHORIZED",
    "FORBIDDEN": "FORBIDDEN",
    "CONFLICT": "CONFLICT",
    "RATE_LIMIT_EXCEEDED": "RATE_LIMIT_EXCEEDED",
    "INTERNAL_ERROR": "INTERNAL_ERROR",
    "SERVICE_UNAVAILABLE": "SERVICE_UNAVAILABLE",
}

# =============================================================================
# METRICS CONSTANTS
# =============================================================================

# Метрики производительности
PERFORMANCE_METRICS = {
    "response_time": "response_time",
    "throughput": "throughput",
    "error_rate": "error_rate",
    "cpu_usage": "cpu_usage",
    "memory_usage": "memory_usage",
    "disk_usage": "disk_usage",
}

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Конфигурационные ключи
CONFIG_KEYS = {
    "DATABASE_URL": "DATABASE_URL",
    "REDIS_URL": "REDIS_URL",
    "S3_ENDPOINT_URL": "S3_ENDPOINT_URL",
    "ML_SERVICE_URL": "ML_SERVICE_URL",
    "JWT_SECRET_KEY": "JWT_SECRET_KEY",
    "ENCRYPTION_KEY": "ENCRYPTION_KEY",
    "CORS_ORIGINS": "CORS_ORIGINS",
    "DEBUG": "DEBUG",
    "LOG_LEVEL": "LOG_LEVEL",
}

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Флаги функций
FEATURE_FLAGS = {
    "enable_auto_enrollment": True,
    "enable_batch_upload": True,
    "enable_webhooks": True,
    "enable_audit_logging": True,
    "enable_rate_limiting": True,
    "enable_cors": True,
    "maintenance_mode": False,
}

# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Регулярные выражения
REGEX_PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "username": r"^[a-zA-Z0-9_-]{3,50}$",
    "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "phone": r"^\+?[\d\s\-()]+$",
    "url": r"^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$",
    "image_data_url": r"^data:image/[a-z]+;base64,[a-zA-Z0-9+/=]+$",
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

# Значения по умолчанию
DEFAULT_VALUES = {
    "page": 1,
    "per_page": 20,
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 5,
    "quality_threshold": 0.8,
    "confidence_threshold": 0.8,
}

# Минимальные числовые значения
EPSILON: float = 1e-9

# Уровни уверенности
CONFIDENCE_LEVELS = {
    "low": 0.5,
    "medium": 0.75,
    "high": 0.9,
}
