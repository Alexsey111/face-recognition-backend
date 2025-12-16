"""
Утилиты и вспомогательные функции.
Общие утилиты для всего приложения.
"""

# Exceptions
from .exceptions import (
    BaseAppException,
    ValidationError,
    ProcessingError,
    DatabaseError,
    NotFoundError,
    UnauthorizedError,
    ForbiddenError,
    ConflictError,
    EncryptionError,
    StorageError,
    CacheError,
    MLServiceError,
    WebhookError,
    RateLimitError,
    ConfigurationError,
    ExternalServiceError,
    BusinessLogicError,
    QuotaExceededError,
    MaintenanceModeError,
    TimeoutError,
    UnsupportedOperationError,
    RetryExhaustedError,
    DataIntegrityError,
)

# Logger
from .logger import (
    setup_logger,
    get_logger,
    LoggerMixin,
    StructuredLogger,
    structured_logger,
    log_function_call,
    configure_logging_for_environment,
)

# Aliases for backward compatibility
AppException = BaseAppException

# Constants and validators are available but not imported by default
# to avoid circular imports and keep imports explicit
# from .validators import *
# from .constants import *
# from .decorators import *

__all__ = [
    # Exceptions
    "AppException",
    "BaseAppException",
    "ValidationError",
    "ProcessingError",
    "DatabaseError",
    "NotFoundError",
    "UnauthorizedError",
    "ForbiddenError",
    "ConflictError",
    "EncryptionError",
    "StorageError",
    "CacheError",
    "MLServiceError",
    "WebhookError",
    "RateLimitError",
    "ConfigurationError",
    "ExternalServiceError",
    "BusinessLogicError",
    "QuotaExceededError",
    "MaintenanceModeError",
    "TimeoutError",
    "UnsupportedOperationError",
    "RetryExhaustedError",
    "DataIntegrityError",
    # Logger
    "setup_logger",
    "get_logger",
    "LoggerMixin",
    "StructuredLogger",
    "structured_logger",
    "log_function_call",
    "configure_logging_for_environment",
]
