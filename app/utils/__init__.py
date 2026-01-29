"""
Утилиты и вспомогательные функции.
Общие утилиты для всего приложения.
"""

# Exceptions
from .exceptions import (
    BaseAppException,
    BusinessLogicError,
    CacheError,
    ConflictError,
    DatabaseError,
    DataIntegrityError,
    ExternalServiceError,
    ForbiddenError,
    MLServiceError,
    NotFoundError,
    RateLimitError,
    RetryExhaustedError,
    StorageError,
    TimeoutError,
    UnauthorizedError,
    ValidationError,
    WebhookError,
)

# File utilities
from .file_utils import FileUtils, ImageValidator

# Helper functions
from .helpers import (
    CacheKeyGenerator,
    Timer,
    batch_process,
    calculate_file_hash,
    calculate_similarity_score,
    create_api_response,
    create_thumbnail_key,
    extract_image_metadata,
    format_duration,
    format_file_size,
    generate_request_id,
    generate_unique_id,
    mask_sensitive_data,
    parse_user_agent,
    retry_with_backoff,
    sanitize_filename,
    validate_email_format,
    validate_image_base64,
    validate_threshold,
)

# Logger
from .logger import (
    get_logger,
    log_function_call,
    setup_logger,
)
from .security import SecurityUtils

# Validators and security
from .validators import Validators

# Aliases for backward compatibility
AppException = BaseAppException

# Structured logging (must be after other imports to avoid circular imports)
from .structured_logging import (
    AuditLogger,
    LogContext,
    LogEntry,
    LoggerFactory,
    LogLevel,
    create_log,
    get_logger,
    log_error,
    log_info,
    log_with_context,
    redact_sensitive_data,
)

# Constants and validators are available but not imported by default
# to avoid circular imports and keep imports explicit
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
    "log_function_call",
    # File utilities
    "FileUtils",
    "ImageValidator",
    # Validators and security
    "Validators",
    "SecurityUtils",
    # Helper functions
    "generate_unique_id",
    "generate_request_id",
    "format_file_size",
    "calculate_similarity_score",
    "validate_image_base64",
    "extract_image_metadata",
    "sanitize_filename",
    "create_thumbnail_key",
    "validate_email_format",
    "mask_sensitive_data",
    "calculate_file_hash",
    "parse_user_agent",
    "create_api_response",
    "validate_threshold",
    "format_duration",
    "batch_process",
    "retry_with_backoff",
    "Timer",
    "CacheKeyGenerator",
    # Structured logging
    "LogLevel",
    "redact_sensitive_data",
    "LogContext",
    "LoggerFactory",
    "get_logger",
    "log_with_context",
    "log_info",
    "log_error",
    "AuditLogger",
    "LogEntry",
    "create_log",
]
