"""
Утилиты и вспомогательные функции.
Общие утилиты для всего приложения.
"""

# Exceptions
from .exceptions import (
    BaseAppException,
    ValidationError,
    DatabaseError,
    NotFoundError,
    UnauthorizedError,
    ForbiddenError,
    ConflictError,
    StorageError,
    CacheError,
    MLServiceError,
    WebhookError,
    RateLimitError,
    ExternalServiceError,
    BusinessLogicError,
    TimeoutError,
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

# File utilities 
from .file_utils import FileUtils, ImageValidator

# Validators and security 
from .validators import Validators
from .security import SecurityUtils

# Helper functions
from .helpers import (
    generate_unique_id,
    generate_request_id,
    format_file_size,
    calculate_similarity_score,
    validate_image_base64,
    extract_image_metadata,
    sanitize_filename,
    create_thumbnail_key,
    validate_email_format,
    mask_sensitive_data,
    calculate_file_hash,
    parse_user_agent,
    create_api_response,
    validate_threshold,
    format_duration,
    batch_process,
    retry_with_backoff,
    Timer,
    CacheKeyGenerator,
)

# Aliases for backward compatibility
AppException = BaseAppException

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
    "LoggerMixin",
    "StructuredLogger",
    "structured_logger",
    "log_function_call",
    "configure_logging_for_environment",
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
]
