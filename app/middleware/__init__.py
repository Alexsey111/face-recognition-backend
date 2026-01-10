"""
Middleware компоненты.
Обработчики промежуточного ПО для FastAPI приложения.
"""

from .auth import AuthMiddleware
from .rate_limit import RateLimitMiddleware
from .logging import LoggingMiddleware
from .error_handler import ErrorHandlerMiddleware
from .request_logging import RequestLoggingMiddleware, RequestIDMiddleware, ContextLoggerMiddleware, timed_operation
from .metrics import (
    MetricsMiddleware,
    track_function_metrics,
    track_db_query,
    track_processing,
    get_metrics,
    get_metrics_content_type,
    initialize_metrics,
    # Metrics functions
    record_verification_start,
    record_verification_success,
    record_verification_failed,
    record_liveness_check,
    record_upload,
    record_reference_created,
    record_reference_deleted,
    record_auth_attempt,
    record_logout,
    record_token_issued,
    record_token_validation,
    record_error,
    record_cache_hit,
    record_cache_miss,
    update_db_connections,
    update_active_sessions,
    update_queue_size,
    update_cache_size,
)

__all__ = [
    # Core middleware
    "AuthMiddleware",
    "RateLimitMiddleware",
    "LoggingMiddleware",
    "ErrorHandlerMiddleware",
    "RequestLoggingMiddleware",
    "RequestIDMiddleware",
    "ContextLoggerMiddleware",
    "MetricsMiddleware",
    
    # Metrics
    "get_metrics",
    "get_metrics_content_type",
    "initialize_metrics",
    "track_function_metrics",
    "track_db_query",
    "track_processing",
    
    # Metrics recording functions
    "record_verification_start",
    "record_verification_success",
    "record_verification_failed",
    "record_liveness_check",
    "record_upload",
    "record_reference_created",
    "record_reference_deleted",
    "record_auth_attempt",
    "record_logout",
    "record_token_issued",
    "record_token_validation",
    "record_error",
    "record_cache_hit",
    "record_cache_miss",
    "update_db_connections",
    "update_active_sessions",
    "update_queue_size",
    "update_cache_size",
    
    # Utilities
    "timed_operation",
]
