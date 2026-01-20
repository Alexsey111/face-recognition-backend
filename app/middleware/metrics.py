"""
Prometheus Metrics Middleware (production-safe).

Собирает:
- HTTP metrics (count, duration, request/response size)
- Business metrics (aggregated, no high-cardinality labels)
- System metrics (DB, cache, queues)
- Error metrics (HTTP vs business)
"""

import time
import re
from typing import Optional
from contextlib import contextmanager
from functools import wraps

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

# =============================================================================
# Prometheus imports with safe fallback
# =============================================================================

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        REGISTRY,
        CollectorRegistry,
    )
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False

    class _NoopMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

        def set(self, *args, **kwargs):
            return None

        def dec(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    Counter = Histogram = Gauge = Info = _NoopMetric
    REGISTRY = None
    CollectorRegistry = None

from ..config import settings

# =============================================================================
# Registry
# =============================================================================

if PROMETHEUS_AVAILABLE and REGISTRY is not None:
    app_registry = REGISTRY
elif PROMETHEUS_AVAILABLE:
    app_registry = CollectorRegistry(auto_describe=True)
else:
    app_registry = None

# =============================================================================
# HTTP Metrics
# =============================================================================

http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=app_registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1,
        0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0
    ],
    registry=app_registry,
)

http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=[100, 1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
    registry=app_registry,
)

http_response_size_bytes = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=[100, 1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
    registry=app_registry,
)

# =============================================================================
# Business Metrics (AGGREGATED, NO user_id)
# =============================================================================

verifications_total = Counter(
    "verifications_total",
    "Total number of verification attempts",
    ["result"],
    registry=app_registry,
)

liveness_checks_total = Counter(
    "liveness_checks_total",
    "Total number of liveness checks",
    ["result"],
    registry=app_registry,
)

uploads_total = Counter(
    "uploads_total",
    "Total number of uploads",
    ["result", "file_type"],
    registry=app_registry,
)

references_total = Counter(
    "references_total",
    "Total number of reference operations",
    ["operation"],
    registry=app_registry,
)

# =============================================================================
# Auth Metrics
# =============================================================================

auth_attempts_total = Counter(
    "auth_attempts_total",
    "Authentication attempts",
    ["result", "method"],
    registry=app_registry,
)

auth_tokens_total = Counter(
    "auth_tokens_total",
    "Issued tokens",
    ["token_type"],
    registry=app_registry,
)

# =============================================================================
# System Metrics
# =============================================================================

database_connections_active = Gauge(
    "database_connections_active",
    "Active database connections",
    registry=app_registry,
)

database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "Database query duration",
    ["query_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    registry=app_registry,
)

cache_hits_total = Counter(
    "cache_hits_total",
    "Cache hits",
    ["cache_type"],
    registry=app_registry,
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Cache misses",
    ["cache_type"],
    registry=app_registry,
)

queue_size = Gauge(
    "queue_size",
    "Queue size",
    ["queue_name"],
    registry=app_registry,
)

processing_time_seconds = Histogram(
    "processing_time_seconds",
    "Processing time",
    ["operation_type"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
    registry=app_registry,
)

# =============================================================================
# Error Metrics
# =============================================================================

http_errors_total = Counter(
    "http_errors_total",
    "HTTP errors by status code",
    ["status_code", "endpoint"],
    registry=app_registry,
)

business_errors_total = Counter(
    "business_errors_total",
    "Business logic errors",
    ["error_type"],
    registry=app_registry,
)

# =============================================================================
# App Info
# =============================================================================

app_info = Info(
    "app",
    "Application information",
    registry=app_registry,
)

# =============================================================================
# Middleware
# =============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """HTTP metrics middleware."""

    SKIP_PATHS = {
        "/metrics", "/health", "/ready", "/live", "/status", "/favicon.ico"
    }

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in self.SKIP_PATHS:
            return await call_next(request)

        method = request.method
        endpoint = self._normalize_path(path)

        start = time.perf_counter()
        status_code = 500

        request_size = int(request.headers.get("content-length", 0))
        http_request_size_bytes.labels(method, endpoint).observe(request_size)

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception:
            http_errors_total.labels(str(status_code), endpoint).inc()
            raise

        finally:
            duration = time.perf_counter() - start

            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            response_size = int(
                getattr(response, "headers", {}).get("content-length", 0)
            ) if "response" in locals() else 0

            http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint,
            ).observe(response_size)

    @staticmethod
    def _normalize_path(path: str) -> str:
        patterns = [
            (r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "/{uuid}"),
            (r"/\d+(?=/|$)", "/{id}"),
            (r"/[a-f0-9]{16,}", "/{hash}"),
        ]
        for pattern, repl in patterns:
            path = re.sub(pattern, repl, path)
        return path

# =============================================================================
# Additional metrics for compatibility
# =============================================================================

active_sessions_gauge = Gauge(
    "active_sessions",
    "Number of active sessions",
    registry=app_registry,
)

queue_size_gauge = Gauge(
    "queue_size_gauge",
    "Queue size gauge",
    ["queue_name"],
    registry=app_registry,
)

cache_size_gauge = Gauge(
    "cache_size_gauge",
    "Cache size",
    ["cache_type"],
    registry=app_registry,
)


# =============================================================================
# Business metric helpers
# =============================================================================

def record_verification(result: str):
    verifications_total.labels(result=result).inc()

def record_liveness(result: str):
    liveness_checks_total.labels(result=result).inc()

def record_upload(result: str, file_type: str):
    uploads_total.labels(result=result, file_type=file_type).inc()

def record_reference(operation: str):
    references_total.labels(operation=operation).inc()

def record_auth(result: str, method: str):
    auth_attempts_total.labels(result=result, method=method).inc()

def record_token(token_type: str):
    auth_tokens_total.labels(token_type=token_type).inc()

def record_business_error(error_type: str):
    business_errors_total.labels(error_type=error_type).inc()

def record_error(error_type: str, endpoint: str):
    """Record an HTTP error for metrics."""
    http_errors_total.labels(status_code=error_type, endpoint=endpoint).inc()


# =============================================================================
# Compatibility functions (alias names)
# =============================================================================

def record_verification_start():
    """Record verification started."""
    verifications_total.labels(result="started").inc()

def record_verification_success():
    """Record successful verification."""
    verifications_total.labels(result="success").inc()

def record_verification_failed():
    """Record failed verification."""
    verifications_total.labels(result="failed").inc()

def record_liveness_check(result: str):
    """Record liveness check result."""
    liveness_checks_total.labels(result=result).inc()

def record_reference_created():
    """Record reference created."""
    references_total.labels(operation="create").inc()

def record_reference_deleted():
    """Record reference deleted."""
    references_total.labels(operation="delete").inc()

def record_auth_attempt(success: bool):
    """Record authentication attempt."""
    result = "success" if success else "failure"
    auth_attempts_total.labels(result=result, method="jwt").inc()

def record_logout():
    """Record user logout."""
    auth_attempts_total.labels(result="logout", method="jwt").inc()

def record_token_issued(token_type: str):
    """Record token issued."""
    auth_tokens_total.labels(token_type=token_type).inc()

def record_token_validation(success: bool):
    """Record token validation."""
    result = "success" if success else "failure"
    auth_attempts_total.labels(result=result, method="token").inc()

def record_cache_hit(cache_type: str):
    """Record cache hit."""
    cache_hits_total.labels(cache_type=cache_type).inc()

def record_cache_miss(cache_type: str):
    """Record cache miss."""
    cache_misses_total.labels(cache_type=cache_type).inc()

def update_db_connections(count: int):
    """Update database connections gauge."""
    database_connections_active.set(count)

def update_active_sessions(count: int):
    """Update active sessions gauge."""
    active_sessions_gauge.set(count)

def update_queue_size(queue_name: str, size: int):
    """Update queue size gauge."""
    queue_size_gauge.labels(queue_name=queue_name).set(size)

def update_cache_size(cache_type: str, size: int):
    """Update cache size gauge."""
    cache_size_gauge.labels(cache_type=cache_type).set(size)


# =============================================================================
# Decorator for function metrics
# =============================================================================

def track_function_metrics(metric_name: str):
    """Decorator to track function execution metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                processing_time_seconds.labels(operation_type=metric_name).observe(
                    time.perf_counter() - start
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                processing_time_seconds.labels(operation_type=metric_name).observe(
                    time.perf_counter() - start
                )

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

# =============================================================================
# Context managers
# =============================================================================

@contextmanager
def track_db_query(query_type: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        database_query_duration_seconds.labels(query_type).observe(
            time.perf_counter() - start
        )

@contextmanager
def track_processing(operation_type: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        processing_time_seconds.labels(operation_type).observe(
            time.perf_counter() - start
        )

# =============================================================================
# Metrics endpoint helpers
# =============================================================================

def initialize_metrics():
    if not PROMETHEUS_AVAILABLE:
        return
    app_info.info({
        "app_name": getattr(settings, "APP_NAME", "face-recognition-service"),
        "version": "1.0.0",
        "environment": getattr(settings, "ENVIRONMENT", "production"),
        "debug": str(getattr(settings, "DEBUG", False)),
    })

def get_metrics() -> str:
    if not PROMETHEUS_AVAILABLE or app_registry is None:
        return ""
    try:
        from prometheus_client import generate_latest
        return generate_latest(app_registry).decode("utf-8")
    except Exception:
        return ""

def get_metrics_content_type() -> str:
    return "text/plain; version=0.0.4; charset=utf-8"
