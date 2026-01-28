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
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
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
# Biometric Security Metrics (152-ФЗ compliance)
# =============================================================================

biometric_data_access_total = Counter(
    "biometric_data_access_total",
    "Total accesses to biometric data",
    ["operation"],  # 'create', 'read', 'update', 'delete'
    registry=app_registry,
)

biometric_data_deleted_total = Counter(
    "biometric_data_deleted_total",
    "Total biometric data deletions",
    ["reason"],  # 'user_request', 'consent_withdrawn', 'retention_expired'
    registry=app_registry,
)

consent_operations_total = Counter(
    "consent_operations_total",
    "Total consent operations",
    ["operation"],  # 'granted', 'withdrawn', 'updated'
    registry=app_registry,
)

encryption_operations_total = Counter(
    "encryption_operations_total",
    "Total encryption/decryption operations",
    ["operation", "status"],  # operation: 'encrypt'/'decrypt', status: 'success'/'failure'
    registry=app_registry,
)

audit_log_entries_total = Counter(
    "audit_log_entries_total",
    "Total audit log entries created",
    ["event_type"],
    registry=app_registry,
)

# =============================================================================
# FAR/FRR Metrics (Biometric Performance)
# =============================================================================

false_accept_total = Counter(
    "false_accept_total",
    "Total false accepts (impostor accepted)",
    ["severity"],  # 'low', 'medium', 'high'
    registry=app_registry,
)

false_reject_total = Counter(
    "false_reject_total",
    "Total false rejects (genuine rejected)",
    ["severity"],
    registry=app_registry,
)

false_accept_rate = Gauge(
    "false_accept_rate",
    "Current False Accept Rate (FAR) percentage",
    registry=app_registry,
)

false_reject_rate = Gauge(
    "false_reject_rate",
    "Current False Reject Rate (FRR) percentage",
    registry=app_registry,
)

equal_error_rate = Gauge(
    "equal_error_rate",
    "Current Equal Error Rate (EER) percentage",
    registry=app_registry,
)

# =============================================================================
# Image Quality Metrics
# =============================================================================

image_quality_score = Histogram(
    "image_quality_score",
    "Image quality score (blur, lighting, etc.)",
    ["quality_metric"],  # 'blur', 'lighting', 'face_size'
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=app_registry,
)

image_preprocessing_failures = Counter(
    "image_preprocessing_failures",
    "Image preprocessing failures",
    ["failure_reason"],  # 'format_unsupported', 'size_too_large', 'corrupted'
    registry=app_registry,
)

# =============================================================================
# Face Detection & Embedding Metrics
# =============================================================================

face_detection_total = Counter(
    "face_detection_total",
    "Total face detection attempts",
    ["faces_detected"],  # '0', '1', 'multiple'
    registry=app_registry,
)

face_detection_duration_seconds = Histogram(
    "face_detection_duration_seconds",
    "Face detection processing time",
    ["model"],  # 'mtcnn', 'retinaface'
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0],
    registry=app_registry,
)

embedding_extraction_duration_seconds = Histogram(
    "embedding_extraction_duration_seconds",
    "Embedding extraction processing time",
    ["model"],  # 'facenet', 'arcface'
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0],
    registry=app_registry,
)

# =============================================================================
# Verification Detailed Metrics
# =============================================================================

verification_duration_seconds = Histogram(
    "verification_duration_seconds",
    "Verification processing time",
    ["method"],  # 'cosine', 'euclidean'
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0],
    registry=app_registry,
)

verification_confidence_score = Histogram(
    "verification_confidence_score",
    "Verification confidence/similarity score",
    ["result"],  # 'match', 'no_match'
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=app_registry,
)

# =============================================================================
# Liveness Detection Detailed Metrics
# =============================================================================

liveness_duration_seconds = Histogram(
    "liveness_duration_seconds",
    "Liveness check processing time",
    ["model"],  # 'minifasnet'
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5],
    registry=app_registry,
)

liveness_confidence_score = Histogram(
    "liveness_confidence_score",
    "Liveness confidence score",
    ["result"],  # 'live', 'spoof'
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=app_registry,
)

spoofing_attacks_detected = Counter(
    "spoofing_attacks_detected",
    "Total spoofing attacks detected",
    ["attack_type"],  # 'print', 'replay', 'mask', 'deepfake', 'unknown'
    registry=app_registry,
)

# =============================================================================
# GPU/System Resources
# =============================================================================

gpu_utilization_percent = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
    registry=app_registry,
)

gpu_memory_used_bytes = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id"],
    registry=app_registry,
)

model_inference_batch_size = Histogram(
    "model_inference_batch_size",
    "Model inference batch size",
    ["model_name"],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
    registry=app_registry,
)

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

    SKIP_PATHS = {"/metrics", "/health", "/ready", "/live", "/status", "/favicon.ico"}

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

            response_size = (
                int(getattr(response, "headers", {}).get("content-length", 0))
                if "response" in locals()
                else 0
            )

            http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint,
            ).observe(response_size)

    @staticmethod
    def _normalize_path(path: str) -> str:
        patterns = [
            (
                r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                "/{uuid}",
            ),
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
# Biometric Security Helpers (152-ФЗ compliance)
# =============================================================================


def record_biometric_access(operation: str):
    """Record biometric data access (152-ФЗ)."""
    biometric_data_access_total.labels(operation=operation).inc()


def record_biometric_deletion(reason: str):
    """Record biometric data deletion (152-ФЗ)."""
    biometric_data_deleted_total.labels(reason=reason).inc()


def record_consent_operation(operation: str):
    """Record consent operation (152-ФЗ)."""
    consent_operations_total.labels(operation=operation).inc()


def record_encryption(operation: str, status: str):
    """Record encryption/decryption operation."""
    encryption_operations_total.labels(operation=operation, status=status).inc()


def record_audit_event(event_type: str):
    """Record audit log entry."""
    audit_log_entries_total.labels(event_type=event_type).inc()


# =============================================================================
# FAR/FRR Metrics Helpers
# =============================================================================


def record_false_accept(severity: str = "medium"):
    """Record false accept event."""
    false_accept_total.labels(severity=severity).inc()


def record_false_reject(severity: str = "medium"):
    """Record false reject event."""
    false_reject_total.labels(severity=severity).inc()


def update_far_frr_rates(far: float, frr: float, eer: float):
    """Update FAR/FRR/EER gauges."""
    false_accept_rate.set(far)
    false_reject_rate.set(frr)
    equal_error_rate.set(eer)


# =============================================================================
# Image Quality Helpers
# =============================================================================


def record_image_quality(quality_metric: str, score: float):
    """Record image quality score."""
    image_quality_score.labels(quality_metric=quality_metric).observe(score)


def record_preprocessing_failure(reason: str):
    """Record preprocessing failure."""
    image_preprocessing_failures.labels(failure_reason=reason).inc()


# =============================================================================
# Face Detection & Embedding Helpers
# =============================================================================


def record_face_detection(faces_detected: str):
    """Record face detection result."""
    face_detection_total.labels(faces_detected=faces_detected).inc()


def record_face_detection_duration(model: str, duration: float):
    """Record face detection duration."""
    face_detection_duration_seconds.labels(model=model).observe(duration)


def record_embedding_extraction(model: str, duration: float):
    """Record embedding extraction duration."""
    embedding_extraction_duration_seconds.labels(model=model).observe(duration)


# =============================================================================
# Verification Detailed Helpers
# =============================================================================


def record_verification_duration(method: str, duration: float):
    """Record verification duration."""
    verification_duration_seconds.labels(method=method).observe(duration)


def record_verification_confidence(result: str, score: float):
    """Record verification confidence score."""
    verification_confidence_score.labels(result=result).observe(score)


# =============================================================================
# Liveness Detection Helpers
# =============================================================================


def record_liveness_duration(model: str, duration: float):
    """Record liveness check duration."""
    liveness_duration_seconds.labels(model=model).observe(duration)


def record_liveness_confidence(result: str, score: float):
    """Record liveness confidence score."""
    liveness_confidence_score.labels(result=result).observe(score)


def record_spoofing_attack(attack_type: str):
    """Record detected spoofing attack."""
    spoofing_attacks_detected.labels(attack_type=attack_type).inc()


# =============================================================================
# GPU/System Helpers
# =============================================================================


def update_gpu_metrics(gpu_id: str, utilization: float, memory_bytes: int):
    """Update GPU metrics."""
    gpu_utilization_percent.labels(gpu_id=gpu_id).set(utilization)
    gpu_memory_used_bytes.labels(gpu_id=gpu_id).set(memory_bytes)


def record_batch_size(model_name: str, size: int):
    """Record model inference batch size."""
    model_inference_batch_size.labels(model_name=model_name).observe(size)


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
    app_info.info(
        {
            "app_name": getattr(settings, "APP_NAME", "face-recognition-service"),
            "version": "1.0.0",
            "environment": getattr(settings, "ENVIRONMENT", "production"),
            "debug": str(getattr(settings, "DEBUG", False)),
        }
    )


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
