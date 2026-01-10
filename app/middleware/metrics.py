"""
Prometheus Metrics Middleware.
Предоставляет метрики для мониторинга приложения:
- HTTP request metrics (count, duration, size)
- Business metrics (verifications, uploads, liveness checks)
- System metrics (database connections, cache hits/misses)
- Error metrics
"""

import time
import functools
from typing import Optional, Callable, Any
from datetime import datetime, timezone
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, REGISTRY, CollectorRegistry
except Exception:
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

    Counter = _NoopMetric
    Histogram = _NoopMetric
    Gauge = _NoopMetric
    Info = _NoopMetric
    REGISTRY = None
    CollectorRegistry = None

from ..config import settings


# ============================================================================
# Registry
# ============================================================================

# Используем отдельный реестр для метрик приложения
app_registry = REGISTRY


# ============================================================================
# HTTP Request Metrics
# ============================================================================

# Total HTTP requests
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=app_registry
)

# Request duration in seconds
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0],
    registry=app_registry
)

# Request size in bytes
http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=[100, 1000, 10000, 100000, 500000, 1000000, 5000000, 10000000],
    registry=app_registry
)

# Response size in bytes
http_response_size_bytes = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=[100, 1000, 10000, 100000, 500000, 1000000, 5000000, 10000000],
    registry=app_registry
)


# ============================================================================
# Business Metrics
# ============================================================================

# Total verification sessions
verifications_total = Counter(
    "verifications_total",
    "Total number of verification sessions started",
    ["user_id", "session_type"],
    registry=app_registry
)

# Successful verifications
verifications_success = Counter(
    "verifications_success",
    "Number of successful verifications (match found)",
    ["user_id", "session_type"],
    registry=app_registry
)

# Failed verifications
verifications_failed = Counter(
    "verifications_failed",
    "Number of failed verifications (no match)",
    ["user_id", "session_type", "reason"],
    registry=app_registry
)

# Liveness checks
liveness_checks_total = Counter(
    "liveness_checks_total",
    "Total number of liveness checks",
    ["user_id"],
    registry=app_registry
)

liveness_checks_passed = Counter(
    "liveness_checks_passed",
    "Number of passed liveness checks",
    ["user_id"],
    registry=app_registry
)

liveness_checks_failed = Counter(
    "liveness_checks_failed",
    "Number of failed liveness checks",
    ["user_id", "reason"],
    registry=app_registry
)

# File uploads
uploads_total = Counter(
    "uploads_total",
    "Total number of file uploads",
    ["user_id", "file_type"],
    registry=app_registry
)

uploads_success = Counter(
    "uploads_success",
    "Number of successful file uploads",
    ["user_id", "file_type"],
    registry=app_registry
)

uploads_failed = Counter(
    "uploads_failed",
    "Number of failed file uploads",
    ["user_id", "file_type", "reason"],
    registry=app_registry
)

# Reference images
references_created = Counter(
    "references_created",
    "Total number of reference images created",
    ["user_id"],
    registry=app_registry
)

references_deleted = Counter(
    "references_deleted",
    "Total number of reference images deleted",
    ["user_id", "deleted_by"],
    registry=app_registry
)


# ============================================================================
# Authentication Metrics
# ============================================================================

auth_login_total = Counter(
    "auth_login_total",
    "Total number of login attempts",
    ["status", "method"],
    registry=app_registry
)

auth_logout_total = Counter(
    "auth_logout_total",
    "Total number of logout actions",
    ["user_id"],
    registry=app_registry
)

auth_token_issued = Counter(
    "auth_token_issued",
    "Total number of tokens issued",
    ["token_type"],
    registry=app_registry
)

auth_token_validated = Counter(
    "auth_token_validated",
    "Total number of token validations",
    ["status"],
    registry=app_registry
)


# ============================================================================
# System Metrics
# ============================================================================

# Database connections
database_connections_active = Gauge(
    "database_connections_active",
    "Number of active database connections",
    registry=app_registry
)

database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "Database query duration in seconds",
    ["query_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=app_registry
)

# Cache metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
    registry=app_registry
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
    registry=app_registry
)

cache_size = Gauge(
    "cache_size",
    "Current cache size",
    ["cache_type"],
    registry=app_registry
)

# Error metrics
errors_total = Counter(
    "errors_total",
    "Total number of errors by type",
    ["error_type", "endpoint"],
    registry=app_registry
)

# Active sessions
active_sessions = Gauge(
    "active_sessions",
    "Number of active verification sessions",
    registry=app_registry
)

# Queue metrics
queue_size = Gauge(
    "queue_size",
    "Current queue size",
    ["queue_name"],
    registry=app_registry
)

# Processing metrics
processing_time_seconds = Histogram(
    "processing_time_seconds",
    "Time spent processing requests",
    ["operation_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=app_registry
)


# ============================================================================
# System Info Metrics
# ============================================================================

app_info = Info(
    "app",
    "Application information",
    registry=app_registry
)


def update_app_info():
    """Обновление информации о приложении."""
    app_info.info({
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "debug": str(settings.DEBUG),
        "app_name": settings.APP_NAME,
    })


# ============================================================================
# Metrics Middleware
# ============================================================================

class MetricsMiddleware:
    """
    Middleware для сбора HTTP метрик.
    Автоматически собирает:
    - Количество запросов
    - Время выполнения
    - Размер запроса/ответа
    """
    
    def __init__(self, app):  # ✅ ДОБАВИЛИ app
        self._app = app  # ✅ СОХРАНЯЕМ app
        self.requests_total = http_requests_total
        self.request_duration = http_request_duration_seconds
        self.request_size = http_request_size_bytes
        self.response_size = http_response_size_bytes
    
    async def __call__(self, scope, receive, send):
        """
        Обработка HTTP запроса.
        
        Args:
            scope: ASGI scope
            receive: Receive callable
            send: Send callable
        """
        if scope["type"] != "http":
            await self._app(scope, receive, send)  # ✅ ИСПОЛЬЗУЕМ self._app
            return
        
        # Извлекаем информацию о запросе
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        
        # Нормализуем путь для метрик (убираем dynamic IDs)
        normalized_path = self._normalize_path(path)
        
        # Время начала запроса
        start_time = time.perf_counter()
        
        # Статус код (будет установлен позже)
        status_code = 200
        
        # Перехватываем send для измерения размера ответа
        response_size = 0
        
        async def traced_send(message):
            nonlocal response_size, status_code
            
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
                
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                response_size += len(body)
            
            await send(message)
        
        try:
            # Продолжаем обработку запроса
            await self._app(scope, receive, traced_send)
            
        except Exception as exc:
            # Логируем ошибку
            status_code = 500
            errors_total.labels(
                error_type=exc.__class__.__name__,
                endpoint=normalized_path
            ).inc()
            raise
        
        finally:
            # Вычисляем длительность
            duration = time.perf_counter() - start_time
            
            # Обновляем метрики
            self.requests_total.labels(
                method=method,
                endpoint=normalized_path,
                status_code=status_code
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=normalized_path
            ).observe(duration)
            
            if response_size > 0:
                self.response_size.labels(
                    method=method,
                    endpoint=normalized_path
                ).observe(response_size)
    
    def _normalize_path(self, path: str) -> str:
        """
        Нормализация пути для метрик.
        Заменяет dynamic IDs на placeholder.
        
        /api/v1/users/123 -> /api/v1/users/{id}
        /api/v1/references/uuid-123 -> /api/v1/references/{id}
        """
        import re
        
        # Паттерны для замены
        patterns = [
            (r'/\d+', '/{id}'),  # Числовые ID
            (r'/[0-9a-fA-F-]{36}', '/{uuid}'),  # UUID
            (r'/[a-f0-9]{16,}', '/{hash}'),  # Hashes
        ]
        
        normalized = path
        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized

# ============================================================================
# Business Metrics Functions
# ============================================================================

def record_verification_start(user_id: str, session_type: str = "standard"):
    """Запись начала верификации."""
    verifications_total.labels(user_id=user_id, session_type=session_type).inc()


def record_verification_success(user_id: str, session_type: str = "standard"):
    """Запись успешной верификации."""
    verifications_success.labels(user_id=user_id, session_type=session_type).inc()


def record_verification_failed(user_id: str, session_type: str = "standard", reason: str = "no_match"):
    """Запись неуспешной верификации."""
    verifications_failed.labels(
        user_id=user_id,
        session_type=session_type,
        reason=reason
    ).inc()


def record_liveness_check(user_id: str, passed: bool, reason: Optional[str] = None):
    """Запись проверки liveness."""
    liveness_checks_total.labels(user_id=user_id).inc()
    
    if passed:
        liveness_checks_passed.labels(user_id=user_id).inc()
    else:
        liveness_checks_failed.labels(
            user_id=user_id,
            reason=reason or "unknown"
        ).inc()


def record_upload(user_id: str, file_type: str, success: bool, reason: Optional[str] = None):
    """Запись загрузки файла."""
    uploads_total.labels(user_id=user_id, file_type=file_type).inc()
    
    if success:
        uploads_success.labels(user_id=user_id, file_type=file_type).inc()
    else:
        uploads_failed.labels(
            user_id=user_id,
            file_type=file_type,
            reason=reason or "unknown"
        ).inc()


def record_reference_created(user_id: str):
    """Запись создания reference."""
    references_created.labels(user_id=user_id).inc()


def record_reference_deleted(user_id: str, deleted_by: str = "user"):
    """Запись удаления reference."""
    references_deleted.labels(user_id=user_id, deleted_by=deleted_by).inc()


def record_auth_attempt(status: str = "success", method: str = "password"):
    """Запись попытки аутентификации."""
    auth_login_total.labels(status=status, method=method).inc()


def record_logout(user_id: str):
    """Запись выхода из системы."""
    auth_logout_total.labels(user_id=user_id).inc()


def record_token_issued(token_type: str = "access"):
    """Запись выдачи токена."""
    auth_token_issued.labels(token_type=token_type).inc()


def record_token_validation(status: str = "valid"):
    """Запись валидации токена."""
    auth_token_validated.labels(status=status).inc()


def record_error(error_type: str, endpoint: str = "unknown"):
    """Запись ошибки."""
    errors_total.labels(error_type=error_type, endpoint=endpoint).inc()


# ============================================================================
# Database Metrics
# ============================================================================

@contextmanager
def track_db_query(query_type: str = "unknown"):
    """Контекстный менеджер для отслеживания времени запросов к БД."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        database_query_duration_seconds.labels(query_type=query_type).observe(duration)


def update_db_connections(count: int):
    """Обновление количества активных соединений."""
    database_connections_active.set(count)


def record_cache_hit(cache_type: str = "redis"):
    """Запись cache hit."""
    cache_hits_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str = "redis"):
    """Запись cache miss."""
    cache_misses_total.labels(cache_type=cache_type).inc()


# ============================================================================
# Processing Metrics
# ============================================================================

@contextmanager
def track_processing(operation_type: str = "unknown"):
    """Контекстный менеджер для отслеживания времени обработки."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        processing_time_seconds.labels(operation_type=operation_type).observe(duration)


def update_active_sessions(count: int):
    """Обновление количества активных сессий."""
    active_sessions.set(count)


def update_queue_size(queue_name: str, size: int):
    """Обновление размера очереди."""
    queue_size.labels(queue_name=queue_name).set(size)


def update_cache_size(cache_type: str, size: int):
    """Обновление размера кэша."""
    cache_size.labels(cache_type=cache_type).set(size)


# ============================================================================
# Metrics Decorators
# ============================================================================

def track_function_metrics(operation_name: Optional[str] = None):
    """
    Декоратор для отслеживания метрик функции.
    
    Args:
        operation_name: Имя операции (по умолчанию - имя функции)
    """
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with track_processing(operation_type=op_name):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with track_processing(operation_type=op_name):
                return await func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================================================
# Metrics Endpoint
# ============================================================================

def get_metrics() -> str:
    """
    Получение всех метрик в формате Prometheus.
    
    Returns:
        Строка с метриками в формате Prometheus text format
    """
    from prometheus_client import generate_latest
    
    # Обновляем app_info
    update_app_info()
    
    return generate_latest(app_registry).decode("utf-8")


def get_metrics_content_type() -> str:
    """Получение content type для метрик."""
    return "text/plain; version=0.0.4; charset=utf-8"


# ============================================================================
# Metrics Initialization
# ============================================================================

def initialize_metrics():
    """Инициализация метрик при запуске приложения."""
    update_app_info()
