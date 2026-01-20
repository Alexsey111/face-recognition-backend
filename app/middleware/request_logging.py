from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import time
import uuid
from typing import Callable
from ..config import settings
from ..utils.logger import get_logger, audit_event

logger = get_logger(__name__)


def _get_request_id(request: Request) -> str:
    header = settings.REQUEST_ID_HEADER
    req_id = request.headers.get(header)
    if not req_id and settings.GENERATE_REQUEST_ID_IF_NOT_PRESENT:
        req_id = str(uuid.uuid4())
    return req_id


def _redact_body(body: dict) -> dict:
    # reuse logging redaction lightly; keep simple here
    sensitive = {"password", "passwd", "embeddings", "image", "file"}
    return {k: ("[REDACTED]" if k.lower() in sensitive else v) for k, v in body.items()}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = _get_request_id(request)
        request.state.request_id = request_id

        start = time.time()

        try:
            # read small request bodies only when enabled
            body = None
            if settings.AUDIT_LOG_INCLUDE_REQUEST_BODY:
                try:
                    body = await request.json()
                    if isinstance(body, dict):
                        body = _redact_body(body)
                except Exception:
                    body = None

            logger.info(
                "request_received",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else None,
                    "body": body,
                },
            )

            response = await call_next(request)

            duration_ms = int((time.time() - start) * 1000)
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client": request.client.host if request.client else None,
                },
            )

            return response

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            logger.exception(
                "request_error",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            # record audit event for failures
            try:
                audit_event(
                    action="request_error",
                    actor_id=None,
                    resource_type="http_request",
                    resource_id=request_id,
                    details={"method": request.method, "path": request.url.path},
                    ip_address=request.client.host if request.client else None,
                    success=False,
                    error_message=str(e),
                )
            except Exception:
                pass
            raise
"""
Request Logging Middleware.
Middleware для логирования всех HTTP запросов:
- Request ID tracing через все логи
- Логирование времени выполнения
- Redaction чувствительных данных
- Интеграция с Prometheus метриками
"""

import time
import uuid
import json
import logging
from typing import Optional, Callable, Dict, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from ..utils.logger import (
    get_logger,
    log_with_context,
    LogContext,
    _redact,
)
from ..utils.helpers import generate_request_id
from ..config import settings
from .metrics import (
    http_requests_total,
    http_request_duration_seconds,
    http_request_size_bytes,
    http_response_size_bytes,
    record_error,
)


logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для логирования HTTP запросов.
    
    Функции:
    - Генерация или извлечение request_id
    - Логирование начала/конца запроса
    - Измерение времени выполнения
    - Логирование метода, пути, статуса
    - Redaction чувствительных данных
    - Интеграция с контекстным логированием
    """
    
    # Пути, которые не нужно логировать
    EXCLUDED_PATHS = {
        "/health",
        "/health/",
        "/metrics",
        "/metrics/",
        "/favicon.ico",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    def __init__(
        self,
        app,
        exclude_paths: Optional[set] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or self.EXCLUDED_PATHS
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Обработка запроса.
        
        Args:
            request: HTTP запрос
            call_next: Следующий middleware/handler
            
        Returns:
            HTTP ответ
        """
        # Проверяем, нужно ли логировать этот путь
        if self._should_skip_logging(request):
            return await call_next(request)
        
        # Генерируем или извлекаем request_id
        request_id = self._get_request_id(request)
        
        # Получаем информацию о клиенте
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")
        
        # Время начала запроса
        start_time = time.perf_counter()
        start_timestamp = time.time()
        
        # Создаём контекст для логирования
        with LogContext(
            request_id=request_id,
            user_id=None,  # Будет установлен из токена, если есть
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
            }
        ):
            try:
                # Логируем начало запроса
                self._log_request_start(request, request_id, client_ip)
                
                # Выполняем запрос
                response = await call_next(request)
                
                # Вычисляем время выполнения
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Логируем завершение запроса
                self._log_request_end(
                    request, response, request_id,
                    duration_ms, client_ip, start_timestamp
                )
                
                # Обновляем статус в контексте
                return response
                
            except Exception as exc:
                # Вычисляем время выполнения
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Логируем ошибку
                self._log_request_error(
                    request, exc, request_id,
                    duration_ms, client_ip
                )
                
                # Пробрасываем исключение дальше
                raise
    
    def _should_skip_logging(self, request: Request) -> bool:
        """Проверка, нужно ли пропустить логирование."""
        path = request.url.path
        return any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    def _get_request_id(self, request: Request) -> str:
        """
        Получение request_id из заголовка или генерация нового.
        
        Priority:
        1. X-Request-ID header
        2. X-Correlation-ID header
        3. Generated request_id
        """
        # Пробуем получить из заголовка
        request_id = (
            request.headers.get(settings.REQUEST_ID_HEADER) or
            request.headers.get("X-Correlation-ID") or
            request.headers.get("X-Request-ID")
        )
        
        if request_id and settings.GENERATE_REQUEST_ID_IF_NOT_PRESENT:
            return request_id
        
        # Генерируем новый
        return generate_request_id()
    
    def _get_client_ip(self, request: Request) -> str:
        """Получение IP адреса клиента."""
        # Проверяем прокси заголовки
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # X-Forwarded-For: client, proxy1, proxy2
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback на direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _log_request_start(
        self,
        request: Request,
        request_id: str,
        client_ip: str
    ) -> None:
        """Логирование начала запроса."""
        
        # Формируем базовые поля
        log_fields = {
            "event": "request_start",
            "method": request.method,
            "path": request.url.path,
            "query_string": request.url.query,
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }
        
        # Redact sensitive data
        log_fields = _redact(log_fields)
        
        # Логируем
        log_with_context(
            logger.info,  # ✅ Изменить
            f"{request.method} {request.url.path}",
            request_id=request_id,
            ip_address=client_ip,
            extra=log_fields
        )
        
    
    def _log_request_end(
        self,
        request: Request,
        response: Response,
        request_id: str,
        duration_ms: float,
        client_ip: str,
        start_timestamp: float
    ) -> None:
        """Логирование завершения запроса."""
        # Нормализуем путь для метрик
        normalized_path = self._normalize_path(request.url.path)
        
        # Получаем статус код
        status_code = response.status_code if hasattr(response, "status_code") else 200
        
        # Обновляем Prometheus метрики
        http_requests_total.labels(
            method=request.method,
            endpoint=normalized_path,
            status_code=str(status_code)
        ).inc()
        
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=normalized_path
        ).observe(duration_ms / 1000)  # Конвертируем в секунды
        
        # Формируем поля лога
        log_fields = {
            "event": "request_end",
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": client_ip,
        }
        
        # Добавляем заголовки ответа (без sensitive)
        if hasattr(response, "headers") and response.headers:
            response_headers = dict(response.headers)
            # Убираем sensitive headers
            sensitive_headers = {"authorization", "cookie", "set-cookie"}
            log_fields["response_headers"] = {
                k: v for k, v in response_headers.items()
                if k.lower() not in sensitive_headers
            }
        
        # Логируем
        level = self._get_log_level_for_status(status_code)

        # Получаем метод логгера по уровню
        if level == logging.DEBUG:
            log_method = logger.debug
        elif level == logging.INFO:
            log_method = logger.info
        elif level == logging.WARNING:
            log_method = logger.warning
        else:
            log_method = logger.error

        log_with_context(
            log_method,  # ✅ ИСПРАВЛЕНО
            f"{request.method} {request.url.path} -> {status_code}",
            request_id=request_id,
            ip_address=client_ip,
            duration_ms=duration_ms,
            extra=log_fields
        )

    
    def _log_request_error(
        self,
        request: Request,
        error: Exception,
        request_id: str,
        duration_ms: float,
        client_ip: str
    ) -> None:
        """Логирование ошибки запроса."""
        # Записываем метрику ошибки
        normalized_path = self._normalize_path(request.url.path)
        record_error(
            error_type=error.__class__.__name__,
            endpoint=normalized_path
        )
        
        # Формируем поля лога
        log_fields = {
            "event": "request_error",
            "method": request.method,
            "path": request.url.path,
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "duration_ms": round(duration_ms, 2),
            "client_ip": client_ip,
        }
        
        # Логируем ошибку
        log_with_context(
            logger.error,  # ✅ Изменить
            f"{request.method} {request.url.path} - Error: {str(error)}",
            request_id=request_id,
            ip_address=client_ip,
            duration_ms=duration_ms,
            error=error,
            extra=log_fields
        )


    
    def _get_log_level_for_status(self, status_code: int) -> int:
        """Определение уровня логирования по status code."""
        if status_code < 400:
            return logging.DEBUG
        elif status_code < 500:
            return logging.WARNING
        else:
            return logging.ERROR
    
    def _normalize_path(self, path: str) -> str:
        """
        Нормализация пути для метрик.
        Заменяет dynamic IDs на placeholder.
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
# Request ID Middleware
# ============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware для добавления X-Request-ID заголовка.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Добавление request_id к запросу и ответу.
        """
        # Генерируем или получаем request_id
        request_id = (
            request.headers.get(settings.REQUEST_ID_HEADER) or
            request.headers.get("X-Correlation-ID") or
            generate_request_id()
        )
        
        # Добавляем request_id в state для использования в路由
        request.state.request_id = request_id
        
        # Выполняем запрос
        response = await call_next(request)
        
        # Добавляем request_id в заголовки ответа
        response.headers[settings.REQUEST_ID_HEADER] = request_id
        
        return response


# ============================================================================
# Context Logger Middleware
# ============================================================================

class ContextLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware для установки контекста логирования.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Установка контекста и выполнение запроса.
        """
        # Получаем request_id из state
        request_id = getattr(request.state, "request_id", None)
        
        # Пытаемся получить user_id из токена (если есть)
        user_id = None
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Здесь можно декодировать токен и получить user_id
            # Пока оставляем None - будет заполнено в auth middleware
            pass
        
        # Создаём контекст
        with LogContext(
            request_id=request_id,
            user_id=user_id,
            extra={
                "method": request.method,
                "path": request.url.path,
            }
        ):
            try:
                response = await call_next(request)
                return response
            finally:
                # Контекст автоматически восстанавливается при выходе
                pass


# ============================================================================
# Request Timing Decorator
# ============================================================================

def timed_operation(operation_name: Optional[str] = None):
    """
    Декоратор для замера времени выполнения операции.
    
    Args:
        operation_name: Имя операции (по умолчанию - имя функции)
    """
    def decorator(func):
        import functools
        import asyncio
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Логируем
                log_with_context(
                    logger.debug,
                    f"Operation {operation_name or func.__name__} completed",
                    duration_ms=duration_ms,
                    extra={"operation": operation_name or func.__name__}
                )
                
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                log_with_context(
                    logger.error,
                    f"Operation {operation_name or func.__name__} failed",
                    duration_ms=duration_ms,
                    error=e,
                    extra={"operation": operation_name or func.__name__}
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                log_with_context(
                    logger.debug,  # ✅ ИСПРАВЛЕНО
                    f"Operation {operation_name or func.__name__} completed",
                    duration_ms=duration_ms,
                    extra={"operation": operation_name or func.__name__}
                )
                
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                log_with_context(
                    logger.error,  # ✅ ИСПРАВЛЕНО
                    f"Operation {operation_name or func.__name__} failed",
                    duration_ms=duration_ms,
                    error=e,
                    extra={"operation": operation_name or func.__name__}
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

