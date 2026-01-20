"""
Logging Middleware (production-safe).

Логирует HTTP запросы и ответы без биометрических данных.
Поддерживает:
- correlation_id (X-Request-ID)
- structured logging
- proxy-aware client IP detection
- безопасное логирование ошибок
"""

import time
import uuid
from typing import Dict, Any, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logger import get_logger, LogContext

logger = get_logger(__name__)


# =============================================================================
# Logging Middleware
# =============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для логирования HTTP запросов и ответов.
    """

    SKIP_PATH_PREFIXES = (
        "/metrics",
        "/health",
        "/ready",
        "/live",
        "/status",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    )

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Пропускаем служебные endpoints
        if self._should_skip_logging(request.url.path):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        start_time = time.perf_counter()

        user_id: Optional[str] = None
        if hasattr(request.state, "user") and isinstance(request.state.user, dict):
            user_id = request.state.user.get("user_id")

        async with LogContext(request_id=request_id, user_id=user_id):
            self._log_request(request)

            response = None
            try:
                response = await call_next(request)
                return response

            except Exception as exc:
                process_time = time.perf_counter() - start_time
                self._log_error(request, exc, process_time)
                raise

            finally:
                process_time = time.perf_counter() - start_time
                if response is not None:
                    self._log_response(request, response.status_code, process_time)
                    response.headers["X-Request-ID"] = request_id

    # =============================================================================
    # Internal helpers
    # =============================================================================

    def _should_skip_logging(self, path: str) -> bool:
        return path.startswith(self.SKIP_PATH_PREFIXES)

    def _log_request(self, request: Request) -> None:
        logger.info(
            "HTTP request received",
            extra={
                "event_type": "http_request",
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
            },
        )

    def _log_response(
        self,
        request: Request,
        status_code: int,
        process_time: float,
    ) -> None:
        logger.info(
            "HTTP response sent",
            extra={
                "event_type": "http_response",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "process_time_ms": round(process_time * 1000, 2),
            },
        )

    def _log_error(
        self,
        request: Request,
        exc: Exception,
        process_time: float,
    ) -> None:
        logger.error(
            "HTTP request failed",
            extra={
                "event_type": "http_error",
                "method": request.method,
                "path": request.url.path,
                "error_type": exc.__class__.__name__,
                "process_time_ms": round(process_time * 1000, 2),
            },
            exc_info=True,
        )

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Получение реального IP клиента с учетом прокси.
        """
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"


# =============================================================================
# Structured Event Logger (Business / Security / Performance)
# =============================================================================

class RequestLogger:
    """
    Утилита для логирования бизнес-, security- и performance-событий.
    """

    @staticmethod
    def log_user_action(
        action: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger.info(
            "User action",
            extra={
                "event_type": "user_action",
                "action": action,
                "user_id": user_id,
                "details": details or {},
            },
        )

    @staticmethod
    def log_verification(
        verification_type: str,
        result: str,
        session_id: Optional[str] = None,
    ) -> None:
        logger.info(
            "Verification event",
            extra={
                "event_type": "verification",
                "verification_type": verification_type,
                "result": result,
                "session_id": session_id,
            },
        )

    @staticmethod
    def log_security_event(
        event: str,
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger.warning(
            "Security event detected",
            extra={
                "event_type": "security_event",
                "security_event": event,
                "severity": severity,
                "details": details or {},
            },
        )

    @staticmethod
    def log_performance_metric(
        metric_name: str,
        value: float,
        unit: str = "ms",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        logger.info(
            "Performance metric",
            extra={
                "event_type": "performance_metric",
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                "tags": tags or {},
            },
        )


# Глобальный экземпляр
request_logger = RequestLogger()
