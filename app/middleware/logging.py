"""Middleware для логирования запросов.
Запись информации о HTTP запросах и ответах без биометрических данных.
"""

from typing import Dict, Any
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для логирования HTTP запросов и ответов.
    """

    def __init__(self, app):
        super().__init__(app)
        # self.db_service = DatabaseService()  # TODO Phase 3

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        if self._should_skip_logging(request):
            return await call_next(request)

        start_time = time.time()

        # Логируем запрос
        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={"request_id": request_id, "client_ip": self._get_client_ip(request)},
        )

        response = await call_next(request)

        process_time = time.time() - start_time

        # Логируем ответ
        response_info = (
            f"← {request.method} {request.url.path} - "
            f"{response.status_code} ({process_time:.3f}s)"
        )
        logger.info(
            response_info,
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time,
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response

    def _should_skip_logging(self, request: Request) -> bool:
        """
        Проверка, нужно ли пропустить логирование запроса.

        Args:
            request: HTTP запрос

        Returns:
            bool: True если логирование не нужно
        """
        path = request.url.path

        # Пути, которые не нужно логировать (избегаем спама)
        skip_paths = [
            "/metrics",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/api/v1/metrics",
            "/api/v1/health",
            "/api/v1/status",
            "/api/v1/ready",
            "/api/v1/live",
        ]

        # Пропускаем пути из списка
        for skip_path in skip_paths:
            if path.startswith(skip_path):
                return True

        return False

    def _get_client_ip(self, request: Request) -> str:
        """
        Получение IP адреса клиента.

        Args:
            request: HTTP запрос

        Returns:
            str: IP адрес
        """
        # Проверяем различные заголовки для получения реального IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Используем IP из соединения
        if request.client:
            return request.client.host

        return "unknown"


class RequestLogger:
    """
    Утилита для логирования специфичных событий.
    """

    @staticmethod
    def log_user_action(user_id: str, action: str, details: Dict[str, Any] = None):
        """
        Логирование действий пользователя.

        Args:
            user_id: ID пользователя
            action: Действие
            details: Дополнительные детали
        """
        logger.info(
            f"User action: {action}",
            extra={
                "event_type": "user_action",
                "user_id": user_id,
                "action": action,
                "details": details or {},
            },
        )

    @staticmethod
    def log_verification_request(
        user_id: str, session_id: str, verification_type: str, success: bool
    ):
        """
        Логирование запроса верификации.

        Args:
            user_id: ID пользователя
            session_id: ID сессии
            verification_type: Тип верификации
            success: Результат
        """
        result_text = "success" if success else "failure"
        logger.info(
            f"Verification request: {verification_type} - {result_text}",
            extra={
                "event_type": "verification_request",
                "user_id": user_id,
                "session_id": session_id,
                "verification_type": verification_type,
                "success": success,
            },
        )

    @staticmethod
    def log_security_event(event_type: str, details: Dict[str, Any]):
        """
        Логирование событий безопасности.

        Args:
            event_type: Тип события безопасности
            details: Детали события
        """
        logger.warning(
            f"Security event: {event_type}",
            extra={
                "event_type": "security_event",
                "security_event_type": event_type,
                "details": details,
            },
        )

    @staticmethod
    def log_performance_metric(
        metric_name: str, value: float, tags: Dict[str, str] = None
    ):
        """
        Логирование метрик производительности.

        Args:
            metric_name: Название метрики
            value: Значение
            tags: Теги для метрики
        """
        logger.info(
            f"Performance metric: {metric_name} = {value}",
            extra={
                "event_type": "performance_metric",
                "metric_name": metric_name,
                "metric_value": value,
                "metric_tags": tags or {},
            },
        )


# Создаем глобальный экземпляр логгера
request_logger = RequestLogger()
