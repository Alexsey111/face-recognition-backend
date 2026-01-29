"""
Middleware для ограничения скорости запросов (Rate Limiting).
Защита от DDoS атак и контроль нагрузки на API.
"""

import hashlib
import time
from collections import defaultdict, deque
from typing import Any, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware для ограничения скорости запросов.
    """

    def __init__(self, app):
        super().__init__(app)
        self.requests_per_minute = settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        self.local_cache: Dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if self._should_skip_rate_limit(request):
            return await call_next(request)

        client_id = await self._get_client_id(request)
        is_allowed, limit_info = await self._check_rate_limit_local(
            client_id, int(time.time()), int(time.time()) - 60
        )

        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
                headers={
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        response = await call_next(request)
        return response

    async def _get_client_id(self, request: Request) -> str:
        """
        Получение уникального идентификатора клиента.

        Args:
            request: HTTP запрос

        Returns:
            str: Идентификатор клиента
        """
        # Приоритет идентификаторов:
        # 1. API ключ в заголовке
        # 2. JWT токен (user_id)
        # 3. IP адрес

        # Проверяем API ключ
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{hashlib.md5(api_key.encode()).hexdigest()[:16]}"

        # Проверяем JWT токен
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Получаем user_id из токена (если есть)
            user_id = getattr(request.state, "user_id", None)
            if user_id:
                return f"user:{user_id}"

        # Используем IP адрес
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"

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
            # Берем первый IP из списка
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Используем IP из соединения
        if request.client:
            return request.client.host

        return "unknown"

    def _should_skip_rate_limit(self, request: Request) -> bool:
        """
        Проверка, нужно ли пропустить rate limiting для endpoint.

        Args:
            request: HTTP запрос

        Returns:
            bool: True если rate limiting не нужен
        """
        # Список endpoint-ов, которые не ограничиваются
        skip_paths = [
            "/health",
            "/status",
            "/ready",
            "/live",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
            "/api/v1/status",
            "/api/v1/ready",
            "/api/v1/live",
            "/api/v1/metrics",
        ]

        path = request.url.path

        # Проверяем точные совпадения и префиксы
        for skip_path in skip_paths:
            if path == skip_path or path.startswith(skip_path + "/"):
                return True

        # Health check endpoints с GET методом
        if request.method == "GET" and (
            path.startswith("/api/v1/health")
            or path.startswith("/api/v1/status")
            or path.startswith("/api/v1/ready")
            or path.startswith("/api/v1/live")
        ):
            return True

        return False

    async def _check_rate_limit_local(
        self, client_id: str, current_time: int, window_start: int
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Проверка лимитов с использованием локального кэша.

        Args:
            client_id: Идентификатор клиента
            current_time: Текущее время
            window_start: Начало временного окна

        Returns:
            tuple[bool, Dict[str, Any]]: (разрешен ли запрос, информация о лимитах)
        """
        now_deque = self.local_cache[client_id]

        # Удаляем старые записи
        while now_deque and now_deque[0] < window_start:
            now_deque.popleft()

        current_requests = len(now_deque)

        # Добавляем текущий запрос
        now_deque.append(current_time)

        # Ограничиваем размер deque для предотвращения утечек памяти
        if len(now_deque) > 1000:
            # Оставляем только последние 1000 записей
            self.local_cache[client_id] = deque(list(now_deque)[-1000:], maxlen=1000)

        # Проверяем лимиты
        is_allowed = current_requests < self.requests_per_minute

        limit_info = {
            "limit": self.requests_per_minute,
            "remaining": max(0, self.requests_per_minute - current_requests - 1),
            "reset_time": current_time + 60,
            "window_start": window_start,
            "current_requests": current_requests + 1,
        }

        return is_allowed, limit_info
