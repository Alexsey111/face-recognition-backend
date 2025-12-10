"""
Middleware для ограничения скорости запросов (Rate Limiting).
Защита от DDoS атак и контроль нагрузки на API.
"""

from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import time
import hashlib
from collections import defaultdict, deque

from ..config import settings
from ..utils.logger import get_logger
from ..services.cache_service import CacheService

logger = get_logger(__name__)


class RateLimitMiddleware:
    """
    Middleware для ограничения скорости запросов.
    """
    
    def __init__(self):
        self.requests_per_minute = settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        self.burst_limit = settings.RATE_LIMIT_BURST
        self.cache_service = CacheService()
        
        # Локальный кэш для быстрой проверки (fallback)
        self.local_cache: Dict[str, deque] = defaultdict(deque)
        self.local_cache_ttl = 60  # 1 минута
    
    async def __call__(self, request: Request, call_next):
        """
        Обработка запроса с проверкой rate limit.
        
        Args:
            request: HTTP запрос
            call_next: Следующий обработчик
            
        Returns:
            Response: HTTP ответ
        """
        try:
            # Получаем идентификатор клиента
            client_id = await self._get_client_id(request)
            
            # Проверяем, нужно ли пропустить rate limiting
            if self._should_skip_rate_limit(request):
                return await call_next(request)
            
            # Проверяем лимиты
            is_allowed, limit_info = await self._check_rate_limit(client_id, request)
            
            if not is_allowed:
                # Превышен лимит
                return await self._handle_rate_limit_exceeded(request, limit_info)
            
            # Выполняем запрос
            response = await call_next(request)
            
            # Добавляем информацию о лимитах в заголовки ответа
            self._add_rate_limit_headers(response, limit_info)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limit middleware error: {str(e)}")
            # В случае ошибки позволяем запрос пройти
            return await call_next(request)
    
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
            user_id = getattr(request.state, 'user_id', None)
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
            str: IP адрес клиента
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
            "/openapi.json"
        ]
        
        path = request.url.path
        
        # Проверяем точные совпадения и префиксы
        for skip_path in skip_paths:
            if path == skip_path or path.startswith(skip_path + "/"):
                return True
        
        # Health check endpoints с GET методом
        if (request.method == "GET" and 
            (path.startswith("/api/v1/health") or 
             path.startswith("/api/v1/status") or
             path.startswith("/api/v1/ready") or
             path.startswith("/api/v1/live"))):
            return True
        
        return False
    
    async def _check_rate_limit(self, client_id: str, request: Request) -> tuple[bool, Dict[str, Any]]:
        """
        Проверка лимитов скорости для клиента.
        
        Args:
            client_id: Идентификатор клиента
            request: HTTP запрос
            
        Returns:
            tuple[bool, Dict[str, Any]]: (разрешен ли запрос, информация о лимитах)
        """
        current_time = int(time.time())
        window_start = current_time - 60  # 1 минута назад
        
        try:
            # Пытаемся использовать Redis для точного подсчета
            if await self.cache_service.health_check():
                return await self._check_rate_limit_redis(client_id, current_time, window_start)
            else:
                # Fallback на локальный кэш
                return await self._check_rate_limit_local(client_id, current_time, window_start)
                
        except Exception as e:
            logger.warning(f"Rate limit check failed for {client_id}: {str(e)}")
            # В случае ошибки разрешаем запрос
            return True, self._get_default_limit_info()
    
    async def _check_rate_limit_redis(self, client_id: str, current_time: int, window_start: int) -> tuple[bool, Dict[str, Any]]:
        """
        Проверка лимитов с использованием Redis.
        
        Args:
            client_id: Идентификатор клиента
            current_time: Текущее время
            window_start: Начало временного окна
            
        Returns:
            tuple[bool, Dict[str, Any]]: (разрешен ли запрос, информация о лимитах)
        """
        key = f"rate_limit:{client_id}"
        
        # Используем Redis sorted set для sliding window
        pipe = self.cache_service._redis.pipeline()
        
        # Удаляем старые записи
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Получаем количество запросов в текущем окне
        pipe.zcard(key)
        
        # Добавляем текущий запрос
        pipe.zadd(key, {str(current_time): current_time})
        
        # Устанавливаем TTL для ключа
        pipe.expire(key, 120)  # 2 минуты
        
        results = await pipe.execute()
        current_requests = results[1] if results[1] else 0
        
        # Проверяем лимиты
        is_allowed = current_requests < self.requests_per_minute
        
        limit_info = {
            "limit": self.requests_per_minute,
            "remaining": max(0, self.requests_per_minute - current_requests - 1),
            "reset_time": current_time + 60,
            "window_start": window_start,
            "current_requests": current_requests + 1,
            "burst_used": min(current_requests + 1, self.burst_limit),
            "burst_limit": self.burst_limit
        }
        
        return is_allowed, limit_info
    
    async def _check_rate_limit_local(self, client_id: str, current_time: int, window_start: int) -> tuple[bool, Dict[str, Any]]:
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
            "burst_used": min(current_requests + 1, self.burst_limit),
            "burst_limit": self.burst_limit
        }
        
        return is_allowed, limit_info
    
    async def _handle_rate_limit_exceeded(self, request: Request, limit_info: Dict[str, Any]) -> Response:
        """
        Обработка превышения лимита скорости.
        
        Args:
            request: HTTP запрос
            limit_info: Информация о лимитах
            
        Returns:
            Response: HTTP ответ с ошибкой
        """
        logger.warning(
            f"Rate limit exceeded for {request.client.host if request.client else 'unknown'}: "
            f"{limit_info['current_requests']}/{limit_info['limit']} requests per minute"
        )
        
        # Создаем ответ с информацией о лимитах
        detail = {
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "limit": limit_info["limit"],
            "window_start": limit_info["window_start"],
            "reset_time": limit_info["reset_time"],
            "retry_after": limit_info["reset_time"] - int(time.time())
        }
        
        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=detail
        )
        
        # Добавляем заголовки с информацией о лимитах
        self._add_rate_limit_headers(response, limit_info)
        
        return response
    
    def _add_rate_limit_headers(self, response: Response, limit_info: Dict[str, Any]):
        """
        Добавление заголовков с информацией о лимитах.
        
        Args:
            response: HTTP ответ
            limit_info: Информация о лимитах
        """
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset_time"])
        response.headers["X-RateLimit-Window"] = "60"  # 1 минута
        
        # Дополнительные заголовки для burst limit
        if "burst_limit" in limit_info:
            response.headers["X-Burst-Limit"] = str(limit_info["burst_limit"])
            response.headers["X-Burst-Used"] = str(limit_info["burst_used"])
    
    def _get_default_limit_info(self) -> Dict[str, Any]:
        """
        Получение информации о лимитах по умолчанию.
        
        Returns:
            Dict[str, Any]: Информация о лимитах
        """
        return {
            "limit": self.requests_per_minute,
            "remaining": self.requests_per_minute,
            "reset_time": int(time.time()) + 60,
            "window_start": int(time.time()) - 60,
            "current_requests": 0,
            "burst_used": 0,
            "burst_limit": self.burst_limit
        }


class RateLimitConfig:
    """
    Конфигурация rate limiting для разных endpoint-ов.
    """
    
    # Лимиты по умолчанию для разных типов endpoint-ов
    DEFAULT_LIMITS = {
        "general": {"requests_per_minute": 60, "burst": 10},
        "upload": {"requests_per_minute": 30, "burst": 5},
        "verify": {"requests_per_minute": 100, "burst": 20},
        "admin": {"requests_per_minute": 20, "burst": 5},
        "health": {"requests_per_minute": 1000, "burst": 100}
    }
    
    @classmethod
    def get_limits_for_endpoint(cls, path: str, method: str) -> Dict[str, int]:
        """
        Получение лимитов для конкретного endpoint-а.
        
        Args:
            path: Путь к endpoint-у
            method: HTTP метод
            
        Returns:
            Dict[str, int]: Лимиты для endpoint-а
        """
        # Определяем тип endpoint-а по пути
        if "/upload" in path:
            return cls.DEFAULT_LIMITS["upload"]
        elif "/verify" in path:
            return cls.DEFAULT_LIMITS["verify"]
        elif "/admin" in path:
            return cls.DEFAULT_LIMITS["admin"]
        elif any(health_path in path for health_path in ["/health", "/status", "/ready", "/live"]):
            return cls.DEFAULT_LIMITS["health"]
        else:
            return cls.DEFAULT_LIMITS["general"]


def rate_limit(requests_per_minute: int = None, burst: int = None):
    """
    Декоратор для применения специфичных лимитов к endpoint-ам.
    
    Args:
        requests_per_minute: Лимит запросов в минуту
        burst: Burst лимит
        
    Returns:
        Декоратор функции
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # TODO: Реализовать применение специфичных лимитов
            # Сейчас просто вызываем функцию
            return await func(*args, **kwargs)
        
        # Сохраняем информацию о лимитах в функции
        wrapper._rate_limits = {
            "requests_per_minute": requests_per_minute,
            "burst": burst
        }
        
        return wrapper
    
    return decorator