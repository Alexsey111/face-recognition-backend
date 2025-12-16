"""
Безопасность и утилиты.
Password hashing, CORS конфигурация, security headers и другие функции безопасности.
"""

import secrets
import hashlib
import hmac
from typing import Optional, List, Dict, Any
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SecurityHash:
    """
    Утилиты для создания безопасных хешей.
    """

    @staticmethod
    def generate_secure_hash(data: str, secret: Optional[str] = None) -> str:
        """
        Создание HMAC SHA256 хеша.
        
        Args:
            data: Данные для хеширования
            secret: Секретный ключ (используется JWT_SECRET_KEY если не указан)
            
        Returns:
            str: HMAC хеш
        """
        try:
            secret_key = secret or settings.JWT_SECRET_KEY
            signature = hmac.new(
                secret_key.encode('utf-8'),
                data.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return signature
            
        except Exception as e:
            logger.error(f"Error generating secure hash: {str(e)}")
            raise

    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """
        Генерация API ключа.
        
        Args:
            length: Длина ключа
            
        Returns:
            str: API ключ
        """
        try:
            key_bytes = secrets.token_bytes(length)
            api_key = key_bytes.hex()
            logger.debug(f"API key generated (length: {length})")
            return api_key
            
        except Exception as e:
            logger.error(f"Error generating API key: {str(e)}")
            raise

    @staticmethod
    def generate_session_id() -> str:
        """
        Генерация уникального ID сессии.
        
        Returns:
            str: ID сессии
        """
        try:
            session_bytes = secrets.token_bytes(16)
            session_id = session_bytes.hex()
            return session_id
            
        except Exception as e:
            logger.error(f"Error generating session ID: {str(e)}")
            raise


class CORSConfig:
    """
    Конфигурация CORS для FastAPI.
    """

    @staticmethod
    def get_cors_middleware_config() -> Dict[str, Any]:
        """
        Получение конфигурации CORS middleware.
        
        Returns:
            Dict[str, Any]: Конфигурация CORS
        """
        return {
            "allow_origins": settings.cors_origins_list,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            "allow_headers": [
                "Authorization",
                "Content-Type",
                "X-Requested-With",
                "Accept",
                "Origin",
                "User-Agent",
                "DNT",
                "Cache-Control",
                "X-Mx-ReqToken",
                "Keep-Alive",
                "X-Requested-With",
                "If-Modified-Since"
            ],
            "expose_headers": [
                "X-Total-Count",
                "X-Rate-Limit-Limit",
                "X-Rate-Limit-Remaining",
                "X-Rate-Limit-Reset"
            ],
            "max_age": 86400,  # 24 часа
            "allow_origin_regex": None,
        }

    @staticmethod
    def get_allowed_origins() -> List[str]:
        """
        Получение списка разрешенных origins.
        
        Returns:
            List[str]: Список разрешенных origins
        """
        return settings.cors_origins_list

    @staticmethod
    def is_origin_allowed(origin: str) -> bool:
        """
        Проверка, разрешен ли origin.
        
        Args:
            origin: Origin для проверки
            
        Returns:
            bool: True если origin разрешен
        """
        allowed_origins = settings.cors_origins_list
        
        # В development режиме разрешаем все
        if settings.DEBUG:
            return True
        
        # Проверяем точное совпадение
        if origin in allowed_origins:
            return True
        
        # Проверяем wildcard patterns
        for allowed in allowed_origins:
            if allowed == "*":
                return True
            if allowed.endswith("*"):
                prefix = allowed[:-1]
                if origin.startswith(prefix):
                    return True
        
        return False


class SecurityHeaders:
    """
    Security headers для HTTP ответов.
    """

    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """
        Получение security headers.
        
        Returns:
            Dict[str, str]: Security headers
        """
        return {
            # Защита от MIME sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Защита от clickjacking
            "X-Frame-Options": "DENY",
            
            # XSS Protection
            "X-XSS-Protection": "1; mode=block",
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Content Security Policy (базовый)
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            ),
            
            # HSTS (только для HTTPS)
            "Strict-Transport-Security": (
                "max-age=31536000; includeSubDomains; preload"
            ) if not settings.DEBUG else "",
            
            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=()"
            ),
            
            # Cache Control для API
            "Cache-Control": "no-cache, no-store, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0"
        }

    @staticmethod
    def get_api_security_headers() -> Dict[str, str]:
        """
        Получение security headers специально для API.
        
        Returns:
            Dict[str, str]: API security headers
        """
        headers = SecurityHeaders.get_security_headers()
        
        # Для API убираем некоторые headers которые могут мешать
        headers.pop("X-Frame-Options", None)
        headers.pop("X-XSS-Protection", None)
        
        # Добавляем API-специфичные headers
        headers.update({
            "API-Version": "v1",
            "X-Content-Type-Options": "nosniff",
            "X-Robots-Tag": "noindex, nofollow"
        })
        
        return headers

    @staticmethod
    def get_cors_headers() -> Dict[str, str]:
        """
        Получение CORS headers.
        
        Returns:
            Dict[str, str]: CORS headers
        """
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Max-Age": "86400"
        }


class RateLimitConfig:
    """
    Конфигурация rate limiting.
    """

    @staticmethod
    def get_rate_limit_config() -> Dict[str, Any]:
        """
        Получение конфигурации rate limiting.
        
        Returns:
            Dict[str, Any]: Конфигурация rate limiting
        """
        return {
            "requests_per_minute": settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "burst_size": settings.RATE_LIMIT_BURST,
            "block_duration": 300,  # 5 минут блокировки
            "skip_successful_requests": False,
            "skip_failed_requests": False,
        }

    @staticmethod
    def get_endpoint_limits() -> Dict[str, Dict[str, Any]]:
        """
        Получение лимитов для конкретных endpoints.
        
        Returns:
            Dict[str, Dict[str, Any]]: Лимиты по endpoint-ам
        """
        return {
            "/api/v1/auth/login": {
                "requests_per_minute": 5,
                "burst_size": 2,
                "block_duration": 900  # 15 минут
            },
            "/api/v1/auth/register": {
                "requests_per_minute": 3,
                "burst_size": 1,
                "block_duration": 1800  # 30 минут
            },
            "/api/v1/auth/refresh": {
                "requests_per_minute": 10,
                "burst_size": 3,
                "block_duration": 300  # 5 минут
            },
            "/api/v1/upload": {
                "requests_per_minute": 20,
                "burst_size": 5,
                "block_duration": 600  # 10 минут
            },
            "/api/v1/verify": {
                "requests_per_minute": 30,
                "burst_size": 10,
                "block_duration": 300  # 5 минут
            }
        }


class InputSanitizer:
    """
    Санитизация входных данных.
    """

    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """
        Санитизация строки.
        
        Args:
            text: Строка для санитизации
            max_length: Максимальная длина
            
        Returns:
            str: Санитизированная строка
        """
        if not text:
            return ""
        
        # Удаляем потенциально опасные символы
        import re
        
        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаляем JavaScript события
        text = re.sub(r'on\w+\s*=', '', text)
        
        # Удаляем javascript: протоколы
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Обрезаем до максимальной длины
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Санитизация имени файла.
        
        Args:
            filename: Имя файла для санитизации
            
        Returns:
            str: Санитизированное имя файла
        """
        if not filename:
            return ""
        
        # Удаляем path traversal
        import os
        filename = os.path.basename(filename)
        
        # Удаляем опасные символы
        import re
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        
        # Ограничиваем длину
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250-len(ext)] + ext
        
        return filename


class SecurityUtils:
    """
    Общие утилиты безопасности.
    """

    @staticmethod
    def generate_csrf_token() -> str:
        """
        Генерация CSRF токена.
        
        Returns:
            str: CSRF токен
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def verify_csrf_token(token: str, session_token: str) -> bool:
        """
        Проверка CSRF токена.
        
        Args:
            token: Токен от клиента
            session_token: Токен из сессии
            
        Returns:
            bool: True если токены совпадают
        """
        try:
            return hmac.compare_digest(token, session_token)
        except Exception:
            return False

    @staticmethod
    def is_secure_context() -> bool:
        """
        Проверка, является ли контекст безопасным (HTTPS).
        
        Returns:
            bool: True если контекст безопасный
        """
        return not settings.DEBUG  # В production всегда HTTPS

    @staticmethod
    def get_client_ip(request) -> str:
        """
        Получение реального IP адреса клиента.
        
        Args:
            request: FastAPI request объект
            
        Returns:
            str: IP адрес клиента
        """
        # Проверяем различные заголовки для получения реального IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        client_ip = request.client.host if request.client else "unknown"
        return client_ip

    @staticmethod
    def detect_suspicious_activity(
        user_agent: str, 
        client_ip: str, 
        request_count: int
    ) -> Dict[str, Any]:
        """
        Обнаружение подозрительной активности.
        
        Args:
            user_agent: User agent браузера
            client_ip: IP адрес клиента
            request_count: Количество запросов
            
        Returns:
            Dict[str, Any]: Результат анализа
        """
        suspicious_indicators = []
        risk_score = 0
        
        # Проверяем User Agent
        if not user_agent or len(user_agent) < 10:
            suspicious_indicators.append("Missing or short User Agent")
            risk_score += 10
        
        # Проверяем на бота
        bot_indicators = ["bot", "crawler", "spider", "scraper"]
        if any(indicator in user_agent.lower() for indicator in bot_indicators):
            suspicious_indicators.append("Bot User Agent detected")
            risk_score += 15
        
        # Проверяем частоту запросов
        if request_count > 100:
            suspicious_indicators.append("High request frequency")
            risk_score += 20
        
        # Проверяем IP (простейшая проверка)
        if client_ip.startswith("10.") or client_ip.startswith("192.168."):
            # Локальный IP - может быть нормально
            pass
        
        risk_level = "low"
        if risk_score >= 30:
            risk_level = "high"
        elif risk_score >= 15:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "suspicious_indicators": suspicious_indicators,
            "requires_review": risk_score >= 30
        }

    # 🟢 В Phase 5 добавь GeoIP блокировку
    @staticmethod
    def is_ip_blacklisted(ip: str) -> bool:
        """Check if IP is in blacklist"""
        # TODO: Integrate with GeoIP database
        pass

    @staticmethod
    def get_ip_geolocation(ip: str) -> Dict[str, Any]:
        """Get IP geolocation data"""
        # TODO: Integrate with MaxMind GeoIP2
        pass