"""
Безопасность и утилиты.
Password hashing, CORS конфигурация, security headers и другие функции безопасности.
"""

import secrets
import hashlib
import hmac
import os
import re
from typing import Optional, List, Dict, Any

from fastapi import Request

from ..config import settings
from .logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Hashing / Tokens
# ==========================================================


class SecurityHash:
    """Утилиты для создания безопасных хешей."""

    @staticmethod
    def generate_secure_hash(data: str, secret: Optional[str] = None) -> str:
        """Создание HMAC-SHA256 хеша."""
        secret_key = secret or settings.JWT_SECRET_KEY

        if not secret_key:
            raise ValueError("Secret key is not configured")

        try:
            return hmac.new(
                secret_key.encode(), data.encode(), hashlib.sha256
            ).hexdigest()
        except Exception as exc:
            logger.exception("Failed to generate secure hash")
            raise exc

    @staticmethod
    def generate_api_key(byte_length: int = 32) -> str:
        """Генерация API ключа."""
        return secrets.token_hex(byte_length)

    @staticmethod
    def generate_session_id() -> str:
        """Генерация ID сессии."""
        return secrets.token_hex(16)


# ==========================================================
# CORS
# ==========================================================


class CORSConfig:
    """Конфигурация CORS для FastAPI."""

    @staticmethod
    def middleware_config() -> Dict[str, Any]:
        return {
            "allow_origins": settings.cors_origins_list,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            "allow_headers": [
                "Authorization",
                "Content-Type",
                "Accept",
                "Origin",
                "User-Agent",
                "Cache-Control",
                "If-Modified-Since",
            ],
            "expose_headers": [
                "X-Total-Count",
                "X-Rate-Limit-Limit",
                "X-Rate-Limit-Remaining",
                "X-Rate-Limit-Reset",
            ],
            "max_age": 86400,
        }

    @staticmethod
    def is_origin_allowed(origin: str) -> bool:
        if settings.DEBUG:
            return True

        for allowed in settings.cors_origins_list:
            if allowed == "*" or origin == allowed:
                return True
            if allowed.endswith("*") and origin.startswith(allowed[:-1]):
                return True

        return False


# ==========================================================
# Security Headers
# ==========================================================


class SecurityHeaders:
    """Security headers для HTTP ответов."""

    @staticmethod
    def base_headers() -> Dict[str, str]:
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
            ),
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "img-src 'self' data: https:; "
                "style-src 'self'; "
                "script-src 'self'; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            ),
        }

        if not settings.DEBUG:
            headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        return headers

    @staticmethod
    def api_headers() -> Dict[str, str]:
        headers = SecurityHeaders.base_headers()
        headers.update(
            {
                "X-Robots-Tag": "noindex, nofollow",
                "API-Version": "v1",
            }
        )
        return headers


# ==========================================================
# Rate limiting
# ==========================================================


class RateLimitConfig:
    """Конфигурация rate limiting."""

    @staticmethod
    def default() -> Dict[str, Any]:
        return {
            "requests_per_minute": settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "burst_size": settings.RATE_LIMIT_BURST,
            "block_duration": 300,
        }

    @staticmethod
    def per_endpoint() -> Dict[str, Dict[str, Any]]:
        return {
            "/api/v1/auth/login": {
                "requests_per_minute": 5,
                "burst_size": 2,
                "block_duration": 900,
            },
            "/api/v1/auth/register": {
                "requests_per_minute": 3,
                "burst_size": 1,
                "block_duration": 1800,
            },
            "/api/v1/auth/refresh": {
                "requests_per_minute": 10,
                "burst_size": 3,
                "block_duration": 300,
            },
        }


# ==========================================================
# Input Sanitization
# ==========================================================


class InputSanitizer:
    """Санитизация входных данных."""

    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        if not text:
            return ""

        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"on\w+\s*=", "", text, flags=re.IGNORECASE)
        text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)

        return text.strip()[:max_length]

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        if not filename:
            return ""

        filename = os.path.basename(filename)
        filename = re.sub(r"[^\w\-.]", "_", filename)

        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[: 250 - len(ext)] + ext

        return filename


# ==========================================================
# Utilities
# ==========================================================


class SecurityUtils:
    """Общие утилиты безопасности."""

    @staticmethod
    def generate_csrf_token() -> str:
        return secrets.token_urlsafe(32)

    @staticmethod
    def verify_csrf_token(token: str, session_token: str) -> bool:
        return hmac.compare_digest(token or "", session_token or "")

    @staticmethod
    def get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    @staticmethod
    def detect_suspicious_activity(
        user_agent: str,
        client_ip: str,
        request_count: int,
    ) -> Dict[str, Any]:

        indicators = []
        score = 0

        if not user_agent or len(user_agent) < 10:
            indicators.append("Invalid User-Agent")
            score += 10

        if any(bot in user_agent.lower() for bot in ("bot", "crawler", "spider")):
            indicators.append("Bot signature detected")
            score += 15

        if request_count > 100:
            indicators.append("High request rate")
            score += 20

        level = "low"
        if score >= 30:
            level = "high"
        elif score >= 15:
            level = "medium"

        return {
            "risk_level": level,
            "risk_score": score,
            "indicators": indicators,
            "requires_review": score >= 30,
        }

    @staticmethod
    def is_ip_blacklisted(ip: str) -> bool:
        raise NotImplementedError("GeoIP blacklist is not implemented yet")

    @staticmethod
    def get_ip_geolocation(ip: str) -> Dict[str, Any]:
        raise NotImplementedError("GeoIP lookup is not implemented yet")
