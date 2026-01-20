"""
API роуты.
Все endpoints приложения.
"""

from . import health, verify, liveness, reference, admin, upload, metrics

__all__ = ["health", "verify", "liveness", "reference", "admin", "upload", "auth", "webhook", "metrics"]
