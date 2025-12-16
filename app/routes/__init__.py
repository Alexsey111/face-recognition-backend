"""
API роуты.
Все endpoints приложения.
"""

from . import health, upload, verify, liveness, reference, admin

__all__ = ["health", "upload", "verify", "liveness", "reference", "admin"]
