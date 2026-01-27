"""
API роуты.
Все endpoints приложения.
"""

from . import health, verify, liveness, reference, admin, upload, metrics, face_recognition

__all__ = [
    "health",
    "verify",
    "liveness",
    "reference",
    "admin",
    "upload",
    "auth",
    "webhook",
    "metrics",
    "face_recognition",
]
