"""
API роуты.
Все endpoints приложения.
"""

from . import (
    admin,
    face_recognition,
    health,
    liveness,
    metrics,
    reference,
    upload,
    verify,
)

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
