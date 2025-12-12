"""
API роуты.
Все endpoints приложения.
"""

# TODO Phase 2: Создать остальные роуты
# from . import upload, verify, liveness, reference, admin
from . import health

__all__ = ["health"]
