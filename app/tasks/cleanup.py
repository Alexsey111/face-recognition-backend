"""
Фоновые асинхронные задачи очистки.
Удаление истекших сессий, старых файлов и данных.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from ..config import settings
from ..services.storage_service import StorageService
from ..services.session_service import SessionService
from ..services.database_service import DatabaseService
from ..utils.logger import get_logger
from ..db.database import AsyncSessionLocal

logger = get_logger(__name__)


UTC_NOW = lambda: datetime.now(timezone.utc)


class CleanupTasks:
    """Асинхронные фоновые задачи очистки"""

    """
    Удаляет старые сессии верификации из БД.

    Удаляются записи старше UPLOAD_EXPIRATION_DAYS дней.

    Returns:
        int: Количество удалённых записей
    """
  
    # ------------------------------------------------------------------
    # Verification sessions (DB)
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_old_verification_sessions() -> int:
        async with AsyncSessionLocal() as db:
            try:
                db_service = DatabaseService(db)
                deleted = await db_service.verification_crud.cleanup_old_sessions(
                    db,
                    days=settings.UPLOAD_EXPIRATION_DAYS,
                )
                await db.commit()

                logger.info(
                    "Cleanup: removed %s old verification sessions", deleted
                )
                return deleted

            except Exception:
                await db.rollback()
                logger.exception(
                    "cleanup_old_verification_sessions failed"
                )
                return 0

    # ------------------------------------------------------------------
    # Audit logs (DB)
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_old_logs() -> int:
        async with AsyncSessionLocal() as db:
            try:
                db_service = DatabaseService(db)
                deleted = await db_service.audit_crud.cleanup_old_logs(
                    db,
                    days=settings.UPLOAD_EXPIRATION_DAYS,
                )
                await db.commit()

                logger.info(
                    "Cleanup: removed %s old audit log records", deleted
                )
                return deleted

            except Exception:
                await db.rollback()
                logger.exception("cleanup_old_logs failed")
                return 0



    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @staticmethod
    async def get_cleanup_stats() -> Dict[str, Any]:
        return {
            "timestamp": UTC_NOW().isoformat(),
            "message": "Cleanup handled automatically by Redis TTL and S3 lifecycle rules",
            "note": "Manual cleanup tasks are minimized. Use run_full_cleanup() for on-demand cleanup."
        }

    # ------------------------------------------------------------------
    # Full cleanup
    # ------------------------------------------------------------------
    @staticmethod
    async def run_full_cleanup() -> Dict[str, int]:
        logger.info("Starting full system cleanup")

        results = {
            "verification_sessions": await CleanupTasks.cleanup_old_verification_sessions(),
            "old_logs": await CleanupTasks.cleanup_old_logs(),
            
        }

        total = sum(results.values())
        logger.info(
            "Full cleanup finished. Total deleted: %s. Details: %s",
            total,
            results,
        )

        return results
