"""
Фоновые асинхронные задачи очистки.
Удаление истекших сессий, старых файлов и данных.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from ..config import settings
from ..services.storage_service import StorageService
from ..services.session_service import SessionService
from ..services.cache_service import CacheService
from ..services.database_service import DatabaseService
from ..utils.logger import get_logger
from ..db.database import get_async_db_manager  # ✅ ИСПРАВЛЕНО

logger = get_logger(__name__)


def utcnow() -> datetime:
    """Единая точка получения UTC-времени"""
    return datetime.now(timezone.utc)


class CleanupTasks:
    """Асинхронные фоновые задачи очистки"""

    # ------------------------------------------------------------------
    # Upload sessions (Redis)
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_expired_upload_sessions() -> int:
        """
        Удаляет истекшие upload sessions из Redis.
        Redis TTL автоматически удаляет ключи, но этот метод для мануального вызова.
        """
        cache = CacheService()
        pattern = "upload_session:*"
        deleted_count = 0
        cursor = 0
        while True:
            cursor, keys = await cache.redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                session_id = key.decode().split(":")[-1]
                session = await SessionService.get_session(session_id)
                if session is None:  # Истекшая или несуществующая
                    deleted_count += 1
            if cursor == 0:
                break
        logger.info(f"Cleanup: found {deleted_count} expired upload sessions")
        return deleted_count

    # ------------------------------------------------------------------
    # Old files (MinIO/S3)
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_old_files_from_storage() -> int:
        """
        Удаляет старые файлы из MinIO (старше UPLOAD_EXPIRATION_DAYS дней).
        """
        storage = StorageService()
        cutoff_date = utcnow() - timedelta(days=settings.UPLOAD_EXPIRATION_DAYS)
        try:
            files = await storage.list_files(prefix="uploads/", limit=1000)
            deleted_count = 0
            for file_info in files:
                last_modified = file_info.get("last_modified")
                if last_modified and last_modified < cutoff_date:
                    try:
                        await storage.delete_image(file_info["key"])
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_info['key']}: {e}")
            logger.info(f"Cleanup: deleted {deleted_count} old files from storage")
            return deleted_count
        except Exception as e:
            logger.error(f"cleanup_old_files_from_storage failed: {e}")
            return 0

    # ------------------------------------------------------------------
    # Verification sessions (DB)
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_old_verification_sessions() -> int:
        """
        Удаляет старые сессии верификации из БД.
        Удаляются записи старше UPLOAD_EXPIRATION_DAYS дней.
        Returns:
            int: Количество удалённых записей
        """
        async with get_async_db_manager().get_session() as db:  # ✅ ИСПРАВЛЕНО
            try:
                db_service = DatabaseService(db)
                deleted = await db_service.verification_crud.cleanup_old_sessions(
                    db,
                    days=settings.UPLOAD_EXPIRATION_DAYS,
                )
                await db.commit()

                logger.info("Cleanup: removed %s old verification sessions", deleted)
                return deleted

            except Exception:
                await db.rollback()
                logger.exception("cleanup_old_verification_sessions failed")
                return 0

    # ------------------------------------------------------------------
    # Audit logs (DB)
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_old_logs() -> int:
        async with get_async_db_manager().get_session() as db:  # ✅ ИСПРАВЛЕНО
            try:
                db_service = DatabaseService(db)
                deleted = await db_service.audit_crud.cleanup_old_logs(
                    db,
                    days=settings.UPLOAD_EXPIRATION_DAYS,
                )
                await db.commit()

                logger.info("Cleanup: removed %s old audit log records", deleted)
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
            "timestamp": utcnow().isoformat(),
            "message": "Cleanup handled automatically by Redis TTL and S3 lifecycle rules",
            "note": "Manual cleanup tasks are minimized. Use run_full_cleanup() for on-demand cleanup.",
        }

    # ------------------------------------------------------------------
    # Full cleanup
    # ------------------------------------------------------------------
    @staticmethod
    async def run_full_cleanup() -> Dict[str, int]:
        logger.info("Starting full system cleanup")

        results = {
            "expired_upload_sessions": await CleanupTasks.cleanup_expired_upload_sessions(),
            "old_files": await CleanupTasks.cleanup_old_files_from_storage(),
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
