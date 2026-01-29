"""
Фоновые асинхронные задачи очистки.
Удаление истекших сессий, старых файлов и данных.

================================================================================
ПОЛИТИКА ХРАНЕНИЯ И АВТОМАТИЧЕСКОГО УДАЛЕНИЯ
================================================================================

| Тип данных              | Срок хранения     | Автоматическое удаление     |
|-------------------------|-------------------|----------------------------|
| Upload sessions         | 24 часа (TTL)     | Redis TTL                  |
| Verification sessions   | 30 дней           | cleanup_old_verification_sessions() |
| Эталонные фото (raw)    | 30 дней           | cleanup_old_files_from_storage()    |
| Audit логи              | 90 дней           | cleanup_old_logs()                  |
| Biometric templates     | 3 года inactivity | cleanup_inactive_biometric_templates() |
| Webhook логи            | 30 дней           | cleanup_old_webhook_logs()          |

================================================================================
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from ..config import settings
from ..db.database import get_async_db_manager
from ..services.cache_service import CacheService
from ..services.database_service import DatabaseService
from ..services.session_service import SessionService
from ..services.storage_service import StorageService
from ..utils.logger import get_logger

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
        async with get_async_db_manager().get_session() as db:
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
        async with get_async_db_manager().get_session() as db:
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
    # Biometric templates (GDPR compliance)
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_inactive_biometric_templates(days: int = 1095) -> int:
        """
        Удаляет биометрические шаблоны пользователей, которые не использовались
        более указанного количества дней (по умолчанию 3 года = 1095 дней).

        Это обеспечивает compliance с GDPR "right to be forgotten" и принципом
        минимизации хранения данных.

        Args:
            days: Количество дней неактивности перед удалением (default: 1095 = 3 года)

        Returns:
            int: Количество удалённых шаблонов
        """
        async with get_async_db_manager().get_session() as db:
            try:
                db_service = DatabaseService(db)

                # Удаляем только soft-deleted записи старше указанного срока
                from sqlalchemy import text

                result = await db.execute(
                    text(
                        f"""
                        DELETE FROM biometric_templates 
                        WHERE is_active = False 
                        AND updated_at < NOW() - INTERVAL '{days} days'
                    """
                    )
                )
                await db.commit()

                deleted = result.rowcount
                logger.info(
                    f"🗑️ Cleanup: removed {deleted} inactive biometric templates "
                    f"(inactive > {days} days)"
                )
                return deleted

            except Exception as e:
                await db.rollback()
                logger.exception("cleanup_inactive_biometric_templates failed")
                return 0

    # ------------------------------------------------------------------
    # Photo retention policy enforcement
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_raw_photos(days: int = 30) -> Dict[str, int]:
        """
        Удаляет исходные (raw) фотографии пользователей согласно политике хранения.

        Эталонные фото хранятся в MinIO в бакете с lifecycle rule:
        - 30 дней для обычных пользователей
        - 90 дней для корпоративных клиентов

        Args:
            days: Количество дней хранения raw фото (default: 30)

        Returns:
            Dict с информацией об удалённых файлах
        """
        storage = StorageService()
        cutoff_date = utcnow() - timedelta(days=days)

        result = {"scanned": 0, "deleted": 0, "errors": 0, "deleted_keys": []}

        try:
            # Сканируем бакеет на предмет старых файлов
            async for file_info in storage.list_files_async(
                prefix="references/", limit=5000
            ):
                result["scanned"] += 1

                last_modified = file_info.get("last_modified")
                if last_modified and last_modified < cutoff_date:
                    try:
                        await storage.delete_image(file_info["key"])
                        result["deleted"] += 1
                        result["deleted_keys"].append(file_info["key"])
                    except Exception as e:
                        logger.error(f"Failed to delete {file_info['key']}: {e}")
                        result["errors"] += 1

            logger.info(
                f"📸 Photo cleanup: scanned={result['scanned']}, "
                f"deleted={result['deleted']}, errors={result['errors']}"
            )
            return result

        except Exception as e:
            logger.error(f"cleanup_raw_photos failed: {e}")
            return result

    # ------------------------------------------------------------------
    # Webhook logs retention
    # ------------------------------------------------------------------
    @staticmethod
    async def cleanup_old_webhook_logs(days: int = 30) -> int:
        """
        Удаляет старые webhook логи согласно политике хранения.
        """
        async with get_async_db_manager().get_session() as db:
            try:
                db_service = DatabaseService(db)
                deleted = await db_service.webhook_crud.cleanup_old_logs(
                    db,
                    days=days,
                )
                await db.commit()

                logger.info("🧹 Cleanup: removed %s old webhook log records", deleted)
                return deleted

            except Exception:
                await db.rollback()
                logger.exception("cleanup_old_webhook_logs failed")
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
            "retention_policy": {
                "upload_sessions_hours": 24,
                "verification_sessions_days": 30,
                "raw_photos_days": 30,
                "audit_logs_days": 90,
                "biometric_templates_inactive_years": 3,
                "webhook_logs_days": 30,
            },
        }

    # ------------------------------------------------------------------
    # Full cleanup (GDPR compliance)
    # ------------------------------------------------------------------
    @staticmethod
    async def run_full_cleanup() -> Dict[str, Any]:
        """
        Полная очистка системы с соблюдением политики хранения данных.

        Выполняет все cleanup задачи согласно retention policy:
        - Upload sessions (Redis TTL)
        - Raw photos (30 дней)
        - Verification sessions (30 дней)
        - Audit logs (90 дней)
        - Biometric templates inactive (3 года)
        - Webhook logs (30 дней)
        """
        logger.info("Starting full GDPR-compliant system cleanup")

        results = {
            # Redis TTL handles upload sessions automatically
            "upload_sessions": "handled_by_redis_ttl",
            # Raw photos cleanup
            "raw_photos": await CleanupTasks.cleanup_raw_photos(
                days=settings.UPLOAD_EXPIRATION_DAYS
            ),
            # Database cleanups
            "verification_sessions": await CleanupTasks.cleanup_old_verification_sessions(),
            "audit_logs": await CleanupTasks.cleanup_old_logs(),
            # GDPR compliance: cleanup inactive biometric templates
            "inactive_biometric_templates": await CleanupTasks.cleanup_inactive_biometric_templates(
                days=1095  # 3 years
            ),
            "webhook_logs": await CleanupTasks.cleanup_old_webhook_logs(),
        }

        # Calculate totals
        total_deleted = (
            results["raw_photos"].get("deleted", 0)
            + results["verification_sessions"]
            + results["audit_logs"]
            + results["inactive_biometric_templates"]
            + results["webhook_logs"]
        )

        logger.info(
            "🧹 Full GDPR-compliant cleanup finished. Total deleted: %s. Details: %s",
            total_deleted,
            results,
        )

        return {
            "total_deleted": total_deleted,
            "details": results,
            "timestamp": utcnow().isoformat(),
            "policy": "GDPR_compliant",
        }
