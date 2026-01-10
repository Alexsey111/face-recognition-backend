"""Планировщик задач для системы."""

from datetime import datetime, timezone
from typing import Optional, Dict, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from ..utils.logger import get_logger
from ..db.database import get_async_db_manager

logger = get_logger(__name__)


class CleanupScheduler:
    """Планировщик для задач очистки данных."""
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.started: bool = False

    def start(self):
        """Запуск планировщика очистки."""
        if self.started:
            logger.warning("CleanupScheduler already started")
            return
            
        self.scheduler = AsyncIOScheduler(timezone=timezone.utc)
        
        from .cleanup import CleanupTasks  # Ленивый импорт

        # Очистка сессий - каждые 12ч
        self.scheduler.add_job(
            CleanupTasks.cleanup_old_verification_sessions,
            IntervalTrigger(hours=12),
            id="cleanup_verification_sessions",
            name="Cleanup Old Verification Sessions",
            replace_existing=True
        )

        # Очистка логов аудита - ежедневно
        self.scheduler.add_job(
            CleanupTasks.cleanup_old_logs,
            IntervalTrigger(hours=24),
            id="cleanup_old_logs",
            name="Cleanup Old Audit Logs",
            replace_existing=True
        )

        # Полная очистка - еженедельно
        self.scheduler.add_job(
            CleanupTasks.run_full_cleanup,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="full_cleanup",
            name="Full System Cleanup",
            replace_existing=True
        )

        self.scheduler.start()
        self.started = True
        logger.info("CleanupScheduler started successfully")

    async def shutdown(self):
        """Остановка планировщика очистки."""
        if self.scheduler and self.started:
            self.scheduler.shutdown(wait=True)
            self.started = False
            logger.info("CleanupScheduler stopped")


class WebhookScheduler:
    """
    Планировщик для задач webhook.
    
    ВАЖНО: Retry логика теперь встроена в WebhookService через asyncio.create_task.
    Этот планировщик используется ТОЛЬКО для:
    - Очистки старых логов webhook
    - Периодических статистических отчетов
    - Мониторинга состояния системы webhook
    """
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.started: bool = False

    def start(self):
        """Запуск планировщика webhook."""
        if self.started:
            logger.warning("WebhookScheduler already started")
            return
            
        self.scheduler = AsyncIOScheduler(timezone=timezone.utc)

        # Очистка старых логов webhook - ежедневно в 3:00 UTC
        self.scheduler.add_job(
            self._cleanup_old_webhook_logs,
            CronTrigger(hour=3, minute=0),
            id="webhook_log_cleanup",
            name="Cleanup Old Webhook Logs",
            replace_existing=True
        )

        # Статистика webhook в лог - каждые 30 минут
        self.scheduler.add_job(
            self._log_webhook_statistics,
            IntervalTrigger(minutes=30),
            id="webhook_stats_logging",
            name="Log Webhook Statistics",
            replace_existing=True
        )
        
        # Проверка "застрявших" webhook в статусе PENDING - каждые 5 минут
        self.scheduler.add_job(
            self._check_stale_webhooks,
            IntervalTrigger(minutes=5),
            id="webhook_stale_check",
            name="Check Stale Webhooks",
            replace_existing=True
        )

        self.scheduler.start()
        self.started = True
        logger.info("WebhookScheduler started successfully")

    async def _cleanup_old_webhook_logs(self):
        """
        Очистка старых логов webhook.
        Удаляет логи старше 30 дней (настраивается).
        """
        try:
            from sqlalchemy import delete, and_
            from datetime import timedelta
            from ..db.models import WebhookLog, WebhookStatus
            
            logger.info("Starting webhook logs cleanup")
            
            # Получаем сессию БД
            async with get_async_db_manager().get_session() as db:
                # Удаляем успешные логи старше 30 дней
                cutoff_date_success = datetime.now(timezone.utc) - timedelta(days=30)
                
                result_success = await db.execute(
                    delete(WebhookLog).where(
                        and_(
                            WebhookLog.status == WebhookStatus.SUCCESS,
                            WebhookLog.created_at < cutoff_date_success
                        )
                    )
                )
                
                # Удаляем failed/expired логи старше 90 дней
                cutoff_date_failed = datetime.now(timezone.utc) - timedelta(days=90)
                
                result_failed = await db.execute(
                    delete(WebhookLog).where(
                        and_(
                            WebhookLog.status.in_([WebhookStatus.FAILED, WebhookStatus.EXPIRED]),
                            WebhookLog.created_at < cutoff_date_failed
                        )
                    )
                )
                
                await db.commit()
                
                total_deleted = result_success.rowcount + result_failed.rowcount
                logger.info(
                    f"Webhook logs cleanup completed: "
                    f"{result_success.rowcount} success logs, "
                    f"{result_failed.rowcount} failed logs deleted, "
                    f"total: {total_deleted}"
                )
                
        except Exception as e:
            logger.error(f"Error during webhook logs cleanup: {str(e)}")

    async def _log_webhook_statistics(self):
        """
        Логирование статистики webhook для мониторинга.
        """
        try:
            from sqlalchemy import select, func, and_
            from ..db.models import WebhookLog, WebhookConfig, WebhookStatus
            from datetime import timedelta
            
            async with get_async_db_manager().get_session() as db:
                # Статистика за последний час
                one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
                
                # Общее количество webhook за последний час
                result_total = await db.execute(
                    select(func.count(WebhookLog.id)).where(
                        WebhookLog.created_at >= one_hour_ago
                    )
                )
                total_webhooks = result_total.scalar() or 0
                
                # Успешные webhook
                result_success = await db.execute(
                    select(func.count(WebhookLog.id)).where(
                        and_(
                            WebhookLog.created_at >= one_hour_ago,
                            WebhookLog.status == WebhookStatus.SUCCESS
                        )
                    )
                )
                success_webhooks = result_success.scalar() or 0
                
                # Failed webhook
                result_failed = await db.execute(
                    select(func.count(WebhookLog.id)).where(
                        and_(
                            WebhookLog.created_at >= one_hour_ago,
                            WebhookLog.status == WebhookStatus.FAILED
                        )
                    )
                )
                failed_webhooks = result_failed.scalar() or 0
                
                # Pending webhook
                result_pending = await db.execute(
                    select(func.count(WebhookLog.id)).where(
                        and_(
                            WebhookLog.created_at >= one_hour_ago,
                            WebhookLog.status == WebhookStatus.PENDING
                        )
                    )
                )
                pending_webhooks = result_pending.scalar() or 0
                
                # Retry webhook
                result_retry = await db.execute(
                    select(func.count(WebhookLog.id)).where(
                        and_(
                            WebhookLog.created_at >= one_hour_ago,
                            WebhookLog.status == WebhookStatus.RETRY
                        )
                    )
                )
                retry_webhooks = result_retry.scalar() or 0
                
                # Средняя скорость обработки
                result_avg_time = await db.execute(
                    select(func.avg(WebhookLog.processing_time)).where(
                        and_(
                            WebhookLog.created_at >= one_hour_ago,
                            WebhookLog.processing_time.isnot(None)
                        )
                    )
                )
                avg_processing_time = result_avg_time.scalar() or 0
                
                # Активные конфигурации
                result_active_configs = await db.execute(
                    select(func.count(WebhookConfig.id)).where(
                        WebhookConfig.is_active == True
                    )
                )
                active_configs = result_active_configs.scalar() or 0
                
                success_rate = (success_webhooks / total_webhooks * 100) if total_webhooks > 0 else 0
                
                logger.info(
                    f"Webhook Stats (last hour): "
                    f"Total={total_webhooks}, "
                    f"Success={success_webhooks}, "
                    f"Failed={failed_webhooks}, "
                    f"Pending={pending_webhooks}, "
                    f"Retry={retry_webhooks}, "
                    f"Success Rate={success_rate:.2f}%, "
                    f"Avg Processing Time={avg_processing_time:.3f}s, "
                    f"Active Configs={active_configs}"
                )
                
        except Exception as e:
            logger.error(f"Error logging webhook statistics: {str(e)}")

    async def _check_stale_webhooks(self):
        """
        Проверка "застрявших" webhook в статусе PENDING.
        Webhook, которые находятся в статусе PENDING более 10 минут, помечаются как EXPIRED.
        """
        try:
            from sqlalchemy import update, and_
            from datetime import timedelta
            from ..db.models import WebhookLog, WebhookStatus
            
            async with get_async_db_manager().get_session() as db:
                # Webhook в статусе PENDING старше 10 минут
                stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
                
                result = await db.execute(
                    update(WebhookLog)
                    .where(
                        and_(
                            WebhookLog.status == WebhookStatus.PENDING,
                            WebhookLog.created_at < stale_cutoff
                        )
                    )
                    .values(
                        status=WebhookStatus.EXPIRED,
                        error_message="Webhook expired - stuck in PENDING status",
                        last_attempt_at=datetime.now(timezone.utc)
                    )
                )
                
                await db.commit()
                
                if result.rowcount > 0:
                    logger.warning(
                        f"Marked {result.rowcount} stale webhooks as EXPIRED "
                        f"(stuck in PENDING > 10 minutes)"
                    )
                    
        except Exception as e:
            logger.error(f"Error checking stale webhooks: {str(e)}")

    async def shutdown(self):
        """Остановка планировщика webhook."""
        if self.scheduler and self.started:
            self.scheduler.shutdown(wait=True)
            self.started = False
            logger.info("WebhookScheduler stopped")


# =============================================================================
# GLOBALS & LIFESPAN HELPERS
# =============================================================================

cleanup_scheduler: Optional[CleanupScheduler] = None
webhook_scheduler: Optional[WebhookScheduler] = None


def start_schedulers():
    """
    Запуск всех планировщиков.
    Вызывается при старте приложения (lifespan event).
    """
    global cleanup_scheduler, webhook_scheduler
    
    try:
        # Запуск планировщика очистки
        cleanup_scheduler = CleanupScheduler()
        cleanup_scheduler.start()
        logger.info("Cleanup scheduler initialized")
        
        # Запуск планировщика webhook
        webhook_scheduler = WebhookScheduler()
        webhook_scheduler.start()
        logger.info("Webhook scheduler initialized")
        
        logger.info("All schedulers started successfully")
        
    except Exception as e:
        logger.error(f"Error starting schedulers: {str(e)}")
        raise


async def stop_schedulers():
    """
    Остановка всех планировщиков.
    Вызывается при остановке приложения (lifespan event).
    """
    global cleanup_scheduler, webhook_scheduler
    
    try:
        if cleanup_scheduler:
            await cleanup_scheduler.shutdown()
            logger.info("Cleanup scheduler stopped")
            
        if webhook_scheduler:
            await webhook_scheduler.shutdown()
            logger.info("Webhook scheduler stopped")
            
        logger.info("All schedulers stopped successfully")
        
    except Exception as e:
        logger.error(f"Error stopping schedulers: {str(e)}")


def get_scheduler_status() -> Dict[str, Any]:
    """
    Получение статуса всех планировщиков.
    Полезно для health checks и мониторинга.
    
    Returns:
        Dict с информацией о статусе планировщиков
    """
    global cleanup_scheduler, webhook_scheduler
    
    return {
        "cleanup_scheduler": {
            "running": cleanup_scheduler.started if cleanup_scheduler else False,
            "jobs": len(cleanup_scheduler.scheduler.get_jobs()) if cleanup_scheduler and cleanup_scheduler.scheduler else 0
        },
        "webhook_scheduler": {
            "running": webhook_scheduler.started if webhook_scheduler else False,
            "jobs": len(webhook_scheduler.scheduler.get_jobs()) if webhook_scheduler and webhook_scheduler.scheduler else 0
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def get_scheduled_jobs() -> Dict[str, list]:
    """
    Получение списка всех запланированных задач.
    
    Returns:
        Dict со списками задач для каждого планировщика
    """
    global cleanup_scheduler, webhook_scheduler
    
    cleanup_jobs = []
    webhook_jobs = []
    
    if cleanup_scheduler and cleanup_scheduler.scheduler:
        for job in cleanup_scheduler.scheduler.get_jobs():
            cleanup_jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
    
    if webhook_scheduler and webhook_scheduler.scheduler:
        for job in webhook_scheduler.scheduler.get_jobs():
            webhook_jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
    
    return {
        "cleanup_jobs": cleanup_jobs,
        "webhook_jobs": webhook_jobs,
        "total_jobs": len(cleanup_jobs) + len(webhook_jobs)
    }
