"""
Планировщик фоновых задач.
Использует APScheduler для автоматического выполнения задач очистки.
"""

import asyncio
from datetime import datetime
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CleanupScheduler:
    """Планировщик фоновых задач очистки"""
    
    def __init__(self):
        self.scheduler: Optional[BackgroundScheduler] = None
        self.is_running = False
        
    def start(self):
        """Запуск планировщика"""
        if self.is_running:
            logger.warning("Планировщик уже запущен")
            return
            
        try:
            # Создаем планировщик
            self.scheduler = BackgroundScheduler()
            
            # Настраиваем задачи очистки
            
            # 1. Очистка истекших сессий загрузки (каждый час)
            self.scheduler.add_job(
                func=self._cleanup_upload_sessions,
                trigger=IntervalTrigger(hours=1),
                id='cleanup_upload_sessions',
                name='Очистка истекших сессий загрузки',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300  # 5 минут
            )
            
            # 2. Очистка старых файлов (каждые 6 часов)
            self.scheduler.add_job(
                func=self._cleanup_old_files,
                trigger=IntervalTrigger(hours=6),
                id='cleanup_old_files',
                name='Очистка старых файлов',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=600  # 10 минут
            )
            
            # 3. Очистка старых сессий верификации (каждые 12 часов)
            self.scheduler.add_job(
                func=self._cleanup_verification_sessions,
                trigger=IntervalTrigger(hours=12),
                id='cleanup_verification_sessions',
                name='Очистка старых сессий верификации',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=900  # 15 минут
            )
            
            # 4. Очистка старых логов (каждые 24 часа)
            self.scheduler.add_job(
                func=self._cleanup_old_logs,
                trigger=IntervalTrigger(hours=24),
                id='cleanup_old_logs',
                name='Очистка старых логов',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=1800  # 30 минут
            )
            
            # 5. Очистка осиротевших файлов (каждые 24 часа)
            self.scheduler.add_job(
                func=self._cleanup_orphaned_files,
                trigger=IntervalTrigger(hours=24),
                id='cleanup_orphaned_files',
                name='Очистка осиротевших файлов',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=1800  # 30 минут
            )
            
            # 6. Очистка временных файлов (каждые 30 минут)
            self.scheduler.add_job(
                func=self._cleanup_temp_files,
                trigger=IntervalTrigger(minutes=30),
                id='cleanup_temp_files',
                name='Очистка временных файлов',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300  # 5 минут
            )
            
            # 7. Полная очистка системы (каждые 7 дней в воскресенье в 2:00)
            self.scheduler.add_job(
                func=self._run_full_cleanup,
                trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
                id='full_cleanup',
                name='Полная очистка системы',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=3600  # 1 час
            )
            
            # Запускаем планировщик
            self.scheduler.start()
            self.is_running = True
            
            logger.info("Планировщик фоновых задач успешно запущен")
            self._log_scheduled_jobs()
            
        except Exception as e:
            logger.error(f"Ошибка запуска планировщика: {e}")
            raise
    
    def stop(self):
        """Остановка планировщика"""
        if not self.is_running or not self.scheduler:
            logger.warning("Планировщик не запущен")
            return
            
        try:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Планировщик остановлен")
        except Exception as e:
            logger.error(f"Ошибка остановки планировщика: {e}")
    
    def pause(self):
        """Приостановка планировщика"""
        if self.scheduler and self.is_running:
            self.scheduler.pause()
            logger.info("Планировщик приостановлен")
    
    def resume(self):
        """Возобновление планировщика"""
        if self.scheduler:
            self.scheduler.resume()
            logger.info("Планировщик возобновлен")
    
    def get_status(self) -> dict:
        """Получение статуса планировщика"""
        if not self.scheduler:
            return {"status": "not_initialized"}
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time,
                "trigger": str(job.trigger)
            })
        
        return {
            "status": "running" if self.is_running else "stopped",
            "jobs_count": len(jobs),
            "jobs": jobs
        }
    
    def add_custom_job(self, func, trigger, job_id: str, name: str, **kwargs):
        """Добавление пользовательской задачи"""
        if not self.scheduler:
            raise RuntimeError("Планировщик не инициализирован")
        
        try:
            self.scheduler.add_job(
                func=func,
                trigger=trigger,
                id=job_id,
                name=name,
                **kwargs
            )
            logger.info(f"Добавлена пользовательская задача: {name}")
        except Exception as e:
            logger.error(f"Ошибка добавления пользовательской задачи: {e}")
            raise
    
    def remove_job(self, job_id: str):
        """Удаление задачи"""
        if not self.scheduler:
            raise RuntimeError("Планировщик не инициализирован")
        
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Задача удалена: {job_id}")
        except Exception as e:
            logger.error(f"Ошибка удаления задачи: {e}")
            raise
    
    def run_job_now(self, job_id: str):
        """Принудительное выполнение задачи"""
        if not self.scheduler:
            raise RuntimeError("Планировщик не инициализирован")
        
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.utcnow())
                self.scheduler.wakeup()
                logger.info(f"Задача запущена принудительно: {job_id}")
            else:
                logger.warning(f"Задача не найдена: {job_id}")
        except Exception as e:
            logger.error(f"Ошибка принудительного запуска задачи: {e}")
            raise
    
    def _log_scheduled_jobs(self):
        """Логирование запланированных задач"""
        if not self.scheduler:
            return
        
        jobs_info = []
        for job in self.scheduler.get_jobs():
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None
            })
        
        logger.info(f"Запланировано {len(jobs_info)} задач очистки")
        for job_info in jobs_info:
            logger.info(
                f"Задача: {job_info['name']} (ID: {job_info['id']}) - "
                f"следующий запуск: {job_info['next_run']}"
            )
    
    # Методы для выполнения задач очистки
    
    async def _cleanup_upload_sessions(self):
        """Очистка истекших сессий загрузки"""
        from .cleanup import CleanupTasks
        return await asyncio.to_thread(CleanupTasks.cleanup_upload_sessions)
    
    async def _cleanup_old_files(self):
        """Очистка старых файлов"""
        from .cleanup import CleanupTasks
        return await asyncio.to_thread(CleanupTasks.cleanup_old_files)
    
    async def _cleanup_verification_sessions(self):
        """Очистка старых сессий верификации"""
        from .cleanup import CleanupTasks
        return await asyncio.to_thread(CleanupTasks.cleanup_old_verification_sessions)
    
    async def _cleanup_old_logs(self):
        """Очистка старых логов"""
        from .cleanup import CleanupTasks
        return await asyncio.to_thread(CleanupTasks.cleanup_old_logs)
    
    async def _cleanup_orphaned_files(self):
        """Очистка осиротевших файлов"""
        from .cleanup import CleanupTasks
        return await asyncio.to_thread(CleanupTasks.cleanup_orphaned_files)
    
    async def _cleanup_temp_files(self):
        """Очистка временных файлов"""
        from .cleanup import CleanupTasks
        return await asyncio.to_thread(CleanupTasks.cleanup_temp_files)
    
    async def _run_full_cleanup(self):
        """Полная очистка системы"""
        from .cleanup import CleanupTasks
        return await asyncio.to_thread(CleanupTasks.run_full_cleanup)


# Глобальный экземпляр планировщика
_global_scheduler: Optional[CleanupScheduler] = None


def get_scheduler() -> Optional[CleanupScheduler]:
    """Получение глобального экземпляра планировщика"""
    return _global_scheduler


def init_scheduler() -> CleanupScheduler:
    """Инициализация глобального планировщика"""
    global _global_scheduler
    
    if _global_scheduler is None:
        _global_scheduler = CleanupScheduler()
        logger.info("Глобальный планировщик инициализирован")
    
    return _global_scheduler


def start_global_scheduler():
    """Запуск глобального планировщика"""
    global _global_scheduler
    
    if _global_scheduler is None:
        _global_scheduler = CleanupScheduler()
    
    if not _global_scheduler.is_running:
        _global_scheduler.start()


def stop_global_scheduler():
    """Остановка глобального планировщика"""
    global _global_scheduler
    
    if _global_scheduler and _global_scheduler.is_running:
        _global_scheduler.stop()


async def start_async_scheduler():
    """Запуск асинхронного планировщика (для веб-приложений)"""
    try:
        scheduler = AsyncIOScheduler()
        
        # Добавляем задачи очистки
        scheduler.add_job(
            func=_cleanup_upload_sessions_async,
            trigger=IntervalTrigger(hours=1),
            id='cleanup_upload_sessions',
            name='Cleanup expired upload sessions'
        )
        
        scheduler.add_job(
            func=_cleanup_old_files_async,
            trigger=IntervalTrigger(hours=6),
            id='cleanup_old_files',
            name='Cleanup old files'
        )
        
        scheduler.start()
        logger.info("Асинхронный планировщик запущен")
        return scheduler
        
    except Exception as e:
        logger.error(f"Ошибка запуска асинхронного планировщика: {e}")
        raise


async def _cleanup_upload_sessions_async():
    """Асинхронная очистка истекших сессий загрузки"""
    from .cleanup import CleanupTasks
    return await asyncio.to_thread(CleanupTasks.cleanup_upload_sessions)


async def _cleanup_old_files_async():
    """Асинхронная очистка старых файлов"""
    from .cleanup import CleanupTasks
    return await asyncio.to_thread(CleanupTasks.cleanup_old_files)