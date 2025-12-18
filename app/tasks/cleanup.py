"""
Фоновые задачи очистки.
Удаление истекших сессий, старых файлов и данных.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
from ..config import settings
from ..services.storage_service import StorageService
from ..services.session_service import SessionService
from ..services.database_service import DatabaseService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CleanupTasks:
    """Фоновые задачи очистки"""
    
    @staticmethod
    def cleanup_upload_sessions() -> int:
        """
        Очистка истекших сессий загрузки
        
        Returns:
            int: Количество удаленных сессий
        """
        try:
            deleted = SessionService.cleanup_expired_sessions()
            logger.info(f"Задача очистки: удалено {deleted} истекших сессий загрузки")
            return deleted
        except Exception as e:
            logger.error(f"Ошибка в cleanup_upload_sessions: {e}")
            return 0
    
    @staticmethod
    async def cleanup_old_files() -> int:
        """
        Очистка старых файлов из MinIO
        
        Returns:
            int: Количество удаленных файлов
        """
        try:
            storage = StorageService()
            cutoff_date = datetime.utcnow() - timedelta(
                days=settings.UPLOAD_EXPIRATION_DAYS
            )
            
            # Получаем список старых файлов
            files = await storage.list_images(prefix="uploads/", max_keys=1000)
            old_files = []
            
            for file_key in files:
                try:
                    file_info = await storage.get_image_info(file_key)
                    
                    if file_info and file_info.get('created_at'):
                        created_at = file_info['created_at']
                        
                        # Преобразуем строку в datetime если нужно
                        if isinstance(created_at, str):
                            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        # Убираем timezone info для корректного сравнения
                        created_at_naive = created_at.replace(tzinfo=None)
                        
                        # Файл считается старым, если его дата создания раньше cutoff_date
                        if created_at_naive < cutoff_date:
                            old_files.append(file_key)
                    else:
                        # Если не удалось получить информацию о файле, пропускаем его
                        logger.warning(f"Не удалось получить информацию о файле {file_key}, пропускаем")
                except Exception as e:
                    logger.warning(f"Ошибка проверки файла {file_key}: {e}")
                    continue  # Важно: пропускаем файл при ошибке
            
            # Удаляем старые файлы
            deleted_count = 0
            for file_key in old_files:
                try:
                    await storage.delete_image(file_key)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Ошибка удаления файла {file_key}: {e}")
            
            logger.info(f"Задача очистки: удалено {deleted_count} старых файлов")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Ошибка в cleanup_old_files: {e}")
            return 0
    
    @staticmethod
    def cleanup_old_verification_sessions() -> int:
        """
        Очистка старых сессий верификации из БД
        
        Returns:
            int: Количество удаленных сессий
        """
        try:
            # Получаем экземпляр БД
            from ..db.database import get_db
            
            # Временное подключение к БД для очистки
            with next(get_db()) as db:
                db_service = DatabaseService(db)
                deleted = db_service.verification_crud.cleanup_old_sessions(
                    db, days=settings.UPLOAD_EXPIRATION_DAYS
                )
                
            logger.info(f"Задача очистки: удалено {deleted} старых сессий верификации")
            return deleted
            
        except Exception as e:
            logger.error(f"Ошибка в cleanup_old_verification_sessions: {e}")
            return 0
    
    @staticmethod
    def cleanup_old_logs() -> int:
        """
        Очистка старых логов аудита
        
        Returns:
            int: Количество удаленных записей
        """
        try:
            from ..db.database import get_db
            
            with next(get_db()) as db:
                db_service = DatabaseService(db)
                deleted = db_service.audit_crud.cleanup_old_logs(
                    db, days=settings.UPLOAD_EXPIRATION_DAYS
                )
                
            logger.info(f"Задача очистки: удалено {deleted} старых записей лога")
            return deleted
            
        except Exception as e:
            logger.error(f"Ошибка в cleanup_old_logs: {e}")
            return 0
    
    @staticmethod
    async def cleanup_orphaned_files() -> int:
        """
        Очистка осиротевших файлов (файлы без соответствующих записей в БД)
        
        Returns:
            int: Количество удаленных файлов
        """
        try:
            storage = StorageService()
            deleted_count = 0
            
            # Получаем все файлы из хранилища
            all_files = await storage.list_images(prefix="uploads/", max_keys=1000)
            
            from ..db.database import get_db
            with next(get_db()) as db:
                db_service = DatabaseService(db)
                
                # Проверяем каждый файл
                for file_key in all_files:
                    try:
                        # Проверяем, существует ли запись в БД
                        file_exists_in_db = db_service.reference_crud.get_reference_by_file_key(
                            db, file_key
                        )
                        
                        if not file_exists_in_db:
                            # Файл не имеет соответствующей записи в БД
                            await storage.delete_image(file_key)
                            deleted_count += 1
                            logger.info(f"Удален осиротевший файл: {file_key}")
                            
                    except Exception as e:
                        logger.warning(f"Ошибка проверки файла {file_key}: {e}")
            
            logger.info(f"Задача очистки: удалено {deleted_count} осиротевших файлов")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Ошибка в cleanup_orphaned_files: {e}")
            return 0
    
    @staticmethod
    async def cleanup_temp_files() -> int:
        """
        Очистка временных файлов (файлы с префиксом temp/)
        
        Returns:
            int: Количество удаленных файлов
        """
        try:
            storage = StorageService()
            
            # Получаем список временных файлов
            temp_files = await storage.list_images(prefix="temp/", max_keys=1000)
            deleted_count = 0
            
            # Удаляем временные файлы старше 1 часа
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            for file_key in temp_files:
                try:
                    file_info = await storage.get_image_info(file_key)
                    if file_info and file_info.get('created_at'):
                        created_at = file_info['created_at']
                        if isinstance(created_at, str):
                            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        
                        if created_at.replace(tzinfo=None) < cutoff_time:
                            await storage.delete_image(file_key)
                            deleted_count += 1
                            
                except Exception as e:
                    logger.warning(f"Ошибка удаления временного файла {file_key}: {e}")
            
            logger.info(f"Задача очистки: удалено {deleted_count} временных файлов")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Ошибка в cleanup_temp_files: {e}")
            return 0
    
    @staticmethod
    def get_cleanup_stats() -> Dict[str, Any]:
        """
        Получение статистики очистки
        
        Returns:
            Dict[str, Any]: Статистика выполненных операций
        """
        try:
            stats = {
                "timestamp": datetime.utcnow(),
                "upload_sessions": SessionService.get_active_sessions_count(),
                "storage_stats": {},
                "cleanup_tasks_last_run": {},
            }
            
            # Статистика хранилища
            try:
                storage = StorageService()
                stats["storage_stats"] = storage.get_storage_stats()
            except Exception as e:
                logger.warning(f"Ошибка получения статистики хранилища: {e}")
                stats["storage_stats"] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики очистки: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def run_full_cleanup() -> Dict[str, int]:
        """
        Выполнение полной очистки всех данных
        
        Returns:
            Dict[str, int]: Статистика удаленных элементов
        """
        logger.info("Начинается полная очистка системы")
        
        results = {
            "upload_sessions": CleanupTasks.cleanup_upload_sessions(),
            "old_files": await CleanupTasks.cleanup_old_files(),
            "verification_sessions": CleanupTasks.cleanup_old_verification_sessions(),
            "old_logs": CleanupTasks.cleanup_old_logs(),
            "orphaned_files": await CleanupTasks.cleanup_orphaned_files(),
            "temp_files": await CleanupTasks.cleanup_temp_files(),
        }
        
        total_deleted = sum(results.values())
        logger.info(
            f"Полная очистка завершена. Удалено элементов: {total_deleted}. "
            f"Детали: {results}"
        )
        
        return results


# Глобальный экземпляр планировщика
_scheduler = None


def start_cleanup_scheduler():
    """Запуск планировщика фоновых задач"""
    global _scheduler
    
    if _scheduler is None:
        try:
            _scheduler = CleanupScheduler()
            _scheduler.start()
            logger.info("Планировщик фоновых задач запущен")
        except Exception as e:
            logger.error(f"Ошибка запуска планировщика: {e}")
            raise


def stop_cleanup_scheduler():
    """Остановка планировщика фоновых задач"""
    global _scheduler
    
    if _scheduler:
        try:
            _scheduler.stop()
            _scheduler = None
            logger.info("Планировщик фоновых задач остановлен")
        except Exception as e:
            logger.error(f"Ошибка остановки планировщика: {e}")


def get_cleanup_scheduler():
    """Получение экземпляра планировщика"""
    return _scheduler