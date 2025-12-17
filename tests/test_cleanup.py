"""
Тесты для cleanup tasks (Phase 5).
Проверка фоновых задач очистки данных.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.tasks.cleanup import CleanupTasks
from app.tasks.scheduler import CleanupScheduler
from app.services.session_service import SessionService


class TestCleanupTasks:
    """Тесты задач очистки"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем сессии для тестов
        SessionService._sessions.clear()
    
    def test_cleanup_upload_sessions(self):
        """Тест очистки истекших сессий загрузки"""
        # Создаем обычную сессию
        valid_session = SessionService.create_session("user123")
        
        # Создаем истекшую сессию
        expired_session = SessionService.create_session("user456")
        expired_session.expiration_at = expired_session.expiration_at.replace(year=2020)
        
        # Очищаем истекшие сессии
        deleted_count = CleanupTasks.cleanup_upload_sessions()
        
        # Должна быть удалена только одна истекшая сессия
        assert deleted_count == 1
        
        # Проверяем, что валидная сессия осталась
        assert SessionService.get_session(valid_session.session_id) is not None
        
        # Проверяем, что истекшая сессия удалена
        assert SessionService.get_session(expired_session.session_id) is None
    
    @patch('app.services.storage_service.StorageService')
    def test_cleanup_old_files(self, mock_storage_class):
        """Тест очистки старых файлов"""
        # Мокируем storage service
        mock_storage_instance = Mock()
        mock_storage_class.return_value = mock_storage_instance
        
        # Мокируем список файлов
        mock_storage_instance.list_images.return_value = [
            "uploads/user1/2020_01_01_old_file.jpg",
            "uploads/user1/2023_01_01_new_file.jpg",
            "uploads/user2/2020_06_01_old_file.png"
        ]
        
        # Мокируем информацию о файлах
        mock_storage_instance.get_image_info.side_effect = [
            {"created_at": datetime(2020, 1, 1)},  # Старый файл
            {"created_at": datetime(2023, 1, 1)},  # Новый файл
            {"created_at": datetime(2020, 6, 1)}   # Старый файл
        ]
        
        # Мокируем удаление файлов
        mock_storage_instance.delete_image.return_value = True
        
        # Выполняем очистку
        deleted_count = CleanupTasks.cleanup_old_files()
        
        # Должны быть удалены 2 старых файла
        assert deleted_count == 2
        
        # Проверяем, что методы storage service вызывались
        mock_storage_instance.list_images.assert_called_once()
        mock_storage_instance.get_image_info.assert_called()
        assert mock_storage_instance.delete_image.call_count == 2
    
    @patch('app.db.database.get_db')
    @patch('app.services.database_service.DatabaseService')
    def test_cleanup_old_verification_sessions(self, mock_db_service_class, mock_get_db):
        """Тест очистки старых сессий верификации"""
        # Мокируем БД сервис
        mock_db_service = Mock()
        mock_db_service_class.return_value = mock_db_service
        mock_db_service.verification_crud.cleanup_old_sessions.return_value = 5
        
        # Мокируем контекст БД
        mock_db_context = Mock()
        mock_db_context.__enter__ = Mock(return_value=mock_db_context)
        mock_db_context.__exit__ = Mock(return_value=None)
        mock_get_db.return_value = mock_db_context
        
        # Выполняем очистку
        deleted_count = CleanupTasks.cleanup_old_verification_sessions()
        
        # Должно быть удалено 5 сессий
        assert deleted_count == 5
        
        # Проверяем вызовы
        mock_db_service_class.assert_called_once()
        mock_db_service.verification_crud.cleanup_old_sessions.assert_called_once()
    
    @patch('app.db.database.get_db')
    @patch('app.services.database_service.DatabaseService')
    def test_cleanup_old_logs(self, mock_db_service_class, mock_get_db):
        """Тест очистки старых логов"""
        # Мокируем БД сервис
        mock_db_service = Mock()
        mock_db_service_class.return_value = mock_db_service
        mock_db_service.audit_crud.cleanup_old_logs.return_value = 10
        
        # Мокируем контекст БД
        mock_db_context = Mock()
        mock_db_context.__enter__ = Mock(return_value=mock_db_context)
        mock_db_context.__exit__ = Mock(return_value=None)
        mock_get_db.return_value = mock_db_context
        
        # Выполняем очистку
        deleted_count = CleanupTasks.cleanup_old_logs()
        
        # Должно быть удалено 10 записей лога
        assert deleted_count == 10
        
        # Проверяем вызовы
        mock_db_service_class.assert_called_once()
        mock_db_service.audit_crud.cleanup_old_logs.assert_called_once()
    
    @patch('app.services.storage_service.StorageService')
    @patch('app.db.database.get_db')
    @patch('app.services.database_service.DatabaseService')
    def test_cleanup_orphaned_files(self, mock_db_service_class, mock_get_db, mock_storage_class):
        """Тест очистки осиротевших файлов"""
        # Мокируем storage service
        mock_storage_instance = Mock()
        mock_storage_instance.list_images.return_value = [
            "uploads/user1/file1.jpg",
            "uploads/user1/file2.jpg",
            "uploads/user2/file3.jpg"
        ]
        mock_storage_class.return_value = mock_storage_instance
        
        # Мокируем БД сервис
        mock_db_service = Mock()
        mock_db_service_class.return_value = mock_db_service
        
        # Первые два файла имеют записи в БД, третий - осиротевший
        mock_db_service.reference_crud.get_reference_by_file_key.side_effect = [
            True,   # file1.jpg - существует в БД
            True,   # file2.jpg - существует в БД
            None    # file3.jpg - осиротевший файл
        ]
        
        # Мокируем контекст БД
        mock_db_context = Mock()
        mock_db_context.__enter__ = Mock(return_value=mock_db_context)
        mock_db_context.__exit__ = Mock(return_value=None)
        mock_get_db.return_value = mock_db_context
        
        # Выполняем очистку
        deleted_count = CleanupTasks.cleanup_orphaned_files()
        
        # Должен быть удален 1 осиротевший файл
        assert deleted_count == 1
        
        # Проверяем вызовы
        mock_storage_instance.list_images.assert_called_once()
        assert mock_db_service.reference_crud.get_reference_by_file_key.call_count == 3
        mock_storage_instance.delete_image.assert_called_once_with("uploads/user2/file3.jpg")
    
    @patch('app.services.storage_service.StorageService')
    def test_cleanup_temp_files(self, mock_storage_class):
        """Тест очистки временных файлов"""
        # Мокируем storage service
        mock_storage_instance = Mock()
        mock_storage_instance.list_images.return_value = [
            "temp/old_file1.jpg",
            "temp/new_file2.jpg",
            "temp/old_file3.jpg"
        ]
        mock_storage_class.return_value = mock_storage_instance
        
        # Мокируем информацию о файлах
        mock_storage_instance.get_image_info.side_effect = [
            {"created_at": datetime.utcnow() - timedelta(hours=2)},  # Старый файл
            {"created_at": datetime.utcnow() - timedelta(minutes=30)}, # Новый файл
            {"created_at": datetime.utcnow() - timedelta(hours=3)}   # Старый файл
        ]
        
        # Мокируем удаление файлов
        mock_storage_instance.delete_image.return_value = True
        
        # Выполняем очистку
        deleted_count = CleanupTasks.cleanup_temp_files()
        
        # Должны быть удалены 2 старых временных файла
        assert deleted_count == 2
        
        # Проверяем вызовы
        mock_storage_instance.list_images.assert_called_once()
        assert mock_storage_instance.get_image_info.call_count == 3
        assert mock_storage_instance.delete_image.call_count == 2
    
    @patch('app.tasks.cleanup.CleanupTasks.cleanup_upload_sessions')
    @patch('app.tasks.cleanup.CleanupTasks.cleanup_old_files')
    @patch('app.tasks.cleanup.CleanupTasks.cleanup_old_verification_sessions')
    @patch('app.tasks.cleanup.CleanupTasks.cleanup_old_logs')
    @patch('app.tasks.cleanup.CleanupTasks.cleanup_orphaned_files')
    @patch('app.tasks.cleanup.CleanupTasks.cleanup_temp_files')
    def test_run_full_cleanup(
        self, 
        mock_temp_cleanup,
        mock_orphaned_cleanup,
        mock_logs_cleanup,
        mock_verification_cleanup,
        mock_files_cleanup,
        mock_sessions_cleanup
    ):
        """Тест полной очистки системы"""
        # Настраиваем возвращаемые значения
        mock_sessions_cleanup.return_value = 2
        mock_files_cleanup.return_value = 5
        mock_verification_cleanup.return_value = 3
        mock_logs_cleanup.return_value = 8
        mock_orphaned_cleanup.return_value = 1
        mock_temp_cleanup.return_value = 4
        
        # Выполняем полную очистку
        results = CleanupTasks.run_full_cleanup()
        
        # Проверяем результаты
        assert results["upload_sessions"] == 2
        assert results["old_files"] == 5
        assert results["verification_sessions"] == 3
        assert results["old_logs"] == 8
        assert results["orphaned_files"] == 1
        assert results["temp_files"] == 4
        
        # Общее количество удаленных элементов
        total_deleted = sum(results.values())
        assert total_deleted == 23
        
        # Проверяем, что все методы были вызваны
        mock_sessions_cleanup.assert_called_once()
        mock_files_cleanup.assert_called_once()
        mock_verification_cleanup.assert_called_once()
        mock_logs_cleanup.assert_called_once()
        mock_orphaned_cleanup.assert_called_once()
        mock_temp_cleanup.assert_called_once()
    
    @patch('app.services.storage_service.StorageService')
    def test_get_cleanup_stats(self, mock_storage_class):
        """Тест получения статистики очистки"""
        # Создаем тестовые сессии
        SessionService.create_session("user1")
        SessionService.create_session("user2")
        
        # Мокируем storage service
        mock_storage_instance = Mock()
        mock_storage_instance.get_storage_stats.return_value = {
            "bucket_name": "face-recognition",
            "total_objects": 100,
            "status": "accessible"
        }
        mock_storage_class.return_value = mock_storage_instance
        
        # Получаем статистику
        stats = CleanupTasks.get_cleanup_stats()
        
        # Проверяем статистику
        assert "timestamp" in stats
        assert "upload_sessions" in stats
        assert "storage_stats" in stats
        assert "cleanup_tasks_last_run" in stats
        
        assert stats["upload_sessions"] == 2  # 2 активные сессии
        
        # Проверяем статистику хранилища
        storage_stats = stats["storage_stats"]
        assert storage_stats["bucket_name"] == "face-recognition"
        assert storage_stats["total_objects"] == 100


class TestCleanupScheduler:
    """Тесты планировщика очистки"""
    
    def test_scheduler_initialization(self):
        """Тест инициализации планировщика"""
        scheduler = CleanupScheduler()
        
        assert scheduler.scheduler is None
        assert scheduler.is_running == False
    
    @patch('app.tasks.scheduler.BackgroundScheduler')
    def test_start_scheduler(self, mock_scheduler_class):
        """Тест запуска планировщика"""
        # Мокируем APScheduler
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = CleanupScheduler()
        scheduler.start()
        
        assert scheduler.is_running == True
        assert scheduler.scheduler is not None
        
        # Проверяем, что планировщик был запущен
        mock_scheduler.start.assert_called_once()
        
        # Проверяем, что были добавлены задачи
        assert mock_scheduler.add_job.call_count >= 6  # Минимум 6 задач
    
    @patch('app.tasks.scheduler.BackgroundScheduler')
    def test_stop_scheduler(self, mock_scheduler_class):
        """Тест остановки планировщика"""
        # Мокируем планировщик
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = CleanupScheduler()
        scheduler.start()
        
        # Останавливаем планировщик
        scheduler.stop()
        
        assert scheduler.is_running == False
        
        # Проверяем, что планировщик был остановлен
        mock_scheduler.shutdown.assert_called_once()
    
    @patch('app.tasks.scheduler.BackgroundScheduler')
    def test_pause_resume_scheduler(self, mock_scheduler_class):
        """Тест приостановки и возобновления планировщика"""
        # Мокируем планировщик
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = CleanupScheduler()
        scheduler.start()
        
        # Приостанавливаем
        scheduler.pause()
        mock_scheduler.pause.assert_called_once()
        
        # Возобновляем
        scheduler.resume()
        mock_scheduler.resume.assert_called_once()
    
    @patch('app.tasks.scheduler.BackgroundScheduler')
    def test_get_status(self, mock_scheduler_class):
        """Тест получения статуса планировщика"""
        # Мокируем планировщик и задачи
        mock_scheduler = Mock()
        mock_scheduler.get_jobs.return_value = [
            Mock(id="job1", name="Job 1", next_run_time=datetime.utcnow()),
            Mock(id="job2", name="Job 2", next_run_time=None)
        ]
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = CleanupScheduler()
        scheduler.start()
        
        status = scheduler.get_status()
        
        assert status["status"] == "running"
        assert status["jobs_count"] == 2
        assert len(status["jobs"]) == 2
    
    def test_scheduler_not_initialized(self):
        """Тест работы с неинициализированным планировщиком"""
        scheduler = CleanupScheduler()
        
        # Статус неинициализированного планировщика
        status = scheduler.get_status()
        assert status["status"] == "not_initialized"
        
        # Попытка добавить задачу должна вызвать ошибку
        with pytest.raises(RuntimeError, match="Планировщик не инициализирован"):
            scheduler.add_job(lambda: None, None, "test_job", "Test Job")
        
        # Попытка удалить задачу должна вызвать ошибку
        with pytest.raises(RuntimeError, match="Планировщик не инициализирован"):
            scheduler.remove_job("test_job")
        
        # Попытка запустить задачу должна вызвать ошибку
        with pytest.raises(RuntimeError, match="Планировщик не инициализирован"):
            scheduler.run_job_now("test_job")


class TestGlobalScheduler:
    """Тесты глобального планировщика"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Сбрасываем глобальный планировщик
        from app.tasks.scheduler import _global_scheduler
        _global_scheduler = None
    
    @patch('app.tasks.scheduler.CleanupScheduler')
    def test_init_global_scheduler(self, mock_scheduler_class):
        """Тест инициализации глобального планировщика"""
        from app.tasks.scheduler import init_scheduler, get_scheduler
        
        # Инициализируем планировщик
        scheduler = init_scheduler()
        
        assert scheduler is not None
        assert get_scheduler() is scheduler
        
        # Повторная инициализация должна вернуть тот же экземпляр
        scheduler2 = init_scheduler()
        assert scheduler2 is scheduler
    
    @patch('app.tasks.scheduler.CleanupScheduler')
    def test_start_stop_global_scheduler(self, mock_scheduler_class):
        """Тест запуска и остановки глобального планировщика"""
        from app.tasks.scheduler import start_global_scheduler, stop_global_scheduler, get_scheduler
        
        # Запускаем глобальный планировщик
        start_global_scheduler()
        
        scheduler = get_scheduler()
        assert scheduler is not None
        assert scheduler.is_running == True
        
        # Останавливаем планировщик
        stop_global_scheduler()
        
        assert scheduler.is_running == False


# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])