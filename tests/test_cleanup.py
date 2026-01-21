"""
Тесты для cleanup tasks (Phase 5).
Проверка фоновых задач очистки.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta


class TestCleanupTasks:
    """Тесты задач очистки"""

    @pytest.mark.asyncio
    async def test_cleanup_expired_upload_sessions(self):
        """Тест очистки просроченных сессий загрузки"""
        from app.tasks.cleanup import CleanupTasks

        with (
            patch("app.tasks.cleanup.CacheService") as mock_cache,
            patch("app.tasks.cleanup.SessionService") as mock_session,
        ):

            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.redis.scan = AsyncMock(return_value=(0, []))
            mock_session.get_session = AsyncMock(return_value=None)

            result = await CleanupTasks.cleanup_expired_upload_sessions()

            assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_files_from_storage(self):
        """Тест очистки старых файлов из MinIO"""
        from app.tasks.cleanup import CleanupTasks

        with (
            patch("app.tasks.cleanup.StorageService") as mock_storage,
            patch("app.tasks.cleanup.settings") as mock_settings,
        ):

            mock_settings.UPLOAD_EXPIRATION_DAYS = 30

            mock_storage_instance = Mock()
            mock_storage.return_value = mock_storage_instance
            mock_storage_instance.list_files = AsyncMock(return_value=[])

            result = await CleanupTasks.cleanup_old_files_from_storage()

            assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_verification_sessions(self):
        """Тест очистки старых верификационных сессий"""
        from app.tasks.cleanup import CleanupTasks

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("app.tasks.cleanup.get_async_db_manager") as mock_db_manager,
            patch("app.tasks.cleanup.DatabaseService") as mock_db_service,
            patch("app.tasks.cleanup.settings") as mock_settings,
        ):

            mock_settings.UPLOAD_EXPIRATION_DAYS = 30

            mock_db_manager.return_value.get_session.return_value = mock_session_context

            mock_db_service_instance = Mock()
            mock_db_service.return_value = mock_db_service_instance
            mock_db_service_instance.verification_crud.cleanup_old_sessions = AsyncMock(
                return_value=10
            )

            result = await CleanupTasks.cleanup_old_verification_sessions()

            assert result == 10

    @pytest.mark.asyncio
    async def test_cleanup_old_logs(self):
        """Тест очистки старых логов"""
        from app.tasks.cleanup import CleanupTasks

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("app.tasks.cleanup.get_async_db_manager") as mock_db_manager,
            patch("app.tasks.cleanup.DatabaseService") as mock_db_service,
            patch("app.tasks.cleanup.settings") as mock_settings,
        ):

            mock_settings.UPLOAD_EXPIRATION_DAYS = 30

            mock_db_manager.return_value.get_session.return_value = mock_session_context

            mock_db_service_instance = Mock()
            mock_db_service.return_value = mock_db_service_instance
            mock_db_service_instance.audit_crud.cleanup_old_logs = AsyncMock(
                return_value=100
            )

            result = await CleanupTasks.cleanup_old_logs()

            assert result == 100

    @pytest.mark.asyncio
    async def test_get_cleanup_stats(self):
        """Тест получения статистики очистки"""
        from app.tasks.cleanup import CleanupTasks

        result = await CleanupTasks.get_cleanup_stats()

        assert "timestamp" in result
        assert result["message"] is not None

    @pytest.mark.asyncio
    async def test_run_full_cleanup(self):
        """Тест полной очистки"""
        from app.tasks.cleanup import CleanupTasks

        with (
            patch.object(
                CleanupTasks, "cleanup_expired_upload_sessions", new_callable=AsyncMock
            ) as mock_sessions,
            patch.object(
                CleanupTasks, "cleanup_old_files_from_storage", new_callable=AsyncMock
            ) as mock_files,
            patch.object(
                CleanupTasks,
                "cleanup_old_verification_sessions",
                new_callable=AsyncMock,
            ) as mock_verif,
            patch.object(
                CleanupTasks, "cleanup_old_logs", new_callable=AsyncMock
            ) as mock_logs,
        ):

            mock_sessions.return_value = 5
            mock_files.return_value = 10
            mock_verif.return_value = 20
            mock_logs.return_value = 30

            result = await CleanupTasks.run_full_cleanup()

            assert result["expired_upload_sessions"] == 5
            assert result["old_files"] == 10
            assert result["verification_sessions"] == 20
            assert result["old_logs"] == 30


class TestSchedulerLifecycle:
    """Тесты жизненного цикла планировщика"""

    def test_scheduler_import(self):
        """Тест импорта планировщика"""
        from app.tasks import cleanup

        assert cleanup is not None

    def test_cleanup_tasks_class_exists(self):
        """Тест существования класса CleanupTasks"""
        from app.tasks.cleanup import CleanupTasks

        assert hasattr(CleanupTasks, "cleanup_expired_upload_sessions")
        assert hasattr(CleanupTasks, "cleanup_old_files_from_storage")
        assert hasattr(CleanupTasks, "cleanup_old_verification_sessions")
        assert hasattr(CleanupTasks, "cleanup_old_logs")
        assert hasattr(CleanupTasks, "run_full_cleanup")


# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
