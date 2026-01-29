"""
Тесты для app/services/session_service.py
Сервис управления сессиями загрузки файлов.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.session_service import SessionService, UploadSession, utcnow


class TestUploadSession:
    """Тесты для модели UploadSession"""

    def test_upload_session_creation(self):
        """Тест создания UploadSession"""
        now = datetime.now(timezone.utc)
        session = UploadSession(
            session_id="test-session-id",
            user_id="user123",
            created_at=now,
            expiration_at=now + timedelta(days=1),
            file_key="uploads/user123/test.jpg",
            file_size=1024,
            file_hash="abc123",
            metadata={"test": "value"},
        )

        assert session.session_id == "test-session-id"
        assert session.user_id == "user123"
        assert session.file_key == "uploads/user123/test.jpg"
        assert session.file_size == 1024
        assert session.file_hash == "abc123"
        assert session.metadata == {"test": "value"}

    def test_upload_session_repr(self):
        """Тест строкового представления"""
        session = UploadSession(
            session_id="test-id",
            user_id="user456",
            created_at=datetime.now(timezone.utc),
            expiration_at=datetime.now(timezone.utc) + timedelta(days=1),
        )

        repr_str = repr(session)
        assert "test-id" in repr_str
        assert "user456" in repr_str

    def test_is_expired_false(self):
        """Тест что не истекшая сессия возвращает False"""
        session = UploadSession(
            session_id="test-id",
            user_id="user123",
            created_at=datetime.now(timezone.utc),
            expiration_at=datetime.now(timezone.utc)
            + timedelta(hours=1),  # Истекает через час
        )

        assert session.is_expired() is False

    def test_is_expired_true(self):
        """Тест что истекшая сессия возвращает True"""
        session = UploadSession(
            session_id="test-id",
            user_id="user123",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expiration_at=datetime.now(timezone.utc)
            - timedelta(hours=1),  # Истекла час назад
        )

        assert session.is_expired() is True

    def test_is_expired_just_expired(self):
        """Тест для только что истекшей сессии"""
        session = UploadSession(
            session_id="test-id",
            user_id="user123",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=61),
            expiration_at=datetime.now(timezone.utc)
            - timedelta(seconds=1),  # Истекла 1 секунду назад
        )

        assert session.is_expired() is True

    def test_from_redis_data_full(self):
        """Тест создания UploadSession из данных Redis (полные данные)"""
        data = {
            "user_id": "user123",
            "created_at": "2024-01-15T10:00:00+00:00",
            "expiration_at": "2024-01-16T10:00:00+00:00",
            "file_key": "uploads/user123/test.jpg",
            "file_size": "2048",
            "file_hash": "def456",
            "metadata": '{"processed": true}',
        }

        session = UploadSession.from_redis_data("session-123", data)

        assert session.session_id == "session-123"
        assert session.user_id == "user123"
        assert session.file_key == "uploads/user123/test.jpg"
        assert session.file_size == 2048
        assert session.file_hash == "def456"
        assert session.metadata == {"processed": True}

    def test_from_redis_data_minimal(self):
        """Тест создания UploadSession из данных Redis (минимальные данные)"""
        data = {
            "user_id": "user456",
            "created_at": "2024-01-15T10:00:00+00:00",
            "expiration_at": "2024-01-16T10:00:00+00:00",
        }

        session = UploadSession.from_redis_data("session-456", data)

        assert session.session_id == "session-456"
        assert session.user_id == "user456"
        assert session.file_key is None
        assert session.file_size is None
        assert session.file_hash is None
        assert session.metadata == {}

    def test_from_redis_data_invalid_metadata(self):
        """Тест обработки некорректного JSON в metadata"""
        data = {
            "user_id": "user789",
            "created_at": "2024-01-15T10:00:00+00:00",
            "expiration_at": "2024-01-16T10:00:00+00:00",
            "metadata": "not valid json",
        }

        # Не должно вызвать исключение, должен использоваться пустой словарь
        session = UploadSession.from_redis_data("session-789", data)

        assert session.metadata == {}

    def test_from_redis_data_empty_metadata(self):
        """Тест обработки пустого metadata"""
        data = {
            "user_id": "user000",
            "created_at": "2024-01-15T10:00:00+00:00",
            "expiration_at": "2024-01-16T10:00:00+00:00",
            "metadata": "",
        }

        session = UploadSession.from_redis_data("session-000", data)

        assert session.metadata == {}


class TestUtcnow:
    """Тесты для функции utcnow"""

    def test_utcnow_returns_datetime(self):
        """Тест что utcnow возвращает datetime"""
        result = utcnow()

        assert isinstance(result, datetime)
        assert result.tzinfo is not None  # Должен быть timezone-aware

    def test_utcnow_is_utc(self):
        """Тест что utcnow возвращает UTC время"""
        result = utcnow()

        assert result.tzinfo == timezone.utc


class TestSessionServiceCreateSession:
    """Тесты для SessionService.create_session"""

    @pytest.mark.asyncio
    async def test_create_session_success(self):
        """Тест успешного создания сессии"""
        user_id = "user123"

        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            MockCacheService.return_value = mock_cache

            session = await SessionService.create_session(user_id)

            assert session is not None
            assert session.session_id is not None
            assert len(session.session_id) == 36  # UUID format
            assert session.user_id == user_id
            assert session.created_at is not None
            assert session.expiration_at is not None
            assert session.expiration_at > session.created_at

    @pytest.mark.asyncio
    async def test_create_session_empty_user_id(self):
        """Тест что create_session вызывает исключение для пустого user_id"""
        with pytest.raises(ValueError, match="user_id is required"):
            await SessionService.create_session("")

    @pytest.mark.asyncio
    async def test_create_session_none_user_id(self):
        """Тест что create_session вызывает исключение для None user_id"""
        with pytest.raises(ValueError, match="user_id is required"):
            await SessionService.create_session(None)

    @pytest.mark.asyncio
    async def test_create_session_calls_cache(self):
        """Тест что create_session сохраняет данные в кэш"""
        user_id = "user_test"

        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            MockCacheService.return_value = mock_cache

            await SessionService.create_session(user_id)

            # Проверяем что CacheService.set был вызван
            mock_cache.set.assert_called_once()

            # Получаем аргументы вызова
            call_args = mock_cache.set.call_args
            key = call_args[0][0]
            data = call_args[0][1]
            expire_seconds = call_args[1]["expire_seconds"]

            assert key.startswith("upload_session:")
            assert data["user_id"] == user_id
            assert "created_at" in data
            assert "expiration_at" in data
            assert expire_seconds > 0


class TestSessionServiceGetSession:
    """Тесты для SessionService.get_session"""

    @pytest.mark.asyncio
    async def test_get_session_found(self):
        """Тест получения существующей сессии"""
        session_id = "test-session-id"
        data = {
            "user_id": "user123",
            "created_at": "2024-01-15T10:00:00+00:00",
            "expiration_at": "2024-01-16T10:00:00+00:00",
        }

        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            mock_cache.get.return_value = data
            MockCacheService.return_value = mock_cache

            session = await SessionService.get_session(session_id)

            assert session is not None
            assert session.session_id == session_id
            assert session.user_id == "user123"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        """Тест получения несуществующей сессии"""
        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            mock_cache.get.return_value = None
            MockCacheService.return_value = mock_cache

            session = await SessionService.get_session("non-existent-id")

            assert session is None

    @pytest.mark.asyncio
    async def test_get_session_empty_session_id(self):
        """Тест что get_session возвращает None для пустого session_id"""
        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            MockCacheService.return_value = mock_cache

            session = await SessionService.get_session("")

            assert session is None
            mock_cache.get.assert_not_called()


class TestSessionServiceAttachFileToSession:
    """Тесты для SessionService.attach_file_to_session"""

    @pytest.mark.asyncio
    async def test_attach_file_success(self):
        """Тест успешного прикрепления файла к сессии"""
        session_id = "test-session-id"
        user_id = "user123"
        file_key = "uploads/user123/test.jpg"
        file_size = 2048
        file_hash = "abc123def456"

        # Мок существующей сессии
        existing_session = UploadSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            expiration_at=datetime.now(timezone.utc) + timedelta(hours=23),
        )

        with (
            patch.object(
                SessionService, "get_session", new_callable=AsyncMock
            ) as mock_get,
            patch("app.services.session_service.CacheService") as MockCacheService,
        ):

            mock_get.return_value = existing_session

            mock_cache = AsyncMock()
            mock_redis = AsyncMock()
            mock_redis.ttl.return_value = 3600  # 1 час TTL
            mock_cache.redis = mock_redis
            MockCacheService.return_value = mock_cache

            result = await SessionService.attach_file_to_session(
                session_id, user_id, file_key, file_size, file_hash
            )

            assert result is not None
            mock_redis.hset.assert_called_once()
            mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_attach_file_session_not_found(self):
        """Тест прикрепления файла к несуществующей сессии"""
        with patch.object(
            SessionService, "get_session", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            with pytest.raises(ValueError, match="Session not found or expired"):
                await SessionService.attach_file_to_session(
                    "non-existent", "user123", "key", 1024, "hash"
                )

    @pytest.mark.asyncio
    async def test_attach_file_wrong_user(self):
        """Тест прикрепления файла к чужой сессии"""
        session = UploadSession(
            session_id="session-id",
            user_id="user_actual",
            created_at=datetime.now(timezone.utc),
            expiration_at=datetime.now(timezone.utc) + timedelta(days=1),
        )

        with patch.object(
            SessionService, "get_session", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = session

            with pytest.raises(
                PermissionError, match="Session does not belong to user"
            ):
                await SessionService.attach_file_to_session(
                    "session-id", "user_wrong", "key", 1024, "hash"
                )


class TestSessionServiceDeleteSession:
    """Тесты для SessionService.delete_session"""

    @pytest.mark.asyncio
    async def test_delete_session_success(self):
        """Тест успешного удаления сессии"""
        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            mock_cache.delete.return_value = True
            MockCacheService.return_value = mock_cache

            result = await SessionService.delete_session("test-session-id")

            assert result is True
            mock_cache.delete.assert_called_once()
            call_args = mock_cache.delete.call_args[0]
            assert call_args[0] == "upload_session:test-session-id"

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self):
        """Тест удаления несуществующей сессии"""
        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            mock_cache.delete.return_value = False
            MockCacheService.return_value = mock_cache

            result = await SessionService.delete_session("non-existent-id")

            assert result is False

    @pytest.mark.asyncio
    async def test_delete_session_empty_id(self):
        """Тест что delete_session возвращает False для пустого ID"""
        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            MockCacheService.return_value = mock_cache

            result = await SessionService.delete_session("")

            assert result is False
            mock_cache.delete.assert_not_called()


class TestSessionServiceValidateSession:
    """Тесты для SessionService.validate_session"""

    @pytest.mark.asyncio
    async def test_validate_session_valid(self):
        """Тест валидной сессии"""
        data = {"user_id": "user123"}

        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            mock_cache.get.return_value = data
            MockCacheService.return_value = mock_cache

            result = await SessionService.validate_session("session-id", "user123")

            assert result is True

    @pytest.mark.asyncio
    async def test_validate_session_wrong_user(self):
        """Тест валидации сессии с неверным user_id"""
        data = {"user_id": "user_actual"}

        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            mock_cache.get.return_value = data
            MockCacheService.return_value = mock_cache

            result = await SessionService.validate_session("session-id", "user_wrong")

            assert result is False

    @pytest.mark.asyncio
    async def test_validate_session_not_found(self):
        """Тест валидации несуществующей сессии"""
        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            mock_cache.get.return_value = None
            MockCacheService.return_value = mock_cache

            result = await SessionService.validate_session("non-existent", "user123")

            assert result is False

    @pytest.mark.asyncio
    async def test_validate_session_empty_session_id(self):
        """Тест валидации с пустым session_id"""
        result = await SessionService.validate_session("", "user123")

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_session_empty_user_id(self):
        """Тест валидации с пустым user_id"""
        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            MockCacheService.return_value = mock_cache

            result = await SessionService.validate_session("session-id", "")

            assert result is False


class TestSessionServiceGetUserSessions:
    """Тесты для SessionService.get_user_sessions"""

    @pytest.mark.asyncio
    async def test_get_user_sessions_empty(self):
        """Тест получения сессий пользователя без сессий"""
        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            mock_redis = AsyncMock()
            # Scan возвращает пустой список
            mock_redis.scan.return_value = (0, [])
            mock_cache.redis = mock_redis
            MockCacheService.return_value = mock_cache

            result = await SessionService.get_user_sessions("user123")

            assert result == []

    @pytest.mark.asyncio
    async def test_get_user_sessions_with_sessions(self):
        """Тест получения сессий пользователя с несколькими сессиями"""
        # Мок данных для скана
        with (
            patch("app.services.session_service.CacheService") as MockCacheService,
            patch.object(
                SessionService, "get_session", new_callable=AsyncMock
            ) as mock_get_session,
        ):

            mock_cache = AsyncMock()
            mock_redis = AsyncMock()
            # Возвращаем ключи сессий
            mock_redis.scan.return_value = (
                0,
                [b"upload_session:session1", b"upload_session:session2"],
            )
            mock_cache.redis = mock_redis
            MockCacheService.return_value = mock_cache

            # Мок сессий пользователя
            session1 = UploadSession(
                session_id="session1",
                user_id="user123",
                created_at=datetime.now(timezone.utc),
                expiration_at=datetime.now(timezone.utc) + timedelta(days=1),
            )
            session2 = UploadSession(
                session_id="session2",
                user_id="user123",
                created_at=datetime.now(timezone.utc),
                expiration_at=datetime.now(timezone.utc) + timedelta(days=1),
            )

            mock_get_session.side_effect = [session1, session2]

            result = await SessionService.get_user_sessions("user123")

            assert len(result) == 2
            assert result[0].session_id == "session1"
            assert result[1].session_id == "session2"

    @pytest.mark.asyncio
    async def test_get_user_sessions_filters_by_user(self):
        """Тест что get_user_sessions фильтрует по user_id"""
        with (
            patch("app.services.session_service.CacheService") as MockCacheService,
            patch.object(
                SessionService, "get_session", new_callable=AsyncMock
            ) as mock_get_session,
        ):

            mock_cache = AsyncMock()
            mock_redis = AsyncMock()
            mock_redis.scan.return_value = (0, [b"upload_session:session1"])
            mock_cache.redis = mock_redis
            MockCacheService.return_value = mock_cache

            # Возвращаем сессию другого пользователя
            session = UploadSession(
                session_id="session1",
                user_id="user_other",
                created_at=datetime.now(timezone.utc),
                expiration_at=datetime.now(timezone.utc) + timedelta(days=1),
            )
            mock_get_session.return_value = session

            result = await SessionService.get_user_sessions("user123")

            assert len(result) == 0  # Сессия отфильтрована


class TestSessionServiceEdgeCases:
    """Тесты для граничных случаев SessionService"""

    @pytest.mark.asyncio
    async def test_create_session_with_special_characters(self):
        """Тест создания сессии с специальными символами в user_id"""
        user_id = "user-with-dashes_and_underscores.123"

        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            MockCacheService.return_value = mock_cache

            session = await SessionService.create_session(user_id)

            assert session.user_id == user_id

    @pytest.mark.asyncio
    async def test_session_expiration_time(self):
        """Тест что срок действия сессии настраивается через settings"""
        from app.config import settings

        with patch("app.services.session_service.CacheService") as MockCacheService:
            mock_cache = AsyncMock()
            MockCacheService.return_value = mock_cache

            session = await SessionService.create_session("user123")

            # Проверяем что expiration_at - created_at = UPLOAD_EXPIRATION_DAYS
            delta = session.expiration_at - session.created_at
            expected_delta = timedelta(days=settings.UPLOAD_EXPIRATION_DAYS)

            # Допускаем небольшую погрешность из-за времени выполнения
            assert abs((delta - expected_delta).total_seconds()) < 5
