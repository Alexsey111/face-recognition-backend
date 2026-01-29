"""
Тесты для storage service (Phase 5).
Проверка работы с MinIO S3 хранилищем.
"""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from PIL import Image


class TestFileUtils:
    """Тесты утилит для работы с файлами"""

    def test_generate_upload_key(self):
        """Тест генерации ключа загрузки"""
        from app.utils.file_utils import FileUtils

        key = FileUtils.generate_upload_key("user123", "photo.jpg")
        assert key.startswith("uploads/user123/")
        assert "photo.jpg" in key

    def test_is_valid_image_format(self):
        """Тест валидации формата изображения"""
        from app.utils.file_utils import FileUtils

        assert FileUtils.is_valid_image_format("photo.jpg") == True
        assert FileUtils.is_valid_image_format("image.png") == True
        assert FileUtils.is_valid_image_format("document.pdf") == False

    def test_get_file_size_mb(self):
        """Тест получения размера файла в МБ"""
        from app.utils.file_utils import FileUtils

        test_data = b"test data" * 1000
        size_mb = FileUtils.get_file_size_mb(test_data)
        assert size_mb > 0.008 and size_mb < 0.012

    def test_calculate_file_hash(self):
        """Тест вычисления хеша файла"""
        from app.utils.file_utils import FileUtils

        test_data = b"test file content"
        hash1 = FileUtils.calculate_file_hash(test_data)
        hash2 = FileUtils.calculate_file_hash(test_data)

        assert len(hash1) == 64
        assert hash1 == hash2


class TestImageValidator:
    """Тесты валидатора изображений"""

    def test_get_image_info(self):
        """Тест получения информации об изображении"""
        from app.utils.file_utils import FileUtils, ImageValidator

        img = Image.new("RGB", (300, 200), color="purple")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_content = img_bytes.getvalue()

        info = ImageValidator.get_info(img_content, "test.jpg")

        assert info["filename"] == "test.jpg"
        assert info["width"] == 300
        assert info["height"] == 200
        assert info["format"] == "JPEG"


class TestStorageService:
    """Тесты сервиса хранилища"""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Тест проверки состояния хранилища"""
        from app.services.storage_service import StorageService

        with patch("app.services.storage_service.boto3") as mock_boto:
            mock_s3 = Mock()
            mock_boto.client.return_value = mock_s3
            mock_s3.head_bucket.return_value = {}

            storage = StorageService()
            result = await storage.health_check()

            assert result == True
            mock_s3.head_bucket.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_image(self):
        """Тест загрузки изображения"""
        from app.services.storage_service import StorageService

        with patch("app.services.storage_service.boto3") as mock_boto:
            mock_s3 = Mock()
            mock_boto.client.return_value = mock_s3
            mock_s3.put_object.return_value = {}

            storage = StorageService()

            img = Image.new("RGB", (100, 100), color="red")
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            test_data = img_bytes.getvalue()

            result = await storage.upload_image(test_data)

            assert "key" in result
            assert "file_url" in result
            assert "content_type" in result
            assert result["file_size"] == len(test_data)
            mock_s3.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_image(self):
        """Тест скачивания изображения"""
        from app.services.storage_service import StorageService

        with patch("app.services.storage_service.boto3") as mock_boto:
            mock_s3 = Mock()
            mock_boto.client.return_value = mock_s3

            mock_response = {"Body": Mock()}
            mock_response["Body"].read.return_value = b"downloaded image data"
            mock_s3.get_object.return_value = mock_response

            storage = StorageService()
            result = await storage.download_image("test_key.jpg")

            assert result == b"downloaded image data"
            mock_s3.get_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_image(self):
        """Тест удаления изображения"""
        from app.services.storage_service import StorageService

        with patch("app.services.storage_service.boto3") as mock_boto:
            mock_s3 = Mock()
            mock_boto.client.return_value = mock_s3
            mock_s3.delete_object.return_value = {}

            storage = StorageService()
            result = await storage.delete_image("test_key.jpg")

            assert result == True
            mock_s3.delete_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_files(self):
        """Тест получения списка файлов"""
        from app.services.storage_service import StorageService

        with patch("app.services.storage_service.boto3") as mock_boto:
            mock_s3 = Mock()
            mock_boto.client.return_value = mock_s3

            mock_s3.list_objects_v2.return_value = {
                "Contents": [
                    {
                        "Key": "uploads/user1/file1.jpg",
                        "Size": 1024,
                        "LastModified": "2024-01-01",
                    },
                    {
                        "Key": "uploads/user2/file2.png",
                        "Size": 2048,
                        "LastModified": "2024-01-02",
                    },
                ]
            }

            storage = StorageService()
            files = await storage.list_files(prefix="uploads/", limit=100)

            assert len(files) == 2
            assert files[0]["key"] == "uploads/user1/file1.jpg"


class TestSessionService:
    """Тесты сервиса сессий"""

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Тест создания сессии"""
        from app.services.session_service import SessionService

        mock_cache_set = AsyncMock(return_value=True)

        with patch("app.services.session_service.CacheService") as mock_cache:
            mock_cache.return_value.set = mock_cache_set

            session = await SessionService.create_session("user123")

            assert session.user_id == "user123"
            assert session.session_id is not None
            assert len(session.session_id) == 36
            assert session.file_key is None
            mock_cache_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Тест получения сессии"""
        from app.services.session_service import SessionService

        mock_cache_get = AsyncMock(
            return_value={
                "user_id": "user123",
                "created_at": "2024-01-01T00:00:00",
                "expiration_at": "2025-02-01T00:00:00",
                "file_key": None,
                "file_size": None,
                "file_hash": None,
                "metadata": "{}",
            }
        )

        with patch("app.services.session_service.CacheService") as mock_cache:
            mock_cache.return_value.get = mock_cache_get

            session = await SessionService.get_session("test-uuid")

            assert session is not None
            assert session.user_id == "user123"

    @pytest.mark.asyncio
    async def test_validate_session(self):
        """Тест валидации сессии"""
        from app.services.session_service import SessionService

        mock_cache_get = AsyncMock(
            return_value={
                "user_id": "user123",
                "created_at": "2024-01-01T00:00:00",
                "expiration_at": "2025-02-01T00:00:00",
            }
        )

        with patch("app.services.session_service.CacheService") as mock_cache:
            mock_cache.return_value.get = mock_cache_get

            result = await SessionService.validate_session("test-uuid", "user123")
            assert result == True

            result = await SessionService.validate_session("test-uuid", "user456")
            assert result == False

    @pytest.mark.asyncio
    async def test_delete_session(self):
        """Тест удаления сессии"""
        from app.services.session_service import SessionService

        mock_cache_delete = AsyncMock(return_value=True)

        with patch("app.services.session_service.CacheService") as mock_cache:
            mock_cache.return_value.delete = mock_cache_delete

            result = await SessionService.delete_session("test-uuid")
            assert result == True


# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
