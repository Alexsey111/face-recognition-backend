"""
Тесты для storage service (Phase 5).
Проверка работы с MinIO S3 хранилищем.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from io import BytesIO
from PIL import Image

from app.services.storage_service import StorageService
from app.services.session_service import SessionService
from app.utils.file_utils import FileUtils, ImageValidator


class TestFileUtils:
    """Тесты утилит для работы с файлами"""
    
    def test_generate_file_key(self):
        """Тест генерации ключа файла"""
        key = FileUtils.generate_file_key("user123", "photo.jpg")
        assert key.startswith("uploads/user123/")
        assert "photo.jpg" in key
        
    def test_is_valid_image_format(self):
        """Тест валидации формата изображения"""
        assert FileUtils.is_valid_image_format("photo.jpg") == True
        assert FileUtils.is_valid_image_format("image.png") == True
        assert FileUtils.is_valid_image_format("image.heic") == True
        assert FileUtils.is_valid_image_format("document.pdf") == False
        
    def test_get_file_size_mb(self):
        """Тест получения размера файла в МБ"""
        test_data = b"test data" * 1000  # ~10KB
        size_mb = FileUtils.get_file_size_mb(test_data)
        assert size_mb > 0.009 and size_mb < 0.011  # ~0.01MB
        
    def test_calculate_file_hash(self):
        """Тест вычисления хеша файла"""
        test_data = b"test file content"
        hash1 = FileUtils.calculate_file_hash(test_data)
        hash2 = FileUtils.calculate_file_hash(test_data)
        
        assert len(hash1) == 64  # SHA256 hex length
        assert hash1 == hash2
        
        # Разные файлы - разные хеши
        different_data = b"different content"
        hash3 = FileUtils.calculate_file_hash(different_data)
        assert hash1 != hash3


class TestSessionService:
    """Тесты сервиса управления сессиями"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем хранилище сессий для тестов
        SessionService._sessions.clear()
    
    def test_create_session(self):
        """Тест создания сессии"""
        session = SessionService.create_session("user123")
        
        assert session.user_id == "user123"
        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID length
        assert not session.is_expired()
        assert session.file_key is None
        assert session.file_size is None
        
    def test_validate_session(self):
        """Тест валидации сессии"""
        session = SessionService.create_session("user123")
        
        # Валидная сессия
        assert SessionService.validate_session(session.session_id, "user123") == True
        
        # Неправильный пользователь
        assert SessionService.validate_session(session.session_id, "user456") == False
        
        # Несуществующая сессия
        assert SessionService.validate_session("nonexistent", "user123") == False
        
    def test_update_session(self):
        """Тест обновления сессии"""
        session = SessionService.create_session("user123")
        original_file_key = session.file_key
        
        updated_session = SessionService.update_session(
            session.session_id,
            file_key="new_file_key.jpg",
            file_size=1.5
        )
        
        assert updated_session is not None
        assert updated_session.file_key == "new_file_key.jpg"
        assert updated_session.file_size == 1.5
        assert original_file_key is None
        
    def test_delete_session(self):
        """Тест удаления сессии"""
        session = SessionService.create_session("user123")
        session_id = session.session_id
        
        # Проверяем, что сессия существует
        assert SessionService.get_session(session_id) is not None
        
        # Удаляем сессию
        result = SessionService.delete_session(session_id)
        assert result == True
        
        # Проверяем, что сессия удалена
        assert SessionService.get_session(session_id) is None
        
    def test_cleanup_expired_sessions(self):
        """Тест очистки истекших сессий"""
        # Создаем сессию
        session = SessionService.create_session("user123")
        
        # Истекаем сессию вручную
        session.expiration_at = session.expiration_at.replace(year=2020)
        
        # Очищаем истекшие сессии
        deleted_count = SessionService.cleanup_expired_sessions()
        assert deleted_count == 1
        
        # Сессия должна быть удалена
        assert SessionService.get_session(session.session_id) is None
        
    def test_get_user_sessions(self):
        """Тест получения сессий пользователя"""
        # Создаем сессии для разных пользователей
        session1 = SessionService.create_session("user123")
        session2 = SessionService.create_session("user456")
        session3 = SessionService.create_session("user123")
        
        # Получаем сессии пользователя user123
        user_sessions = SessionService.get_user_sessions("user123")
        assert len(user_sessions) == 2
        
        # Все сессии должны принадлежать user123
        for session in user_sessions:
            assert session.user_id == "user123"


class TestImageValidator:
    """Тесты валидатора изображений"""
    
    def test_validate_valid_image(self):
        """Тест валидации корректного изображения"""
        # Создаем тестовое изображение
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_content = img_bytes.getvalue()
        
        is_valid, error = ImageValidator.validate_image(img_content, "test.jpg")
        assert is_valid == True
        assert error == ""
        
    def test_validate_small_image(self):
        """Тест валидации слишком маленького изображения"""
        img = Image.new('RGB', (10, 10), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_content = img_bytes.getvalue()
        
        is_valid, error = ImageValidator.validate_image(img_content, "test.jpg")
        assert is_valid == False
        assert "слишком маленькое" in error.lower()
        
    def test_validate_invalid_format(self):
        """Тест валидации неподдерживаемого формата"""
        fake_content = b"this is not an image"
        
        is_valid, error = ImageValidator.validate_image(fake_content, "test.pdf")
        assert is_valid == False
        assert "недействительный формат" in error.lower()
        
    def test_get_image_info(self):
        """Тест получения информации об изображении"""
        img = Image.new('RGB', (200, 150), color='blue')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_content = img_bytes.getvalue()
        
        info = ImageValidator.get_image_info(img_content, "test.jpg")
        
        assert info["filename"] == "test.jpg"
        assert info["width"] == 200
        assert info["height"] == 150
        assert info["size_mb"] > 0
        assert info["format"] == "JPEG"
        assert info["is_valid"] == True
        assert "file_hash" in info
        
    def test_convert_image_to_jpg(self):
        """Тест конвертации изображения в JPG"""
        # Создаем PNG изображение
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        png_content = img_bytes.getvalue()
        
        # Конвертируем в JPG
        jpg_content, new_filename = FileUtils.convert_image_to_jpg(
            png_content, "test.png"
        )
        
        assert new_filename == "test.jpg"
        assert len(jpg_content) > 0
        
        # Проверяем, что это действительно JPG
        converted_img = Image.open(BytesIO(jpg_content))
        assert converted_img.format == "JPEG"
        
    def test_resize_image(self):
        """Тест изменения размера изображения"""
        # Создаем большое изображение
        img = Image.new('RGB', (2000, 2000), color='green')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        large_content = img_bytes.getvalue()
        
        # Изменяем размер
        resized_content = FileUtils.resize_image(large_content, 1024, 1024)
        
        # Проверяем, что размер уменьшился
        assert len(resized_content) < len(large_content)
        
        # Проверяем новые размеры
        resized_img = Image.open(BytesIO(resized_content))
        assert resized_img.width <= 1024
        assert resized_img.height <= 1024


class TestStorageService:
    """Тесты сервиса хранилища (базовые тесты)"""
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Тест проверки состояния хранилища"""
        with patch('app.services.storage_service.boto3.client') as mock_client:
            mock_s3 = Mock()
            mock_client.return_value = mock_s3
            mock_s3.head_bucket.return_value = {}
            
            storage = StorageService()
            result = await storage.health_check()
            
            assert result == True
            mock_s3.head_bucket.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_upload_image(self):
        """Тест загрузки изображения"""
        with patch('app.services.storage_service.boto3.client') as mock_client:
            mock_s3 = Mock()
            mock_client.return_value = mock_s3
            mock_s3.put_object.return_value = {}
            
            storage = StorageService()
            test_data = b"test image data"
            
            result = await storage.upload_image(test_data)
            
            assert "image_id" in result
            assert "file_url" in result
            assert result["file_size"] == len(test_data)
            mock_s3.put_object.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_download_image(self):
        """Тест скачивания изображения"""
        with patch('app.services.storage_service.boto3.client') as mock_client:
            mock_s3 = Mock()
            mock_client.return_value = mock_s3
            
            # Мокируем ответ
            mock_response = {
                'Body': Mock()
            }
            mock_response['Body'].read.return_value = b"downloaded image data"
            mock_s3.get_object.return_value = mock_response
            
            storage = StorageService()
            result = await storage.download_image("test_key.jpg")
            
            assert result == b"downloaded image data"
            mock_s3.get_object.assert_called_once_with(
                Bucket=storage.bucket_name, 
                Key="test_key.jpg"
            )
            
    @pytest.mark.asyncio
    async def test_delete_image(self):
        """Тест удаления изображения"""
        with patch('app.services.storage_service.boto3.client') as mock_client:
            mock_s3 = Mock()
            mock_client.return_value = mock_s3
            mock_s3.delete_object.return_value = {}
            
            storage = StorageService()
            result = await storage.delete_image("test_key.jpg")
            
            assert result == True
            mock_s3.delete_object.assert_called_once_with(
                Bucket=storage.bucket_name,
                Key="test_key.jpg"
            )


class TestIntegration:
    """Интеграционные тесты"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем хранилище сессий
        SessionService._sessions.clear()
    
    @pytest.mark.asyncio
    async def test_full_upload_workflow(self):
        """Тест полного workflow загрузки"""
        user_id = "test_user_123"
        
        # 1. Создаем сессию
        session = SessionService.create_session(user_id)
        assert session is not None
        
        # 2. Валидируем сессию
        assert SessionService.validate_session(session.session_id, user_id) == True
        
        # 3. Создаем тестовое изображение
        img = Image.new('RGB', (300, 300), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        image_content = img_bytes.getvalue()
        
        # 4. Валидируем изображение
        is_valid, error = ImageValidator.validate_image(image_content, "test.jpg")
        assert is_valid == True
        
        # 5. Генерируем ключ файла
        file_key = FileUtils.generate_file_key(user_id, "test.jpg")
        assert file_key.startswith(f"uploads/{user_id}/")
        
        # 6. Вычисляем хеш
        file_hash = FileUtils.calculate_file_hash(image_content)
        assert len(file_hash) == 64
        
        # 7. Обновляем сессию с информацией о файле
        updated_session = SessionService.update_session(
            session.session_id,
            file_key=file_key,
            file_size=FileUtils.get_file_size_mb(image_content),
            file_hash=file_hash
        )
        
        assert updated_session.file_key == file_key
        assert updated_session.file_hash == file_hash
        
        # 8. Получаем статус сессии
        status = SessionService.get_session(session.session_id)
        assert status is not None
        assert status.file_key == file_key
        
        # 9. Удаляем сессию
        result = SessionService.delete_session(session.session_id)
        assert result == True
        
        # 10. Проверяем, что сессия удалена
        assert SessionService.get_session(session.session_id) is None
    
    def test_session_expiration(self):
        """Тест истечения сессии"""
        session = SessionService.create_session("user123")
        
        # Сессия должна быть валидна
        assert not session.is_expired()
        assert SessionService.validate_session(session.session_id, "user123") == True
        
        # Истекаем сессию
        session.expiration_at = session.expiration_at.replace(year=2020)
        
        # Сессия должна быть недействительна
        assert session.is_expired()
        assert SessionService.get_session(session.session_id) is None
        assert SessionService.validate_session(session.session_id, "user123") == False


# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])