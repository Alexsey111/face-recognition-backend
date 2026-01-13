"""
Тесты для upload endpoints (Phase 5).
Проверка API endpoints для загрузки файлов с сессиями.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
from io import BytesIO
from PIL import Image
import json

import os
from app.main import create_test_app
from app.routes.auth import get_current_user
from app.db.database import get_db
from app.services.session_service import SessionService
from app.utils.file_utils import FileUtils, ImageValidator
from fastapi import Request
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

# =============================================================================
# НАСТРОЙКА ТЕСТОВОЙ БАЗЫ ДАННЫХ
# =============================================================================

# Используем SQLite для тестирования
SQLALCHEMY_TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_async.db"

# Создаем асинхронный движок для тестов
async_engine = create_async_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Фабрика сессий для тестов
TestingSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def override_get_async_db():
    """Переопределение get_async_db для тестирования"""
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()

# Создаем тестовое приложение без middleware для авторизации
test_app = create_test_app()

# Mock функция для замены get_current_user
async def mock_get_current_user():
    return "test_user_123"

# Переопределяем зависимости
from app.routes.auth import get_current_user
from app.db.database import get_async_db
test_app.dependency_overrides[get_current_user] = mock_get_current_user
test_app.dependency_overrides[get_async_db] = override_get_async_db

client = TestClient(test_app)


def create_test_image(width=300, height=300, format='JPEG', color='red'):
    """Создание тестового изображения (общая функция)"""
    img = Image.new('RGB', (width, height), color=color)
    img_bytes = BytesIO()
    img.save(img_bytes, format=format)
    return img_bytes.getvalue()


class TestUploadEndpoints:
    """Тесты endpoints загрузки файлов"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем сессии
        SessionService._sessions.clear()
    
    def test_create_upload_session(self):
        """Тест создания сессии загрузки"""
        # Устанавливаем тестовое окружение
        os.environ["TESTING"] = "True"
        
        # Тестируем endpoint (без middleware авторизации)
        response = client.post("/api/v1/upload/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "session_id" in data
        assert "expires_at" in data
        assert "max_file_size_mb" in data
        assert data["max_file_size_mb"] == 10.0
        
        # Проверяем, что сессия создана через API
        session_from_api = SessionService.get_session(data["session_id"])
        assert session_from_api is not None
        assert session_from_api.user_id == "test_user_123"
    
    def test_upload_file_to_session(self):
        """Тест загрузки файла в сессию"""
        # Создаем сессию
        session = SessionService.create_session("test_user_123")
        
        # Создаем тестовое изображение
        image_content = create_test_image()
        
        with patch('app.routes.upload.get_current_user') as mock_auth, \
             patch('app.routes.upload.get_storage_service') as mock_storage_factory:
            
            mock_auth.return_value = "test_user_123"
            
            # Мокируем storage service
            mock_storage_instance = AsyncMock()
            mock_storage_instance.upload_image.return_value = {
                "image_id": "test_key.jpg",
                "file_url": "http://minio:9000/bucket/test_key.jpg",
                "file_size": len(image_content)
            }
            mock_storage_factory.return_value = mock_storage_instance
            
            # Создаем файл для загрузки
            files = {
                "file": ("test.jpg", BytesIO(image_content), "image/jpeg")
            }
            
            response = client.post(
                f"/api/v1/upload/{session.session_id}/file",
                files=files
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "file_key" in data
            assert "file_url" in data
            assert "file_size_mb" in data
            assert "file_hash" in data
            assert data["session_id"] == session.session_id

            # Проверяем обновление сессии
            updated_session = SessionService.get_session(session.session_id)
            assert updated_session.file_key is not None

    def test_upload_file_invalid_session(self):
        """Тест загрузки с недействительной сессией"""
        image_content = create_test_image()
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            files = {
                "file": ("test.jpg", BytesIO(image_content), "image/jpeg")
            }
            
            response = client.post(
                "/api/v1/upload/invalid_session_id/file",
                files=files
            )
            
            assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_upload_file_wrong_user(self):
        """Тест загрузки с неправильным пользователем"""
        # Создаем сессию для другого пользователя
        session = SessionService.create_session("other_user")
        
        image_content = create_test_image()
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            files = {
                "file": ("test.jpg", BytesIO(image_content), "image/jpeg")
            }
            
            response = client.post(
                f"/api/v1/upload/{session.session_id}/file",
                files=files
            )
            
            assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_upload_invalid_image_format(self):
        """Тест загрузки файла с неподдерживаемым форматом"""
        session = SessionService.create_session("test_user_123")
        
        # Создаем файл с неподдерживаемым форматом
        invalid_content = b"this is not an image"
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            files = {
                "file": ("test.pdf", BytesIO(invalid_content), "application/pdf")
            }
            
            response = client.post(
                f"/api/v1/upload/{session.session_id}/file",
                files=files
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_upload_small_image(self):
        """Тест загрузки слишком маленького изображения"""
        session = SessionService.create_session("test_user_123")
        
        # Создаем слишком маленькое изображение
        small_image = create_test_image(width=10, height=10)
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            files = {
                "file": ("small.jpg", BytesIO(small_image), "image/jpeg")
            }
            
            response = client.post(
                f"/api/v1/upload/{session.session_id}/file",
                files=files
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            response_data = response.json()
            
            # Извлекаем сообщение об ошибке из правильной структуры ответа
            error_message = ""
            if "error_details" in response_data and isinstance(response_data["error_details"], dict):
                error_details = response_data["error_details"]
                if "error" in error_details:
                    error_message = str(error_details["error"])
                elif "detail" in error_details:
                    error_message = str(error_details["detail"])
            elif "detail" in response_data:
                error_message = str(response_data["detail"])
            elif "message" in response_data:
                error_message = str(response_data["message"])
            elif "error" in response_data:
                error_message = str(response_data["error"])
            
            # Проверяем, что сообщение об ошибке содержит нужный текст
            assert "слишком маленькое" in error_message.lower(), f"Expected 'слишком маленькое' in error message, got: '{error_message}'"
    
    def test_get_upload_status(self):
        """Тест получения статуса сессии"""
        session = SessionService.create_session("test_user_123")
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            response = client.get(f"/api/v1/upload/{session.session_id}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["session_id"] == session.session_id
            assert data["user_id"] == "test_user_123"
            assert data["file_key"] is None
            assert data["file_size_mb"] is None
            assert data["is_expired"] == False
    
    def test_get_upload_status_nonexistent(self):
        """Тест получения статуса несуществующей сессии"""
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            response = client.get("/api/v1/upload/nonexistent_session")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_upload_status_wrong_user(self):
        """Тест получения статуса сессии другого пользователя"""
        session = SessionService.create_session("other_user")
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            response = client.get(f"/api/v1/upload/{session.session_id}")
            
            assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_delete_upload_session(self):
        """Тест удаления сессии загрузки"""
        session = SessionService.create_session("test_user_123")
        
        with patch('app.routes.upload.get_current_user') as mock_auth, \
             patch('app.routes.upload.get_storage_service') as mock_storage_factory:
            
            mock_auth.return_value = "test_user_123"
            
            # Мокируем storage service
            mock_storage_instance = AsyncMock()
            mock_storage_instance.delete_image.return_value = True
            mock_storage_factory.return_value = mock_storage_instance
            
            response = client.delete(f"/api/v1/upload/{session.session_id}")
            
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["message"] == "Сессия загрузки удалена"
            
            # Проверяем, что сессия удалена
            assert SessionService.get_session(session.session_id) is None
    
    def test_delete_nonexistent_session(self):
        """Тест удаления несуществующей сессии"""
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            response = client.delete("/api/v1/upload/nonexistent_session")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_delete_session_wrong_user(self):
        """Тест удаления сессии другого пользователя"""
        session = SessionService.create_session("other_user")
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            response = client.delete(f"/api/v1/upload/{session.session_id}")
            
            assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_get_active_sessions(self):
        """Тест получения активных сессий пользователя"""
        # Создаем несколько сессий
        session1 = SessionService.create_session("test_user_123")
        session2 = SessionService.create_session("test_user_123")
        session3 = SessionService.create_session("other_user")
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            response = client.get("/api/v1/upload/sessions/active")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["user_id"] == "test_user_123"
            assert data["active_sessions_count"] == 2
            assert len(data["sessions"]) == 2
    
    def test_cleanup_expired_sessions(self):
        """Тест принудительной очистки истекших сессий"""
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            response = client.post("/api/v1/upload/cleanup")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "message" in data
            assert "deleted_sessions" in data
    
    def test_legacy_upload(self):
        """Тест устаревшего endpoint прямой загрузки"""
        image_content = create_test_image()
        
        with patch('app.routes.upload.get_current_user') as mock_auth:
            mock_auth.return_value = "test_user_123"
            
            files = {
                "file": ("legacy.jpg", BytesIO(image_content), "image/jpeg")
            }
            
            # POST /api/v1/upload/ с файлом на самом деле создает сессию загрузки
            response = client.post("/api/v1/upload/", files=files)
            
            # Ожидаем успешное создание сессии
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            
            # Проверяем структуру ответа создания сессии
            assert "session_id" in response_data
            assert "expires_at" in response_data
            assert "max_file_size_mb" in response_data
            assert response_data["max_file_size_mb"] == 10.0
            
            # Проверяем, что сессия действительно создана
            session_from_api = SessionService.get_session(response_data["session_id"])
            assert session_from_api is not None
            assert session_from_api.user_id == "test_user_123"


class TestUploadWorkflows:
    """Тесты полных workflow загрузки"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        SessionService._sessions.clear()
    
    @patch('app.routes.upload.get_current_user')
    @patch('app.routes.upload.get_storage_service')
    def test_complete_upload_workflow(self, mock_storage_factory, mock_auth):
        """Тест полного workflow загрузки файла"""
        # Настройка моков
        mock_auth.return_value = "test_user_123"
        mock_storage_instance = AsyncMock()
        mock_storage_instance.upload_image.return_value = {
            "image_id": "workflow_key.jpg",
            "file_url": "http://minio:9000/bucket/workflow_key.jpg",
            "file_size": 1024
        }
        mock_storage_factory.return_value = mock_storage_instance
        
        image_content = create_test_image()
        
        # 1. Создание сессии
        response = client.post("/api/v1/upload/")
        assert response.status_code == status.HTTP_200_OK
        session_data = response.json()
        session_id = session_data["session_id"]

        # 2. Загрузка файла
        files = {
            "file": ("workflow.jpg", BytesIO(image_content), "image/jpeg")
        }
        response = client.post(f"/api/v1/upload/{session_id}/file", files=files)
        assert response.status_code == status.HTTP_200_OK
        upload_data = response.json()

        assert upload_data["session_id"] == session_id
        assert "file_key" in upload_data

        # 3. Проверка статуса
        response = client.get(f"/api/v1/upload/{session_id}")
        assert response.status_code == status.HTTP_200_OK
        status_data = response.json()
        
        assert status_data["session_id"] == session_id
        assert status_data["file_key"] is not None
        
        # 4. Получение активных сессий
        response = client.get("/api/v1/upload/sessions/active")
        assert response.status_code == status.HTTP_200_OK
        sessions_data = response.json()
        
        assert sessions_data["active_sessions_count"] >= 1
        
        # 5. Удаление сессии
        response = client.delete(f"/api/v1/upload/{session_id}")
        assert response.status_code == status.HTTP_200_OK
        
        # 6. Проверка удаления
        response = client.get(f"/api/v1/upload/{session_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @patch('app.routes.upload.get_current_user')
    @patch('app.routes.upload.get_storage_service')
    def test_png_to_jpg_conversion(self, mock_storage_factory, mock_auth):
        """Тест конвертации PNG в JPG"""
        # Настройка моков
        mock_auth.return_value = "test_user_123"
        mock_storage_instance = AsyncMock()
        mock_storage_instance.upload_image.return_value = {
            "image_id": "converted_key.jpg",
            "file_url": "http://minio:9000/bucket/converted_key.jpg",
            "file_size": 1024
        }
        mock_storage_factory.return_value = mock_storage_instance
        
        # Создаем PNG изображение
        img = Image.new('RGB', (300, 300), color='green')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        png_content = img_bytes.getvalue()
        
        # Создаем сессию
        session = SessionService.create_session("test_user_123")
        
        # Загружаем PNG файл
        files = {
            "file": ("test.png", BytesIO(png_content), "image/png")
        }
        response = client.post(f"/api/v1/upload/{session.session_id}/file", files=files)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Файл должен быть загружен (конвертация происходит автоматически)
        assert "file_key" in data
        assert "file_url" in data
    
    @patch('app.routes.upload.get_current_user')
    @patch('app.routes.upload.get_storage_service')
    def test_large_image_resize(self, mock_storage_factory, mock_auth):
        """Тест изменения размера большого изображения"""
        mock_auth.return_value = "test_user_123"
        
        # Мокируем storage service
        mock_storage_instance = AsyncMock()
        mock_storage_instance.upload_image.return_value = {
            "image_id": "large_key.jpg",
            "file_url": "http://minio:9000/bucket/large_key.jpg",
            "file_size": 1024
        }
        mock_storage_factory.return_value = mock_storage_instance
        
        # Создаем очень большое изображение
        img = Image.new('RGB', (3000, 3000), color='yellow')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        large_content = img_bytes.getvalue()
        
        # Создаем сессию
        session = SessionService.create_session("test_user_123")
        
        # Загружаем большой файл
        files = {
            "file": ("large.jpg", BytesIO(large_content), "image/jpeg")
        }
        response = client.post(f"/api/v1/upload/{session.session_id}/file", files=files)
        
        # Файл должен быть принят и изменен в размере
        assert response.status_code == status.HTTP_200_OK


# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
