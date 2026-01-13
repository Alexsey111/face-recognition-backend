"""
Финальные исправленные тесты для Reference Routes API.
Все тесты полностью адаптированы для работы с асинхронным API.
"""

import pytest
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from datetime import datetime, timezone

# Сначала импортируем все необходимые модули
from app.main import create_test_app  # ✅ Используем тестовое приложение без AuthMiddleware
from app.routes.reference import router
from app.models.request import ReferenceCreateRequest, ReferenceUpdateRequest
from app.models.response import ReferenceResponse, ReferenceListResponse
from app.utils.exceptions import ValidationError, ProcessingError, NotFoundError

# Mock настроек до импорта маршрутов
with patch('app.config.settings') as mock_settings:
    mock_settings.MAX_UPLOAD_SIZE = 10485760  # 10MB
    mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
    mock_settings.STORE_ORIGINAL_IMAGES = False
    mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
    mock_settings.cors_origins_list = ["*"]
    mock_settings.return_value = mock_settings
    pass
    

class TestReferenceRoutes:
    """Тесты для Reference Routes API"""
    
    @pytest.fixture
    def app(self):
        """Фикстура для создания FastAPI приложения"""
        # Используем create_test_app как в интеграционных тестах
        app = create_test_app()
        app.include_router(router, prefix="/api/v1")
        return app
    
    @pytest.fixture
    def client(self, app):
        """Фикстура для создания тестового клиента"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_reference_data(self):
        """Фикстура с образцом данных эталона"""
        return {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "label": "Test Reference",
            "file_url": "http://minio/test-image.jpg",
            "file_size": 102400,
            "image_format": "JPEG",
            "image_dimensions": {"width": 224, "height": 224},
            "quality_score": 0.85,
            "is_active": True,
            "usage_count": 0,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "metadata": {"test": "data"}
        }
    
    @pytest.fixture
    def sample_image_data(self):
        """Фикстура с образцом данных изображения"""
        import base64
        from PIL import Image
        import io
        
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_data = img_bytes.getvalue()
        return f"data:image/jpeg;base64,{base64.b64encode(img_data).decode()}"
    
    
    
    # === GET /api/v1/reference - Получение списка эталонов ===
    
    @pytest.mark.asyncio
    async def test_get_references_success(self, client, sample_reference_data):
        """Тест успешного получения списка эталонов"""
        with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            # Создаем mock объекты правильно
            mock_reference = Mock()
            mock_reference.id = str(uuid.uuid4())
            mock_reference.user_id = "user123"
            mock_reference.label = sample_reference_data.get("label", "Test Reference")
            mock_reference.file_url = sample_reference_data.get("file_url")
            mock_reference.created_at = datetime.now(timezone.utc)
            mock_reference.updated_at = None
            mock_reference.quality_score = sample_reference_data.get("quality_score", 0.8)
            mock_reference.usage_count = 0
            mock_reference.last_used = None
            mock_reference.metadata = sample_reference_data.get("metadata")
            mock_reference.is_active = True
            
            # Настраиваем моки правильно
            mock_get_refs.return_value = [mock_reference]
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            response = client.get("/api/v1/reference?user_id=user123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["references"]) == 1
            assert data["total_count"] == 1
            assert "request_id" in data
            
    @pytest.mark.asyncio
    async def test_get_references_with_filters(self, client, sample_reference_data):
        """Тест получения эталонов с фильтрами"""
        with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            mock_reference = Mock()
            mock_reference.id = str(uuid.uuid4())
            mock_reference.user_id = "123"
            mock_reference.label = "test"
            mock_reference.file_url = sample_reference_data.get("file_url")
            mock_reference.created_at = datetime.now(timezone.utc)
            mock_reference.updated_at = None
            mock_reference.quality_score = 0.7
            mock_reference.usage_count = 0
            mock_reference.last_used = None
            mock_reference.metadata = sample_reference_data.get("metadata")
            mock_reference.is_active = True
            
            mock_get_refs.return_value = [mock_reference]
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            response = client.get(
                "/api/v1/reference?user_id=123&label=test&is_active=true&quality_min=0.5&quality_max=0.9"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["references"]) == 1
            assert data["total_count"] == 1

    @pytest.mark.asyncio
    async def test_get_references_with_pagination(self, client, sample_reference_data):
        """Тест получения эталонов с пагинацией"""
        with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            mock_reference = Mock()
            mock_reference.id = str(uuid.uuid4())
            mock_reference.user_id = "user123"
            mock_reference.label = sample_reference_data.get("label", "Test Reference")
            mock_reference.file_url = sample_reference_data.get("file_url")
            mock_reference.created_at = datetime.now(timezone.utc)
            mock_reference.updated_at = None
            mock_reference.quality_score = sample_reference_data.get("quality_score", 0.8)
            mock_reference.usage_count = 0
            mock_reference.last_used = None
            mock_reference.metadata = sample_reference_data.get("metadata")
            mock_reference.is_active = True
            
            # Создаем список из 25 элементов для пагинации
            all_references = [mock_reference for _ in range(25)]
            mock_get_refs.return_value = all_references
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            response = client.get("/api/v1/reference?page=2&per_page=10&user_id=user123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["references"]) == 10  # page=2, per_page=10
            assert data["total_count"] == 25
            assert data["page"] == 2
            assert data["per_page"] == 10
            assert data["has_next"] is True
            assert data["has_prev"] is True

    @pytest.mark.asyncio
    async def test_get_references_empty_list(self, client, sample_reference_data):
        """Тест получения пустого списка эталонов"""
        with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            mock_get_refs.return_value = []
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            response = client.get("/api/v1/reference?user_id=user123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["references"]) == 0
            assert data["total_count"] == 0
    
    # === GET /api/v1/reference/{reference_id} - Получение конкретного эталона ===
    
    @pytest.mark.asyncio
    async def test_get_reference_success(self, client, sample_reference_data):
        """Тест успешного получения конкретного эталона"""
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            reference_id = str(uuid.uuid4())
            
            # Создаем полные данные для ReferenceResponse
            mock_reference_data = {
                "id": reference_id,
                "reference_id": reference_id,  # Добавляем reference_id
                "user_id": "user123",
                "label": sample_reference_data.get("label", "Test Reference"),
                "file_url": sample_reference_data.get("file_url"),
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
                "quality_score": sample_reference_data.get("quality_score", 0.8),
                "usage_count": 0,
                "last_used": None,
                "metadata": sample_reference_data.get("metadata"),
                "is_active": True,
                "embedding": "fake_embedding_data"  # Это должно быть удалено из ответа
            }
            
            mock_get_ref.return_value = mock_reference_data
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            response = client.get(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["reference_id"] == reference_id
            assert "request_id" in data
            # Проверяем что embedding удален из ответа
            assert "embedding" not in data

    @pytest.mark.asyncio
    async def test_get_reference_not_found(self, client):
        """Тест получения несуществующего эталона"""
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            reference_id = str(uuid.uuid4())
            
            mock_get_ref.return_value = None
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            response = client.get(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 404
    
    # === POST /api/v1/reference - Создание эталона ===
    
    @pytest.mark.asyncio
    async def test_create_reference_success(self, client, sample_reference_data, sample_image_data):
        """Тест успешного создания эталона"""
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.StorageService') as mock_storage_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.EncryptionService') as mock_encryption_service, \
             patch('app.db.crud.ReferenceCRUD.create_reference', new_callable=AsyncMock) as mock_create_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            # Настраиваем моки сервисов
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            storage_service = mock_storage_service.return_value
            storage_service.upload_image = AsyncMock(return_value={
                "file_url": "http://minio/test-image.jpg",
                "file_size": 102400,
                "image_id": "test-image-id"
            })
            
            ml_service = mock_ml_service.return_value
            ml_service.generate_embedding = AsyncMock(return_value={
                "success": True,
                "embedding": [0.1, 0.2, 0.3],
                "quality_score": 0.85,
                "model_version": "facenet-vggface2-optimized"
            })
            
            encryption_service = mock_encryption_service.return_value
            encryption_service.encrypt_embedding = AsyncMock(return_value=b"encrypted_embedding")
            
            # Настраиваем мок для создания эталона с правильными полями
            reference_id = str(uuid.uuid4())
            created_reference = {
                "id": reference_id,
                "reference_id": reference_id,  # Добавляем reference_id для Pydantic модели
                "user_id": "user123",
                "label": "Test Reference",
                "file_url": "http://minio/test-image.jpg",
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
                "quality_score": 0.85,
                "usage_count": 0,
                "last_used": None,
                "metadata": {"test": "data"},
                "is_active": True,
                "embedding": "fake_embedding_data"
            }
            
            mock_create_ref.return_value = created_reference
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {
                "user_id": "user123",
                "label": "Test Reference",
                "image_data": sample_image_data,
                "quality_threshold": 0.5,
                "metadata": {"test": "data"}
            }
            
            response = client.post("/api/v1/reference", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["label"] == "Test Reference"
            assert "request_id" in data
            
    @pytest.mark.asyncio
    async def test_create_reference_validation_failed(self, client, sample_image_data):
        """Тест создания эталона с невалидными данными"""
        with patch('app.routes.reference.ValidationService') as mock_validation_service:
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = False
            mock_validation_result.error_message = "Invalid image format"
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            request_data = {
                "user_id": "user123",
                "label": "Test Reference",
                "image_data": sample_image_data
            }
            
            response = client.post("/api/v1/reference", json=request_data)
            
            assert response.status_code == 400
            
    @pytest.mark.asyncio
    def test_create_reference_ml_failed(self, client, sample_image_data):
        """Тест создания эталона при ошибке ML сервиса"""
        # Mock всех сервисов внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Настраиваем мок валидации
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Настраиваем мок ML сервиса с ошибкой
            ml_service = mock_ml_service.return_value
            ml_service.generate_embedding = AsyncMock(return_value={"success": False, "error": "ML processing failed"})
            
            request_data = {
                "user_id": "user123",
                "label": "Test Reference",
                "image_data": sample_image_data
            }
            
            response = client.post("/api/v1/reference", json=request_data)
            
            assert response.status_code == 422
            data = response.json()
            assert data["detail"]["error_code"] == "PROCESSING_ERROR"
            assert "Failed to generate embedding" in data["detail"]["error_details"]["error"]
            
    @pytest.mark.asyncio
    def test_create_reference_quality_below_threshold(self, client, sample_image_data):
        """Тест создания эталона с качеством ниже порога"""
        # Mock всех сервисов внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Настраиваем мок валидации
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Настраиваем мок ML сервиса с низким качеством
            ml_service = mock_ml_service.return_value
            ml_service.generate_embedding = AsyncMock(return_value={
                "success": True,
                "embedding": [0.1, 0.2, 0.3],
                "quality_score": 0.3,  # Низкое качество
                "model_version": "facenet-vggface2-optimized"
            })
            
            request_data = {
                "user_id": "user123",
                "label": "Test Reference",
                "image_data": sample_image_data,
                "quality_threshold": 0.8  # Высокий порог
            }
            
            response = client.post("/api/v1/reference", json=request_data)
            
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error_code"] == "VALIDATION_ERROR"
            assert "quality" in data["detail"]["error_details"]["error"].lower()
            
    @pytest.mark.asyncio
    def test_create_reference_database_error(self, client, sample_image_data):
        """Тест создания эталона при ошибке базы данных"""
        # Mock всех сервисов внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.StorageService') as mock_storage_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.EncryptionService') as mock_encryption_service, \
             patch('app.db.crud.ReferenceCRUD.create_reference', new_callable=AsyncMock) as mock_create_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock успешной валидации
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            storage_service = mock_storage_service.return_value
            storage_service.upload_image = AsyncMock(return_value={"file_url": "http://minio/test.jpg"})
            
            ml_service = mock_ml_service.return_value
            ml_service.generate_embedding = AsyncMock(return_value={
                "success": True,
                "embedding": [0.1, 0.2, 0.3],
                "quality_score": 0.8
            })
            
            encryption_service = mock_encryption_service.return_value
            encryption_service.encrypt_embedding = AsyncMock(return_value=b"encrypted")
            
            # Ошибка при создании в БД
            mock_create_ref.side_effect = Exception("Database error")
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {
                "user_id": "user123",
                "label": "Test Reference",
                "image_data": sample_image_data
            }
            
            response = client.post("/api/v1/reference", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert data["detail"]["error_code"] == "INTERNAL_ERROR"
    
    # === PUT /api/v1/reference/{reference_id} - Обновление эталона ===
    
    @pytest.mark.asyncio
    def test_update_reference_success(self, client, sample_reference_data, sample_image_data):
        """Тест успешного обновления эталона"""
        reference_id = sample_reference_data["id"]
        
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.update_reference', new_callable=AsyncMock) as mock_update_ref, \
             patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.StorageService') as mock_storage_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.EncryptionService') as mock_encryption_service, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock существующий эталон
            existing_reference = {
                "id": reference_id,
                "user_id": "user123",
                "label": "Test Reference",
                "file_url": "http://minio/test.jpg",
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
                "quality_score": 0.8,
                "usage_count": 0,
                "last_used": None,
                "metadata": {"test": "data"},
                "is_active": True,
                "quality_threshold": 0.8
            }
            
            mock_get_ref.return_value = existing_reference
            
            # Mock обновленный эталон
            updated_reference = {
                "id": reference_id,
                "reference_id": reference_id,
                "user_id": "user123",
                "label": "Updated Reference",
                "file_url": "http://minio/test.jpg",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "quality_score": 0.8,
                "usage_count": 0,
                "last_used": None,
                "metadata": {"updated": True},
                "is_active": True,
                "embedding": "fake_embedding_data"
            }
            
            mock_update_ref.return_value = updated_reference
            
            # Mock валидации
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Mock ML и encryption сервисы
            ml_service = mock_ml_service.return_value
            ml_service.generate_embedding = AsyncMock(return_value={
                "success": True,
                "embedding": [0.1, 0.2, 0.3],
                "quality_score": 0.8
            })
            
            encryption_service = mock_encryption_service.return_value
            encryption_service.encrypt_embedding = AsyncMock(return_value=b"encrypted")
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {
                "label": "Updated Reference",
                "metadata": {"updated": True}
            }
            
            response = client.put(f"/api/v1/reference/{reference_id}", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["label"] == "Updated Reference"
    
    @pytest.mark.asyncio
    def test_update_reference_not_found(self, client):
        """Тест обновления несуществующего эталона"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            reference_id = str(uuid.uuid4())
            
            # Mock - эталон не найден
            mock_get_ref.return_value = None
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {"label": "Updated Reference"}
            
            response = client.put(f"/api/v1/reference/{reference_id}", json=request_data)
            
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error_code"] == "VALIDATION_ERROR"
            assert "not found" in data["detail"]["error_details"]["error"]
            
    @pytest.mark.asyncio
    def test_update_reference_with_new_image(self, client, sample_reference_data, sample_image_data):
        """Тест обновления эталона с новым изображением"""
        reference_id = sample_reference_data["id"]
        
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.update_reference', new_callable=AsyncMock) as mock_update_ref, \
             patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.StorageService') as mock_storage_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.EncryptionService') as mock_encryption_service, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock существующий эталон
            existing_reference = {
                "id": reference_id,
                "user_id": "user123",
                "label": "Test Reference",
                "file_url": "http://minio/test.jpg",
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
                "quality_score": 0.8,
                "usage_count": 0,
                "last_used": None,
                "metadata": {"test": "data"},
                "is_active": True,
                "quality_threshold": 0.8
            }
            
            mock_get_ref.return_value = existing_reference
            mock_update_ref.return_value = Mock()
            
            # Mock валидации нового изображения
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Mock ML и encryption сервисы
            ml_service = mock_ml_service.return_value
            ml_service.generate_embedding = AsyncMock(return_value={
                "success": True,
                "embedding": [0.1, 0.2, 0.3],
                "quality_score": 0.8
            })
            
            encryption_service = mock_encryption_service.return_value
            encryption_service.encrypt_embedding = AsyncMock(return_value=b"encrypted")
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {
                "label": "Updated Reference",
                "image_data": sample_image_data
            }
            
            response = client.put(f"/api/v1/reference/{reference_id}", json=request_data)
            
            assert response.status_code == 200
            # Проверяем что вызывались методы для обработки нового изображения
            validation_service.validate_image.assert_called()
            ml_service.generate_embedding.assert_called()
    
    # === PUT /api/v1/update-reference - Алиас для обновления ===
    
    @pytest.mark.asyncio
    def test_update_reference_alias(self, client, sample_reference_data):
        """Тест алиаса для обновления эталона"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.update_reference', new_callable=AsyncMock) as mock_update_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            reference_id = sample_reference_data["id"]
            
            # Mock существующий эталон
            existing_reference = {
                "id": reference_id,
                "user_id": "user123",
                "label": "Test Reference",
                "file_url": "http://minio/test.jpg",
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
                "quality_score": 0.8,
                "usage_count": 0,
                "last_used": None,
                "metadata": {"test": "data"},
                "is_active": True,
                "quality_threshold": 0.8
            }
            
            mock_get_ref.return_value = existing_reference
            mock_update_ref.return_value = Mock()
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {"label": "Test Reference"}
            
            response = client.put(f"/api/v1/update-reference?reference_id={reference_id}", json=request_data)
            
            assert response.status_code == 200
    
    # === DELETE /api/v1/reference/{reference_id} - Удаление эталона ===
    
    def test_delete_reference_success(self, client, sample_reference_data):
        """Тест успешного удаления эталона"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.delete_reference', new_callable=AsyncMock) as mock_delete_ref, \
             patch('app.routes.reference.StorageService') as mock_storage_service, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            reference_id = sample_reference_data["id"]
            reference_data = sample_reference_data.copy()
            reference_data["file_url"] = "http://minio/test-image.jpg"
            
            # Mock получения эталона
            mock_get_ref.return_value = reference_data
            mock_delete_ref.return_value = True
            
            # Mock storage service
            storage_service = mock_storage_service.return_value
            storage_service.delete_image_by_url = AsyncMock()
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            response = client.delete(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "deleted successfully" in data["message"]
    
    def test_delete_reference_not_found(self, client):
        """Тест удаления несуществующего эталона"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            reference_id = str(uuid.uuid4())
            
            # Mock - эталон не найден
            mock_get_ref.return_value = None
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            response = client.delete(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["error_code"] == "REFERENCE_NOT_FOUND"
    
    def test_delete_reference_storage_error(self, client, sample_reference_data):
        """Тест удаления эталона с ошибкой хранилища"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.delete_reference', new_callable=AsyncMock) as mock_delete_ref, \
             patch('app.routes.reference.StorageService') as mock_storage_service, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            reference_id = sample_reference_data["id"]
            reference_data = sample_reference_data.copy()
            reference_data["file_url"] = "http://minio/test-image.jpg"
            
            # Mock получения эталона и удаления из БД
            mock_get_ref.return_value = reference_data
            mock_delete_ref.return_value = True
            
            # Mock storage service с ошибкой
            storage_service = mock_storage_service.return_value
            storage_service.delete_image_by_url = AsyncMock(side_effect=Exception("Storage error"))
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            # Ошибка хранилища не должна влиять на успешное удаление из БД
            response = client.delete(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    # === POST /api/v1/compare - Сравнение с эталонами ===
    
    @pytest.mark.asyncio
    def test_compare_with_references_success(self, client, sample_reference_data, sample_image_data):
        """Тест успешного сравнения с эталонами"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.EncryptionService') as mock_encryption_service, \
             patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock валидации изображения
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Mock эталонов
            reference_data = {
                "id": str(uuid.uuid4()),
                "user_id": "user123",
                "label": "Test Reference",
                "file_url": "http://minio/test.jpg",
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
                "quality_score": 0.8,
                "usage_count": 0,
                "last_used": None,
                "metadata": {"test": "data"},
                "is_active": True,
                "embedding": "encrypted_embedding_data"
            }
            
            mock_get_refs.return_value = [reference_data]
            
            # Mock дешифрации и сравнения
            encryption_service = mock_encryption_service.return_value
            encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            
            ml_service = mock_ml_service.return_value
            ml_service.compare_faces = AsyncMock(return_value={
                "success": True,
                "similarity_score": 0.85,
                "distance": 0.3,
                "processing_time": 0.5
            })
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {
                "user_id": "user123",
                "image_data": sample_image_data,
                "threshold": 0.8,
                "max_results": 10
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) == 1
            assert data["threshold_used"] == 0.8
            assert "processing_time" in data
    
    @pytest.mark.asyncio
    def test_compare_with_references_by_ids(self, client, sample_reference_data, sample_image_data):
        """Тест сравнения с конкретными ID эталонов"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.EncryptionService') as mock_encryption_service, \
             patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock валидации изображения
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Mock получения эталона по ID
            reference_data = {
                "id": sample_reference_data["id"],
                "user_id": "user123",
                "label": "Test Reference",
                "file_url": "http://minio/test.jpg",
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
                "quality_score": 0.8,
                "usage_count": 0,
                "last_used": None,
                "metadata": {"test": "data"},
                "is_active": True,
                "embedding": "encrypted_embedding_data"
            }
            
            mock_get_ref.return_value = reference_data
            
            # Mock дешифрации и сравнения
            encryption_service = mock_encryption_service.return_value
            encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            
            ml_service = mock_ml_service.return_value
            ml_service.compare_faces = AsyncMock(return_value={
                "success": True,
                "similarity_score": 0.85,
                "distance": 0.3,
                "processing_time": 0.5
            })
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {
                "reference_ids": [sample_reference_data["id"]],
                "image_data": sample_image_data,
                "threshold": 0.8
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) == 1
    
    @pytest.mark.asyncio
    def test_compare_with_references_no_references(self, client, sample_image_data):
        """Тест сравнения без доступных эталонов"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock валидации изображения
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Mock отсутствия эталонов
            mock_get_refs.return_value = []
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {
                "user_id": "user123",
                "image_data": sample_image_data
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["error_code"] == "NOT_FOUND"
            assert "No active references found" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_compare_with_references_too_many_references(self, client, sample_image_data):
        """Тест сравнения с слишком большим количеством эталонов"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock валидации изображения
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Mock большого количества эталонов
            many_references = [{"id": str(uuid.uuid4()), "is_active": True, "embedding": "test"} for _ in range(150)]
            mock_get_refs.return_value = many_references
            
            # Настраиваем мок для базы данных
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            request_data = {
                "user_id": "user123",
                "image_data": sample_image_data
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error_code"] == "VALIDATION_ERROR"
            assert "Too many references" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_compare_with_references_validation_failed(self, client, sample_image_data):
        """Тест сравнения с невалидным изображением"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock невалидной валидации
            mock_validation_result = Mock()
            mock_validation_result.is_valid = False
            mock_validation_result.error_message = "Invalid image format"
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            request_data = {
                "user_id": "user123",
                "image_data": sample_image_data
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error_code"] == "VALIDATION_ERROR"
            assert "Image validation failed" in data["detail"]["error_details"]["error"]
            
    @pytest.mark.asyncio
    def test_compare_with_references_no_user_or_ids(self, client, sample_image_data):
        """Тест сравнения без user_id и reference_ids"""
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настраиваем мок настроек
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock валидации изображения
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            request_data = {
                "image_data": sample_image_data
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error_code"] == "VALIDATION_ERROR"
            assert "Either reference_ids or user_id must be provided" in data["detail"]["error_details"]["error"]
            
    # === ОБРАБОТКА ОШИБОК ===
    
    def test_get_references_validation_error(self, client):
        """Тест обработки ValidationError при получении эталонов"""
        from app.utils.exceptions import ValidationError
        
        # Mock настроек для валидации
        with patch('app.routes.reference.settings') as mock_settings:
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Просто тестируем валидацию параметров сортировки
            response = client.get("/api/v1/reference?sort_by=invalid_field")
            
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error_code"] == "VALIDATION_ERROR"
            assert "Invalid sort_by field" in data["detail"]["error_details"]["error"]
    
    def test_get_reference_not_found_error(self, client):
        """Тест обработки NotFoundError при получении эталона"""
        from app.utils.exceptions import NotFoundError
        
        # Mock всех зависимостей внутри функции маршрута
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            # Мокаем ReferenceCRUD.get_reference_by_id, чтобы вызвать NotFoundError
            mock_get_ref.side_effect = NotFoundError("Reference not found")
            
            mock_db_instance = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            reference_id = str(uuid.uuid4())
            response = client.get(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["error_code"] == "REFERENCE_NOT_FOUND"
    
    @pytest.mark.asyncio
    def test_create_reference_processing_error(self, client, sample_image_data):
        """Тест обработки ProcessingError при создании эталона"""
        from app.utils.exceptions import ProcessingError
        
        # Mock всех зависимостей
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager, \
             patch('app.routes.reference.settings') as mock_settings:
            
            # Настройка моков
            mock_settings.MAX_UPLOAD_SIZE = 10485760
            mock_settings.ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
            mock_settings.STORE_ORIGINAL_IMAGES = False
            mock_settings.DELETE_SOURCE_AFTER_PROCESSING = False
            
            # Mock DatabaseManager для асинхронной сессии
            mock_db_instance = Mock()
            mock_db_instance.get_session = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            # Mock валидации как async
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Ошибка при генерации эмбеддинга как async
            ml_service = mock_ml_service.return_value
            ml_service.generate_embedding = AsyncMock(side_effect=ProcessingError("ML processing failed"))
            
            request_data = {
                "user_id": "user123",
                "label": "Test Reference",
                "image_data": f"data:image/jpeg;base64,{sample_image_data}"
            }
            
            response = client.post("/api/v1/reference", json=request_data)
            
            # Просто проверяем статус код - главное, что ProcessingError обрабатывается
            assert response.status_code == 422


# === ИНТЕГРАЦИОННЫЕ ТЕСТЫ ===

class TestReferenceRoutesIntegration:
    """Интеграционные тесты для Reference Routes"""
    
    @pytest.mark.asyncio
    async def test_full_reference_lifecycle(self):
        """Тест полного жизненного цикла эталона"""
        app = create_test_app()  # ✅ Используем тестовое приложение
        app.include_router(router)
        client = TestClient(app)
        
        with patch('app.config.settings') as mock_settings, \
             patch('app.routes.reference.ValidationService') as mock_validation, \
             patch('app.routes.reference.MLService') as mock_ml, \
             patch('app.routes.reference.EncryptionService') as mock_encryption, \
             patch('app.routes.reference.DatabaseService') as mock_db, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            # Настройка моков
            validation_service = mock_validation.return_value
            ml_service = mock_ml.return_value
            encryption_service = mock_encryption.return_value
            db_service = mock_db.return_value
            
            # Отключаем загрузку изображений в тесте
            mock_settings.return_value.STORE_ORIGINAL_IMAGES = False
            
            # Mock сервисов как async
            validation_service.validate_image = AsyncMock(return_value=Mock(
                is_valid=True,
                image_data=b"image_data",
                image_format="JPEG",
                dimensions={"width": 224, "height": 224}
            ))
            
            # Mock DatabaseManager для асинхронной сессии
            mock_db_instance = Mock()
            mock_db_instance.get_session = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            
            # Определяем переменные раньше
            reference_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            
            # Настройка моков
            mock_db_manager.return_value = mock_db_instance
            
            # Mock CRUD методов как async
            with patch('app.db.crud.ReferenceCRUD.create_reference', new_callable=AsyncMock) as mock_create, \
                 patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get, \
                 patch('app.db.crud.ReferenceCRUD.update_reference', new_callable=AsyncMock) as mock_update, \
                 patch('app.db.crud.ReferenceCRUD.delete_reference', new_callable=AsyncMock) as mock_delete:
                
                # Настройка мока для create_reference
                mock_create.return_value = {
                    "id": reference_id,
                    "reference_id": reference_id,  # Добавляем поле reference_id
                    "user_id": user_id,
                    "label": "Test Reference",
                    "quality_score": 0.8,
                    "is_active": True,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "file_url": "http://minio/test.jpg",
                    "file_size": 102400,
                    "image_format": "JPEG",
                    "metadata": {},
                    "embedding": b"encrypted_embedding",  # Добавляем embedding для теста
                    "usage_count": 0,  # Добавляем обязательные поля
                    "last_used": None
                }
                
                # Mock сервисов как async
                ml_service.generate_embedding = AsyncMock(return_value={
                    "success": True,
                    "embedding": [0.1, 0.2, 0.3],
                    "quality_score": 0.8
                })
                encryption_service.encrypt_embedding = AsyncMock(return_value=b"encrypted")
                
                # Настройка CRUD моков
                
                mock_get.return_value = {
                    "id": reference_id,
                    "user_id": user_id,
                    "label": "Test Reference",
                    "quality_score": 0.8,
                    "is_active": True,
                    "file_url": "http://minio/test.jpg"
                }
                
                mock_update.return_value = {
                    "id": reference_id,
                    "user_id": user_id,
                    "label": "Updated Reference",
                    "quality_score": 0.8,
                    "is_active": True,
                    "file_url": "http://minio/test.jpg"
                }
                
                mock_delete.return_value = True
                
                # 1. Создание эталона
                # Используем валидные base64 данные (минимальный JPEG файл)
                valid_jpeg_base64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                
                create_data = {
                    "user_id": user_id,
                    "label": "Test Reference",
                    "image_data": valid_jpeg_base64,
                    "quality_threshold": 0.5
                }
                
                create_response = client.post("/api/v1/reference", json=create_data)
                print(f"Create response status: {create_response.status_code}")
                print(f"Create response content: {create_response.content}")
                assert create_response.status_code == 200
                
                # 2. Получение эталона
                get_response = client.get(f"/api/v1/reference/{reference_id}")
                assert get_response.status_code == 200
                
                # 3. Обновление эталона
                update_data = {"label": "Updated Reference"}
                update_response = client.put(f"/api/v1/reference/{reference_id}", json=update_data)
                assert update_response.status_code == 200
                
                # 4. Удаление эталона
                delete_response = client.delete(f"/api/v1/reference/{reference_id}")
                assert delete_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_reference_comparison_workflow(self):
        """Тест рабочего процесса сравнения с эталонами"""
        app = create_test_app()  # ✅ Используем тестовое приложение
        app.include_router(router)
        client = TestClient(app)
        
        with patch('app.routes.reference.settings'), \
             patch('app.routes.reference.ValidationService') as mock_validation, \
             patch('app.routes.reference.MLService') as mock_ml, \
             patch('app.routes.reference.DatabaseService') as mock_db, \
             patch('app.routes.reference.EncryptionService') as mock_encryption, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            # Настройка моков
            validation_service = mock_validation.return_value
            ml_service = mock_ml.return_value
            db_service = mock_db.return_value
            encryption_service = mock_encryption.return_value
            
            # Mock DatabaseManager для асинхронной сессии
            mock_db_instance = Mock()
            mock_db_instance.get_session = Mock()
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_db_instance.get_session.return_value = mock_session
            mock_db_manager.return_value = mock_db_instance
            
            # Mock CRUD методов для избежания обращения к реальной БД
            # Создаем простые объекты с атрибутом is_active
            reference1 = Mock()
            reference1.id = 'ref1'
            reference1.label = 'Reference 1'
            reference1.user_id = 'user123'
            reference1.quality_score = 0.8
            reference1.is_active = True
            reference1.embedding = b"encrypted_embedding_1"
            reference1.get = lambda key, default=None: getattr(reference1, key, default)
            reference1.__getitem__ = lambda self, key: getattr(reference1, key)
            
            reference2 = Mock()
            reference2.id = 'ref2'
            reference2.label = 'Reference 2'
            reference2.user_id = 'user123'
            reference2.quality_score = 0.9
            reference2.is_active = True
            reference2.embedding = b"encrypted_embedding_2"
            reference2.get = lambda key, default=None: getattr(reference2, key, default)
            reference2.__getitem__ = lambda self, key: getattr(reference2, key)
            
            # Мокаем ReferenceCRUD.get_all_references как async метод
            with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_all_refs:
                mock_get_all_refs.return_value = [reference1, reference2]
                
                # Mock валидации изображения (async)
                validation_service.validate_image = AsyncMock(return_value=Mock(
                    is_valid=True,
                    image_data=b"image_data",
                    image_format="JPEG"
                ))
                
                # Mock дешифрации и сравнения (async)
                encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
                ml_service.compare_faces = AsyncMock(return_value={
                    "success": True,
                    "similarity_score": 0.85,
                    "distance": 0.3,
                    "processing_time": 0.5
                })
                
                compare_data = {
                    "user_id": "user123",
                    "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD",
                    "threshold": 0.8,
                    "max_results": 10
                }
                
                response = client.post("/api/v1/compare", json=compare_data)
                assert response.status_code == 200
                
                data = response.json()
                assert data["success"] is True
                assert len(data["results"]) == 2
                assert all("similarity_score" in result for result in data["results"])
