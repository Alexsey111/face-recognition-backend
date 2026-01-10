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

    # === PUT /api/v1/reference/{reference_id} - Обновление эталона ===

    @pytest.mark.asyncio
    async def test_update_reference_success(self, client, sample_reference_data):
        """Тест успешного обновления эталона"""
        reference_id = sample_reference_data["id"]
        
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.update_reference', new_callable=AsyncMock) as mock_update_ref, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
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
            
            mock_get_ref.return_value = existing_reference
            mock_update_ref.return_value = updated_reference
            
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
    async def test_update_reference_not_found(self, client):
        """Тест обновления несуществующего эталона"""
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
            
            request_data = {"label": "Updated Reference"}
            
            response = client.put(f"/api/v1/reference/{reference_id}", json=request_data)
            
            assert response.status_code == 400

    # === DELETE /api/v1/reference/{reference_id} - Удаление эталона ===

    @pytest.mark.asyncio
    async def test_delete_reference_success(self, client, sample_reference_data):
        """Тест успешного удаления эталона"""
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.delete_reference', new_callable=AsyncMock) as mock_delete_ref, \
             patch('app.routes.reference.StorageService') as mock_storage_service, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            reference_id = sample_reference_data["id"]
            reference_data = sample_reference_data.copy()
            reference_data["file_url"] = "http://minio/test-image.jpg"
            
            mock_get_ref.return_value = reference_data
            mock_delete_ref.return_value = True
            
            storage_service = mock_storage_service.return_value
            storage_service.delete_image_by_url = AsyncMock()
            
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

    @pytest.mark.asyncio
    async def test_delete_reference_not_found(self, client):
        """Тест удаления несуществующего эталона"""
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
            
            response = client.delete(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 404

    # === POST /api/v1/compare - Сравнение с эталонами ===

    @pytest.mark.asyncio
    async def test_compare_with_references_success(self, client, sample_reference_data, sample_image_data):
        """Тест успешного сравнения с эталонами"""
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.routes.reference.MLService') as mock_ml_service, \
             patch('app.routes.reference.EncryptionService') as mock_encryption_service, \
             patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
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
            
            encryption_service = mock_encryption_service.return_value
            encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            
            ml_service = mock_ml_service.return_value
            ml_service.compare_faces = AsyncMock(return_value={
                "success": True,
                "similarity_score": 0.85,
                "distance": 0.15,
                "processing_time": 0.05
            })
            
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
                "max_results": 10,
                "include_metadata": True
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["total_references_compared"] == 1

    @pytest.mark.asyncio
    async def test_compare_with_references_no_references(self, client, sample_image_data):
        """Тест сравнения когда нет доступных эталонов"""
        with patch('app.routes.reference.ValidationService') as mock_validation_service, \
             patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Возвращаем пустой список эталонов
            mock_get_refs.return_value = []
            
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
            
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_compare_with_references_validation_failed(self, client):
        """Тест сравнения с невалидными данными"""
        with patch('app.routes.reference.ValidationService') as mock_validation_service:
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = False
            mock_validation_result.error_message = "Invalid image format"
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            request_data = {
                "user_id": "user123",
                "image_data": "invalid_image_data",
                "threshold": 0.8
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_compare_with_references_no_user_or_ids(self, client, sample_image_data):
        """Тест сравнения без указания user_id или reference_ids"""
        with patch('app.routes.reference.ValidationService') as mock_validation_service:
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
            mock_validation_result.image_format = "JPEG"
            mock_validation_result.dimensions = {"width": 224, "height": 224}
            
            validation_service = mock_validation_service.return_value
            validation_service.validate_image = AsyncMock(return_value=mock_validation_result)
            
            # Не указываем ни user_id, ни reference_ids
            request_data = {
                "image_data": sample_image_data,
                "threshold": 0.8
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 400