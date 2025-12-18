"""
Тесты для Reference Routes API.
Критически важный модуль с низким покрытием - цель: увеличить с 12.50% до 80%+
"""

import pytest
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from datetime import datetime, timezone

# Mock всех внешних зависимостей
with patch('app.routes.reference.settings'), \
     patch('app.routes.reference.DatabaseService'), \
     patch('app.routes.reference.StorageService'), \
     patch('app.routes.reference.MLService'), \
     patch('app.routes.reference.EncryptionService'), \
     patch('app.routes.reference.ValidationService'):
    
    from app.main import create_app
    from app.routes.reference import router
    from app.models.request import ReferenceCreateRequest, ReferenceUpdateRequest
    from app.models.response import ReferenceResponse, ReferenceListResponse
    from app.utils.exceptions import ValidationError, ProcessingError, NotFoundError


class TestReferenceRoutes:
    """Тесты для Reference Routes API"""
    
    @pytest.fixture
    def app(self):
        """Фикстура для создания FastAPI приложения"""
        app = create_app()
        app.include_router(router)
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
        return base64.b64encode(img_bytes.getvalue()).decode()
    
    @pytest.fixture
    def mock_db_service(self):
        """Фикстура для мока DatabaseService"""
        with patch('app.routes.reference.DatabaseService') as mock_service:
            yield mock_service.return_value
    
    @pytest.fixture
    def mock_validation_service(self):
        """Фикстура для мока ValidationService"""
        with patch('app.routes.reference.ValidationService') as mock_service:
            yield mock_service.return_value
    
    @pytest.fixture
    def mock_storage_service(self):
        """Фикстура для мока StorageService"""
        with patch('app.routes.reference.StorageService') as mock_service:
            yield mock_service.return_value
    
    @pytest.fixture
    def mock_ml_service(self):
        """Фикстура для мока MLService"""
        with patch('app.routes.reference.MLService') as mock_service:
            yield mock_service.return_value
    
    @pytest.fixture
    def mock_encryption_service(self):
        """Фикстура для мока EncryptionService"""
        with patch('app.routes.reference.EncryptionService') as mock_service:
            yield mock_service.return_value
    
    # === GET /api/v1/reference - Получение списка эталонов ===
    
    def test_get_references_success(self, client, mock_db_service, sample_reference_data):
        """Тест успешного получения списка эталонов"""
        # Mock ответ от БД
        mock_db_service.get_references.return_value = {
            "items": [sample_reference_data],
            "total_count": 1,
            "has_next": False,
            "has_prev": False
        }
        
        response = client.get("/api/v1/reference")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["references"]) == 1
        assert data["total_count"] == 1
        assert "request_id" in data
    
    def test_get_references_with_filters(self, client, mock_db_service, sample_reference_data):
        """Тест получения эталонов с фильтрами"""
        mock_db_service.get_references.return_value = {
            "items": [sample_reference_data],
            "total_count": 1,
            "has_next": False,
            "has_prev": False
        }
        
        response = client.get(
            "/api/v1/reference?user_id=123&label=test&is_active=true&quality_min=0.5&quality_max=0.9"
        )
        
        assert response.status_code == 200
        # Проверяем что метод был вызван с правильными фильтрами
        mock_db_service.get_references.assert_called_once()
        call_args = mock_db_service.get_references.call_args
        assert "filters" in call_args.kwargs
        filters = call_args.kwargs["filters"]
        assert filters["user_id"] == "123"
        assert filters["label"] == "test"
        assert filters["is_active"] is True
        assert filters["quality_min"] == 0.5
        assert filters["quality_max"] == 0.9
    
    def test_get_references_with_pagination(self, client, mock_db_service, sample_reference_data):
        """Тест получения эталонов с пагинацией"""
        mock_db_service.get_references.return_value = {
            "items": [sample_reference_data],
            "total_count": 25,
            "has_next": True,
            "has_prev": False
        }
        
        response = client.get("/api/v1/reference?page=2&per_page=10")
        
        assert response.status_code == 200
        call_args = mock_db_service.get_references.call_args
        assert call_args.kwargs["page"] == 2
        assert call_args.kwargs["per_page"] == 10
        assert response.json()["has_next"] is True
    
    def test_get_references_with_sorting(self, client, mock_db_service, sample_reference_data):
        """Тест получения эталонов с сортировкой"""
        mock_db_service.get_references.return_value = {
            "items": [sample_reference_data],
            "total_count": 1,
            "has_next": False,
            "has_prev": False
        }
        
        response = client.get("/api/v1/reference?sort_by=quality_score&sort_order=asc")
        
        assert response.status_code == 200
        call_args = mock_db_service.get_references.call_args
        assert call_args.kwargs["sort_by"] == "quality_score"
        assert call_args.kwargs["sort_order"] == "asc"
    
    def test_get_references_invalid_sort_field(self, client):
        """Тест получения эталонов с невалидным полем сортировки"""
        response = client.get("/api/v1/reference?sort_by=invalid_field")
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "Invalid sort_by field" in data["detail"]["error_details"]["error"]
    
    def test_get_references_invalid_sort_order(self, client):
        """Тест получения эталонов с невалидным порядком сортировки"""
        response = client.get("/api/v1/reference?sort_order=invalid")
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "sort_order must be 'asc' or 'desc'" in data["detail"]["error_details"]["error"]
    
    def test_get_references_empty_list(self, client, mock_db_service):
        """Тест получения пустого списка эталонов"""
        mock_db_service.get_references.return_value = {
            "items": [],
            "total_count": 0,
            "has_next": False,
            "has_prev": False
        }
        
        response = client.get("/api/v1/reference")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["references"]) == 0
        assert data["total_count"] == 0
    
    def test_get_references_database_error(self, client, mock_db_service):
        """Тест обработки ошибки базы данных при получении эталонов"""
        mock_db_service.get_references.side_effect = Exception("Database error")
        
        response = client.get("/api/v1/reference")
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["error_code"] == "INTERNAL_ERROR"
    
    # === GET /api/v1/reference/{reference_id} - Получение конкретного эталона ===
    
    def test_get_reference_success(self, client, mock_db_service, sample_reference_data):
        """Тест успешного получения конкретного эталона"""
        reference_id = str(uuid.uuid4())
        sample_reference_data["id"] = reference_id
        mock_db_service.get_reference_by_id.return_value = sample_reference_data
        
        response = client.get(f"/api/v1/reference/{reference_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["id"] == reference_id
        assert "request_id" in data
        # Проверяем что embedding удален из ответа
        assert "embedding" not in data
    
    def test_get_reference_not_found(self, client, mock_db_service):
        """Тест получения несуществующего эталона"""
        reference_id = str(uuid.uuid4())
        mock_db_service.get_reference_by_id.return_value = None
        
        response = client.get(f"/api/v1/reference/{reference_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error_code"] == "REFERENCE_NOT_FOUND"
        assert "not found" in data["detail"]["error_details"]["error"]
    
    def test_get_reference_database_error(self, client, mock_db_service):
        """Тест обработки ошибки базы данных при получении эталона"""
        reference_id = str(uuid.uuid4())
        mock_db_service.get_reference_by_id.side_effect = Exception("Database error")
        
        response = client.get(f"/api/v1/reference/{reference_id}")
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["error_code"] == "INTERNAL_ERROR"
    
    # === POST /api/v1/reference - Создание эталона ===
    
    @pytest.mark.asyncio
    def test_create_reference_success(self, client, mock_validation_service, mock_storage_service, 
                                     mock_ml_service, mock_encryption_service, mock_db_service, 
                                     sample_reference_data, sample_image_data):
        """Тест успешного создания эталона"""
        # Mock всех сервисов
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        mock_storage_service.upload_image.return_value = {
            "file_url": "http://minio/test-image.jpg",
            "file_size": 102400,
            "image_id": "test-image-id"
        }
        
        mock_ml_service.generate_embedding.return_value = {
            "success": True,
            "embedding": [0.1, 0.2, 0.3],
            "quality_score": 0.85,
            "model_version": "facenet-vggface2-optimized"
        }
        
        mock_encryption_service.encrypt_embedding.return_value = b"encrypted_embedding"
        mock_db_service.create_reference.return_value = sample_reference_data
        
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
    def test_create_reference_validation_failed(self, client, mock_validation_service, sample_image_data):
        """Тест создания эталона с невалидными данными"""
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.error_message = "Invalid image format"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        request_data = {
            "user_id": "user123",
            "label": "Test Reference",
            "image_data": sample_image_data
        }
        
        response = client.post("/api/v1/reference", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "Image validation failed" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_create_reference_ml_failed(self, client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест создания эталона при ошибке ML сервиса"""
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        mock_ml_service.generate_embedding.return_value = {"success": False, "error": "ML processing failed"}
        
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
    def test_create_reference_quality_below_threshold(self, client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест создания эталона с качеством ниже порога"""
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        mock_ml_service.generate_embedding.return_value = {
            "success": True,
            "embedding": [0.1, 0.2, 0.3],
            "quality_score": 0.3,  # Низкое качество
            "model_version": "facenet-vggface2-optimized"
        }
        
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
    def test_create_reference_database_error(self, client, mock_validation_service, mock_storage_service, 
                                           mock_ml_service, mock_encryption_service, mock_db_service, sample_image_data):
        """Тест создания эталона при ошибке базы данных"""
        # Mock успешной валидации
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        mock_storage_service.upload_image.return_value = {"file_url": "http://minio/test.jpg"}
        mock_ml_service.generate_embedding.return_value = {
            "success": True,
            "embedding": [0.1, 0.2, 0.3],
            "quality_score": 0.8
        }
        mock_encryption_service.encrypt_embedding.return_value = b"encrypted"
        
        # Ошибка при создании в БД
        mock_db_service.create_reference.side_effect = Exception("Database error")
        
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
    def test_update_reference_success(self, client, mock_db_service, mock_validation_service, 
                                    mock_storage_service, mock_ml_service, mock_encryption_service, sample_reference_data, sample_image_data):
        """Тест успешного обновления эталона"""
        reference_id = sample_reference_data["id"]
        
        # Mock существующий эталон
        mock_db_service.get_reference_by_id.return_value = sample_reference_data
        mock_db_service.update_reference.return_value = {**sample_reference_data, "label": "Updated Reference"}
        
        # Mock валидации
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        mock_ml_service.generate_embedding.return_value = {
            "success": True,
            "embedding": [0.1, 0.2, 0.3],
            "quality_score": 0.8
        }
        mock_encryption_service.encrypt_embedding.return_value = b"encrypted"
        
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
    def test_update_reference_not_found(self, client, mock_db_service):
        """Тест обновления несуществующего эталона"""
        reference_id = str(uuid.uuid4())
        mock_db_service.get_reference_by_id.return_value = None
        
        request_data = {"label": "Updated Reference"}
        
        response = client.put(f"/api/v1/reference/{reference_id}", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "not found" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_update_reference_with_new_image(self, client, mock_db_service, mock_validation_service, 
                                           mock_storage_service, mock_ml_service, mock_encryption_service, sample_reference_data, sample_image_data):
        """Тест обновления эталона с новым изображением"""
        reference_id = sample_reference_data["id"]
        
        # Mock существующий эталон
        mock_db_service.get_reference_by_id.return_value = sample_reference_data
        mock_db_service.update_reference.return_value = sample_reference_data
        
        # Mock валидации нового изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        mock_ml_service.generate_embedding.return_value = {
            "success": True,
            "embedding": [0.1, 0.2, 0.3],
            "quality_score": 0.8
        }
        mock_encryption_service.encrypt_embedding.return_value = b"encrypted"
        
        request_data = {
            "label": "Updated Reference",
            "image_data": sample_image_data
        }
        
        response = client.put(f"/api/v1/reference/{reference_id}", json=request_data)
        
        assert response.status_code == 200
        # Проверяем что вызывались методы для обработки нового изображения
        mock_validation_service.validate_image.assert_called()
        mock_ml_service.generate_embedding.assert_called()
    
    # === PUT /api/v1/update-reference - Алиас для обновления ===
    
    @pytest.mark.asyncio
    def test_update_reference_alias(self, client, mock_db_service, sample_reference_data):
        """Тест алиаса для обновления эталона"""
        reference_id = sample_reference_data["id"]
        
        with patch('app.routes.reference.update_reference') as mock_update:
            mock_update.return_value = sample_reference_data
            
            request_data = {"label": "Test Reference"}
            
            response = client.put(f"/api/v1/update-reference?reference_id={reference_id}", json=request_data)
            
            assert response.status_code == 200
            mock_update.assert_called_once()
    
    # === DELETE /api/v1/reference/{reference_id} - Удаление эталона ===
    
    def test_delete_reference_success(self, client, mock_db_service, mock_storage_service, sample_reference_data):
        """Тест успешного удаления эталона"""
        reference_id = sample_reference_data["id"]
        sample_reference_data["file_url"] = "http://minio/test-image.jpg"
        
        mock_db_service.get_reference_by_id.return_value = sample_reference_data
        mock_db_service.delete_reference.return_value = True
        
        response = client.delete(f"/api/v1/reference/{reference_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["message"]
    
    def test_delete_reference_not_found(self, client, mock_db_service):
        """Тест удаления несуществующего эталона"""
        reference_id = str(uuid.uuid4())
        mock_db_service.get_reference_by_id.return_value = None
        
        response = client.delete(f"/api/v1/reference/{reference_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error_code"] == "REFERENCE_NOT_FOUND"
    
    def test_delete_reference_storage_error(self, client, mock_db_service, mock_storage_service, sample_reference_data):
        """Тест удаления эталона с ошибкой хранилища"""
        reference_id = sample_reference_data["id"]
        sample_reference_data["file_url"] = "http://minio/test-image.jpg"
        
        mock_db_service.get_reference_by_id.return_value = sample_reference_data
        mock_db_service.delete_reference.return_value = True
        mock_storage_service.delete_image_by_url.side_effect = Exception("Storage error")
        
        # Ошибка хранилища не должна влиять на успешное удаление из БД
        response = client.delete(f"/api/v1/reference/{reference_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    # === POST /api/v1/reference/compare - Сравнение с эталонами ===
    
    @pytest.mark.asyncio
    def test_compare_with_references_success(self, client, mock_validation_service, mock_ml_service, 
                                           mock_db_service, mock_encryption_service, sample_reference_data, sample_image_data):
        """Тест успешного сравнения с эталонами"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock эталонов
        mock_db_service.get_active_references_by_user.return_value = [sample_reference_data]
        
        # Mock дешифрации и сравнения
        mock_encryption_service.decrypt_embedding.return_value = [0.1, 0.2, 0.3]
        mock_ml_service.compare_faces.return_value = {
            "success": True,
            "similarity_score": 0.85,
            "distance": 0.3,
            "processing_time": 0.5
        }
        
        request_data = {
            "user_id": "user123",
            "image_data": sample_image_data,
            "threshold": 0.8,
            "max_results": 10
        }
        
        response = client.post("/api/v1/reference/compare", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["threshold_used"] == 0.8
        assert "processing_time" in data
    
    @pytest.mark.asyncio
    def test_compare_with_references_by_ids(self, client, mock_validation_service, mock_ml_service, 
                                          mock_db_service, mock_encryption_service, sample_reference_data, sample_image_data):
        """Тест сравнения с конкретными ID эталонов"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock получения эталона по ID
        mock_db_service.get_reference_by_id.return_value = sample_reference_data
        
        # Mock дешифрации и сравнения
        mock_encryption_service.decrypt_embedding.return_value = [0.1, 0.2, 0.3]
        mock_ml_service.compare_faces.return_value = {
            "success": True,
            "similarity_score": 0.85,
            "distance": 0.3,
            "processing_time": 0.5
        }
        
        request_data = {
            "reference_ids": [sample_reference_data["id"]],
            "image_data": sample_image_data,
            "threshold": 0.8
        }
        
        response = client.post("/api/v1/reference/compare", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
    
    @pytest.mark.asyncio
    def test_compare_with_references_no_references(self, client, mock_validation_service, mock_db_service, sample_image_data):
        """Тест сравнения без доступных эталонов"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock отсутствия эталонов
        mock_db_service.get_active_references_by_user.return_value = []
        
        request_data = {
            "user_id": "user123",
            "image_data": sample_image_data
        }
        
        response = client.post("/api/v1/reference/compare", json=request_data)
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error_code"] == "NOT_FOUND"
        assert "No active references found" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_compare_with_references_too_many_references(self, client, mock_validation_service, mock_db_service, sample_image_data):
        """Тест сравнения с слишком большим количеством эталонов"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock большого количества эталонов
        many_references = [{"id": str(uuid.uuid4()), "is_active": True} for _ in range(150)]
        mock_db_service.get_active_references_by_user.return_value = many_references
        
        request_data = {
            "user_id": "user123",
            "image_data": sample_image_data
        }
        
        response = client.post("/api/v1/reference/compare", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "Too many references" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_compare_with_references_validation_failed(self, client, mock_validation_service, sample_image_data):
        """Тест сравнения с невалидным изображением"""
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.error_message = "Invalid image format"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        request_data = {
            "user_id": "user123",
            "image_data": sample_image_data
        }
        
        response = client.post("/api/v1/reference/compare", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "Image validation failed" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_compare_with_references_no_user_or_ids(self, client, mock_validation_service, sample_image_data):
        """Тест сравнения без user_id и reference_ids"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        request_data = {
            "image_data": sample_image_data
        }
        
        response = client.post("/api/v1/reference/compare", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "Either reference_ids or user_id must be provided" in data["detail"]["error_details"]["error"]
    
    # === ОБРАБОТКА ОШИБОК ===
    
    def test_get_references_validation_error(self, client, mock_db_service):
        """Тест обработки ValidationError при получении эталонов"""
        from app.utils.exceptions import ValidationError
        mock_db_service.get_references.side_effect = ValidationError("Invalid parameters")
        
        response = client.get("/api/v1/reference")
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
    
    def test_get_reference_not_found_error(self, client, mock_db_service):
        """Тест обработки NotFoundError при получении эталона"""
        from app.utils.exceptions import NotFoundError
        mock_db_service.get_reference_by_id.side_effect = NotFoundError("Reference not found")
        
        reference_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/reference/{reference_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error_code"] == "REFERENCE_NOT_FOUND"
    
    @pytest.mark.asyncio
    def test_create_reference_processing_error(self, client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест обработки ProcessingError при создании эталона"""
        from app.utils.exceptions import ProcessingError
        
        # Mock валидации
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Ошибка при генерации эмбеддинга
        mock_ml_service.generate_embedding.side_effect = ProcessingError("ML processing failed")
        
        request_data = {
            "user_id": "user123",
            "label": "Test Reference",
            "image_data": sample_image_data
        }
        
        response = client.post("/api/v1/reference", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["detail"]["error_code"] == "PROCESSING_ERROR"


# === ИНТЕГРАЦИОННЫЕ ТЕСТЫ ===

class TestReferenceRoutesIntegration:
    """Интеграционные тесты для Reference Routes"""
    
    @pytest.mark.asyncio
    async def test_full_reference_lifecycle(self):
        """Тест полного жизненного цикла эталона"""
        app = create_app()
        app.include_router(router)
        client = TestClient(app)
        
        with patch('app.routes.reference.settings'), \
             patch('app.routes.reference.DatabaseService') as mock_db, \
             patch('app.routes.reference.ValidationService') as mock_validation, \
             patch('app.routes.reference.StorageService') as mock_storage, \
             patch('app.routes.reference.MLService') as mock_ml, \
             patch('app.routes.reference.EncryptionService') as mock_encryption:
            
            # Настройка моков
            db_service = mock_db.return_value
            validation_service = mock_validation.return_value
            storage_service = mock_storage.return_value
            ml_service = mock_ml.return_value
            encryption_service = mock_encryption.return_value
            
            reference_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            
            # 1. Создание эталона
            mock_validation.validate_image.return_value = Mock(
                is_valid=True,
                image_data=b"image_data",
                image_format="JPEG",
                dimensions={"width": 224, "height": 224}
            )
            storage_service.upload_image.return_value = {"file_url": "http://minio/test.jpg"}
            ml_service.generate_embedding.return_value = {
                "success": True,
                "embedding": [0.1, 0.2, 0.3],
                "quality_score": 0.8
            }
            encryption_service.encrypt_embedding.return_value = b"encrypted"
            db_service.create_reference.return_value = {
                "id": reference_id,
                "user_id": user_id,
                "label": "Test Reference",
                "quality_score": 0.8,
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            create_data = {
                "user_id": user_id,
                "label": "Test Reference",
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD",
                "quality_threshold": 0.5
            }
            
            create_response = client.post("/api/v1/reference", json=create_data)
            assert create_response.status_code == 200
            
            # 2. Получение эталона
            db_service.get_reference_by_id.return_value = {
                "id": reference_id,
                "user_id": user_id,
                "label": "Test Reference",
                "quality_score": 0.8,
                "is_active": True
            }
            
            get_response = client.get(f"/api/v1/reference/{reference_id}")
            assert get_response.status_code == 200
            
            # 3. Обновление эталона
            db_service.update_reference.return_value = {
                "id": reference_id,
                "user_id": user_id,
                "label": "Updated Reference",
                "quality_score": 0.8,
                "is_active": True
            }
            
            update_data = {"label": "Updated Reference"}
            update_response = client.put(f"/api/v1/reference/{reference_id}", json=update_data)
            assert update_response.status_code == 200
            
            # 4. Удаление эталона
            db_service.get_reference_by_id.return_value = {
                "id": reference_id,
                "file_url": "http://minio/test.jpg"
            }
            db_service.delete_reference.return_value = True
            
            delete_response = client.delete(f"/api/v1/reference/{reference_id}")
            assert delete_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_reference_comparison_workflow(self):
        """Тест рабочего процесса сравнения с эталонами"""
        app = create_app()
        app.include_router(router)
        client = TestClient(app)
        
        with patch('app.routes.reference.settings'), \
             patch('app.routes.reference.ValidationService') as mock_validation, \
             patch('app.routes.reference.MLService') as mock_ml, \
             patch('app.routes.reference.DatabaseService') as mock_db, \
             patch('app.routes.reference.EncryptionService') as mock_encryption:
            
            # Настройка моков
            validation_service = mock_validation.return_value
            ml_service = mock_ml.return_value
            db_service = mock_db.return_value
            encryption_service = mock_encryption.return_value
            
            # Mock валидации изображения
            validation_service.validate_image.return_value = Mock(
                is_valid=True,
                image_data=b"image_data",
                image_format="JPEG"
            )
            
            # Mock эталонов
            db_service.get_active_references_by_user.return_value = [
                {
                    "id": str(uuid.uuid4()),
                    "label": "Reference 1",
                    "user_id": "user123",
                    "quality_score": 0.8,
                    "is_active": True
                },
                {
                    "id": str(uuid.uuid4()),
                    "label": "Reference 2", 
                    "user_id": "user123",
                    "quality_score": 0.9,
                    "is_active": True
                }
            ]
            
            # Mock дешифрации и сравнения
            encryption_service.decrypt_embedding.return_value = [0.1, 0.2, 0.3]
            ml_service.compare_faces.return_value = {
                "success": True,
                "similarity_score": 0.85,
                "distance": 0.3,
                "processing_time": 0.5
            }
            
            compare_data = {
                "user_id": "user123",
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD",
                "threshold": 0.8,
                "max_results": 10
            }
            
            response = client.post("/api/v1/reference/compare", json=compare_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) == 2
            assert all("similarity_score" in result for result in data["results"])