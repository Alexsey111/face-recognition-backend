"""
Финальные исправленные тесты для Reference Routes API.
Все тесты полностью адаптированы для работы с асинхронным API.
"""

import pytest
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from datetime import datetime, timezone
from types import SimpleNamespace
from contextlib import asynccontextmanager

from app.routes.reference import router as reference_router
from app.routes.health import router as health_router
from app.utils.exceptions import ValidationError, ProcessingError, NotFoundError


def create_test_app_for_reference():
    """Создает FastAPI приложение ТОЛЬКО для тестирования reference routes"""
    app = FastAPI(title="Reference Routes Test", version="1.0.0")
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(reference_router, prefix="/api/v1")
    return app


def create_mock_reference(overrides=None):
    """Создает mock объект с атрибутами SQLAlchemy модели Reference"""
    ref_id = str(uuid.uuid4())
    data = {
        "id": ref_id, "user_id": "user123", "label": "Test Reference",
        "file_url": "http://minio/test-image.jpg", "created_at": datetime.now(timezone.utc),
        "updated_at": None, "quality_score": 0.85, "usage_count": 0, "last_used": None,
        "metadata": {"test": "data"}, "is_active": True, "embedding_encrypted": b"encrypted_embedding",
        "embedding_version": "1", "embedding_hash": "testhash123", "image_filename": "test.jpg",
        "image_size_mb": 0.1, "image_format": "JPEG", "face_landmarks": None, "previous_reference_id": None,
    }
    if overrides:
        data.update(overrides)
    return SimpleNamespace(**data)


def create_mock_db_manager():
    """Создает мок DatabaseManager с async context manager"""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=Mock(scalars=Mock(all=Mock(return_value=[]))))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    
    mock_cm = Mock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    
    mock_db_manager = Mock()
    mock_db_manager.get_session = Mock(return_value=mock_cm)
    return mock_db_manager, mock_session


class TestReferenceRoutes:
    """Тесты для Reference Routes API"""
    
    @pytest.fixture
    def app(self):
        return create_test_app_for_reference()
    
    @pytest.fixture
    def client(self, app):
        return TestClient(app, raise_server_exceptions=False)
    
    @pytest.fixture
    def sample_reference_data(self):
        return {
            "id": str(uuid.uuid4()), "user_id": str(uuid.uuid4()), "label": "Test Reference",
            "file_url": "http://minio/test-image.jpg", "file_size": 102400, "image_format": "JPEG",
            "image_dimensions": {"width": 224, "height": 224}, "quality_score": 0.85, "is_active": True,
            "usage_count": 0, "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
            "metadata": {"test": "data"}
        }
    
    @pytest.fixture
    def sample_image_data(self):
        import base64
        from PIL import Image
        import io
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        return f"data:image/jpeg;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"

    # === GET /api/v1/reference - Получение списка эталонов ===
    
    def test_get_references_success(self, client, sample_reference_data):
        mock_reference = create_mock_reference()
        mock_reference.label = sample_reference_data.get("label", "Test Reference")
        mock_reference.file_url = sample_reference_data.get("file_url")
        mock_reference.quality_score = sample_reference_data.get("quality_score", 0.8)
        mock_reference.metadata = sample_reference_data.get("metadata")
        
        mock_db_manager, mock_session = create_mock_db_manager()
        mock_session.execute = AsyncMock(return_value=Mock(scalars=Mock(all=Mock(return_value=[mock_reference]))))
        
        with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_refs.return_value = [mock_reference]
            
            response = client.get("/api/v1/reference?user_id=user123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["references"]) == 1
            assert data["total_count"] == 1
            assert "request_id" in data

    def test_get_references_with_filters(self, client, sample_reference_data):
        mock_reference = create_mock_reference()
        mock_reference.user_id = "123"
        mock_reference.label = "test"
        mock_reference.quality_score = 0.7
        
        mock_db_manager, mock_session = create_mock_db_manager()
        mock_session.execute = AsyncMock(return_value=Mock(scalars=Mock(all=Mock(return_value=[mock_reference]))))
        
        with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_refs.return_value = [mock_reference]
            
            response = client.get("/api/v1/reference?user_id=123&label=test&is_active=true&quality_min=0.5&quality_max=0.9")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["references"]) == 1
            assert data["total_count"] == 1

    def test_get_references_with_pagination(self, client, sample_reference_data):
        mock_reference = create_mock_reference()
        mock_reference.label = sample_reference_data.get("label", "Test Reference")
        mock_reference.file_url = sample_reference_data.get("file_url")
        mock_reference.quality_score = sample_reference_data.get("quality_score", 0.8)
        mock_reference.metadata = sample_reference_data.get("metadata")
        
        all_references = [create_mock_reference() for _ in range(25)]
        
        mock_db_manager, mock_session = create_mock_db_manager()
        mock_session.execute = AsyncMock(return_value=Mock(scalars=Mock(all=Mock(return_value=all_references))))
        
        with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_refs.return_value = all_references
            
            response = client.get("/api/v1/reference?page=2&per_page=10&user_id=user123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["references"]) == 10
            assert data["total_count"] == 25
            assert data["page"] == 2
            assert data["per_page"] == 10
            assert data["has_next"] is True
            assert data["has_prev"] is True

    def test_get_references_empty_list(self, client, sample_reference_data):
        mock_db_manager, mock_session = create_mock_db_manager()
        mock_session.execute = AsyncMock(return_value=Mock(scalars=Mock(all=Mock(return_value=[]))))
        
        with patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
           
            mock_get_refs.return_value = []
            
            response = client.get("/api/v1/reference?user_id=user123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["references"]) == 0
            assert data["total_count"] == 0

    # === GET /api/v1/reference/{reference_id} - Получение конкретного эталона ===

    def test_get_reference_success(self, client, sample_reference_data):
        reference_id = str(uuid.uuid4())
        
        mock_reference = create_mock_reference({
            "id": reference_id, "user_id": "user123",
            "label": sample_reference_data.get("label", "Test Reference"),
            "file_url": sample_reference_data.get("file_url"),
            "quality_score": sample_reference_data.get("quality_score", 0.8),
            "metadata": sample_reference_data.get("metadata"),
        })
        
        mock_db_manager, mock_session = create_mock_db_manager()
        
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_ref.return_value = mock_reference
            
            response = client.get(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["reference_id"] == reference_id
            assert "request_id" in data

    def test_get_reference_not_found(self, client):
        reference_id = str(uuid.uuid4())
        
        mock_db_manager, mock_session = create_mock_db_manager()
        
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_ref.return_value = None
            
            response = client.get(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 404

    # === POST /api/v1/reference - Создание эталона ===

    def test_create_reference_success(self, client, sample_reference_data, sample_image_data):
        reference_id = str(uuid.uuid4())
        created_reference = create_mock_reference({
            "id": reference_id, "user_id": "user123", "label": "Test Reference",
            "file_url": "http://minio/test-image.jpg", "quality_score": 0.85,
            "metadata": {"test": "data"},
        })
        
        mock_db_manager, mock_session = create_mock_db_manager()
        mock_session.execute = AsyncMock(return_value=Mock(scalar_one_or_none=Mock(return_value=None)))
        
        # Мок-экземпляр ValidationService
        mock_validation_instance = Mock()
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_result.error_message = None
        mock_validation_instance.validate_image = AsyncMock(return_value=mock_validation_result)
        mock_validation_instance.aclose = AsyncMock()
        
        mock_storage_instance = Mock()
        mock_storage_instance.upload_image = AsyncMock(return_value={
            "file_url": "http://minio/test-image.jpg", "file_size": 102400, "image_id": "test-image-id"
        })
        
        mock_ml_instance = Mock()
        mock_ml_instance.generate_embedding = AsyncMock(return_value={
            "success": True, "embedding": [0.1, 0.2, 0.3],
            "quality_score": 0.85, "model_version": "facenet-vggface2-optimized"
        })
        
        mock_encryption_instance = Mock()
        mock_encryption_instance.encrypt_embedding = AsyncMock(return_value=b"encrypted_embedding")
        
        with \
             patch('app.services.validation_service.ValidationService', return_value=mock_validation_instance), \
             patch('app.services.storage_service.StorageService', return_value=mock_storage_instance), \
             patch('app.services.ml_service.MLService', return_value=mock_ml_instance), \
             patch('app.services.encryption_service.EncryptionService', return_value=mock_encryption_instance), \
             patch('app.db.crud.ReferenceCRUD.create_reference', new_callable=AsyncMock) as mock_create_ref, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_create_ref.return_value = created_reference
            
            request_data = {
                "user_id": "user123", "label": "Test Reference",
                "image_data": sample_image_data, "quality_threshold": 0.5,
                "metadata": {"test": "data"}
            }
            
            response = client.post("/api/v1/reference", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["label"] == "Test Reference"
            assert "request_id" in data

    def test_create_reference_validation_failed(self, client, sample_image_data):
        mock_db_manager, mock_session = create_mock_db_manager()
        
        mock_validation_instance = Mock()
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.error_message = "Invalid image format"
        mock_validation_instance.validate_image = AsyncMock(return_value=mock_validation_result)
        mock_validation_instance.aclose = AsyncMock()
        
        with patch('app.services.validation_service.ValidationService', return_value=mock_validation_instance), \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            request_data = {
                "user_id": "user123", "label": "Test Reference", "image_data": sample_image_data
            }
            
            response = client.post("/api/v1/reference", json=request_data)
            
            assert response.status_code == 400

    # === PUT /api/v1/reference/{reference_id} - Обновление эталона ===

    def test_update_reference_success(self, client, sample_reference_data):
        reference_id = sample_reference_data["id"]
        
        existing_reference = create_mock_reference({
            "id": reference_id, "user_id": "user123", "label": "Test Reference",
            "file_url": "http://minio/test.jpg", "quality_score": 0.8,
            "metadata": {"test": "data"}, "quality_threshold": 0.8,
        })
        
        updated_reference = create_mock_reference({
            "id": reference_id, "user_id": "user123", "label": "Updated Reference",
            "file_url": "http://minio/test.jpg", "quality_score": 0.8,
            "metadata": {"updated": True}, "quality_threshold": 0.8,
        })
        
        mock_db_manager, mock_session = create_mock_db_manager()
        
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.update_reference', new_callable=AsyncMock) as mock_update_ref, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_ref.return_value = existing_reference
            mock_update_ref.return_value = updated_reference
            
            request_data = {"label": "Updated Reference", "metadata": {"updated": True}}
            
            response = client.put(f"/api/v1/reference/{reference_id}", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["label"] == "Updated Reference"

    def test_update_reference_not_found(self, client):
        reference_id = str(uuid.uuid4())
        
        mock_db_manager, mock_session = create_mock_db_manager()
        
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_ref.return_value = None
            
            request_data = {"label": "Updated Reference"}
            
            response = client.put(f"/api/v1/reference/{reference_id}", json=request_data)
            
            assert response.status_code == 404

    # === DELETE /api/v1/reference/{reference_id} - Удаление эталона ===

    def test_delete_reference_success(self, client, sample_reference_data):
        reference_id = sample_reference_data["id"]
        existing_reference = create_mock_reference({
            "id": reference_id, "file_url": "http://minio/test-image.jpg",
        })
        
        mock_db_manager, mock_session = create_mock_db_manager()
        
        mock_storage_instance = Mock()
        mock_storage_instance.delete_image_by_url = AsyncMock()
        
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.crud.ReferenceCRUD.delete_reference', new_callable=AsyncMock) as mock_delete_ref, \
             patch('app.services.storage_service.StorageService', return_value=mock_storage_instance), \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_ref.return_value = existing_reference
            mock_delete_ref.return_value = True
            
            response = client.delete(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "deleted" in data["message"].lower()

    def test_delete_reference_not_found(self, client):
        reference_id = str(uuid.uuid4())
        
        mock_db_manager, mock_session = create_mock_db_manager()
        
        with patch('app.db.crud.ReferenceCRUD.get_reference_by_id', new_callable=AsyncMock) as mock_get_ref, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_ref.return_value = None
            
            response = client.delete(f"/api/v1/reference/{reference_id}")
            
            assert response.status_code == 404

    # === POST /api/v1/compare - Сравнение с эталонами ===

    def test_compare_with_references_success(self, client, sample_reference_data, sample_image_data):
        reference_data = create_mock_reference({
            "user_id": "user123", "label": "Test Reference",
            "file_url": "http://minio/test.jpg", "quality_score": 0.8,
            "metadata": {"test": "data"},
        })
        
        mock_db_manager, mock_session = create_mock_db_manager()
        
        mock_validation_instance = Mock()
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_instance.validate_image = AsyncMock(return_value=mock_validation_result)
        mock_validation_instance.aclose = AsyncMock()
        
        mock_encryption_instance = Mock()
        mock_encryption_instance.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        mock_ml_instance = Mock()
        mock_ml_instance.compare_faces = AsyncMock(return_value={
            "success": True, "similarity_score": 0.85,
            "distance": 0.15, "processing_time": 0.05
        })
        
        with patch('app.services.validation_service.ValidationService', return_value=mock_validation_instance), \
             patch('app.services.ml_service.MLService', return_value=mock_ml_instance), \
             patch('app.services.encryption_service.EncryptionService', return_value=mock_encryption_instance), \
             patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_refs.return_value = [reference_data]
            
            request_data = {
                "user_id": "user123", "image_data": sample_image_data,
                "threshold": 0.8, "max_results": 10, "quality_threshold": 0.5
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "results" in data
            assert len(data["results"]) == 1

    def test_compare_with_references_no_references(self, client, sample_image_data):
        mock_db_manager, mock_session = create_mock_db_manager()
        
        mock_validation_instance = Mock()
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_instance.validate_image = AsyncMock(return_value=mock_validation_result)
        mock_validation_instance.aclose = AsyncMock()
        
        with patch('app.services.validation_service.ValidationService', return_value=mock_validation_instance), \
             patch('app.db.crud.ReferenceCRUD.get_all_references', new_callable=AsyncMock) as mock_get_refs, \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            mock_get_refs.return_value = []
            
            request_data = {
                "user_id": "user123", "image_data": sample_image_data,
                "threshold": 0.8, "max_results": 10, "quality_threshold": 0.5
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) == 0

    def test_compare_with_references_validation_failed(self, client):
        mock_db_manager, mock_session = create_mock_db_manager()
        
        mock_validation_instance = Mock()
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.error_message = "Invalid image format"
        mock_validation_instance.validate_image = AsyncMock(return_value=mock_validation_result)
        mock_validation_instance.aclose = AsyncMock()
        
        with patch('app.services.validation_service.ValidationService', return_value=mock_validation_instance), \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            request_data = {
                "user_id": "user123", "image_data": "invalid_image_data",
                "threshold": 0.8, "quality_threshold": 0.5
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 400

    def test_compare_with_references_no_user_or_ids(self, client, sample_image_data):
        mock_db_manager, mock_session = create_mock_db_manager()
        mock_session.execute = AsyncMock(return_value=Mock(scalars=Mock(all=Mock(return_value=[]))))
        
        mock_validation_instance = Mock()
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.image_format = "JPEG"
        mock_validation_result.dimensions = {"width": 224, "height": 224}
        mock_validation_instance.validate_image = AsyncMock(return_value=mock_validation_result)
        mock_validation_instance.aclose = AsyncMock()
        
        with patch('app.services.validation_service.ValidationService', return_value=mock_validation_instance), \
             patch('app.db.database.get_async_db_manager', return_value=mock_db_manager):
            
            request_data = {
                "image_data": sample_image_data, "threshold": 0.8, "quality_threshold": 0.5
            }
            
            response = client.post("/api/v1/compare", json=request_data)
            
            assert response.status_code == 200