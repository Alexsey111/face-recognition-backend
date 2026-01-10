"""
Тесты для Verify Routes API.
Критически важный модуль для верификации лиц.
"""

import pytest
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from datetime import datetime, timezone

# Mock всех внешних зависимостей при импорте
with patch('app.routes.verify.settings'), \
     patch('app.routes.verify.DatabaseService'), \
     patch('app.routes.verify.CacheService'), \
     patch('app.routes.verify.MLService'), \
     patch('app.routes.verify.ValidationService'), \
     patch('app.routes.verify.EncryptionService'), \
     patch('app.routes.verify.StorageService'):

    from app.main import create_test_app
    from app.routes.verify import router
    from app.models.request import VerifyRequest
    from app.models.response import VerifyResponse
    from app.utils.exceptions import ValidationError, ProcessingError, NotFoundError


class TestVerifyRoutes:
    """Тесты для Verify Routes API"""

    @pytest.fixture
    def app(self):
        """Фикстура для создания FastAPI приложения"""
        return create_test_app()

    @pytest.fixture
    def client(self, app):
        """Фикстура для создания тестового клиента"""
        return TestClient(app)

    @pytest.fixture
    def sample_image_data(self):
        """Фикстура с образцом данных изображения (валидное base64 JPEG)"""
        import base64
        from PIL import Image
        import io

        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        base64_data = base64.b64encode(img_bytes.getvalue()).decode()
        return f"data:image/jpeg;base64,{base64_data}"

    @pytest.fixture
    def valid_validation_result(self, sample_image_data):
        """Фикстура: результат успешной валидации изображения"""
        result = Mock()
        result.is_valid = True
        result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        result.quality_score = 0.8
        result.image_format = "JPEG"
        return result

    @pytest.fixture
    def mock_validation_service(self):
        with patch('app.routes.verify.ValidationService') as mock_service:
            yield mock_service.return_value

    @pytest.fixture
    def mock_ml_service(self):
        with patch('app.routes.verify.MLService') as mock_service:
            yield mock_service.return_value

    @pytest.fixture
    def mock_db_service(self):
        with patch('app.routes.verify.DatabaseService') as mock_service:
            yield mock_service.return_value

    @pytest.fixture
    def mock_cache_service(self):
        with patch('app.routes.verify.CacheService') as mock_service:
            yield mock_service.return_value

    @pytest.fixture
    def mock_encryption_service(self):
        with patch('app.routes.verify.EncryptionService') as mock_service:
            yield mock_service.return_value

    @pytest.fixture
    def sample_reference_data(self):
        return {
            "id": str(uuid.uuid4()),
            "user_id": "user123",
            "label": "Test Reference",
            "embedding": b"encrypted_embedding",
            "quality_score": 0.85,
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    # === POST /api/v1/verify - Основная верификация ===

    @pytest.mark.asyncio
    async def test_verify_face_success(
        self, client, mock_validation_service, mock_ml_service,
        mock_db_service, mock_encryption_service, sample_image_data, sample_reference_data
    ):
        """Тест успешной верификации лица"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data,
            quality_score=0.8,
            image_format="JPEG"
        ))

        mock_reference = Mock(
            id=sample_reference_data["id"],
            user_id=sample_reference_data["user_id"],
            label=sample_reference_data["label"],
            embedding=sample_reference_data["embedding"],
            is_active=True
        )
        mock_db_service.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        mock_ml_service.verify_face = AsyncMock(return_value={
            "success": True,
            "verified": True,
            "confidence": 0.9,
            "similarity_score": 0.85,
            "face_detected": True,
            "face_quality": 0.8,
            "model_version": "facenet-vggface2"
        })

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "reference_id": sample_reference_data["id"],
            "threshold": 0.8,
            "auto_enroll": False
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["verified"] is True
        assert data["confidence"] == 0.9
        assert data["similarity_score"] == 0.85
        assert data["threshold_used"] == 0.8
        assert data["face_detected"] is True
        assert "request_id" in data

    @pytest.mark.asyncio
    async def test_verify_face_no_match(
        self, client, mock_validation_service, mock_ml_service,
        mock_db_service, mock_encryption_service, sample_image_data, sample_reference_data
    ):
        """Тест верификации без совпадения"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode(),
            quality_score=0.8,
            image_format="JPEG"
        ))

        mock_reference = Mock(
            id=sample_reference_data["id"],
            user_id=sample_reference_data["user_id"],
            label=sample_reference_data["label"],
            embedding=sample_reference_data["embedding"],
            is_active=True
        )
        mock_db_service.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        mock_ml_service.verify_face = AsyncMock(return_value={
            "success": True,
            "verified": False,
            "confidence": 0.3,
            "similarity_score": 0.4,
            "face_detected": True,
            "face_quality": 0.7,
            "model_version": "facenet-vggface2"
        })

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "reference_id": sample_reference_data["id"],
            "threshold": 0.8
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["verified"] is False
        assert data["confidence"] == 0.3
        assert data["similarity_score"] == 0.4

    @pytest.mark.asyncio
    async def test_verify_face_with_user_id(
        self, client, mock_validation_service, mock_ml_service,
        mock_db_service, mock_encryption_service, sample_image_data, sample_reference_data
    ):
        """Тест верификации с user_id вместо reference_id"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode(),
            quality_score=0.8,
            image_format="JPEG"
        ))

        mock_reference = Mock(
            id=sample_reference_data["id"],
            user_id=sample_reference_data["user_id"],
            label=sample_reference_data["label"],
            embedding=sample_reference_data["embedding"],
            is_active=True
        )
        mock_db_service.get_active_references_by_user = AsyncMock(return_value=[mock_reference])
        mock_encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        mock_ml_service.verify_face = AsyncMock(return_value={
            "success": True,
            "verified": True,
            "confidence": 0.9,
            "similarity_score": 0.85,
            "face_detected": True,
            "face_quality": 0.8,
            "model_version": "facenet-vggface2"
        })

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "user_id": "user123",
            "threshold": 0.8
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["verified"] is True
        mock_db_service.get_active_references_by_user.assert_called_once_with("user123")

    @pytest.mark.asyncio
    async def test_verify_face_no_reference_found(self, client, mock_validation_service, mock_db_service, sample_image_data):
        """Тест верификации когда эталон не найден"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode(),
            quality_score=0.8,
            image_format="JPEG"
        ))
        mock_db_service.get_reference_by_id = AsyncMock(return_value=None)

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "reference_id": "nonexistent_id"
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == 404  # HTTP статус код, а не строковый код
        assert "not found" in data["error_details"]["error"]

    @pytest.mark.asyncio
    async def test_verify_face_no_user_references(self, client, mock_validation_service, mock_db_service, sample_image_data):
        """Тест верификации когда у пользователя нет эталонов"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode(),
            quality_score=0.8,
            image_format="JPEG"
        ))
        mock_db_service.get_active_references_by_user = AsyncMock(return_value=[])

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "user_id": "user_without_references"
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == 404
        assert "No active references found" in data["error_details"]["error"]

    @pytest.mark.asyncio
    async def test_verify_face_inactive_reference(self, client, mock_validation_service, mock_db_service, sample_image_data):
        """Тест верификации с неактивным эталоном"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode(),
            quality_score=0.8,
            image_format="JPEG"
        ))

        mock_reference = Mock(id="ref123", is_active=False)
        mock_db_service.get_reference_by_id = AsyncMock(return_value=mock_reference)

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "reference_id": "inactive_ref"
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == 400
        assert "not active" in data["error_details"]["error"]

    @pytest.mark.asyncio
    async def test_verify_face_no_reference_id_or_user_id(self, client, mock_validation_service, sample_image_data):
        """Тест верификации без reference_id и user_id"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode(),
            quality_score=0.8,
            image_format="JPEG"
        ))

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == 400
        assert "Either reference_id or user_id must be provided" in data["error_details"]["error"]

    @pytest.mark.asyncio
    async def test_verify_face_validation_failed(self, client, mock_validation_service, sample_image_data):
        """Тест верификации с невалидным изображением"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=False,
            error_message="Invalid image format"
        ))

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "reference_id": "ref123"
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == 400
        assert "Image validation failed" in data["error_details"]["error"]

    @pytest.mark.asyncio
    async def test_verify_face_ml_failed(
        self, client, mock_validation_service, mock_ml_service,
        mock_db_service, mock_encryption_service, sample_image_data, sample_reference_data
    ):
        """Тест верификации при ошибке ML сервиса"""
        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode(),
            quality_score=0.8,
            image_format="JPEG"
        ))

        mock_reference = Mock(
            id=sample_reference_data["id"],
            embedding=sample_reference_data["embedding"],
            is_active=True
        )
        mock_db_service.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        mock_ml_service.verify_face = AsyncMock(return_value={
            "success": False,
            "error": "Face detection failed"
        })

        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "reference_id": sample_reference_data["id"]
        }

        response = client.post("/api/v1/verify/", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == 422
        assert "ML verification failed" in data["error_details"]["error"]

    # === POST /api/v1/verify/session - Создание сессии ===

    @pytest.mark.asyncio
    async def test_create_verification_session_success(self, client, mock_db_service, mock_cache_service):
        """Тест успешного создания сессии верификации"""
        with patch('app.db.crud.VerificationSessionCRUD') as mock_crud, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:

            # Mock database session context manager
            mock_async_session = AsyncMock()
            mock_async_session.__aenter__ = AsyncMock(return_value=mock_async_session)
            mock_async_session.__aexit__ = AsyncMock(return_value=None)
            mock_db_manager.return_value.get_session.return_value = mock_async_session

            # Mock CRUD create_session
            session_id = str(uuid.uuid4())
            mock_crud.create_session = AsyncMock()
            
            # Mock cache service
            mock_cache_service.set_verification_session = AsyncMock(return_value=True)

            request_data = {
                "session_type": "verification",
                "user_id": "user123",
                "expires_in_minutes": 30
            }

            response = client.post("/api/v1/verify/session", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_type"] == "verification"
            assert data["user_id"] == "user123"
            assert "session_id" in data  # Просто проверяем, что session_id есть (генерируется автоматически)
            assert "expires_at" in data
            assert "request_id" in data

    @pytest.mark.asyncio
    async def test_create_verification_session_invalid_type(self, client, mock_db_service):
        """Тест создания сессии с невалидным типом (обрабатывается валидатором Pydantic до роута)"""
        request_data = {
            "session_type": "invalid_type",
            "user_id": "user123"
        }

        response = client.post("/api/v1/verify/session", json=request_data)

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    # === POST /api/v1/verify/session/{session_id} - Верификация в сессии ===

    @pytest.mark.asyncio
    async def test_verify_in_session_success(
        self, client, mock_cache_service, mock_validation_service,
        mock_ml_service, mock_db_service, mock_encryption_service,
        sample_image_data, sample_reference_data
    ):
        """Тест успешной верификации в рамках сессии"""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": "user123",
            "session_type": "verification",
            "status": "pending",
            "reference_id": sample_reference_data["id"]
        }
        mock_cache_service.get_verification_session = AsyncMock(return_value=session_data)
        mock_cache_service.set_verification_session = AsyncMock()

        mock_validation_service.validate_image = AsyncMock(return_value=Mock(
            is_valid=True,
            image_data=sample_image_data.encode(),
            quality_score=0.8,
            image_format="JPEG"
        ))

        # Create a simple mock reference object that can be serialized (not unittest.mock.Mock)
        class MockReference:
            def __init__(self):
                self.id = sample_reference_data["id"]
                self.embedding = sample_reference_data["embedding"]
                self.is_active = True
                self.user_id = "user123"
                self.label = "Test Reference"
                self.version = 1

        mock_reference = MockReference()
        mock_db_service.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_encryption_service.decrypt_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        mock_ml_service.verify_face = AsyncMock(return_value={
            "success": True,
            "verified": True,
            "confidence": 0.9,
            "similarity_score": 0.85,
            "face_detected": True,
            "face_quality": 0.8,
            "model_version": "facenet-vggface2"
        })

        request_data = {
            "image_data": sample_image_data,
            "threshold": 0.8
        }

        response = client.post(f"/api/v1/verify/session/{session_id}", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["verified"] is True
        assert data["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_verify_in_session_not_found(self, client, mock_cache_service, sample_image_data):
        """Тест верификации в несуществующей сессии"""
        mock_cache_service.get_verification_session = AsyncMock(return_value=None)

        request_data = {
            "image_data": sample_image_data
        }

        response = client.post(f"/api/v1/verify/session/nonexistent_session", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == 404
        assert "not found or expired" in data["error_details"]["error"]

    @pytest.mark.asyncio
    async def test_verify_in_session_not_active(self, client, mock_cache_service, sample_image_data):
        """Тест верификации в неактивной сессии"""
        session_id = "inactive_session"
        session_data = {
            "session_id": session_id,
            "status": "completed",  # ← не pending
            "user_id": "user123",
            "session_type": "verification",
            "reference_id": "ref123"
        }
        mock_cache_service.get_verification_session = AsyncMock(return_value=session_data)

        request_data = {
            "image_data": sample_image_data,
            "threshold": 0.8,
        }

        response = client.post(f"/api/v1/verify/session/{session_id}", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == 400
        assert f"Verification session {session_id} is not active" in data["error_details"]["error"]

    # === GET /api/v1/verify/{session_id} - Получение результата ===

    @pytest.mark.asyncio
    async def test_get_verification_result_success(self, client, sample_reference_data):
        """Тест успешного получения результата верификации"""
        from app.db.crud import VerificationSessionCRUD

        session_id = str(uuid.uuid4())

        class MockSession:
            def __init__(self, sid, ref_id):
                self.session_id = sid
                self.user_id = "user123"
                self.reference_id = ref_id
                self.session_type = "verification"
                self.status = "success"
                self.is_match = True
                self.similarity_score = 0.85
                self.confidence = 0.9
                self.face_detected = True
                self.face_quality_score = 0.8
                self.processing_time = 0.5
                self.created_at = datetime.now(timezone.utc)
                self.started_at = datetime.now(timezone.utc)
                self.completed_at = datetime.now(timezone.utc)
                self.error_code = None
                self.error_message = None

        mock_session = MockSession(session_id, sample_reference_data["id"])

        with patch('app.db.crud.VerificationSessionCRUD.get_session', return_value=mock_session), \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:

            mock_db_manager.return_value.get_session.return_value.__aenter__ = AsyncMock()
            mock_db_manager.return_value.get_session.return_value.__aexit__ = AsyncMock()

            response = client.get(f"/api/v1/verify/{session_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == session_id
            assert data["verified"] is True
            assert data["similarity_score"] == 0.85
            assert data["confidence"] == 0.9
            assert data["status"] == "success"
            assert "request_id" in data

    @pytest.mark.asyncio
    async def test_get_verification_result_not_found(self, client):
        """Тест получения результата несуществующей сессии"""
        with patch('app.db.crud.VerificationSessionCRUD.get_session', return_value=None), \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:

            mock_db_manager.return_value.get_session.return_value.__aenter__ = AsyncMock()
            mock_db_manager.return_value.get_session.return_value.__aexit__ = AsyncMock()

            response = client.get(f"/api/v1/verify/nonexistent_session")

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == 404
            assert "not found" in data["error_details"]["error"]

    @pytest.mark.asyncio
    async def test_get_verification_result_wrong_type(self, client):
        """Тест получения результата сессии неправильного типа"""
        class MockSession:
            def __init__(self):
                self.session_id = "liveness_session"
                self.session_type = "liveness"
                self.user_id = "user123"
                self.reference_id = "ref123"
                self.status = "completed"
                self.is_match = True
                self.similarity_score = 0.85
                self.confidence = 0.9
                self.face_detected = True
                self.face_quality_score = 0.8
                self.processing_time = 0.5
                self.created_at = datetime.now(timezone.utc)
                self.completed_at = datetime.now(timezone.utc)
                self.error_code = None
                self.error_message = None

        mock_session = MockSession()

        with patch('app.db.crud.VerificationSessionCRUD.get_session', return_value=mock_session), \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:

            mock_db_manager.return_value.get_session.return_value.__aenter__ = AsyncMock()
            mock_db_manager.return_value.get_session.return_value.__aexit__ = AsyncMock()

            response = client.get(f"/api/v1/verify/liveness_session")

            assert response.status_code == 400
            data = response.json()
            assert data["error_code"] == 400
            assert "not a verification session" in data["error_details"]["error"]

    # === GET /api/v1/verify/history - История верификации ===

    @pytest.mark.asyncio
    async def test_get_verification_history_success(self, client):
        """Тест успешного получения истории верификации"""
        class MockSession:
            def __init__(self, sid, is_match, sim_score, conf):
                self.session_id = sid
                self.user_id = "user123"
                self.reference_id = f"ref_{sid}"
                self.status = "success"
                self.is_match = is_match
                self.similarity_score = sim_score
                self.confidence = conf
                self.face_detected = True
                self.face_quality_score = 0.8
                self.processing_time = 0.5
                self.created_at = datetime.now(timezone.utc)
                self.completed_at = datetime.now(timezone.utc)
                self.error_code = None
                self.error_message = None

        sessions = [
            MockSession("session1", True, 0.85, 0.9),
            MockSession("session2", False, 0.3, 0.4)
        ]

        with patch('app.db.crud.VerificationSessionCRUD.get_user_sessions', return_value=sessions), \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:

            mock_db_manager.return_value.get_session.return_value.__aenter__ = AsyncMock()
            mock_db_manager.return_value.get_session.return_value.__aexit__ = AsyncMock()

            response = client.get("/api/v1/verify/history?user_id=user123&limit=10&offset=0")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["sessions"]) == 2
            assert data["total_count"] == 2
            assert data["limit"] == 10
            assert data["offset"] == 0
            assert "request_id" in data

    @pytest.mark.asyncio
    async def test_get_verification_history_with_filters(self, client):
        class MockSession:
            def __init__(self):
                self.session_id = "session1"
                self.user_id = "user123"
                self.reference_id = "ref123"
                self.status = "success"
                self.is_match = True
                self.similarity_score = 0.85
                self.confidence = 0.9
                self.face_detected = True
                self.face_quality_score = 0.8
                self.processing_time = 0.5
                self.created_at = datetime.now(timezone.utc)
                self.completed_at = datetime.now(timezone.utc)
                self.error_code = None
                self.error_message = None

        with patch('app.db.crud.VerificationSessionCRUD.get_user_sessions', return_value=[MockSession()]), \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:

            mock_db_manager.return_value.get_session.return_value.__aenter__ = AsyncMock()
            mock_db_manager.return_value.get_session.return_value.__aexit__ = AsyncMock()

            response = client.get(
                "/api/v1/verify/history?user_id=user123&status=success&verified=true&limit=5&offset=0"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["sessions"]) == 1
            assert data["filters_applied"]["user_id"] == "user123"
            assert data["filters_applied"]["status"] == "success"
            assert data["filters_applied"]["verified"] is True

    @pytest.mark.asyncio
    async def test_get_verification_history_invalid_date_format(self, client):
        """Тест получения истории с невалидным форматом даты"""
        response = client.get("/api/v1/verify/history?date_from=invalid_date")

        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == 400
        assert "Invalid date_from format" in data["error_details"]["error"]

    @pytest.mark.asyncio
    async def test_get_verification_history_invalid_status(self, client):
        """Тест получения истории с невалидным статусом"""
        response = client.get("/api/v1/verify/history?status=invalid_status")

        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == 400
        assert "Invalid status" in data["error_details"]["error"]
        assert "Must be one of:" in data["error_details"]["error"]

    # === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

    def test_choose_threshold(self):
        """Тест функции выбора порога"""
        from app.routes.verify import _choose_threshold

        # Пользовательский порог
        threshold = _choose_threshold(
            requested_threshold=0.7,
            quality_score=0.8,
            default=0.6,
            min_thr=0.5,
            max_thr=0.9
        )
        assert threshold == 0.7

        # Адаптивный (низкое качество → выше порог)
        threshold = _choose_threshold(
            requested_threshold=None,
            quality_score=0.5,
            default=0.6,
            min_thr=0.5,
            max_thr=0.9
        )
        assert 0.6 <= threshold <= 0.9  # quality_score < default → порог ↑

        # Адаптивный (высокое качество → ниже порог)
        threshold = _choose_threshold(
            requested_threshold=None,
            quality_score=0.9,
            default=0.6,
            min_thr=0.5,
            max_thr=0.9
        )
        assert 0.5 <= threshold < 0.6  # quality_score > default → порог ↓

    def test_confidence_level(self):
        """Тест функции определения уровня уверенности"""
        from app.routes.verify import _confidence_level

        assert _confidence_level(0.9) == "high"
        assert _confidence_level(0.7) == "medium"
        assert _confidence_level(0.4) == "low"
        assert _confidence_level(0.2) == "very_low"


# === ИНТЕГРАЦИОННЫЕ ТЕСТЫ ===

class TestVerifyRoutesIntegration:
    """Интеграционные тесты для Verify Routes"""

    @pytest.fixture
    def sample_image_data(self):
        """Фикстура с образцом данных изображения (валидное base64 JPEG)"""
        import base64
        from PIL import Image
        import io

        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        base64_data = base64.b64encode(img_bytes.getvalue()).decode()
        return f"data:image/jpeg;base64,{base64_data}"

    @pytest.mark.asyncio
    async def test_full_verification_workflow(self, sample_image_data):
        """Тест полного workflow верификации"""
        # Create a simple mock reference object that can be serialized
        class MockReference:
            def __init__(self):
                self.id = "ref123"
                self.user_id = "user123"
                self.label = "Test Reference"
                self.embedding = b"encrypted_embedding"
                self.is_active = True
                self.version = 1

        with patch('app.routes.verify.settings'), \
             patch('app.routes.verify.DatabaseService') as mock_db, \
             patch('app.routes.verify.CacheService') as mock_cache, \
             patch('app.routes.verify.ValidationService') as mock_val, \
             patch('app.routes.verify.MLService') as mock_ml, \
             patch('app.routes.verify.EncryptionService') as mock_enc, \
             patch('app.routes.verify.WebhookService') as mock_webhook, \
             patch('app.db.crud.VerificationSessionCRUD') as mock_crud, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:

            # Setup mocks
            db_service = mock_db.return_value
            cache_service = mock_cache.return_value
            validation_service = mock_val.return_value
            ml_service = mock_ml.return_value
            enc_service = mock_enc.return_value
            webhook_service = mock_webhook.return_value
            webhook_service.send_verification_result = AsyncMock()

            # Mock session creation
            session_id = str(uuid.uuid4())
            session_data = {
                "id": session_id,
                "user_id": "user123",
                "session_type": "verification",
                "status": "pending",
                "expires_at": datetime.now(timezone.utc).isoformat()
            }
            db_service.create_verification_session = AsyncMock(return_value=session_data)
            cache_service.set_verification_session = AsyncMock()

            # Mock async CRUD operations
            mock_crud.create_session = AsyncMock(return_value=session_data)
            mock_crud.get_session = AsyncMock(return_value=None)
            mock_crud.get_user_sessions = AsyncMock(return_value=[])
            
            # Mock database session context manager
            mock_async_session = AsyncMock()
            mock_async_session.__aenter__ = AsyncMock(return_value=mock_async_session)
            mock_async_session.__aexit__ = AsyncMock(return_value=None)
            mock_db_manager.return_value.get_session.return_value = mock_async_session

            # Mock cache service methods
            session_cache_data = {
                "session_id": session_id,
                "user_id": "user123",
                "session_type": "verification",
                "status": "pending",
                "reference_id": "ref123"
            }
            cache_service.get_verification_session = AsyncMock(return_value=session_cache_data)
            cache_service.set_verification_session = AsyncMock()

            # Mock validation
            validation_service.validate_image = AsyncMock(return_value=Mock(
                is_valid=True,
                image_data=b"fake_image_bytes",
                quality_score=0.8,
                image_format="JPEG"
            ))

            # Mock reference with proper object (not Mock)
            mock_ref = MockReference()
            db_service.get_reference_by_id = AsyncMock(return_value=mock_ref)
            enc_service.decrypt_embedding = AsyncMock(return_value=[0.1] * 512)

            # Mock ML - Упрощаем максимально
            ml_service.verify_face = AsyncMock(return_value={
                "success": True,
                "verified": True,
                "confidence": 1.0,
                "similarity_score": 1.0,
                "face_detected": True,
                "face_quality": 1.0,
                "model_version": "facenet"
            })

            # Use create_test_app like other tests
            app = create_test_app()
            client = TestClient(app)

            # Step 1: Create session
            resp1 = client.post("/api/v1/verify/session", json={
                "session_type": "verification",
                "user_id": "user123",
                "expires_in_minutes": 30
            })
            assert resp1.status_code == 200
            sid = resp1.json()["session_id"]

            # Step 2: Verify in session (use valid image data)
            resp2 = client.post(f"/api/v1/verify/session/{sid}", json={
                "image_data": sample_image_data  # Use valid image data from fixture
            })
            assert resp2.status_code == 200
            data = resp2.json()
            assert data["success"] is True
            assert data["verified"] is True
            assert data["session_id"] == sid
            
            