"""
Тесты для Liveness Routes API.
Критически важный модуль для проверки живости лица.
"""

import pytest
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from datetime import datetime, timezone

# Mock всех внешних зависимостей
with patch('app.routes.liveness.settings'), \
     patch('app.routes.liveness.DatabaseService'), \
     patch('app.routes.liveness.CacheService'), \
     patch('app.routes.liveness.MLService'), \
     patch('app.routes.liveness.ValidationService'):
    
    from app.main import create_test_app
    from app.routes.liveness import router
    from app.models.request import LivenessRequest
    from app.models.response import LivenessResponse
    from app.utils.exceptions import ValidationError, ProcessingError, NotFoundError


class TestLivenessRoutes:
    """Тесты для Liveness Routes API"""
    
    @pytest.fixture
    def app(self):
        """Фикстура для создания FastAPI приложения"""
        app = create_test_app()
        # Подключаем liveness router к тестовому приложению с префиксом
        app.include_router(router, prefix="/api/v1")
        return app
    
    @pytest.fixture
    def liveness_client(self, app):
        """Фикстура для создания тестового клиента с liveness router"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_data(self):
        """Фикстура с образцом данных изображения"""
        import base64
        from PIL import Image
        import io
        
        img = Image.new('RGB', (224, 224), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        return base64.b64encode(img_bytes.getvalue()).decode()
    
    @pytest.fixture
    def mock_validation_service(self):
        """Фикстура для мока ValidationService"""
        with patch('app.routes.liveness.ValidationService') as mock_service:
            yield mock_service.return_value
    
    @pytest.fixture
    def mock_ml_service(self):
        """Фикстура для мока MLService"""
        with patch('app.routes.liveness.MLService') as mock_service:
            yield mock_service.return_value
    
    @pytest.fixture
    def mock_db_service(self):
        """Фикстура для мока DatabaseService"""
        with patch('app.routes.liveness.DatabaseService') as mock_service:
            yield mock_service.return_value
    
    @pytest.fixture
    def mock_cache_service(self):
        """Фикстура для мока CacheService"""
        with patch('app.routes.liveness.CacheService') as mock_service:
            yield mock_service.return_value
    
    # === POST /api/v1/liveness - Основная проверка живости ===
    
    def test_check_liveness_passive_success(self, liveness_client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест успешной пассивной проверки живости"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.quality_score = 0.8
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock эвристического анализа спуфинга
        mock_validation_service.analyze_spoof_signs.return_value = {
            "score": 0.8,
            "flags": []
        }
        
        # Mock ML сервиса
        mock_ml_service.check_liveness.return_value = {
            "success": True,
            "liveness_detected": True,
            "confidence": 0.9,
            "anti_spoofing_score": 0.85,
            "face_detected": True,
            "multiple_faces": False,
            "image_quality": 0.8,
            "recommendations": [],
            "model_version": "liveness-v1"
        }
        
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["liveness_detected"] is True
        assert data["confidence"] == 0.9
        assert data["challenge_type"] == "passive"
        assert data["anti_spoofing_score"] == 0.85
        assert data["face_detected"] is True
        assert data["multiple_faces"] is False
        assert "request_id" in data
    
    def test_check_liveness_failed(self, liveness_client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест неуспешной проверки живости (спуфинг)"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.quality_score = 0.8
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock эвристического анализа спуфинга
        mock_validation_service.analyze_spoof_signs.return_value = {
            "score": 0.3,
            "flags": ["glare", "reflection"]
        }
        
        # Mock ML сервиса - спуфинг обнаружен
        mock_ml_service.check_liveness.return_value = {
            "success": True,
            "liveness_detected": False,
            "confidence": 0.2,
            "anti_spoofing_score": 0.25,
            "face_detected": True,
            "multiple_faces": False,
            "image_quality": 0.8,
            "recommendations": ["Improve lighting", "Remove glare"],
            "model_version": "liveness-v1"
        }
        
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["liveness_detected"] is False
        assert data["confidence"] == 0.2
        assert data["anti_spoofing_score"] == 0.25
        assert len(data["recommendations"]) > 0
    
    def test_check_liveness_active_challenge(self, liveness_client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест активной проверки живости с челленджем"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.quality_score = 0.8
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock эвристического анализа
        mock_validation_service.analyze_spoof_signs.return_value = {
            "score": 0.9,
            "flags": []
        }
        
        # Mock ML сервиса для активной проверки
        mock_ml_service.check_liveness.return_value = {
            "success": True,
            "liveness_detected": True,
            "confidence": 0.95,
            "anti_spoofing_score": 0.9,
            "face_detected": True,
            "multiple_faces": False,
            "image_quality": 0.85,
            "recommendations": [],
            "depth_analysis": {"depth_score": 0.8, "stereo_vision": True},
            "model_version": "liveness-v1"
        }
        
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "active",
            "challenge_data": {
                "rotation_x": 15,
                "rotation_y": 10,
                "instruction": "Turn head slightly"
            }
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["liveness_detected"] is True
        assert data["challenge_type"] == "active"
        assert "depth_analysis" in data
    
    def test_check_liveness_blink_detection(self, liveness_client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест проверки живости с обнаружением моргания"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.quality_score = 0.8
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock эвристического анализа
        mock_validation_service.analyze_spoof_signs.return_value = {
            "score": 0.85,
            "flags": []
        }
        
        # Mock ML сервиса для проверки моргания
        mock_ml_service.check_liveness.return_value = {
            "success": True,
            "liveness_detected": True,
            "confidence": 0.88,
            "anti_spoofing_score": 0.8,
            "face_detected": True,
            "multiple_faces": False,
            "image_quality": 0.8,
            "recommendations": [],
            "blink_detected": True,
            "model_version": "liveness-v1"
        }
        
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "blink"
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["liveness_detected"] is True
        assert data["challenge_type"] == "blink"
    
    def test_check_liveness_no_face_detected(self, liveness_client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест проверки живости когда лицо не обнаружено"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.quality_score = 0.8
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock ML сервиса - лицо не обнаружено
        mock_ml_service.check_liveness.return_value = {
            "success": True,
            "liveness_detected": False,
            "confidence": 0.0,
            "anti_spoofing_score": None,
            "face_detected": False,
            "multiple_faces": False,
            "image_quality": 0.8,
            "recommendations": ["Position face in frame", "Ensure good lighting"],
            "model_version": "liveness-v1"
        }
        
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["liveness_detected"] is False
        assert data["face_detected"] is False
        assert len(data["recommendations"]) > 0
    
    def test_check_liveness_multiple_faces(self, liveness_client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест проверки живости с несколькими лицами"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.quality_score = 0.8
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock ML сервиса - несколько лиц
        mock_ml_service.check_liveness.return_value = {
            "success": True,
            "liveness_detected": False,
            "confidence": 0.1,
            "anti_spoofing_score": 0.3,
            "face_detected": True,
            "multiple_faces": True,
            "image_quality": 0.8,
            "recommendations": ["Ensure only one person in frame"],
            "model_version": "liveness-v1"
        }
        
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["multiple_faces"] is True
        assert data["liveness_detected"] is False
    
    def test_check_liveness_active_challenge_missing_data(self, liveness_client, mock_validation_service, sample_image_data):
        """Тест активной проверки без обязательных данных"""
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "active"  # Требует challenge_data
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "challenge_data is required" in data["detail"]["error_details"]["error"]
    
    def test_check_liveness_validation_failed(self, liveness_client, mock_validation_service, sample_image_data):
        """Тест проверки живости с невалидным изображением"""
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.error_message = "Invalid image format"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "Image validation failed" in data["detail"]["error_details"]["error"]
    
    def test_check_liveness_ml_failed(self, liveness_client, mock_validation_service, mock_ml_service, sample_image_data):
        """Тест проверки живости при ошибке ML сервиса"""
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.quality_score = 0.8
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock ошибки ML сервиса
        mock_ml_service.check_liveness.return_value = {
            "success": False,
            "error": "Model inference failed"
        }
        
        request_data = {
            "session_id": str(uuid.uuid4()),
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post("/api/v1/liveness", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["detail"]["error_code"] == "PROCESSING_ERROR"
        assert "ML liveness check failed" in data["detail"]["error_details"]["error"]
    
    # === POST /api/v1/liveness/session - Создание сессии ===
    
    def test_create_liveness_session_success(self, liveness_client, mock_db_service, mock_cache_service):
        """Тест успешного создания сессии проверки живости"""
        # Mock создания сессии в БД
        session_data = {
            "id": str(uuid.uuid4()),
            "user_id": "user123",
            "session_type": "liveness",
            "status": "pending",
            "expires_at": datetime.now(timezone.utc)
        }
        mock_db_service.create_verification_session.return_value = session_data
        
        # Mock кэша
        mock_cache_service.set_verification_session.return_value = True
        
        request_data = {
            "session_type": "liveness",
            "user_id": "user123",
            "expires_in_minutes": 30
        }
        
        response = liveness_client.post("/api/v1/liveness/session", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_type"] == "liveness"
        assert data["user_id"] == "user123"
        assert "session_id" in data
        assert "expires_at" in data
        assert "request_id" in data
    
    def test_create_liveness_session_invalid_type(self, liveness_client, mock_db_service):
        """Тест создания сессии с неправильным типом"""
        request_data = {
            "session_type": "verification",  # Не liveness
            "user_id": "user123"
        }
        
        response = liveness_client.post("/api/v1/liveness/session", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "Session type must be 'liveness'" in data["detail"]["error_details"]["error"]
    
    # === POST /api/v1/liveness/session/{session_id} - Проверка в сессии ===
    
    @pytest.mark.asyncio
    def test_check_liveness_in_session_success(self, liveness_client, mock_cache_service, mock_validation_service, 
                                              mock_ml_service, sample_image_data):
        """Тест успешной проверки живости в рамках сессии"""
        session_id = str(uuid.uuid4())
        
        # Mock данных сессии
        session_data = {
            "session_id": session_id,
            "user_id": "user123",
            "session_type": "liveness",
            "status": "pending"
        }
        mock_cache_service.get_verification_session.return_value = session_data
        
        # Mock валидации изображения
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.image_data = sample_image_data.encode() if isinstance(sample_image_data, str) else sample_image_data
        mock_validation_result.quality_score = 0.8
        mock_validation_result.image_format = "JPEG"
        mock_validation_service.validate_image.return_value = mock_validation_result
        
        # Mock эвристического анализа
        mock_validation_service.analyze_spoof_signs.return_value = {
            "score": 0.8,
            "flags": []
        }
        
        # Mock ML сервиса
        mock_ml_service.check_liveness.return_value = {
            "success": True,
            "liveness_detected": True,
            "confidence": 0.9,
            "anti_spoofing_score": 0.85,
            "face_detected": True,
            "multiple_faces": False,
            "image_quality": 0.8,
            "recommendations": [],
            "model_version": "liveness-v1"
        }
        
        request_data = {
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post(f"/api/v1/liveness/session/{session_id}", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["liveness_detected"] is True
        assert data["session_id"] == session_id
    
    @pytest.mark.asyncio
    def test_check_liveness_in_session_not_found(self, liveness_client, mock_cache_service, sample_image_data):
        """Тест проверки живости в несуществующей сессии"""
        session_id = "nonexistent_session"
        mock_cache_service.get_verification_session.return_value = None
        
        request_data = {
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post(f"/api/v1/liveness/session/{session_id}", json=request_data)
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "not found or expired" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_check_liveness_in_session_wrong_type(self, liveness_client, mock_cache_service, sample_image_data):
        """Тест проверки живости в сессии неправильного типа"""
        session_id = "verification_session"
        session_data = {
            "session_id": session_id,
            "session_type": "verification",  # Не liveness
            "status": "pending"
        }
        mock_cache_service.get_verification_session.return_value = session_data
        
        request_data = {
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post(f"/api/v1/liveness/session/{session_id}", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "not a liveness session" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_check_liveness_in_session_not_active(self, liveness_client, mock_cache_service, sample_image_data):
        """Тест проверки живости в неактивной сессии"""
        session_id = "inactive_session"
        session_data = {
            "session_id": session_id,
            "session_type": "liveness",
            "status": "completed"
        }
        mock_cache_service.get_verification_session.return_value = session_data
        
        request_data = {
            "image_data": sample_image_data,
            "challenge_type": "passive"
        }
        
        response = liveness_client.post(f"/api/v1/liveness/session/{session_id}", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "not active" in data["detail"]["error_details"]["error"]
    
    # === GET /api/v1/liveness/challenges - Получение доступных челленджей ===
    
    def test_get_available_challenges(self, liveness_client):
        """Тест получения списка доступных челленджей"""
        response = liveness_client.get("/api/v1/liveness/challenges")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "challenges" in data
        assert "default_challenge" in data
        assert data["default_challenge"] == "passive"
        
        challenges = data["challenges"]
        assert "passive" in challenges
        assert "active" in challenges
        assert "blink" in challenges
        assert "smile" in challenges
        assert "turn_head" in challenges
        
        # Проверяем структуру челленджа
        passive_challenge = challenges["passive"]
        assert "name" in passive_challenge
        assert "description" in passive_challenge
        assert "required_data" in passive_challenge
        assert "difficulty" in passive_challenge
        assert "processing_time" in passive_challenge
    
    # === GET /api/v1/liveness/session/{session_id} - Получение статуса сессии ===
    
    @pytest.mark.asyncio
    def test_get_liveness_session_status(self, liveness_client, mock_cache_service):
        """Тест получения статуса сессии проверки живости"""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "session_type": "liveness",
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc),
            "user_id": "user123",
            "metadata": {"test": "data"}
        }
        mock_cache_service.get_verification_session.return_value = session_data
        
        response = liveness_client.get(f"/api/v1/liveness/session/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == session_id
        assert data["session_type"] == "liveness"
        assert data["status"] == "pending"
        assert data["user_id"] == "user123"
        assert "request_id" in data
    
    @pytest.mark.asyncio
    def test_get_liveness_session_status_completed(self, liveness_client, mock_cache_service):
        """Тест получения статуса завершенной сессии с результатами"""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "session_type": "liveness",
            "status": "completed",
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc),
            "user_id": "user123",
            "response_data": {
                "success": True,
                "liveness_detected": True,
                "confidence": 0.9
            }
        }
        mock_cache_service.get_verification_session.return_value = session_data
        
        response = liveness_client.get(f"/api/v1/liveness/session/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "completed"
        assert "results" in data
        assert data["results"]["liveness_detected"] is True
    
    @pytest.mark.asyncio
    def test_get_liveness_session_status_not_found(self, liveness_client, mock_cache_service):
        """Тест получения статуса несуществующей сессии"""
        session_id = "nonexistent_session"
        mock_cache_service.get_verification_session.return_value = None
        
        response = liveness_client.get(f"/api/v1/liveness/session/{session_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "not found" in data["detail"]["error_details"]["error"]
    
    @pytest.mark.asyncio
    def test_get_liveness_session_status_wrong_type(self, liveness_client, mock_cache_service):
        """Тест получения статуса сессии неправильного типа"""
        session_id = "verification_session"
        session_data = {
            "session_id": session_id,
            "session_type": "verification",  # Не liveness
            "status": "pending"
        }
        mock_cache_service.get_verification_session.return_value = session_data
        
        response = liveness_client.get(f"/api/v1/liveness/session/{session_id}")
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "VALIDATION_ERROR"
        assert "not a liveness session" in data["detail"]["error_details"]["error"]
    
    # === GET /api/v1/liveness/{session_id} - Получение результата ===
    
    def test_get_liveness_result_success(self, liveness_client):
        """Тест успешного получения результата проверки живости"""
        from app.db.crud import VerificationSessionCRUD
        
        session_id = str(uuid.uuid4())
        
        # Mock CRUD метод и менеджер БД
        with patch('app.db.crud.VerificationSessionCRUD.get_session') as mock_get_session, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            # Создаем mock сессии
            mock_session = Mock()
            mock_session.session_id = session_id
            mock_session.user_id = "user123"
            mock_session.session_type = "liveness"
            mock_session.status = "success"
            mock_session.is_liveness_passed = True
            mock_session.liveness_score = 0.9
            mock_session.liveness_method = "passive"
            mock_session.confidence = 0.9
            mock_session.face_detected = True
            mock_session.face_quality_score = 0.8
            mock_session.processing_time = 0.5
            mock_session.created_at = datetime.now(timezone.utc)
            mock_session.completed_at = datetime.now(timezone.utc)
            mock_session.error_code = None
            mock_session.error_message = None
            
            mock_get_session.return_value = mock_session
            mock_db_instance = Mock()
            mock_db_manager.return_value = mock_db_instance
            mock_db_instance.get_session.return_value.__aenter__ = AsyncMock()
            mock_db_instance.get_session.return_value.__aexit__ = AsyncMock()
            
            response = liveness_client.get(f"/api/v1/liveness/{session_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == session_id
            assert data["is_live"] is True
            assert data["liveness_score"] == 0.9
            assert data["liveness_method"] == "passive"
            assert data["confidence"] == 0.9
            assert data["status"] == "success"
            assert "request_id" in data
    
    def test_get_liveness_result_not_found(self, liveness_client):
        """Тест получения результата несуществующей сессии"""
        session_id = "nonexistent_session"
        
        # Mock CRUD метод и менеджер БД
        with patch('app.db.crud.VerificationSessionCRUD.get_session') as mock_get_session, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            mock_get_session.return_value = None
            mock_db_instance = Mock()
            mock_db_manager.return_value = mock_db_instance
            mock_db_instance.get_session.return_value.__aenter__ = AsyncMock()
            mock_db_instance.get_session.return_value.__aexit__ = AsyncMock()
            
            response = liveness_client.get(f"/api/v1/liveness/{session_id}")
            
            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["error_code"] == "SESSION_NOT_FOUND"
            assert "not found" in data["detail"]["error_details"]["error"]
    
    def test_get_liveness_result_wrong_type(self, liveness_client):
        """Тест получения результата сессии неправильного типа"""
        session_id = "verification_session"
        
        # Mock CRUD метод и менеджер БД
        with patch('app.db.crud.VerificationSessionCRUD.get_session') as mock_get_session, \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            # Создаем mock сессии неправильного типа
            mock_session = Mock()
            mock_session.session_id = session_id
            mock_session.session_type = "verification"  # Не liveness
            mock_get_session.return_value = mock_session
            mock_db_instance = Mock()
            mock_db_manager.return_value = mock_db_instance
            mock_db_instance.get_session.return_value.__aenter__ = AsyncMock()
            mock_db_instance.get_session.return_value.__aexit__ = AsyncMock()
            
            response = liveness_client.get(f"/api/v1/liveness/{session_id}")
            
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error_code"] == "VALIDATION_ERROR"
            assert "not a liveness session" in data["detail"]["error_details"]["error"]


# === ИНТЕГРАЦИОННЫЕ ТЕСТЫ ===

class TestLivenessRoutesIntegration:
    """Интеграционные тесты для Liveness Routes"""
    
    @pytest.mark.asyncio
    async def test_full_liveness_workflow(self):
        """Тест полного workflow проверки живости"""
        app = create_test_app()
        # Подключаем liveness router к тестовому приложению с префиксом
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)
        
        with patch('app.routes.liveness.settings'), \
             patch('app.routes.liveness.DatabaseService') as mock_db, \
             patch('app.routes.liveness.ValidationService') as mock_validation, \
             patch('app.routes.liveness.MLService') as mock_ml, \
             patch('app.routes.liveness.CacheService') as mock_cache:
            
            # Настройка моков
            db_service = mock_db.return_value
            validation_service = mock_validation.return_value
            ml_service = mock_ml.return_value
            cache_service = mock_cache.return_value
            
            # 1. Создание сессии
            cache_service.set_verification_session.return_value = True
            
            session_response = client.post("/api/v1/liveness/session", json={
                "session_type": "liveness",
                "user_id": "user123"
            })
            assert session_response.status_code == 200
            session_id = session_response.json()["session_id"]
            
            # 2. Проверка живости в сессии
            validation_service.validate_image.return_value = Mock(
                is_valid=True,
                image_data=b"test_image_data",
                quality_score=0.8,
                image_format="JPEG"
            )
            validation_service.analyze_spoof_signs.return_value = {
                "score": 0.8,
                "flags": []
            }
            
            ml_service.check_liveness.return_value = {
                "success": True,
                "liveness_detected": True,
                "confidence": 0.9,
                "anti_spoofing_score": 0.85,
                "face_detected": True,
                "multiple_faces": False,
                "image_quality": 0.8,
                "recommendations": [],
                "model_version": "liveness-v1"
            }
            
            liveness_response = client.post(f"/api/v1/liveness/session/{session_id}", json={
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD",
                "challenge_type": "passive"
            })
            assert liveness_response.status_code == 200
            liveness_data = liveness_response.json()
            assert liveness_data["liveness_detected"] is True
            
            # 3. Получение доступных челленджей
            challenges_response = client.get("/api/v1/liveness/challenges")
            assert challenges_response.status_code == 200
            challenges_data = challenges_response.json()
            assert "passive" in challenges_data["challenges"]
            assert challenges_data["default_challenge"] == "passive"
