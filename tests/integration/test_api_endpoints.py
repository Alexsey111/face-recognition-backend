import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import create_app
from app.config import Config


class TestHealthEndpoints:
    """Интеграционные тесты для health check endpoints"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_health_check(self):
        """Тест основного health check"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_health_detailed(self):
        """Тест детального health check"""
        response = self.client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert "database" in data["services"]
        assert "redis" in data["services"]
        assert "minio" in data["services"]
    
    def test_health_ready(self):
        """Тест readiness probe"""
        response = self.client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert data["ready"] is True


class TestReferenceEndpoints:
    """Интеграционные тесты для эталонных изображений"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_create_reference_unauthorized(self):
        """Тест создания эталонного изображения без авторизации"""
        response = self.client.post("/api/v1/reference")
        
        assert response.status_code == 401
    
    def test_create_reference_invalid_data(self):
        """Тест создания эталонного изображения с невалидными данными"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = self.client.post(
            "/api/v1/reference",
            headers=headers,
            json={"invalid": "data"}
        )
        
        assert response.status_code in [401, 422]  # Unauthorized или Validation Error
    
    @patch('app.routes.reference.ReferenceCRUD')
    def test_create_reference_success(self, mock_crud):
        """Тест успешного создания эталонного изображения"""
        # Мок успешного создания
        mock_crud.return_value.create.return_value = Mock(
            id="550e8400-e29b-41d4-a716-446655440000",
            user_id="550e8400-e29b-41d4-a716-446655440001",
            label="Test Reference",
            is_active=True
        )
        
        headers = {"Authorization": "Bearer sk_test_1234567890abcdef"}
        reference_data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440001",
            "label": "Test Reference",
            "image_data": "base64_encoded_image",
            "metadata": {"source": "upload"}
        }
        
        response = self.client.post(
            "/api/v1/reference",
            headers=headers,
            json=reference_data
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["label"] == "Test Reference"
    
    def test_get_reference_unauthorized(self):
        """Тест получения эталонного изображения без авторизации"""
        response = self.client.get("/api/v1/reference/550e8400-e29b-41d4-a716-446655440000")
        
        assert response.status_code == 401
    
    @patch('app.routes.reference.ReferenceCRUD')
    def test_get_reference_success(self, mock_crud):
        """Тест успешного получения эталонного изображения"""
        # Мок успешного получения
        mock_crud.return_value.get.return_value = Mock(
            id="550e8400-e29b-41d4-a716-446655440000",
            user_id="550e8400-e29b-41d4-a716-446655440001",
            label="Test Reference",
            is_active=True
        )
        
        headers = {"Authorization": "Bearer sk_test_1234567890abcdef"}
        response = self.client.get(
            "/api/v1/reference/550e8400-e29b-41d4-a716-446655440000",
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "550e8400-e29b-41d4-a716-446655440000"
    
    def test_get_reference_not_found(self):
        """Тест получения несуществующего эталонного изображения"""
        headers = {"Authorization": "Bearer sk_test_1234567890abcdef"}
        response = self.client.get(
            "/api/v1/reference/nonexistent_id",
            headers=headers
        )
        
        assert response.status_code == 404


class TestVerificationEndpoints:
    """Интеграционные тесты для верификации"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_verify_unauthorized(self):
        """Тест верификации без авторизации"""
        response = self.client.post("/api/v1/verify")
        
        assert response.status_code == 401
    
    def test_verify_invalid_data(self):
        """Тест верификации с невалидными данными"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = self.client.post(
            "/api/v1/verify",
            headers=headers,
            json={"invalid": "data"}
        )
        
        assert response.status_code in [401, 422]
    
    @patch('app.routes.verify.VerificationService')
    def test_verify_success(self, mock_service):
        """Тест успешной верификации"""
        # Мок успешной верификации
        mock_service.return_value.verify.return_value = Mock(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            is_match=True,
            confidence=0.95,
            threshold_used=0.8,
            processing_time=0.123
        )
        
        headers = {"Authorization": "Bearer sk_test_1234567890abcdef"}
        verify_data = {
            "image_data": "base64_encoded_test_image",
            "reference_id": "550e8400-e29b-41d4-a716-446655440000",
            "threshold": 0.8
        }
        
        response = self.client.post(
            "/api/v1/verify",
            headers=headers,
            json=verify_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["is_match"] is True
        assert data["confidence"] == 0.95
    
    @patch('app.routes.verify.VerificationService')
    def test_verify_no_match(self, mock_service):
        """Тест верификации без совпадения"""
        # Мок верификации без совпадения
        mock_service.return_value.verify.return_value = Mock(
            session_id="550e8400-e29b-41d4-a716-446655440001",
            is_match=False,
            confidence=0.45,
            threshold_used=0.8,
            processing_time=0.098
        )
        
        headers = {"Authorization": "Bearer sk_test_1234567890abcdef"}
        verify_data = {
            "image_data": "base64_encoded_test_image",
            "reference_id": "550e8400-e29b-41d4-a716-446655440000"
        }
        
        response = self.client.post(
            "/api/v1/verify",
            headers=headers,
            json=verify_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is False
        assert data["confidence"] == 0.45


class TestLivenessEndpoints:
    """Интеграционные тесты для liveness detection"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_liveness_unauthorized(self):
        """Тест liveness detection без авторизации"""
        response = self.client.post("/api/v1/liveness")
        
        assert response.status_code == 401
    
    def test_liveness_invalid_data(self):
        """Тест liveness detection с невалидными данными"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = self.client.post(
            "/api/v1/liveness",
            headers=headers,
            json={"invalid": "data"}
        )
        
        assert response.status_code in [401, 422]
    
    @patch('app.routes.liveness.LivenessService')
    def test_liveness_live(self, mock_service):
        """Тест liveness detection для живого лица"""
        # Мок для живого лица
        mock_service.return_value.detect_liveness.return_value = Mock(
            is_live=True,
            confidence=0.92,
            processing_time=0.087
        )
        
        headers = {"Authorization": "Bearer sk_test_1234567890abcdef"}
        liveness_data = {
            "image_data": "base64_encoded_live_face"
        }
        
        response = self.client.post(
            "/api/v1/liveness",
            headers=headers,
            json=liveness_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_live"] is True
        assert data["confidence"] == 0.92
    
    @patch('app.routes.liveness.LivenessService')
    def test_liveness_spoof(self, mock_service):
        """Тест liveness detection для подделки (фото)"""
        # Мок для подделки
        mock_service.return_value.detect_liveness.return_value = Mock(
            is_live=False,
            confidence=0.88,
            processing_time=0.065
        )
        
        headers = {"Authorization": "Bearer sk_test_1234567890abcdef"}
        liveness_data = {
            "image_data": "base64_encoded_photo"
        }
        
        response = self.client.post(
            "/api/v1/liveness",
            headers=headers,
            json=liveness_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_live"] is False
        assert data["confidence"] == 0.88


class TestAdminEndpoints:
    """Интеграционные тесты для административных функций"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_admin_unauthorized(self):
        """Тест административных функций без авторизации"""
        response = self.client.get("/api/v1/admin/users")
        
        assert response.status_code == 401
    
    def test_admin_invalid_token(self):
        """Тест административных функций с невалидным токеном"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = self.client.get("/api/v1/admin/users", headers=headers)
        
        assert response.status_code == 403
    
    @patch('app.routes.admin.AdminService')
    def test_admin_get_users(self, mock_service):
        """Тест получения списка пользователей"""
        # Мок списка пользователей
        mock_service.return_value.get_users.return_value = [
            Mock(
                id="550e8400-e29b-41d4-a716-446655440000",
                username="user1",
                email="user1@example.com",
                is_active=True
            ),
            Mock(
                id="550e8400-e29b-41d4-a716-446655440001",
                username="user2",
                email="user2@example.com",
                is_active=True
            )
        ]
        
        headers = {"Authorization": "Bearer sk_admin_1234567890abcdef"}
        response = self.client.get("/api/v1/admin/users", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["username"] == "user1"
    
    @patch('app.routes.admin.AdminService')
    def test_admin_get_stats(self, mock_service):
        """Тест получения статистики"""
        # Мок статистики
        mock_service.return_value.get_stats.return_value = Mock(
            total_users=150,
            active_users=120,
            total_references=300,
            total_verifications=5000,
            successful_verifications=4750,
            failed_verifications=250
        )
        
        headers = {"Authorization": "Bearer sk_admin_1234567890abcdef"}
        response = self.client.get("/api/v1/admin/stats", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_users" in data
        assert "total_verifications" in data
        assert data["total_users"] == 150
        assert data["total_verifications"] == 5000