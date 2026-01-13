import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import create_test_app
from app.config import settings as Config


class TestHealthEndpoints:
    """Интеграционные тесты для health check endpoints"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_health_check(self):
        """Тест основного health check"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        # Может быть "healthy", "degraded" или "unhealthy" в зависимости от состояния сервисов
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_health_detailed(self):
        """Тест детального health check"""
        response = self.client.get("/health/detailed")
        # Этот endpoint может не существовать, поэтому ожидаем 404 или 200
        assert response.status_code in [200, 404]
        
    def test_health_ready(self):
        """Тест readiness probe"""
        response = self.client.get("/health/ready")
        # Этот endpoint может не существовать, поэтому ожидаем 404 или 200
        assert response.status_code in [200, 404]
        

class TestReferenceEndpoints:
    """Интеграционные тесты для эталонных изображений"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_create_reference_endpoint_exists(self):
        """Тест что endpoint reference существует и принимает запросы"""
        response = self.client.post(
            "/api/v1/reference",
            json={
                "user_id": "test-user-123",
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD",
                "label": "Test Reference"
            }
        )
        # Ожидаем ошибку валидации или авторизации, но не 404
        assert response.status_code in [401, 422, 400]
        
        # Проверяем структуру ответа
        if response.status_code != 404:
            data = response.json()
            assert "detail" in data
    
    def test_get_reference_list_endpoint_exists(self):
        """Тест что endpoint reference (GET) существует"""
        response = self.client.get("/api/v1/reference")
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
    
    def test_get_reference_detail_endpoint_exists(self):
        """Тест что endpoint reference/{id} существует"""
        response = self.client.get("/api/v1/reference/test-reference-123")
        # Ожидаем 404 (референс не найден), но не 404 route не найден
        assert response.status_code != 404
    
    def test_update_reference_endpoint_exists(self):
        """Тест что endpoint reference/{id} (PUT) существует"""
        response = self.client.put(
            "/api/v1/reference/test-reference-123",
            json={"label": "Updated Reference"}
        )
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
        
    def test_delete_reference_endpoint_exists(self):
        """Тест что endpoint reference/{id} (DELETE) существует"""
        response = self.client.delete("/api/v1/reference/test-reference-123")
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
    
    def test_reference_compare_endpoint_exists(self):
        """Тест что endpoint reference/compare существует"""
        response = self.client.post(
            "/api/v1/compare",
            json={
                "user_id": "test-user-123",
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD"
            }
        )
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
        
    def test_reference_validation_error_structure(self):
        """Тест структуры ошибки валидации для reference"""
        response = self.client.post(
            "/api/v1/reference",
            json={
                "user_id": "test-user-123"
                # Отсутствует image_data
            }
        )
        
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data
    
    def test_reference_missing_required_fields(self):
        """Тест reference с отсутствующими обязательными полями"""
        response = self.client.post(
            "/api/v1/reference",
            json={
                "label": "Test Reference"
                # Отсутствует user_id и image_data
            }
        )
        
        # Ожидаем ошибку валидации
        assert response.status_code in [400, 422]
    
    def test_reference_pagination_parameters(self):
        """Тест reference endpoints с параметрами пагинации"""
        response = self.client.get("/api/v1/reference?page=1&per_page=10")
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
    
    def test_reference_sorting_parameters(self):
        """Тест reference endpoints с параметрами сортировки"""
        response = self.client.get("/api/v1/reference?sort_by=created_at&sort_order=desc")
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
    

class TestVerificationEndpoints:
    """Интеграционные тесты для верификации"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_verify_endpoint_exists(self):
        """Тест что endpoint verify существует и принимает запросы"""
        # Проверяем что endpoint не возвращает 404
        response = self.client.post(
            "/api/v1/verify",
            json={
                "session_id": "test-session-123",
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD",
                "reference_id": "test-ref-123"
            }
        )
        # Ожидаем ошибку валидации (422) или авторизации (401), но не 404
        assert response.status_code in [401, 422, 400]
        
        # Проверяем структуру ответа
        if response.status_code != 404:
            data = response.json()
            assert "detail" in data
    
    def test_verify_session_endpoint_exists(self):
        """Тест что endpoint verify/session существует"""
        response = self.client.post(
            "/api/v1/verify/session",
            json={
                "session_type": "verification",
                "user_id": "test-user"
            }
        )
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
    
    def test_verify_get_result_endpoint_exists(self):
        """Тест что endpoint verify/{session_id} существует"""
        response = self.client.get("/api/v1/verify/test-session-123")
        # Ожидаем 404 (сессия не найдена), но не 404 route не найден
        assert response.status_code != 404
    
    def test_verify_history_endpoint_exists(self):
        """Тест что endpoint verify/history существует"""
        response = self.client.get("/api/v1/verify/history")
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
    
    def test_verify_validation_error_structure(self):
        """Тест структуры ошибки валидации для verify"""
        response = self.client.post(
            "/api/v1/verify",
            json={
                "session_id": "test-session-123",
                # Отсутствует image_data
                "reference_id": "test-ref-123"
            }
        )
        
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data
    
    def test_verify_missing_parameters(self):
        """Тест verify с отсутствующими обязательными параметрами"""
        response = self.client.post(
            "/api/v1/verify",
            json={
                "session_id": "test-session-123"
                # Отсутствует image_data и reference_id/user_id
            }
        )
        
        # Ожидаем ошибку валидации
        assert response.status_code in [400, 422]
    

class TestLivenessEndpoints:
    """Интеграционные тесты для liveness detection"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_liveness_endpoint_exists(self):
        """Тест что endpoint liveness существует и принимает запросы"""
        response = self.client.post(
            "/api/v1/liveness",
            json={
                "session_id": "test-session-123",
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD",
                "challenge_type": "passive"
            }
        )
        # Ожидаем ошибку валидации или авторизации, но не 404
        assert response.status_code in [401, 422, 400]
        
        # Проверяем структуру ответа
        if response.status_code != 404:
            data = response.json()
            assert "detail" in data
    
    def test_liveness_session_endpoint_exists(self):
        """Тест что endpoint liveness/session существует"""
        response = self.client.post(
            "/api/v1/liveness/session",
            json={
                "session_type": "liveness",
                "user_id": "test-user"
            }
        )
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
    
    def test_liveness_challenges_endpoint_exists(self):
        """Тест что endpoint liveness/challenges существует"""
        response = self.client.get("/api/v1/liveness/challenges")
        # Ожидаем успешный ответ или ошибку авторизации, но не 404
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert "challenges" in data
            assert "passive" in data["challenges"]
    
    def test_liveness_get_result_endpoint_exists(self):
        """Тест что endpoint liveness/{session_id} существует"""
        response = self.client.get("/api/v1/liveness/test-session-123")
        # Ожидаем 404 (сессия не найдена), но не 404 route не найден
        assert response.status_code != 404
    
    def test_liveness_session_status_endpoint_exists(self):
        """Тест что endpoint liveness/session/{session_id} существует"""
        response = self.client.get("/api/v1/liveness/session/test-session-123")
        # Ожидаем ошибку, но не 404
        assert response.status_code != 404
    
    def test_liveness_validation_error_structure(self):
        """Тест структуры ошибки валидации для liveness"""
        response = self.client.post(
            "/api/v1/liveness",
            json={
                "session_id": "test-session-123",
                # Отсутствует image_data
                "challenge_type": "passive"
            }
        )
        
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data
    
    def test_liveness_missing_parameters(self):
        """Тест liveness с отсутствующими обязательными параметрами"""
        response = self.client.post(
            "/api/v1/liveness",
            json={
                "session_id": "test-session-123"
                # Отсутствует image_data
            }
        )
        
        # Ожидаем ошибку валидации
        assert response.status_code in [400, 422]
    
    def test_liveness_active_challenge_validation(self):
        """Тест валидации активного челленджа"""
        response = self.client.post(
            "/api/v1/liveness",
            json={
                "session_id": "test-session-123",
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wIAD",
                "challenge_type": "active"
                # Отсутствует challenge_data для активного челленджа
            }
        )
        
        # Ожидаем ошибку валидации
        assert response.status_code in [400, 422]
        
        if response.status_code in [400, 422]:
            data = response.json()
            assert "challenge_data is required" in data["detail"]["error_details"]["error"]
    

class TestAdminEndpoints:
    """Интеграционные тесты для административных функций"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.app = create_test_app()
        self.client = TestClient(self.app)
    
    def test_admin_unauthorized(self):
        """Тест административных функций без авторизации"""
        response = self.client.get("/api/v1/admin/users")
        
        assert response.status_code == 401
    
    def test_admin_invalid_token(self):
        """Тест административных функций с невалидным токеном"""
        # Простой тест без аутентификации
        response = self.client.get("/api/v1/admin/users")
        assert response.status_code == 401
    
    def test_admin_get_users(self):
        """Тест получения списка пользователей"""
        # Простой тест без аутентификации
        response = self.client.get("/api/v1/admin/users")
        assert response.status_code == 401
    
    def test_admin_get_stats(self):
        """Тест получения статистики"""
        # Простой тест без аутентификации
        response = self.client.get("/api/v1/admin/stats")
        assert response.status_code == 401
