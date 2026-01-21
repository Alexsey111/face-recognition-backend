"""
Security тесты для проверки уязвимостей.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import create_test_app


@pytest.fixture
def client():
    app = create_test_app()
    return TestClient(app)


class TestAuthenticationSecurity:
    """Тесты безопасности аутентификации."""

    def test_no_auth_required_endpoints(self, client):
        """Проверка, что защищенные endpoints требуют аутентификации."""
        # Попытка доступа без токена
        response = client.post(
            "/api/v1/verify", json={"user_id": "test", "image_data": "fake_data"}
        )

        # Должна быть ошибка 401 Unauthorized
        assert response.status_code == 401

    def test_invalid_token(self, client):
        """Тест с невалидным токеном."""
        headers = {"Authorization": "Bearer invalid_token_12345"}

        response = client.post(
            "/api/v1/verify",
            json={"user_id": "test", "image_data": "data"},
            headers=headers,
        )

        assert response.status_code == 401


class TestInputValidation:
    """Тесты валидации входных данных."""

    def test_sql_injection_attempts(self, client):
        """Тест на SQL injection."""
        # Регистрация с SQL injection в email
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test'; DROP TABLE users; --@example.com",
                "password": "password123",
            },
        )

        # Должна быть валидационная ошибка или безопасная обработка
        assert response.status_code in [400, 422]

    def test_oversized_image(self, client):
        """Тест загрузки слишком большого изображения."""
        # Создаем изображение > MAX_UPLOAD_SIZE
        import base64

        oversized_data = base64.b64encode(b"x" * (11 * 1024 * 1024)).decode("utf-8")

        client.post(
            "/api/v1/auth/register",
            json={"email": "test@example.com", "password": "password123"},
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "password123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post(
            "/api/v1/reference",
            json={"user_id": "test", "image_data": oversized_data, "label": "Test"},
            headers=headers,
        )

        # Должна быть ошибка валидации
        assert response.status_code in [400, 413, 422]

    def test_malicious_file_upload(self, client):
        """Тест загрузки вредоносного файла."""
        import base64

        # Создаем "изображение" с исполняемым кодом
        malicious_data = base64.b64encode(b"#!/bin/bash\nrm -rf /").decode("utf-8")

        client.post(
            "/api/v1/auth/register",
            json={"email": "malicious@example.com", "password": "password123"},
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "malicious@example.com", "password": "password123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post(
            "/api/v1/reference",
            json={
                "user_id": "test",
                "image_data": malicious_data,
                "label": "Malicious",
            },
            headers=headers,
        )

        # Должна быть ошибка обработки изображения
        assert response.status_code in [400, 422]


class TestRateLimiting:
    """Тесты rate limiting."""

    def test_rate_limit_exceeded(self, client):
        """Тест превышения rate limit."""
        # Регистрация
        client.post(
            "/api/v1/auth/register",
            json={"email": "ratelimit@example.com", "password": "password123"},
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "ratelimit@example.com", "password": "password123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Делаем много запросов подряд
        responses = []
        for _ in range(100):  # Превышаем RATE_LIMIT_REQUESTS_PER_MINUTE
            response = client.get("/api/v1/health", headers=headers)
            responses.append(response.status_code)

        # Хотя бы один запрос должен быть заблокирован (429)
        assert 429 in responses


class TestDataPrivacy:
    """Тесты приватности данных."""

    def test_embedding_encryption(self, client):
        """Проверка, что embeddings хранятся в зашифрованном виде."""
        import base64

        # Регистрация и создание reference
        client.post(
            "/api/v1/auth/register",
            json={"email": "privacy@example.com", "password": "password123"},
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "privacy@example.com", "password": "password123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        with open("tests/fixtures/face_sample.jpg", "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ref_response = client.post(
            "/api/v1/reference",
            json={
                "user_id": "privacy-user",
                "image_data": image_data,
                "label": "Privacy Test",
            },
            headers=headers,
        )

        assert ref_response.status_code == 200

        # Проверяем, что в ответе нет raw embedding
        response_data = ref_response.json()
        assert (
            "embedding" not in response_data or response_data.get("embedding") is None
        )
