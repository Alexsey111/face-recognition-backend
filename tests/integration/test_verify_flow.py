"""
Integration тесты для полного flow верификации.
Тестируют взаимодействие ML Service, DB, Cache, Storage.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import create_test_app
import base64


@pytest.fixture
def client():
    """Фикстура для тестового клиента."""
    app = create_test_app()
    return TestClient(app)


@pytest.fixture
def test_user_token(client):
    """Создание тестового пользователя и получение токена."""
    # Регистрация пользователя
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User",
        },
    )

    # Вход
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.com", "password": "testpassword123"},
    )

    return response.json()["access_token"]


@pytest.fixture
def face_image_base64():
    """Загрузка тестового изображения лица в base64."""
    with open("tests/fixtures/face_sample.jpg", "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


class TestReferenceCreation:
    """Тесты создания эталонного изображения."""

    def test_create_reference(self, client, test_user_token, face_image_base64):
        """Тест создания reference."""
        headers = {"Authorization": f"Bearer {test_user_token}"}

        response = client.post(
            "/api/v1/reference",
            json={
                "user_id": "test-user-123",
                "image_data": face_image_base64,
                "label": "Test Reference",
                "quality_threshold": 0.5,
            },
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "reference_id" in data
        assert data["quality_score"] > 0

    def test_create_reference_low_quality(self, client, test_user_token):
        """Тест создания reference с низким качеством."""
        # Создаем низкокачественное изображение
        from PIL import Image
        import io

        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=10)
        low_quality_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        headers = {"Authorization": f"Bearer {test_user_token}"}

        response = client.post(
            "/api/v1/reference",
            json={
                "user_id": "test-user-123",
                "image_data": low_quality_data,
                "label": "Low Quality Test",
                "quality_threshold": 0.7,  # Высокий порог
            },
            headers=headers,
        )

        # Должно быть отклонено из-за низкого качества
        assert response.status_code == 400


class TestVerificationFlow:
    """Тесты полного flow верификации."""

    def test_verify_with_reference(self, client, test_user_token, face_image_base64):
        """Тест верификации с существующим reference."""
        headers = {"Authorization": f"Bearer {test_user_token}"}

        # 1. Создаем reference
        ref_response = client.post(
            "/api/v1/reference",
            json={
                "user_id": "test-user-123",
                "image_data": face_image_base64,
                "label": "Test Reference",
            },
            headers=headers,
        )

        assert ref_response.status_code == 200
        reference_id = ref_response.json()["reference_id"]

        # 2. Верифицируем то же изображение
        verify_response = client.post(
            "/api/v1/verify",
            json={
                "user_id": "test-user-123",
                "image_data": face_image_base64,
                "reference_id": reference_id,
                "threshold": 0.6,
            },
            headers=headers,
        )

        assert verify_response.status_code == 200
        verify_data = verify_response.json()

        assert verify_data["verified"] is True
        assert verify_data["similarity_score"] > 0.9
        assert verify_data["confidence"] > 0.8

    def test_verify_without_reference(self, client, test_user_token, face_image_base64):
        """Тест верификации без reference."""
        headers = {"Authorization": f"Bearer {test_user_token}"}

        response = client.post(
            "/api/v1/verify",
            json={"user_id": "nonexistent-user", "image_data": face_image_base64},
            headers=headers,
        )

        # Должна быть ошибка 404
        assert response.status_code == 404

    def test_verify_session_flow(self, client, test_user_token, face_image_base64):
        """Тест верификации через сессию."""
        headers = {"Authorization": f"Bearer {test_user_token}"}

        # 1. Создаем сессию верификации
        session_response = client.post(
            "/api/v1/verify/session",
            json={"user_id": "test-user-123", "expires_in_minutes": 15},
            headers=headers,
        )

        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        # 2. Верифицируем в рамках сессии
        verify_response = client.post(
            f"/api/v1/verify/session/{session_id}",
            json={"image_data": face_image_base64},
            headers=headers,
        )

        assert verify_response.status_code == 200
        verify_data = verify_response.json()

        assert "verified" in verify_data
        assert verify_data["session_id"] == session_id


class TestLivenessIntegration:
    """Integration тесты для liveness detection."""

    def test_liveness_check(self, client, test_user_token, face_image_base64):
        """Тест проверки живости."""
        headers = {"Authorization": f"Bearer {test_user_token}"}

        response = client.post(
            "/api/v1/liveness",
            json={
                "user_id": "test-user-123",
                "image_data": face_image_base64,
                "challenge_type": "passive",
            },
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "liveness_detected" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_liveness_with_depth_analysis(
        self, client, test_user_token, face_image_base64
    ):
        """Тест liveness с depth анализом."""
        headers = {"Authorization": f"Bearer {test_user_token}"}

        response = client.post(
            "/api/v1/liveness/anti-spoofing/advanced",
            json={
                "user_id": "test-user-123",
                "image_data": face_image_base64,
                "analysis_type": "depth_certified",
            },
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "depth_analysis" in data
        assert "anti_spoofing_score" in data


class TestCacheIntegration:
    """Тесты интеграции с кэшем."""

    def test_verification_result_cached(
        self, client, test_user_token, face_image_base64
    ):
        """Тест кэширования результатов верификации."""
        headers = {"Authorization": f"Bearer {test_user_token}"}

        # Первый запрос
        response1 = client.post(
            "/api/v1/verify",
            json={"user_id": "test-user-123", "image_data": face_image_base64},
            headers=headers,
        )

        # Второй запрос (должен быть быстрее из-за кэша)
        import time

        start = time.time()

        response2 = client.post(
            "/api/v1/verify",
            json={"user_id": "test-user-123", "image_data": face_image_base64},
            headers=headers,
        )

        elapsed = time.time() - start

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Второй запрос должен быть значительно быстрее
        # (если есть кэш эмбеддингов)
