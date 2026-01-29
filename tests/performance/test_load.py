"""
Performance тесты для проверки производительности под нагрузкой.
"""

import base64
import statistics
import time

import pytest
from locust import HttpUser, between, task


class FaceVerificationUser(HttpUser):
    """
    Locust user для load testing.

    Запуск:
    locust -f tests/performance/test_load.py --host=http://localhost:8000
    """

    wait_time = between(1, 3)

    def on_start(self):
        """Инициализация: регистрация и вход."""
        # Регистрация
        self.client.post(
            "/api/v1/auth/register",
            json={
                "email": f"loadtest_{time.time()}@example.com",
                "password": "password123",
            },
        )

        # Вход
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "email": f"loadtest_{time.time()}@example.com",
                "password": "password123",
            },
        )

        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

        # Загружаем тестовое изображение
        with open("tests/fixtures/face_sample.jpg", "rb") as f:
            self.image_data = base64.b64encode(f.read()).decode("utf-8")

    @task(3)
    def verify_face(self):
        """Верификация лица (вес 3 - самая частая операция)."""
        self.client.post(
            "/api/v1/verify",
            json={"user_id": "loadtest-user", "image_data": self.image_data},
            headers=self.headers,
        )

    @task(2)
    def check_liveness(self):
        """Проверка живости (вес 2)."""
        self.client.post(
            "/api/v1/liveness",
            json={
                "user_id": "loadtest-user",
                "image_data": self.image_data,
                "challenge_type": "passive",
            },
            headers=self.headers,
        )

    @task(1)
    def create_reference(self):
        """Создание reference (вес 1 - редкая операция)."""
        self.client.post(
            "/api/v1/reference",
            json={
                "user_id": "loadtest-user",
                "image_data": self.image_data,
                "label": f"Load Test {time.time()}",
            },
            headers=self.headers,
        )


class TestPerformanceMetrics:
    """Pytest тесты для измерения производительности."""

    @pytest.mark.performance
    def test_embedding_generation_speed(self):
        """Тест скорости генерации эмбеддингов."""
        import asyncio

        from app.services.ml_service import OptimizedMLService

        async def run_test():
            ml_service = OptimizedMLService()
            await ml_service.initialize()

            with open("tests/fixtures/face_sample.jpg", "rb") as f:
                image_data = f.read()

            # Прогрев (warm-up)
            await ml_service.generate_embedding(image_data)

            # Измеряем 10 итераций
            times = []
            for _ in range(10):
                start = time.time()
                await ml_service.generate_embedding(image_data)
                elapsed = time.time() - start
                times.append(elapsed)

            return times

        times = asyncio.run(run_test())

        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]

        print(f"\n🔹 Embedding Generation Performance:")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   Median: {median_time:.3f}s")
        print(f"   P95: {p95_time:.3f}s")

        # Assertions
        assert avg_time < 0.5  # < 500ms в среднем
        assert p95_time < 1.0  # < 1s для 95% запросов

    @pytest.mark.performance
    def test_verification_speed(self):
        """Тест скорости верификации."""
        import asyncio

        import numpy as np

        from app.services.ml_service import OptimizedMLService

        async def run_test():
            ml_service = OptimizedMLService()
            await ml_service.initialize()

            with open("tests/fixtures/face_sample.jpg", "rb") as f:
                image_data = f.read()

            # Генерируем reference embedding
            ref_result = await ml_service.generate_embedding(image_data)
            reference_embedding = np.array(ref_result["embedding"])

            # Прогрев
            await ml_service.verify_face(image_data, reference_embedding)

            # Измеряем 10 итераций
            times = []
            for _ in range(10):
                start = time.time()
                await ml_service.verify_face(image_data, reference_embedding)
                elapsed = time.time() - start
                times.append(elapsed)

            return times

        times = asyncio.run(run_test())

        avg_time = statistics.mean(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]

        print(f"\n🔹 Verification Performance:")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   P95: {p95_time:.3f}s")

        assert avg_time < 0.6  # < 600ms
        assert p95_time < 1.2  # < 1.2s
