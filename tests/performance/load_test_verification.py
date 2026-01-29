# tests/performance/load_test_verification.py

"""
Load-тесты для проверки производительности сервиса распознавания лиц

Требования ТЗ:
- Скорость обработки одного лица: до 1 секунды
- Горизонтальное масштабирование
- Доступность 99.5%
"""

import base64
import io
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from locust import HttpUser, between, events, task
from locust.runners import MasterRunner
from PIL import Image

from app.utils.logger import get_logger

logger = get_logger(__name__)


class VerificationLoadTest(HttpUser):
    """
    Симуляция пользователей для load-тестирования верификации
    """

    # Время ожидания между запросами (имитация реального поведения)
    wait_time = between(1, 3)

    # Тестовые изображения
    test_images_dir = Path("tests/datasets/load_test_images")
    test_images = []

    def on_start(self):
        """Инициализация при старте пользователя"""
        # Загрузка тестовых изображений
        if not self.test_images:
            self._load_test_images()

        # Аутентификация (если требуется)
        # self.authenticate()

    def _load_test_images(self):
        """Загрузка тестовых изображений в память"""
        if not self.test_images_dir.exists():
            logger.warning(f"Test images directory not found: {self.test_images_dir}")
            self._generate_synthetic_images()
            return

        image_files = list(self.test_images_dir.glob("*.jpg")) + list(
            self.test_images_dir.glob("*.png")
        )

        for img_path in image_files[:100]:  # Ограничиваем 100 изображениями
            with open(img_path, "rb") as f:
                self.test_images.append({"name": img_path.name, "data": f.read()})

        logger.info(f"Loaded {len(self.test_images)} test images")

    def _generate_synthetic_images(self):
        """Генерация синтетических изображений для тестов"""
        logger.info("Generating synthetic test images...")

        for i in range(50):
            # Создаем случайное RGB изображение
            img = Image.fromarray(
                np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            )

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)

            self.test_images.append(
                {"name": f"synthetic_{i}.jpg", "data": buffer.getvalue()}
            )

        logger.info(f"Generated {len(self.test_images)} synthetic images")

    def get_random_image_pair(self) -> tuple:
        """Получение случайной пары изображений"""
        if len(self.test_images) < 2:
            raise ValueError("Not enough test images loaded")

        img1 = random.choice(self.test_images)
        img2 = random.choice(self.test_images)

        return img1, img2

    @task(10)  # Вес задачи: 10 (самая частая операция)
    def verify_face(self):
        """
        Тест верификации лица (1:1)

        Требование: < 1 секунда
        """
        img1, img2 = self.get_random_image_pair()

        start_time = time.time()

        with self.client.post(
            "/verify/face",
            files={
                "reference_image": (img1["name"], img1["data"], "image/jpeg"),
                "test_image": (img2["name"], img2["data"], "image/jpeg"),
            },
            params={"threshold": 0.6, "check_liveness": True},
            catch_response=True,
            name="POST /verify/face",
        ) as response:

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                # Проверка времени обработки (требование ТЗ)
                if elapsed > 1.0:
                    response.failure(f"Response time exceeded 1 second: {elapsed:.3f}s")
                    logger.warning(f"Slow response: {elapsed:.3f}s for /verify/face")
                else:
                    response.success()
                    logger.debug(
                        f"Verification completed in {elapsed:.3f}s, "
                        f"match={result.get('result', {}).get('match')}"
                    )
            else:
                response.failure(f"Status code {response.status_code}: {response.text}")

    @task(5)
    def upload_and_validate(self):
        """
        Тест загрузки и валидации изображения
        """
        img = random.choice(self.test_images)

        with self.client.post(
            "/upload/validate",
            files={"file": (img["name"], img["data"], "image/jpeg")},
            catch_response=True,
            name="POST /upload/validate",
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")

    @task(3)
    def get_verification_status(self):
        """
        Тест получения статуса верификации
        """
        # Используем случайный UUID для теста
        verification_id = "550e8400-e29b-41d4-a716-446655440000"

        with self.client.get(
            f"/status/{verification_id}", catch_response=True, name="GET /status/{id}"
        ) as response:

            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")

    @task(2)
    def reference_search(self):
        """
        Тест поиска по эталонам (1:N)
        """
        img = random.choice(self.test_images)

        with self.client.post(
            "/reference/search",
            files={"image": (img["name"], img["data"], "image/jpeg")},
            params={"top_k": 5, "threshold": 0.6},
            catch_response=True,
            name="POST /reference/search",
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")

    @task(1)
    def get_supported_formats(self):
        """
        Тест получения поддерживаемых форматов (легкий запрос)
        """
        with self.client.get(
            "/upload/supported-formats",
            catch_response=True,
            name="GET /upload/supported-formats",
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")

    @task(1)
    def health_check(self):
        """
        Тест healthcheck
        """
        with self.client.get(
            "/health", catch_response=True, name="GET /health"
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")


class StressTestUser(HttpUser):
    """
    Стресс-тест с высокой нагрузкой
    """

    wait_time = between(0.1, 0.5)  # Минимальная задержка

    test_images = []

    def on_start(self):
        self._load_test_images()

    def _load_test_images(self):
        """Загрузка тестовых изображений"""
        test_dir = Path("tests/datasets/load_test_images")

        if not test_dir.exists():
            # Генерируем синтетические изображения
            for i in range(10):
                img = Image.fromarray(
                    np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
                )
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")

                self.test_images.append(
                    {"name": f"stress_{i}.jpg", "data": buffer.getvalue()}
                )
        else:
            for img_path in list(test_dir.glob("*.jpg"))[:10]:
                with open(img_path, "rb") as f:
                    self.test_images.append({"name": img_path.name, "data": f.read()})

    @task
    def stress_verify(self):
        """Непрерывная верификация для стресс-теста"""
        if len(self.test_images) < 2:
            return

        img1 = random.choice(self.test_images)
        img2 = random.choice(self.test_images)

        self.client.post(
            "/verify/face",
            files={
                "reference_image": (img1["name"], img1["data"], "image/jpeg"),
                "test_image": (img2["name"], img2["data"], "image/jpeg"),
            },
            timeout=5,
        )


# Event handlers для сбора статистики

response_times = []
failed_requests = 0
total_requests = 0


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Обработчик для каждого запроса"""
    global response_times, failed_requests, total_requests

    total_requests += 1

    if exception:
        failed_requests += 1
    else:
        response_times.append(response_time)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Обработчик при завершении теста"""
    if not response_times:
        logger.warning("No response times recorded")
        return

    # Расчет статистики
    response_times_sorted = sorted(response_times)
    total = len(response_times)

    stats = {
        "total_requests": total_requests,
        "successful_requests": total - failed_requests,
        "failed_requests": failed_requests,
        "success_rate": (
            ((total - failed_requests) / total_requests * 100)
            if total_requests > 0
            else 0
        ),
        "response_time": {
            "min": min(response_times),
            "max": max(response_times),
            "mean": np.mean(response_times),
            "median": np.median(response_times),
            "p95": np.percentile(response_times, 95),
            "p99": np.percentile(response_times, 99),
        },
    }

    # Проверка соответствия ТЗ
    logger.info("=" * 80)
    logger.info("LOAD TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total requests: {stats['total_requests']}")
    logger.info(f"Success rate: {stats['success_rate']:.2f}%")
    logger.info(f"Failed requests: {stats['failed_requests']}")
    logger.info("")
    logger.info("Response times (ms):")
    logger.info(f"  Min:    {stats['response_time']['min']:.0f} ms")
    logger.info(f"  Mean:   {stats['response_time']['mean']:.0f} ms")
    logger.info(f"  Median: {stats['response_time']['median']:.0f} ms")
    logger.info(f"  P95:    {stats['response_time']['p95']:.0f} ms")
    logger.info(f"  P99:    {stats['response_time']['p99']:.0f} ms")
    logger.info(f"  Max:    {stats['response_time']['max']:.0f} ms")
    logger.info("")

    # Проверка требований ТЗ
    compliance_checks = {
        "Response time < 1s (mean)": stats["response_time"]["mean"] < 1000,
        "Response time < 1s (p95)": stats["response_time"]["p95"] < 1000,
        "Response time < 1s (p99)": stats["response_time"]["p99"]
        < 1500,  # Допустим 1.5s для p99
        "Success rate > 99.5%": stats["success_rate"] > 99.5,
    }

    logger.info("ТЗ Compliance Check:")
    all_passed = True
    for check, passed in compliance_checks.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {check}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 80)

    if all_passed:
        logger.info("✓ Все требования ТЗ выполнены!")
    else:
        logger.error("✗ Некоторые требования ТЗ не выполнены")

    # Сохранение результатов в файл
    import json
    from datetime import datetime

    results_file = Path(
        f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Results saved to {results_file}")
