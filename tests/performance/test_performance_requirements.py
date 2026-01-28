# tests/performance/test_performance_requirements.py

"""
Pytest тесты для проверки требований производительности из ТЗ
"""

import pytest
import asyncio
import time
import statistics
from pathlib import Path
import numpy as np

from app.services.face_verification_service import FaceVerificationService
from app.services.anti_spoofing_service import AntiSpoofingService
from app.utils.file_utils import ImageFileHandler


@pytest.mark.performance
class TestPerformanceRequirements:
    """Тесты требований производительности из ТЗ"""
    
    @pytest.fixture(scope="class")
    def verification_service(self):
        """Fixture для сервиса верификации"""
        return FaceVerificationService()
    
    @pytest.fixture(scope="class")
    def anti_spoofing_service(self):
        """Fixture для anti-spoofing сервиса"""
        return AntiSpoofingService()
    
    @pytest.fixture(scope="class")
    def test_images(self):
        """Загрузка тестовых изображений"""
        test_dir = Path("tests/datasets/performance_test")
        
        if not test_dir.exists():
            pytest.skip("Performance test dataset not found")
        
        images = []
        for img_path in list(test_dir.glob("*.jpg"))[:50]:
            images.append(ImageFileHandler.load_image_from_bytes(
                img_path.read_bytes(),
                img_path.name
            ))
        
        return images
    
    @pytest.mark.asyncio
    async def test_single_verification_time(
        self, 
        verification_service, 
        anti_spoofing_service,
        test_images
    ):
        """
        Тест: Скорость обработки одного лица до 1 секунды (ТЗ п.6)
        """
        if len(test_images) < 2:
            pytest.skip("Not enough test images")
        
        times = []
        
        for i in range(min(20, len(test_images) - 1)):
            img1 = test_images[i]
            img2 = test_images[i + 1]
            
            start = time.time()
            
            # Полный pipeline верификации
            # 1. Face detection (не включаем, т.к. это preprocessing)
            # 2. Liveness check
            liveness1 = await anti_spoofing_service.detect_spoofing(img1)
            liveness2 = await anti_spoofing_service.detect_spoofing(img2)
            
            # 3. Embedding extraction
            emb1 = await verification_service.extract_embedding(img1)
            emb2 = await verification_service.extract_embedding(img2)
            
            # 4. Comparison
            similarity = verification_service.compare_embeddings(emb1, emb2)
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Статистика
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        p95_time = np.percentile(times, 95)
        max_time = max(times)
        
        print(f"\nVerification Performance:")
        print(f"  Mean:   {mean_time*1000:.0f} ms")
        print(f"  Median: {median_time*1000:.0f} ms")
        print(f"  P95:    {p95_time*1000:.0f} ms")
        print(f"  Max:    {max_time*1000:.0f} ms")
        
        # Проверка требования ТЗ: < 1 секунда
        assert mean_time < 1.0, f"Mean time {mean_time:.3f}s exceeds 1 second requirement"
        assert p95_time < 1.0, f"P95 time {p95_time:.3f}s exceeds 1 second requirement"
        
        # Предупреждение если близко к лимиту
        if mean_time > 0.8:
            pytest.warn(
                f"Performance warning: mean time {mean_time:.3f}s is close to 1s limit"
            )
    
    @pytest.mark.asyncio
    async def test_throughput(self, verification_service, test_images):
        """
        Тест пропускной способности
        """
        if len(test_images) < 10:
            pytest.skip("Not enough test images")
        
        batch_size = 10
        images_batch = test_images[:batch_size]
        
        start = time.time()
        
        # Параллельная обработка
        tasks = [
            verification_service.extract_embedding(img)
            for img in images_batch
        ]
        embeddings = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start
        throughput = batch_size / elapsed
        
        print(f"\nThroughput: {throughput:.2f} faces/second")
        print(f"Time per face: {elapsed/batch_size*1000:.0f} ms")
        
        # Минимальная пропускная способность: 5 лиц/сек (CPU)
        # или 20 лиц/сек (GPU)
        min_throughput = 5  # Консервативная оценка для CPU
        
        assert throughput >= min_throughput, \
            f"Throughput {throughput:.2f} faces/s is below minimum {min_throughput}"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, verification_service, test_images):
        """
        Тест обработки конкурентных запросов
        """
        if len(test_images) < 20:
            pytest.skip("Not enough test images")
        
        num_concurrent = 10
        
        async def single_verification(img1, img2):
            start = time.time()
            emb1 = await verification_service.extract_embedding(img1)
            emb2 = await verification_service.extract_embedding(img2)
            similarity = verification_service.compare_embeddings(emb1, emb2)
            return time.time() - start
        
        # Запускаем 10 конкурентных верификаций
        tasks = []
        for i in range(num_concurrent):
            img1 = test_images[i * 2]
            img2 = test_images[i * 2 + 1]
            tasks.append(single_verification(img1, img2))
        
        start = time.time()
        times = await asyncio.gather(*tasks)
        total_time = time.time() - start
        
        mean_time = statistics.mean(times)
        max_time = max(times)
        
        print(f"\nConcurrent Processing ({num_concurrent} requests):")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Mean per request: {mean_time*1000:.0f} ms")
        print(f"  Max per request: {max_time*1000:.0f} ms")
        
        # Среднее время не должно сильно увеличиваться при конкурентной обработке
        # (допускаем увеличение до 2x при CPU-bound операциях)
        assert mean_time < 2.0, \
            f"Mean concurrent time {mean_time:.3f}s exceeds 2 seconds"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, verification_service, test_images):
        """
        Тест утечек памяти при многократной обработке
        """
        if len(test_images) < 1:
            pytest.skip("No test images")
        
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Начальная память
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Обрабатываем 100 изображений
        img = test_images[0]
        for _ in range(100):
            emb = await verification_service.extract_embedding(img)
            del emb
        
        # Финальная память
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Утечка памяти не должна превышать 50 MB на 100 операций
        assert memory_increase < 50, \
            f"Memory leak detected: {memory_increase:.1f} MB increase"
    
    @pytest.mark.parametrize("image_size", [
        (160, 160),   # Минимальный
        (640, 480),   # SD
        (1280, 720),  # HD
        (1920, 1080), # Full HD
        (3024, 4032), # iPhone 12 Pro
    ])
    @pytest.mark.asyncio
    async def test_different_image_sizes(
        self, 
        verification_service, 
        image_size
    ):
        """
        Тест производительности для различных разрешений
        """
        width, height = image_size
        
        # Генерируем тестовое изображение
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        start = time.time()
        emb = await verification_service.extract_embedding(img)
        elapsed = time.time() - start
        
        print(f"\n{width}x{height}: {elapsed*1000:.0f} ms")
        
        # Время должно быть разумным для всех размеров
        # (предобработка включает resize, поэтому время схожее)
        assert elapsed < 1.0, \
            f"Processing {width}x{height} took {elapsed:.3f}s (> 1s)"


@pytest.mark.performance
class TestScalability:
    """Тесты масштабируемости"""
    
    @pytest.mark.skip(reason="Requires multiple instances")
    def test_horizontal_scaling(self):
        """
        Тест горизонтального масштабирования
        
        Требуется:
        - Kubernetes cluster с несколькими подами
        - Load balancer
        """
        # TODO: Implement with K8s test environment
        pass
    
    @pytest.mark.skip(reason="Requires long-running test")
    def test_availability_99_5_percent(self):
        """
        Тест доступности 99.5% (ТЗ п.6)
        
        Требуется:
        - Длительное тестирование (24+ часов)
        - Мониторинг uptime
        """
        # TODO: Implement with monitoring integration
        pass
