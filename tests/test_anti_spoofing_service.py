"""
Unit и integration тесты для AntiSpoofingService.
"""

import asyncio
import gc
import io

import numpy as np
import pytest
import pytest_asyncio
import torch
from PIL import Image

from app.services.anti_spoofing_service import (
    AntiSpoofingService,
    MiniFASNetV2,
    analyze_image_for_spoofing_indicators,
    estimate_face_texture_quality,
    get_anti_spoofing_service,
    reset_anti_spoofing_service,
)
from app.utils.exceptions import MLServiceError, ProcessingError, ValidationError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_image_bytes():
    """Создание тестового изображения."""
    img = Image.new("RGB", (80, 80), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_large_bytes():
    """Создание большого тестового изображения."""
    img = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="function")
def service():
    """
    Fixture для AntiSpoofingService - СИНХРОННЫЙ!
    Возвращает неинициализированный сервис.
    """
    svc = AntiSpoofingService()
    yield svc

    # Cleanup
    if svc.model is not None:
        svc.model.cpu()
        del svc.model
        svc.model = None

    svc.is_initialized = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()


@pytest_asyncio.fixture(scope="function")
async def initialized_service():
    """
    Fixture для инициализированного сервиса - ASYNC!
    """
    await reset_anti_spoofing_service()

    svc = AntiSpoofingService()
    await svc.initialize()

    yield svc

    # Cleanup
    if svc.model is not None:
        svc.model.cpu()
        del svc.model
        svc.model = None

    svc.is_initialized = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()

    await reset_anti_spoofing_service()


# =============================================================================
# Model Architecture Tests (Sync)
# =============================================================================


def test_minifasnet_v2_architecture():
    """Тест архитектуры модели."""
    model = MiniFASNetV2(input_channels=3, embedding_size=128, out_classes=2)

    # Переводим модель в режим оценки для корректной работы BatchNorm
    model.eval()

    assert hasattr(model, "conv1")
    assert hasattr(model, "conv2_dw")
    assert hasattr(model, "fc")
    assert hasattr(model, "global_avg_pool")

    dummy_input = torch.randn(1, 3, 80, 80)
    with torch.no_grad():  # Отключаем градиенты для теста
        output = model(dummy_input)

    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"

    del model, dummy_input, output
    gc.collect()


def test_minifasnet_v2_forward_pass():
    """Тест forward pass модели."""
    model = MiniFASNetV2()
    model.eval()

    with torch.no_grad():
        input_tensor = torch.randn(2, 3, 80, 80)
        output = model(input_tensor)

    assert output.shape == (2, 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    del model, input_tensor, output
    gc.collect()


def test_minifasnet_v2_embedding():
    """Тест извлечения эмбеддингов."""
    model = MiniFASNetV2(embedding_size=128)
    model.eval()

    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 80, 80)
        embedding = model.get_embedding(input_tensor)

    assert embedding.shape == (1, 256)

    del model, input_tensor, embedding
    gc.collect()


# =============================================================================
# Service Initialization Tests (Async)
# =============================================================================


@pytest.mark.asyncio
async def test_service_initialization():
    """Тест инициализации сервиса."""
    await reset_anti_spoofing_service()
    service = AntiSpoofingService()

    assert not service.is_initialized

    await service.initialize()

    assert service.is_initialized
    assert service.model is not None
    assert service.model_version is not None

    await reset_anti_spoofing_service()


@pytest.mark.asyncio
async def test_singleton_pattern():
    """Тест singleton pattern."""
    await reset_anti_spoofing_service()

    service1 = await get_anti_spoofing_service()
    service2 = await get_anti_spoofing_service()

    assert service1 is service2

    await reset_anti_spoofing_service()


# =============================================================================
# Input Validation Tests (Async)
# =============================================================================


@pytest.mark.asyncio
async def test_empty_image_data(initialized_service):
    """Тест на пустые данные."""
    with pytest.raises(ValidationError, match="Empty image data"):
        await initialized_service.check_liveness(b"")


@pytest.mark.asyncio
async def test_oversized_image(initialized_service):
    """Тест на слишком большое изображение."""
    large_data = b"x" * (11 * 1024 * 1024)

    with pytest.raises(ValidationError, match="exceeds maximum"):
        await initialized_service.check_liveness(large_data)


@pytest.mark.asyncio
async def test_invalid_image_format(initialized_service):
    """Тест на невалидный формат изображения."""
    invalid_data = b"not an image"

    with pytest.raises(ProcessingError):
        await initialized_service.check_liveness(invalid_data)


# =============================================================================
# Liveness Check Tests (Async)
# =============================================================================


@pytest.mark.asyncio
async def test_basic_liveness_check(initialized_service, sample_image_bytes):
    """Базовый тест проверки живости."""
    result = await initialized_service.check_liveness(sample_image_bytes)

    assert result["success"] is True
    assert "liveness_detected" in result
    assert "confidence" in result
    assert "real_probability" in result
    assert "spoof_probability" in result
    assert "processing_time" in result
    assert "inference_time" in result

    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["real_probability"] <= 1
    assert 0 <= result["spoof_probability"] <= 1
    assert abs(result["real_probability"] + result["spoof_probability"] - 1.0) < 0.001


@pytest.mark.asyncio
async def test_liveness_check_with_features(initialized_service, sample_image_bytes):
    """Тест с извлечением признаков."""
    result = await initialized_service.check_liveness(
        sample_image_bytes, return_features=True
    )

    assert "features" in result
    assert "embedding" in result["features"]
    assert isinstance(result["features"]["embedding"], list)


@pytest.mark.asyncio
async def test_liveness_check_with_auxiliary(initialized_service, sample_image_bytes):
    """Тест с дополнительными проверками."""
    result = await initialized_service.check_liveness(
        sample_image_bytes, enable_auxiliary_checks=True
    )

    assert "auxiliary_checks" in result or result["success"]


@pytest.mark.asyncio
async def test_batch_liveness_check(initialized_service, sample_image_bytes):
    """Тест пакетной проверки."""
    images = [sample_image_bytes] * 3

    results = await initialized_service.batch_check_liveness(images)

    assert len(results) == 3
    assert all(r["success"] for r in results)


# =============================================================================
# Statistics Tests (Async)
# =============================================================================


@pytest.mark.asyncio
async def test_statistics_update(initialized_service, sample_image_bytes):
    """Тест обновления статистики."""
    initial_stats = initialized_service.get_stats()
    initial_count = initial_stats["total_checks"]

    await initialized_service.check_liveness(sample_image_bytes)

    updated_stats = initialized_service.get_stats()

    assert updated_stats["total_checks"] == initial_count + 1
    assert updated_stats["avg_processing_time"] > 0


@pytest.mark.asyncio
async def test_statistics_reset(initialized_service, sample_image_bytes):
    """Тест сброса статистики."""
    await initialized_service.check_liveness(sample_image_bytes)

    initialized_service.reset_stats()
    stats = initialized_service.get_stats()

    assert stats["total_checks"] == 0
    assert stats["real_detected"] == 0
    assert stats["spoof_detected"] == 0


# =============================================================================
# Threshold Tests (Async)
# =============================================================================


@pytest.mark.asyncio
async def test_threshold_adjustment(initialized_service):
    """Тест изменения порога."""
    original_threshold = initialized_service.threshold

    initialized_service.set_threshold(0.7)
    assert initialized_service.threshold == 0.7

    initialized_service.set_threshold(0.3)
    assert initialized_service.threshold == 0.3

    initialized_service.set_threshold(original_threshold)


def test_invalid_threshold():
    """Тест невалидного порога."""
    service = AntiSpoofingService()

    with pytest.raises(ValueError):
        service.set_threshold(1.5)

    with pytest.raises(ValueError):
        service.set_threshold(-0.1)


# =============================================================================
# Auxiliary Functions Tests (Sync)
# =============================================================================


def test_texture_quality_estimation(sample_image_bytes):
    """Тест оценки качества текстуры."""
    score = estimate_face_texture_quality(sample_image_bytes)

    assert 0 <= score <= 1
    assert isinstance(score, float)


def test_spoofing_indicators_analysis(sample_image_bytes):
    """Тест анализа признаков подделки."""
    result = analyze_image_for_spoofing_indicators(sample_image_bytes)

    assert "spoof_indicators" in result
    assert "combined_spoof_probability" in result
    assert "is_likely_spoof" in result

    indicators = result["spoof_indicators"]
    assert "moire_score" in indicators
    assert "uniform_regions_score" in indicators
    assert "edge_density" in indicators
    assert "texture_quality" in indicators


# =============================================================================
# Health Check Tests (Async)
# =============================================================================


@pytest.mark.asyncio
async def test_health_check(initialized_service):
    """Тест health check."""
    health = await initialized_service.health_check()

    assert "status" in health
    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert health["model_loaded"] is True
    assert "stats" in health


# =============================================================================
# Performance Tests (Async)
# =============================================================================


@pytest.mark.asyncio
async def test_inference_speed(initialized_service, sample_image_bytes):
    """Тест скорости инференса."""
    # Warmup
    await initialized_service.check_liveness(sample_image_bytes)

    import time

    start = time.time()
    result = await initialized_service.check_liveness(sample_image_bytes)
    elapsed = time.time() - start

    if initialized_service.device.type == "cuda":
        assert result["inference_time"] < 0.05, "GPU inference too slow"
    else:
        assert result["inference_time"] < 0.2, "CPU inference too slow"


@pytest.mark.asyncio
async def test_concurrent_requests(initialized_service, sample_image_bytes):
    """Тест конкурентных запросов."""
    tasks = [initialized_service.check_liveness(sample_image_bytes) for _ in range(10)]

    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert all(r["success"] for r in results)


# =============================================================================
# Edge Cases Tests (Async)
# =============================================================================


@pytest.mark.asyncio
async def test_grayscale_image(initialized_service):
    """Тест на grayscale изображение."""
    img = Image.new("L", (80, 80), color=128)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    result = await initialized_service.check_liveness(buffer.getvalue())

    assert result["success"] is True


@pytest.mark.asyncio
async def test_rgba_image(initialized_service):
    """Тест на RGBA изображение."""
    img = Image.new("RGBA", (80, 80), color=(128, 128, 128, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    result = await initialized_service.check_liveness(buffer.getvalue())

    assert result["success"] is True


@pytest.mark.asyncio
async def test_small_image(initialized_service):
    """Тест на маленькое изображение."""
    img = Image.new("RGB", (20, 20), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    result = await initialized_service.check_liveness(buffer.getvalue())

    assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
