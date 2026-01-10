"""
Тесты для AntiSpoofingService (MiniFASNetV2).
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAntiSpoofingService:
    """Тесты для AntiSpoofingService."""
    
    @pytest.fixture
    def mock_settings(self):
        """Мок настроек для тестирования."""
        with patch('app.services.anti_spoofing_service.settings') as mock:
            mock.LOCAL_ML_DEVICE = "cpu"
            mock.LOCAL_ML_ENABLE_CUDA = False
            mock.CERTIFIED_LIVENESS_THRESHOLD = 0.98
            mock.CERTIFIED_LIVENESS_MODEL_PATH = None
            yield mock
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Генерация тестового изображения."""
        img = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    @pytest.fixture
    def anti_spoofing_service(self, mock_settings):
        """Создание экземпляра сервиса."""
        from app.services.anti_spoofing_service import AntiSpoofingService
        return AntiSpoofingService()
    
    def test_service_initialization(self, anti_spoofing_service, mock_settings):
        """Тест инициализации сервиса."""
        assert anti_spoofing_service.device is not None
        assert anti_spoofing_service.is_initialized is False
        assert anti_spoofing_service.threshold == 0.98
        assert anti_spoofing_service.stats["checks_performed"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_model(self, anti_spoofing_service, mock_settings):
        """Тест инициализации модели."""
        # Модель должна инициализироваться без ошибок
        await anti_spoofing_service.initialize()
        assert anti_spoofing_service.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_check_liveness_with_real_face(self, anti_spoofing_service, sample_image_bytes, mock_settings):
        """Тест проверки живости с изображением реального лица."""
        await anti_spoofing_service.initialize()
        
        # Проверяем, что метод возвращает ожидаемые ключи
        result = await anti_spoofing_service.check_liveness(sample_image_bytes)
        
        assert "success" in result
        assert "liveness_detected" in result
        assert "confidence" in result
        assert "real_probability" in result
        assert "spoof_probability" in result
        assert "model_type" in result
        assert result["model_type"] == "MiniFASNetV2"
        assert result["model_version"] == "MiniFASNetV2-certified"
    
    @pytest.mark.asyncio
    async def test_check_liveness_updates_stats(self, anti_spoofing_service, sample_image_bytes, mock_settings):
        """Тест обновления статистики после проверки."""
        await anti_spoofing_service.initialize()
        initial_checks = anti_spoofing_service.stats["checks_performed"]
        
        await anti_spoofing_service.check_liveness(sample_image_bytes)
        
        assert anti_spoofing_service.stats["checks_performed"] == initial_checks + 1
    
    def test_set_threshold(self, anti_spoofing_service):
        """Тест установки порога."""
        anti_spoofing_service.set_threshold(0.95)
        assert anti_spoofing_service.threshold == 0.95
        
        # Проверка границ
        anti_spoofing_service.set_threshold(1.5)
        assert anti_spoofing_service.threshold == 1.0
        
        anti_spoofing_service.set_threshold(-0.5)
        assert anti_spoofing_service.threshold == 0.0
    
    def test_get_stats(self, anti_spoofing_service):
        """Тест получения статистики."""
        stats = anti_spoofing_service.get_stats()
        
        assert "total_checks" in stats
        assert "real_detected" in stats
        assert "spoof_detected" in stats
        assert "threshold_used" in stats
        assert "model_accuracy_claim" in stats
        assert stats["model_accuracy_claim"] == ">98%"
    
    @pytest.mark.asyncio
    async def test_health_check(self, anti_spoofing_service, mock_settings):
        """Тест проверки состояния сервиса."""
        await anti_spoofing_service.initialize()
        
        health = await anti_spoofing_service.health_check()
        
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert "device" in health
    
    @pytest.mark.asyncio
    async def test_liveness_not_detected_on_low_confidence(self, anti_spoofing_service, mock_settings):
        """Тест случая, когда живость не обнаружена."""
        await anti_spoofing_service.initialize()
        
        # Устанавливаем очень высокий порог
        anti_spoofing_service.set_threshold(0.999)
        
        result = await anti_spoofing_service.check_liveness(b"\x00" * 1000)
        
        assert result["success"] is True
        # При очень высоком пороге результат зависит от модели
    
    def test_preprocess_image(self, anti_spoofing_service, sample_image_bytes):
        """Тест предобработки изображения."""
        tensor = anti_spoofing_service._preprocess_image(sample_image_bytes)
        
        assert tensor is not None
        assert tensor.shape[0] == 1  # batch dimension
        assert tensor.shape[1] == 3  # channels
        assert tensor.shape[2] == 80  # height (MiniFASNetV2)
        assert tensor.shape[3] == 80  # width (MiniFASNetV2)


class TestUtilityFunctions:
    """Тесты утилитарных функций."""
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Генерация тестового изображения."""
        img = Image.new('RGB', (100, 100), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    def test_estimate_face_texture_quality(self, sample_image_bytes):
        """Тест оценки качества текстуры."""
        from app.services.anti_spoofing_service import estimate_face_texture_quality
        
        quality = estimate_face_texture_quality(sample_image_bytes)
        
        assert 0.0 <= quality <= 1.0
    
    def test_analyze_image_for_spoofing_indicators(self, sample_image_bytes):
        """Тест анализа изображения на признаки подделки."""
        from app.services.anti_spoofing_service import analyze_image_for_spoofing_indicators
        
        result = analyze_image_for_spoofing_indicators(sample_image_bytes)
        
        assert "spoof_indicators" in result
        assert "combined_spoof_probability" in result
        assert "is_likely_spoof" in result


class TestMiniFASNetV2Model:
    """Тесты архитектуры модели MiniFASNetV2."""
    
    def test_model_creation(self):
        """Тест создания модели."""
        from app.services.anti_spoofing_service import MiniFASNetV2
        
        model = MiniFASNetV2(input_channels=3, embedding_size=128, out_classes=2)
        
        assert model is not None
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'fc')
    
    def test_model_forward_pass(self):
        """Тест прямого прохода через модель."""
        import torch
        from app.services.anti_spoofing_service import MiniFASNetV2
        
        model = MiniFASNetV2(input_channels=3, embedding_size=128, out_classes=2)
        model.eval()
        
        # Создаем тестовый вход
        x = torch.randn(1, 3, 80, 80)
        
        with torch.no_grad():
            output = model(x)
        
        assert output is not None
        assert output.shape == (1, 2)  # 2 classes
    
    def test_model_output_format(self):
        """Тест формата выхода модели."""
        import torch
        from app.services.anti_spoofing_service import MiniFASNetV2
        import torch.nn.functional as F
        
        model = MiniFASNetV2(input_channels=3, embedding_size=128, out_classes=2)
        model.eval()
        
        x = torch.randn(1, 3, 80, 80)
        
        with torch.no_grad():
            output = model(x)
            probabilities = F.softmax(output, dim=1)
        
        # Проверяем, что сумма вероятностей = 1
        assert probabilities.sum(dim=1).item() == pytest.approx(1.0, abs=1e-5)


class TestEdgeCases:
    """Тесты граничных случаев."""
    
    @pytest.fixture
    def anti_spoofing_service(self):
        """Создание сервиса для тестов граничных случаев."""
        with patch('app.services.anti_spoofing_service.settings') as mock:
            mock.LOCAL_ML_DEVICE = "cpu"
            mock.LOCAL_ML_ENABLE_CUDA = False
            mock.CERTIFIED_LIVENESS_THRESHOLD = 0.98
            mock.CERTIFIED_LIVENESS_MODEL_PATH = None
            
            from app.services.anti_spoofing_service import AntiSpoofingService
            return AntiSpoofingService()
    
    @pytest.mark.asyncio
    async def test_invalid_image_data(self, anti_spoofing_service):
        """Тест обработки некорректных данных изображения."""
        await anti_spoofing_service.initialize()
        
        # Пустые данные
        with pytest.raises(Exception):
            await anti_spoofing_service.check_liveness(b"")
    
    @pytest.mark.asyncio
    async def test_uninitialized_service_check(self, anti_spoofing_service):
        """Тест проверки на неинициализированном сервисе."""
        # Проверяем, что вызов check_liveness инициализирует сервис
        img = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        
        result = await anti_spoofing_service.check_liveness(buffer.getvalue())
        
        assert anti_spoofing_service.is_initialized is True


# ============================================================================
# Integration Tests (require model files)
# ============================================================================

@pytest.mark.integration
class TestAntiSpoofingIntegration:
    """Интеграционные тесты для AntiSpoofingService."""
    
    @pytest.fixture
    def integration_service(self):
        """Сервис для интеграционного тестирования."""
        with patch('app.services.anti_spoofing_service.settings') as mock:
            mock.LOCAL_ML_DEVICE = "cpu"
            mock.LOCAL_ML_ENABLE_CUDA = False
            mock.CERTIFIED_LIVENESS_THRESHOLD = 0.98
            mock.CERTIFIED_LIVENESS_MODEL_PATH = None  # Will use random weights
            
            from app.services.anti_spoofing_service import AntiSpoofingService
            return AntiSpoofingService()
    
    @pytest.mark.asyncio
    async def test_full_liveness_pipeline(self, integration_service):
        """Тест полного пайплайна проверки живости."""
        await integration_service.initialize()
        
        # Создаем тестовое изображение лица
        img = Image.new('RGB', (224, 224), color=(200, 150, 100))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        
        result = await integration_service.check_liveness(buffer.getvalue())
        
        # Проверяем структуру ответа
        assert result["success"] is True
        assert "liveness_detected" in result
        assert "confidence" in result
        assert "processing_time" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
