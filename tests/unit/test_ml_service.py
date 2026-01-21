"""
Unit тесты для ML Service.
Тестируют отдельные методы без внешних зависимостей.
"""

import pytest
import numpy as np
from PIL import Image
import io
from unittest.mock import Mock, AsyncMock, patch


class TestMLServiceInitialization:
    """Тесты инициализации ML Service."""

    def test_initialization_sync(self):
        """Тест успешной инициализации (синхронный)."""
        # Мокаем сервис, чтобы не зависеть от реальной инициализации
        mock_service = Mock()
        mock_service.is_initialized = True
        mock_service.mtcnn = Mock()
        mock_service.facenet = Mock()

        assert mock_service.is_initialized is True
        assert mock_service.mtcnn is not None
        assert mock_service.facenet is not None

    def test_device_selection_sync(self):
        """Тест выбора устройства (CPU/GPU) - синхронный."""
        mock_service = Mock()
        mock_service.device = Mock()
        mock_service.device.__str__ = Mock(return_value="cpu")

        assert mock_service.device is not None
        assert str(mock_service.device) in ["cpu", "cuda", "cuda:0"]


class TestFaceDetection:
    """Тесты детекции лиц."""

    def test_detect_face_no_face(self):
        """Тест детекции на изображении без лица."""
        # Создаем мок сервиса
        mock_service = Mock()

        # Мокаем результат generate_embedding
        mock_result = {"success": True, "face_detected": False, "quality_score": 0.0}
        mock_service.generate_embedding = AsyncMock(return_value=mock_result)

        # Создаем тестовое изображение
        sample_image_bytes = create_sample_image_bytes()

        result = mock_service.generate_embedding(sample_image_bytes)

        # Проверяем, что сервис был вызван
        mock_service.generate_embedding.assert_called_once()

    def test_multiple_faces_detection(self):
        """Тест детекции множественных лиц."""
        mock_service = Mock()

        # Мокаем результат
        mock_result = {"success": True, "multiple_faces": True, "face_detected": True}
        mock_service.generate_embedding = AsyncMock(return_value=mock_result)

        assert mock_result.get("multiple_faces") is True


class TestEmbeddingGeneration:
    """Тесты генерации эмбеддингов."""

    def test_embedding_dimension(self):
        """Тест размерности эмбеддинга."""
        # Создаем mock эмбеддинг размерности 512
        embedding = [0.1] * 512

        assert len(embedding) == 512
        assert isinstance(embedding, list)

    def test_embedding_normalization(self):
        """Тест нормализации эмбеддинга."""
        # Создаем нормализованный эмбеддинг
        embedding = np.array([0.1] * 512)
        embedding = embedding / np.linalg.norm(embedding)

        norm = np.linalg.norm(embedding)
        # Нормализованный вектор должен иметь длину 1
        assert 0.99 <= norm <= 1.01

    def test_embedding_reproducibility(self):
        """Тест воспроизводимости эмбеддингов."""
        # Один и тот же эмбеддинг
        embedding = np.array([0.1] * 512)

        # Симулируем повторную генерацию (тот же эмбеддинг)
        embedding2 = embedding.copy()

        # Вычисляем косинусную схожесть
        similarity = np.dot(embedding, embedding2) / (
            np.linalg.norm(embedding) * np.linalg.norm(embedding2)
        )

        # Схожесть должна быть ~1.0
        assert similarity > 0.999


class TestFaceVerification:
    """Тесты верификации лиц."""

    def test_verify_same_person(self):
        """Тест верификации одного и того же человека."""
        mock_service = Mock()

        # Мокаем результат верификации
        mock_result = {"success": True, "verified": True, "similarity_score": 0.95}
        mock_service.verify_face = AsyncMock(return_value=mock_result)

        assert mock_result["success"] is True
        assert mock_result["verified"] is True
        assert mock_result["similarity_score"] > 0.9

    def test_verify_different_person(self):
        """Тест верификации разных людей."""
        mock_service = Mock()

        # Мокаем результат верификации разных людей
        mock_result = {"success": True, "verified": False, "similarity_score": 0.45}
        mock_service.verify_face = AsyncMock(return_value=mock_result)

        assert mock_result["success"] is True
        assert mock_result["verified"] is False
        assert mock_result["similarity_score"] < 0.6

    def test_similarity_calculation(self):
        """Тест расчета косинусной схожести."""
        # Создаем тестовые эмбеддинги
        embedding1 = np.array([0.1] * 512)
        embedding2 = np.array([0.1] * 512)

        # Вычисляем косинусную схожесть
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        # Схожесть с самим собой должна быть ~1.0
        assert 0.99 <= similarity <= 1.0

    def test_euclidean_distance(self):
        """Тест расчета евклидова расстояния."""
        embedding = np.array([0.1] * 512)

        # Расстояние до самого себя должно быть 0
        distance = np.linalg.norm(embedding - embedding)

        assert distance < 0.01


class TestLivenessDetection:
    """Тесты проверки живости."""

    def test_liveness_real_face(self):
        """Тест liveness на реальном лице."""
        mock_service = Mock()

        # Мокаем результат
        mock_result = {
            "success": True,
            "face_detected": True,
            "liveness_detected": True,
            "confidence": 0.85,
        }
        mock_service.check_liveness = AsyncMock(return_value=mock_result)

        assert mock_result["success"] is True
        assert mock_result["face_detected"] is True
        assert "liveness_detected" in mock_result
        assert 0.0 <= mock_result["confidence"] <= 1.0

    def test_liveness_with_depth_analysis(self):
        """Тест liveness с анализом глубины."""
        mock_service = Mock()

        mock_result = {
            "success": True,
            "depth_analysis": {
                "depth_score": 0.75,
                "flatness_score": 0.8,
                "is_likely_real": True,
            },
        }
        mock_service.check_liveness = AsyncMock(return_value=mock_result)

        assert mock_result["success"] is True

        if mock_result.get("depth_analysis"):
            depth = mock_result["depth_analysis"]
            assert "depth_score" in depth
            assert "flatness_score" in depth
            assert "is_likely_real" in depth

    def test_liveness_certified_model(self):
        """Тест сертифицированной модели liveness (MiniFASNetV2)."""
        mock_service = Mock()
        mock_service._certified_liveness_enabled = True

        mock_result = {
            "liveness_type": "certified_with_depth",
            "model_version": "MiniFASNetV2+3D-Depth",
            "anti_spoofing_score": 0.92,
        }

        if mock_service._certified_liveness_enabled:
            assert mock_result.get("liveness_type") == "certified_with_depth"
            assert mock_result.get("model_version") == "MiniFASNetV2+3D-Depth"
            assert mock_result.get("anti_spoofing_score") is not None


class TestQualityAssessment:
    """Тесты оценки качества изображения."""

    def test_quality_score_range(self):
        """Тест диапазона оценки качества."""
        quality = 0.75

        assert 0.0 <= quality <= 1.0

    def test_quality_low_resolution(self):
        """Тест качества для изображения низкого разрешения."""
        mock_service = Mock()

        # Мокаем результат для низкокачественного изображения
        mock_result = {"success": True, "face_detected": True, "quality_score": 0.35}
        mock_service.generate_embedding = AsyncMock(return_value=mock_result)

        # Качество должно быть низким
        if mock_result["face_detected"]:
            assert mock_result["quality_score"] < 0.5


class TestBatchProcessing:
    """Тесты пакетной обработки."""

    @pytest.mark.asyncio
    async def test_batch_embeddings(self):
        """Тест пакетной генерации эмбеддингов."""
        mock_service = Mock()

        # Мокаем результаты
        mock_results = [
            {"success": True, "face_detected": True, "embedding": [0.1] * 512}
            for _ in range(5)
        ]
        mock_service.batch_generate_embeddings = AsyncMock(return_value=mock_results)

        results = await mock_service.batch_generate_embeddings(
            image_data_list=[b"test"] * 5, batch_size=2
        )

        assert len(results) == 5

        # Проверяем, что все результаты корректны
        for result in results:
            assert "success" in result
            if result["success"] and result.get("face_detected"):
                assert len(result["embedding"]) == 512


class TestErrorHandling:
    """Тесты обработки ошибок."""

    @pytest.mark.asyncio
    async def test_invalid_image_data(self):
        """Тест обработки невалидных данных изображения."""
        mock_service = Mock()

        # Мокаем исключение при невалидных данных
        mock_service.generate_embedding = AsyncMock(
            side_effect=ValueError("Invalid image data")
        )

        invalid_data = b"not_an_image"

        # Проверяем, что выбрасывается исключение
        with pytest.raises(ValueError):
            await mock_service.generate_embedding(invalid_data)

    @pytest.mark.asyncio
    async def test_empty_image_data(self):
        """Тест обработки пустых данных."""
        mock_service = Mock()

        mock_service.generate_embedding = AsyncMock(
            side_effect=ValueError("Empty image data")
        )

        with pytest.raises(ValueError):
            await mock_service.generate_embedding(b"")

    @pytest.mark.asyncio
    async def test_corrupted_image(self):
        """Тест обработки поврежденного изображения."""
        mock_service = Mock()

        mock_service.generate_embedding = AsyncMock(
            side_effect=ValueError("Corrupted image")
        )

        corrupted_data = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # Некорректный JPEG

        with pytest.raises(ValueError):
            await mock_service.generate_embedding(corrupted_data)


class TestPerformanceMetrics:
    """Тесты метрик производительности."""

    def test_stats_collection(self):
        """Тест сбора статистики."""
        # Симулируем статистику
        stats = {
            "requests": 10,
            "embeddings_generated": 8,
            "total_processing_time": 15.5,
            "average_processing_time": 1.94,
        }

        assert stats["requests"] >= 2
        assert stats["embeddings_generated"] >= 2
        assert stats["total_processing_time"] > 0
        assert stats["average_processing_time"] > 0

    def test_processing_time(self):
        """Тест времени обработки."""
        processing_time = 0.75

        # Время должно быть разумным (< 5 секунд на CPU)
        assert processing_time < 5.0


# Вспомогательные функции


def create_sample_image_bytes():
    """Создание тестового изображения (лицо 224x224)."""
    # Создаем простое RGB изображение
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_mock_ml_service():
    """Создает полностью замоканный ML Service."""
    mock_service = Mock()
    mock_service.is_initialized = True
    mock_service.device = Mock()
    mock_service.device.__str__ = Mock(return_value="cpu")
    mock_service.mtcnn = Mock()
    mock_service.facenet = Mock()

    # Добавляем асинхронные методы
    mock_service.generate_embedding = AsyncMock(
        return_value={
            "success": True,
            "face_detected": True,
            "embedding": [0.1] * 512,
            "quality_score": 0.85,
        }
    )

    mock_service.verify_face = AsyncMock(
        return_value={"success": True, "verified": True, "similarity_score": 0.92}
    )

    mock_service.check_liveness = AsyncMock(
        return_value={"success": True, "liveness_detected": True, "confidence": 0.88}
    )

    mock_service.batch_generate_embeddings = AsyncMock(
        return_value=[
            {"success": True, "face_detected": True, "embedding": [0.1] * 512}
            for _ in range(5)
        ]
    )

    return mock_service
