"""
Тесты для ML сервиса (OptimizedMLService).
Критически важный модуль с низким покрытием - цель: увеличить с 14.64% до 80%+
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import torch
from PIL import Image
import io

# Mock всех внешних зависимостей для избежания конфликтов
with patch('app.services.ml_service.MTCNN'), \
     patch('app.services.ml_service.InceptionResnetV1'), \
     patch('app.services.ml_service.cv2'), \
     patch('app.services.ml_service.settings'):
    
    from app.services.ml_service import OptimizedMLService, MLServiceError
    from app.utils.exceptions import ProcessingError


class TestOptimizedMLService:
    """Тесты для OptimizedMLService"""
    
    @pytest.fixture
    def ml_service(self):
        """Фикстура для создания ML сервиса"""
        with patch('app.services.ml_service.settings') as mock_settings:
            mock_settings.LOCAL_ML_DEVICE = "cpu"
            mock_settings.LOCAL_ML_ENABLE_CUDA = False
            mock_settings.LOCAL_ML_FACE_DETECTION_THRESHOLD = 0.6
            mock_settings.LOCAL_ML_QUALITY_THRESHOLD = 0.5
            mock_settings.LOCAL_ML_BATCH_SIZE = 32
            mock_settings.LOCAL_ML_ENABLE_PERFORMANCE_MONITORING = True
            
            service = OptimizedMLService()
            
            # Mock модели
            service.mtcnn = Mock()
            service.facenet = Mock()
            service.is_initialized = True
            
            return service
    
    @pytest.fixture
    def sample_image_data(self):
        """Фикстура с образцом данных изображения"""
        # Создаем простое тестовое изображение
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    @pytest.fixture
    def sample_embedding(self):
        """Фикстура с образцом эмбеддинга"""
        return np.random.rand(512).astype(np.float32)
    
    # === ИНИЦИАЛИЗАЦИЯ ===
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Тест успешной инициализации ML сервиса"""
        with patch('app.services.ml_service.settings') as mock_settings, \
             patch('app.services.ml_service.MTCNN') as mock_mtcnn, \
             patch('app.services.ml_service.InceptionResnetV1') as mock_facenet:
            
            mock_settings.LOCAL_ML_DEVICE = "cpu"
            mock_settings.LOCAL_ML_ENABLE_CUDA = False
            
            service = OptimizedMLService()
            service.is_initialized = False
            
            # Вызываем инициализацию
            await service.initialize()
            
            # Проверяем что модели созданы
            assert service.mtcnn is not None
            assert service.facenet is not None
            assert service.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Тест повторной инициализации (должна быть пропущена)"""
        service = OptimizedMLService()
        service.is_initialized = True
        
        with patch.object(service, 'mtcnn', Mock()) as mock_mtcnn:
            await service.initialize()
            mock_mtcnn.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Тест неудачной инициализации"""
        with patch('app.services.ml_service.MTCNN', side_effect=Exception("Init error")):
            service = OptimizedMLService()
            service.is_initialized = False
            
            with pytest.raises(MLServiceError):
                await service.initialize()
    
    # === ПРОВЕРКА ЗДОРОВЬЯ ===
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, ml_service):
        """Тест успешной проверки здоровья"""
        result = await ml_service.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Тест проверки здоровья без инициализации"""
        service = OptimizedMLService()
        service.is_initialized = False
        
        with patch.object(service, 'initialize', new_callable=AsyncMock):
            result = await service.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Тест неудачной проверки здоровья"""
        service = OptimizedMLService()
        service.is_initialized = False
        
        with patch.object(service, 'initialize', side_effect=Exception("Init failed")):
            result = await service.health_check()
            assert result is False
    
    # === ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ ===
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, ml_service, sample_image_data, sample_embedding):
        """Тест успешной генерации эмбеддинга"""
        # Mock MTCNN возвращает face_crop и высокую вероятность
        mock_face_crop = torch.rand(3, 224, 224)
        ml_service.mtcnn.return_value = (mock_face_crop, 0.9)
        
        # Mock генерации эмбеддинга и оценки качества
        with patch.object(ml_service, '_generate_face_embedding', return_value=sample_embedding), \
             patch.object(ml_service, '_assess_face_quality', return_value=0.8):
            
            result = await ml_service.generate_embedding(sample_image_data)
        
        assert result["success"] is True
        assert result["face_detected"] is True
        assert "embedding" in result
        assert result["quality_score"] == 0.8
        assert "processing_time" in result
    
    @pytest.mark.asyncio
    async def test_generate_embedding_no_face(self, ml_service, sample_image_data):
        """Тест генерации эмбеддинга без лица"""
        # Mock MTCNN возвращает None (лицо не найдено)
        ml_service.mtcnn.return_value = (None, 0.1)
        
        result = await ml_service.generate_embedding(sample_image_data)
        
        assert result["success"] is True
        assert result["face_detected"] is False
        assert "error" in result
        assert result["error"] == "No face detected"
    
    @pytest.mark.asyncio
    async def test_generate_embedding_low_probability(self, ml_service, sample_image_data):
        """Тест генерации эмбеддинга с низкой вероятностью лица"""
        # Mock MTCNN возвращает низкую вероятность
        ml_service.mtcnn.return_value = (torch.rand(3, 224, 224), 0.3)
        
        result = await ml_service.generate_embedding(sample_image_data)
        
        assert result["success"] is True
        assert result["face_detected"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_generate_embedding_processing_error(self, ml_service, sample_image_data):
        """Тест ошибки при генерации эмбеддинга"""
        ml_service.mtcnn.side_effect = Exception("Processing failed")
        
        with pytest.raises(MLServiceError):
            await ml_service.generate_embedding(sample_image_data)
    
    # === ВЕРИФИКАЦИЯ ЛИЦ ===
    
    @pytest.mark.asyncio
    async def test_verify_face_success(self, ml_service, sample_image_data, sample_embedding):
        """Тест успешной верификации лица"""
        # Mock генерации эмбеддинга
        embedding_result = {
            "success": True,
            "face_detected": True,
            "embedding": sample_embedding,
            "quality_score": 0.8
        }
        ml_service.generate_embedding = AsyncMock(return_value=embedding_result)
        ml_service._compute_cosine_similarity = Mock(return_value=0.85)
        ml_service._compute_euclidean_distance = Mock(return_value=0.3)
        
        result = await ml_service.verify_face(sample_image_data, sample_embedding, threshold=0.8)
        
        assert result["success"] is True
        assert result["verified"] is True
        assert result["confidence"] == 0.85
        assert result["similarity_score"] == 0.85
        assert "processing_time" in result
    
    @pytest.mark.asyncio
    async def test_verify_face_no_face_detected(self, ml_service, sample_image_data, sample_embedding):
        """Тест верификации без обнаруженного лица"""
        # Mock генерации эмбеддинга без лица
        embedding_result = {
            "success": True,
            "face_detected": False
        }
        ml_service.generate_embedding = AsyncMock(return_value=embedding_result)
        
        result = await ml_service.verify_face(sample_image_data, sample_embedding)
        
        assert result["success"] is True
        assert result["verified"] is False
        assert result["face_detected"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_verify_face_below_threshold(self, ml_service, sample_image_data, sample_embedding):
        """Тест верификации ниже порога"""
        # Mock генерации эмбеддинга
        embedding_result = {
            "success": True,
            "face_detected": True,
            "embedding": sample_embedding,
            "quality_score": 0.8
        }
        ml_service.generate_embedding = AsyncMock(return_value=embedding_result)
        ml_service._compute_cosine_similarity = Mock(return_value=0.6)
        ml_service._compute_euclidean_distance = Mock(return_value=0.8)
        
        result = await ml_service.verify_face(sample_image_data, sample_embedding, threshold=0.8)
        
        assert result["success"] is True
        assert result["verified"] is False  # 0.6 < 0.8
        assert result["confidence"] == 0.6
    
    # === ПРОВЕРКА ЖИВОСТИ ===
    
    @pytest.mark.asyncio
    async def test_check_liveness_success(self, ml_service, sample_image_data):
        """Тест успешной проверки живости"""
        # Mock детекции лица и эвристической проверки
        with patch.object(ml_service, '_detect_faces_optimized', return_value=(True, torch.rand(3, 224, 224), False, 1)), \
             patch.object(ml_service, '_heuristic_liveness_check', return_value={
                 "liveness_detected": True,
                 "confidence": 0.8,
                 "image_quality": 0.7,
                 "recommendations": []
             }):
            
            result = await ml_service.check_liveness(sample_image_data)
        
        assert result["success"] is True
        assert result["liveness_detected"] is True
        assert result["confidence"] == 0.8
        assert "face_detected" in result
        assert "processing_time" in result
    
    @pytest.mark.asyncio
    async def test_check_liveness_no_face(self, ml_service, sample_image_data):
        """Тест проверки живости без лица"""
        with patch.object(ml_service, '_detect_faces_optimized', return_value=(False, None, False, 0)):
            result = await ml_service.check_liveness(sample_image_data)
        
        assert result["success"] is True
        assert result["liveness_detected"] is False
        assert result["face_detected"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_check_liveness_challenge_type(self, ml_service, sample_image_data):
        """Тест проверки живости с типом вызова"""
        with patch.object(ml_service, '_detect_faces_optimized', return_value=(True, torch.rand(3, 224, 224), False, 1)), \
             patch.object(ml_service, '_heuristic_liveness_check', return_value={
                 "liveness_detected": True,
                 "confidence": 0.8
             }):
            
            result = await ml_service.check_liveness(sample_image_data, challenge_type="active")
        
        assert result["success"] is True
        assert result["liveness_type"] == "heuristic_passive_non_certified"
    
    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===
    
    def test_load_image_pil_success(self, ml_service, sample_image_data):
        """Тест успешной загрузки изображения PIL"""
        result = ml_service._load_image_pil(sample_image_data)
        
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
    
    def test_load_image_pil_invalid_data(self, ml_service):
        """Тест загрузки невалидных данных изображения"""
        invalid_data = b"invalid image data"
        
        with pytest.raises(ProcessingError):
            ml_service._load_image_pil(invalid_data)
    
    def test_image_to_numpy(self, ml_service):
        """Тест конвертации PIL в numpy"""
        img = Image.new('RGB', (100, 100), color='red')
        result = ml_service._image_to_numpy(img)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)
    
    def test_detect_faces_optimized_success(self, ml_service):
        """Тест оптимизированной детекции лиц"""
        # Mock MTCNN
        mock_face_crop = torch.rand(3, 224, 224)
        ml_service.mtcnn.return_value = (mock_face_crop, 0.9)
        ml_service.mtcnn.detect.return_value = (np.array([[10, 10, 100, 100]]), np.array([0.9]))
        
        img = Image.new('RGB', (224, 224))
        face_detected, face_crop, multiple_faces, faces_count = ml_service._detect_faces_optimized(img)
        
        assert face_detected is True
        assert face_crop is not None
        assert multiple_faces is False
        assert faces_count == 1
    
    def test_detect_faces_optimized_no_face(self, ml_service):
        """Тест оптимизированной детекции лиц без лица"""
        ml_service.mtcnn.return_value = (None, 0.1)
        
        img = Image.new('RGB', (224, 224))
        face_detected, face_crop, multiple_faces, faces_count = ml_service._detect_faces_optimized(img)
        
        assert face_detected is False
        assert face_crop is None
        assert multiple_faces is False
        assert faces_count == 0
    
    def test_assess_face_quality_success(self, ml_service):
        """Тест оценки качества лица"""
        mock_face_crop = torch.rand(3, 224, 224)
        
        with patch('app.services.ml_service.cv2') as mock_cv2:
            mock_cv2.cvtColor.return_value = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            mock_cv2.Laplacian.return_value.var.return_value = 500
            
            result = ml_service._assess_face_quality(mock_face_crop)
            
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
    
    def test_assess_face_quality_error(self, ml_service):
        """Тест оценки качества лица с ошибкой"""
        mock_face_crop = torch.rand(3, 224, 224)
        
        with patch('app.services.ml_service.cv2', side_effect=Exception("CV2 error")):
            result = ml_service._assess_face_quality(mock_face_crop)
            
            assert result == 0.5  # Значение по умолчанию при ошибке
    
    def test_generate_face_embedding_success(self, ml_service, sample_embedding):
        """Тест генерации эмбеддинга лица"""
        mock_face_crop = torch.rand(1, 3, 224, 224)
        
        # Mock Facenet возвращает тензор
        mock_embedding = torch.tensor(sample_embedding).unsqueeze(0)
        ml_service.facenet.return_value = mock_embedding
        
        # Mock нормализации
        with patch('torch.norm') as mock_norm, \
             patch('torch.no_grad'):
            mock_norm.return_value = torch.tensor(1.0)
            
            result = ml_service._generate_face_embedding(mock_face_crop)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (512,)
    
    def test_generate_face_embedding_error(self, ml_service):
        """Тест генерации эмбеддинга лица с ошибкой"""
        mock_face_crop = torch.rand(1, 3, 224, 224)
        ml_service.facenet.side_effect = Exception("Facenet error")
        
        with pytest.raises(ProcessingError):
            ml_service._generate_face_embedding(mock_face_crop)
    
    def test_compute_cosine_similarity_success(self, ml_service, sample_embedding):
        """Тест вычисления косинусной схожести"""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])
        
        result = ml_service._compute_cosine_similarity(embedding1, embedding2)
        
        assert isinstance(result, float)
        assert result == 1.0  # Идентичные векторы
    
    def test_compute_cosine_similarity_opposite(self, ml_service):
        """Тест косинусной схожести противоположных векторов"""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([-1.0, 0.0, 0.0])
        
        result = ml_service._compute_cosine_similarity(embedding1, embedding2)
        
        assert result == -1.0
    
    def test_compute_cosine_similarity_error(self, ml_service):
        """Тест ошибки при вычислении косинусной схожести"""
        with patch('numpy.linalg.norm', side_effect=Exception("Norm error")):
            result = ml_service._compute_cosine_similarity(np.array([1, 2, 3]), np.array([4, 5, 6]))
            
            assert result == 0.0  # Значение по умолчанию при ошибке
    
    def test_compute_euclidean_distance_success(self, ml_service):
        """Тест вычисления евклидового расстояния"""
        embedding1 = np.array([0.0, 0.0, 0.0])
        embedding2 = np.array([3.0, 4.0, 0.0])
        
        result = ml_service._compute_euclidean_distance(embedding1, embedding2)
        
        assert result == 5.0  # sqrt(3^2 + 4^2) = 5
    
    def test_compute_euclidean_distance_error(self, ml_service):
        """Тест ошибки при вычислении евклидового расстояния"""
        with patch('numpy.linalg.norm', side_effect=Exception("Norm error")):
            result = ml_service._compute_euclidean_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))
            
            assert result == float("inf")
    
    def test_heuristic_liveness_check_success(self, ml_service):
        """Тест эвристической проверки живости"""
        mock_face_crop = torch.rand(3, 224, 224)
        mock_image = Image.new('RGB', (224, 224))
        
        with patch.object(ml_service, '_assess_image_quality', return_value=0.8):
            result = ml_service._heuristic_liveness_check(mock_image, mock_face_crop)
        
        assert "liveness_detected" in result
        assert "confidence" in result
        assert "image_quality" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_heuristic_liveness_check_error(self, ml_service):
        """Тест эвристической проверки живости с ошибкой"""
        mock_face_crop = torch.rand(3, 224, 224)
        mock_image = Image.new('RGB', (224, 224))
        
        with patch.object(ml_service, '_assess_image_quality', side_effect=Exception("Analysis error")):
            result = ml_service._heuristic_liveness_check(mock_image, mock_face_crop)
        
        assert result["liveness_detected"] is False
        assert result["confidence"] == 0.0
        assert "analysis_failed" in result["recommendations"]
    
    def test_assess_image_quality_success(self, ml_service, sample_image_data):
        """Тест оценки качества изображения"""
        img = Image.new('RGB', (100, 100))
        
        with patch('app.services.ml_service.cv2') as mock_cv2:
            mock_cv2.cvtColor.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            mock_cv2.Laplacian.return_value.var.return_value = 500
            
            result = ml_service._assess_image_quality(np.array(img))
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_assess_image_quality_error(self, ml_service):
        """Тест оценки качества изображения с ошибкой"""
        with patch('app.services.ml_service.cv2', side_effect=Exception("CV2 error")):
            result = ml_service._assess_image_quality(np.array(Image.new('RGB', (100, 100))))
        
        assert result == 0.5
    
    def test_update_stats(self, ml_service):
        """Тест обновления статистики"""
        initial_requests = ml_service.stats["requests"]
        
        ml_service._update_stats(0.5, "face_detections")
        
        assert ml_service.stats["requests"] == initial_requests + 1
        assert ml_service.stats["face_detections"] == 1
        assert ml_service.stats["total_processing_time"] == 0.5
    
    def test_update_stats_monitoring_disabled(self, ml_service):
        """Тест обновления статистики с отключенным мониторингом"""
        ml_service.enable_performance_monitoring = False
        initial_requests = ml_service.stats["requests"]
        
        ml_service._update_stats(0.5, "face_detections")
        
        assert ml_service.stats["requests"] == initial_requests  # Не изменилось
    
    def test_get_stats(self, ml_service):
        """Тест получения статистики"""
        result = ml_service.get_stats()
        
        assert "stats" in result
        assert "device" in result
        assert "models_initialized" in result
        assert "cuda_available" in result
        assert "optimizations_applied" in result
        
        assert result["models_initialized"] is True
        assert isinstance(result["optimizations_applied"], list)
    
    # === ОБРАБОТКА ОШИБОК ===
    
    @pytest.mark.asyncio
    async def test_generate_embedding_with_processing_error(self, ml_service):
        """Тест обработки ошибки при генерации эмбеддинга"""
        ml_service.mtcnn.side_effect = ProcessingError("Processing failed")
        
        with pytest.raises(MLServiceError):
            await ml_service.generate_embedding(b"test data")
    
    @pytest.mark.asyncio
    async def test_verify_face_with_processing_error(self, ml_service, sample_embedding):
        """Тест обработки ошибки при верификации"""
        ml_service.generate_embedding = AsyncMock(side_effect=ProcessingError("ML processing failed"))
        
        with pytest.raises(MLServiceError):
            await ml_service.verify_face(b"test data", sample_embedding)
    
    @pytest.mark.asyncio
    async def test_check_liveness_with_processing_error(self, ml_service):
        """Тест обработки ошибки при проверке живости"""
        with patch.object(ml_service, '_detect_faces_optimized', side_effect=Exception("Detection failed")):
            with pytest.raises(MLServiceError):
                await ml_service.check_liveness(b"test data")


# === ТЕСТЫ АЛИАСОВ ===

class TestMLServiceAlias:
    """Тесты для обратной совместимости (MLService alias)"""
    
    def test_ml_service_alias(self):
        """Тест что MLService это алиас OptimizedMLService"""
        from app.services.ml_service import MLService, OptimizedMLService
        
        assert MLService is OptimizedMLService


# === ИНТЕГРАЦИОННЫЕ ТЕСТЫ ===

class TestMLServiceIntegration:
    """Интеграционные тесты для ML сервиса"""
    
    @pytest.mark.asyncio
    async def test_full_embedding_workflow(self):
        """Тест полного рабочего процесса генерации эмбеддинга"""
        with patch('app.services.ml_service.settings') as mock_settings, \
             patch('app.services.ml_service.MTCNN') as mock_mtcnn, \
             patch('app.services.ml_service.InceptionResnetV1'):
            
            mock_settings.LOCAL_ML_DEVICE = "cpu"
            mock_settings.LOCAL_ML_ENABLE_CUDA = False
            
            service = OptimizedMLService()
            service.is_initialized = True
            
            # Создаем Mock объекты для mtcnn и facenet
            service.mtcnn = Mock()
            service.facenet = Mock()
            
            # Настраиваем правильные значения для атрибутов
            service.face_detection_threshold = 0.6
            service.quality_threshold = 0.5
            
            # Mock всех необходимых методов
            service.mtcnn.return_value = (torch.rand(3, 224, 224), 0.9)
            
            with patch.object(service, '_generate_face_embedding', return_value=np.random.rand(512)), \
                 patch.object(service, '_assess_face_quality', return_value=0.8):
                
                # Создаем тестовое изображение
                img = Image.new('RGB', (224, 224), color='red')
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                
                result = await service.generate_embedding(img_bytes.getvalue())
            
            # Проверяем результат
            assert result["success"] is True
            assert result["face_detected"] is True
            assert "embedding" in result
            assert isinstance(result["embedding"], np.ndarray)
            assert result["quality_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_full_verification_workflow(self):
        """Тест полного рабочего процесса верификации"""
        service = OptimizedMLService()
        service.is_initialized = True
        
        # Mock всех необходимых методов
        embedding_result = {
            "success": True,
            "face_detected": True,
            "embedding": np.random.rand(512),
            "quality_score": 0.8
        }
        service.generate_embedding = AsyncMock(return_value=embedding_result)
        service._compute_cosine_similarity = Mock(return_value=0.85)
        service._compute_euclidean_distance = Mock(return_value=0.3)
        
        result = await service.verify_face(b"test image", np.random.rand(512), threshold=0.8)
        
        assert result["success"] is True
        assert result["verified"] is True
        assert result["similarity_score"] == 0.85
        assert result["confidence"] == 0.85
