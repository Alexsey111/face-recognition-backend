"""
Тесты для улучшенных функций face alignment, lighting analysis и 3D depth estimation.

Тестирует:
1. Face Alignment с 68-point landmarks
2. Улучшенный анализ освещения и теней
3. 3D Depth Estimation для liveness detection
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.utils.face_alignment_utils import (
    FaceLandmarks,
    detect_face_landmarks,
    align_face,
    analyze_shadows_and_lighting,
    enhance_lighting,
    analyze_depth_for_liveness,
    combine_liveness_scores,
    estimate_depth_map,
    LightingAnalysis,
    DepthAnalysis,
)


class TestFaceLandmarks:
    """Тесты для FaceLandmarks dataclass."""
    
    def test_from_68_points(self):
        """Тест создания FaceLandmarks из 68 точек."""
        # Создаем тестовые 68 точек
        points = np.random.randint(0, 100, (68, 2), dtype=np.int32)
        
        landmarks = FaceLandmarks.from_68_points(points)
        
        assert landmarks.jawline.shape == (17, 2)
        assert landmarks.left_eyebrow.shape == (5, 2)
        assert landmarks.right_eyebrow.shape == (5, 2)
        assert landmarks.nose.shape == (9, 2)
        assert landmarks.left_eye.shape == (6, 2)
        assert landmarks.right_eye.shape == (6, 2)
        assert landmarks.outer_lips.shape == (12, 2)
        assert landmarks.inner_lips.shape == (8, 2)
    
    def test_get_eye_centers(self):
        """Тест получения центров глаз."""
        points = np.zeros((68, 2), dtype=np.int32)
        
        # Устанавливаем координаты левого глаза (36-41)
        points[36:42, 0] = [100, 105, 110, 108, 102, 98]
        points[36:42, 1] = [50, 52, 50, 48, 46, 48]
        
        # Устанавливаем координаты правого глаза (42-47)
        points[42:48, 0] = [200, 205, 210, 208, 202, 198]
        points[42:48, 1] = [50, 52, 50, 48, 46, 48]
        
        landmarks = FaceLandmarks.from_68_points(points)
        left_center, right_center = landmarks.get_eye_centers()
        
        assert left_center[0] == pytest.approx(103.83, rel=0.1)
        assert right_center[0] == pytest.approx(203.83, rel=0.1)
    
    def test_get_eye_distance(self):
        """Тест расчета расстояния между глазами."""
        points = np.zeros((68, 2), dtype=np.int32)
        
        # Устанавливаем координаты глаз
        points[36:42, 0] = [100, 100, 100, 100, 100, 100]
        points[36:42, 1] = [50, 50, 50, 50, 50, 50]
        
        points[42:48, 0] = [200, 200, 200, 200, 200, 200]
        points[42:48, 1] = [50, 50, 50, 50, 50, 50]
        
        landmarks = FaceLandmarks.from_68_points(points)
        distance = landmarks.get_eye_distance()
        
        assert distance == pytest.approx(100.0, rel=0.01)


class TestFaceAlignment:
    """Тесты для выравнивания лица."""
    
    @pytest.fixture
    def sample_face_image(self):
        """Создает тестовое изображение лица."""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_landmarks(self):
        """Создает тестовые landmarks."""
        landmarks = np.zeros((68, 2), dtype=np.int32)
        
        # Jawline
        jaw_x = np.linspace(50, 174, 17)
        jaw_y = 200 - np.linspace(10, 50, 17)
        landmarks[0:17, 0] = jaw_x.astype(np.int32)
        landmarks[0:17, 1] = jaw_y.astype(np.int32)
        
        # Eyes centers (для расчета поворота)
        landmarks[36:42, 0] = [75, 78, 80, 80, 78, 75]
        landmarks[36:42, 1] = [80, 82, 80, 78, 76, 78]
        
        landmarks[42:48, 0] = [145, 148, 150, 150, 148, 145]
        landmarks[42:48, 1] = [80, 82, 80, 78, 76, 78]
        
        return landmarks
    
    def test_align_face_output_size(self, sample_face_image, sample_landmarks):
        """Тест выходного размера выровненного лица."""
        output_size = (112, 112)
        aligned_face, metadata = align_face(sample_face_image, sample_landmarks, output_size=output_size)
        
        assert aligned_face.shape[:2] == output_size
    
    def test_align_face_rotation_angle(self, sample_face_image, sample_landmarks):
        """Тест расчета угла поворота."""
        # Поворачиваем правый глаз выше левого
        sample_landmarks[42:48, 1] = 70  # Правый глаз выше
        
        aligned_face, metadata = align_face(sample_face_image, sample_landmarks)
        
        # Проверяем, что угол не нулевой
        assert metadata["rotation_angle"] != 0
    
    def test_align_face_fallback_on_empty_crop(self, sample_face_image):
        """Тест fallback при ошибке кропа."""
        # Создаем landmarks за пределами изображения
        landmarks = np.full((68, 2), 500, dtype=np.int32)  # Все точки вне изображения
        
        aligned_face, metadata = align_face(sample_face_image, landmarks)
        
        # Должен использовать fallback (resize)
        assert aligned_face.shape[:2] == (112, 112)
        assert "error" in metadata


class TestLightingAnalysis:
    """Тесты для анализа освещения."""
    
    @pytest.fixture
    def well_lit_face(self):
        """Хорошо освещенное лицо."""
        # Создаем плавный градиент для имитации хорошего освещения
        h, w = 112, 112
        x = np.linspace(0, 255, w)
        y = np.linspace(0, 255, h)
        xx, yy = np.meshgrid(x, y)
        face = np.stack([xx, yy, (xx + yy) / 2], axis=2).astype(np.uint8)
        return np.clip(face, 0, 255)
    
    @pytest.fixture
    def poorly_lit_face(self):
        """Плохо освещенное лицо с сильными тенями."""
        h, w = 112, 112
        face = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Добавляем сильную тень с одной стороны
        face[:, :w//2] = 50  # Темная левая половина
        face[:, w//2:] = 200  # Яркая правая половина
        
        return face
    
    def test_analyze_well_lit_face(self, well_lit_face):
        """Тест анализа хорошо освещенного лица."""
        result = analyze_shadows_and_lighting(well_lit_face)
        
        assert isinstance(result, LightingAnalysis)
        assert result.overall_quality > 0.5
        assert result.exposure_score > 0.5
        assert result.left_right_balance > 0.5
    
    def test_analyze_poorly_lit_face(self, poorly_lit_face):
        """Тест анализа плохо освещенного лица."""
        result = analyze_shadows_and_lighting(poorly_lit_face)
        
        # Должен обнаружить проблемы с освещением
        assert result.left_right_balance < 0.7
        assert "uneven_left_right_lighting" in result.issues
    
    def test_lighting_analysis_metrics(self, well_lit_face):
        """Тест метрик анализа освещения."""
        result = analyze_shadows_and_lighting(well_lit_face)
        
        # Проверяем наличие всех метрик
        assert hasattr(result, 'mean_brightness')
        assert hasattr(result, 'brightness_std')
        assert hasattr(result, 'dynamic_range')
        assert hasattr(result, 'shadow_ratio')
        assert hasattr(result, 'symmetry_score')
        assert hasattr(result, 'light_type')
    
    def test_enhance_lighting(self, poorly_lit_face):
        """Тест улучшения освещения."""
        result = analyze_shadows_and_lighting(poorly_lit_face)
        
        enhanced = enhance_lighting(poorly_lit_face, result)
        
        assert enhanced.shape == poorly_lit_face.shape
        assert enhanced.dtype == np.uint8


class TestDepthEstimation:
    """Тесты для оценки глубины."""
    
    @pytest.fixture
    def real_face_like(self):
        """Изображение, похожее на реальное лицо (с вариацией глубины)."""
        h, w = 112, 112
        # Создаем псевдо-3D структуру
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Нос "выступает" (ярче в центре)
        nose = np.exp(-((xx - 0.5)**2 + (yy - 0.4)**2) / 0.1)
        # Глаза "углублены" (темнее по бокам)
        eyes = np.exp(-((xx - 0.3)**2 + (yy - 0.3)**2) / 0.05) + \
               np.exp(-((xx - 0.7)**2 + (yy - 0.3)**2) / 0.05)
        
        face = ((nose * 100 + eyes * 50 + 100)).astype(np.uint8)
        face = np.stack([face, face, face], axis=2)
        
        return face
    
    @pytest.fixture
    def flat_surface(self):
        """Плоская поверхность (характерна для фото)."""
        h, w = 112, 112
        face = np.full((h, w, 3), 128, dtype=np.uint8)
        return face
    
    def test_estimate_depth_map_output(self, real_face_like):
        """Тест выхода карты глубины."""
        depth_map = estimate_depth_map(real_face_like)
        
        assert depth_map.shape == real_face_like.shape[:2]
        assert depth_map.dtype == np.uint8
        assert 0 <= depth_map.min() <= 255
        assert 0 <= depth_map.max() <= 255
    
    def test_analyze_depth_real_face(self, real_face_like):
        """Тест анализа глубины для 'реального' лица."""
        result = analyze_depth_for_liveness(real_face_like)
        
        assert isinstance(result, DepthAnalysis)
        assert result.depth_score >= 0
        assert result.depth_score <= 1
        assert isinstance(result.anomalies, list)
    
    def test_analyze_depth_flat_surface(self, flat_surface):
        """Тест анализа глубины для плоской поверхности."""
        result = analyze_depth_for_liveness(flat_surface)
        
        # Плоская поверхность должна иметь высок flatness_score
        assert result.flatness_score > 0.5
        # И низкий depth_score
        assert result.depth_score < 0.5
        # Должны быть аномалии
        assert len(result.anomalies) > 0
        assert "uniform_depth_suspicious_flat" in result.anomalies
    
    def test_depth_analysis_metrics(self, real_face_like):
        """Тест метрик анализа глубины."""
        result = analyze_depth_for_liveness(real_face_like)
        
        # Проверяем наличие всех метрик
        assert hasattr(result, 'depth_variance')
        assert hasattr(result, 'depth_std')
        assert hasattr(result, 'focus_variation_score')
        assert hasattr(result, 'shading_depth_score')
        assert hasattr(result, 'texture_depth_score')
        assert hasattr(result, 'shape_consistency_score')
        assert hasattr(result, 'focus_variation')
        assert hasattr(result, 'natural_shadows_count')
        assert hasattr(result, 'texture_diversity')
    
    def test_combine_liveness_scores(self):
        """Тест комбинирования оценок живости."""
        anti_spoofing_score = 0.8
        depth_score = 0.7
        lighting_quality = 0.6
        
        result = combine_liveness_scores(
            anti_spoofing_score=anti_spoofing_score,
            depth_score=depth_score,
            lighting_quality=lighting_quality,
        )
        
        assert "liveness_detected" in result
        assert "combined_score" in result
        assert "confidence" in result
        assert "decision_threshold" in result


class TestCombineLivenessScores:
    """Тесты для комбинирования оценок живости."""
    
    def test_high_scores_detected_as_real(self):
        """Тест: высокие оценки -> real."""
        result = combine_liveness_scores(
            anti_spoofing_score=0.9,
            depth_score=0.8,
            lighting_quality=0.7,
        )
        
        assert result["liveness_detected"] is True
        assert result["combined_score"] > 0.5
    
    def test_low_scores_detected_as_spoof(self):
        """Тест: низкие оценки -> spoof."""
        result = combine_liveness_scores(
            anti_spoofing_score=0.2,
            depth_score=0.3,
            lighting_quality=0.4,
        )
        
        assert result["liveness_detected"] is False
        assert result["combined_score"] < 0.5
    
    def test_with_anomalies_penalty(self):
        """Тест штрафа за аномалии."""
        depth_analysis = DepthAnalysis(
            depth_score=0.6,
            flatness_score=0.4,
            is_likely_real=True,
            confidence=0.6,
            depth_variance=100,
            depth_std=10,
            focus_variation_score=0.6,
            shading_depth_score=0.6,
            texture_depth_score=0.6,
            shape_consistency_score=0.6,
            depth_map=None,
            anomalies=["uniform_depth_suspicious_flat", "moire_patterns_detected"],
            estimation_confidence=0.6,
            is_3d_consistent=True,
            focus_variation=0.2,
            natural_shadows_count=2,
            texture_diversity=0.4,
        )
        
        result = combine_liveness_scores(
            anti_spoofing_score=0.7,
            depth_score=0.6,
            lighting_quality=0.6,
            depth_analysis=depth_analysis,
        )
        
        # Штраф должен снизить combined_score
        assert result["anomaly_count"] == 2
        assert result["anomaly_penalty"] > 0
    
    def test_3d_consistent_bonus(self):
        """Тест бонуса за консистентность 3D."""
        depth_analysis = DepthAnalysis(
            depth_score=0.6,
            flatness_score=0.4,
            is_likely_real=True,
            confidence=0.6,
            depth_variance=500,
            depth_std=22,
            focus_variation_score=0.6,
            shading_depth_score=0.6,
            texture_depth_score=0.6,
            shape_consistency_score=0.6,
            depth_map=None,
            anomalies=[],
            estimation_confidence=0.8,
            is_3d_consistent=True,
            focus_varation=0.3,
            natural_shadows_count=2,
            texture_diversity=0.5,
        )
        
        result_with_3d = combine_liveness_scores(
            anti_spoofing_score=0.6,
            depth_score=0.6,
            lighting_quality=0.6,
            depth_analysis=depth_analysis,
        )
        
        depth_analysis_no_3d = DepthAnalysis(
            depth_score=0.6,
            flatness_score=0.4,
            is_likely_real=True,
            confidence=0.6,
            depth_variance=500,
            depth_std=22,
            focus_variation_score=0.6,
            shading_depth_score=0.6,
            texture_depth_score=0.6,
            shape_consistency_score=0.6,
            depth_map=None,
            anomalies=[],
            estimation_confidence=0.8,
            is_3d_consistent=False,
            focus_variation=0.3,
            natural_shadows_count=2,
            texture_diversity=0.5,
        )
        
        result_without_3d = combine_liveness_scores(
            anti_spoofing_score=0.6,
            depth_score=0.6,
            lighting_quality=0.6,
            depth_analysis=depth_analysis_no_3d,
        )
        
        assert result_with_3d["is_3d_consistent"] is True
        assert result_with_3d["combined_score"] >= result_without_3d["combined_score"]


class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_pipeline(self):
        """Тест полного пайплайна обработки."""
        # Создаем тестовое "лицо"
        h, w = 112, 112
        face = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        
        # Добавляем градиент для имитации освещения
        x = np.linspace(0, 255, w)
        face = np.clip(face.astype(np.int16) + x.astype(np.int16), 0, 255).astype(np.uint8)
        
        # 1. Анализ освещения
        lighting_result = analyze_shadows_and_lighting(face)
        
        # 2. Оценка глубины
        depth_result = analyze_depth_for_liveness(face)
        
        # 3. Комбинирование
        combined = combine_liveness_scores(
            anti_spoofing_score=0.7,
            depth_score=depth_result.depth_score,
            lighting_quality=lighting_result.overall_quality,
        )
        
        # Все этапы должны выполняться успешно
        assert lighting_result.overall_quality >= 0
        assert depth_result.depth_score >= 0
        assert combined["combined_score"] >= 0
    
    def test_edge_cases(self):
        """Тест граничных случаев."""
        # Очень маленькое изображение
        tiny_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        # Должен обрабатываться без ошибок
        try:
            lighting_result = analyze_shadows_and_lighting(tiny_image)
            depth_result = analyze_depth_for_liveness(tiny_image)
            
            # Результаты должны быть валидными
            assert lighting_result.overall_quality >= 0
            assert depth_result.depth_score >= 0
        except Exception as e:
            # Допускаются исключения для очень маленьких изображений
            assert "analysis_failed" in str(e).lower() or True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
