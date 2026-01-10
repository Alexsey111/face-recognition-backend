"""
Утилиты для выравнивания лица, анализа освещения и 3D depth estimation.
Используются для улучшения качества эмбеддингов и проверки живости.

Улучшения:
- Enhanced Face Alignment с 68-point landmarks и улучшенным кропом
- Advanced Shadow Analysis с детекцией конкретных типов теней
- Multi-scale Lighting Analysis для разных условий освещения
- 3D Depth Estimation с depth-from-defocus и shading-based depth
"""

from __future__ import annotations

import io
import base64
import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

from ..utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enhanced Face Alignment with 68-point Landmarks
# =============================================================================

@dataclass
class FaceLandmarks:
    """Лицевые landmarks для выравнивания лица."""
    # 68-point landmark indices (dlib style)
    jawline: np.ndarray  # 0-16
    left_eyebrow: np.ndarray  # 17-21
    right_eyebrow: np.ndarray  # 22-26
    nose: np.ndarray  # 27-35
    left_eye: np.ndarray  # 36-41
    right_eye: np.ndarray  # 42-47
    outer_lips: np.ndarray  # 48-59
    inner_lips: np.ndarray  # 60-67
    
    @classmethod
    def from_68_points(cls, points: np.ndarray) -> "FaceLandmarks":
        """Создание из массива 68 точек."""
        if len(points) != 68:
            raise ValueError(f"Expected 68 points, got {len(points)}")
        return cls(
            jawline=points[0:17],
            left_eyebrow=points[17:22],
            right_eyebrow=points[22:27],
            nose=points[27:36],
            left_eye=points[36:42],
            right_eye=points[42:48],
            outer_lips=points[48:60],
            inner_lips=points[60:68],
        )
    
    def get_eye_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Получение центров глаз."""
        left_center = np.mean(self.left_eye, axis=0)
        right_center = np.mean(self.right_eye, axis=0)
        return left_center, right_center
    
    def get_nose_tip(self) -> np.ndarray:
        """Получение кончика носа."""
        return self.nose[6]
    
    def get_face_center(self) -> np.ndarray:
        """Получение центра лица."""
        return np.mean(self.jawline, axis=0)
    
    def get_eye_distance(self) -> float:
        """Расстояние между глазами."""
        left_center, right_center = self.get_eye_centers()
        return np.linalg.norm(right_center - left_center)
    
    def get_face_width(self) -> float:
        """Ширина лица по jawline."""
        return np.linalg.norm(self.jawline[0] - self.jawline[-1])
    
    def get_face_height(self) -> float:
        """Высота лица (от подбородка до центра лба)."""
        chin = np.mean(self.jawline[0:3], axis=0)
        brow_center = np.mean(self.left_eyebrow[-1] + self.right_eyebrow[0]) / 2
        # Estimate forehead top
        forehead_top = self.jawline[-1] + (self.jawline[-1] - self.jawline[0]) * 0.3
        return np.linalg.norm(forehead_top - chin)
    
    def get_face_ratio(self) -> float:
        """Соотношение сторон лица (width/height)."""
        return self.get_face_width() / (self.get_face_height() + 1e-6)


def detect_face_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Детекция 68 лицевых landmarks с использованием MTCNN + fallback.
    
    Returns:
        Массив 68x2 с координатами landmarks или None если не найдены
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Попытка использовать dlib если доступен
        try:
            import dlib
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(dlib.shape_predictor_model_path())
            
            faces = detector(gray)
            if faces:
                face = faces[0]
                shape = predictor(gray, face)
                points = np.array([[p.x, p.y] for p in shape.parts()])
                return points
        except (ImportError, AttributeError, OSError):
            pass
        
        # Используем MTCNN из facenet_pytorch
        try:
            from facenet_pytorch import MTCNN
            mtcnn = MTCNN(image_size=160, device='cpu')
            
            # MTCNN возвращает 5 точек: left_eye, right_eye, nose, left_mouth, right_mouth
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
            
            if boxes is not None and len(boxes) > 0:
                # Расширяем 5 точек до 68 используя эвристики
                lm = landmarks[0]  # Берем первую детекцию
                return _expand_mtcnn_to_68_points(image, lm)
                
        except ImportError:
            pass
        
        # Fallback: Используем каскады + эвристики
        return _heuristic_landmarks(image)
        
    except Exception as e:
        logger.warning(f"Landmark detection failed: {e}")
        return None


def _expand_mtcnn_to_68_points(image: np.ndarray, mtcnn_points: np.ndarray) -> np.ndarray:
    """
    Расширение 5 точек MTCNN до 68 точек dlib-стиля.
    
    MTCNN points: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    h, w = image.shape[:2]
    
    if mtcnn_points is None or len(mtcnn_points) != 5:
        return _heuristic_landmarks(image)
    
    left_eye = mtcnn_points[0]
    right_eye = mtcnn_points[1]
    nose = mtcnn_points[2]
    left_mouth = mtcnn_points[3]
    right_mouth = mtcnn_points[4]
    
    # Eye centers
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    
    eye_distance = np.linalg.norm(right_eye - left_eye)
    eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    
    # Face dimensions estimate
    face_width = eye_distance * 2.5
    face_height = face_width * 1.3
    
    # Jawline - 17 points
    jaw_width = face_width * 0.8
    jaw_start_x = eye_center_x - jaw_width / 2
    jaw_y_base = nose[1] + eye_distance * 0.8
    
    jawline = np.zeros((17, 2))
    for i in range(17):
        angle = -np.pi + (i / 16) * np.pi  # Arc from left to right
        jawline[i, 0] = jaw_start_x + (jaw_width / 2) * (1 + np.cos(angle))
        jawline[i, 1] = jaw_y_base + (jaw_width / 3) * np.sin(angle) * 0.5
    
    # Eyebrows - 10 points (5 each)
    brow_y = left_eye[1] - eye_distance * 0.4
    brow_length = eye_distance * 0.7
    brow_width = eye_distance * 0.25
    
    left_eyebrow = np.zeros((5, 2))
    right_eyebrow = np.zeros((5, 2))
    
    for i in range(5):
        t = i / 4.0
        # Left eyebrow
        left_eyebrow[i, 0] = left_eye[0] - brow_length/2 + t * brow_length
        left_eyebrow[i, 1] = brow_y - brow_width * np.abs(t - 0.5)
        # Right eyebrow
        right_eyebrow[i, 0] = right_eye[0] - brow_length/2 + t * brow_length
        right_eyebrow[i, 1] = brow_y - brow_width * np.abs(t - 0.5)
    
    # Nose - 9 points
    nose_points = np.zeros((9, 2))
    nose_tip_y = nose[1]
    nose_top_y = left_eye[1] * 0.8 + nose_tip_y * 0.2
    nose_width_base = eye_distance * 0.4
    nose_width_tip = eye_distance * 0.3
    
    # Nose bridge
    for i in range(4):
        t = i / 3.0
        nose_points[i, 0] = nose[0]
        nose_points[i, 1] = nose_top_y + t * (nose_tip_y - nose_top_y)
    
    # Nose base (3 points)
    nose_points[4, 0] = nose[0] - nose_width_base / 2
    nose_points[4, 1] = nose_tip_y + eye_distance * 0.15
    nose_points[5, 0] = nose[0]
    nose_points[5, 1] = nose_tip_y + eye_distance * 0.2
    nose_points[6, 0] = nose[0] + nose_width_base / 2
    nose_points[6, 1] = nose_tip_y + eye_distance * 0.15
    
    # Nostrils (2 points)
    nose_points[7, 0] = nose[0] - nose_width_tip / 2
    nose_points[7, 1] = nose_tip_y + eye_distance * 0.05
    nose_points[8, 0] = nose[0] + nose_width_tip / 2
    nose_points[8, 1] = nose_tip_y + eye_distance * 0.05
    
    # Eyes - 12 points (6 each) - approximated as ellipses
    eye_radius = eye_distance * 0.15
    
    left_eye_points = np.zeros((6, 2))
    right_eye_points = np.zeros((6, 2))
    
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
        left_eye_points[i, 0] = left_eye[0] + eye_radius * np.cos(angle)
        left_eye_points[i, 1] = left_eye[1] + eye_radius * np.sin(angle) * 0.6
        right_eye_points[i, 0] = right_eye[0] + eye_radius * np.cos(angle)
        right_eye_points[i, 1] = right_eye[1] + eye_radius * np.sin(angle) * 0.6
    
    # Lips - 20 points (12 outer + 8 inner)
    lip_center_y = (left_mouth[1] + right_mouth[1]) / 2
    lip_center_x = (left_mouth[0] + right_mouth[0]) / 2
    lip_width = np.linalg.norm(right_mouth - left_mouth)
    lip_height = lip_width * 0.4
    
    outer_lips = np.zeros((12, 2))
    inner_lips = np.zeros((8, 2))
    
    # Outer lip contour
    for i in range(6):
        angle = np.pi + (i / 5.0) * np.pi
        outer_lips[i, 0] = lip_center_x + (lip_width / 2) * np.cos(angle)
        outer_lips[i, 1] = lip_center_y + (lip_height / 2) * np.sin(angle)
    
    for i in range(6):
        angle = 0 + (i / 5.0) * np.pi
        outer_lips[i + 6, 0] = lip_center_x + (lip_width / 2) * np.cos(angle)
        outer_lips[i + 6, 1] = lip_center_y + (lip_height / 2) * np.sin(angle)
    
    # Inner lip contour
    inner_lip_scale = 0.5
    for i in range(4):
        angle = np.pi + (i / 3.0) * np.pi
        inner_lips[i, 0] = lip_center_x + (lip_width / 2 * inner_lip_scale) * np.cos(angle)
        inner_lips[i, 1] = lip_center_y + (lip_height / 2 * inner_lip_scale) * np.sin(angle)
    
    for i in range(4):
        angle = 0 + (i / 3.0) * np.pi
        inner_lips[i + 4, 0] = lip_center_x + (lip_width / 2 * inner_lip_scale) * np.cos(angle)
        inner_lips[i + 4, 1] = lip_center_y + (lip_height / 2 * inner_lip_scale) * np.sin(angle)
    
    # Combine all points
    all_points = np.vstack([
        jawline, left_eyebrow, right_eyebrow, nose_points,
        left_eye_points, right_eye_points, outer_lips, inner_lips
    ])
    
    # Clip to image bounds
    all_points[:, 0] = np.clip(all_points[:, 0], 0, w - 1)
    all_points[:, 1] = np.clip(all_points[:, 1], 0, h - 1)
    
    return all_points.astype(np.int32)


def _heuristic_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Эвристическая генерация landmarks на основе каскадного детектора.
    Используется когда dlib и MTCNN недоступны.
    """
    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Детекция лица через каскады
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return None
        
        x, y, fw, fh = faces[0]
        
        # Эвристические пропорции лица
        landmarks = np.zeros((68, 2), dtype=np.int32)
        
        # Jawline
        jaw_x = np.linspace(x + int(0.15*fw), x + int(0.85*fw), 17)
        jaw_y = y + fh - np.linspace(int(0.05*fh), int(0.25*fh), 17)
        landmarks[0:17, 0] = jaw_x.astype(np.int32)
        landmarks[0:17, 1] = jaw_y.astype(np.int32)
        
        # Eyebrows
        landmarks[17:22, 0] = np.linspace(x + int(0.20*fw), x + int(0.45*fw), 5)
        landmarks[17:22, 1] = y + int(0.20*fh) + np.linspace(0, -int(0.05*fh), 5)
        landmarks[22:27, 0] = np.linspace(x + int(0.55*fw), x + int(0.80*fw), 5)
        landmarks[22:27, 1] = y + int(0.20*fh) + np.linspace(0, -int(0.05*fh), 5)
        
        # Nose
        landmarks[27:36, 0] = x + int(0.50*fw) + np.linspace(-int(0.10*fw), int(0.10*fw), 9)
        landmarks[27:36, 1] = y + int(0.35*fh) + np.linspace(0, int(0.15*fh), 9)
        
        # Eyes
        left_eye_center = (x + int(0.32*fw), y + int(0.30*fh))
        right_eye_center = (x + int(0.68*fw), y + int(0.30*fh))
        
        eye_radius = int(0.05*fw)
        for i, angle in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
            landmarks[36+i, 0] = int(left_eye_center[0] + eye_radius * np.cos(angle))
            landmarks[36+i, 1] = int(left_eye_center[1] + eye_radius * np.sin(angle))
        
        for i, angle in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
            landmarks[42+i, 0] = int(right_eye_center[0] + eye_radius * np.cos(angle))
            landmarks[42+i, 1] = int(right_eye_center[1] + eye_radius * np.sin(angle))
        
        # Lips
        lip_center_y = y + int(0.55*fh)
        lip_width = int(0.20*fw)
        
        landmarks[48:60, 0] = np.concatenate([
            np.linspace(x + int(0.35*fw), x + int(0.65*fw), 6),
            np.linspace(x + int(0.65*fw), x + int(0.35*fw), 6)
        ])
        landmarks[48:60, 1] = lip_center_y + np.concatenate([
            np.zeros(6),
            np.ones(6) * int(0.02*fh)
        ])
        
        landmarks[60:68, 0] = np.concatenate([
            np.linspace(x + int(0.40*fw), x + int(0.60*fw), 4),
            np.linspace(x + int(0.60*fw), x + int(0.40*fw), 4)
        ])
        landmarks[60:68, 1] = lip_center_y + np.concatenate([
            np.zeros(4),
            np.ones(4) * int(0.01*fh)
        ])
        
        return landmarks.astype(np.int32)
        
    except Exception as e:
        logger.warning(f"Heuristic landmarks failed: {e}")
        return None


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
    scale_factor: float = 1.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Выравнивание лица по landmarks с улучшенным алгоритмом.
    
    Алгоритм:
    1. Вычисление угла поворота по линии между глазами
    2. Применение аффинного преобразования
    3. Извлечение области лица с оптимальным кропом
    4. Ресайз к целевому размеру
    
    Args:
        image: Входное изображение (RGB)
        landmarks: 68 лицевых точек
        output_size: Выходной размер (width, height)
        scale_factor: Дополнительный масштаб для кропа
        
    Returns:
        Tuple of (aligned_image, metadata)
    """
    try:
        # Создаём FaceLandmarks для удобства
        face_landmarks = FaceLandmarks.from_68_points(landmarks)
        
        # Центры глаз
        left_eye = np.mean(face_landmarks.left_eye, axis=0)
        right_eye = np.mean(face_landmarks.right_eye, axis=0)
        
        # Вычисляем угол поворота
        eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        rotation_angle = -eye_angle * 180.0 / np.pi
        
        # Центр изображения
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Матрица вращения
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Поворачиваем изображение
        aligned_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # Поворачиваем landmarks для определения области кропа
        landmarks_homo = np.hstack([landmarks, np.ones((68, 1))])
        landmarks_rotated = (rotation_matrix @ landmarks_homo.T).T
        landmarks_rotated = landmarks_rotated[:, :2]
        
        # Вычисляем bounding box по landmarks с отступом
        x_min, y_min = np.min(landmarks_rotated, axis=0)
        x_max, y_max = np.max(landmarks_rotated, axis=0)
        
        # Расширяем bounding box
        face_width = x_max - x_min
        face_height = y_max - y_min
        
        margin_x = face_width * 0.25 * scale_factor
        margin_y = face_height * 0.35 * scale_factor  # Больше сверху для лба
        
        x_min = max(0, int(x_min - margin_x))
        y_min = max(0, int(y_min - margin_y))
        x_max = min(image.shape[1], int(x_max + margin_x))
        y_max = min(image.shape[0], int(y_max + margin_y * 0.5))
        
        # Обрезаем лицо
        cropped_face = aligned_image[y_min:y_max, x_min:x_max]
        
        if cropped_face.size == 0:
            # Fallback: возврат центрированного resized изображения
            resized = cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)
            return resized, {"error": "Empty crop, used fallback"}
        
        # Изменяем размер
        aligned_face = cv2.resize(cropped_face, output_size, interpolation=cv2.INTER_CUBIC)
        
        # Метаданные для отладки
        metadata = {
            "rotation_angle": float(rotation_angle),
            "eye_distance": float(np.linalg.norm(right_eye - left_eye)),
            "face_width": float(face_width),
            "face_height": float(face_height),
            "crop_box": [x_min, y_min, x_max, y_max],
            "face_ratio": float(face_landmarks.get_face_ratio()),
        }
        
        return aligned_face, metadata
        
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}")
        resized = cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)
        return resized, {"error": str(e)}


# =============================================================================
# Enhanced Shadow and Lighting Analysis
# =============================================================================

@dataclass
class LightingAnalysis:
    """Результат расширенного анализа освещения и теней."""
    # Overall lighting quality [0, 1]
    overall_quality: float
    # Component scores
    exposure_score: float
    shadow_evenness: float
    left_right_balance: float
    vertical_balance: float
    contrast_score: float
    shadow_naturalness: float
    # Detailed metrics
    mean_brightness: float
    brightness_std: float
    dynamic_range: float
    # Shadow analysis
    shadow_regions: List[Dict[str, Any]]
    shadow_ratio: float
    # Light symmetry
    symmetry_score: float
    light_type: str
    # Detected issues
    issues: List[str]
    recommendations: List[str]
    # Enhanced metrics
    highlights_ratio: float
    red_eye_ratio: float
    gradient_smoothness: float
    texture_contrast: float


def analyze_shadows_and_lighting(face_region: np.ndarray) -> LightingAnalysis:
    """
    Комплексный анализ теней и освещения на лице с улучшенными методами.
    
    Detects:
    - Harsh shadows (backlight, side lighting)
    - Uneven illumination (left/right imbalance)
    - Over/underexposure
    - Specular highlights (flash artifacts)
    - Red-eye (indicates flash photography)
    - Shadow patterns characteristic of 3D vs flat surfaces
    
    Returns:
        LightingAnalysis с оценками и рекомендациями
    """
    issues = []
    recommendations = []
    
    try:
        # Конвертация в grayscale для анализа
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Разделяем на регионы
        mid_x = w // 2
        mid_y = h // 2
        
        # =================================================================
        # 1. EXPOSURE ANALYSIS
        # =================================================================
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Гистограммный анализ
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        
        # Процентили
        cumulative = np.cumsum(hist)
        p10 = np.searchsorted(cumulative, 0.1)
        p90 = np.searchsorted(cumulative, 0.9)
        p25 = np.searchsorted(cumulative, 0.25)
        p75 = np.searchsorted(cumulative, 0.75)
        
        dynamic_range = p90 - p10
        
        # Оценка экспозиции
        if mean_brightness < 50:
            exposure_score = mean_brightness / 50.0
            issues.append("underexposed")
            recommendations.append("increase_illumination")
        elif mean_brightness > 220:
            exposure_score = 1.0 - (mean_brightness - 220) / 35.0
            issues.append("overexposed")
            recommendations.append("reduce_direct_light")
        else:
            # Optimal range 80-180
            optimal_center = 130
            exposure_score = 1.0 - abs(mean_brightness - optimal_center) / 80.0
        
        exposure_score = max(0.0, min(1.0, exposure_score))
        
        # =================================================================
        # 2. CONTRAST ANALYSIS
        # =================================================================
        contrast_score = min(brightness_std / 100.0, 1.0)
        if contrast_score < 0.3:
            issues.append("low_contrast")
            recommendations.append("improve_contrast_or_avoid_flat_lighting")
        
        # =================================================================
        # 3. LEFT/RIGHT BALANCE
        # =================================================================
        left_half = gray[:, :mid_x]
        right_half = gray[:, mid_x:]
        
        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)
        left_right_ratio = min(left_mean, right_mean) / max(left_mean, right_mean)
        left_right_balance = left_right_ratio
        
        if left_right_balance < 0.7:
            issues.append("uneven_left_right_lighting")
            recommendations.append("move_to_evenly_lit_area")
        
        # =================================================================
        # 4. VERTICAL GRADIENT (top/bottom balance)
        # =================================================================
        top_half = gray[:h//3, :]
        bottom_half = gray[2*h//3:, :]
        
        top_mean = np.mean(top_half)
        bottom_mean = np.mean(bottom_half)
        vertical_ratio = min(top_mean, bottom_mean) / max(top_mean, bottom_mean)
        vertical_balance = vertical_ratio
        
        if vertical_balance < 0.6:
            issues.append("strong_vertical_gradient")
            recommendations.append("avoid_backlight_or_under_chin_lighting")
        
        # =================================================================
        # 5. SHADOW ANALYSIS (Enhanced)
        # =================================================================
        
        # Адаптивное пороговое значение для детекции теней
        block_size = 51
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 2
        )
        
        # Морфологическая обработка
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Контуры теней
        contours, _ = cv2.findContours(
            shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Анализ регионов теней
        shadow_regions = []
        total_shadow_pixels = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if area > h * w * 0.005:  # Значимые регионы
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    bbox = cv2.boundingRect(c)
                    
                    shadow_regions.append({
                        "area": float(area),
                        "center": (cx, cy),
                        "bbox": bbox,
                        "aspect_ratio": bbox[2] / max(bbox[3], 1),
                    })
                    total_shadow_pixels += area
        
        shadow_ratio = total_shadow_pixels / (h * w)
        
        # Детекция характерных теней лица
        has_nose_shadow = False
        has_eye_shadows = False
        has_chin_shadow = False
        
        nose_region_y = (int(h * 0.35), int(h * 0.65))
        eye_region_y = (int(h * 0.25), int(h * 0.45))
        chin_region_y = (int(h * 0.75), h)
        
        for shadow in shadow_regions:
            cy = shadow["center"][1]
            
            if nose_region_y[0] < cy < nose_region_y[1]:
                if shadow["area"] > h * w * 0.02:
                    has_nose_shadow = True
            
            if eye_region_y[0] < cy < eye_region_y[1]:
                if shadow["area"] > h * w * 0.01:
                    has_eye_shadows = True
            
            if chin_region_y[0] < cy < chin_region_y[1]:
                if shadow["area"] > h * w * 0.015:
                    has_chin_shadow = True
        
        natural_shadow_count = sum([has_nose_shadow, has_eye_shadows, has_chin_shadow])
        
        # Оценка естественности теней
        # Живое лицо имеет тени в характерных местах
        if natural_shadow_count >= 2:
            shadow_naturalness = 0.8
        elif natural_shadow_count == 1:
            shadow_naturalness = 0.5
        else:
            shadow_naturalness = 0.3
            issues.append("unnatural_shadow_pattern")
            recommendations.append("ensure_natural_lighting_conditions")
        
        # Равномерность распределения теней
        if len(shadow_regions) > 1:
            shadow_positions = [s["center"][0] / w for s in shadow_regions]
            shadow_spread = np.std(shadow_positions)
            shadow_evenness = 1.0 - min(shadow_spread * 2, 1.0)
        else:
            shadow_evenness = 0.5
        
        # =================================================================
        # 6. GRADIENT ANALYSIS
        # =================================================================
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Плавность градиентов (характерно для естественного освещения)
        smooth_ratio = np.sum(gradient_mag < 30) / (h * w)
        gradient_smoothness = smooth_ratio
        
        # Текстурный контраст
        texture_contrast = np.std(gradient_mag) / 50.0
        texture_contrast = min(texture_contrast, 1.0)
        
        # =================================================================
        # 7. FLASH/HIGHLIGHT DETECTION
        # =================================================================
        _, high_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        highlights_ratio = np.sum(high_mask > 0) / (h * w)
        
        if highlights_ratio > 0.05:
            issues.append("specular_highlights_detected")
            recommendations.append("avoid_flash_direct_reflection")
        
        # Red-eye detection
        if len(face_region.shape) == 3:
            r_channel = face_region[:, :, 0]
            g_channel = face_region[:, :, 1]
            b_channel = face_region[:, :, 2]
            
            # Красные глаза: R > G и R > B
            red_eye_mask = (r_channel > 150) & (r_channel > g_channel) & (r_channel > b_channel)
            red_eye_ratio = np.sum(red_eye_mask) / (h * w)
            
            if red_eye_ratio > 0.005:
                issues.append("red_eye_detected")
                recommendations.append("disable_flash_or_use_red_eye_reduction")
        else:
            red_eye_ratio = 0
        
        # =================================================================
        # 8. LIGHT SYMMETRY
        # =================================================================
        brightness_diff = np.abs(left_half.astype(float) - right_half.astype(float))
        avg_brightness_diff = float(np.mean(brightness_diff))
        symmetry_score = 1.0 - (avg_brightness_diff / 255.0)
        
        if left_right_balance > 0.85:
            light_type = "symmetric_front"
        elif left_right_balance > 0.7:
            light_type = "side_light"
        elif left_right_balance > 0.5:
            light_type = "strong_side_light"
        else:
            light_type = "mixed_uneven"
        
        # =================================================================
        # 9. OVERALL QUALITY SCORE
        # =================================================================
        overall_quality = (
            exposure_score * 0.20 +
            contrast_score * 0.15 +
            left_right_balance * 0.15 +
            vertical_balance * 0.10 +
            shadow_naturalness * 0.20 +
            symmetry_score * 0.10 +
            (1 - min(highlights_ratio * 5, 1)) * 0.10
        )
        
        overall_quality = max(0.0, min(1.0, overall_quality))
        
        return LightingAnalysis(
            overall_quality=overall_quality,
            exposure_score=exposure_score,
            shadow_evenness=shadow_evenness,
            left_right_balance=left_right_balance,
            vertical_balance=vertical_balance,
            contrast_score=contrast_score,
            shadow_naturalness=shadow_naturalness,
            mean_brightness=mean_brightness,
            brightness_std=brightness_std,
            dynamic_range=dynamic_range,
            shadow_regions=shadow_regions,
            shadow_ratio=shadow_ratio,
            symmetry_score=symmetry_score,
            light_type=light_type,
            issues=issues,
            recommendations=recommendations,
            highlights_ratio=highlights_ratio,
            red_eye_ratio=red_eye_ratio,
            gradient_smoothness=gradient_smoothness,
            texture_contrast=texture_contrast,
        )
        
    except Exception as e:
        logger.warning(f"Lighting analysis failed: {e}")
        return LightingAnalysis(
            overall_quality=0.5,
            exposure_score=0.5,
            shadow_evenness=0.5,
            left_right_balance=0.5,
            vertical_balance=0.5,
            contrast_score=0.5,
            shadow_naturalness=0.5,
            mean_brightness=128,
            brightness_std=0,
            dynamic_range=0,
            shadow_regions=[],
            shadow_ratio=0,
            symmetry_score=0.5,
            light_type="unknown",
            issues=["analysis_failed"],
            recommendations=["retry_with_better_lighting"],
            highlights_ratio=0,
            red_eye_ratio=0,
            gradient_smoothness=0.5,
            texture_contrast=0.5,
        )


def enhance_lighting(
    image: np.ndarray,
    lighting: LightingAnalysis
) -> np.ndarray:
    """
    Улучшение освещения на основе анализа с адаптивными методами.
    """
    try:
        result = image.copy()
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # 1. Gamma correction для экспозиции
        if lighting.overall_quality < 0.6:
            if lighting.mean_brightness < 100:
                #underexposed - увеличиваем яркость
                gamma = 0.7
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                l_channel = cv2.LUT(l_channel, table)
            elif lighting.mean_brightness > 180:
                #overexposed - уменьшаем яркость
                gamma = 1.3
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                l_channel = cv2.LUT(l_channel, table)
        
        # 2. CLAHE для контраста
        if lighting.contrast_score < 0.5:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
        
        # 3. Выравнивание локальной яркости
        if lighting.shadow_evenness < 0.6:
            # Добавляем мягкое освещение в теневые области
            blurred = cv2.GaussianBlur(l_channel, (51, 51), 0)
            l_channel = cv2.addWeighted(l_channel, 1.5, blurred, -0.5, 0)
        
        # 4. Коррекция left/right imbalance
        if lighting.left_right_balance < 0.7:
            h, w = l_channel.shape
            mid_x = w // 2
            
            left_mean = np.mean(l_channel[:, :mid_x])
            right_mean = np.mean(l_channel[:, mid_x:])
            
            if left_mean < right_mean:
                # Левая сторона темнее - осветляем
                correction = (right_mean - left_mean) * 0.5
                l_channel[:, :mid_x] = np.clip(l_channel[:, :mid_x].astype(np.int16) + correction, 0, 255).astype(np.uint8)
            else:
                # Правая сторона темнее - осветляем
                correction = (left_mean - right_mean) * 0.5
                l_channel[:, mid_x:] = np.clip(l_channel[:, mid_x:].astype(np.int16) + correction, 0, 255).astype(np.uint8)
        
        lab[:, :, 0] = l_channel
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        logger.warning(f"Lighting enhancement failed: {e}")
        return image


# =============================================================================
# Enhanced 3D Depth Estimation for Liveness Detection
# =============================================================================

@dataclass
class DepthAnalysis:
    """Результат расширенного анализа глубины для проверки живости."""
    depth_score: float
    flatness_score: float
    is_likely_real: bool
    confidence: float
    
    # Detailed metrics
    depth_variance: float
    depth_std: float
    focus_variation_score: float
    shading_depth_score: float
    texture_depth_score: float
    shape_consistency_score: float
    
    # Depth map
    depth_map: Optional[np.ndarray]
    
    # Anomalies detected
    anomalies: List[str]
    
    # Quality metrics
    estimation_confidence: float
    is_3d_consistent: bool
    
    # Additional features
    focus_variation: float
    natural_shadows_count: int
    texture_diversity: float


def estimate_depth_map(face_region: np.ndarray) -> np.ndarray:
    """
    Оценка карты глубины с использованием множественных методов.
    
    Combines:
    - Depth from defocus (Laplacian variance)
    - Depth from shading (gradient magnitude)
    - Shape cues (intensity-based depth)
    """
    try:
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # 1. Depth from defocus
        blur_small = cv2.GaussianBlur(gray, (3, 3), 0)
        blur_large = cv2.GaussianBlur(gray, (15, 15), 0)
        local_variance = cv2.absdiff(blur_small, blur_large)
        local_variance = cv2.GaussianBlur(local_variance, (5, 5), 0)
        
        # 2. Depth from shading (gradients)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_mag = cv2.GaussianBlur(gradient_mag, (5, 5), 0)
        
        # 3. Depth from intensity (assuming front/top lighting)
        intensity_depth = (255 - gray).astype(np.float32)
        intensity_depth = cv2.GaussianBlur(intensity_depth, (7, 7), 0)
        
        # Combine all depth cues
        depth_map = np.zeros((h, w), dtype=np.float32)
        depth_map += local_variance * 0.3    # Defocus
        depth_map += gradient_mag * 1.5      # Shading gradients
        depth_map += intensity_depth * 0.5   # Intensity
        
        # Normalize
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        depth_map = (depth_map * 255).astype(np.uint8)
        
        # Smooth the final depth map
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
        return depth_map
        
    except Exception as e:
        logger.warning(f"Depth estimation failed: {e}")
        return np.zeros(face_region.shape[:2], dtype=np.uint8)


def analyze_depth_for_liveness(face_region: np.ndarray) -> DepthAnalysis:
    """
    Комплексный анализ глубины для детекции живости.
    
    Evaluates:
    - Focus variation across face regions
    - Shading patterns (3D structure shadows)
    - Texture gradients (surface variation)
    - Shape consistency (face proportions)
    - Depth map statistics
    
    Returns:
        DepthAnalysis с оценкой живости
    """
    anomalies = []
    
    try:
        h, w = face_region.shape[:2]
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        
        # 1. DEPTH MAP ESTIMATION
        depth_map = estimate_depth_map(face_region)
        
        depth_variance = np.var(depth_map)
        depth_std = np.std(depth_map)
        
        # 2. FOCUS VARIATION ANALYSIS
        regions = {
            "forehead": (0, int(h * 0.25), 0, w),
            "left_eye": (int(h * 0.25), int(h * 0.45), 0, int(w * 0.45)),
            "right_eye": (int(h * 0.25), int(h * 0.45), int(w * 0.55), w),
            "nose": (int(h * 0.4), int(h * 0.65), int(w * 0.35), int(w * 0.65)),
            "left_cheek": (int(h * 0.5), int(h * 0.8), 0, int(w * 0.35)),
            "right_cheek": (int(h * 0.5), int(h * 0.8), int(w * 0.65), w),
            "chin": (int(h * 0.8), h, int(w * 0.25), int(w * 0.75)),
        }
        
        region_blur_scores = {}
        for name, (y1, y2, x1, x2) in regions.items():
            region = gray[y1:y2, x1:x2]
            laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
            region_blur_scores[name] = laplacian_var
        
        blur_values = list(region_blur_scores.values())
        blur_mean = np.mean(blur_values)
        blur_std = np.std(blur_values)
        blur_cv = blur_std / (blur_mean + 1e-6)  # Coefficient of variation
        
        # Focus variation score
        focus_variation_score = min(blur_cv / 0.3, 1.0)
        focus_variation = blur_cv
        
        # 3. SHADING DEPTH ANALYSIS
        _, shadow_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_nose_shadow = False
        has_eye_shadows = False
        has_chin_shadow = False
        
        for c in contours:
            area = cv2.contourArea(c)
            if area > h * w * 0.01:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cy = int(M["m01"] / M["m00"])
                    
                    if int(h * 0.4) < cy < int(h * 0.65):
                        has_nose_shadow = True
                    elif int(h * 0.25) < cy < int(h * 0.45):
                        has_eye_shadows = True
                    elif cy > int(h * 0.8):
                        has_chin_shadow = True
        
        natural_shadows_count = sum([has_nose_shadow, has_eye_shadows, has_chin_shadow])
        
        shading_depth_score = (
            min(natural_shadows_count / 3.0, 1.0) * 0.5 +
            min(blur_cv / 0.4, 1.0) * 0.5
        )
        
        # 4. TEXTURE GRADIENT ANALYSIS
        lbp = _compute_lbp(gray)
        lbp_std = np.std(lbp)
        lbp_mean = np.mean(lbp)
        texture_diversity = lbp_std / (lbp_mean + 1e-6)
        
        glcm = _compute_glcm_simple(gray)
        contrast = glcm.get("contrast", 0)
        entropy = glcm.get("entropy", 0)
        
        texture_depth_score = (
            min(texture_diversity / 0.5, 1.0) * 0.4 +
            min(contrast / 500, 1.0) * 0.3 +
            min(entropy / 5, 1.0) * 0.3
        )
        
        # 5. SHAPE CONSISTENCY ANALYSIS
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                axis_a, axis_b = ellipse[1]
                axis_ratio = axis_a / (axis_b + 1e-6)
                
                shape_score = 1.0 - min(abs(axis_ratio - 0.8) / 0.3, 1.0)
            else:
                shape_score = 0.5
        else:
            shape_score = 0.5
        
        shape_consistency_score = shape_score
        
        # 6. FLAT SPOOFING DETECTION
        if depth_std < 15:
            anomalies.append("uniform_depth_suspicious_flat")
            flatness_score = 0.8
        elif depth_std < 25:
            flatness_score = 0.5
        else:
            flatness_score = 0.2
        
        if blur_cv < 0.1:
            anomalies.append("uniform_blur_suspicious_flat")
            flatness_score = max(flatness_score, 0.7)
        
        if natural_shadows_count < 2:
            anomalies.append("missing_natural_shadows")
            flatness_score = max(flatness_score, 0.6)
        
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        center_h, center_w = h // 2, w // 2
        center_region = magnitude_spectrum[
            max(0, center_h-15):min(h, center_h+15),
            max(0, center_w-15):min(w, center_w+15)
        ]
        
        if np.std(center_region) > np.mean(center_region) * 2.5:
            anomalies.append("moire_patterns_detected")
            flatness_score = max(flatness_score, 0.7)
        
        # 7. DEPTH SCORE CALCULATION
        depth_score = (
            focus_variation_score * 0.25 +
            shading_depth_score * 0.30 +
            texture_depth_score * 0.20 +
            shape_consistency_score * 0.25
        )
        
        anomaly_penalty = min(len(anomalies) * 0.15, 0.4)
        depth_score *= (1.0 - anomaly_penalty)
        
        # 8. FINAL DECISION
        is_likely_real = depth_score > 0.45 and len(anomalies) < 2
        
        score_variance = np.std([focus_variation_score, shading_depth_score, 
                                 texture_depth_score, shape_consistency_score])
        estimation_confidence = 1.0 - min(score_variance / 0.3 + len(anomalies) * 0.1, 1.0)
        
        is_3d_consistent = (
            focus_variation_score > 0.4 and
            shading_depth_score > 0.4 and
            natural_shadows_count >= 2
        )
        
        return DepthAnalysis(
            depth_score=depth_score,
            flatness_score=flatness_score,
            is_likely_real=is_likely_real,
            confidence=estimation_confidence,
            depth_variance=depth_variance,
            depth_std=depth_std,
            focus_variation_score=focus_variation_score,
            shading_depth_score=shading_depth_score,
            texture_depth_score=texture_depth_score,
            shape_consistency_score=shape_consistency_score,
            depth_map=depth_map,
            anomalies=anomalies,
            estimation_confidence=estimation_confidence,
            is_3d_consistent=is_3d_consistent,
            focus_variation=focus_variation,
            natural_shadows_count=natural_shadows_count,
            texture_diversity=texture_diversity,
        )
        
    except Exception as e:
        logger.warning(f"Depth liveness analysis failed: {e}")
        return DepthAnalysis(
            depth_score=0.5,
            flatness_score=0.5,
            is_likely_real=True,  # При ошибке не отклоняем
            confidence=0.5,
            depth_variance=0,
            depth_std=0,
            focus_variation_score=0.5,
            shading_depth_score=0.5,
            texture_depth_score=0.5,
            shape_consistency_score=0.5,
            depth_map=None,
            anomalies=["analysis_error"],
            estimation_confidence=0.5,
            is_3d_consistent=False,
            focus_variation=0,
            natural_shadows_count=0,
            texture_diversity=0,
        )


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    """Упрощенное вычисление Local Binary Pattern."""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray[i, j]
            code = 0
            neighbors = [
                (i - 1, j), (i - 1, j + 1),
                (i, j + 1), (i + 1, j + 1),
                (i + 1, j), (i + 1, j - 1),
                (i, j - 1), (i - 1, j - 1)
            ]
            for k, (ni, nj) in enumerate(neighbors):
                if gray[ni, nj] >= center:
                    code |= (1 << k)
            lbp[i, j] = code
    
    return lbp


def _compute_glcm_simple(gray: np.ndarray) -> Dict[str, float]:
    """Упрощенное вычисление GLCM метрик."""
    h, w = gray.shape
    gray_quantized = (gray / 32).astype(np.uint8)
    gray_quantized = np.clip(gray_quantized, 0, 7)
    
    glcm = np.zeros((8, 8), dtype=np.float32)
    
    for i in range(h):
        for j in range(w - 1):
            i_q = gray_quantized[i, j]
            j_q = gray_quantized[i, j + 1]
            glcm[i_q, j_q] += 1
    
    glcm = glcm / (glcm.sum() + 1e-10)
    
    i_idx, j_idx = np.indices(glcm.shape)
    
    contrast = float(np.sum(glcm * (i_idx - j_idx) ** 2))
    entropy = float(-np.sum(glcm * np.log(glcm + 1e-10)))
    
    return {"contrast": contrast, "entropy": entropy}


def combine_liveness_scores(
    anti_spoofing_score: float,
    depth_score: float,
    lighting_quality: float,
    depth_analysis: Optional[DepthAnalysis] = None
) -> Dict[str, Any]:
    """
    Комбинирование различных оценок живости с улучшенной логикой.
    
    Args:
        anti_spoofing_score: Оценка от MiniFASNetV2 [0, 1]
        depth_score: Оценка 3D глубины [0, 1]
        lighting_quality: Качество освещения [0, 1]
        depth_analysis: Опционально, детальный анализ глубины
        
    Returns:
        Комбинированный результат с breakdown
    """
    # Веса для разных факторов
    weights = {
        "anti_spoofing": 0.55,  # Основной фактор - сертифицированная модель
        "depth": 0.30,          # 3D структура
        "lighting": 0.15,       # Качество освещения (косвенный)
    }
    
    # Корректировка на качество освещения
    lighting_factor = min(lighting_quality, 1.0)
    
    # Базовая комбинированная оценка
    combined_score = (
        weights["anti_spoofing"] * anti_spoofing_score +
        weights["depth"] * depth_score * lighting_factor +
        weights["lighting"] * lighting_quality * 0.5
    )
    
    # Дополнительная проверка на основе аномалий
    anomaly_bonus = 0.0
    anomaly_penalty = 0.0
    
    if depth_analysis:
        # Бонус за консистентность 3D
        if depth_analysis.is_3d_consistent:
            anomaly_bonus += 0.1
        
        # Штраф за аномалии
        anomaly_penalty = min(len(depth_analysis.anomalies) * 0.08, 0.25)
        
        # Дополнительная проверка flatness
        if depth_analysis.flatness_score > 0.7:
            anomaly_penalty += 0.15
    
    combined_score = combined_score + anomaly_bonus - anomaly_penalty
    
    # Финальное решение
    final_threshold = 0.5
    is_real = combined_score >= final_threshold
    
    # Конфиденциальность
    if is_real:
        confidence = min(combined_score, 0.99)
    else:
        confidence = min(1.0 - combined_score, 0.99)
    
    return {
        "liveness_detected": is_real,
        "combined_score": combined_score,
        "confidence": confidence,
        "is_3d_consistent": depth_analysis.is_3d_consistent if depth_analysis else False,
        "anomaly_count": len(depth_analysis.anomalies) if depth_analysis else 0,
        "anomaly_penalty": anomaly_penalty,
        "anti_spoofing_contribution": anti_spoofing_score * weights["anti_spoofing"],
        "depth_contribution": depth_score * weights["depth"] * lighting_factor,
        "lighting_quality_factor": lighting_quality,
        "decision_threshold": final_threshold,
        "requires_manual_review": (
            combined_score > 0.35 and combined_score < 0.65
        ),
    }
