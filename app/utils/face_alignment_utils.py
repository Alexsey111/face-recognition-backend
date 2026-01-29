"""
Face alignment utilities with MediaPipe.
Совместимость с MediaPipe 0.10.0 - 0.10.32+
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# MediaPipe Import (Compatible with all versions 0.10+)
# ============================================================================

HAS_MEDIAPIPE = False
mp = None
mp_face_mesh = None
import_method = None

try:
    import mediapipe as mp

    # MediaPipe 0.10.10+ использует НОВЫЙ API через tasks
    if hasattr(mp, "tasks"):
        try:
            from mediapipe.tasks.python import vision

            # Проверяем что FaceLandmarker доступен
            if hasattr(vision, "FaceLandmarker"):
                mp_face_mesh = vision.FaceLandmarker
                HAS_MEDIAPIPE = True
                import_method = "mp.tasks.vision.FaceLandmarker"
                logger.info(f"MediaPipe 0.10.10+ detected, using tasks API")
        except (ImportError, AttributeError) as e:
            logger.warning(f"MediaPipe tasks API not available: {e}")

    # MediaPipe 0.10.0-0.10.9 использует СТАРЫЙ API через solutions
    if not HAS_MEDIAPIPE and hasattr(mp, "solutions"):
        try:
            if hasattr(mp.solutions, "face_mesh"):
                mp_face_mesh = mp.solutions.face_mesh
                HAS_MEDIAPIPE = True
                import_method = "mp.solutions.face_mesh"
                logger.info(f"MediaPipe 0.10.0-0.10.9 detected, using solutions API")
        except (ImportError, AttributeError) as e:
            logger.warning(f"MediaPipe solutions API not available: {e}")

except ImportError as e:
    HAS_MEDIAPIPE = False
    mp_face_mesh = None
    import_method = None
    raise ImportError(
        f"MediaPipe is required but import failed: {e}\n"
        "Install it with: pip install mediapipe>=0.10.0"
    )

# Проверяем что MediaPipe установлен корректно
if not HAS_MEDIAPIPE or mp_face_mesh is None:
    error_msg = (
        "MediaPipe is not properly installed or face_mesh module is not available.\n"
        "Debugging info:\n"
        f"  - MediaPipe installed: {mp is not None}\n"
        f"  - MediaPipe version: {getattr(mp, '__version__', 'unknown')}\n"
        f"  - Has 'solutions': {hasattr(mp, 'solutions') if mp else False}\n"
        f"  - Has 'tasks': {hasattr(mp, 'tasks') if mp else False}\n"
        f"  - HAS_MEDIAPIPE: {HAS_MEDIAPIPE}\n"
        f"  - mp_face_mesh: {mp_face_mesh}\n"
        f"  - import_method: {import_method}\n"
        "\nSolutions:\n"
        "  1. Try: pip uninstall mediapipe && pip install mediapipe==0.10.9\n"
        "  2. Or: pip install --upgrade mediapipe\n"
        "  3. Check Python version (MediaPipe requires Python 3.8-3.11)"
    )
    raise ImportError(error_msg)

logger.info(
    f"✅ MediaPipe loaded successfully\n"
    f"   Version: {getattr(mp, '__version__', 'unknown')}\n"
    f"   Import method: {import_method}"
)

# ============================================================================

# MediaPipe → dlib 68-point mapping
# ============================================================================

MEDIAPIPE_TO_DLIB_68 = {
    # Контур лица (0-16)
    0: 234,
    1: 227,
    2: 137,
    3: 177,
    4: 215,
    5: 138,
    6: 135,
    7: 169,
    8: 152,  # Подбородок
    9: 378,
    10: 365,
    11: 397,
    12: 288,
    13: 447,
    14: 366,
    15: 401,
    16: 454,
    # Левая бровь (17-21)
    17: 70,
    18: 63,
    19: 105,
    20: 66,
    21: 107,
    # Правая бровь (22-26)
    22: 336,
    23: 296,
    24: 334,
    25: 293,
    26: 300,
    # Нос (27-35)
    27: 168,
    28: 6,
    29: 197,
    30: 195,
    31: 5,
    32: 4,
    33: 1,
    34: 19,
    35: 164,
    # Левый глаз (36-41)
    36: 33,
    37: 160,
    38: 158,
    39: 133,
    40: 153,
    41: 144,
    # Правый глаз (42-47)
    42: 362,
    43: 385,
    44: 387,
    45: 263,
    46: 373,
    47: 380,
    # Рот внешний (48-59)
    48: 61,
    49: 185,
    50: 40,
    51: 39,
    52: 37,
    53: 0,
    54: 267,
    55: 269,
    56: 270,
    57: 409,
    58: 291,
    59: 375,
    # Рот внутренний (60-67)
    60: 78,
    61: 191,
    62: 80,
    63: 81,
    64: 82,
    65: 13,
    66: 312,
    67: 311,
}

# ============================================================================

# FaceLandmarks Class (dlib-compatible)
# ============================================================================


@dataclass
class FaceLandmarks:
    """
    Класс для работы с 68-point facial landmarks.
    Совместим с dlib форматом, но использует MediaPipe под капотом.
    """

    points: np.ndarray  # Shape: (68, 2)

    @classmethod
    def from_68_points(cls, landmarks: np.ndarray) -> FaceLandmarks:
        """
        Создание из 68-point landmarks.

        Args:
            landmarks: Массив формы (68, 2) с координатами

        Returns:
            FaceLandmarks объект
        """
        if landmarks.shape != (68, 2):
            raise ValueError(f"Expected shape (68, 2), got {landmarks.shape}")

        return cls(points=landmarks)

    @classmethod
    def from_mediapipe(
        cls, mediapipe_landmarks, image_width: int, image_height: int
    ) -> FaceLandmarks:
        """
        Создание из MediaPipe landmarks (468 точек → 68 точек).

        Args:
            mediapipe_landmarks: MediaPipe FaceLandmarkList
            image_width: Ширина изображения
            image_height: Высота изображения

        Returns:
            FaceLandmarks объект
        """
        landmarks_68 = np.zeros((68, 2), dtype=np.float32)

        for dlib_idx, mediapipe_idx in MEDIAPIPE_TO_DLIB_68.items():
            lm = mediapipe_landmarks.landmark[mediapipe_idx]
            landmarks_68[dlib_idx] = [lm.x * image_width, lm.y * image_height]

        return cls(points=landmarks_68)

    def get_eye_distance(self) -> float:
        """Расстояние между центрами глаз."""
        # Левый глаз: точки 36-41
        left_eye_center = np.mean(self.points[36:42], axis=0)

        # Правый глаз: точки 42-47
        right_eye_center = np.mean(self.points[42:48], axis=0)

        distance = np.linalg.norm(right_eye_center - left_eye_center)
        return float(distance)

    def get_face_ratio(self) -> float:
        """
        Соотношение ширины и высоты лица.
        Идеально: ~0.75-0.85
        """
        # Ширина: расстояние между точками 0 и 16 (края лица)
        face_width = np.linalg.norm(self.points[16] - self.points[0])

        # Высота: расстояние от точки 8 (подбородок) до точки 27 (переносица)
        face_height = np.linalg.norm(self.points[8] - self.points[27])

        if face_height == 0:
            return 0.0

        ratio = face_width / face_height
        return float(ratio)

    def get_nose_to_chin_distance(self) -> float:
        """Расстояние от носа до подбородка."""
        nose_tip = self.points[30]  # Кончик носа
        chin = self.points[8]  # Подбородок

        distance = np.linalg.norm(chin - nose_tip)
        return float(distance)

    def get_mouth_width(self) -> float:
        """Ширина рта."""
        left_corner = self.points[48]  # Левый угол рта
        right_corner = self.points[54]  # Правый угол рта

        width = np.linalg.norm(right_corner - left_corner)
        return float(width)

    def to_array(self) -> np.ndarray:
        """Возвращает landmarks как numpy array."""
        return self.points


# ============================================================================

# Landmark Detection
# ============================================================================


def detect_face_landmarks(
    image: np.ndarray,
    return_468: bool = False,
) -> Optional[np.ndarray]:
    """
    Детекция facial landmarks с помощью MediaPipe.

    Args:
        image: Изображение в формате RGB
        return_468: Если True, возвращает все 468 точек MediaPipe,
                    иначе 68 точек в dlib формате

    Returns:
        np.ndarray: (68, 2) или (468, 2) или None
    """
    if not HAS_MEDIAPIPE:
        raise ImportError("MediaPipe is not installed")

    try:
        h, w = image.shape[:2]

        # ✅ ИСПРАВЛЕНО: Проверяем какой API использовать
        # Проверяем API MediaPipe
        if import_method == "mp.tasks.vision.FaceLandmarker":
            # ✅ Новый API (MediaPipe 0.10.10+)
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision import FaceLandmarkerOptions

            # Создаем опции
            base_options = mp.tasks.BaseOptions(
                model_asset_path=None,  # Используем встроенную модель
            )
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # Создаем детектор
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                # Конвертируем в RGB MediaPipe Image
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image

                # MediaPipe Image требует uint8
                if image_rgb.dtype != np.uint8:
                    image_rgb = (image_rgb * 255).astype(np.uint8)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                # Детекция
                detection_result = landmarker.detect(mp_image)

                if not detection_result.face_landmarks:
                    return None

                face_landmarks = detection_result.face_landmarks[0]

                if return_468:
                    # Все 468 точек
                    landmarks = np.zeros((468, 2), dtype=np.float32)
                    for idx, landmark in enumerate(face_landmarks):
                        landmarks[idx] = [landmark.x * w, landmark.y * h]
                    return landmarks
                else:
                    # 68 точек (dlib-compatible)
                    landmarks_68 = np.zeros((68, 2), dtype=np.float32)
                    for dlib_idx, mediapipe_idx in MEDIAPIPE_TO_DLIB_68.items():
                        lm = face_landmarks[mediapipe_idx]
                        landmarks_68[dlib_idx] = [lm.x * w, lm.y * h]
                    return landmarks_68

        elif hasattr(mp_face_mesh, "FaceMesh"):
            # ✅ Старый API (MediaPipe 0.10.0-0.10.9)
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            ) as face_mesh:

                # Конвертируем в RGB если нужно
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image

                results = face_mesh.process(image_rgb)

                if not results.multi_face_landmarks:
                    return None

                face_landmarks = results.multi_face_landmarks[0]

                if return_468:
                    # Все 468 точек
                    landmarks = np.zeros((468, 2), dtype=np.float32)
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        landmarks[idx] = [landmark.x * w, landmark.y * h]
                    return landmarks
                else:
                    # 68 точек (dlib-compatible)
                    landmarks_68 = np.zeros((68, 2), dtype=np.float32)
                    for dlib_idx, mediapipe_idx in MEDIAPIPE_TO_DLIB_68.items():
                        lm = face_landmarks.landmark[mediapipe_idx]
                        landmarks_68[dlib_idx] = [lm.x * w, lm.y * h]
                    return landmarks_68
        else:
            # Fallback: возможно это новый API или legacy
            logger.error(
                f"Unsupported MediaPipe API structure. "
                f"mp_face_mesh type: {type(mp_face_mesh)}, import_method: {import_method}"
            )
            raise NotImplementedError(
                "This MediaPipe version is not yet supported. "
                "Please use MediaPipe 0.10.0-0.10.9"
            )

    except Exception as e:
        logger.error(f"MediaPipe detection failed: {str(e)}", exc_info=True)
        return None


# ============================================================================

# Face Alignment
# ============================================================================


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
    eye_line_target_y: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Выравнивание лица по landmarks.

    Args:
        image: Исходное изображение
        landmarks: 68-point landmarks
        output_size: Размер выходного изображения
        eye_line_target_y: Целевая Y-координата линии глаз (0-1)

    Returns:
        Tuple[aligned_face, alignment_info]
    """
    # Центры глаз
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    # Вычисляем угол поворота
    delta = right_eye - left_eye
    angle = np.degrees(np.arctan2(delta[1], delta[0]))

    # Расстояние между глазами
    eye_distance = np.linalg.norm(delta)

    # Центр между глазами
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # Вычисляем scale
    desired_eye_distance = output_size[0] * 0.4
    scale = desired_eye_distance / eye_distance

    # Матрица поворота
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Корректируем центр
    target_eyes_center_y = output_size[1] * eye_line_target_y
    M[1, 2] += target_eyes_center_y - eyes_center[1]
    M[0, 2] += output_size[0] / 2 - eyes_center[0]

    # Применяем трансформацию
    aligned = cv2.warpAffine(
        image,
        M,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Метаданные
    alignment_info = {
        "rotation_angle": float(angle),
        "scale": float(scale),
        "eye_distance": float(eye_distance),
        "eyes_center": tuple(map(float, eyes_center)),
        "face_ratio": float(eye_distance / output_size[0]),
    }

    return aligned, alignment_info


# ============================================================================

# Lighting Analysis
# ============================================================================


@dataclass
class LightingAnalysis:
    """Результаты анализа освещения."""

    overall_quality: float
    exposure_score: float
    shadow_evenness: float
    left_right_balance: float
    contrast_score: float
    issues: List[str]
    recommendations: List[str]


def analyze_shadows_and_lighting(face_image: np.ndarray) -> LightingAnalysis:
    """
    Анализ освещения и теней на лице.

    Args:
        face_image: Изображение лица (RGB)

    Returns:
        LightingAnalysis с метриками
    """
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)

    # 1. Exposure
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    if mean_brightness < 50:
        exposure_score = mean_brightness / 50.0
        exposure_issue = "too_dark"
    elif mean_brightness > 200:
        exposure_score = (255 - mean_brightness) / 55.0
        exposure_issue = "overexposed"
    else:
        exposure_score = 1.0
        exposure_issue = None

    # 2. Contrast
    contrast_score = min(std_brightness / 50.0, 1.0)

    # 3. Shadow evenness
    h, w = gray.shape
    top_half = gray[: h // 2, :]
    bottom_half = gray[h // 2 :, :]

    top_brightness = np.mean(top_half)
    bottom_brightness = np.mean(bottom_half)

    shadow_diff = abs(top_brightness - bottom_brightness)
    shadow_evenness = max(0.0, 1.0 - shadow_diff / 50.0)

    # 4. Left-right balance
    left_half = gray[:, : w // 2]
    right_half = gray[:, w // 2 :]

    left_brightness = np.mean(left_half)
    right_brightness = np.mean(right_half)

    lr_diff = abs(left_brightness - right_brightness)
    left_right_balance = max(0.0, 1.0 - lr_diff / 30.0)

    # Overall quality
    overall_quality = (
        exposure_score * 0.4
        + contrast_score * 0.2
        + shadow_evenness * 0.2
        + left_right_balance * 0.2
    )

    # Issues
    issues = []
    recommendations = []

    if exposure_issue == "too_dark":
        issues.append("insufficient_lighting")
        recommendations.append("Increase lighting or use flash")
    elif exposure_issue == "overexposed":
        issues.append("overexposure")
        recommendations.append("Reduce lighting intensity")

    if contrast_score < 0.4:
        issues.append("low_contrast")
        recommendations.append("Improve lighting contrast")

    if shadow_evenness < 0.6:
        issues.append("uneven_shadows")
        recommendations.append("Use diffused lighting")

    if left_right_balance < 0.7:
        issues.append("asymmetric_lighting")
        recommendations.append("Balance left/right lighting")

    return LightingAnalysis(
        overall_quality=overall_quality,
        exposure_score=exposure_score,
        shadow_evenness=shadow_evenness,
        left_right_balance=left_right_balance,
        contrast_score=contrast_score,
        issues=issues,
        recommendations=recommendations,
    )


def enhance_lighting(
    face_image: np.ndarray, lighting_analysis: LightingAnalysis
) -> np.ndarray:
    """
    Улучшение освещения лица.

    Args:
        face_image: Изображение лица
        lighting_analysis: Результаты анализа

    Returns:
        Улучшенное изображение
    """
    enhanced = face_image.copy()

    # CLAHE для улучшения контраста
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    # Гамма-коррекция при необходимости
    if lighting_analysis.exposure_score < 0.6:
        gamma = 1.5 if lighting_analysis.overall_quality < 0.5 else 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        enhanced = cv2.LUT(enhanced, table)

    return enhanced


# ============================================================================

# Depth Analysis
# ============================================================================


@dataclass
class DepthAnalysis:
    """Результаты анализа глубины."""

    depth_score: float
    flatness_score: float
    is_likely_real: bool
    confidence: float
    anomalies: List[str]


def analyze_depth_for_liveness(face_image: np.ndarray) -> DepthAnalysis:
    """
    Простой анализ 3D глубины для liveness detection.

    Args:
        face_image: Изображение лица

    Returns:
        DepthAnalysis
    """
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)

    # Gradient analysis
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_std = np.std(gradient_magnitude)

    # Flatness (плоское лицо = фото)
    flatness_score = 1.0 - min(gradient_std / 50.0, 1.0)

    # Depth score (высокий = 3D объект)
    depth_score = min(gradient_std / 30.0, 1.0)

    # Anomalies
    anomalies = []
    if flatness_score > 0.7:
        anomalies.append("high_flatness")
    if depth_score < 0.3:
        anomalies.append("insufficient_depth")

    is_likely_real = depth_score > 0.4 and flatness_score < 0.6
    confidence = depth_score if is_likely_real else 1.0 - depth_score

    return DepthAnalysis(
        depth_score=depth_score,
        flatness_score=flatness_score,
        is_likely_real=is_likely_real,
        confidence=confidence,
        anomalies=anomalies,
    )


def combine_liveness_scores(
    anti_spoofing_score: float,
    depth_score: float,
    lighting_quality: float,
    depth_analysis: Optional[DepthAnalysis] = None,
) -> Dict[str, Any]:
    """
    Комбинирование различных liveness оценок.

    Args:
        anti_spoofing_score: Оценка MiniFASNetV2 (0-1)
        depth_score: Оценка 3D глубины (0-1)
        lighting_quality: Качество освещения (0-1)
        depth_analysis: Детальный анализ глубины

    Returns:
        Dict с комбинированными результатами
    """
    # Веса компонентов
    weights = {
        "anti_spoofing": 0.6,
        "depth": 0.25,
        "lighting": 0.15,
    }

    # Взвешенная сумма
    combined_score = (
        anti_spoofing_score * weights["anti_spoofing"]
        + depth_score * weights["depth"]
        + lighting_quality * weights["lighting"]
    )

    # Корректировка на аномалии
    if depth_analysis and len(depth_analysis.anomalies) > 0:
        penalty = len(depth_analysis.anomalies) * 0.1
        combined_score = max(0.0, combined_score - penalty)

    liveness_detected = combined_score > 0.5

    return {
        "liveness_detected": liveness_detected,
        "confidence": combined_score,
        "anti_spoofing_score": anti_spoofing_score,
        "depth_score": depth_score,
        "lighting_quality": lighting_quality,
        "weights": weights,
        "method": "weighted_combination",
    }
