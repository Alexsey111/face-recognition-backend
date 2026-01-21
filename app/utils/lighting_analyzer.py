"""
Улучшенный анализ освещения и теней для face recognition и anti-spoofing.

Анализирует:
- Распределение освещения на лице
- Наличие и характер теней
- Симметричность освещения (естественное vs искусственное)
- Градиенты освещения
- Следы вспышки / блики

Используется для:
- Улучшения качества эмбеддингов
- Детекции фото-атак (неравномерное освещение на фото)
- Оценки качества входного изображения
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
import asyncio

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LightingAnalyzer:
    """
    Комплексный анализ освещения и теней на изображении лица.

    Метрики:
    - Brightness distribution - равномерность освещения
    - Shadow analysis - анализ теней
    - Light symmetry - симметричность освещения
    - Flash detection - детекция вспышки
    - Ambient ratio - соотношение окружающего и направленного света
    - Gradient analysis - анализ градиентов освещения

    Применение:
    - Улучшение качества эмбеддингов при неравномерном освещении
    - Детекция фото-атак (характерные паттерны освещения экрана/печати)
    - Оценка условий съемки
    """

    def __init__(
        self,
        face_region: Optional[Tuple[int, int, int, int]] = None,  # (x1, y1, x2, y2)
        sensitivity: float = 1.0,  # 0.5-2.0, выше = более чувствительный
    ):
        """
        Args:
            face_region: Область лица для анализа (если известна)
            sensitivity: Чувствительность анализа
        """
        self.face_region = face_region
        self.sensitivity = sensitivity

        # Пороговые значения
        self.BRIGHTNESS_MIN = 40 * sensitivity
        self.BRIGHTNESS_MAX = 220 * sensitivity
        self.SHADOW_THRESHOLD = 30 * sensitivity
        self.SYMMETRY_TOLERANCE = 0.15  # 15% допуск для симметрии
        self.FLASH_INTENSITY_THRESHOLD = 230

        logger.info("LightingAnalyzer initialized")

    async def analyze(
        self, image_data: bytes, face_landmarks: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Комплексный анализ освещения.

        Args:
            image_data: Байты изображения
            face_landmarks: Опционально, landmarks лица для точного кропа

        Returns:
            Dict с результатами анализа
        """
        try:
            # Декодирование изображения
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            if image is None:
                return {"success": False, "error": "Failed to decode image"}

            # Конвертация в RGB для анализа
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Извлечение области лица
            face_img = self._extract_face_region(image_rgb, face_landmarks)

            if face_img is None or face_img.size == 0:
                return {"success": False, "error": "No face region detected"}

            # Вычисление метрик освещения
            brightness_analysis = await asyncio.to_thread(
                self._analyze_brightness, face_img
            )

            shadow_analysis = await asyncio.to_thread(self._analyze_shadows, face_img)

            symmetry_analysis = await asyncio.to_thread(
                self._analyze_light_symmetry, face_img
            )

            flash_analysis = await asyncio.to_thread(
                self._detect_flash_artifacts, face_img
            )

            gradient_analysis = await asyncio.to_thread(
                self._analyze_gradients, face_img
            )

            # Общая оценка качества освещения
            lighting_quality = self._calculate_lighting_quality(
                brightness_analysis, shadow_analysis, symmetry_analysis
            )

            # Оценка вероятности spoofing на основе паттернов освещения
            spoof_indicators = self._detect_spoofing_patterns(
                brightness_analysis,
                shadow_analysis,
                symmetry_analysis,
                flash_analysis,
                gradient_analysis,
            )

            return {
                "success": True,
                "lighting_quality": lighting_quality,
                "brightness": brightness_analysis,
                "shadows": shadow_analysis,
                "light_symmetry": symmetry_analysis,
                "flash_detection": flash_analysis,
                "gradients": gradient_analysis,
                "spoof_indicators": spoof_indicators,
                "is_good_lighting": lighting_quality["overall_score"] > 0.6,
                "recommendations": self._get_recommendations(
                    brightness_analysis, shadow_analysis, symmetry_analysis
                ),
            }

        except Exception as e:
            logger.error(f"Lighting analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _extract_face_region(
        self, image: np.ndarray, landmarks: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Извлечение области лица для анализа."""
        h, w = image.shape[:2]

        if landmarks is not None and len(landmarks) >= 5:
            # Используем landmarks для определения области лица
            # MTCNN landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            xs = landmarks[:, 0]
            ys = landmarks[:, 1]

            # Расширяем область для включения лба и подбородка
            face_x1 = max(0, int(xs.min() * 0.7))
            face_y1 = max(0, int(ys.min() * 0.6))
            face_x2 = min(w, int(xs.max() * 1.3))
            face_y2 = min(h, int(ys.max() * 1.5))

            return image[face_y1:face_y2, face_x1:face_x2]

        elif self.face_region is not None:
            x1, y1, x2, y2 = self.face_region
            return image[y1:y2, x1:x2]

        else:
            # Используем центр изображения
            center_size = min(w, h) // 2
            cx, cy = w // 2, h // 2
            x1 = cx - center_size // 2
            y1 = cy - center_size // 2
            x2 = x1 + center_size
            y2 = y1 + center_size

            return image[y1:y2, x1:x2]

    def _analyze_brightness(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Анализ яркости и экспозиции."""
        # Конвертация в оттенки серого
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

        # Базовые метрики яркости
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        median_brightness = float(np.median(gray))

        # Гистограммный анализ
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Процентили яркости
        cumulative = np.cumsum(hist)
        p10 = np.searchsorted(cumulative, 0.1)
        p90 = np.searchsorted(cumulative, 0.9)
        p25 = np.searchsorted(cumulative, 0.25)
        p75 = np.searchsorted(cumulative, 0.75)

        # Динамический диапазон
        dynamic_range = p90 - p10

        # Контрастность (Michelson contrast)
        if mean_brightness > 0:
            contrast = (p90 - p10) / (2 * mean_brightness + 1)
        else:
            contrast = 0

        # Оценка экспозиции
        exposure_score = 1.0
        if mean_brightness < self.BRIGHTNESS_MIN:
            exposure_score = mean_brightness / self.BRIGHTNESS_MIN
        elif mean_brightness > self.BRIGHTNESS_MAX:
            exposure_score = 1.0 - (mean_brightness - self.BRIGHTNESS_MAX) / (
                255 - self.BRIGHTNESS_MAX
            )

        # Равномерность освещения (коэффициент вариации)
        if mean_brightness > 0:
            uniformity = 1.0 - min(std_brightness / mean_brightness, 1.0)
        else:
            uniformity = 0

        return {
            "mean": mean_brightness,
            "std": std_brightness,
            "median": median_brightness,
            "percentile_10": float(p10),
            "percentile_25": float(p25),
            "percentile_75": float(p75),
            "percentile_90": float(p90),
            "dynamic_range": float(dynamic_range),
            "contrast": float(contrast),
            "exposure_score": float(exposure_score),
            "uniformity": float(uniformity),
            "is_properly_exposed": 0.4 < mean_brightness / 255 < 0.8,
            "has_good_contrast": dynamic_range > 80,
            "is_uniform": uniformity > 0.7,
        }

    def _analyze_shadows(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Анализ теней на лице."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

        # Сегментация теней с помощью адаптивного порога
        block_size = 51  # Нечетное для адаптивного порога
        binary_shadow = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            2,
        )

        # Морфологическая обработка для удаления шума
        kernel = np.ones((3, 3), np.uint8)
        binary_shadow = cv2.morphologyEx(binary_shadow, cv2.MORPH_OPEN, kernel)

        # Анализ контуров теней
        contours, _ = cv2.findContours(
            binary_shadow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Характеристики теней
        shadow_pixels = np.sum(binary_shadow > 0)
        total_pixels = gray.shape[0] * gray.shape[1]
        shadow_ratio = shadow_pixels / total_pixels

        # Размеры и распределение теней
        shadow_areas = [
            cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100
        ]

        if shadow_areas:
            avg_shadow_size = np.mean(shadow_areas)
            max_shadow_size = max(shadow_areas)
            shadow_distribution = len(shadow_areas) / (
                total_pixels / 10000
            )  # плотность теней
        else:
            avg_shadow_size = 0
            max_shadow_size = 0
            shadow_distribution = 0

        # Анализ градиента теней (переходы света/тени)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Средний градиент (резкость переходов)
        avg_gradient = float(np.mean(gradient_magnitude))

        # Естественность теней (мягкие переходы характерны для живых лиц)
        sharp_transitions = np.sum(gradient_magnitude > 100) / total_pixels

        # Классификация паттерна теней
        if shadow_ratio < 0.1:
            shadow_pattern = "minimal"
        elif shadow_ratio < 0.25:
            shadow_pattern = "soft"
        elif shadow_ratio < 0.4:
            shadow_pattern = "moderate"
        elif shadow_ratio < 0.6:
            shadow_pattern = "heavy"
        else:
            shadow_pattern = "extreme"

        # Оценка естественности теней
        natural_shadow_score = 1.0
        if shadow_pattern in ["minimal", "soft"]:
            natural_shadow_score = 0.8 - abs(avg_gradient - 20) / 100
        elif shadow_pattern == "moderate":
            natural_shadow_score = 0.6
        else:
            natural_shadow_score = 0.3

        return {
            "shadow_ratio": float(shadow_ratio),
            "shadow_pattern": shadow_pattern,
            "avg_shadow_size": float(avg_shadow_size) if shadow_areas else 0,
            "max_shadow_size": float(max_shadow_size) if shadow_areas else 0,
            "shadow_density": float(shadow_distribution),
            "avg_gradient": avg_gradient,
            "sharp_transitions": float(sharp_transitions),
            "natural_shadow_score": max(0.0, min(1.0, natural_shadow_score)),
            "is_natural_shadow": natural_shadow_score > 0.5,
        }

    def _analyze_light_symmetry(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Анализ симметричности освещения на лице."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Разделение на левую и правую половины
        mid_x = w // 2

        left_half = gray[:, :mid_x]
        right_half = gray[:, mid_x:]

        # Зеркалим правую половину для сравнения
        right_half_flipped = cv2.flip(right_half, 1)

        # Сравнение размеров (могут отличаться на 1 пиксель)
        min_w = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half_cropped = left_half[:, :min_w]
        right_half_cropped = right_half_flipped[:, :min_w]

        # Разница яркости между половинами
        brightness_diff = np.abs(
            left_half_cropped.astype(float) - right_half_cropped.astype(float)
        )
        avg_brightness_diff = float(np.mean(brightness_diff))

        # Отношение средней яркости
        left_mean = float(np.mean(left_half))
        right_mean = float(np.mean(right_half))
        brightness_ratio = min(left_mean, right_mean) / (
            max(left_mean, right_mean) + 1e-6
        )

        # Стандартное отклонение разницы (неоднородность)
        brightness_std = float(np.std(brightness_diff))

        # Анализ симметрии по регионам (верхняя/нижняя часть лица)
        mid_y = h // 2
        top_half = gray[:mid_y, :]
        bottom_half = gray[mid_y:, :]

        top_mean = float(np.mean(top_half))
        bottom_mean = float(np.mean(bottom_half))
        vertical_brightness_ratio = min(top_mean, bottom_mean) / (
            max(top_mean, bottom_mean) + 1e-6
        )

        # Symmetry score (1.0 = полная симметрия)
        symmetry_score = 1.0 - (brightness_diff / 255).mean()

        # Определение типа освещения по симметрии
        if brightness_ratio > 0.85:
            light_type = "symmetric_front"
        elif brightness_ratio > 0.7:
            light_type = "side_light"
        elif brightness_ratio > 0.5:
            light_type = "strong_side_light"
        else:
            light_type = "mixed_uneven"

        # Оценка естественности освещения
        natural_light_score = 1.0
        if light_type in ["symmetric_front", "side_light"]:
            natural_light_score = 0.8
        elif light_type == "strong_side_light":
            natural_light_score = 0.6
        else:
            natural_light_score = 0.4

        return {
            "brightness_difference": avg_brightness_diff,
            "brightness_ratio": brightness_ratio,
            "brightness_std": brightness_std,
            "vertical_brightness_ratio": vertical_brightness_ratio,
            "symmetry_score": float(symmetry_score),
            "light_type": light_type,
            "natural_light_score": float(natural_light_score),
            "is_symmetric": brightness_ratio > 0.8,
            "is_natural": natural_light_score > 0.6,
        }

    def _detect_flash_artifacts(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Детекция артефактов вспышки и бликов."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

        # Поиск пересвеченных областей (возможные блики от вспышки)
        _, bright_mask = cv2.threshold(
            gray, self.FLASH_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        bright_pixels = np.sum(bright_mask > 0)
        total_pixels = gray.shape[0] * gray.shape[1]
        bright_ratio = bright_pixels / total_pixels

        # Анализ формы бликов (естественные блики имеют неправильную форму)
        contours, _ = cv2.findContours(
            bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Circularity для каждого контура
            circularities = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 10:
                    perimeter = cv2.arcLength(c, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter**2)
                        circularities.append(circularity)

            if circularities:
                avg_circularity = np.mean(circularities)
                max_circularity = max(circularities)
            else:
                avg_circularity = 0
                max_circularity = 0
        else:
            avg_circularity = 0
            max_circularity = 0

        # Анализ области глаз (характерное расположение бликов от вспышки)
        h, w = gray.shape
        eye_region_y = int(h * 0.3)
        eye_region_h = int(h * 0.25)
        eye_region = gray[eye_region_y : eye_region_y + eye_region_h, :]

        eye_bright_pixels = np.sum(eye_region > 200)
        eye_total = eye_region.shape[0] * eye_region.shape[1]
        eye_bright_ratio = eye_bright_pixels / eye_total

        # Red-eye detection (характерно для фото со вспышкой)
        if len(face_img.shape) == 3:
            # Анализ цветовых каналов в ярких областях
            r_channel = face_img[:, :, 0]
            g_channel = face_img[:, :, 1]
            b_channel = face_img[:, :, 2]

            # Красные глаза имеют R > G и R > B
            red_eye_mask = (
                (r_channel > 150) & (r_channel > g_channel) & (r_channel > b_channel)
            )
            red_eye_pixels = np.sum(red_eye_mask)

            red_eye_ratio = red_eye_pixels / total_pixels
        else:
            red_eye_ratio = 0

        # Оценка вероятности использования вспышки
        flash_probability = 0.0
        if bright_ratio > 0.05:
            flash_probability += 0.3
        if max_circularity > 0.7:
            flash_probability += 0.2
        if eye_bright_ratio > 0.1:
            flash_probability += 0.3
        if red_eye_ratio > 0.005:
            flash_probability += 0.4

        return {
            "bright_pixel_ratio": float(bright_ratio),
            "avg_bright_region_circularity": float(avg_circularity),
            "max_bright_region_circularity": float(max_circularity),
            "eye_region_bright_ratio": float(eye_bright_ratio),
            "red_eye_ratio": float(red_eye_ratio),
            "flash_probability": min(1.0, flash_probability),
            "has_flash_artifacts": flash_probability > 0.3,
            "is_natural_light": flash_probability < 0.2,
        }

    def _analyze_gradients(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Анализ градиентов освещения."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Вычисление градиентов по обеим осям
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Магнитуда градиента
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Направление градиента
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        # Статистики градиента
        mean_gradient = float(np.mean(gradient_magnitude))
        std_gradient = float(np.std(gradient_magnitude))
        max_gradient = float(np.max(gradient_magnitude))

        # Анализ направления преобладающего освещения
        # Гистограмма направлений
        directions = gradient_direction.flatten()
        magnitudes = gradient_magnitude.flatten()

        # Взвешенная гистограмма по магнитуде
        hist_bins = 8
        hist_range = (-np.pi, np.pi)
        hist_weights = magnitudes / (magnitudes.sum() + 1e-6)

        # Определяем преобладающее направление
        dominant_direction = float(np.median(directions))

        # Горизонтальная/вертикальная составляющая
        horizontal_component = np.abs(np.cos(dominant_direction))
        vertical_component = np.abs(np.sin(dominant_direction))

        if horizontal_component > vertical_component:
            light_direction = "horizontal"
        else:
            light_direction = "vertical"

        # Оценка плавности градиентов (для естественного освещения характерны плавные переходы)
        smooth_gradient_ratio = np.sum(gradient_magnitude < 50) / (h * w)

        # Оценка сложности текстуры
        texture_complexity = std_gradient / (mean_gradient + 1e-6)

        return {
            "mean_gradient": mean_gradient,
            "std_gradient": std_gradient,
            "max_gradient": max_gradient,
            "dominant_direction_degrees": float(np.degrees(dominant_direction)),
            "light_direction": light_direction,
            "smooth_gradient_ratio": float(smooth_gradient_ratio),
            "texture_complexity": float(texture_complexity),
            "is_natural_lighting": smooth_gradient_ratio > 0.6,
        }

    def _calculate_lighting_quality(
        self, brightness: Dict, shadows: Dict, symmetry: Dict
    ) -> Dict[str, Any]:
        """Расчет общей оценки качества освещения."""
        # Веса для разных компонентов
        exposure_weight = 0.3
        shadow_weight = 0.35
        symmetry_weight = 0.35

        # Оценка экспозиции
        exposure_score = brightness.get("exposure_score", 0.5)

        # Оценка теней
        shadow_score = shadows.get("natural_shadow_score", 0.5)

        # Оценка симметрии
        symmetry_score = symmetry.get("symmetry_score", 0.5)

        # Комбинированная оценка
        overall_score = (
            exposure_weight * exposure_score
            + shadow_weight * shadow_score
            + symmetry_weight * symmetry_score
        )

        # Определение качества
        if overall_score > 0.8:
            quality_level = "excellent"
        elif overall_score > 0.65:
            quality_level = "good"
        elif overall_score > 0.5:
            quality_level = "fair"
        elif overall_score > 0.35:
            quality_level = "poor"
        else:
            quality_level = "bad"

        return {
            "overall_score": float(overall_score),
            "exposure_score": float(exposure_score),
            "shadow_score": float(shadow_score),
            "symmetry_score": float(symmetry_score),
            "quality_level": quality_level,
        }

    def _detect_spoofing_patterns(
        self,
        brightness: Dict,
        shadows: Dict,
        symmetry: Dict,
        flash: Dict,
        gradients: Dict,
    ) -> Dict[str, Any]:
        """Детекция паттернов, характерных для spoofing-атак."""

        # Неестественно равномерное освещение (характерно для экрана)
        uniform_spoof = 0.0
        if brightness.get("uniformity", 0) > 0.9:
            uniform_spoof += 0.4
        if brightness.get("dynamic_range", 255) < 60:
            uniform_spoof += 0.3

        # Неестественные тени
        shadow_spoof = 0.0
        if not shadows.get("is_natural_shadow", True):
            shadow_spoof += 0.3
        if shadows.get("shadow_pattern") in ["minimal", "extreme"]:
            shadow_spoof += 0.2

        # Асимметрия освещения
        symmetry_spoof = 0.0
        if not symmetry.get("is_symmetric", True):
            symmetry_spoof += 0.2
        if symmetry.get("light_type") == "mixed_uneven":
            symmetry_spoof += 0.3

        # Артефакты вспышки
        flash_spoof = flash.get("flash_probability", 0)

        # Градиенты
        gradient_spoof = 0.0
        if not gradients.get("is_natural_lighting", True):
            gradient_spoof += 0.2

        # Общая оценка вероятности spoofing
        combined_spoof_prob = (
            uniform_spoof * 0.25
            + shadow_spoof * 0.25
            + symmetry_spoof * 0.2
            + flash_spoof * 0.2
            + gradient_spoof * 0.1
        )

        return {
            "uniform_light_spoof_score": uniform_spoof,
            "shadow_spoof_score": shadow_spoof,
            "symmetry_spoof_score": symmetry_spoof,
            "flash_spoof_score": flash_spoof,
            "gradient_spoof_score": gradient_spoof,
            "combined_spoof_probability": min(1.0, combined_spoof_prob),
            "is_likely_spoof": combined_spoof_prob > 0.4,
        }

    def _get_recommendations(
        self, brightness: Dict, shadows: Dict, symmetry: Dict
    ) -> list[str]:
        """Генерация рекомендаций по улучшению освещения."""
        recommendations = []

        if not brightness.get("is_properly_exposed", True):
            if brightness.get("mean", 128) < 50:
                recommendations.append("image_too_dark")
            elif brightness.get("mean", 128) > 200:
                recommendations.append("image_too_bright")

        if not brightness.get("is_uniform", True):
            recommendations.append("uneven_lighting")

        if not shadows.get("is_natural_shadow", True):
            recommendations.append("unnatural_shadow_pattern")

        if not symmetry.get("is_symmetric", True):
            if symmetry.get("light_type") == "strong_side_light":
                recommendations.append("side_lighting_too_extreme")
            else:
                recommendations.append("uneven_lighting_symmetry")

        return recommendations


# =============================================================================
# Utility functions
# =============================================================================


async def quick_lighting_check(image_data: bytes) -> Dict[str, Any]:
    """
    Быстрая проверка качества освещения.

    Returns:
        Dict с основными метриками и оценкой
    """
    analyzer = LightingAnalyzer()
    result = await analyzer.analyze(image_data)

    if result.get("success"):
        return {
            "lighting_quality": result["lighting_quality"]["overall_score"],
            "is_good": result["is_good_lighting"],
            "recommendations": result.get("recommendations", []),
        }

    return {"lighting_quality": 0.5, "is_good": False, "error": result.get("error")}


# =============================================================================
# Singleton
# =============================================================================

_analyzer: Optional[LightingAnalyzer] = None


async def get_lighting_analyzer(
    face_region: Optional[Tuple[int, int, int, int]] = None
) -> LightingAnalyzer:
    """Получение singleton экземпляра LightingAnalyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LightingAnalyzer(face_region=face_region)
    return _analyzer
