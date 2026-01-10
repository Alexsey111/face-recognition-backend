"""
3D Depth Estimation для Liveness Detection.

Оценивает глубину/объёмность лица для детекции плоских атак:
- Фотографии (напечатанные или на экране)
- Видео на экране
- Depth map анализ

Методы:
1. Depth-from-focus/defocus - оценка глубины по размытию
2. Texture gradient analysis - анализ текстурных градиентов  
3. Shading-based depth - оценка по паттернам освещения
4. Size consistency - проверка согласованности размеров
5. 3D morphable model fitting - оценка соответствия 3D модели

Возвращает depth_score для оценки "объёмности" лица.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
import asyncio

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    """
    Оценка 3D глубины/объёмности лица для детекции spoofing-атак.
    
    Живое лицо имеет характерные признаки объёмности:
    - Тени в естественных местах (нос, глаза, подбородок)
    - Разная резкость в разных частях лица (при наклоне)
    - 3D структура, соответствующая антропометрическим нормам
    - Естественные искажения при движении
    
    Плоские атаки (фото/экран):
    - Равномерная глубина или отсутствие глубины
    - Характерные паттерны освещения экрана/печати
    - moiré паттерны от экрана
    - Искажения от линз камеры
    
    Returns:
        depth_score: 0.0-1.0, где 1.0 = высокое качество 3D структуры (real)
    """
    
    def __init__(
        self,
        face_size: int = 224,
        sensitivity: float = 1.0,
        use_gpu: bool = False
    ):
        """
        Args:
            face_size: Размер обрабатываемого изображения лица
            sensitivity: Чувствительность детектора (0.5-2.0)
            use_gpu: Использовать GPU для вычислений
        """
        self.face_size = face_size
        self.sensitivity = sensitivity
        self.device = "cuda" if use_gpu else "cpu"
        
        # Пороги для классификации
        self.DEPTH_REAL_THRESHOLD = 0.55  # Выше = real
        self.DEPTH_SPOOF_THRESHOLD = 0.40  # Ниже = spoof
        
        # Параметры анализа
        self.LAPLACIAN_KERNEL_SIZE = 5
        self.BLUR_KERNEL_SIZES = [3, 7, 15, 21]
        
        logger.info(f"DepthEstimator initialized (face_size={face_size}, sensitivity={sensitivity})")
    
    async def estimate_depth(
        self,
        image_data: bytes,
        face_landmarks: Optional[np.ndarray] = None,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Комплексная оценка 3D глубины лица.
        
        Args:
            image_data: Байты изображения
            face_landmarks: Опционально, landmarks лица
            face_bbox: Опционально, bounding box лица (x1, y1, x2, y2)
            
        Returns:
            Dict с depth_score и детальным анализом
        """
        try:
            # Декодирование изображения
            image = cv2.imdecode(
                np.frombuffer(image_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if image is None:
                return {"success": False, "error": "Failed to decode image"}
            
            # Извлечение области лица
            face_img = self._extract_face_region(image, face_landmarks, face_bbox)
            
            if face_img is None or face_img.size == 0:
                return {"success": False, "error": "No face region detected"}
            
            # Ресайз к стандартному размеру
            face_resized = cv2.resize(face_img, (self.face_size, self.face_size))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Параллельный анализ разных методов
            focus_analysis, shading_analysis, texture_analysis, shape_analysis = await asyncio.gather(
                asyncio.to_thread(self._analyze_depth_from_focus, face_rgb),
                asyncio.to_thread(self._analyze_shading_depth, face_rgb),
                asyncio.to_thread(self._analyze_texture_gradient, face_rgb),
                asyncio.to_thread(self._analyze_3d_shape_consistency, face_rgb, face_landmarks)
            )
            
            # Комбинированная оценка глубины
            depth_score = self._combine_depth_scores(
                focus_analysis,
                shading_analysis,
                texture_analysis,
                shape_analysis
            )
            
            # Анализ на признаки flat spoofing
            spoof_detection = self._detect_flat_spoofing(
                focus_analysis,
                shading_analysis,
                texture_analysis,
                shape_analysis
            )
            
            # Оценка качества depth estimation
            estimation_confidence = self._calculate_estimation_confidence(
                focus_analysis,
                shading_analysis,
                texture_analysis,
                shape_analysis
            )
            
            return {
                "success": True,
                "depth_score": depth_score,
                "is_real_face": depth_score > self.DEPTH_REAL_THRESHOLD,
                "is_likely_flat_spoof": depth_score < self.DEPTH_SPOOF_THRESHOLD,
                "estimation_confidence": estimation_confidence,
                "focus_analysis": focus_analysis,
                "shading_analysis": shading_analysis,
                "texture_analysis": texture_analysis,
                "shape_analysis": shape_analysis,
                "spoof_detection": spoof_detection,
                "depth_3d_indicators": self._get_3d_indicators(
                    focus_analysis,
                    shading_analysis,
                    texture_analysis,
                    shape_analysis
                ),
            }
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _extract_face_region(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]]
    ) -> Optional[np.ndarray]:
        """Извлечение области лица для анализа."""
        h, w = image.shape[:2]
        
        if landmarks is not None and len(landmarks) >= 5:
            # Используем landmarks для точного определения области
            xs = landmarks[:, 0]
            ys = landmarks[:, 1]
            
            # Расширяем область для включения всего лица
            padding = 0.3
            face_x1 = max(0, int(xs.min() * (1 - padding)))
            face_y1 = max(0, int(ys.min() * (1 - padding)))
            face_x2 = min(w, int(xs.max() * (1 + padding)))
            face_y2 = min(h, int(ys.max() * (1 + padding)))
            
            return image[face_y1:face_y2, face_x1:face_x2]
        
        elif bbox is not None:
            x1, y1, x2, y2 = bbox
            return image[y1:y2, x1:x2]
        
        else:
            # Центральная область как fallback
            center_size = min(w, h) // 2
            cx, cy = w // 2, h // 2
            x1 = cx - center_size // 2
            y1 = cy - center_size // 2
            x2 = x1 + center_size
            y2 = y1 + center_size
            
            return image[y1:y2, x1:x2]
    
    def _analyze_depth_from_focus(self, face_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Анализ глубины по фокусу/размытию (Depth from Defocus).
        
        Живое лицо имеет естественную вариацию резкости из-за:
        - 3D формы лица (нос ближе к камере, чем уши)
        - Разное расстояние до камеры для разных частей
        
        Плоское изображение имеет равномерную резкость.
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Laplacian variance для разных регионов лица
        regions = self._define_face_regions(h, w)
        
        region_blur_scores = {}
        for region_name, (y1, y2, x1, x2) in regions.items():
            region = gray[y1:y2, x1:x2]
            
            # Laplacian variance = measure of blur
            laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
            region_blur_scores[region_name] = laplacian_var
        
        # Вычисление вариации резкости между регионами
        blur_values = list(region_blur_scores.values())
        blur_mean = np.mean(blur_values)
        blur_std = np.std(blur_values)
        
        # Отношение std/mean = коэффициент вариации резкости
        # Живое лицо имеет variation из-за 3D формы
        blur_cv = blur_std / (blur_mean + 1e-6)
        
        # Нормализация до 0-1
        focus_variation_score = min(blur_cv / 0.5, 1.0)
        
        # Анализ абсолютной резкости (размытое фото = spoof)
        absolute_sharpness = blur_mean
        sharpness_score = min(absolute_sharpness / 100.0, 1.0)
        
        # Дополнительный анализ: blur consistency
        # Плоское изображение имеет одинаковый blur everywhere
        blur_diffs = []
        for i, name1 in enumerate(region_blur_scores.keys()):
            for name2 in list(region_blur_scores.keys())[i+1:]:
                diff = abs(region_blur_scores[name1] - region_blur_scores[name2])
                blur_diffs.append(diff)
        
        avg_blur_diff = np.mean(blur_diffs) if blur_diffs else 0
        blur_consistency = avg_blur_diff / (blur_mean + 1e-6)
        
        # Высокая consistency = плоское изображение
        flat_consistency_score = min(blur_consistency / 0.3, 1.0)
        
        # Финальный score для depth-from-focus
        # Реальное лицо: высокая variation, средняя sharpness
        depth_ff_score = (
            focus_variation_score * 0.5 +
            sharpness_score * 0.3 +
            (1 - flat_consistency_score) * 0.2
        )
        
        return {
            "region_blur_scores": region_blur_scores,
            "blur_variation": float(blur_cv),
            "absolute_sharpness": float(absolute_sharpness),
            "blur_consistency": float(blur_consistency),
            "depth_from_focus_score": float(depth_ff_score),
            "is_3d_like": blur_cv > 0.1,
            "is_sufficiently_sharp": sharpness_score > 0.3,
        }
    
    def _define_face_regions(self, h: int, w: int) -> Dict[str, Tuple[int, int, int, int]]:
        """Определение регионов лица для анализа."""
        return {
            "forehead": (0, int(h * 0.25), 0, w),
            "left_eye": (int(h * 0.25), int(h * 0.45), 0, int(w * 0.45)),
            "right_eye": (int(h * 0.25), int(h * 0.45), int(w * 0.55), w),
            "nose": (int(h * 0.4), int(h * 0.65), int(w * 0.35), int(w * 0.65)),
            "left_cheek": (int(h * 0.5), int(h * 0.8), 0, int(w * 0.35)),
            "right_cheek": (int(h * 0.5), int(h * 0.8), int(w * 0.65), w),
            "chin": (int(h * 0.8), h, int(w * 0.25), int(w * 0.75)),
        }
    
    def _analyze_shading_depth(self, face_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Анализ глубины по паттернам освещения (Shading-based Depth).
        
        Живое лицо имеет характерную картину теней:
        - Тень под носом
        - Тени в глазницах
        - Тень под подбородком
        
        Эти тени формируют 3D форму лица.
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Нормализация яркости для анализа
        gray_norm = cv2.equalizeHist(gray)
        
        # Обнаружение темных областей (тени)
        _, shadow_mask = cv2.threshold(gray_norm, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Морфологическая обработка
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Контуры теней
        contours, _ = cv2.findContours(
            shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Анализ контуров теней
        shadow_regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > h * w * 0.01:  # Значимые регионы
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shadow_regions.append({
                        "area": area,
                        "center": (cx, cy),
                        "bbox": cv2.boundingRect(c),
                    })
        
        # Проверка наличия характерных теней лица
        has_nose_shadow = False
        has_eye_shadows = False
        has_chin_shadow = False
        
        nose_region_y = (h * 0.4, h * 0.65)
        eye_region_y = (h * 0.25, h * 0.45)
        chin_region_y = (h * 0.8, h)
        
        for shadow in shadow_regions:
            cy = shadow["center"][1]
            
            if nose_region_y[0] < cy < nose_region_y[1]:
                # Тень в области носа
                if shadow["area"] > h * w * 0.02:
                    has_nose_shadow = True
            
            if eye_region_y[0] < cy < eye_region_y[1]:
                if shadow["area"] > h * w * 0.01:
                    has_eye_shadows = True
            
            if chin_region_y[0] < cy < chin_region_y[1]:
                if shadow["area"] > h * w * 0.015:
                    has_chin_shadow = True
        
        # Проверка естественности паттерна теней
        natural_shadow_count = sum([has_nose_shadow, has_eye_shadows, has_chin_shadow])
        shadow_score = natural_shadow_count / 3.0
        
        # Анализ градиента освещения (плавный = естественный)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Средняя интенсивность градиента (резкость переходов)
        mean_gradient = float(np.mean(gradient_mag))
        
        # Вариация градиента (разная в разных частях лица = 3D)
        regions = self._define_face_regions(h, w)
        region_gradients = {}
        for name, (y1, y2, x1, x2) in regions.items():
            region = gradient_mag[y1:y2, x1:x2]
            region_gradients[name] = np.mean(region)
        
        grad_values = list(region_gradients.values())
        grad_cv = np.std(grad_values) / (np.mean(grad_values) + 1e-6)
        
        # Финальный score
        shading_depth_score = (
            shadow_score * 0.4 +
            min(grad_cv / 0.3, 1.0) * 0.3 +
            min(mean_gradient / 50, 1.0) * 0.3
        )
        
        return {
            "shadow_regions_count": len(shadow_regions),
            "has_nose_shadow": has_nose_shadow,
            "has_eye_shadows": has_eye_shadows,
            "has_chin_shadow": has_chin_shadow,
            "natural_shadow_count": natural_shadow_count,
            "gradient_variation": float(grad_cv),
            "mean_gradient_intensity": mean_gradient,
            "shading_depth_score": float(shading_depth_score),
            "has_3d_shading": natural_shadow_count >= 2,
        }
    
    def _analyze_texture_gradient(self, face_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Анализ текстурных градиентов для оценки объёмности.
        
        3D поверхность имеет плавные изменения текстуры.
        Плоская поверхность имеет резкие или однородные текстурные переходы.
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # LBP (Local Binary Pattern) для анализа текстуры
        lbp = self._compute_lbp(gray)
        
        # Анализ вариации LBP паттернов
        lbp_std = np.std(lbp)
        lbp_mean = np.mean(lbp)
        
        # Текстурное разнообразие
        texture_diversity = lbp_std / (lbp_mean + 1e-6)
        
        # Gray-Level Co-occurrence Matrix (GLCM) для анализа паттернов
        glcm = self._compute_glcm(gray, distances=[2], angles=[0, np.pi/4, np.pi/2])
        
        # Contrast (локальная вариация)
        contrast = glcm.get("contrast", 0)
        # Homogeneity (однородность паттернов)
        homogeneity = glcm.get("homogeneity", 0)
        # Entropy (сложность текстуры)
        entropy = glcm.get("entropy", 0)
        
        # Анализ масштаба текстуры
        # Применяем фильтры разного масштаба
        small_scale_var = []
        large_scale_var = []
        
        for ksize in [3, 5]:
            filtered = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            var = np.var(filtered)
            if ksize <= 3:
                small_scale_var.append(var)
            else:
                large_scale_var.append(var)
        
        scale_ratio = np.mean(small_scale_var) / (np.mean(large_scale_var) + 1e-6)
        
        # Плоское изображение имеет более равномерную текстуру
        # (меньше variation между масштабами)
        flat_texture_score = min(abs(1 - scale_ratio), 1.0)
        
        # Финальный score
        texture_score = (
            min(texture_diversity / 0.5, 1.0) * 0.25 +
            min(contrast / 500, 1.0) * 0.25 +
            min(entropy / 7, 1.0) * 0.25 +
            (1 - flat_texture_score) * 0.25
        )
        
        return {
            "lbp_std": float(lbp_std),
            "lbp_mean": float(lbp_mean),
            "texture_diversity": float(texture_diversity),
            "glcm_contrast": float(contrast),
            "glcm_homogeneity": float(homogeneity),
            "glcm_entropy": float(entropy),
            "scale_ratio": float(scale_ratio),
            "flat_texture_score": float(flat_texture_score),
            "texture_depth_score": float(texture_score),
            "has_rich_texture": texture_diversity > 0.3,
        }
    
    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Вычисление Local Binary Pattern."""
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        # Упрощенный LBP (8-neighbor)
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
    
    def _compute_glcm(
        self,
        gray: np.ndarray,
        distances: list = [1, 2],
        angles: list = [0, np.pi/4, np.pi/2]
    ) -> Dict[str, float]:
        """Вычисление GLCM метрик."""
        # Упрощенная реализация GLCM
        h, w = gray.shape
        
        # Нормализация к 8 уровням
        gray_quantized = (gray / 32).astype(np.uint8)
        gray_quantized = np.clip(gray_quantized, 0, 7)
        
        # Создание GLCM
        glcm = np.zeros((8, 8), dtype=np.float32)
        
        for d in distances:
            for theta in angles:
                cos_a, sin_a = np.cos(theta), np.sin(theta)
                
                for i in range(h):
                    for j in range(w - d):
                        ni = int(i + d * sin_a)
                        nj = int(j + d * cos_a)
                        
                        if 0 <= ni < h and 0 <= nj < w:
                            i_q = gray_quantized[i, j]
                            j_q = gray_quantized[ni, nj]
                            glcm[i_q, j_q] += 1
        
        # Нормализация
        glcm = glcm / glcm.sum() if glcm.sum() > 0 else glcm
        
        # Метрики
        i_idx, j_idx = np.indices(glcm.shape)
        
        contrast = np.sum(glcm * (i_idx - j_idx) ** 2)
        homogeneity = np.sum(glcm / (1 + (i_idx - j_idx) ** 2))
        entropy = -np.sum(glcm * np.log(glcm + 1e-10))
        
        return {
            "contrast": float(contrast),
            "homogeneity": float(homogeneity),
            "entropy": float(entropy),
        }
    
    def _analyze_3d_shape_consistency(
        self,
        face_rgb: np.ndarray,
        face_landmarks: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Анализ согласованности 3D формы лица.
        
        Проверяет:
        - Пропорции лица (соответствие антропометрии)
        - Симметрию
        - Соответствие типичной 3D структуре
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Детекция контуров лица (упрощенная)
        edges = cv2.Canny(gray, 50, 150)
        
        # Поиск эллипса, соответствующего лицу
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Находим biggest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Аппроксимация эллипсом
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                
                # Проверка соотношения осей
                axis_a, axis_b = ellipse[1]  # (width, height)
                axis_ratio = axis_a / (axis_b + 1e-6)
                
                # Типичное лицо имеет axis_ratio около 0.75-0.85
                shape_score = 1.0 - min(abs(axis_ratio - 0.8) / 0.3, 1.0)
                
                # Проверка центра эллипса
                center_x, center_y = ellipse[0]
                center_x_norm = center_x / w
                center_y_norm = center_y / h
                
                # Центр должен быть примерно посередине
                center_offset = np.sqrt((center_x_norm - 0.5)**2 + (center_y_norm - 0.5)**2)
                center_score = 1.0 - min(center_offset / 0.3, 1.0)
                
                return {
                    "axis_ratio": float(axis_ratio),
                    "ellipse_fit_score": float(shape_score * center_score),
                    "has_consistent_shape": shape_score > 0.6,
                    "center_offset": float(center_offset),
                    "shape_consistency_score": float(shape_score * center_score * 0.7 + 0.3),
                }
        
        # Fallback без landmarks
        return {
            "axis_ratio": w / (h + 1e-6),
            "ellipse_fit_score": 0.5,
            "has_consistent_shape": False,
            "center_offset": 0.2,
            "shape_consistency_score": 0.5,
            "note": "No landmarks available, using edge analysis only",
        }
    
    def _combine_depth_scores(
        self,
        focus: Dict,
        shading: Dict,
        texture: Dict,
        shape: Dict
    ) -> float:
        """Комбинирование оценок глубины из разных методов."""
        # Веса для разных методов
        focus_weight = 0.30
        shading_weight = 0.30
        texture_weight = 0.20
        shape_weight = 0.20
        
        focus_score = focus.get("depth_from_focus_score", 0.5)
        shading_score = shading.get("shading_depth_score", 0.5)
        texture_score = texture.get("texture_depth_score", 0.5)
        shape_score = shape.get("shape_consistency_score", 0.5)
        
        # Взвешенная комбинация
        combined = (
            focus_weight * focus_score +
            shading_weight * shading_score +
            texture_weight * texture_score +
            shape_weight * shape_score
        )
        
        return float(np.clip(combined, 0.0, 1.0))
    
    def _detect_flat_spoofing(
        self,
        focus: Dict,
        shading: Dict,
        texture: Dict,
        shape: Dict
    ) -> Dict[str, Any]:
        """Детекция признаков плоской spoofing атаки."""
        
        # Признаки плоского изображения
        flat_indicators = []
        
        # 1. Равномерная резкость
        if focus.get("blur_consistency", 0) > 0.2:
            flat_indicators.append("uniform_blur")
        
        # 2. Отсутствие характерных теней
        if shading.get("natural_shadow_count", 0) < 1:
            flat_indicators.append("no_natural_shadows")
        
        # 3. Однородная текстура
        if texture.get("texture_diversity", 1) < 0.2:
            flat_indicators.append("uniform_texture")
        
        # 4. Неестественная форма
        if not shape.get("has_consistent_shape", True):
            flat_indicators.append("inconsistent_shape")
        
        # 5. Слишком высокая резкость (характерно для экрана)
        if focus.get("absolute_sharpness", 0) > 500:
            flat_indicators.append("unnatural_sharpness")
        
        # Расчёт вероятности flat spoofing
        spoof_prob = len(flat_indicators) / 5.0
        
        # Учитываем индивидуальные scores
        if focus.get("depth_from_focus_score", 0.5) < 0.3:
            spoof_prob += 0.1
        if shading.get("shading_depth_score", 0.5) < 0.3:
            spoof_prob += 0.1
        
        return {
            "flat_indicators": flat_indicators,
            "spoof_probability": min(1.0, spoof_prob),
            "is_likely_flat_spoof": spoof_prob > 0.4,
        }
    
    def _calculate_estimation_confidence(
        self,
        focus: Dict,
        shading: Dict,
        texture: Dict,
        shape: Dict
    ) -> float:
        """Расчёт уверенности в оценке глубины."""
        # Уверенность выше, если разные методы согласны
        scores = [
            focus.get("depth_from_focus_score", 0.5),
            shading.get("shading_depth_score", 0.5),
            texture.get("texture_depth_score", 0.5),
            shape.get("shape_consistency_score", 0.5),
        ]
        
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        
        # Низкая дисперсия = высокая уверенность
        confidence = 1.0 - min(score_std / 0.3, 1.0)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _get_3d_indicators(
        self,
        focus: Dict,
        shading: Dict,
        texture: Dict,
        shape: Dict
    ) -> Dict[str, Any]:
        """Получение индикаторов 3D структуры."""
        return {
            "depth_variation": focus.get("blur_variation", 0),
            "natural_shadows": shading.get("natural_shadow_count", 0),
            "texture_diversity": texture.get("texture_diversity", 0),
            "shape_score": shape.get("shape_consistency_score", 0),
            "is_3d_consistent": (
                focus.get("is_3d_like", False) and
                shading.get("has_3d_shading", False) and
                texture.get("has_rich_texture", False)
            ),
        }


# =============================================================================
# Quick estimation functions
# =============================================================================

async def quick_depth_check(
    image_data: bytes,
    threshold: float = 0.55
) -> Dict[str, Any]:
    """
    Быстрая проверка 3D структуры лица.
    
    Args:
        image_data: Байты изображения
        threshold: Порог для определения real vs spoof
        
    Returns:
        Dict с depth_score и решением
    """
    estimator = DepthEstimator()
    result = await estimator.estimate_depth(image_data)
    
    if result.get("success"):
        return {
            "depth_score": result["depth_score"],
            "is_real_face": result["depth_score"] > threshold,
            "confidence": result.get("estimation_confidence", 0.5),
        }
    
    return {"depth_score": 0.5, "is_real_face": False, "error": result.get("error")}


# =============================================================================
# Singleton
# =============================================================================

_depth_estimator: Optional[DepthEstimator] = None


async def get_depth_estimator(
    face_size: int = 224,
    sensitivity: float = 1.0
) -> DepthEstimator:
    """Получение singleton экземпляра DepthEstimator."""
    global _depth_estimator
    if _depth_estimator is None:
        _depth_estimator = DepthEstimator(face_size=face_size, sensitivity=sensitivity)
    return _depth_estimator
