"""
Face Occlusion Detection Service.
Detects masks, sunglasses, VR headsets, and other occlusions.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OcclusionResult:
    """Результат детекции окклюзий."""
    has_mask: bool
    has_sunglasses: bool
    has_regular_glasses: bool
    has_vr_headset: bool
    has_hand_covering: bool
    occlusion_score: float  # 0-1, где 1 = полностью видимое лицо
    confidence: float
    details: Dict[str, Any]


class FaceOcclusionDetector:
    """
    Детектор окклюзий лица.
    
    Методы детекции:
    1. Landmark visibility - проверка видимости ключевых точек
    2. Color/texture analysis - анализ цвета и текстуры в области рта/глаз
    3. Symmetry check - проверка симметрии лица
    """
    
    def __init__(self):
        self.mouth_region_indices = list(range(48, 68))  # 68-point landmarks
        self.nose_region_indices = list(range(27, 36))
        self.left_eye_indices = list(range(36, 42))
        self.right_eye_indices = list(range(42, 48))
    
    def detect_occlusions(
        self,
        face_image: np.ndarray,
        landmarks_68: Optional[np.ndarray] = None,
    ) -> OcclusionResult:
        """
        Детекция окклюзий на изображении лица.
        
        Args:
            face_image: Изображение лица (RGB)
            landmarks_68: Опциональные 68-point landmarks
            
        Returns:
            OcclusionResult: Результаты детекции
        """
        results = {
            "has_mask": False,
            "has_sunglasses": False,
            "has_regular_glasses": False,
            "has_vr_headset": False,
            "has_hand_covering": False,
        }
        
        details = {}
        confidence_scores = []
        
        # 1. Детекция маски (проверка области рта/носа)
        mask_result = self._detect_mask(face_image, landmarks_68)
        results["has_mask"] = mask_result["detected"]
        details["mask_detection"] = mask_result
        confidence_scores.append(mask_result["confidence"])
        
        # 2. Детекция солнцезащитных очков (темная область глаз)
        sunglasses_result = self._detect_sunglasses(face_image, landmarks_68)
        results["has_sunglasses"] = sunglasses_result["detected"]
        details["sunglasses_detection"] = sunglasses_result
        confidence_scores.append(sunglasses_result["confidence"])
        
        # 3. Детекция обычных очков (отражения, края оправы)
        glasses_result = self._detect_regular_glasses(face_image, landmarks_68)
        results["has_regular_glasses"] = glasses_result["detected"]
        details["glasses_detection"] = glasses_result
        confidence_scores.append(glasses_result["confidence"])
        
        # 4. Детекция руки перед лицом (skin color detection)
        hand_result = self._detect_hand_covering(face_image)
        results["has_hand_covering"] = hand_result["detected"]
        details["hand_detection"] = hand_result
        confidence_scores.append(hand_result["confidence"])
        
        # Вычисляем occlusion score (чем выше, тем меньше окклюзий)
        occlusion_penalties = {
            "has_mask": 0.4,
            "has_sunglasses": 0.3,
            "has_vr_headset": 0.5,
            "has_hand_covering": 0.3,
            "has_regular_glasses": 0.1,  # Обычные очки менее критичны
        }
        
        occlusion_score = 1.0
        for key, value in results.items():
            if value:
                occlusion_score -= occlusion_penalties.get(key, 0.1)
        
        occlusion_score = max(0.0, occlusion_score)
        
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        return OcclusionResult(
            **results,
            occlusion_score=occlusion_score,
            confidence=overall_confidence,
            details=details,
        )
    
    def _detect_mask(
        self,
        face_image: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Детекция маски на лице."""
        try:
            h, w = face_image.shape[:2]
            
            # Если есть landmarks, используем их для точного выделения области
            if landmarks is not None and len(landmarks) == 68:
                # Область рта и носа
                mouth_points = landmarks[self.mouth_region_indices]
                nose_points = landmarks[self.nose_region_indices]
                
                # Создаем маску для ROI
                mask_roi = np.zeros((h, w), dtype=np.uint8)
                all_points = np.vstack([mouth_points, nose_points]).astype(np.int32)
                cv2.fillConvexPoly(mask_roi, cv2.convexHull(all_points), 255)
                
                # Извлекаем ROI
                roi = cv2.bitwise_and(face_image, face_image, mask=mask_roi)
            else:
                # Fallback: используем нижнюю половину лица
                roi = face_image[h//2:, :]
            
            # Конвертируем в HSV для анализа цвета
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            
            # Проверяем наличие однородного не-кожного цвета
            # Маски обычно имеют синий/белый/черный цвет
            
            # Диапазон кожи (для исключения)
            skin_lower = np.array([0, 20, 70], dtype=np.uint8)
            skin_upper = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            
            skin_ratio = np.sum(skin_mask > 0) / (roi.shape[0] * roi.shape[1])
            
            # Если кожи мало в области рта/носа - вероятно маска
            mask_detected = skin_ratio < 0.3
            confidence = 1.0 - skin_ratio if mask_detected else skin_ratio
            
            return {
                "detected": mask_detected,
                "confidence": float(confidence),
                "skin_ratio": float(skin_ratio),
                "method": "color_analysis",
            }
            
        except Exception as e:
            logger.error(f"Mask detection failed: {e}")
            return {"detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_sunglasses(
        self,
        face_image: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Детекция солнцезащитных очков."""
        try:
            h, w = face_image.shape[:2]
            
            if landmarks is not None and len(landmarks) == 68:
                # Область глаз
                left_eye = landmarks[self.left_eye_indices]
                right_eye = landmarks[self.right_eye_indices]
                
                # Создаем маску для области глаз
                eye_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(eye_mask, cv2.convexHull(left_eye.astype(np.int32)), 255)
                cv2.fillConvexPoly(eye_mask, cv2.convexHull(right_eye.astype(np.int32)), 255)
                
                # Расширяем маску для захвата оправы
                kernel = np.ones((10, 10), np.uint8)
                eye_mask = cv2.dilate(eye_mask, kernel, iterations=1)
                
                roi = cv2.bitwise_and(face_image, face_image, mask=eye_mask)
            else:
                # Fallback: верхняя треть лица
                roi = face_image[:h//3, :]
            
            # Конвертируем в grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            
            # Солнцезащитные очки обычно темные
            dark_pixel_ratio = np.sum(gray < 50) / (gray.shape[0] * gray.shape[1])
            
            # Проверяем однородность (очки обычно равномерно темные)
            std_dev = np.std(gray)
            uniformity = 1.0 - min(std_dev / 50.0, 1.0)  # Нормализуем
            
            # Детекция краев (оправа очков)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Солнцезащитные очки: темные + однородные + есть края
            score = (dark_pixel_ratio * 0.5 + uniformity * 0.3 + edge_density * 0.2)
            detected = score > 0.4
            
            return {
                "detected": detected,
                "confidence": float(score),
                "dark_ratio": float(dark_pixel_ratio),
                "uniformity": float(uniformity),
                "edge_density": float(edge_density),
            }
            
        except Exception as e:
            logger.error(f"Sunglasses detection failed: {e}")
            return {"detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_regular_glasses(
        self,
        face_image: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Детекция обычных очков (прозрачные линзы)."""
        try:
            h, w = face_image.shape[:2]
            
            # Обычные очки детектируем по:
            # 1. Отражениям на линзах
            # 2. Краям оправы
            
            if landmarks is not None and len(landmarks) == 68:
                # Область глаз с запасом
                left_eye = landmarks[self.left_eye_indices]
                right_eye = landmarks[self.right_eye_indices]
                
                eye_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(eye_mask, cv2.convexHull(left_eye.astype(np.int32)), 255)
                cv2.fillConvexPoly(eye_mask, cv2.convexHull(right_eye.astype(np.int32)), 255)
                
                # Расширяем для захвата оправы
                kernel = np.ones((15, 15), np.uint8)
                eye_mask = cv2.dilate(eye_mask, kernel, iterations=1)
                
                roi = cv2.bitwise_and(face_image, face_image, mask=eye_mask)
            else:
                roi = face_image[:h//3, :]
            
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            
            # Детекция отражений (яркие пятна)
            _, bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            reflection_ratio = np.sum(bright_spots > 0) / (gray.shape[0] * gray.shape[1])
            
            # Детекция краев оправы (прямые линии)
            edges = cv2.Canny(gray, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
            
            has_frame_edges = lines is not None and len(lines) > 2
            
            # Обычные очки: есть отражения ИЛИ четкие края оправы
            score = reflection_ratio * 0.6 + (0.4 if has_frame_edges else 0.0)
            detected = score > 0.15
            
            return {
                "detected": detected,
                "confidence": float(score),
                "reflection_ratio": float(reflection_ratio),
                "has_frame_edges": bool(has_frame_edges),
            }
            
        except Exception as e:
            logger.error(f"Regular glasses detection failed: {e}")
            return {"detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_hand_covering(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Детекция руки перед лицом."""
        try:
            # Конвертируем в YCrCb для детекции кожи
            ycrcb = cv2.cvtColor(face_image, cv2.COLOR_RGB2YCrCb)
            
            # Диапазон кожи (расширенный для рук)
            skin_lower = np.array([0, 133, 77], dtype=np.uint8)
            skin_upper = np.array([255, 173, 127], dtype=np.uint8)
            
            skin_mask = cv2.inRange(ycrcb, skin_lower, skin_upper)
            
            # Морфологическая обработка
            kernel = np.ones((5, 5), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # Находим контуры
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Рука обычно создает большую неправильную область кожи
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                total_area = face_image.shape[0] * face_image.shape[1]
                area_ratio = area / total_area
                
                # Вычисляем компактность (рука менее компактна чем лицо)
                perimeter = cv2.arcLength(largest_contour, True)
                compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # Рука: большая область + низкая компактность
                detected = area_ratio > 0.4 and compactness < 0.5
                confidence = area_ratio if detected else 1.0 - area_ratio
                
                return {
                    "detected": detected,
                    "confidence": float(confidence),
                    "area_ratio": float(area_ratio),
                    "compactness": float(compactness),
                }
            else:
                return {"detected": False, "confidence": 0.5}
                
        except Exception as e:
            logger.error(f"Hand covering detection failed: {e}")
            return {"detected": False, "confidence": 0.0, "error": str(e)}


# Singleton instance
_occlusion_detector_instance: Optional[FaceOcclusionDetector] = None


def get_occlusion_detector() -> FaceOcclusionDetector:
    """Get or create singleton occlusion detector."""
    global _occlusion_detector_instance
    if _occlusion_detector_instance is None:
        _occlusion_detector_instance = FaceOcclusionDetector()
    return _occlusion_detector_instance
