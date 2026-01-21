"""
Утилита для выравнивания лица (Face Alignment).
Выравнивает лицо по ключевым точкам (landmarks) для улучшения качества эмбеддингов.

Использует MTCNN для получения landmarks и аффинное преобразование
для выравнивания по центрам глаз.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import asyncio
import torch
from facenet_pytorch import MTCNN

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FaceAligner:
    """
    Выравнивание лица по ключевым точкам.

    Алгоритм:
    1. Детекция лица и landmarks с помощью MTCNN
    2. Определение центров левого и правого глаза
    3. Аффинное преобразование для выравнивания
    4. Кроп и ресайз к целевому размеру

    Преимущества:
    - Улучшает качество эмбеддингов при нецентрированных лицах
    - Нормализует поворот лица
    - Повышает точность верификации
    """

    # Стандартные координаты для целевого выравнивания
    # Используем типичное расположение глаз для VGGFace2/FaceNet
    LEFT_EYE_TARGET = (0.35, 0.4)  # Относительные координаты в выходном изображении
    RIGHT_EYE_TARGET = (0.65, 0.4)

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        landmarks_enabled: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            output_size: Размер выходного изображения (width, height)
            landmarks_enabled: Использовать landmarks для точного выравнивания
            device: Устройство для вычислений ('cpu' или 'cuda')
        """
        self.output_size = output_size
        self.landmarks_enabled = landmarks_enabled
        self.device = torch.device(device)

        # MTCNN для получения landmarks
        self.mtcnn = MTCNN(
            image_size=min(output_size),
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            device=self.device,
            select_largest=False,
        )

        logger.info(f"FaceAligner initialized with output_size: {output_size}")

    async def align_face(
        self, image: Image.Image, return_landmarks: bool = False
    ) -> Dict[str, Any]:
        """
        Выравнивание лица на изображении.

        Args:
            image: PIL Image (RGB)
            return_landmarks: Вернуть ли coordinates landmarks

        Returns:
            Dict с выравненным изображением и метаданными
        """
        try:
            # Детекция лица и landmarks
            face_data = await asyncio.to_thread(self._detect_face_with_landmarks, image)

            if face_data is None or face_data["landmarks"] is None:
                logger.warning("No face or landmarks detected for alignment")
                return {
                    "aligned_image": None,
                    "success": False,
                    "error": "No face detected",
                }

            landmarks = face_data["landmarks"]
            face_crop = face_data["face"]
            bounding_box = face_data["box"]

            # Выравнивание по глазам
            aligned_image = await asyncio.to_thread(
                self._apply_alignment, image, landmarks
            )

            # Конвертация обратно в numpy для возврата
            aligned_np = np.array(aligned_image)

            # Вычисление углов выравнивания
            alignment_angle = self._calculate_alignment_angle(landmarks)

            result = {
                "aligned_image": aligned_np,
                "success": True,
                "face_crop": face_crop,  # Оригинальный кроп без выравнивания
                "bounding_box": bounding_box,
                "landmarks": landmarks,
                "alignment_angle": alignment_angle,
                "alignment_applied": True,
            }

            if return_landmarks:
                result["landmarks_coordinates"] = self._get_landmarks_coords(landmarks)

            logger.debug(f"Face aligned successfully, angle: {alignment_angle:.2f}°")

            return result

        except Exception as e:
            logger.error(f"Face alignment failed: {str(e)}")
            return {
                "aligned_image": None,
                "success": False,
                "error": str(e),
            }

    def _detect_face_with_landmarks(
        self, image: Image.Image
    ) -> Optional[Dict[str, Any]]:
        """Детекция лица и получение landmarks."""
        try:
            # MTCNN возвращает (boxes, probs, landmarks)
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)

            if boxes is None or len(boxes) == 0:
                return None

            # Берем лицо с наибольшей вероятностью
            best_idx = np.argmax(probs) if probs is not None else 0

            box = boxes[best_idx].astype(int)
            prob = probs[best_idx] if probs is not None else 1.0
            lm = landmarks[best_idx] if landmarks is not None else None

            # Извлекаем кроп лица
            face = self.mtcnn.extract(image, [box], save_path=None)

            return {
                "box": box.tolist(),
                "probability": float(prob),
                "landmarks": lm,
                "face": face[0] if face is not None else None,
            }

        except Exception as e:
            logger.error(f"Face detection with landmarks failed: {str(e)}")
            return None

    def _apply_alignment(
        self, image: Image.Image, landmarks: np.ndarray
    ) -> Image.Image:
        """
        Применение аффинного преобразования для выравнивания лица.

        Args:
            image: Оригинальное PIL изображение
            landmarks: Массив landmarks (5 точек: left_eye, right_eye, nose, left_mouth, right_mouth)

        Returns:
            Выравненное PIL изображение
        """
        if landmarks is None or len(landmarks) < 5:
            return image

        img = np.array(image)
        h, w = img.shape[:2]

        # Координаты левого и правого глаза в исходном изображении
        # MTCNN landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Целевые координаты глаз в выходном изображении
        target_left_eye = (
            self.LEFT_EYE_TARGET[0] * self.output_size[0],
            self.LEFT_EYE_TARGET[1] * self.output_size[1],
        )
        target_right_eye = (
            self.RIGHT_EYE_TARGET[0] * self.output_size[0],
            self.RIGHT_EYE_TARGET[1] * self.output_size[1],
        )

        # Вычисление угла поворота
        angle = self._calculate_rotation_angle(left_eye, right_eye)

        # Масштаб (поддерживаем соотношение сторон)
        scale = 1.0

        # Центр изображения для поворота
        center = (w / 2, h / 2)

        # Матрица аффинного преобразования
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # Корректировка для выравнивания глаз
        # Находим точку центра между глазами
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )
        target_eye_center = (
            (target_left_eye[0] + target_right_eye[0]) / 2,
            (target_left_eye[1] + target_right_eye[1]) / 2,
        )

        # Смещение для центрирования глаз
        dx = target_eye_center[0] - eye_center[0]
        dy = target_eye_center[1] - eye_center[1]

        rot_matrix[0, 2] += dx
        rot_matrix[1, 2] += dy

        # Применяем преобразование
        aligned = cv2.warpAffine(
            img,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(127, 127, 127),
        )

        # Кроп области лица с запасом
        face_size = min(w, h)
        margin = int(face_size * 0.3)

        x1 = max(0, int(eye_center[0] - face_size / 2 - margin))
        y1 = max(0, int(eye_center[1] - face_size / 2 - margin))
        x2 = min(w, int(eye_center[0] + face_size / 2 + margin))
        y2 = min(h, int(eye_center[1] + face_size / 2 + margin))

        face_aligned = aligned[y1:y2, x1:x2]

        # Ресайз к целевому размеру
        face_resized = cv2.resize(face_aligned, self.output_size)

        return Image.fromarray(face_resized.astype(np.uint8))

    def _calculate_rotation_angle(
        self, left_eye: Tuple[float, float], right_eye: Tuple[float, float]
    ) -> float:
        """Вычисление угла поворота по положению глаз."""
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def _calculate_alignment_angle(self, landmarks: np.ndarray) -> float:
        """Вычисление примененного угла выравнивания."""
        if landmarks is None or len(landmarks) < 2:
            return 0.0

        left_eye = landmarks[0]
        right_eye = landmarks[1]

        return self._calculate_rotation_angle(left_eye, right_eye)

    def _get_landmarks_coords(
        self, landmarks: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Получение именованных координат landmarks."""
        names = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
        return {
            name: (float(landmarks[i][0]), float(landmarks[i][1]))
            for i, name in enumerate(names)
            if i < len(landmarks)
        }

    async def batch_align(self, images: list[Image.Image]) -> list[Dict[str, Any]]:
        """
        Пакетное выравнивание лиц.

        Args:
            images: Список PIL изображений

        Returns:
            Список результатов для каждого изображения
        """
        results = []
        for image in images:
            result = await self.align_face(image)
            results.append(result)
        return results

    def get_alignment_info(self) -> Dict[str, Any]:
        """Получение информации о параметрах выравнивания."""
        return {
            "output_size": self.output_size,
            "landmarks_enabled": self.landmarks_enabled,
            "left_eye_target": self.LEFT_EYE_TARGET,
            "right_eye_target": self.RIGHT_EYE_TARGET,
        }


# =============================================================================
# Вспомогательные функции для интеграции с MTCNN
# =============================================================================


async def extract_aligned_face(
    image: Image.Image, output_size: Tuple[int, int] = (224, 224), device: str = "cpu"
) -> Optional[np.ndarray]:
    """
    Удобная функция для быстрого выравнивания лица.

    Args:
        image: PIL Image (RGB)
        output_size: Размер выходного изображения
        device: Устройство для вычислений

    Returns:
        numpy array выравненного лица или None
    """
    aligner = FaceAligner(output_size=output_size, device=device)
    result = await aligner.align_face(image)

    if result["success"] and result["aligned_image"] is not None:
        return result["aligned_image"]
    return None


# =============================================================================
# Singleton
# =============================================================================

_aligner: Optional[FaceAligner] = None


async def get_face_aligner(
    output_size: Tuple[int, int] = (224, 224), device: str = "cpu"
) -> FaceAligner:
    """Получение singleton экземпляра FaceAligner."""
    global _aligner
    if _aligner is None:
        _aligner = FaceAligner(output_size=output_size, device=device)
    return _aligner
