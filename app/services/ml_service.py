"""
ml_service.py
Оптимизированный локальный ML сервис для обработки изображений и распознавания лиц.

Features:
- FaceNet/InceptionResnetV1 для извлечения эмбеддингов
- MTCNN для детекции лиц
- MediaPipe для 68-point facial landmarks (вместо dlib)
- Face Alignment: выравнивание лица по landmarks
- Shadow/Lighting Analysis: анализ освещения и теней
- 3D Depth Estimation: оценка глубины для anti-spoofing
- MiniFASNetV2: сертифицированная проверка живости
"""

from __future__ import annotations

import asyncio
import io
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

try:
    from decord import VideoReader
    from decord import cpu as decord_cpu

    _HAS_DECORD = True
except Exception:
    VideoReader = None
    decord_cpu = None
    _HAS_DECORD = False

from ..config import settings
from ..utils.exceptions import MLServiceError, ProcessingError

# Импорты утилит для выравнивания, анализа освещения и depth estimation
# ✅ ВСЕ ЭТИ ФУНКЦИИ ТЕПЕРЬ ИСПОЛЬЗУЮТ MediaPipe ПОД КАПОТОМ
from ..utils.face_alignment_utils import DepthAnalysis  # Dataclass для depth
from ..utils.face_alignment_utils import FaceLandmarks  # Класс для работы с landmarks
from ..utils.face_alignment_utils import LightingAnalysis  # Dataclass для результатов
from ..utils.face_alignment_utils import align_face  # Выравнивание по landmarks
from ..utils.face_alignment_utils import (
    analyze_depth_for_liveness,  # 3D depth estimation
)
from ..utils.face_alignment_utils import (
    analyze_shadows_and_lighting,  # Анализ освещения
)
from ..utils.face_alignment_utils import (
    combine_liveness_scores,  # Комбинирование оценок
)
from ..utils.face_alignment_utils import (
    detect_face_landmarks,  # MediaPipe → 68-point landmarks
)
from ..utils.face_alignment_utils import enhance_lighting  # Улучшение освещения
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OptimizedMLService:
    """
    Оптимизированный локальный сервис для работы с машинным обучением.

    Технологии:
    - FaceNet (VGGFace2) для эмбеддингов - 99.65% accuracy
    - MTCNN для детекции лиц - fast & accurate
    - MediaPipe для 468→68 point landmarks - кросс-платформенный
    - MiniFASNetV2 для anti-spoofing - 98.5% certified

    Обновления:
    - ✅ Полностью удалена зависимость от dlib
    - ✅ Используется MediaPipe для всех landmarks
    - ✅ Автоматическая конвертация 468→68 точек для совместимости
    """

    def __init__(self):
        # Определение устройства для вычислений
        device_setting = settings.LOCAL_ML_DEVICE.lower()
        if device_setting == "auto":
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available() and settings.LOCAL_ML_ENABLE_CUDA
                else "cpu"
            )
        elif device_setting == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Initializing Optimized MLService on device: {self.device}")

        # Инициализация моделей
        self.mtcnn = None
        self.facenet = None
        self.is_initialized = False

        # Сертифицированная модель liveness (MiniFASNetV2)
        self._anti_spoofing_service = None
        self._certified_liveness_enabled = settings.USE_CERTIFIED_LIVENESS

        # Настройки из конфигурации
        self.face_detection_threshold = settings.LOCAL_ML_FACE_DETECTION_THRESHOLD
        self.quality_threshold = settings.LOCAL_ML_QUALITY_THRESHOLD
        self.batch_size = settings.LOCAL_ML_BATCH_SIZE
        self.enable_performance_monitoring = (
            settings.LOCAL_ML_ENABLE_PERFORMANCE_MONITORING
        )

        # Настройки для выравнивания и анализа
        self._alignment_enabled = True
        self._lighting_enhancement_enabled = True
        self._depth_estimation_enabled = True

        # Оптимизированная статистика
        self.stats = {
            "requests": 0,
            "face_detections": 0,
            "embeddings_generated": 0,
            "liveness_checks": 0,
            "depth_checks": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "max_processing_time": 0.0,
            "min_processing_time": float("inf"),
        }

    async def initialize(self):
        """Инициализация ML моделей."""
        if self.is_initialized:
            return

        try:
            logger.info("Initializing optimized ML models...")
            start_time = time.time()

            # Инициализация MTCNN для детекции лиц
            self.mtcnn = MTCNN(
                image_size=224,
                margin=20,
                min_face_size=50,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                device=self.device,
            )
            logger.info("MTCNN model initialized")

            # Инициализация FaceNet для извлечения эмбеддингов
            self.facenet = (
                InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            )
            logger.info("FaceNet model initialized")

            # Инициализация сертифицированной модели Anti-Spoofing
            if self._certified_liveness_enabled:
                try:
                    from .anti_spoofing_service import get_anti_spoofing_service

                    self._anti_spoofing_service = await get_anti_spoofing_service()
                    logger.info(
                        "MiniFASNetV2 Anti-Spoofing model initialized (CERTIFIED)"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize certified anti-spoofing model: {e}"
                    )
                    self._certified_liveness_enabled = False

            initialization_time = time.time() - start_time
            logger.info(f"ML models initialized in {initialization_time:.2f}s")

            self.is_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize ML models: {str(e)}")
            raise MLServiceError(f"ML model initialization failed: {str(e)}")

    async def health_check(self) -> bool:
        """Проверка состояния ML сервиса."""
        try:
            if not self.is_initialized:
                await self.initialize()

            if not torch.cuda.is_available() and self.device.type == "cuda":
                logger.warning("CUDA not available, using CPU")

            return True

        except Exception as e:
            logger.error(f"Local ML service health check failed: {str(e)}")
            return False

    def _load_image_pil(self, image_data: bytes) -> Image.Image:
        """Загрузка изображения в формате PIL."""
        try:
            image = Image.open(io.BytesIO(image_data))
            return image.convert("RGB")
        except Exception as e:
            raise ProcessingError(f"Failed to load image: {str(e)}")

    def _image_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Конвертация PIL в numpy для OpenCV анализа."""
        return np.array(image)

    async def _detect_faces_optimized(
        self, image_pil: Image.Image
    ) -> Tuple[bool, Optional[torch.Tensor], bool, int]:
        """
        Оптимизированная детекция лиц.

        Returns:
            (face_detected, face_crop, multiple_faces, faces_count)
        """
        try:
            # Один вызов MTCNN для получения кропа лица
            face_crop, prob = await asyncio.to_thread(
                self.mtcnn, image_pil, return_prob=True
            )

            if face_crop is None or prob < self.face_detection_threshold:
                return False, None, False, 0

            # Второй вызов MTCNN для определения количества лиц
            boxes, probs = await asyncio.to_thread(self.mtcnn.detect, image_pil)
            faces_count = len(boxes) if boxes is not None else 1
            multiple_faces = faces_count > 1

            return True, face_crop, multiple_faces, faces_count

        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return False, None, False, 0

    async def _assess_face_quality(self, face_crop: torch.Tensor) -> float:
        """Оценка качества лица (выполняется в отдельном потоке)."""

        def _sync_assess(face_crop_cpu: torch.Tensor) -> float:
            try:
                # Конвертация в numpy
                face_np = face_crop_cpu[0].permute(1, 2, 0).numpy()

                # Конвертация в оттенки серого
                if len(face_np.shape) == 3:
                    gray = cv2.cvtColor(
                        (face_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                    )
                else:
                    gray = (face_np * 255).astype(np.uint8)

                # Laplacian variance для резкости
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_score = min(laplacian_var / 500.0, 1.0)

                # Контрастность
                contrast = gray.std() / 128.0
                contrast_score = min(contrast, 1.0)

                # Общая оценка
                quality_score = (sharpness_score + contrast_score) / 2.0

                return max(0.0, min(1.0, quality_score))

            except Exception as e:
                logger.warning(f"Face quality assessment failed: {str(e)}")
                return 0.5

        # Переносим тензор на CPU заранее
        face_crop_cpu = face_crop.cpu()

        # Тяжёлые вычисления OpenCV — в отдельный поток
        quality_score = await asyncio.to_thread(_sync_assess, face_crop_cpu)

        return quality_score

    def _generate_face_embedding(self, face_crop: torch.Tensor) -> np.ndarray:
        """Генерация эмбеддинга лица с помощью FaceNet."""
        try:
            with torch.no_grad():
                # Перенос на устройство
                face_crop = face_crop.to(self.device)

                # Нормализация
                face_crop = (face_crop - 0.5) * 2.0

                # Генерация эмбеддинга
                embedding = self.facenet(face_crop)

                # Нормализация эмбеддинга
                embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)

                # Конвертация в numpy
                embedding_np = embedding.cpu().numpy()

                return embedding_np.flatten()

        except Exception as e:
            raise ProcessingError(f"Failed to generate face embedding: {str(e)}")

    async def generate_embedding(
        self,
        image_data: bytes,
        apply_face_alignment: bool = True,
        enhance_lighting_flag: bool = True,
        apply_depth_check: bool = True,
    ) -> Dict[str, Any]:
        """
        Генерация эмбеддинга лица из изображения с улучшенным выравниванием.

        Features:
        - Face Alignment: выравнивание лица по MediaPipe landmarks (468→68 точек)
        - Lighting Analysis: расширенный анализ освещения и теней
        - 3D Depth Estimation: оценка глубины для anti-spoofing

        Args:
            image_data: Байты изображения
            apply_face_alignment: Применить выравнивание лица (по умолчанию True)
            enhance_lighting_flag: Улучшить освещение если нужно (по умолчанию True)
            apply_depth_check: Применить проверку глубины (по умолчанию True)

        Returns:
            Dict с эмбеддингом и детальными метаданными
        """
        start_time = time.time()

        try:
            if not self.is_initialized:
                await self.initialize()

            logger.info(
                "Generating face embedding with MediaPipe alignment and analysis"
            )

            # Загрузка изображения (храним как PIL)
            image_pil = self._load_image_pil(image_data)
            image_np = self._image_to_numpy(image_pil)

            # Оптимизированная детекция лица
            face_detected, face_crop, multiple_faces, faces_count = (
                await self._detect_faces_optimized(image_pil)
            )

            if not face_detected or face_crop is None:
                logger.warning("No face detected in image")
                return {
                    "success": True,
                    "face_detected": False,
                    "multiple_faces": False,
                    "quality_score": 0.0,
                    "error": "No face detected",
                    "face_alignment_applied": False,
                    "lighting_analysis": None,
                    "depth_analysis": None,
                }

            # Конвертируем face_crop в numpy для дальнейшей обработки
            face_crop_np = face_crop[0].permute(1, 2, 0).cpu().numpy()
            face_crop_uint8 = (face_crop_np * 255).astype(np.uint8)

            # ========================================
            # ENHANCED FACE ALIGNMENT (MediaPipe)
            # ========================================
            alignment_metadata = {
                "face_alignment_applied": False,
                "rotation_angle": 0.0,
                "alignment_method": "none",
                "landmarks_type": "none",
            }

            aligned_face_np = None

            if apply_face_alignment:
                # ✅ Детекция 68-point landmarks через MediaPipe
                # detect_face_landmarks() теперь использует MediaPipe внутри
                # и конвертирует 468 точек → 68 точек (dlib-compatible)
                landmarks = await asyncio.to_thread(detect_face_landmarks, image_np)

                if landmarks is not None and len(landmarks) == 68:
                    # Выравнивание лица по 68 landmarks
                    aligned_face_np, alignment_info = await asyncio.to_thread(
                        align_face, image_np, landmarks, output_size=(112, 112)
                    )

                    # Расширенные метаданные выравнивания
                    left_eye = np.mean(landmarks[36:42], axis=0)
                    right_eye = np.mean(landmarks[42:48], axis=0)
                    rotation_angle = (
                        np.arctan2(
                            right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
                        )
                        * 180.0
                        / np.pi
                    )

                    # ✅ FaceLandmarks теперь работает с MediaPipe landmarks
                    face_landmarks = FaceLandmarks.from_68_points(landmarks)

                    alignment_metadata = {
                        "face_alignment_applied": True,
                        "rotation_angle": float(rotation_angle),
                        "alignment_method": "68_point_landmarks",
                        "landmarks_type": "mediapipe_converted_to_dlib_68",  # ✅ Обновлено
                        "landmarks_detected": len(landmarks),
                        "eye_distance": float(face_landmarks.get_eye_distance()),
                        "face_ratio": float(face_landmarks.get_face_ratio()),
                        "alignment_quality": alignment_info.get("face_ratio", 0.8),
                    }

                    # Конвертируем выровненное лицо в тензор для FaceNet
                    aligned_tensor = (
                        torch.from_numpy(aligned_face_np.transpose(2, 0, 1)).float()
                        / 255.0
                    )
                    face_crop = aligned_tensor.unsqueeze(0).to(self.device)

                    # Используем выровненное изображение для анализа
                    face_crop_uint8 = aligned_face_np.copy()

                    logger.info(
                        f"Face aligned (MediaPipe): rotation={rotation_angle:.1f}°, "
                        f"face_ratio={face_landmarks.get_face_ratio():.2f}"
                    )

            # ========================================
            # ENHANCED LIGHTING ANALYSIS
            # ========================================
            lighting_analysis: Optional[LightingAnalysis] = None
            if enhance_lighting_flag:
                lighting_analysis = await asyncio.to_thread(
                    analyze_shadows_and_lighting, face_crop_uint8
                )

                # Улучшаем освещение если качество низкое
                if lighting_analysis and lighting_analysis.overall_quality < 0.55:
                    enhanced_face = await asyncio.to_thread(
                        enhance_lighting, face_crop_uint8, lighting_analysis
                    )

                    # Конвертируем улучшенное лицо обратно в тензор
                    enhanced_tensor = (
                        torch.from_numpy(enhanced_face.transpose(2, 0, 1)).float()
                        / 255.0
                    )
                    face_crop = enhanced_tensor.unsqueeze(0).to(self.device)

                    logger.info(
                        f"Lighting enhanced: quality={lighting_analysis.overall_quality:.2f}"
                    )

            # ========================================
            # 3D DEPTH ESTIMATION (Anti-Spoofing)
            # ========================================
            depth_analysis: Optional[DepthAnalysis] = None
            if apply_depth_check:
                depth_analysis = await asyncio.to_thread(
                    analyze_depth_for_liveness, face_crop_uint8
                )

                if depth_analysis:
                    if not depth_analysis.is_likely_real:
                        logger.warning(
                            f"Depth analysis suggests possible spoof: {depth_analysis.anomalies}"
                        )

                    logger.debug(
                        f"Depth analysis: score={depth_analysis.depth_score:.3f}, "
                        f"flatness={depth_analysis.flatness_score:.3f}"
                    )

            # ========================================
            # EMBEDDING GENERATION
            # ========================================
            embedding = await asyncio.to_thread(
                self._generate_face_embedding, face_crop
            )

            # ========================================
            # FACE QUALITY ASSESSMENT
            # ========================================
            quality_score = await self._assess_face_quality(face_crop)

            processing_time = time.time() - start_time
            self._update_stats(processing_time, "embeddings_generated")

            # ========================================
            # FORMATTED RESULT
            # ========================================
            result = {
                "success": True,
                "embedding": embedding,
                "quality_score": quality_score,
                "face_detected": True,
                "multiple_faces": multiple_faces,
                "model_version": "facenet-vggface2-mediapipe-enhanced",  # ✅ Обновлено
                "processing_time": processing_time,
                "face_alignment": alignment_metadata,
                "lighting_analysis": (
                    {
                        "overall_quality": (
                            lighting_analysis.overall_quality
                            if lighting_analysis
                            else None
                        ),
                        "exposure_score": (
                            lighting_analysis.exposure_score
                            if lighting_analysis
                            else None
                        ),
                        "shadow_evenness": (
                            lighting_analysis.shadow_evenness
                            if lighting_analysis
                            else None
                        ),
                        "left_right_balance": (
                            lighting_analysis.left_right_balance
                            if lighting_analysis
                            else None
                        ),
                        "contrast_score": (
                            lighting_analysis.contrast_score
                            if lighting_analysis
                            else None
                        ),
                        "issues": lighting_analysis.issues if lighting_analysis else [],
                        "recommendations": (
                            lighting_analysis.recommendations
                            if lighting_analysis
                            else []
                        ),
                    }
                    if lighting_analysis
                    else None
                ),
                "depth_analysis": (
                    {
                        "depth_score": (
                            depth_analysis.depth_score if depth_analysis else None
                        ),
                        "flatness_score": (
                            depth_analysis.flatness_score if depth_analysis else None
                        ),
                        "is_likely_real": (
                            depth_analysis.is_likely_real if depth_analysis else None
                        ),
                        "confidence": (
                            depth_analysis.confidence if depth_analysis else None
                        ),
                        "anomalies": depth_analysis.anomalies if depth_analysis else [],
                    }
                    if depth_analysis
                    else None
                ),
            }

            logger.info(
                f"Embedding generated (MediaPipe): quality={quality_score:.3f}, "
                f"aligned={alignment_metadata['face_alignment_applied']}, "
                f"time={processing_time:.3f}s"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Embedding generation failed: {str(e)}")
            raise MLServiceError(f"Failed to generate embedding: {str(e)}")

    def _compute_cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Вычисление косинусной схожести между эмбеддингами."""
        try:
            # Нормализуем векторы
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)

            # Вычисляем косинусную схожесть (от -1 до 1)
            similarity = np.dot(embedding1_norm, embedding2_norm)

            return float(np.clip(similarity, -1.0, 1.0))

        except Exception as e:
            logger.error(f"Error computing cosine similarity: {str(e)}")
            return 0.0

    def _compute_euclidean_distance(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Вычисление евклидового расстояния между эмбеддингами."""
        try:
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)

        except Exception as e:
            logger.error(f"Error computing euclidean distance: {str(e)}")
            return float("inf")

    async def verify_face(
        self, image_data: bytes, reference_embedding: np.ndarray, threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Верификация лица по эталонному эмбеддингу."""
        start_time = time.time()

        try:
            if not self.is_initialized:
                await self.initialize()

            logger.info("Verifying face")

            # Генерация эмбеддинга для изображения
            embedding_result = await self.generate_embedding(image_data)

            if not embedding_result.get("face_detected"):
                return {
                    "success": True,
                    "verified": False,
                    "confidence": 0.0,
                    "similarity_score": 0.0,
                    "face_detected": False,
                    "error": "No face detected in image",
                }

            # Вычисляем косинусную схожесть
            current_embedding = embedding_result["embedding"]
            similarity_score = self._compute_cosine_similarity(
                current_embedding, reference_embedding
            )

            # Определяем результат верификации
            verified = similarity_score >= threshold
            distance = self._compute_euclidean_distance(
                current_embedding, reference_embedding
            )

            processing_time = time.time() - start_time

            logger.info(
                f"Face verification: {verified} (similarity: {similarity_score:.3f}, "
                f"threshold: {threshold}, time: {processing_time:.3f}s)"
            )

            return {
                "success": True,
                "verified": verified,
                "confidence": similarity_score,
                "similarity_score": similarity_score,
                "threshold": threshold,
                "face_detected": True,
                "face_quality": embedding_result.get("quality_score", 0.0),
                "distance": distance,
                "model_version": "facenet-vggface2-mediapipe",  # ✅ Обновлено
                "processing_time": processing_time,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Face verification failed: {str(e)}")
            raise MLServiceError(f"Failed to verify face: {str(e)}")

    async def check_liveness(
        self,
        image_data: bytes,
        challenge_type: str = "passive",
        challenge_data: Optional[Dict[str, Any]] = None,
        use_3d_depth: bool = True,
    ) -> Dict[str, Any]:
        """
        Проверка живости лица с поддержкой 3D Depth Estimation.

        Args:
            image_data: Байты изображения
            challenge_type: Тип челленджа (passive, blink, smile, turn_head)
            challenge_data: Данные челленджа
            use_3d_depth: Использовать 3D depth estimation (по умолчанию True)

        Returns:
            Dict с результатами проверки живости
        """
        start_time = time.time()

        try:
            if not self.is_initialized:
                await self.initialize()

            # Загрузка изображения
            image_pil = self._load_image_pil(image_data)
            image_np = self._image_to_numpy(image_pil)

            # Детекция лица
            face_detected, face_crop, multiple_faces, faces_count = (
                await self._detect_faces_optimized(image_pil)
            )

            if not face_detected or face_crop is None:
                return {
                    "success": True,
                    "liveness_detected": False,
                    "confidence": 0.0,
                    "face_detected": False,
                    "error": "No face detected",
                    "liveness_type": "no_face",
                }

            # Конвертируем face_crop в numpy для анализа
            face_crop_np = face_crop[0].permute(1, 2, 0).cpu().numpy()
            face_crop_uint8 = (face_crop_np * 255).astype(np.uint8)

            # ========================================
            # 1. CERTIFIED ANTI-SPOOFING (MiniFASNetV2)
            # ========================================
            anti_spoofing_result = None
            if self._anti_spoofing_service:
                try:
                    anti_spoofing_result = (
                        await self._anti_spoofing_service.check_liveness(image_data)
                    )
                    logger.info(
                        f"Anti-spoofing model: {'REAL' if anti_spoofing_result['liveness_detected'] else 'SPOOF'} "
                        f"(confidence: {anti_spoofing_result['confidence']:.3f})"
                    )
                except Exception as e:
                    logger.warning(f"Anti-spoofing model failed: {str(e)}")

            anti_spoofing_score = (
                anti_spoofing_result.get("real_probability", 0.5)
                if anti_spoofing_result
                else 0.5
            )

            # ========================================
            # 2. 3D DEPTH ESTIMATION
            # ========================================
            depth_analysis: Optional[DepthAnalysis] = None
            if use_3d_depth:
                depth_analysis = await asyncio.to_thread(
                    analyze_depth_for_liveness, face_crop_uint8
                )

                if depth_analysis:
                    self._update_stats(time.time() - start_time, "depth_checks")

                    logger.info(
                        f"Depth analysis: score={depth_analysis.depth_score:.3f}, "
                        f"flatness={depth_analysis.flatness_score:.3f}, "
                        f"is_real={depth_analysis.is_likely_real}"
                    )

            depth_score = depth_analysis.depth_score if depth_analysis else 0.5

            # ========================================
            # 3. SHADOW/LIGHTING ANALYSIS
            # ========================================
            lighting_analysis = await asyncio.to_thread(
                analyze_shadows_and_lighting, face_crop_uint8
            )

            lighting_quality = (
                lighting_analysis.overall_quality if lighting_analysis else 0.5
            )

            # ========================================
            # 4. COMBINE ALL SCORES
            # ========================================
            if anti_spoofing_result:
                # Используем комбинированный скор с весами
                combined_result = combine_liveness_scores(
                    anti_spoofing_score=anti_spoofing_score,
                    depth_score=depth_score,
                    lighting_quality=lighting_quality,
                    depth_analysis=depth_analysis,
                )

                liveness_detected = combined_result["liveness_detected"]
                confidence = combined_result["confidence"]

                # Дополнительная проверка depth anomalies
                if depth_analysis and depth_analysis.anomalies:
                    confidence *= max(0.5, 1.0 - len(depth_analysis.anomalies) * 0.1)
                    if len(depth_analysis.anomalies) >= 3:
                        liveness_detected = False

                result = {
                    "success": True,
                    "liveness_detected": liveness_detected,
                    "confidence": confidence,
                    "anti_spoofing_score": anti_spoofing_score,
                    "depth_score": depth_score,
                    "lighting_quality": lighting_quality,
                    "face_detected": True,
                    "multiple_faces": multiple_faces,
                    "liveness_type": "certified_with_depth",
                    "model_version": "MiniFASNetV2+3D-Depth+MediaPipe",  # ✅ Обновлено
                    "accuracy_claim": ">98%",
                    "processing_time": time.time() - start_time,
                    "depth_analysis": (
                        {
                            "depth_score": (
                                depth_analysis.depth_score if depth_analysis else None
                            ),
                            "flatness_score": (
                                depth_analysis.flatness_score
                                if depth_analysis
                                else None
                            ),
                            "is_likely_real": (
                                depth_analysis.is_likely_real
                                if depth_analysis
                                else None
                            ),
                            "anomalies": (
                                depth_analysis.anomalies if depth_analysis else []
                            ),
                        }
                        if depth_analysis
                        else None
                    ),
                    "lighting_analysis": (
                        {
                            "overall_quality": (
                                lighting_analysis.overall_quality
                                if lighting_analysis
                                else None
                            ),
                            "issues": (
                                lighting_analysis.issues if lighting_analysis else []
                            ),
                        }
                        if lighting_analysis
                        else None
                    ),
                    "score_breakdown": combined_result,
                }

            else:
                # Fallback на heuristic + depth analysis
                liveness_score = depth_score * 0.5 + lighting_quality * 0.3 + 0.2

                # Корректировка на основе аномалий
                if depth_analysis and depth_analysis.anomalies:
                    liveness_score *= max(
                        0.3, 1.0 - len(depth_analysis.anomalies) * 0.15
                    )

                liveness_detected = liveness_score > 0.5
                confidence = (
                    liveness_score if liveness_detected else 1.0 - liveness_score
                )

                result = {
                    "success": True,
                    "liveness_detected": liveness_detected,
                    "confidence": confidence,
                    "anti_spoofing_score": 0.5,
                    "depth_score": depth_score,
                    "lighting_quality": lighting_quality,
                    "face_detected": True,
                    "multiple_faces": multiple_faces,
                    "liveness_type": "depth_heuristic_non_certified",
                    "model_version": "3D-Depth-Heuristic+MediaPipe",  # ✅ Обновлено
                    "accuracy_claim": "non-certified",
                    "processing_time": time.time() - start_time,
                    "depth_analysis": (
                        {
                            "depth_score": (
                                depth_analysis.depth_score if depth_analysis else None
                            ),
                            "anomalies": (
                                depth_analysis.anomalies if depth_analysis else []
                            ),
                        }
                        if depth_analysis
                        else None
                    ),
                }

            # Обновляем статистику
            self._update_stats(time.time() - start_time, "liveness_checks")

            logger.info(
                f"Liveness check: {'REAL' if result['liveness_detected'] else 'SPOOF'} "
                f"(confidence: {result['confidence']:.3f}, time: {result['processing_time']:.3f}s)"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Liveness check failed: {str(e)}")
            raise MLServiceError(f"Failed to check liveness: {str(e)}")

    def _update_stats(self, processing_time: float, operation_type: str):
        """Обновление статистики."""
        if self.enable_performance_monitoring:
            self.stats["requests"] += 1
            self.stats[operation_type] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["max_processing_time"] = max(
                self.stats["max_processing_time"], processing_time
            )
            self.stats["min_processing_time"] = min(
                self.stats["min_processing_time"], processing_time
            )

            # Среднее время на запрос
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["requests"]
            )

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности."""
        stats = {
            "requests": self.stats["requests"],
            "face_detections": self.stats["face_detections"],
            "embeddings_generated": self.stats["embeddings_generated"],
            "liveness_checks": self.stats["liveness_checks"],
            "depth_checks": self.stats["depth_checks"],
            "total_processing_time": self.stats["total_processing_time"],
            "average_processing_time": self.stats["average_processing_time"],
            "max_processing_time": self.stats["max_processing_time"],
            "min_processing_time": self.stats["min_processing_time"],
            "device": str(self.device),
            "models_initialized": self.is_initialized,
            "cuda_available": torch.cuda.is_available(),
            "certified_liveness_enabled": self._certified_liveness_enabled,
            "landmarks_backend": "mediapipe",  # ✅ Добавлено
        }

        # Добавляем статистику Anti-Spoofing если доступна
        if self._anti_spoofing_service:
            try:
                anti_spoofing_stats = self._anti_spoofing_service.get_stats()
                stats["anti_spoofing"] = anti_spoofing_stats
            except Exception:
                stats["anti_spoofing"] = {"status": "error"}

        return stats

    async def verify(
        self, reference_image: bytes, probe_image: bytes, threshold: float = 0.8
    ):
        """
        Верификация двух изображений (alias для совместимости).
        """
        # Генерируем эмбеддинг для reference
        ref_result = await self.generate_embedding(reference_image)
        if not ref_result.get("face_detected"):
            return {"success": False, "error": "No face detected in reference image"}

        # Верифицируем probe относительно reference
        verify_result = await self.verify_face(
            image_data=probe_image,
            reference_embedding=ref_result["embedding"],
            threshold=threshold,
        )

        return verify_result

    async def analyze_video_liveness(
        self, video_data: bytes, challenge_type: str, frame_count: int = 10
    ):
        """Video liveness analysis - STUB for Phase 6."""
        logger.warning("analyze_video_liveness called but not fully implemented yet")
        # TODO: Implement proper video frame extraction
        return {
            "success": True,
            "liveness_detected": False,
            "confidence": 0.5,
            "frames_processed": 0,
            "error": "Video liveness not yet implemented",
        }

    async def check_active_liveness(
        self, image_data: bytes, challenge_type: str, challenge_data: dict
    ):
        """Active liveness - delegates to check_liveness."""
        return await self.check_liveness(image_data, challenge_type, challenge_data)

    async def batch_generate_embeddings(
        self,
        image_data_list: list,
        batch_size: int = 10,
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None,
        fail_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Batch generation of face embeddings with production-grade features.

        Features:
        - Parallel processing with semaphore-controlled concurrency
        - Automatic retries with exponential backoff
        - Progress tracking via callback
        - Graceful error handling with detailed reporting
        - Memory management between batches
        - Comprehensive metrics and logging

        Args:
            image_data_list: List of image bytes
            batch_size: Number of images per batch (for memory management)
            max_concurrent: Maximum concurrent tasks (default: 5)
            progress_callback: Optional callback(current, total, result)
            fail_on_error: Raise exception on first error (default: False)

        Returns:
            Dict with 'results', 'metrics', and 'errors'
        """
        total_count = len(image_data_list)
        if total_count == 0:
            return {
                "results": [],
                "metrics": {"total": 0, "success": 0, "failed": 0},
                "errors": [],
            }

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        semaphore = asyncio.Semaphore(max_concurrent)

        # Track per-batch metrics
        batch_metrics = {"total": total_count, "success": 0, "failed": 0, "retries": 0}

        async def process_with_retry(img_data: bytes, index: int) -> Dict[str, Any]:
            """Process single image with retry logic."""
            max_retries = 3
            base_delay = 0.5

            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        result = await self.generate_embedding(img_data)
                        return {
                            "index": index,
                            "success": True,
                            "data": result,
                        }
                except Exception as e:
                    batch_metrics["retries"] += 1
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        await asyncio.sleep(delay)
                    else:
                        # Final attempt failed
                        return {
                            "index": index,
                            "success": False,
                            "error": str(e),
                            "attempts": attempt + 1,
                        }

        # Process in batches for memory management
        batch_results: List[Dict[str, Any]] = []
        for batch_start in range(0, total_count, batch_size):
            batch_end = min(batch_start + batch_size, total_count)
            batch = image_data_list[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // batch_size + 1}: "
                f"images {batch_start + 1}-{batch_end} of {total_count}"
            )

            # Process batch concurrently
            batch_tasks = [
                process_with_retry(img_data, batch_start + i)
                for i, img_data in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Collect results and handle errors
            for i, result in enumerate(batch_results):
                actual_index = batch_start + i

                if isinstance(result, Exception):
                    error_entry = {
                        "index": actual_index,
                        "error": str(result),
                        "type": type(result).__name__,
                    }
                    errors.append(error_entry)
                    batch_metrics["failed"] += 1

                    if fail_on_error:
                        raise result

                    # Add placeholder for failed result
                    results.append(
                        {
                            "index": actual_index,
                            "success": False,
                            "error": str(result),
                        }
                    )
                elif not result.get("success", False):
                    error_entry = {
                        "index": actual_index,
                        "error": result.get("error", "Unknown error"),
                        "attempts": result.get("attempts", 1),
                    }
                    errors.append(error_entry)
                    batch_metrics["failed"] += 1

                    if fail_on_error:
                        raise ProcessingError(error_entry["error"])

                    results.append(
                        {
                            "index": actual_index,
                            "success": False,
                            "error": result.get("error"),
                        }
                    )
                else:
                    results.append(
                        {
                            "index": actual_index,
                            "success": True,
                            "embedding": result["data"].get("embedding"),
                            "quality_score": result["data"].get("quality_score"),
                            "face_detected": result["data"].get("face_detected", False),
                            "metadata": {
                                "processing_time": result["data"].get(
                                    "processing_time"
                                ),
                                "model_version": result["data"].get("model_version"),
                            },
                        }
                    )
                    batch_metrics["success"] += 1

            # Progress callback
            processed = batch_end
            if progress_callback:
                try:
                    progress_callback(processed, total_count, results[-len(batch) :])
                except Exception as cb_err:
                    logger.warning(f"Progress callback failed: {cb_err}")

            # Memory cleanup between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Sort results by original index
        results.sort(key=lambda x: x["index"])

        total_time = time.monotonic() - start_time
        metrics = {
            **batch_metrics,
            "total_time_seconds": round(total_time, 3),
            "avg_time_per_image": (
                round(total_time / total_count, 3) if total_count > 0 else 0
            ),
            "throughput_per_second": (
                round(total_count / total_time, 2) if total_time > 0 else 0
            ),
            "error_rate": (
                round(batch_metrics["failed"] / total_count, 3)
                if total_count > 0
                else 0
            ),
        }

        logger.info(
            f"Batch embedding generation completed: "
            f"{batch_metrics['success']}/{total_count} successful, "
            f"{batch_metrics['failed']} failed, "
            f"{total_time:.2f}s total"
        )

        return {
            "results": results,
            "metrics": metrics,
            "errors": errors,
        }

    async def advanced_anti_spoofing_check(self, image_data: bytes, analysis_type: str):
        """Advanced anti-spoofing analysis."""
        result = await self.check_liveness(
            image_data, challenge_type="passive", use_3d_depth=True
        )

        # Добавляем analysis_results для совместимости с response model
        result["analysis_results"] = {
            "depth_analysis": result.get("depth_analysis"),
            "lighting_analysis": result.get("lighting_analysis"),
            "certified_analysis": {
                "is_certified_passed": result.get("liveness_detected", False),
                "certification_level": (
                    "MiniFASNetV2" if self._certified_liveness_enabled else "heuristic"
                ),
            },
        }
        result["component_scores"] = {
            "anti_spoofing": result.get("anti_spoofing_score", 0.5),
            "depth": result.get("depth_score", 0.5),
            "lighting": result.get("lighting_quality", 0.5),
        }

        return result


# Alias для обратной совместимости
MLService = OptimizedMLService

# Module-level singleton
_ml_service: Optional[OptimizedMLService] = None


async def get_ml_service() -> OptimizedMLService:
    """Get or create ML service singleton."""
    global _ml_service
    if _ml_service is None:
        _ml_service = OptimizedMLService()
        await _ml_service.initialize()
    return _ml_service
