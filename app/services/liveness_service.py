"""
Liveness Service - Проверка живости лица (Anti-Spoofing).

Ответственность:
- Passive liveness detection
- Active liveness detection (blink, smile, turn_head)
- Video liveness (multi-frame analysis)
- Advanced anti-spoofing (MiniFASNetV2)
- Challenge-response механизм
"""

import time
import uuid
import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.crud import VerificationSessionCRUD
from ..services.ml_service import MLService
from ..services.anti_spoofing_service import AntiSpoofingService
from ..services.validation_service import ValidationService
from ..services.cache_service import CacheService
from ..services.webhook_service import WebhookService
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError
from ..middleware.metrics import record_liveness

logger = get_logger(__name__)


class LivenessService:
    """Сервис для проверки живости лица."""

    SUPPORTED_CHALLENGES = {
        "passive": "Passive liveness detection (no user action required)",
        "active": "Active liveness with challenge-response",
        "blink": "Blink detection",
        "smile": "Smile detection",
        "turn_head": "Head turn detection",
        "video_blink": "Video-based blink detection",
        "video_smile": "Video-based smile detection",
        "video_head_turn": "Video-based head turn detection",
    }

    def __init__(self, db: AsyncSession):
        """
        Инициализация сервиса.

        Args:
            db: Асинхронная сессия базы данных
        """
        self.db = db
        self.ml_service = MLService()
        self.anti_spoofing_service = AntiSpoofingService()
        self.validation_service = ValidationService()
        self.cache_service = CacheService()
        self.webhook_service = WebhookService(db)

    # =========================================================================
    # Основная проверка живости
    # =========================================================================

    async def check_liveness(
        self,
        image_data: bytes,
        challenge_type: str = "passive",
        challenge_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Основная проверка живости лица.

        Args:
            image_data: Бинарные данные изображения
            challenge_type: Тип проверки (passive, active, blink, smile, turn_head)
            challenge_data: Данные для активной проверки
            user_id: ID пользователя (для webhook)
            session_id: ID сессии

        Returns:
            Словарь с результатами проверки живости

        Raises:
            ValidationError: Если изображение не прошло валидацию или challenge_type неверный
            ProcessingError: Если ML обработка не удалась
        """
        start_time = time.monotonic()
        request_id = session_id or str(uuid.uuid4())

        logger.info(f"Starting liveness check: type={challenge_type}, request={request_id}")

        # 1. Валидация challenge_type
        if challenge_type not in self.SUPPORTED_CHALLENGES:
            raise ValidationError(
                f"Unsupported challenge_type: {challenge_type}. "
                f"Supported: {list(self.SUPPORTED_CHALLENGES.keys())}"
            )

        # 2. Проверка требований к challenge_data
        if challenge_type in ("active", "blink", "smile", "turn_head") and not challenge_data:
            raise ValidationError(
                f"challenge_data is required for challenge_type={challenge_type}"
            )

        # 3. Валидация изображения
        validation_result = await self.validation_service.validate_image(
            image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # 4. Выполнение проверки в зависимости от типа
        if challenge_type == "passive":
            liveness_result = await self.check_passive_liveness(
                validation_result.image_data
            )
        else:
            liveness_result = await self.check_active_liveness(
                image_data=validation_result.image_data,
                challenge_type=challenge_type,
                challenge_data=challenge_data,
            )

        # 5. Определение результата
        liveness_detected = liveness_result.get("liveness_detected", False)
        confidence = liveness_result.get("confidence", 0.0)
        anti_spoofing_score = liveness_result.get("anti_spoofing_score")

        # 6. Дополнительные эвристики (fallback)
        heuristic_result = await self._analyze_spoof_heuristics(
            validation_result.image_data
        )

        if anti_spoofing_score is None:
            anti_spoofing_score = heuristic_result.get("score", 0.5)

        processing_time = time.monotonic() - start_time

        result = {
            "success": True,
            "session_id": request_id,
            "liveness_detected": liveness_detected,
            "confidence": confidence,
            "challenge_type": challenge_type,
            "processing_time": processing_time,
            "anti_spoofing_score": anti_spoofing_score,
            "face_detected": liveness_result.get("face_detected", False),
            "multiple_faces": liveness_result.get("multiple_faces", False),
            "image_quality": liveness_result.get("image_quality"),
            "recommendations": liveness_result.get("recommendations", []),
            "liveness_type": liveness_result.get("liveness_type", "unknown"),
            "depth_analysis": liveness_result.get("depth_analysis"),
            "heuristic_flags": heuristic_result.get("flags", []),
            "request_id": request_id,
        }

        logger.info(
            f"Liveness check completed: detected={liveness_detected}, "
            f"confidence={confidence:.3f}, "
            f"type={challenge_type}, "
            f"time={processing_time:.3f}s"
        )

        # Запись метрики liveness
        result_label = "passed" if liveness_detected else "failed"
        record_liveness(result_label)

        return result

    # =========================================================================
    # Passive liveness
    # =========================================================================

    async def check_passive_liveness(
        self,
        image_data: bytes,
    ) -> Dict[str, Any]:
        """
        Passive liveness detection (без активных действий пользователя).

        Args:
            image_data: Бинарные данные изображения

        Returns:
            Результат passive liveness проверки
        """
        logger.info("Performing passive liveness check")

        # 1. ML проверка через основной сервис
        ml_result = await self.ml_service.check_liveness(
            image_data=image_data,
            challenge_type="passive",
        )

        # 2. Certified anti-spoofing (если включен)
        if settings.USE_CERTIFIED_LIVENESS:
            certified_result = await self.anti_spoofing_service.check_liveness(
                image_data=image_data
            )

            # Комбинируем результаты
            ml_score = ml_result.get("confidence", 0.0)
            certified_score = certified_result.get("liveness_score", 0.0)

            # Weighted average (certified имеет больший вес)
            combined_confidence = (ml_score * 0.3) + (certified_score * 0.7)

            liveness_detected = (
                certified_result.get("is_real", False) and
                combined_confidence >= settings.LIVENESS_CONFIDENCE_THRESHOLD
            )

            return {
                "liveness_detected": liveness_detected,
                "confidence": combined_confidence,
                "liveness_type": "passive_certified",
                "face_detected": ml_result.get("face_detected", False),
                "multiple_faces": ml_result.get("multiple_faces", False),
                "image_quality": ml_result.get("image_quality"),
                "anti_spoofing_score": certified_score,
                "recommendations": ml_result.get("recommendations", []),
                "depth_analysis": certified_result.get("depth_analysis"),
            }
        else:
            # Только базовая ML проверка
            return {
                "liveness_detected": ml_result.get("liveness_detected", False),
                "confidence": ml_result.get("confidence", 0.0),
                "liveness_type": "passive_basic",
                "face_detected": ml_result.get("face_detected", False),
                "multiple_faces": ml_result.get("multiple_faces", False),
                "image_quality": ml_result.get("image_quality"),
                "recommendations": ml_result.get("recommendations", []),
            }

    # =========================================================================
    # Active liveness
    # =========================================================================

    async def check_active_liveness(
        self,
        image_data: bytes,
        challenge_type: str,
        challenge_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Active liveness detection с челленджами.

        Args:
            image_data: Бинарные данные изображения
            challenge_type: Тип челленджа (blink, smile, turn_head, active)
            challenge_data: Данные для проверки челленджа

        Returns:
            Результат active liveness проверки
        """
        logger.info(f"Performing active liveness check: {challenge_type}")

        # ML проверка
        ml_result = await self.ml_service.check_active_liveness(
            image_data=image_data,
            challenge_type=challenge_type,
            challenge_data=challenge_data,
        )

        return {
            "liveness_detected": ml_result.get("liveness_detected", False),
            "confidence": ml_result.get("confidence", 0.0),
            "liveness_type": f"active_{challenge_type}",
            "face_detected": ml_result.get("face_detected", False),
            "image_quality": ml_result.get("image_quality"),
            "anti_spoofing_score": ml_result.get("anti_spoofing_score"),
            "recommendations": ml_result.get("recommendations", []),
            "challenge_specific_data": ml_result.get("challenge_specific_data", {}),
        }

    # =========================================================================
    # Video liveness
    # =========================================================================

    async def analyze_video_liveness(
        self,
        video_frames: List[bytes],
        challenge_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multi-frame анализ для видео liveness.

        Args:
            video_frames: Список кадров видео (binary data)
            challenge_type: Опциональный тип челленджа для видео

        Returns:
            Результат video liveness проверки
        """
        logger.info(f"Performing video liveness analysis: {len(video_frames)} frames")

        if not video_frames:
            raise ValidationError("No video frames provided")

        # ML анализ видео
        ml_result = await self.ml_service.analyze_video_liveness(
            video_data=video_frames,  # Передаём список кадров
            challenge_type=challenge_type,
            frame_count=len(video_frames),
        )

        return {
            "liveness_detected": ml_result.get("liveness_detected", False),
            "confidence": ml_result.get("confidence", 0.0),
            "liveness_type": f"video_{challenge_type or 'passive'}",
            "frames_processed": ml_result.get("frames_processed", 0),
            "sequence_data": ml_result.get("sequence_data"),
            "anti_spoofing_score": ml_result.get("anti_spoofing_score"),
            "face_detected": ml_result.get("face_detected", False),
            "recommendations": ml_result.get("recommendations", []),
        }

    # =========================================================================
    # Advanced anti-spoofing
    # =========================================================================

    async def perform_anti_spoofing_check(
        self,
        image_data: bytes,
        analysis_type: str = "certified",
        include_reasoning: bool = False,
    ) -> Dict[str, Any]:
        """
        Продвинутая anti-spoofing проверка.

        Args:
            image_data: Бинарные данные изображения
            analysis_type: Тип анализа (certified, depth, texture, comprehensive)
            include_reasoning: Включать ли multi-turn reasoning

        Returns:
            Результат anti-spoofing проверки
        """
        logger.info(f"Performing advanced anti-spoofing: {analysis_type}")

        # Валидация изображения
        validation_result = await self.validation_service.validate_image(
            image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # ML анализ
        ml_result = await self.ml_service.advanced_anti_spoofing_check(
            image_data=validation_result.image_data,
            analysis_type=analysis_type,
        )

        return {
            "liveness_detected": ml_result.get("liveness_detected", False),
            "confidence": ml_result.get("confidence", 0.0),
            "anti_spoofing_score": ml_result.get("anti_spoofing_score", 0.0),
            "analysis_type": analysis_type,
            "depth_analysis": ml_result.get("analysis_results", {}).get("depth_analysis"),
            "texture_analysis": ml_result.get("analysis_results", {}).get("texture_analysis"),
            "certified_analysis": ml_result.get("analysis_results", {}).get("certified_analysis"),
            "reasoning_result": ml_result.get("analysis_results", {}).get("reasoning_result") if include_reasoning else None,
            "component_scores": ml_result.get("component_scores"),
            "certification_level": ml_result.get("analysis_results", {}).get("certified_analysis", {}).get("certification_level"),
            "certification_passed": ml_result.get("analysis_results", {}).get("certified_analysis", {}).get("is_certified_passed", False),
            "face_detected": ml_result.get("face_detected", False),
            "recommendations": ml_result.get("recommendations", []),
        }

    # =========================================================================
    # Challenge generation
    # =========================================================================

    async def generate_challenge(
        self,
        challenge_type: str,
    ) -> Dict[str, Any]:
        """
        Генерация челленджа для активной проверки живости.

        Args:
            challenge_type: Тип челленджа (blink, smile, turn_head, random)

        Returns:
            Данные челленджа
        """
        if challenge_type == "random":
            challenge_type = random.choice(["blink", "smile", "turn_head"])

        challenges = {
            "blink": {
                "action": "Blink your eyes 2 times",
                "expected_blinks": 2,
                "timeout_seconds": 5,
                "instructions": "Please blink your eyes twice within 5 seconds",
            },
            "smile": {
                "action": "Smile",
                "timeout_seconds": 3,
                "instructions": "Please smile naturally",
            },
            "turn_head": {
                "action": "Turn your head to the left, then right",
                "directions": ["left", "right"],
                "timeout_seconds": 5,
                "instructions": "Please turn your head left, then right within 5 seconds",
            },
        }

        challenge = challenges.get(challenge_type, challenges["blink"])
        challenge["challenge_id"] = str(uuid.uuid4())
        challenge["challenge_type"] = challenge_type
        challenge["created_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Challenge generated: {challenge_type}")
        return challenge

    # =========================================================================
    # Вспомогательные методы
    # =========================================================================

    async def _analyze_spoof_heuristics(
        self,
        image_data: bytes,
    ) -> Dict[str, Any]:
        """
        Эвристический анализ признаков подделки (fallback).

        Args:
            image_data: Бинарные данные изображения

        Returns:
            Результат эвристического анализа
        """
        # Простые эвристики через ValidationService
        try:
            import cv2
            import numpy as np

            # Преобразуем в numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"score": 0.5, "flags": ["invalid_image"]}

            flags = []

            # 1. Проверка на moiré паттерны
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            # Высокая частота в FFT может указывать на moiré
            high_freq_ratio = np.sum(magnitude > np.percentile(magnitude, 95)) / magnitude.size
            if high_freq_ratio > 0.05:
                flags.append("moire_detected")

            # 2. Проверка на низкую вариацию (screen photo)
            variance = np.var(gray)
            if variance < 100:
                flags.append("low_variance")

            # 3. Проверка на резкие края (print photo)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            if edge_ratio > 0.15:
                flags.append("high_edge_density")

            # Итоговый score
            score = 1.0 - (len(flags) * 0.2)
            score = max(0.0, min(1.0, score))

            return {
                "score": score,
                "flags": flags,
                "variance": float(variance),
                "edge_ratio": float(edge_ratio),
                "high_freq_ratio": float(high_freq_ratio),
            }

        except Exception as e:
            logger.warning(f"Heuristic analysis failed: {e}")
            return {"score": 0.5, "flags": ["analysis_error"]}

    @classmethod
    def get_supported_challenges(cls) -> Dict[str, str]:
        """Получение списка поддерживаемых типов проверки."""
        return cls.SUPPORTED_CHALLENGES.copy()
