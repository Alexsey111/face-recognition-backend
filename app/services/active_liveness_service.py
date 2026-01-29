"""
Active Liveness Detection Service.

Supports:
- Blink detection (моргание)
- Head movement detection (повороты головы)
- Smile detection (улыбка)
- Combined active + passive liveness verification
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.active_liveness import (
    ActiveLivenessChallengeResponse,
    ActiveLivenessVerifyResponse,
    BlinkDetectionResult,
    ChallengeInstruction,
    ChallengeSession,
    ChallengeType,
    CombinedLivenessScore,
    HeadMovementResult,
    OcclusionDetectionResult,
    SmileDetectionResult,
)
from ..services.face_occlusion_detector import get_occlusion_detector
from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.eye_blink_detector import BlinkDetector, detect_blinks_in_sequence
from ..utils.face_alignment_utils import detect_face_landmarks
from ..utils.logger import get_logger
from ..utils.mouth_detector import (
    calculate_mouth_aspect_ratio,
    detect_open_mouth_in_sequence,
    detect_smile_in_sequence,
)
from ..utils.video_processing import calculate_video_quality, extract_frames_from_video

logger = get_logger(__name__)


class ActiveLivenessService:
    """
    Сервис Active Liveness Detection.

    Features:
    - Challenge-based liveness (blink, smile, head turn)
    - Video/image sequence processing
    - Combined active + passive verification
    - Challenge session management
    """

    def __init__(self):
        # Хранилище активных челленджей (in-memory, для production используйте Redis)
        self._active_challenges: Dict[str, ChallengeSession] = {}

        # Конфигурация
        self.default_timeout_seconds = 15
        self.max_video_duration_seconds = 30
        self.min_frames_required = 20

        # Пороги детекции
        self.blink_ear_threshold = 0.21
        self.head_movement_min_angle = 15.0
        self.smile_intensity_threshold = 0.6

        # Статистика
        self.stats = {
            "challenges_created": 0,
            "challenges_completed": 0,
            "challenges_failed": 0,
        }

    # ========================================================================
    # Challenge Creation
    # ========================================================================

    async def create_challenge(
        self,
        challenge_type: ChallengeType = "random",
        timeout_seconds: int = 10,
        difficulty: str = "medium",
    ) -> ActiveLivenessChallengeResponse:
        """
        Создание нового Active Liveness челленджа.

        Args:
            challenge_type: Тип челленджа
            timeout_seconds: Время на выполнение
            difficulty: Сложность (easy/medium/hard)

        Returns:
            ActiveLivenessChallengeResponse с инструкциями
        """
        # Генерируем random challenge если запрошено
        if challenge_type == "random":
            import random

            challenge_type = random.choice(
                [
                    "blink",
                    "smile",
                    "turn_head_left",
                    "turn_head_right",
                ]
            )

        # Создаем уникальный ID
        challenge_id = str(uuid.uuid4())

        # Получаем инструкцию
        instruction = self._get_challenge_instruction(challenge_type, difficulty)

        # Параметры челленджа
        parameters = self._get_challenge_parameters(challenge_type, difficulty)

        # Создаем сессию
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=timeout_seconds)

        session = ChallengeSession(
            challenge_id=challenge_id,
            challenge_type=challenge_type,
            created_at=now,
            expires_at=expires_at,
            parameters=parameters,
            status="pending",
        )

        # Сохраняем
        self._active_challenges[challenge_id] = session
        self.stats["challenges_created"] += 1

        logger.info(f"Created challenge: {challenge_id}, type={challenge_type}")

        return ActiveLivenessChallengeResponse(
            success=True,
            challenge_id=challenge_id,
            challenge_type=challenge_type,
            instruction=instruction,
            expires_at=expires_at,
            expected_duration_seconds=instruction.duration_seconds,
            min_frames_required=parameters.get("min_frames", 30),
            recommended_fps=30,
        )

    def _get_challenge_instruction(
        self,
        challenge_type: ChallengeType,
        difficulty: str,
    ) -> ChallengeInstruction:
        """Получение инструкции для челленджа."""

        instructions = {
            "blink": ChallengeInstruction(
                text="Моргните 2 раза" if difficulty != "hard" else "Моргните 3 раза",
                icon="👁️",
                duration_seconds=5 if difficulty == "easy" else 3,
            ),
            "smile": ChallengeInstruction(
                text="Улыбнитесь",
                icon="😊",
                duration_seconds=3,
            ),
            "turn_head_left": ChallengeInstruction(
                text="Поверните голову влево",
                icon="⬅️",
                duration_seconds=3,
            ),
            "turn_head_right": ChallengeInstruction(
                text="Поверните голову вправо",
                icon="➡️",
                duration_seconds=3,
            ),
            "turn_head_up": ChallengeInstruction(
                text="Наклоните голову вверх",
                icon="⬆️",
                duration_seconds=3,
            ),
            "turn_head_down": ChallengeInstruction(
                text="Наклоните голову вниз",
                icon="⬇️",
                duration_seconds=3,
            ),
            "open_mouth": ChallengeInstruction(
                text="Откройте рот",
                icon="😮",
                duration_seconds=2,
            ),
        }

        return instructions.get(challenge_type, instructions["blink"])

    def _get_challenge_parameters(
        self,
        challenge_type: ChallengeType,
        difficulty: str,
    ) -> Dict[str, Any]:
        """Параметры для детекции челленджа."""

        difficulty_multipliers = {
            "easy": 0.8,
            "medium": 1.0,
            "hard": 1.3,
        }

        multiplier = difficulty_multipliers.get(difficulty, 1.0)

        base_params = {
            "blink": {
                "min_blinks": 2 if difficulty != "hard" else 3,
                "ear_threshold": self.blink_ear_threshold,
                "min_frames": 20,
            },
            "smile": {
                "intensity_threshold": self.smile_intensity_threshold * multiplier,
                "min_duration_frames": int(15 * multiplier),
            },
            "turn_head_left": {
                "min_angle": self.head_movement_min_angle * multiplier,
                "direction": "left",
            },
            "turn_head_right": {
                "min_angle": self.head_movement_min_angle * multiplier,
                "direction": "right",
            },
            "turn_head_up": {
                "min_angle": self.head_movement_min_angle * multiplier,
                "direction": "up",
            },
            "turn_head_down": {
                "min_angle": self.head_movement_min_angle * multiplier,
                "direction": "down",
            },
            "open_mouth": {
                "mouth_aspect_ratio_threshold": 0.5 * multiplier,
                "min_duration_frames": 10,
            },
        }

        return base_params.get(challenge_type, base_params["blink"])

    # ========================================================================
    # Challenge Verification
    # ========================================================================

    async def verify_challenge(
        self,
        challenge_id: str,
        video_data: Optional[bytes] = None,
        image_sequence: Optional[List[bytes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActiveLivenessVerifyResponse:
        """
        Верификация выполнения челленджа.

        Args:
            challenge_id: ID челленджа
            video_data: Видео (если есть)
            image_sequence: Последовательность изображений
            metadata: Дополнительные данные

        Returns:
            ActiveLivenessVerifyResponse с результатами
        """
        start_time = asyncio.get_event_loop().time()

        # Проверяем наличие челленджа
        if challenge_id not in self._active_challenges:
            raise ValidationError(f"Challenge {challenge_id} not found or expired")

        session = self._active_challenges[challenge_id]

        # Проверяем срок действия
        if datetime.now(timezone.utc) > session.expires_at:
            session.status = "expired"
            raise ValidationError(f"Challenge {challenge_id} has expired")

        # Проверяем что данные предоставлены
        if not video_data and not image_sequence:
            raise ValidationError(
                "Either video_data or image_sequence must be provided"
            )

        try:
            # 1. Извлекаем кадры
            if video_data:
                frames = await asyncio.to_thread(
                    extract_frames_from_video,
                    video_data,
                    max_frames=300,
                    target_fps=30,
                )
            else:
                # Конвертируем image_sequence в frames
                frames = await self._load_image_sequence(image_sequence)

            if len(frames) < self.min_frames_required:
                raise ProcessingError(
                    f"Insufficient frames: got {len(frames)}, need {self.min_frames_required}"
                )

            # 2. Оценка качества видео
            video_quality = await asyncio.to_thread(calculate_video_quality, frames)

            if video_quality["quality_score"] < 0.3:
                raise ProcessingError(
                    f"Video quality too low: {video_quality['quality_score']:.2f}"
                )

            # 3. Детекция landmarks для всех кадров
            landmarks_sequence = await self._extract_landmarks_sequence(frames)

            if len(landmarks_sequence) < self.min_frames_required:
                raise ProcessingError(
                    f"Face not detected in enough frames: {len(landmarks_sequence)}/{len(frames)}"
                )

            # 4. Проверка окклюзий
            occlusion_result = await self._check_occlusions(
                frames[0], landmarks_sequence[0]
            )

            # 5. Выполняем специфичную для челленджа детекцию
            challenge_result = await self._verify_specific_challenge(
                session.challenge_type,
                session.parameters,
                frames,
                landmarks_sequence,
            )

            # 6. Пассивная liveness проверка (для дополнительной безопасности)
            passive_result = await self._run_passive_liveness(frames[0])

            # 7. Комбинированная оценка
            combined_score = self._calculate_combined_score(
                challenge_result,
                passive_result,
                occlusion_result,
            )

            # 8. Финальное решение
            liveness_detected = combined_score["overall_score"] > 0.7
            challenge_completed = challenge_result["success"]

            success = liveness_detected and challenge_completed

            # Обновляем статус челленджа
            session.status = "completed" if success else "failed"
            session.result = {
                "liveness_detected": liveness_detected,
                "challenge_completed": challenge_completed,
                "confidence": combined_score["overall_score"],
            }

            # Статистика
            if success:
                self.stats["challenges_completed"] += 1
            else:
                self.stats["challenges_failed"] += 1

            processing_time = asyncio.get_event_loop().time() - start_time

            logger.info(
                f"Challenge {challenge_id} verified: success={success}, "
                f"type={session.challenge_type}, time={processing_time:.2f}s"
            )

            # Формируем ответ
            response = ActiveLivenessVerifyResponse(
                success=success,
                liveness_detected=liveness_detected,
                challenge_completed=challenge_completed,
                confidence=combined_score["overall_score"],
                passive_liveness_score=passive_result["liveness_score"],
                anti_spoofing_score=passive_result["anti_spoofing_score"],
                video_quality=video_quality,
                processing_time_seconds=processing_time,
                frames_analyzed=len(frames),
                failure_reasons=challenge_result.get("failure_reasons", []),
                warnings=challenge_result.get("warnings", []),
            )

            # Добавляем специфичные результаты
            if session.challenge_type == "blink":
                response.blink_result = challenge_result.get("blink_result")
            elif session.challenge_type in [
                "turn_head_left",
                "turn_head_right",
                "turn_head_up",
                "turn_head_down",
            ]:
                response.head_movement_result = challenge_result.get(
                    "head_movement_result"
                )

            response.occlusion_result = occlusion_result

            return response

        except Exception as e:
            session.status = "failed"
            logger.error(f"Challenge verification failed: {str(e)}")
            raise

    # ========================================================================
    # Specific Challenge Verification
    # ========================================================================

    async def _verify_specific_challenge(
        self,
        challenge_type: ChallengeType,
        parameters: Dict[str, Any],
        frames: List[np.ndarray],
        landmarks_sequence: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Верификация специфичного типа челленджа."""

        if challenge_type == "blink":
            return await self._verify_blink_challenge(parameters, landmarks_sequence)

        elif challenge_type == "smile":
            return await self._verify_smile_challenge(parameters, landmarks_sequence)

        elif challenge_type in [
            "turn_head_left",
            "turn_head_right",
            "turn_head_up",
            "turn_head_down",
        ]:
            return await self._verify_head_movement_challenge(
                challenge_type,
                parameters,
                landmarks_sequence,
            )

        elif challenge_type == "open_mouth":
            return await self._verify_mouth_open_challenge(
                parameters, landmarks_sequence
            )

        else:
            raise ValidationError(f"Unknown challenge type: {challenge_type}")

    async def _verify_blink_challenge(
        self,
        parameters: Dict[str, Any],
        landmarks_sequence: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Верификация моргания."""

        min_blinks = parameters.get("min_blinks", 2)
        fps = 30.0  # Assumed FPS

        success, blink_count, stats = await asyncio.to_thread(
            detect_blinks_in_sequence,
            landmarks_sequence,
            fps=fps,
            min_blinks=min_blinks,
        )

        blink_result = BlinkDetectionResult(
            blinks_detected=blink_count,
            blinks_required=min_blinks,
            success=success,
            confidence=min(1.0, blink_count / min_blinks),
            average_ear=stats.get("avg_ear", 0.0),
            blink_frames=stats.get("blink_frames", []),
            total_frames_analyzed=stats.get("total_frames", 0),
            quality_issues=[],
        )

        failure_reasons = []
        if not success:
            if blink_count == 0:
                failure_reasons.append("no_blinks_detected")
            else:
                failure_reasons.append(
                    f"insufficient_blinks: {blink_count}/{min_blinks}"
                )

        return {
            "success": success,
            "blink_result": blink_result,
            "failure_reasons": failure_reasons,
            "warnings": [],
        }

    async def _verify_head_movement_challenge(
        self,
        challenge_type: ChallengeType,
        parameters: Dict[str, Any],
        landmarks_sequence: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Верификация поворота головы."""

        min_angle = parameters.get("min_angle", 15.0)
        direction = parameters.get("direction", "left")

        # Вычисляем Euler angles для каждого кадра
        euler_angles = []
        for landmarks in landmarks_sequence:
            angles = await asyncio.to_thread(self._calculate_euler_angles, landmarks)
            euler_angles.append(angles)

        # Находим максимальный угол поворота в нужном направлении
        if direction == "left":
            yaw_angles = [angles["yaw"] for angles in euler_angles]
            max_angle = max(yaw_angles)
            detected = max_angle > min_angle
        elif direction == "right":
            yaw_angles = [angles["yaw"] for angles in euler_angles]
            max_angle = abs(min(yaw_angles))
            detected = max_angle > min_angle
        elif direction == "up":
            pitch_angles = [angles["pitch"] for angles in euler_angles]
            max_angle = max(pitch_angles)
            detected = max_angle > min_angle
        elif direction == "down":
            pitch_angles = [angles["pitch"] for angles in euler_angles]
            max_angle = abs(min(pitch_angles))
            detected = max_angle > min_angle
        else:
            detected = False
            max_angle = 0.0

        head_movement_result = HeadMovementResult(
            movement_detected=detected,
            movement_type=direction,
            angle_degrees=max_angle,
            required_angle=min_angle,
            confidence=min(1.0, max_angle / min_angle) if detected else 0.5,
            yaw=euler_angles[-1]["yaw"],
            pitch=euler_angles[-1]["pitch"],
            roll=euler_angles[-1]["roll"],
            frames_analyzed=len(landmarks_sequence),
        )

        failure_reasons = []
        if not detected:
            failure_reasons.append(
                f"insufficient_head_movement: {max_angle:.1f}°/{min_angle:.1f}°"
            )

        return {
            "success": detected,
            "head_movement_result": head_movement_result,
            "failure_reasons": failure_reasons,
            "warnings": [],
        }

    async def _verify_smile_challenge(
        self,
        parameters: Dict[str, Any],
        landmarks_sequence: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Верификация улыбки с использованием mouth_detector."""

        intensity_threshold = parameters.get("intensity_threshold", 0.6)
        min_duration_frames = parameters.get("min_duration_frames", 15)

        # Используем detect_smile_in_sequence из mouth_detector
        success, smile_intensity, stats = await asyncio.to_thread(
            detect_smile_in_sequence,
            landmarks_sequence,
            min_intensity=intensity_threshold,
            min_frames_with_smile=min_duration_frames,
        )

        # Извлекаем данные из stats
        smile_frames = stats.get("smile_frames", [])
        max_mar = stats.get("max_mar", 0.0)
        avg_mar = stats.get("avg_mar", 0.0)

        smile_result = SmileDetectionResult(
            smile_detected=success,
            confidence=smile_intensity if success else 0.5,
            mouth_aspect_ratio=avg_mar,
            smile_intensity=smile_intensity,
            frames_with_smile=smile_frames,
            total_frames=stats.get("total_frames", len(landmarks_sequence)),
        )

        failure_reasons = []
        if not success:
            if len(smile_frames) == 0:
                failure_reasons.append("no_smile_detected")
            else:
                failure_reasons.append(
                    f"smile_too_short: {len(smile_frames)}/{min_duration_frames} frames"
                )

        return {
            "success": success,
            "smile_result": smile_result,
            "failure_reasons": failure_reasons,
            "warnings": [],
        }

    async def _verify_mouth_open_challenge(
        self,
        parameters: Dict[str, Any],
        landmarks_sequence: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Верификация открытого рта с использованием mouth_detector."""

        mar_threshold = parameters.get("mouth_aspect_ratio_threshold", 0.5)
        min_duration_frames = parameters.get("min_duration_frames", 10)

        # Используем detect_open_mouth_in_sequence из mouth_detector
        success, max_mar, stats = await asyncio.to_thread(
            detect_open_mouth_in_sequence,
            landmarks_sequence,
            min_mar=mar_threshold,
            min_frames_with_open_mouth=min_duration_frames,
        )

        # Извлекаем данные из stats
        open_frames = stats.get("open_frames", [])
        avg_mar = stats.get("avg_mar", 0.0)

        failure_reasons = []
        if not success:
            if len(open_frames) == 0:
                failure_reasons.append("mouth_not_opened")
            else:
                failure_reasons.append(
                    f"mouth_open_too_short: {len(open_frames)}/{min_duration_frames} frames"
                )

        return {
            "success": success,
            "mouth_open_result": {
                "detected": success,
                "mar": max_mar,
                "avg_mar": avg_mar,
                "frames_with_open_mouth": open_frames,
                "total_frames": stats.get("total_frames", len(landmarks_sequence)),
            },
            "failure_reasons": failure_reasons,
            "warnings": [],
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _load_image_sequence(
        self, image_sequence: List[bytes]
    ) -> List[np.ndarray]:
        """Загрузка последовательности изображений."""
        import io

        from PIL import Image

        frames = []
        for img_data in image_sequence:
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            frame = np.array(img)
            frames.append(frame)

        return frames

    async def _extract_landmarks_sequence(
        self,
        frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Извлечение landmarks для каждого кадра."""

        landmarks_sequence = []

        for frame in frames:
            landmarks = await asyncio.to_thread(detect_face_landmarks, frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)

        return landmarks_sequence

    async def _check_occlusions(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> OcclusionDetectionResult:
        """Проверка окклюзий."""

        detector = get_occlusion_detector()
        result = await asyncio.to_thread(detector.detect_occlusions, frame, landmarks)

        return OcclusionDetectionResult(
            has_mask=result.has_mask,
            has_sunglasses=result.has_sunglasses,
            has_regular_glasses=result.has_regular_glasses,
            has_vr_headset=result.has_vr_headset,
            has_hand_covering=result.has_hand_covering,
            occlusion_score=result.occlusion_score,
            confidence=result.confidence,
            details=result.details,
        )

    async def _run_passive_liveness(self, frame: np.ndarray) -> Dict[str, float]:
        """Пассивная liveness проверка (для дополнительной безопасности)."""

        # Импортируем ML service
        from .ml_service import get_ml_service

        ml_service = await get_ml_service()

        # Конвертируем frame в bytes
        import io

        from PIL import Image

        pil_image = Image.fromarray(frame)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()

        # Запускаем liveness check
        result = await ml_service.check_liveness(img_bytes, use_3d_depth=True)

        return {
            "liveness_score": result.get("confidence", 0.5),
            "anti_spoofing_score": result.get("anti_spoofing_score", 0.5),
        }

    def _calculate_combined_score(
        self,
        challenge_result: Dict[str, Any],
        passive_result: Dict[str, float],
        occlusion_result: OcclusionDetectionResult,
    ) -> Dict[str, float]:
        """Комбинированная оценка живости."""

        # Веса компонентов
        weights = {
            "challenge": 0.5,  # Active challenge
            "passive": 0.3,  # Passive liveness
            "anti_spoofing": 0.15,  # Anti-spoofing
            "occlusion": 0.05,  # Occlusion check
        }

        # Оценки компонентов
        challenge_score = 1.0 if challenge_result["success"] else 0.3
        passive_score = passive_result["liveness_score"]
        anti_spoofing_score = passive_result["anti_spoofing_score"]
        occlusion_score = occlusion_result.occlusion_score

        # Взвешенная сумма
        overall_score = (
            challenge_score * weights["challenge"]
            + passive_score * weights["passive"]
            + anti_spoofing_score * weights["anti_spoofing"]
            + occlusion_score * weights["occlusion"]
        )

        return {
            "overall_score": overall_score,
            "challenge_score": challenge_score,
            "passive_score": passive_score,
            "anti_spoofing_score": anti_spoofing_score,
            "occlusion_score": occlusion_score,
        }

    def _calculate_euler_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Вычисление Euler angles из landmarks."""

        # Упрощенный метод через PnP
        # Модельные 3D координаты ключевых точек лица
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
            ]
        )

        # Соответствующие индексы в 68-point landmarks
        image_points = np.array(
            [
                landmarks[30],  # Nose tip
                landmarks[8],  # Chin
                landmarks[36],  # Left eye left corner
                landmarks[45],  # Right eye right corner
                landmarks[48],  # Left mouth corner
                landmarks[54],  # Right mouth corner
            ],
            dtype="double",
        )

        # Camera internals (assumed)
        focal_length = 1.0
        center = (0.0, 0.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        import cv2

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # Конвертируем в Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Извлекаем углы
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0

        # Конвертируем в градусы
        return {
            "yaw": np.degrees(yaw),
            "pitch": np.degrees(pitch),
            "roll": np.degrees(roll),
        }

    def _count_consecutive_frames(self, frame_indices: List[int]) -> int:
        """Подсчет максимальной последовательности кадров."""

        if not frame_indices:
            return 0

        frame_indices = sorted(frame_indices)
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(frame_indices)):
            if frame_indices[i] == frame_indices[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return max_consecutive

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики сервиса."""
        return {
            **self.stats,
            "active_challenges": len(self._active_challenges),
            "success_rate": (
                self.stats["challenges_completed"]
                / max(
                    1,
                    self.stats["challenges_completed"]
                    + self.stats["challenges_failed"],
                )
            ),
        }

    # ========================================================================
    # Active Liveness Check (улучшенная версия)
    # ========================================================================

    async def active_liveness_check(
        self,
        video_frames: List[bytes],
        instructions: List[str],
        require_liveness: bool = True,
    ) -> Dict[str, Any]:
        """
        Активная проверка живости с детекцией выполнения инструкций.

        Args:
            video_frames: Список кадров видео (bytes JPEG)
            instructions: Список инструкций для выполнения
                ["turn_left", "turn_right", "smile", "blink", "open_mouth", "look_up", "look_down"]
            require_liveness: Требовать пассивную проверку живости

        Returns:
            Dict с результатами:
            - passed: bool - все инструкции выполнены + все кадры живые
            - results: список результатов по каждому кадру/инструкции
            - overall_score: float - общая оценка (0-1)
            - failed_instructions: список невыполненных инструкций
        """
        start_time = asyncio.get_event_loop().time()

        if not video_frames or not instructions:
            raise ValidationError("video_frames and instructions must not be empty")

        if len(video_frames) < len(instructions):
            raise ValidationError(
                f"Got {len(video_frames)} frames for {len(instructions)} instructions"
            )

        logger.info(
            f"Starting active liveness check: {len(instructions)} instructions, "
            f"{len(video_frames)} frames"
        )

        try:
            # 1. Конвертация frames в numpy arrays
            frames_np = await self._frames_bytes_to_numpy(video_frames)

            # 2. Оценка качества видео
            video_quality = self._assess_video_quality(frames_np)
            if video_quality["quality_score"] < 0.4:
                logger.warning(f"Video quality too low: {video_quality}")

            # 3. Извлечение landmarks для всех кадров
            all_landmarks = await self._extract_all_landmarks(frames_np)

            if len(all_landmarks) < len(video_frames) * 0.7:
                raise ProcessingError(
                    f"Face not detected in enough frames: {len(all_landmarks)}/{len(video_frames)}"
                )

            # 4. Проверка окклюзий на первом кадре
            occlusion_result = await self._check_occlusions(
                frames_np[0], all_landmarks[0] if all_landmarks else None
            )

            # 5. Пассивная liveness проверка
            passive_result = None
            if require_liveness:
                passive_result = await self._run_passive_liveness(frames_np[0])

            # 6. Проверка каждой инструкции
            instruction_results = []

            for idx, instruction in enumerate(instructions):
                # Получаем соответствующий кадр (или последний если не хватает)
                frame_idx = min(idx, len(frames_np) - 1)
                frame_landmarks = (
                    all_landmarks[frame_idx] if frame_idx < len(all_landmarks) else None
                )

                # Проверяем действие
                action_result = await self._detect_action(
                    instruction, frames_np[frame_idx], frame_landmarks
                )

                # Пассивная проверка живости для этого кадра
                liveness_result = None
                if require_liveness:
                    liveness_result = await self._run_passive_liveness(
                        frames_np[frame_idx]
                    )

                instruction_results.append(
                    {
                        "instruction": instruction,
                        "instruction_index": idx,
                        "action_detected": action_result["detected"],
                        "action_confidence": action_result.get("confidence", 0.0),
                        "action_details": action_result.get("details", {}),
                        "liveness": liveness_result,
                    }
                )

            # 7. Комбинированная оценка
            action_scores = [r["action_confidence"] for r in instruction_results]
            avg_action_score = np.mean(action_scores) if action_scores else 0.0

            liveness_scores = [
                r["liveness"]["liveness_score"] if r["liveness"] else 0.5
                for r in instruction_results
                if r["liveness"]
            ]
            avg_liveness_score = np.mean(liveness_scores) if liveness_scores else 0.5

            # Финальная оценка
            overall_score = (
                avg_action_score * 0.5
                + avg_liveness_score * 0.3
                + (1 - occlusion_result.occlusion_score) * 0.2
            )

            # Определение успеха
            all_actions_passed = all(r["action_detected"] for r in instruction_results)
            all_liveness_passed = all(
                r["liveness"]["liveness_detected"] if r["liveness"] else True
                for r in instruction_results
            )

            passed = all_actions_passed and (
                not require_liveness or all_liveness_passed
            )

            # Список неудачных инструкций
            failed_instructions = [
                r["instruction"]
                for r in instruction_results
                if not r["action_detected"]
            ]

            processing_time = asyncio.get_event_loop().time() - start_time

            return {
                "passed": passed,
                "overall_score": round(overall_score, 4),
                "action_score": round(avg_action_score, 4),
                "liveness_score": round(avg_liveness_score, 4),
                "instructions_passed": len(instructions) - len(failed_instructions),
                "instructions_total": len(instructions),
                "failed_instructions": failed_instructions,
                "results": instruction_results,
                "occlusion_check": {
                    "occlusion_detected": occlusion_result.occlusion_score > 0.5,
                    "occlusion_score": occlusion_result.occlusion_score,
                },
                "video_quality": video_quality,
                "processing_time": round(processing_time, 4),
                "model_type": "ActiveLiveness",
            }

        except Exception as e:
            logger.error(f"Active liveness check failed: {str(e)}")
            raise ProcessingError(f"Active liveness check failed: {str(e)}")

    async def _detect_action(
        self,
        instruction: str,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Детекция выполнения конкретной инструкции.

        Args:
            instruction: Тип инструкции
            frame: Кадр изображения
            landmarks: landmarks лица

        Returns:
            Dict с detected, confidence, details
        """
        instruction = instruction.lower()

        # Проверка наличия landmarks
        if landmarks is None:
            # Пытаемся получить landmarks
            landmarks = await asyncio.to_thread(detect_face_landmarks, frame)
            if landmarks is None:
                return {
                    "detected": False,
                    "confidence": 0.0,
                    "error": "No face detected",
                }

        # Детекция по типу инструкции
        if "blink" in instruction or "глаз" in instruction:
            return await self._detect_blink_action(landmarks)

        elif "smile" in instruction or "улыб" in instruction:
            return await self._detect_smile_action(landmarks)

        elif (
            "turn_left" in instruction
            or "поверните_влево" in instruction
            or "left" in instruction
        ):
            return await self._detect_head_turn_action(landmarks, "left")

        elif (
            "turn_right" in instruction
            or "поверните_вправо" in instruction
            or "right" in instruction
        ):
            return await self._detect_head_turn_action(landmarks, "right")

        elif "look_up" in instruction or "вверх" in instruction or "up" in instruction:
            return await self._detect_head_turn_action(landmarks, "up")

        elif (
            "look_down" in instruction or "вниз" in instruction or "down" in instruction
        ):
            return await self._detect_head_turn_action(landmarks, "down")

        elif (
            "open_mouth" in instruction
            or "рот" in instruction
            or "mouth" in instruction
        ):
            return await self._detect_mouth_open_action(landmarks)

        else:
            return {
                "detected": False,
                "confidence": 0.0,
                "error": f"Unknown instruction: {instruction}",
            }

    async def _detect_blink_action(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Детекция моргания на одном кадре."""
        # EAR (Eye Aspect Ratio)
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]

        def calc_ear(eye):
            # Вертикальные расстояния
            v1 = np.linalg.norm(eye[1] - eye[5])
            v2 = np.linalg.norm(eye[2] - eye[4])
            # Горизонтальное расстояние
            h = np.linalg.norm(eye[0] - eye[3])
            return (v1 + v2) / (2 * h + 1e-6)

        left_ear = calc_ear(left_eye)
        right_ear = calc_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2

        # Порог для моргания (обычно < 0.21)
        is_blinking = avg_ear < self.blink_ear_threshold

        return {
            "detected": is_blinking,
            "confidence": 1.0 - min(avg_ear / 0.21, 1.0),
            "details": {
                "eye_aspect_ratio": round(avg_ear, 4),
                "threshold": self.blink_ear_threshold,
            },
        }

    async def _detect_smile_action(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Детекция улыбки с использованием mouth_detector."""
        # Используем calculate_mouth_aspect_ratio из mouth_detector
        mar = calculate_mouth_aspect_ratio(landmarks)

        # Улыбка при MAR > 0.25
        is_smile = mar > 0.25

        return {
            "detected": is_smile,
            "confidence": min(mar / 0.5, 1.0),
            "details": {
                "mouth_aspect_ratio": round(mar, 4),
                "threshold": 0.25,
            },
        }

    async def _detect_head_turn_action(
        self,
        landmarks: np.ndarray,
        direction: str,
    ) -> Dict[str, Any]:
        """Детекция поворота головы."""
        # Вычисляем yaw из landmarks
        nose = landmarks[30]
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)

        # Yaw angle
        eye_center = (left_eye + right_eye) / 2
        nose_offset = nose[0] - eye_center[0]

        # Приблизительный угол поворота
        yaw = np.degrees(np.arctan2(nose_offset, eye_center[0] - nose[0]))

        if direction == "left":
            detected = yaw > self.head_movement_min_angle
            angle = yaw
        elif direction == "right":
            detected = yaw < -self.head_movement_min_angle
            angle = abs(yaw)
        elif direction == "up":
            # Pitch вверх - нос выше центра между глазами
            nose_y = nose[1]
            eye_center_y = eye_center[1]
            detected = nose_y < eye_center_y - 10
            angle = abs(eye_center_y - nose_y)
        elif direction == "down":
            detected = nose_y > eye_center_y + 10
            angle = abs(nose_y - eye_center_y)
        else:
            detected = False
            angle = 0.0

        return {
            "detected": detected,
            "confidence": min(angle / 30.0, 1.0),
            "details": {
                "direction": direction,
                "angle_degrees": round(angle, 2),
                "required_angle": self.head_movement_min_angle,
            },
        }

    async def _detect_mouth_open_action(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Детекция открытого рта с использованием mouth_detector."""
        # Используем calculate_mouth_aspect_ratio из mouth_detector
        mar = calculate_mouth_aspect_ratio(landmarks)

        is_open = mar > 0.5

        return {
            "detected": is_open,
            "confidence": min(mar / 0.7, 1.0),
            "details": {
                "mouth_aspect_ratio": round(mar, 4),
                "threshold": 0.5,
            },
        }

    async def _frames_bytes_to_numpy(self, frames: List[bytes]) -> List[np.ndarray]:
        """Конвертация bytes в numpy arrays."""
        import io

        from PIL import Image

        result = []
        for frame_bytes in frames:
            img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            result.append(np.array(img))

        return result

    async def _extract_all_landmarks(
        self,
        frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Извлечение landmarks для всех кадров."""
        landmarks_list = []

        for frame in frames:
            landmarks = await asyncio.to_thread(detect_face_landmarks, frame)
            if landmarks is not None:
                landmarks_list.append(landmarks)

        return landmarks_list

    def _assess_video_quality(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Оценка качества видео."""
        if not frames:
            return {"quality_score": 0.0, "blur_score": 0.0, "brightness_score": 0.0}

        # Оценка на первом кадре (достаточно)
        frame = frames[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Резкость
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.std(laplacian) / 500.0

        # Яркость
        brightness = np.mean(gray) / 255.0

        # Комбинированная оценка
        quality_score = min(blur_score, 1.0) * 0.5 + abs(brightness - 0.5) * 2 * 0.5

        return {
            "quality_score": round(min(quality_score, 1.0), 4),
            "blur_score": round(min(blur_score, 1.0), 4),
            "brightness_score": round(brightness, 4),
        }


# Singleton
_active_liveness_service: Optional[ActiveLivenessService] = None


async def get_active_liveness_service() -> ActiveLivenessService:
    """Get or create singleton service."""
    global _active_liveness_service
    if _active_liveness_service is None:
        _active_liveness_service = ActiveLivenessService()
    return _active_liveness_service
