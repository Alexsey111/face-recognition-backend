"""
API роуты для проверки живости (Liveness Detection).

Endpoints:
- POST /liveness/check - Пассивная liveness (одно изображение)
- POST /liveness/video - Пассивная liveness (видео)
- POST /liveness/active/challenge - Создание Active Liveness челленджа
- POST /liveness/active/verify - Верификация Active Liveness челленджа
- POST /liveness/blink - Специфичная детекция моргания
- POST /liveness/head-movement - Детекция движения головы
- GET /liveness/active/stats - Статистика Active Liveness
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Body
from typing import Optional, List
import base64

from ..models.response import BaseResponse
from ..models.request import LivenessRequest, VideoLivenessRequest
from ..models.active_liveness import (
    ActiveLivenessChallengeRequest,
    ActiveLivenessVerifyRequest,
    ActiveLivenessChallengeResponse,
    ActiveLivenessVerifyResponse,
    BlinkDetectionRequest,
    HeadMovementRequest,
    ChallengeType,
)
from ..services.ml_service import get_ml_service
from ..services.active_liveness_service import get_active_liveness_service
from ..dependencies import get_current_user
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError

router = APIRouter(prefix="/liveness", tags=["liveness"])
logger = get_logger(__name__)


# ============================================================================
# PASSIVE LIVENESS (существующие endpoints)
# ============================================================================

@router.post("/check", response_model=BaseResponse)
async def check_liveness(
    file: UploadFile = File(..., description="Изображение для проверки живости"),
    challenge_type: str = Form(default="passive", description="Тип челленджа"),
    user: dict = Depends(get_current_user),
):
    """
    Проверка живости по одному изображению (пассивная проверка).
    
    Features:
    - MiniFASNetV2 anti-spoofing (>98% accuracy)
    - 3D depth analysis
    - Lighting/shadow analysis
    - Texture analysis
    
    **Методы:**
    - Certified: MiniFASNetV2 (если включено)
    - Heuristic: 3D depth + lighting (fallback)
    """
    try:
        # Проверка типа файла
        if not file.content_type.startswith("image/"):
            raise ValidationError(f"Invalid file type: {file.content_type}. Expected image.")
        
        # Читаем изображение
        image_data = await file.read()
        
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise ValidationError("Image size exceeds 10MB limit")
        
        # Получаем ML сервис
        ml_service = await get_ml_service()
        
        # Проверка живости
        result = await ml_service.check_liveness(
            image_data=image_data,
            challenge_type=challenge_type,
            use_3d_depth=True,
        )
        
        logger.info(
            f"Liveness check: user={user['user_id']}, "
            f"detected={result.get('liveness_detected')}, "
            f"confidence={result.get('confidence'):.3f}"
        )
        
        return BaseResponse(
            success=True,
            message="Liveness check completed",
            data=result,
        )
        
    except ValidationError as e:
        logger.warning(f"Liveness validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        logger.error(f"Liveness processing error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/video", response_model=BaseResponse)
async def check_video_liveness(
    file: UploadFile = File(..., description="Видео для проверки живости (MP4, WebM)"),
    challenge_type: str = Form(default="passive", description="Тип челленджа"),
    max_frames: int = Form(default=30, description="Максимум кадров для анализа"),
    user: dict = Depends(get_current_user),
):
    """
    Проверка живости по видео (пассивная проверка).
    
    Анализирует несколько кадров видео для более точной детекции.
    """
    try:
        # Проверка типа файла
        if not file.content_type.startswith("video/"):
            raise ValidationError(f"Invalid file type: {file.content_type}. Expected video.")
        
        # Читаем видео
        video_data = await file.read()
        
        if len(video_data) > 50 * 1024 * 1024:  # 50MB limit
            raise ValidationError("Video size exceeds 50MB limit")
        
        # Получаем ML сервис
        ml_service = await get_ml_service()
        
        # Анализ видео (пока используем первый кадр, TODO: полный анализ)
        from ..utils.video_processing import extract_frames_from_video
        
        frames = await extract_frames_from_video(
            video_data,
            max_frames=max_frames,
            target_fps=10,
        )
        
        if not frames:
            raise ProcessingError("No frames could be extracted from video")
        
        # Проверяем первый и последний кадр
        from PIL import Image
        import io
        
        first_frame = Image.fromarray(frames[0])
        img_byte_arr = io.BytesIO()
        first_frame.save(img_byte_arr, format='JPEG')
        first_frame_bytes = img_byte_arr.getvalue()
        
        result = await ml_service.check_liveness(
            image_data=first_frame_bytes,
            challenge_type=challenge_type,
            use_3d_depth=True,
        )
        
        result["frames_analyzed"] = len(frames)
        result["video_duration_frames"] = len(frames)
        
        logger.info(
            f"Video liveness check: user={user['user_id']}, "
            f"frames={len(frames)}, "
            f"detected={result.get('liveness_detected')}"
        )
        
        return BaseResponse(
            success=True,
            message="Video liveness check completed",
            data=result,
        )
        
    except ValidationError as e:
        logger.warning(f"Video liveness validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        logger.error(f"Video liveness processing error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Video liveness check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# ACTIVE LIVENESS (новые endpoints)
# ============================================================================

@router.post("/active/challenge", response_model=ActiveLivenessChallengeResponse)
async def create_active_liveness_challenge(
    request: ActiveLivenessChallengeRequest,
    user: dict = Depends(get_current_user),
):
    """
    Создание Active Liveness челленджа.
    
    **Challenge Types:**
    - `blink` - Моргание (2-3 раза)
    - `smile` - Улыбка
    - `turn_head_left` - Поворот головы влево
    - `turn_head_right` - Поворот головы вправо
    - `turn_head_up` - Наклон головы вверх
    - `turn_head_down` - Наклон головы вниз
    - `open_mouth` - Открыть рот
    - `random` - Случайный челлендж (генерируется сервером)
    
    **Difficulty:**
    - `easy` - Легко (меньше требований)
    - `medium` - Средне (стандарт)
    - `hard` - Сложно (строгие требования)
    
    **Response содержит:**
    - `challenge_id` - ID для последующей верификации
    - `instruction` - Инструкция для пользователя
    - `expires_at` - Время истечения челленджа
    
    **Пример использования:**
    1. Создайте челлендж (получите `challenge_id`)
    2. Покажите пользователю инструкцию
    3. Запишите видео выполнения
    4. Отправьте на `/active/verify` с `challenge_id`
    """
    try:
        service = await get_active_liveness_service()
        
        response = await service.create_challenge(
            challenge_type=request.challenge_type,
            timeout_seconds=request.timeout_seconds,
            difficulty=request.difficulty,
        )
        
        logger.info(
            f"Challenge created: user={user['user_id']}, "
            f"type={request.challenge_type}, "
            f"id={response.challenge_id}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to create challenge: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create challenge")


@router.post("/active/verify", response_model=ActiveLivenessVerifyResponse)
async def verify_active_liveness_challenge(
    challenge_id: str = Form(..., description="ID челленджа"),
    file: Optional[UploadFile] = File(None, description="Видео выполнения челленджа"),
    metadata: Optional[str] = Form(None, description="Метаданные в JSON"),
    user: dict = Depends(get_current_user),
):
    """
    Верификация выполнения Active Liveness челленджа.
    
    **Требования:**
    - Видео должно содержать выполнение челленджа
    - Минимум 20 кадров (0.5-1 секунда при 30 FPS)
    - Качество: резкость, освещение, видимость лица
    - Лицо должно быть видно на всех кадрах
    
    **Проверки:**
    1. **Active Challenge** - выполнение конкретного действия
    2. **Passive Liveness** - MiniFASNetV2 anti-spoofing
    3. **Occlusion Detection** - маски, очки, руки
    4. **Video Quality** - резкость, освещение, стабильность
    
    **Response:**
    - `liveness_detected` - общая живость
    - `challenge_completed` - выполнен ли челлендж
    - `confidence` - уверенность (0-1)
    - Детальные результаты каждой проверки
    """
    try:
        if not file:
            raise ValidationError("Video file is required")
        
        # Проверка типа файла
        if not file.content_type.startswith("video/"):
            raise ValidationError(f"Invalid file type: {file.content_type}. Expected video.")
        
        # Читаем видео
        video_data = await file.read()
        
        if len(video_data) > 100 * 1024 * 1024:  # 100MB limit
            raise ValidationError("Video size exceeds 100MB limit")
        
        # Парсим метаданные
        import json
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metadata: {metadata}")
        
        # Верификация
        service = await get_active_liveness_service()
        
        response = await service.verify_challenge(
            challenge_id=challenge_id,
            video_data=video_data,
            metadata=metadata_dict,
        )
        
        logger.info(
            f"Challenge verified: user={user['user_id']}, "
            f"id={challenge_id}, "
            f"success={response.success}, "
            f"confidence={response.confidence:.3f}"
        )
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Challenge verification validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        logger.error(f"Challenge verification processing error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Challenge verification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/blink", response_model=BaseResponse)
async def detect_blink(
    file: UploadFile = File(..., description="Видео с морганием"),
    min_blinks: int = Form(default=1, ge=1, le=5, description="Минимум морганий"),
    user: dict = Depends(get_current_user),
):
    """
    Детекция моргания в видео (без создания челленджа).
    
    **Прямой метод** для детекции моргания без предварительного создания челленджа.
    
    **Параметры:**
    - `min_blinks` - Требуемое количество морганий (1-5)
    
    **Алгоритм:**
    - Eye Aspect Ratio (EAR) по методу Soukupová & Čech
    - Последовательность: глаз открыт → закрыт → открыт
    - Проверка длительности (100-400ms)
    
    **Response:**
    - Количество обнаруженных морганий
    - Уверенность детекции
    - Номера кадров с морганиями
    """
    try:
        # Читаем видео
        video_data = await file.read()
        
        if len(video_data) > 50 * 1024 * 1024:
            raise ValidationError("Video size exceeds 50MB limit")
        
        # Извлекаем кадры
        from ..utils.video_processing import extract_frames_from_video
        
        frames = extract_frames_from_video(video_data, max_frames=300, target_fps=30)
        
        if len(frames) < 10:
            raise ProcessingError("Insufficient frames in video")
        
        # Детекция landmarks
        from ..utils.face_alignment_utils import detect_face_landmarks
        
        landmarks_sequence = []
        for frame in frames:
            landmarks = detect_face_landmarks(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
        
        if len(landmarks_sequence) < 10:
            raise ProcessingError("Face not detected in enough frames")
        
        # Детекция морганий
        from ..utils.eye_blink_detector import detect_blinks_in_sequence
        
        success, blink_count, stats = detect_blinks_in_sequence(
            landmarks_sequence,
            fps=30.0,
            min_blinks=min_blinks,
        )
        
        result = {
            "blinks_detected": blink_count,
            "blinks_required": min_blinks,
            "success": success,
            "confidence": min(1.0, blink_count / min_blinks),
            "blink_frames": stats.get("blink_frames", []),
            "total_frames": len(landmarks_sequence),
            "average_ear": stats.get("avg_ear", 0.0),
        }
        
        logger.info(
            f"Blink detection: user={user['user_id']}, "
            f"blinks={blink_count}/{min_blinks}, "
            f"success={success}"
        )
        
        return BaseResponse(
            success=True,
            message="Blink detection completed",
            data=result,
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Blink detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/head-movement", response_model=BaseResponse)
async def detect_head_movement(
    file: UploadFile = File(..., description="Видео с движением головы"),
    movement_type: str = Form(..., description="Тип движения: left/right/up/down"),
    min_angle: float = Form(default=15.0, description="Минимальный угол (градусы)"),
    user: dict = Depends(get_current_user),
):
    """
    Детекция движения головы в видео.
    
    **Типы движений:**
    - `left` - Поворот влево (yaw > 0)
    - `right` - Поворот вправо (yaw < 0)
    - `up` - Наклон вверх (pitch > 0)
    - `down` - Наклон вниз (pitch < 0)
    
    **Алгоритм:**
    - PnP pose estimation с 6 ключевыми точками
    - Вычисление Euler angles (yaw, pitch, roll)
    - Проверка максимального угла в нужном направлении
    
    **Response:**
    - Обнаружено ли движение
    - Угол поворота (градусы)
    - Euler angles (yaw, pitch, roll)
    """
    try:
        # Проверка типа движения
        valid_movements = ["left", "right", "up", "down"]
        if movement_type not in valid_movements:
            raise ValidationError(f"Invalid movement_type. Must be one of: {valid_movements}")
        
        # Читаем видео
        video_data = await file.read()
        
        if len(video_data) > 50 * 1024 * 1024:
            raise ValidationError("Video size exceeds 50MB limit")
        
        # Извлекаем кадры
        from ..utils.video_processing import extract_frames_from_video
        
        frames = extract_frames_from_video(video_data, max_frames=300, target_fps=30)
        
        if len(frames) < 10:
            raise ProcessingError("Insufficient frames in video")
        
        # Детекция landmarks
        from ..utils.face_alignment_utils import detect_face_landmarks
        
        landmarks_sequence = []
        for frame in frames:
            landmarks = detect_face_landmarks(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)
        
        if len(landmarks_sequence) < 10:
            raise ProcessingError("Face not detected in enough frames")
        
        # Вычисляем углы для каждого кадра
        service = await get_active_liveness_service()
        
        euler_angles = []
        for landmarks in landmarks_sequence:
            angles = service._calculate_euler_angles(landmarks)
            euler_angles.append(angles)
        
        # Находим максимальный угол
        if movement_type == "left":
            yaw_angles = [angles["yaw"] for angles in euler_angles]
            max_angle = max(yaw_angles)
            detected = max_angle > min_angle
        elif movement_type == "right":
            yaw_angles = [angles["yaw"] for angles in euler_angles]
            max_angle = abs(min(yaw_angles))
            detected = max_angle > min_angle
        elif movement_type == "up":
            pitch_angles = [angles["pitch"] for angles in euler_angles]
            max_angle = max(pitch_angles)
            detected = max_angle > min_angle
        else:  # down
            pitch_angles = [angles["pitch"] for angles in euler_angles]
            max_angle = abs(min(pitch_angles))
            detected = max_angle > min_angle
        
        result = {
            "movement_detected": detected,
            "movement_type": movement_type,
            "angle_degrees": max_angle,
            "required_angle": min_angle,
            "confidence": min(1.0, max_angle / min_angle) if detected else 0.5,
            "euler_angles": euler_angles[-1] if euler_angles else {},
            "frames_analyzed": len(landmarks_sequence),
        }
        
        logger.info(
            f"Head movement detection: user={user['user_id']}, "
            f"type={movement_type}, "
            f"angle={max_angle:.1f}°, "
            f"detected={detected}"
        )
        
        return BaseResponse(
            success=True,
            message="Head movement detection completed",
            data=result,
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Head movement detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/active/stats", response_model=BaseResponse)
async def get_active_liveness_stats(
    user: dict = Depends(get_current_user),
):
    """
    Статистика Active Liveness сервиса.
    
    **Метрики:**
    - Всего созданных челленджей
    - Успешных/провальных верификаций
    - Success rate
    - Активные челленджи
    
    **Доступ:** только для аутентифицированных пользователей
    """
    try:
        service = await get_active_liveness_service()
        stats = service.get_stats()
        
        return BaseResponse(
            success=True,
            message="Active liveness statistics",
            data=stats,
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/supported-challenges", response_model=BaseResponse)
async def get_supported_challenges():
    """
    Список поддерживаемых типов челленджей.
    
    **Публичный endpoint** - не требует аутентификации.
    """
    challenges = [
        {
            "type": "blink",
            "name": "Моргание",
            "description": "Моргните 2-3 раза",
            "difficulty": ["easy", "medium", "hard"],
            "icon": "👁️",
        },
        {
            "type": "smile",
            "name": "Улыбка",
            "description": "Улыбнитесь",
            "difficulty": ["easy", "medium", "hard"],
            "icon": "😊",
        },
        {
            "type": "turn_head_left",
            "name": "Поворот влево",
            "description": "Поверните голову влево",
            "difficulty": ["easy", "medium", "hard"],
            "icon": "⬅️",
        },
        {
            "type": "turn_head_right",
            "name": "Поворот вправо",
            "description": "Поверните голову вправо",
            "difficulty": ["easy", "medium", "hard"],
            "icon": "➡️",
        },
        {
            "type": "turn_head_up",
            "name": "Наклон вверх",
            "description": "Наклоните голову вверх",
            "difficulty": ["easy", "medium", "hard"],
            "icon": "⬆️",
        },
        {
            "type": "turn_head_down",
            "name": "Наклон вниз",
            "description": "Наклоните голову вниз",
            "difficulty": ["easy", "medium", "hard"],
            "icon": "⬇️",
        },
        {
            "type": "open_mouth",
            "name": "Открыть рот",
            "description": "Откройте рот",
            "difficulty": ["easy", "medium", "hard"],
            "icon": "😮",
        },
        {
            "type": "random",
            "name": "Случайный",
            "description": "Сервер выберет случайный челлендж",
            "difficulty": ["medium"],
            "icon": "🎲",
        },
    ]
    
    return BaseResponse(
        success=True,
        message="Supported challenge types",
        data={
            "challenges": challenges,
            "total": len(challenges),
        },
    )

