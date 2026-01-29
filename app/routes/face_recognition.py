"""
Face Recognition API Routes - Прямое сравнение лиц.

Endpoints:
- POST /api/v1/face/verify/direct - Сравнение двух изображений
- POST /api/v1/face/liveness/check - Проверка живости (anti-spoofing)
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..dependencies import get_current_user
from ..models.response import BaseResponse
from ..services.anti_spoofing_service import get_anti_spoofing_service
from ..services.face_verification_service import get_face_verification_service
from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/face", tags=["Face Recognition"])


# =============================================================================
# Security utilities - Secure data deletion
# =============================================================================


def _secure_delete(data: bytearray) -> None:
    """
    Безопасное удаление данных из памяти.

    Перезаписывает данные случайными байтами перед удалением
    для защиты от memory forensics.
    """
    import secrets

    try:
        for i in range(len(data)):
            data[i] = secrets.randbelow(256)
        for i in range(len(data)):
            data[i] = 0
    except Exception:
        pass


async def _secure_delete_async(data: bytes) -> None:
    """
    Асинхронное безопасное удаление bytes.

    Создаёт копию для модификации, затем очищает.
    """
    import gc
    import secrets

    try:
        # Создаём mutable копию
        mutable_data = bytearray(data)
        # Перезапись
        for i in range(len(mutable_data)):
            mutable_data[i] = secrets.randbelow(256)
        for i in range(len(mutable_data)):
            mutable_data[i] = 0
        # Принудительный GC
        del mutable_data
        gc.collect()
    except Exception:
        pass


@router.post("/verify/direct", response_model=BaseResponse)
async def verify_two_faces(
    image1: UploadFile = File(..., description="Первое изображение"),
    image2: UploadFile = File(..., description="Второе изображение для сравнения"),
    threshold: float = File(0.60, ge=0.0, le=1.0, description="Порог схожести"),
    require_liveness: bool = File(False, description="Требовать проверку живости"),
    current_user: str = Depends(get_current_user),
):
    """
    Прямое сравнение двух изображений лиц.

    **Использование:**
    1. Загрузите оба изображения
    2. Получите similarity score и уровень совпадения
    3. Решение на основе threshold

    **Response:**
    - `is_match`: bool - результат сравнения
    - `similarity`: float - косинусная схожесть (0-1)
    - `match_level`: str - high/medium/low/none
    - `threshold_used`: использованный порог

    **Пример:**
    ```
    similarity > 0.80 → "high"
    similarity > 0.60 → "medium"
    similarity > 0.40 → "low"
    ```
    """
    try:
        # Валидация типов файлов
        if not image1.content_type.startswith("image/"):
            raise ValidationError(
                f"Invalid file type for image1: {image1.content_type}"
            )
        if not image2.content_type.startswith("image/"):
            raise ValidationError(
                f"Invalid file type for image2: {image2.content_type}"
            )

        # Чтение изображений
        image1_bytes = await image1.read()
        image2_bytes = await image2.read()

        try:
            # Проверка размера
            max_size = 10 * 1024 * 1024  # 10MB
            if len(image1_bytes) > max_size or len(image2_bytes) > max_size:
                raise ValidationError("Image size exceeds 10MB limit")

            # Получение сервиса
            service = await get_face_verification_service()

            # Верификация
            result = await service.verify_face(
                image1=image1_bytes,
                image2=image2_bytes,
                threshold=threshold,
                require_liveness=require_liveness,
            )

            return BaseResponse(
                success=True,
                message="Face verification completed",
                data=result,
            )
        finally:
            # Гарантированное удаление изображений из памяти
            _secure_delete(image1_bytes)
            _secure_delete(image2_bytes)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Face verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/liveness/check", response_model=BaseResponse)
async def check_liveness(
    file: UploadFile = File(..., description="Изображение для проверки живости"),
    challenge_type: str = "passive",
    current_user: str = Depends(get_current_user),
):
    """
    Проверка живости лица (anti-spoofing).

    **Методы:**
    - MiniFASNetV2: сертифицированная модель (>98% accuracy)
    - 3D Depth Analysis: оценка глубины для детекции фото/экрана
    - Lighting Analysis: анализ освещения и теней

    **Response:**
    - `liveness_detected`: bool - определено как живое лицо
    - `confidence`: float - уверенность (0-1)
    - `spoof_probability`: вероятность подделки
    """
    try:
        if not file.content_type.startswith("image/"):
            raise ValidationError(f"Invalid file type: {file.content_type}")

        image_bytes = await file.read()

        try:
            if len(image_bytes) > 10 * 1024 * 1024:
                raise ValidationError("Image size exceeds 10MB limit")

            service = await get_anti_spoofing_service()
            result = await service.check_liveness(image_bytes)

            return BaseResponse(
                success=True,
                message="Liveness check completed",
                data={
                    "liveness_detected": result.get("liveness_detected"),
                    "confidence": result.get("confidence"),
                    "spoof_probability": result.get("spoof_probability"),
                    "model_version": result.get("model_version"),
                    "processing_time": result.get("processing_time"),
                },
            )
        finally:
            # Гарантированное удаление изображения из памяти
            await _secure_delete_async(image_bytes)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/liveness/active", response_model=BaseResponse)
async def active_liveness_check(
    instructions: list,
    video_frames: list,
    require_liveness: bool = True,
    current_user: str = Depends(get_current_user),
):
    """
    Активная проверка живости с детекцией действий.

    **Аргументы:**
    - `instructions`: ["turn_left", "smile", "blink", ...]
    - `video_frames`: Бинарные данные кадров видео

    **Поддерживаемые инструкции:**
    - `blink` / `глаз` - моргание
    - `smile` / `улыб` - улыбка
    - `turn_left` / `left` - поворот влево
    - `turn_right` / `right` - поворот вправо
    - `look_up` / `up` - поднять голову
    - `look_down` / `down` - опустить голову
    - `open_mouth` / `рот` - открыть рот

    **Response:**
    - `passed`: bool - все инструкции выполнены
    - `overall_score`: общая оценка (0-1)
    - `results`: детали по каждой инструкции
    """
    try:
        from ..services.active_liveness_service import get_active_liveness_service

        service = await get_active_liveness_service()

        result = await service.active_liveness_check(
            video_frames=video_frames,
            instructions=instructions,
            require_liveness=require_liveness,
        )

        return BaseResponse(
            success=True,
            message="Active liveness check completed",
            data=result,
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Active liveness check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{session_id}", response_model=BaseResponse)
async def get_session_status(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    """
    Получение статуса сессии верификации.

    **Response:**
    - `session_id`: ID сессии
    - `status`: pending/processing/completed/failed
    - `verified`: результат верификации
    - `similarity_score`: оценка схожести
    """
    try:
        from ..services.session_service import SessionService

        session = await SessionService.get_session(session_id)

        if not session:
            raise ProcessingError(f"Session {session_id} not found")

        return BaseResponse(
            success=True,
            message="Session status retrieved",
            data={
                "session_id": session.session_id,
                "status": session.status,
                "verified": getattr(session, "is_match", None),
                "similarity_score": getattr(session, "similarity_score", None),
                "created_at": session.created_at,
                "expires_at": session.expiration_at,
            },
        )

    except ProcessingError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/update-reference", response_model=BaseResponse)
async def update_reference_image(
    user_id: str,
    file: UploadFile = File(..., description="Новое эталонное изображение"),
    current_user: str = Depends(get_current_user),
):
    """
    Обновление эталонного изображения пользователя.

    **Process:**
    1. Валидация нового изображения
    2. Извлечение эмбеддинга
    3. Сохранение в БД
    4. Инвалидация кэша
    """
    try:
        if not file.content_type.startswith("image/"):
            raise ValidationError(f"Invalid file type: {file.content_type}")

        image_bytes = await file.read()

        try:
            if len(image_bytes) > 10 * 1024 * 1024:
                raise ValidationError("Image size exceeds 10MB limit")

            # Используем существующий reference service
            from ..services.cache_service import get_cache_service
            from ..services.reference_service import ReferenceService

            reference_service = ReferenceService()
            cache = await get_cache_service()

            # Создаем новый reference
            ref = await reference_service.create_reference(
                user_id=user_id,
                image_data=image_bytes,
                label="updated_reference",
                quality_threshold=0.7,
            )

            # Инвалидируем кэш
            await cache.invalidate_reference(user_id)

            return BaseResponse(
                success=True,
                message="Reference image updated",
                data={
                    "user_id": user_id,
                    "reference_id": ref.id,
                    "quality_score": ref.quality_score,
                },
            )
        finally:
            # Гарантированное удаление изображения из памяти
            await _secure_delete_async(image_bytes)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update reference: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
