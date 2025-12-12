"""API роуты для верификации."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["Verify"])

# TODO Phase 4: Реализовать verify endpoints

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from typing import Optional
import uuid
import time
from ..config import settings
from datetime import timedelta

# Now you can use timedelta
time_difference = timedelta(days=1, hours=2, minutes=30)

from ..models.request import VerifyRequest
from ..models.response import VerifyResponse, SessionResponse
from ..models.verification import (
    VerificationSessionCreate,
    VerificationRequest,
    VerificationResult,
)
from ..services.ml_service import MLService
from ..services.database_service import DatabaseService
from ..services.cache_service import CacheService
from ..services.validation_service import ValidationService
from ..services.encryption_service import EncryptionService
from ..services.storage_service import StorageService
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError
from ..utils.constants import EPSILON, CONFIDENCE_LEVELS
import numpy as np

logger = get_logger(__name__)


@router.post("/verify", response_model=VerifyResponse)
async def verify_face(request: VerifyRequest, http_request: Request):
    """
    Верификация лица по эталонному изображению.

    Args:
        request: Данные запроса верификации
        http_request: HTTP запрос для получения метаданных

    Returns:
        VerifyResponse: Результат верификации
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting face verification request {request_id}")

        # Инициализация сервисов
        ml_service = MLService()
        validation_service = ValidationService()
        cache_service = CacheService()
        db_service = DatabaseService()

        # Валидация изображения
        logger.info(f"Validating image for verification request {request_id}")
        validation_result = await validation_service.validate_image(
            request.image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # Получение эталонного изображения
        reference_embedding = None
        reference_info = None

        if request.reference_id:
            logger.info(
                f"Getting reference image {request.reference_id} for verification request {request_id}"
            )
            reference_info = await db_service.get_reference_by_id(request.reference_id)

            if not reference_info:
                raise NotFoundError(f"Reference image {request.reference_id} not found")

            if not reference_info.is_active:
                raise ValidationError(
                    f"Reference image {request.reference_id} is not active"
                )

            # Дешифрация эмбеддинга эталона
            if reference_info.embedding:
                encryption_service = EncryptionService()
                reference_embedding = await encryption_service.decrypt_embedding(
                    reference_info.embedding
                )
        elif request.user_id:
            # Получение активных эталонов пользователя
            logger.info(
                f"Getting active references for user {request.user_id} in verification request {request_id}"
            )
            user_references = await db_service.get_active_references_by_user(
                request.user_id
            )

            if not user_references:
                raise NotFoundError(
                    f"No active references found for user {request.user_id}"
                )

            # Используем первый активный эталон
            reference_info = user_references[0]
            encryption_service = EncryptionService()
            reference_embedding = await encryption_service.decrypt_embedding(
                reference_info.embedding
            )
        else:
            raise ValidationError("Either reference_id or user_id must be provided")

        # Определяем эффективный порог
        threshold_used = _choose_threshold(
            requested_threshold=request.threshold,
            quality_score=validation_result.quality_score,
            default=settings.THRESHOLD_DEFAULT,
            min_thr=settings.THRESHOLD_MIN,
            max_thr=settings.THRESHOLD_MAX,
        )

        # Обработка изображения ML сервисом
        logger.info(
            f"Processing image with ML service for verification request {request_id} (threshold={threshold_used})"
        )
        ml_result = await ml_service.verify_face(
            image_data=validation_result.image_data,
            reference_embedding=reference_embedding,
            threshold=threshold_used,
        )

        if not ml_result.get("success", False):
            raise ProcessingError(
                f"ML verification failed: {ml_result.get('error', 'Unknown error')}"
            )

        # Определение результата верификации
        verified = ml_result.get("verified", False)
        confidence = ml_result.get("confidence", 0.0)
        similarity_score = ml_result.get("similarity_score", 0.0)
        confidence_level = _confidence_level(confidence)

        # Автоматическое добавление в эталоны (если включено)
        if verified and request.auto_enroll and request.user_id:
            logger.info(
                f"Auto-enrolling verified face for user {request.user_id} in verification request {request_id}"
            )
            try:
                await auto_enroll_reference(
                    user_id=request.user_id,
                    image_data=validation_result.image_data,
                    verification_result=ml_result,
                    metadata={
                        "source": "auto_enroll",
                        "verification_request_id": request_id,
                        "original_reference_id": request.reference_id,
                    },
                )
            except Exception as e:
                logger.warning(f"Auto-enroll failed for request {request_id}: {str(e)}")
                # Не прерываем процесс, просто логируем

        processing_time = time.time() - start_time

        # Формирование ответа
        response = VerifyResponse(
            success=True,
            session_id=request_id,
            verified=verified,
            confidence=confidence,
            similarity_score=similarity_score,
            threshold_used=threshold_used,
            reference_id=request.reference_id,
            processing_time=processing_time,
            face_detected=ml_result.get("face_detected", False),
            face_quality=ml_result.get("face_quality"),
            metadata={
                "reference_label": reference_info.label if reference_info else None,
                "image_quality": validation_result.quality_score,
                "ml_model_version": ml_result.get("model_version", "unknown"),
                "auto_enrolled": request.auto_enroll and verified,
                "confidence_level": confidence_level,
                "target_far": settings.TARGET_FAR,
                "target_frr": settings.TARGET_FRR,
            },
            request_id=request_id,
        )

        logger.info(
            f"Face verification completed: {verified} (confidence: {confidence:.3f}), request {request_id}"
        )
        return response

    except (ValidationError, NotFoundError) as e:
        logger.warning(
            f"Validation/NotFound error for verification request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except ProcessingError as e:
        logger.error(
            f"Processing error for verification request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(
            f"Unexpected error for verification request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.post("/verify/session", response_model=SessionResponse)
async def create_verification_session(
    request: VerificationSessionCreate, http_request: Request
):
    """
    Создание сессии верификации.

    Args:
        request: Данные для создания сессии
        http_request: HTTP запрос

    Returns:
        SessionResponse: Информация о созданной сессии
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Creating verification session, request {request_id}")

        # Инициализация сервисов
        db_service = DatabaseService()
        cache_service = CacheService()

        # Создание сессии в БД
        session_data = {
            "id": request_id,
            "user_id": request.user_id,
            "session_type": "verification",
            "status": "pending",
            "reference_id": request.reference_id,
            "metadata": request.metadata,
            "expires_at": datetime.now(timezone.utc)
            + timedelta(minutes=request.expires_in_minutes),
            "ip_address": http_request.client.host if http_request.client else None,
            "user_agent": http_request.headers.get("user-agent"),
        }

        await db_service.create_verification_session(session_data)

        # Сохранение в кэш для быстрого доступа
        await cache_service.set_verification_session(
            session_id=request_id,
            session_data=session_data,
            expire_seconds=request.expires_in_minutes * 60,
        )

        response = SessionResponse(
            success=True,
            session_id=request_id,
            session_type="verification",
            expires_at=session_data["expires_at"],
            user_id=request.user_id,
            metadata=request.metadata,
            request_id=request_id,
        )

        logger.info(f"Verification session created successfully: {request_id}")
        return response

    except Exception as e:
        logger.error(
            f"Error creating verification session, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "SESSION_CREATE_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.post("/verify/session/{session_id}", response_model=VerifyResponse)
async def verify_face_in_session(
    session_id: str, request: VerificationRequest, http_request: Request
):
    """
    Верификация лица в рамках сессии.

    Args:
        session_id: ID сессии верификации
        request: Данные запроса верификации
        http_request: HTTP запрос

    Returns:
        VerifyResponse: Результат верификации
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Starting verification in session {session_id}, request {request_id}"
        )

        # Проверка сессии
        cache_service = CacheService()
        session_data = await cache_service.get_verification_session(session_id)

        if not session_data:
            raise NotFoundError(
                f"Verification session {session_id} not found or expired"
            )

        if session_data.get("status") != "pending":
            raise ValidationError(f"Verification session {session_id} is not active")

        # Обновление статуса сессии
        session_data["status"] = "processing"
        session_data["started_at"] = datetime.now(timezone.utc)
        await cache_service.set_verification_session(
            session_id, session_data, expire_seconds=1800
        )  # 30 минут

        # Создание запроса верификации
        verify_request = VerifyRequest(
            session_id=session_id,
            image_data=request.image_data,
            reference_id=request.reference_id or session_data.get("reference_id"),
            threshold=request.threshold,
            auto_enroll=request.auto_enroll,
        )

        # Выполнение верификации
        verify_response = await verify_face(verify_request, http_request)

        # Обновление статуса сессии
        session_data["status"] = "completed"
        session_data["completed_at"] = datetime.now(timezone.utc)
        session_data["response_data"] = verify_response.dict()
        await cache_service.set_verification_session(
            session_id, session_data, expire_seconds=3600
        )  # 1 час

        logger.info(
            f"Verification in session {session_id} completed successfully, request {request_id}"
        )
        return verify_response

    except (ValidationError, NotFoundError) as e:
        logger.warning(
            f"Validation/NotFound error in verification session {session_id}, request {request_id}: {str(e)}"
        )

        # Обновляем статус сессии на "failed"
        if session_id:
            try:
                cache_service = CacheService()
                session_data = await cache_service.get_verification_session(session_id)
                if session_data:
                    session_data["status"] = "failed"
                    session_data["error_message"] = str(e)
                    await cache_service.set_verification_session(
                        session_id, session_data, expire_seconds=3600
                    )
            except Exception:
                pass

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(
            f"Unexpected error in verification session {session_id}, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


async def auto_enroll_reference(
    user_id: str, image_data: str, verification_result: dict, metadata: dict
):
    """
    Автоматическое добавление верифицированного лица в эталоны.

    Args:
        user_id: ID пользователя
        image_data: Данные изображения
        verification_result: Результат верификации
        metadata: Дополнительные метаданные
    """
    try:
        db_service = DatabaseService()
        storage_service = StorageService()
        ml_service = MLService()
        encryption_service = EncryptionService()

        # Загрузка изображения
        upload_result = await storage_service.upload_image(
            image_data=image_data,
            metadata={
                "source": "auto_enroll",
                "verification_confidence": verification_result.get("confidence"),
                **metadata,
            },
        )

        # Генерация эмбеддинга
        embedding_result = await ml_service.generate_embedding(image_data)
        if not embedding_result.get("success"):
            raise ProcessingError("Failed to generate embedding for auto-enroll")

        # Шифрование эмбеддинга
        encrypted_embedding = await encryption_service.encrypt_embedding(
            embedding_result["embedding"]
        )

        # Сохранение в БД
        reference_data = {
            "user_id": user_id,
            "label": f"auto_enroll_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "file_url": upload_result.get("file_url"),
            "embedding": encrypted_embedding,
            "quality_score": verification_result.get("face_quality"),
            "metadata": {
                "source": "auto_enroll",
                "verification_confidence": verification_result.get("confidence"),
                "original_threshold": verification_result.get("threshold"),
                **metadata,
            },
        }

        await db_service.create_reference(reference_data)

        logger.info(f"Auto-enroll completed for user {user_id}")

    except Exception as e:
        logger.error(f"Auto-enroll failed: {str(e)}")
        raise


def _choose_threshold(
    requested_threshold: Optional[float],
    quality_score: Optional[float],
    default: float,
    min_thr: float,
    max_thr: float,
) -> float:
    """
    Определяем эффективный порог верификации:
    - Используем пользовательский, если задан.
    - Иначе адаптивно корректируем от качества (ниже качество — выше порог).
    """
    if requested_threshold:
        return float(np.clip(requested_threshold, min_thr, max_thr))

    # Адаптивная поправка: если качество низкое, повышаем порог в пределах [min,max]
    q = quality_score if quality_score is not None else 0.7
    # quality in [0,1], map to delta in [-0.05, +0.05]
    delta = (0.5 - q) * 0.1
    thr = default + delta
    return float(np.clip(thr, min_thr, max_thr))


def _confidence_level(confidence: float) -> str:
    """
    Возвращаем категорию уверенности для диагностики.
    """
    if confidence >= CONFIDENCE_LEVELS["high"]:
        return "high"
    if confidence >= CONFIDENCE_LEVELS["medium"]:
        return "medium"
    if confidence >= CONFIDENCE_LEVELS["low"]:
        return "low"
    return "very_low"
