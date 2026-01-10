"""API роуты для liveness detection."""

import asyncio
from fastapi import APIRouter

router = APIRouter(tags=["Liveness"])

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import uuid
import time
import base64
import numpy as np
from ..config import settings

from ..models.request import (
    LivenessRequest, 
    VideoLivenessRequest, 
    BatchEmbeddingRequest, 
    BatchVerificationRequest,
    AdvancedAntiSpoofingRequest
)
from ..models.response import (
    LivenessResponse, 
    SessionResponse, 
    VideoLivenessResponse,
    BatchEmbeddingResponse,
    BatchVerificationResponse,
    AdvancedAntiSpoofingResponse,
    ChallengeResponse,
    ActiveLivenessResponse
)
from ..models.verification import (
    VerificationSessionCreate,
    LivenessResult,
)
from ..services.ml_service import MLService
from ..services.database_service import DatabaseService
from ..services.cache_service import CacheService
from ..services.validation_service import ValidationService
from ..services.webhook_service import WebhookService
from ..db.database import get_async_db
from sqlalchemy.ext.asyncio import AsyncSession
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError

router = APIRouter()
logger = get_logger(__name__)


@router.post("/liveness", response_model=LivenessResponse)
async def check_liveness(
    request: LivenessRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Проверка живости лица.

    Args:
        request: Данные запроса проверки живости
        http_request: HTTP запрос для получения метаданных
        db: Сессия базы данных

    Returns:
        LivenessResponse: Результат проверки живости
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting liveness check request {request_id}")

        # Инициализация сервисов
        ml_service = MLService()
        validation_service = ValidationService()

        # Проверка требований к активным челленджам
        challenge_type = request.challenge_type or "passive"
        if (
            challenge_type in ("active", "blink", "smile", "turn_head")
            and not request.challenge_data
        ):
            raise ValidationError(
                f"challenge_data is required for challenge_type={challenge_type}"
            )

        # Валидация изображения
        logger.info(f"Validating image for liveness request {request_id}")
        validation_result = await validation_service.validate_image(
            request.image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # Обработка изображения ML сервисом для проверки живости
        logger.info(
            f"Processing image with ML service for liveness request {request_id}"
        )
        ml_result = await ml_service.check_liveness(
            image_data=validation_result.image_data,
            challenge_type=challenge_type,
            challenge_data=request.challenge_data,
        )

        if not ml_result.get("success", False):
            raise ProcessingError(
                f"ML liveness check failed: {ml_result.get('error', 'Unknown error')}"
            )

        # Извлечение результатов
        liveness_detected = ml_result.get("liveness_detected", False)
        confidence = ml_result.get("confidence", 0.0)
        anti_spoofing_score = ml_result.get("anti_spoofing_score")
        face_detected = ml_result.get("face_detected", False)
        multiple_faces = ml_result.get("multiple_faces", False)
        image_quality = ml_result.get("image_quality")
        recommendations = ml_result.get("recommendations", [])
        depth_analysis = ml_result.get("depth_analysis")

        # Дополнительные эвристики антиспуфинга (fallback/diagnostics)
        spoof_analysis = await validation_service.analyze_spoof_signs(
            validation_result.image_data
        )
        heuristic_score = spoof_analysis.get("score")
        if anti_spoofing_score is None:
            anti_spoofing_score = heuristic_score
        # Добавляем рекомендации при флагах
        if spoof_analysis.get("flags"):
            recommendations = recommendations + [
                f"heuristic:{flag}" for flag in spoof_analysis["flags"]
            ]

        processing_time = time.time() - start_time

        # Формирование ответа
        response = LivenessResponse(
            success=True,
            session_id=request_id,
            liveness_detected=liveness_detected,
            confidence=confidence,
            challenge_type=request.challenge_type or "passive",
            processing_time=processing_time,
            anti_spoofing_score=anti_spoofing_score,
            face_detected=face_detected,
            multiple_faces=multiple_faces,
            image_quality=image_quality,
            recommendations=recommendations,
            request_id=request_id,
        )

        logger.info(
            f"Liveness check completed: {liveness_detected} (confidence: {confidence:.3f}), request {request_id}"
        )
        
        # Автоматическая отправка webhook при завершении проверки живости
        if request.user_id:
            try:
                webhook_service = WebhookService(db)
                
                # Подготавливаем данные для webhook согласно Phase 8 спецификации
                webhook_payload = {
                    "session_id": request_id,
                    "is_live": liveness_detected,
                    "confidence": confidence,
                    "challenge_type": challenge_type,
                    "anti_spoofing_score": anti_spoofing_score,
                    "processing_time_ms": int(processing_time * 1000),
                    "face_detected": face_detected,
                    "multiple_faces": multiple_faces,
                    "recommendations": recommendations,
                    "image_quality": image_quality,
                    "liveness_type": ml_result.get("liveness_type", "unknown"),
                    "depth_analysis": depth_analysis,
                    "heuristic_score": heuristic_score,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                
                # Отправляем webhook асинхронно (не блокируем основной поток)
                asyncio.create_task(
                    webhook_service.emit_event(
                        event_type="liveness.completed",
                        user_id=request.user_id,
                        payload=webhook_payload
                    )
                )
                logger.info(f"Webhook event queued for liveness result: {request_id}")
                
            except Exception as webhook_error:
                # Логируем ошибку webhook, но не прерываем основной процесс
                logger.warning(f"Failed to queue webhook for liveness {request_id}: {str(webhook_error)}")
        
        return response

    except ValidationError as e:
        logger.warning(f"Validation error for liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except ProcessingError as e:
        logger.error(f"Processing error for liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error for liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.post("/session", response_model=SessionResponse)
async def create_liveness_session(
    request: VerificationSessionCreate, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Создание сессии проверки живости.

    Args:
        request: Данные для создания сессии
        http_request: HTTP запрос
        db: Сессия базы данных

    Returns:
        SessionResponse: Информация о созданной сессии
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Creating liveness session, request {request_id}")

        # Инициализация сервисов
        db_service = DatabaseService()
        cache_service = CacheService()

        # Валидация типа сессии
        if request.session_type != "liveness":
            raise ValidationError(
                "Session type must be 'liveness' for liveness session creation"
            )

        # Создание сессии в БД
        session_data = {
            "id": request_id,
            "user_id": request.user_id,
            "session_type": "liveness",
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
            session_type="liveness",
            expires_at=session_data["expires_at"],
            user_id=request.user_id,
            metadata=request.metadata,
            request_id=request_id,
        )

        logger.info(f"Liveness session created successfully: {request_id}")
        
        # Отправляем webhook о создании сессии
        if request.user_id:
            try:
                webhook_service = WebhookService(db)
                asyncio.create_task(
                    webhook_service.emit_event(
                        event_type="liveness.session_created",
                        user_id=request.user_id,
                        payload={
                            "session_id": request_id,
                            "session_type": "liveness",
                            "expires_at": session_data["expires_at"].isoformat(),
                            "metadata": request.metadata,
                        }
                    )
                )
            except Exception as webhook_error:
                logger.warning(f"Failed to queue webhook for session creation {request_id}: {str(webhook_error)}")
        
        return response

    except ValidationError as e:
        logger.warning(
            f"Validation error creating liveness session, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Error creating liveness session, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "SESSION_CREATE_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.post("/session/{session_id}", response_model=LivenessResponse)
async def check_liveness_in_session(
    session_id: str, 
    request: LivenessRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Проверка живости в рамках сессии.

    Args:
        session_id: ID сессии проверки живости
        request: Данные запроса проверки живости
        http_request: HTTP запрос
        db: Сессия базы данных

    Returns:
        LivenessResponse: Результат проверки живости
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Starting liveness check in session {session_id}, request {request_id}"
        )

        # Проверка сессии
        cache_service = CacheService()
        session_data = await cache_service.get_verification_session(session_id)

        if not session_data:
            raise NotFoundError(f"Liveness session {session_id} not found or expired")

        if session_data.get("status") != "pending":
            raise ValidationError(f"Liveness session {session_id} is not active")

        if session_data.get("session_type") != "liveness":
            raise ValidationError(f"Session {session_id} is not a liveness session")

        # Обновление статуса сессии
        session_data["status"] = "processing"
        session_data["started_at"] = datetime.now(timezone.utc).isoformat()
        await cache_service.set_verification_session(
            session_id, session_data, expire_seconds=1800
        )  # 30 минут

        # Выполнение проверки живости
        liveness_response = await check_liveness(request, http_request, db)

        # Обновление статуса сессии
        session_data["status"] = "completed"
        session_data["completed_at"] = datetime.now(timezone.utc).isoformat()
        session_data["response_data"] = liveness_response.dict()
        await cache_service.set_verification_session(
            session_id, session_data, expire_seconds=3600
        )  # 1 час
        
        # Отправляем webhook о завершении сессии
        user_id = session_data.get("user_id")
        if user_id:
            try:
                webhook_service = WebhookService(db)
                asyncio.create_task(
                    webhook_service.emit_event(
                        event_type="liveness.session_completed",
                        user_id=user_id,
                        payload={
                            "session_id": session_id,
                            "status": "completed",
                            "liveness_detected": liveness_response.liveness_detected,
                            "confidence": liveness_response.confidence,
                            "processing_time": time.time() - start_time,
                        }
                    )
                )
            except Exception as webhook_error:
                logger.warning(f"Failed to queue webhook for session completion {session_id}: {str(webhook_error)}")

        logger.info(
            f"Liveness check in session {session_id} completed successfully, request {request_id}"
        )
        return liveness_response

    except (ValidationError, NotFoundError) as e:
        logger.warning(
            f"Validation/NotFound error in liveness session {session_id}, request {request_id}: {str(e)}"
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
                    
                    # Отправляем webhook об ошибке
                    user_id = session_data.get("user_id")
                    if user_id:
                        try:
                            webhook_service = WebhookService(db)
                            asyncio.create_task(
                                webhook_service.emit_event(
                                    event_type="liveness.session_failed",
                                    user_id=user_id,
                                    payload={
                                        "session_id": session_id,
                                        "status": "failed",
                                        "error": str(e),
                                    }
                                )
                            )
                        except Exception:
                            pass
            except Exception:
                pass

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(
            f"Unexpected error in liveness session {session_id}, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.get("/challenges", response_model=dict)
async def get_available_challenges(http_request: Request):
    """
    Получение списка доступных типов проверки живости.

    Args:
        http_request: HTTP запрос

    Returns:
        dict: Список доступных типов проверки с описаниями
    """
    try:
        challenges = {
            "passive": {
                "name": "Passive Liveness",
                "description": "Проверка живости без активных действий пользователя",
                "required_data": [],
                "difficulty": "easy",
                "processing_time": "fast",
            },
            "active": {
                "name": "Active Liveness",
                "description": "Проверка живости с активными действиями",
                "required_data": ["challenge_data"],
                "difficulty": "medium",
                "processing_time": "medium",
            },
            "blink": {
                "name": "Blink Detection",
                "description": "Обнаружение моргания",
                "required_data": [],
                "difficulty": "medium",
                "processing_time": "medium",
            },
            "smile": {
                "name": "Smile Detection",
                "description": "Обнаружение улыбки",
                "required_data": [],
                "difficulty": "medium",
                "processing_time": "medium",
            },
            "turn_head": {
                "name": "Head Turn Detection",
                "description": "Обнаружение поворота головы",
                "required_data": ["challenge_data"],
                "difficulty": "hard",
                "processing_time": "slow",
            },
        }

        return {
            "success": True,
            "challenges": challenges,
            "default_challenge": "passive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting available challenges: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.get("/session/{session_id}", response_model=dict)
async def get_liveness_session_status(
    session_id: str, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Получение статуса сессии проверки живости.

    Args:
        session_id: ID сессии
        http_request: HTTP запрос
        db: Сессия базы данных

    Returns:
        dict: Статус сессии и результаты
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Getting liveness session status for {session_id}, request {request_id}"
        )

        cache_service = CacheService()
        session_data = await cache_service.get_verification_session(session_id)

        if not session_data:
            raise NotFoundError(f"Session {session_id} not found")

        if session_data.get("session_type") != "liveness":
            raise ValidationError(f"Session {session_id} is not a liveness session")

        # Формирование ответа со статусом
        response = {
            "success": True,
            "session_id": session_id,
            "session_type": session_data.get("session_type"),
            "status": session_data.get("status"),
            "created_at": session_data.get("created_at"),
            "expires_at": session_data.get("expires_at"),
            "started_at": session_data.get("started_at"),
            "completed_at": session_data.get("completed_at"),
            "user_id": session_data.get("user_id"),
            "metadata": session_data.get("metadata"),
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Добавляем результаты, если сессия завершена
        if session_data.get("status") == "completed" and session_data.get(
            "response_data"
        ):
            response["results"] = session_data["response_data"]

        # Добавляем информацию об ошибке, если есть
        if session_data.get("error_message"):
            response["error_message"] = session_data["error_message"]

        logger.info(
            f"Liveness session status retrieved for {session_id}, request {request_id}"
        )
        return response

    except (ValidationError, NotFoundError) as e:
        logger.warning(
            f"Validation/NotFound error getting liveness session status {session_id}, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(
            f"Unexpected error getting liveness session status {session_id}, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.get("/{session_id}", response_model=dict)
async def get_liveness_result(
    session_id: str, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Получение результата проверки живости по session_id.

    Args:
        session_id: ID сессии проверки живости
        http_request: HTTP запрос
        db: Сессия базы данных

    Returns:
        dict: Результат проверки живости
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Getting liveness result for session {session_id}, request {request_id}"
        )

        # Инициализация сервисов
        from app.db.crud import VerificationSessionCRUD
        from app.db.database import get_async_db_manager

        # Получение сессии из БД
        async with get_async_db_manager().get_session() as db_session:
            session = await VerificationSessionCRUD.get_session(db_session, session_id)

        if not session:
            raise NotFoundError(f"Liveness session {session_id} not found")

        # Проверяем, что это сессия проверки живости
        if session.session_type != "liveness":
            raise ValidationError(
                f"Session {session_id} is not a liveness session"
            )

        # Формирование ответа
        response = {
            "success": True,
            "session_id": session.session_id,
            "user_id": session.user_id,
            "status": session.status,
            "is_live": session.is_liveness_passed,
            "liveness_score": session.liveness_score,
            "liveness_method": session.liveness_method,
            "confidence": session.confidence,
            "face_detected": session.face_detected,
            "face_quality_score": session.face_quality_score,
            "processing_time": session.processing_time,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "error_code": session.error_code,
            "error_message": session.error_message,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Liveness result retrieved for session {session_id}, request {request_id}"
        )
        return response

    except NotFoundError as e:
        logger.warning(
            f"Liveness session {session_id} not found, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": "SESSION_NOT_FOUND",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except ValidationError as e:
        logger.warning(
            f"Validation error for liveness session {session_id}, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(
            f"Unexpected error getting liveness result for session {session_id}, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


# ========================================
# NEW ENDPOINTS FOR ADVANCED FEATURES
# ========================================

@router.post("/video", response_model=VideoLivenessResponse)
async def analyze_video_liveness(
    request: VideoLivenessRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Анализ видео для проверки живости (multi-frame analysis).
    
    Поддерживает:
    - video_blink: анализ моргания в видео
    - video_smile: анализ улыбки в видео  
    - video_head_turn: анализ поворота головы в видео
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting video liveness analysis request {request_id}")

        # Инициализация сервисов
        ml_service = MLService()
        validation_service = ValidationService()

        # Валидация видео данных
        logger.info(f"Validating video data for request {request_id}")
        
        # Проверяем формат данных (base64 или URL)
        video_data = request.video_data
        if video_data.startswith("data:video/"):
            # Извлекаем base64 данные
            header, encoded = video_data.split(",", 1)
            video_bytes = base64.b64decode(encoded)
        elif video_data.startswith("http://") or video_data.startswith("https://"):
            # TODO: Загрузка по URL (пока заглушка)
            raise ValidationError("Video URL download not implemented yet")
        else:
            raise ValidationError("Video data must be base64 or URL")

        # Проверяем размер видео
        if len(video_bytes) > settings.MAX_UPLOAD_SIZE * 10:  # Видео может быть больше
            raise ValidationError(f"Video size exceeds maximum limit")

        # Анализ видео с ML сервисом
        logger.info(f"Processing video with ML service for request {request_id}")
        ml_result = await ml_service.analyze_video_liveness(
            video_data=video_bytes,
            challenge_type=request.challenge_type,
            frame_count=request.frame_count,
        )

        if not ml_result.get("success", False):
            raise ProcessingError(
                f"ML video analysis failed: {ml_result.get('error', 'Unknown error')}"
            )

        processing_time = time.time() - start_time

        # Формирование ответа
        response = VideoLivenessResponse(
            success=True,
            session_id=request.session_id,
            liveness_detected=ml_result.get("liveness_detected", False),
            confidence=ml_result.get("confidence", 0.0),
            challenge_type=request.challenge_type,
            frames_processed=ml_result.get("frames_processed", 0),
            processing_time=processing_time,
            sequence_data=ml_result.get("sequence_data"),
            anti_spoofing_score=ml_result.get("anti_spoofing_score"),
            face_detected=ml_result.get("face_detected", False),
            recommendations=ml_result.get("recommendations", []),
            request_id=request_id,
        )

        logger.info(
            f"Video liveness analysis completed: {response.liveness_detected} "
            f"(confidence: {response.confidence:.3f}), request {request_id}"
        )
        
        # Отправляем webhook для видео анализа
        if request.user_id:
            try:
                webhook_service = WebhookService(db)
                asyncio.create_task(
                    webhook_service.emit_event(
                        event_type="liveness.video_completed",
                        user_id=request.user_id,
                        payload={
                            "session_id": request.session_id,
                            "liveness_detected": response.liveness_detected,
                            "confidence": response.confidence,
                            "challenge_type": request.challenge_type,
                            "frames_processed": response.frames_processed,
                            "processing_time": processing_time,
                        }
                    )
                )
            except Exception as webhook_error:
                logger.warning(f"Failed to queue webhook for video analysis {request_id}: {str(webhook_error)}")
        
        return response

    except ValidationError as e:
        logger.warning(f"Validation error for video liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except ProcessingError as e:
        logger.error(f"Processing error for video liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error for video liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.post("/active", response_model=ActiveLivenessResponse)
async def check_active_liveness(
    request: LivenessRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Активная проверка живости с различными типами челленджей.
    
    Поддерживает:
    - blink: проверка через обнаружение моргания
    - smile: проверка через обнаружение улыбки
    - turn_head: проверка через анализ поворота головы
    - active: общий challenge-response механизм
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting active liveness check request {request_id}")

        # Инициализация сервисов
        ml_service = MLService()
        validation_service = ValidationService()

        # Проверка требований к активным челленджам
        challenge_type = request.challenge_type
        if challenge_type in ("active", "blink", "smile", "turn_head") and not request.challenge_data:
            raise ValidationError(
                f"challenge_data is required for challenge_type={challenge_type}"
            )

        # Валидация изображения
        logger.info(f"Validating image for active liveness request {request_id}")
        validation_result = await validation_service.validate_image(
            request.image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # Активная проверка живости с ML сервисом
        logger.info(f"Processing with active liveness check for request {request_id}")
        ml_result = await ml_service.check_active_liveness(
            image_data=validation_result.image_data,
            challenge_type=challenge_type,
            challenge_data=request.challenge_data,
        )

        if not ml_result.get("success", False):
            raise ProcessingError(
                f"ML active liveness check failed: {ml_result.get('error', 'Unknown error')}"
            )

        processing_time = time.time() - start_time

        # Формирование ответа на основе типа челленджа
        response_data = {
            "success": True,
            "session_id": request.session_id,
            "challenge_type": challenge_type,
            "liveness_detected": ml_result.get("liveness_detected", False),
            "confidence": ml_result.get("confidence", 0.0),
            "processing_time": processing_time,
            "anti_spoofing_score": ml_result.get("anti_spoofing_score"),
            "face_detected": ml_result.get("face_detected", False),
            "image_quality": ml_result.get("image_quality"),
            "recommendations": ml_result.get("recommendations", []),
            "request_id": request_id,
        }

        # Добавляем специфичные данные в зависимости от типа челленджа
        challenge_specific_data = ml_result.get("challenge_specific_data", {})
        if challenge_type == "blink":
            response_data["blink_analysis"] = challenge_specific_data
        elif challenge_type == "smile":
            response_data["smile_analysis"] = challenge_specific_data
        elif challenge_type == "turn_head":
            response_data["head_turn_analysis"] = challenge_specific_data
        elif challenge_type == "active":
            response_data["challenge_response_analysis"] = challenge_specific_data

        response = ActiveLivenessResponse(**response_data)

        logger.info(
            f"Active liveness check completed: {response.liveness_detected} "
            f"(confidence: {response.confidence:.3f}, type: {challenge_type}), request {request_id}"
        )
        
        # Отправляем webhook для активной проверки
        if request.user_id:
            try:
                webhook_service = WebhookService(db)
                asyncio.create_task(
                    webhook_service.emit_event(
                        event_type="liveness.active_completed",
                        user_id=request.user_id,
                        payload={
                            "session_id": request.session_id,
                            "challenge_type": challenge_type,
                            "liveness_detected": response.liveness_detected,
                            "confidence": response.confidence,
                            "processing_time": processing_time,
                        }
                    )
                )
            except Exception as webhook_error:
                logger.warning(f"Failed to queue webhook for active liveness {request_id}: {str(webhook_error)}")
        
        return response

    except ValidationError as e:
        logger.warning(f"Validation error for active liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except ProcessingError as e:
        logger.error(f"Processing error for active liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error for active liveness request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.post("/batch/embeddings", response_model=BatchEmbeddingResponse)
async def batch_generate_embeddings(
    request: BatchEmbeddingRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Пакетная генерация эмбеддингов для повышения производительности.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    batch_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting batch embedding generation: {len(request.images)} images, request {request_id}")

        # Инициализация сервисов
        ml_service = MLService()
        validation_service = ValidationService()

        # Валидация всех изображений
        logger.info(f"Validating {len(request.images)} images for batch request {request_id}")
        validated_images = []
        
        for i, image_data in enumerate(request.images):
            validation_result = await validation_service.validate_image(
                image_data,
                max_size=settings.MAX_UPLOAD_SIZE,
                allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
            )
            
            if not validation_result.is_valid:
                logger.warning(f"Image {i} validation failed: {validation_result.error_message}")
                # Добавляем ошибку в результат, но продолжаем обработку
                validated_images.append(None)
            else:
                validated_images.append(validation_result.image_data)

        # Фильтруем успешно валидированные изображения
        valid_images = [img for img in validated_images if img is not None]
        
        if not valid_images:
            raise ValidationError("No valid images found in batch")

        # Пакетная генерация эмбеддингов
        logger.info(f"Processing batch embeddings: {len(valid_images)} valid images")
        ml_results = await ml_service.batch_generate_embeddings(
            image_data_list=valid_images,
            batch_size=request.batch_size,
        )

        # Обработка результатов
        results = []
        successful_embeddings = 0
        failed_embeddings = 0

        valid_idx = 0
        for i, original_image in enumerate(validated_images):
            if original_image is None:
                # Изображение не прошло валидацию
                results.append({
                    "image_index": i,
                    "success": False,
                    "error": "Image validation failed",
                })
                failed_embeddings += 1
            else:
                # Обрабатываем результат ML
                ml_result = ml_results[valid_idx] if valid_idx < len(ml_results) else {"success": False, "error": "ML processing failed"}
                
                result_item = {
                    "image_index": i,
                    "success": ml_result.get("success", False),
                    "face_detected": ml_result.get("face_detected", False),
                    "quality_score": ml_result.get("quality_score"),
                    "processing_time": ml_result.get("processing_time"),
                }
                
                if ml_result.get("success", False):
                    result_item["embedding"] = ml_result.get("embedding", [])
                    result_item["model_version"] = ml_result.get("model_version")
                    successful_embeddings += 1
                else:
                    result_item["error"] = ml_result.get("error", "Unknown error")
                    failed_embeddings += 1
                
                results.append(result_item)
                valid_idx += 1

        processing_time = time.time() - start_time

        # Формирование ответа
        response = BatchEmbeddingResponse(
            success=True,
            batch_id=batch_id,
            total_images=len(request.images),
            successful_embeddings=successful_embeddings,
            failed_embeddings=failed_embeddings,
            processing_time=processing_time,
            results=results,
            performance_metrics={
                "images_per_second": len(valid_images) / processing_time if processing_time > 0 else 0,
                "success_rate": successful_embeddings / len(request.images) if len(request.images) > 0 else 0,
                "average_processing_time": processing_time / len(valid_images) if len(valid_images) > 0 else 0,
            },
            request_id=request_id,
        )

        logger.info(
            f"Batch embedding generation completed: {successful_embeddings}/{len(request.images)} successful, "
            f"request {request_id}"
        )
        
        # Отправляем webhook для batch операции
        if request.user_id:
            try:
                webhook_service = WebhookService(db)
                asyncio.create_task(
                    webhook_service.emit_event(
                        event_type="liveness.batch_completed",
                        user_id=request.user_id,
                        payload={
                            "batch_id": batch_id,
                            "total_images": len(request.images),
                            "successful": successful_embeddings,
                            "failed": failed_embeddings,
                            "processing_time": processing_time,
                        }
                    )
                )
            except Exception as webhook_error:
                logger.warning(f"Failed to queue webhook for batch operation {request_id}: {str(webhook_error)}")
        
        return response

    except ValidationError as e:
        logger.warning(f"Validation error for batch embedding request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except ProcessingError as e:
        logger.error(f"Processing error for batch embedding request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error for batch embedding request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.post("/anti-spoofing/advanced", response_model=AdvancedAntiSpoofingResponse)
async def advanced_anti_spoofing_check(
    request: AdvancedAntiSpoofingRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Продвинутая проверка anti-spoofing с depth analysis, texture analysis и multi-turn reasoning.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting advanced anti-spoofing check: {request.analysis_type}, request {request_id}")

        # Инициализация сервисов
        ml_service = MLService()
        validation_service = ValidationService()

        # Валидация изображения
        logger.info(f"Validating image for advanced anti-spoofing request {request_id}")
        validation_result = await validation_service.validate_image(
            request.image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # Продвинутый anti-spoofing анализ
        logger.info(f"Processing with advanced anti-spoofing analysis for request {request_id}")
        ml_result = await ml_service.advanced_anti_spoofing_check(
            image_data=validation_result.image_data,
            analysis_type=request.analysis_type,
        )

        if not ml_result.get("success", False):
            raise ProcessingError(
                f"ML advanced anti-spoofing check failed: {ml_result.get('error', 'Unknown error')}"
            )

        processing_time = time.time() - start_time

        # Извлечение результатов анализа
        analysis_results = ml_result.get("analysis_results", {})
        reasoning_result = analysis_results.get("reasoning_result", {})

        # Формирование ответа
        response = AdvancedAntiSpoofingResponse(
            success=True,
            session_id=request.session_id,
            liveness_detected=ml_result.get("liveness_detected", False),
            confidence=ml_result.get("confidence", 0.0),
            analysis_type=request.analysis_type,
            processing_time=processing_time,
            anti_spoofing_score=ml_result.get("anti_spoofing_score", 0.0),
            
            # Результаты различных анализов
            depth_analysis=analysis_results.get("depth_analysis"),
            texture_analysis=analysis_results.get("texture_analysis"),
            certified_analysis=analysis_results.get("certified_analysis"),
            
            # Multi-turn reasoning
            reasoning_result=reasoning_result if request.include_reasoning else None,
            reasoning_summary=reasoning_result.get("reasoning_summary") if request.include_reasoning else None,
            
            # Компонентные оценки
            component_scores=ml_result.get("component_scores"),
            
            # Сертификация
            certification_level=analysis_results.get("certified_analysis", {}).get("certification_level"),
            certification_passed=analysis_results.get("certified_analysis", {}).get("is_certified_passed", False),
            
            face_detected=ml_result.get("face_detected", False),
            multiple_faces=ml_result.get("multiple_faces", False),
            recommendations=ml_result.get("recommendations", []),
            request_id=request_id,
        )

        logger.info(
            f"Advanced anti-spoofing check completed: {response.liveness_detected} "
            f"(confidence: {response.confidence:.3f}, type: {request.analysis_type}), request {request_id}"
        )
        
        # Отправляем webhook для продвинутого anti-spoofing
        if request.user_id:
            try:
                webhook_service = WebhookService(db)
                asyncio.create_task(
                    webhook_service.emit_event(
                        event_type="liveness.anti_spoofing_completed",
                        user_id=request.user_id,
                        payload={
                            "session_id": request.session_id,
                            "analysis_type": request.analysis_type,
                            "liveness_detected": response.liveness_detected,
                            "confidence": response.confidence,
                            "anti_spoofing_score": response.anti_spoofing_score,
                            "certification_passed": response.certification_passed,
                            "processing_time": processing_time,
                        }
                    )
                )
            except Exception as webhook_error:
                logger.warning(f"Failed to queue webhook for anti-spoofing {request_id}: {str(webhook_error)}")
        
        return response

    except ValidationError as e:
        logger.warning(f"Validation error for advanced anti-spoofing request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except ProcessingError as e:
        logger.error(f"Processing error for advanced anti-spoofing request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error for advanced anti-spoofing request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
