"""API роуты для liveness detection."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["Liveness"])

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
import uuid
import time
from ..config import settings

from ..models.request import LivenessRequest
from ..models.response import LivenessResponse, SessionResponse
from ..models.verification import (
    VerificationSessionCreate,
    LivenessRequest as SessionLivenessRequest,
    LivenessResult,
)
from ..services.ml_service import MLService
from ..services.database_service import DatabaseService
from ..services.cache_service import CacheService
from ..services.validation_service import ValidationService
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError

router = APIRouter()
logger = get_logger(__name__)


@router.post("/liveness", response_model=LivenessResponse)
async def check_liveness(request: LivenessRequest, http_request: Request):
    """
    Проверка живости лица.

    Args:
        request: Данные запроса проверки живости
        http_request: HTTP запрос для получения метаданных

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
                "timestamp": datetime.now(timezone.utc),
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
                "timestamp": datetime.now(timezone.utc),
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
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.post("/liveness/session", response_model=SessionResponse)
async def create_liveness_session(
    request: VerificationSessionCreate, http_request: Request
):
    """
    Создание сессии проверки живости.

    Args:
        request: Данные для создания сессии
        http_request: HTTP запрос

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
                "timestamp": datetime.now(timezone.utc),
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
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.post("/liveness/session/{session_id}", response_model=LivenessResponse)
async def check_liveness_in_session(
    session_id: str, request: SessionLivenessRequest, http_request: Request
):
    """
    Проверка живости в рамках сессии.

    Args:
        session_id: ID сессии проверки живости
        request: Данные запроса проверки живости
        http_request: HTTP запрос

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
        session_data["started_at"] = datetime.now(timezone.utc)
        await cache_service.set_verification_session(
            session_id, session_data, expire_seconds=1800
        )  # 30 минут

        # Создание запроса проверки живости
        liveness_request = LivenessRequest(
            session_id=session_id,
            image_data=request.image_data,
            challenge_type=request.challenge_type,
            challenge_data=request.challenge_data,
        )

        # Выполнение проверки живости
        liveness_response = await check_liveness(liveness_request, http_request)

        # Обновление статуса сессии
        session_data["status"] = "completed"
        session_data["completed_at"] = datetime.now(timezone.utc)
        session_data["response_data"] = liveness_response.dict()
        await cache_service.set_verification_session(
            session_id, session_data, expire_seconds=3600
        )  # 1 час

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
            f"Unexpected error in liveness session {session_id}, request {request_id}: {str(e)}"
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


@router.get("/liveness/challenges", response_model=dict)
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
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        logger.error(f"Error getting available challenges: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.get("/liveness/session/{session_id}", response_model=dict)
async def get_liveness_session_status(session_id: str, http_request: Request):
    """
    Получение статуса сессии проверки живости.

    Args:
        session_id: ID сессии
        http_request: HTTP запрос

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
            "timestamp": datetime.now(timezone.utc),
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
                "timestamp": datetime.now(timezone.utc),
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
                "timestamp": datetime.now(timezone.utc),
            },
        )
