"""
Liveness API Routes - Проверка живости лица.

Упрощённая версия с использованием LivenessService.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.database import get_async_db
from ..models.request import (
    LivenessRequest,
    VideoLivenessRequest,
    BatchEmbeddingRequest,
    AdvancedAntiSpoofingRequest,
)
from ..models.response import (
    LivenessResponse,
    SessionResponse,
    VideoLivenessResponse,
    BatchEmbeddingResponse,
    AdvancedAntiSpoofingResponse,
    ActiveLivenessResponse,
)
from ..models.verification import VerificationSessionCreate
from ..services.liveness_service import LivenessService
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError
from ..utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Liveness"])


# ======================================================================
# MAIN LIVENESS CHECK
# ======================================================================

@router.post("/liveness", response_model=LivenessResponse)
async def check_liveness(
    request: LivenessRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Основная проверка живости лица.
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting liveness check: {request.challenge_type}")
        
        liveness_service = LivenessService(db)

        # Проверка через сервис
        result = await liveness_service.check_liveness(
            image_data=request.image_data,
            challenge_type=request.challenge_type or "passive",
            challenge_data=request.challenge_data,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        return LivenessResponse(**result)

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# SESSION MANAGEMENT
# ======================================================================

@router.post("/liveness/session", response_model=SessionResponse)
async def create_liveness_session(
    request: VerificationSessionCreate,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Создание сессии проверки живости.
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Используем VerifyService для создания сессии (можно вынести в отдельный SessionService)
        from ..services.verify_service import VerifyService
        
        verify_service = VerifyService(db)
        
        session = await verify_service.create_verification_session(
            user_id=request.user_id,
            reference_id=request.reference_id,
            metadata={**(request.metadata or {}), "session_type": "liveness"},
            expires_in_minutes=request.expires_in_minutes,
            ip_address=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent"),
        )

        return SessionResponse(
            success=True,
            session_id=session.session_id,
            session_type="liveness",
            expires_at=session.expires_at.isoformat(),
            user_id=session.user_id,
            metadata=request.metadata,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Error creating liveness session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.post("/liveness/session/{session_id}", response_model=LivenessResponse)
async def check_liveness_in_session(
    session_id: str,
    request: LivenessRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Проверка живости в рамках сессии.
    """
    try:
        liveness_service = LivenessService(db)

        # Проверка через сервис
        result = await liveness_service.check_liveness(
            image_data=request.image_data,
            challenge_type=request.challenge_type or "passive",
            challenge_data=request.challenge_data,
            user_id=request.user_id,
            session_id=session_id,
        )

        return LivenessResponse(**result)

    except (NotFoundError, ValidationError) as e:
        logger.warning(f"Error in session liveness check: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# ACTIVE LIVENESS
# ======================================================================

@router.post("/liveness/active", response_model=ActiveLivenessResponse)
async def check_active_liveness(
    request: LivenessRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Активная проверка живости с челленджами.
    """
    request_id = str(uuid.uuid4())
    
    try:
        liveness_service = LivenessService(db)

        # Проверка active liveness
        result = await liveness_service.check_active_liveness(
            image_data=request.image_data,
            challenge_type=request.challenge_type,
            challenge_data=request.challenge_data,
        )

        return ActiveLivenessResponse(
            success=True,
            session_id=request.session_id or request_id,
            challenge_type=request.challenge_type,
            liveness_detected=result["liveness_detected"],
            confidence=result["confidence"],
            processing_time=0.0,  # Будет заполнено в сервисе
            anti_spoofing_score=result.get("anti_spoofing_score"),
            face_detected=result.get("face_detected", False),
            image_quality=result.get("image_quality"),
            recommendations=result.get("recommendations", []),
            request_id=request_id,
        )

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in active liveness: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# VIDEO LIVENESS
# ======================================================================

@router.post("/liveness/video", response_model=VideoLivenessResponse)
async def analyze_video_liveness(
    request: VideoLivenessRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Анализ видео для проверки живости.
    """
    request_id = str(uuid.uuid4())
    
    try:
        liveness_service = LivenessService(db)

        # Декодируем video_data (если base64)
        import base64
        if request.video_data.startswith("data:video/"):
            header, encoded = request.video_data.split(",", 1)
            video_bytes = base64.b64decode(encoded)
        else:
            raise ValidationError("Invalid video_data format")

        # Извлекаем кадры (упрощённая версия)
        # В реальной реализации нужно использовать opencv или ffmpeg
        video_frames = [video_bytes]  # Заглушка

        # Анализ видео
        result = await liveness_service.analyze_video_liveness(
            video_frames=video_frames,
            challenge_type=request.challenge_type,
        )

        return VideoLivenessResponse(
            success=True,
            session_id=request.session_id or request_id,
            liveness_detected=result["liveness_detected"],
            confidence=result["confidence"],
            challenge_type=request.challenge_type,
            frames_processed=result.get("frames_processed", 0),
            processing_time=0.0,
            sequence_data=result.get("sequence_data"),
            anti_spoofing_score=result.get("anti_spoofing_score"),
            face_detected=result.get("face_detected", False),
            recommendations=result.get("recommendations", []),
            request_id=request_id,
        )

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in video liveness: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# ADVANCED ANTI-SPOOFING
# ======================================================================

@router.post("/liveness/anti-spoofing/advanced", response_model=AdvancedAntiSpoofingResponse)
async def advanced_anti_spoofing_check(
    request: AdvancedAntiSpoofingRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Продвинутая anti-spoofing проверка.
    """
    request_id = str(uuid.uuid4())
    
    try:
        liveness_service = LivenessService(db)

        # Advanced anti-spoofing
        result = await liveness_service.perform_anti_spoofing_check(
            image_data=request.image_data,
            analysis_type=request.analysis_type,
            include_reasoning=request.include_reasoning,
        )

        return AdvancedAntiSpoofingResponse(
            success=True,
            session_id=request.session_id or request_id,
            liveness_detected=result["liveness_detected"],
            confidence=result["confidence"],
            analysis_type=request.analysis_type,
            processing_time=0.0,
            anti_spoofing_score=result["anti_spoofing_score"],
            depth_analysis=result.get("depth_analysis"),
            texture_analysis=result.get("texture_analysis"),
            certified_analysis=result.get("certified_analysis"),
            reasoning_result=result.get("reasoning_result"),
            reasoning_summary=result.get("reasoning_result", {}).get("reasoning_summary") if request.include_reasoning else None,
            component_scores=result.get("component_scores"),
            certification_level=result.get("certification_level"),
            certification_passed=result.get("certification_passed", False),
            face_detected=result.get("face_detected", False),
            multiple_faces=result.get("multiple_faces", False),
            recommendations=result.get("recommendations", []),
            request_id=request_id,
        )

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in advanced anti-spoofing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# CHALLENGES
# ======================================================================

@router.get("/liveness/challenges", response_model=dict)
async def get_available_challenges(http_request: Request):
    """
    Получение списка доступных типов проверки живости.
    """
    try:
        challenges = LivenessService.get_supported_challenges()

        return {
            "success": True,
            "challenges": {
                name: {"description": desc}
                for name, desc in challenges.items()
            },
            "default_challenge": "passive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting challenges: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/liveness/challenge/generate", response_model=dict)
async def generate_challenge(
    challenge_type: str = "random",
    http_request: Request = None,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Генерация челленджа для активной проверки.
    """
    try:
        liveness_service = LivenessService(db)
        challenge = await liveness_service.generate_challenge(challenge_type)

        return {
            "success": True,
            "challenge": challenge,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generating challenge: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# GET RESULT BY SESSION ID
# ======================================================================

@router.get("/liveness/{session_id}", response_model=dict)
async def get_liveness_result(
    session_id: str,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Получение результата liveness по session_id.
    """
    request_id = str(uuid.uuid4())
    
    try:
        from ..db.crud import VerificationSessionCRUD

        # Получение сессии из БД
        session = await VerificationSessionCRUD.get_session(db, session_id)

        if not session:
            raise NotFoundError(f"Liveness session {session_id} not found")

        if session.session_type != "liveness":
            raise ValidationError(f"Session {session_id} is not a liveness session")

        return {
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

    except NotFoundError as e:
        logger.warning(f"Session not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting liveness result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# BATCH EMBEDDINGS (BONUS)
# ======================================================================

@router.post("/liveness/batch/embeddings", response_model=BatchEmbeddingResponse)
async def batch_generate_embeddings(
    request: BatchEmbeddingRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Пакетная генерация embeddings (используется MLService напрямую).
    """
    request_id = str(uuid.uuid4())
    batch_id = str(uuid.uuid4())
    
    try:
        from ..services.ml_service import MLService
        from ..services.validation_service import ValidationService
        
        ml_service = MLService()
        validation_service = ValidationService()

        # Валидация всех изображений
        validated_images = []
        for image_data in request.images:
            validation_result = await validation_service.validate_image(
                image_data,
                max_size=settings.MAX_UPLOAD_SIZE,
                allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
            )
            
            if validation_result.is_valid:
                validated_images.append(validation_result.image_data)
            else:
                validated_images.append(None)

        # Фильтруем только валидные
        valid_images = [img for img in validated_images if img is not None]

        if not valid_images:
            raise ValidationError("No valid images in batch")

        # Пакетная генерация
        ml_results = await ml_service.batch_generate_embeddings(
            image_data_list=valid_images,
            batch_size=request.batch_size,
        )

        # Формирование результатов
        results = []
        successful = 0
        failed = 0

        for i, (validated, ml_result) in enumerate(zip(validated_images, ml_results)):
            if validated is None:
                results.append({
                    "image_index": i,
                    "success": False,
                    "error": "Image validation failed",
                })
                failed += 1
            else:
                if ml_result.get("success"):
                    results.append({
                        "image_index": i,
                        "success": True,
                        "embedding": ml_result.get("embedding", []),
                        "quality_score": ml_result.get("quality_score"),
                        "face_detected": ml_result.get("face_detected", False),
                    })
                    successful += 1
                else:
                    results.append({
                        "image_index": i,
                        "success": False,
                        "error": ml_result.get("error", "Unknown error"),
                    })
                    failed += 1

        return BatchEmbeddingResponse(
            success=True,
            batch_id=batch_id,
            total_images=len(request.images),
            successful_embeddings=successful,
            failed_embeddings=failed,
            processing_time=0.0,
            results=results,
            performance_metrics={
                "success_rate": successful / len(request.images) if request.images else 0,
            },
            request_id=request_id,
        )

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch embeddings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
