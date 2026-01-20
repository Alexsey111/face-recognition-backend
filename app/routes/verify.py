"""
Verify API Routes - Верификация лиц.

Упрощённая версия с использованием VerifyService.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.database import get_async_db
from ..models.request import VerifyRequest
from ..models.response import VerifyResponse, SessionResponse
from ..models.verification import VerificationSessionCreate, VerificationRequest
from ..routes.auth import get_current_user
from ..services.verify_service import VerifyService
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError
from ..utils.logger import get_logger

from ..models.reference import Reference
from ..services.cache_service import CacheService
from ..dependencies import get_cache_service

router = APIRouter(tags=["Verify"])
logger = get_logger(__name__)


# ======================================================================
# MAIN VERIFY ENDPOINT
# ======================================================================

@router.post("/verify", response_model=VerifyResponse)
async def verify_face(
    request: VerifyRequest,
    http_request: Request,
    current_user: str = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Основная верификация лица против reference.
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting verification for user {current_user}")
        user_id = current_user
        
        # ==================== STEP 1: Get Reference (with caching) ====================
        # Try cache first (FAST PATH - ~10ms)
        cached_reference = await cache.get_reference_embedding(user_id)
        reference_embedding = None
        reference_version = None

        if cached_reference:
            logger.info(f"✅ Using cached reference for user {user_id}")
            reference_embedding = cached_reference["embedding"]
            reference_version = cached_reference["version"]
        else:
            # Fallback to database (SLOW PATH - ~100-200ms)
            logger.info(f"⚠️ Cache miss for user {user_id}, querying DB")

            # Get reference from database using async query
            from sqlalchemy import select
            result_ref = await db.execute(
                select(Reference)
                .where(Reference.user_id == user_id)
                .where(Reference.is_active == True)
                .order_by(Reference.version.desc())
            )
            reference_obj = result_ref.scalar_one_or_none()

            if not reference_obj:
                raise NotFoundError("No reference image found. Please upload reference first.")

            reference_embedding = reference_obj.embedding
            reference_version = reference_obj.version

            # Cache for future requests
            await cache.cache_reference_embedding(
                user_id=user_id,
                embedding=reference_embedding,
                version=reference_version,
                metadata={
                    "quality_score": getattr(reference_obj, 'quality_score', None),
                    "created_at": reference_obj.created_at.isoformat() if reference_obj.created_at else None
                }
            )
            logger.info(f"📦 Cached reference for user {user_id}")

        # ==================== STEP 2: ML Processing ====================
        verify_service = VerifyService(db)

        # Верификация через сервис с кэшированным embedding
        result = await verify_service.verify_face(
            user_id=user_id,
            image_data=request.image_data,
            threshold=request.threshold,
            session_id=request.session_id,
            reference_id=request.reference_id,
            reference_embedding=reference_embedding,
            reference_version=reference_version,
            auto_enroll=request.auto_enroll,
        )

        # Отправка webhook (асинхронно)
        if result["verified"]:
            asyncio.create_task(
                verify_service.send_verification_webhook(
                    user_id=user_id,
                    verification_id=result["verification_id"],
                    verified=result["verified"],
                    similarity=result["similarity_score"],
                    confidence=result["confidence"],
                )
            )

        # ==================== STEP 3: Cache Invalidation ====================
        # Invalidate user stats (they changed after new verification)
        await cache.invalidate_user_stats(user_id)

        return VerifyResponse(**result)

    except NotFoundError as e:
        logger.warning(f"Reference not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
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

@router.post("/verify/session", response_model=SessionResponse)
async def create_verification_session(
    request: VerificationSessionCreate,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Создание сессии верификации.
    """
    request_id = str(uuid.uuid4())
    
    try:
        verify_service = VerifyService(db)

        # Создание сессии через сервис
        session = await verify_service.create_verification_session(
            user_id=request.user_id,
            reference_id=request.reference_id,
            metadata=request.metadata,
            expires_in_minutes=request.expires_in_minutes,
            ip_address=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent"),
        )

        return SessionResponse(
            success=True,
            session_id=session.session_id,
            session_type="verification",
            expires_at=session.expires_at.isoformat(),
            user_id=session.user_id,
            metadata=request.metadata,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.post("/verify/session/{session_id}", response_model=VerifyResponse)
async def verify_face_in_session(
    session_id: str,
    request: VerificationRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Верификация в рамках существующей сессии.
    """
    try:
        verify_service = VerifyService(db)

        # Получение сессии
        session = await verify_service.get_verification_session(session_id)
        
        if not session:
            raise NotFoundError(f"Session {session_id} not found or expired")

        if session.status != "pending":
            raise ValidationError(f"Session {session_id} is not active")

        # Верификация через сервис
        result = await verify_service.verify_face(
            user_id=session.user_id,
            image_data=request.image_data,
            threshold=request.threshold,
            session_id=session_id,
            reference_id=request.reference_id or session.reference_id,
            auto_enroll=request.auto_enroll,
        )

        # Обновляем session_id в ответе
        result["session_id"] = session_id

        # Отправка webhook
        if result["verified"]:
            asyncio.create_task(
                verify_service.send_verification_webhook(
                    user_id=session.user_id,
                    verification_id=result["verification_id"],
                    verified=result["verified"],
                    similarity=result["similarity_score"],
                    confidence=result["confidence"],
                )
            )

        return VerifyResponse(**result)

    except (NotFoundError, ValidationError) as e:
        logger.warning(f"Error in session verification: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in session verification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# HISTORY
# ======================================================================

@router.get("/verify/history", response_model=dict)
async def get_verification_history(
    user_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    verified: Optional[bool] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    http_request: Request = None,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Получение истории верификаций с фильтрацией.
    """
    request_id = str(uuid.uuid4())
    
    try:
        verify_service = VerifyService(db)

        # Подготовка фильтров
        filters = {}
        if status:
            filters["status"] = status
        if verified is not None:
            filters["verified"] = verified
        if date_from:
            filters["date_from"] = date_from
        if date_to:
            filters["date_to"] = date_to

        # Получение истории через сервис
        sessions = await verify_service.get_verification_history(
            user_id=user_id or "all",  # Если user_id не указан, получаем все
            filters=filters,
            limit=limit,
            offset=offset,
        )

        # Формирование результатов
        results = [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "reference_id": s.reference_id,
                "status": s.status,
                "verified": s.is_match,
                "similarity_score": s.similarity_score,
                "confidence": s.confidence,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            }
            for s in sessions
        ]

        total_count = len(results)

        return {
            "success": True,
            "sessions": results,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_next": (offset + limit) < total_count,
            "has_prev": offset > 0,
            "filters_applied": filters,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except ValidationError as e:
        logger.warning(f"Validation error in history: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting verification history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ======================================================================
# GET RESULT BY SESSION ID
# ======================================================================

@router.get("/verify/{session_id}", response_model=dict)
async def get_verification_result(
    session_id: str,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Получение результата верификации по session_id.
    """
    request_id = str(uuid.uuid4())
    
    try:
        verify_service = VerifyService(db)

        # Получение сессии
        session = await verify_service.get_verification_session(session_id)

        if not session:
            raise NotFoundError(f"Verification session {session_id} not found")

        if session.session_type != "verification":
            raise ValidationError(f"Session {session_id} is not a verification session")

        return {
            "success": True,
            "session_id": session.session_id,
            "user_id": session.user_id,
            "reference_id": session.reference_id,
            "status": session.status,
            "verified": session.is_match,
            "similarity_score": session.similarity_score,
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
        logger.error(f"Error getting verification result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
