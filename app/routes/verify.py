"""API роуты для верификации."""
import asyncio
import time
import uuid
import json  # <--- ДОБАВЛЕНО
from datetime import datetime, timezone, timedelta
from typing import Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query, Depends, status # <--- ДОБАВЛЕН status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.database import get_async_db
from ..db.models import VerificationSession
from ..models.user import UserModel
from ..models.request import VerifyRequest
from ..models.response import VerifyResponse, SessionResponse
from ..models.verification import (
    VerificationSessionCreate,
    VerificationRequest,
)
from ..routes.auth import get_current_user
from ..services.cache_service import CacheService
from ..services.database_service import DatabaseService
from ..services.encryption_service import EncryptionService
from ..services.ml_service import MLService
from ..services.storage_service import StorageService
from ..services.validation_service import ValidationService
from ..services.webhook_service import WebhookService
from ..utils.constants import CONFIDENCE_LEVELS
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError
from ..utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Verify"])


def _ensure_scalar(value, default: float = 0.0):
    """Убеждаемся, что значение является скаляром, а не numpy массивом."""
    if value is None:
        return default
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if hasattr(value, "__len__") and hasattr(value, "__getitem__") and len(value) == 1:
        return value[0]
    return value


def _sanitize_ml_result(ml_result):
    """Рекурсивно обрабатываем ml_result и превращаем numpy в скаляры."""
    if isinstance(ml_result, dict):
        return {k: _sanitize_ml_result(v) for k, v in ml_result.items()}
    if isinstance(ml_result, (list, tuple)):
        return [_sanitize_ml_result(item) for item in ml_result]
    return _ensure_scalar(ml_result)


def _safe_float(value, default: float = 0.0) -> float:
    """Безопасно преобразуем значение в float, обрабатывая numpy."""
    try:
        if hasattr(value, "item"):  # numpy scalar
            return float(value.item())
        if hasattr(value, "__len__") and hasattr(value, "__getitem__") and len(value) == 1:
            return float(value[0])
        return float(value)
    except (ValueError, TypeError, IndexError):
        return default


async def _verify_face_internal(
    request: VerifyRequest,
    http_request: Request,
    current_user: str,
    db: AsyncSession,
) -> VerifyResponse:
    """
    Внутренняя функция верификации лица (без Depends).
    
    Flow:
    1. Инициализация и логирование.
    2. Поиск эталона (Reference Image).
    3. Вызов ML-движка (сравнение + liveness).
    4. Очистка данных (Sanitization) и Decision Logic.
    5. Сохранение в базу данных (Persistence).
    6. Асинхронное уведомление через Webhook (с соблюдением сигнатуры payload).
    7. Кэширование результата с гарантией JSON-сериализации.
    8. Возврат структурированного ответа.
    """
    start_time = time.monotonic()
    request_id = str(uuid.uuid4())

    logger.info(
        "Face verification process started",
        extra={
            "request_id": request_id,
            "user_id": current_user,
            "ip": http_request.client.host if http_request.client else "unknown",
        },
    )

    # --- Инициализация сервисов ---
    ml_service = MLService()
    db_service = DatabaseService(db)
    cache_service = CacheService()
    webhook_service = WebhookService(db)

    # --- 1. Поиск эталонного изображения ---
    reference = await db_service.get_reference_image(user_id=current_user)
    if not reference:
        logger.warning(f"Verification aborted: No reference for user {current_user}")
        raise HTTPException(
            status_code=404,
            detail="Reference image not found. Please register your face first.",
        )

    # --- 2. ML Верификация ---
    try:
        # Расшифровываем reference embedding
        crypto = EncryptionService()
        reference_embedding = await crypto.decrypt_embedding(reference.embedding_encrypted)
        raw_ml_result = await ml_service.verify_face(
            image_data=request.image_data,
            reference_embedding=reference_embedding,
            threshold=request.threshold or 0.8
        )
    except Exception as exc:
        logger.exception("ML Service Error", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            detail="Error during biometric analysis",
        ) from exc

    # --- 3. Обработка данных и принятие решения ---
    # Очищаем результат от типов numpy (float32 -> float)
    ml_result = _sanitize_ml_result(raw_ml_result)

    # Определяем динамический порог на основе качества фото
    threshold = _choose_threshold(
        requested_threshold=request.threshold,
        quality_score=_safe_float(ml_result.get("face_quality"), 0.7),
        default=0.65,
        min_thr=0.4,
        max_thr=0.9
    )

    similarity = _safe_float(ml_result.get("similarity"), 0.0)
    confidence = _safe_float(ml_result.get("confidence"), 0.0)
    liveness = _safe_float(ml_result.get("liveness"), 0.0)
    
    is_verified = similarity >= threshold

    # --- 4. Сохранение результата в БД ---
    # Создаем запись о транзакции верификации
    verification = await db_service.create_verification(
        user_id=current_user,
        similarity=similarity,
        confidence=confidence,
        liveness=liveness,
        threshold=threshold,
        verified=is_verified,
        request_id=request_id,
    )

    # --- 5. Кэширование (Безопасная сериализация) ---
    try:
        # Превращаем результат в строку через JSON для гарантии совместимости с Redis
        safe_cache_data = json.loads(json.dumps(ml_result, default=str))
        await cache_service.set(
            key=f"verification:{verification.id}",
            value=safe_cache_data,
            ttl=300,
        )
    except Exception as e:
        logger.warning(f"Cache write failed (non-critical): {str(e)}")

    # --- 6. Отправка Webhook (Асинхронно) ---
    # Важно: используем именованный аргумент 'payload', как требует сигнатура emit_event
    webhook_payload = {
        "event": "face.verified",
        "verification_id": str(verification.id),
        "user_id": str(current_user),
        "verified": is_verified,
        "similarity": similarity,
        "confidence": confidence,
        "liveness": liveness,
        "threshold_used": threshold,
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Запускаем фоновую задачу, не блокируя HTTP-ответ
    asyncio.create_task(
        webhook_service.emit_event(
            event_type="face.verified",
            user_id=str(current_user),
            payload=webhook_payload,
        )
    )

    # --- 7. Формирование ответа ---
    duration_ms = int((time.monotonic() - start_time) * 1000)

    logger.info(
        "Face verification finished",
        extra={
            "request_id": request_id,
            "verified": is_verified,
            "similarity": similarity,
            "duration_ms": duration_ms,
        },
    )

    return VerifyResponse(
        verification_id=str(verification.id),
        session_id=request_id,
        verified=is_verified,
        similarity_score=similarity,
        confidence=confidence,
        threshold_used=threshold,
        processing_time=duration_ms / 1000.0,  # Конвертируем мс в секунды
        face_detected=ml_result.get("face_detected", True),
        face_quality=ml_result.get("quality_score"),
        liveness_score=liveness,
        liveness_passed=liveness >= 0.5 if liveness > 0 else None,
        request_id=request_id,
    )


@router.post("/verify", response_model=VerifyResponse)
async def verify_face(
    request: VerifyRequest,
    http_request: Request,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Полная реализация верификации лица (публичный endpoint).
    """
    return await _verify_face_internal(request, http_request, current_user, db)


def _choose_threshold(
    requested_threshold: Optional[float],
    quality_score: Optional[float],
    default: float,
    min_thr: float,
    max_thr: float,
) -> float:
    """Определяем эффективный порог верификации."""
    if requested_threshold is not None:
        # Убеждаемся, что requested_threshold является скаляром
        try:
            if hasattr(requested_threshold, 'item'):  # numpy scalar
                requested_threshold = requested_threshold.item()
            elif hasattr(requested_threshold, '__len__') and hasattr(requested_threshold, '__getitem__') and len(requested_threshold) == 1:  # array with one element
                requested_threshold = requested_threshold[0]
            elif isinstance(requested_threshold, (list, tuple)) and len(requested_threshold) == 1:
                requested_threshold = requested_threshold[0]
            
            # Проверяем, что теперь это скаляр
            if hasattr(requested_threshold, '__len__') and not isinstance(requested_threshold, str):
                raise ValueError(f"requested_threshold is still an array: {requested_threshold}")
                
            return float(np.clip(float(requested_threshold), min_thr, max_thr))
        except Exception as e:
            logger.warning(f"Error processing requested_threshold {requested_threshold}: {e}")
            # Fallback to default threshold
            threshold = default
            return float(np.clip(threshold, min_thr, max_thr))

    q = quality_score if quality_score is not None else 0.7
    # Убеждаемся, что q является скаляром
    if hasattr(q, 'item'):  # numpy scalar
        q = q.item()
    elif hasattr(q, '__len__') and len(q) == 1:  # array with one element
        q = q[0]
    
    # quality in [0,1], map to delta in [-0.05, +0.05]
    delta = (0.5 - q) * 0.1
    thr = default + delta
    return float(np.clip(thr, min_thr, max_thr))


def _confidence_level(confidence: float) -> str:
    """Возвращаем категорию уверенности для диагностики."""
    # Убеждаемся, что confidence является скаляром
    if hasattr(confidence, 'item'):  # numpy scalar
        confidence = confidence.item()
    elif hasattr(confidence, '__len__') and len(confidence) == 1:  # array with one element
        confidence = confidence[0]
    
    if confidence >= CONFIDENCE_LEVELS["high"]:
        return "high"
    if confidence >= CONFIDENCE_LEVELS["medium"]:
        return "medium"
    if confidence >= CONFIDENCE_LEVELS["low"]:
        return "low"
    return "very_low"


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
        cache_service = CacheService()

        # Импортируем CRUD для работы с сессиями
        from app.db.crud import VerificationSessionCRUD
        from app.db.database import get_async_db_manager

        # Создание сессии в БД через CRUD
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=request.expires_in_minutes)
        
        async with get_async_db_manager().get_session() as db:
            await VerificationSessionCRUD.create_session(
                db=db,
                user_id=request.user_id,
                session_id=request_id,
                image_filename="verification_session",  # Значение по умолчанию для verification
                image_size_mb=0.0,  # Значение по умолчанию для verification
                expires_at=expires_at
            )

        # Сохранение в кэш для быстрого доступа
        cache_data = {
            "session_id": request_id,
            "user_id": request.user_id,
            "session_type": "verification",
            "status": "pending",
            "reference_id": request.reference_id,
            "metadata": request.metadata,
            "expires_at": expires_at.isoformat(),  # Конвертируем в строку для кэша
            "ip_address": http_request.client.host if http_request.client else None,
            "user_agent": http_request.headers.get("user-agent"),
        }
        
        await cache_service.set_verification_session(
            session_id=request_id,
            session_data=cache_data,
            expire_seconds=request.expires_in_minutes * 60,
        )

        response = SessionResponse(
            success=True,
            session_id=request_id,
            session_type="verification",
            expires_at=expires_at.isoformat(),  # Строка для JSON сериализации
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
        logger.info(f"Session data: {session_data}")

        if not session_data:
            raise NotFoundError(
                f"Verification session {session_id} not found or expired"
            )

        if session_data.get("status") != "pending":
            raise ValidationError(f"Verification session {session_id} is not active")

        # Обновление статуса сессии
        session_data["status"] = "processing"
        session_data["started_at"] = datetime.now(timezone.utc).isoformat()
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
            user_id=session_data.get("user_id"),
        )

        # Выполнение верификации
        # Получаем user и db из session_data
        from ..db.database import get_async_db_manager
        from ..db.crud import UserCRUD
        
        async with get_async_db_manager().get_session() as db:
            user_id = session_data.get("user_id")
            if not user_id:
                raise ValidationError("User ID not found in session")
            
            user = await UserCRUD.get_user(db, user_id)
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            # Вызываем внутреннюю функцию верификации
            verify_response = await _verify_face_internal(
                verify_request, 
                http_request, 
                user, 
                db
            )

        # Создаем новый ответ с правильным session_id сессии
        corrected_response = VerifyResponse(
            verification_id=verify_response.verification_id,
            session_id=session_id,  # Используем session_id сессии, а не request_id
            verified=verify_response.verified,
            confidence=verify_response.confidence,
            similarity_score=verify_response.similarity_score,
            threshold_used=verify_response.threshold_used,
            reference_id=verify_response.reference_id,
            processing_time=verify_response.processing_time,
            face_detected=verify_response.face_detected,
            face_quality=verify_response.face_quality,
            liveness_score=verify_response.liveness_score,
            liveness_passed=verify_response.liveness_passed,
            metadata=verify_response.metadata,
            request_id=verify_response.request_id,
        )

        # Обновление статуса сессии
        session_data["status"] = "completed"
        session_data["completed_at"] = datetime.now(timezone.utc).isoformat()
        session_data["response_data"] = corrected_response.dict()
        await cache_service.set_verification_session(
            session_id, session_data, expire_seconds=3600
        )  # 1 час

        logger.info(
            f"Verification in session {session_id} completed successfully, request {request_id}"
        )
        return corrected_response

    except NotFoundError as e:
        logger.warning(
            f"Not found error in verification session {session_id}, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": "NOT_FOUND",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except ValidationError as e:
        logger.warning(
            f"Validation error in verification session {session_id}, request {request_id}: {str(e)}"
        )

        # Обновляем статус сессии на "failed"
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except ProcessingError as e:
        logger.error(
            f"Processing error in verification session {session_id}, request {request_id}: {str(e)}"
        )

        # Обновляем статус сессии на "failed"
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
        logger.error(
            f"Unexpected error in verification session {session_id}, request {request_id}: {str(e)}"
        )
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Обновляем статус сессии на "failed"
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
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
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


@router.get("/verify/history", response_model=dict)
async def get_verification_history(
    user_id: Optional[str] = Query(None, description="ID пользователя для фильтрации"),
    limit: int = Query(100, ge=1, le=1000, description="Количество записей (1-1000)"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    status: Optional[str] = Query(None, description="Фильтр по статусу"),
    verified: Optional[bool] = Query(None, description="Фильтр по результату верификации"),
    date_from: Optional[str] = Query(None, description="Дата начала (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Дата окончания (YYYY-MM-DD)"),
    http_request: Request = None,
):
    """
    Получение истории верификации для пользователя.

    Args:
        user_id: ID пользователя для фильтрации
        limit: Количество записей (1-1000)
        offset: Смещение для пагинации
        status: Фильтр по статусу сессии
        verified: Фильтр по результату верификации
        date_from: Дата начала периода
        date_to: Дата окончания периода
        http_request: HTTP запрос

    Returns:
        dict: Список сессий верификации с пагинацией
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Getting verification history, request {request_id}"
        )

        # Инициализация сервисов
        from app.db.crud import VerificationSessionCRUD
        from app.db.database import get_async_db_manager
        from datetime import datetime as dt

        # Валидация дат
        parsed_date_from = None
        parsed_date_to = None
        
        if date_from:
            try:
                parsed_date_from = dt.fromisoformat(date_from)
            except ValueError:
                raise ValidationError("Invalid date_from format. Use YYYY-MM-DD")
                
        if date_to:
            try:
                parsed_date_to = dt.fromisoformat(date_to)
            except ValueError:
                raise ValidationError("Invalid date_to format. Use YYYY-MM-DD")

        # Валидация статуса
        valid_statuses = ["pending", "processing", "success", "failed", "completed"]
        if status and status not in valid_statuses:
            raise ValidationError(f"Invalid status. Must be one of: {valid_statuses}")

        # Получение сессий из БД
        async with get_async_db_manager().get_session() as db:
            # Если user_id не указан, получаем все сессии верификации
            # В реальном приложении здесь должна быть аутентификация
            if user_id:
                all_sessions = await VerificationSessionCRUD.get_user_sessions(db, user_id, limit=limit*2)  # Получаем больше для фильтрации
            else:
                # Упрощенная реализация - получаем все сессии верификации
                from sqlalchemy import select
                stmt = (
                    select(VerificationSession)
                    .where(VerificationSession.session_type == "verification")
                    .order_by(VerificationSession.created_at.desc())
                    .limit(limit * 2)  # Получаем больше для фильтрации
                )
                result = await db.execute(stmt)
                all_sessions = list(result.scalars().all())

        # Применение фильтров
        filtered_sessions = []
        for session in all_sessions:
            # Фильтр по статусу
            if status and session.status != status:
                continue
                
            # Фильтр по результату верификации
            if verified is not None:
                if session.is_match != verified:
                    continue
                    
            # Фильтр по дате
            if parsed_date_from and session.created_at < parsed_date_from:
                continue
            if parsed_date_to and session.created_at > parsed_date_to:
                continue
                
            filtered_sessions.append(session)

        # Применение пагинации
        total_count = len(filtered_sessions)
        paginated_sessions = filtered_sessions[offset:offset + limit]

        # Формирование результатов
        results = []
        for session in paginated_sessions:
            session_data = {
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
            }
            results.append(session_data)

        # Вычисление пагинации
        has_next = (offset + limit) < total_count
        has_prev = offset > 0

        response = {
            "success": True,
            "sessions": results,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_next": has_next,
            "has_prev": has_prev,
            "filters_applied": {
                "user_id": user_id,
                "status": status,
                "verified": verified,
                "date_from": date_from,
                "date_to": date_to,
            },
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Verification history retrieved: {len(results)} sessions, request {request_id}"
        )
        return response

    except ValidationError as e:
        logger.warning(
            f"Validation error getting verification history, request {request_id}: {str(e)}"
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
            f"Unexpected error getting verification history, request {request_id}: {str(e)}"
        )
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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


@router.get("/verify/{session_id}", response_model=dict)
async def get_verification_result(
    session_id: str, http_request: Request
):
    """
    Получение результата верификации по session_id.

    Args:
        session_id: ID сессии верификации
        http_request: HTTP запрос

    Returns:
        dict: Результат верификации
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Getting verification result for session {session_id}, request {request_id}"
        )

        # Инициализация сервисов
        from app.db.crud import VerificationSessionCRUD
        from app.db.database import get_async_db_manager

        # Получение сессии из БД
        async with get_async_db_manager().get_session() as db:
            session = await VerificationSessionCRUD.get_session(db, session_id)

        if not session:
            raise NotFoundError(f"Verification session {session_id} not found")

        # Проверяем, что это сессия верификации
        if session.session_type != "verification":
            raise ValidationError(
                f"Session {session_id} is not a verification session"
            )

        # Формирование ответа
        response = {
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
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "error_code": session.error_code,
            "error_message": session.error_message,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Verification result retrieved for session {session_id}, request {request_id}"
        )
        return response

    except NotFoundError as e:
        logger.warning(
            f"Verification session {session_id} not found, request {request_id}: {str(e)}"
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
            f"Validation error for verification session {session_id}, request {request_id}: {str(e)}"
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
            f"Unexpected error getting verification result for session {session_id}, request {request_id}: {str(e)}"
        )
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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



