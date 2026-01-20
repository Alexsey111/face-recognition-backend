"""
Reference API Routes - Управление эталонными изображениями.

Упрощённая версия с использованием ReferenceService.
"""

from fastapi import APIRouter, Request, Query, Depends
from typing import List, Optional

from ..config import settings
from ..models.request import ReferenceCreateRequest, ReferenceUpdateRequest
from ..models.response import (
    ReferenceResponse,
    ReferenceListResponse,
    BaseResponse,
)
from ..models.reference import ReferenceCompare
from ..services.reference_service import ReferenceService
from ..db.database import get_async_db
from ..routes.auth import get_current_user
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, NotFoundError
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

router = APIRouter(tags=["Reference"])
logger = get_logger(__name__)


# ======================================================================
# GET LIST
# ======================================================================

@router.get("/reference", response_model=ReferenceListResponse)
async def get_references(
    user_id: Optional[str] = Query(None),
    label: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    quality_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    quality_max: Optional[float] = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    http_request: Request = None,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Получение списка references с фильтрацией и пагинацией.
    """
    request_id = str(uuid.uuid4())
    
    try:
        reference_service = ReferenceService(db)
        
        # Получение всех references (с фильтрацией на уровне сервиса)
        if user_id:
            all_references = await reference_service.get_all_references(
                user_id=user_id,
                include_inactive=(is_active is None),
            )
        else:
            # Если user_id не указан, получаем все через прямой запрос
            from sqlalchemy import select
            from ..db.models import Reference
            result = await db.execute(select(Reference))
            all_references = list(result.scalars().all())

        # Применение фильтров
        filtered = []
        for ref in all_references:
            # Фильтр по label
            if label and label not in (ref.label or ""):
                continue
            # Фильтр по is_active
            if is_active is not None and ref.is_active != is_active:
                continue
            # Фильтр по quality
            if quality_min is not None and (ref.quality_score or 0) < quality_min:
                continue
            if quality_max is not None and (ref.quality_score or 0) > quality_max:
                continue
            
            filtered.append(ref)

        # Сортировка
        sort_key_map = {
            "created_at": lambda r: r.created_at or "",
            "updated_at": lambda r: r.updated_at or "",
            "quality_score": lambda r: r.quality_score or 0,
            "usage_count": lambda r: r.usage_count or 0,
            "label": lambda r: r.label or "",
        }
        
        if sort_by not in sort_key_map:
            raise ValidationError(f"Invalid sort field: {sort_by}")
        
        filtered.sort(key=sort_key_map[sort_by], reverse=(sort_order == "desc"))

        # Пагинация
        total = len(filtered)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = filtered[start:end]

        # Формирование ответа
        responses = [
            ReferenceResponse(
                success=True,
                reference_id=ref.id,
                user_id=ref.user_id,
                label=ref.label,
                file_url=ref.file_url,
                created_at=ref.created_at,
                updated_at=ref.updated_at,
                quality_score=ref.quality_score,
                usage_count=ref.usage_count or 0,
                last_used=ref.last_used,
                metadata=ref.metadata,
            )
            for ref in page_items
        ]

        return ReferenceListResponse(
            success=True,
            references=responses,
            total_count=total,
            page=page,
            per_page=per_page,
            has_next=end < total,
            has_prev=start > 0,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Error getting references: {e}")
        raise


# ======================================================================
# GET ONE
# ======================================================================

@router.get("/reference/{reference_id}", response_model=ReferenceResponse)
async def get_reference(
    reference_id: str, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Получение конкретного reference по ID.
    """
    request_id = str(uuid.uuid4())
    
    try:
        reference_service = ReferenceService(db)
        ref = await reference_service.get_reference(reference_id)

        if not ref:
            raise NotFoundError(f"Reference {reference_id} not found")

        return ReferenceResponse(
            success=True,
            reference_id=ref.id,
            user_id=ref.user_id,
            label=ref.label,
            file_url=ref.file_url,
            created_at=ref.created_at,
            updated_at=ref.updated_at,
            quality_score=ref.quality_score,
            usage_count=ref.usage_count or 0,
            last_used=ref.last_used,
            metadata=ref.metadata,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Error getting reference {reference_id}: {e}")
        raise


# ======================================================================
# CREATE
# ======================================================================

@router.post("/reference", response_model=ReferenceResponse)
async def create_reference(
    request: ReferenceCreateRequest, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Создание нового reference.
    """
    request_id = str(uuid.uuid4())
    
    try:
        reference_service = ReferenceService(db)

        # Создание reference через сервис
        ref = await reference_service.create_reference(
            user_id=request.user_id,
            image_data=request.image_data,
            label=request.label,
            quality_threshold=request.quality_threshold,
            metadata=request.metadata,
            store_original=settings.STORE_ORIGINAL_IMAGES,
        )

        logger.info(f"Reference created: {ref.id}")

        return ReferenceResponse(
            success=True,
            reference_id=ref.id,
            user_id=ref.user_id,
            label=ref.label,
            file_url=ref.file_url,
            created_at=ref.created_at,
            quality_score=ref.quality_score,
            metadata=ref.metadata,
            request_id=request_id,
        )

    except ValidationError as e:
        logger.warning(f"Validation error creating reference: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating reference: {e}")
        raise


# ======================================================================
# UPDATE
# ======================================================================

@router.put("/reference/{reference_id}", response_model=ReferenceResponse)
async def update_reference(
    reference_id: str,
    request: ReferenceUpdateRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Обновление metadata reference.
    """
    request_id = str(uuid.uuid4())
    
    try:
        reference_service = ReferenceService(db)

        # Обновление через сервис
        updated = await reference_service.update_reference(
            reference_id=reference_id,
            label=request.label,
            metadata=request.metadata,
            is_active=request.is_active,
        )

        logger.info(f"Reference updated: {reference_id}")

        return ReferenceResponse(
            success=True,
            reference_id=updated.id,
            user_id=updated.user_id,
            label=updated.label,
            file_url=updated.file_url,
            created_at=updated.created_at,
            updated_at=updated.updated_at,
            quality_score=updated.quality_score,
            metadata=updated.metadata,
            request_id=request_id,
        )

    except NotFoundError as e:
        logger.warning(f"Reference not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error updating reference: {e}")
        raise


# ======================================================================
# DELETE
# ======================================================================

@router.delete("/reference/{reference_id}", response_model=BaseResponse)
async def delete_reference(
    reference_id: str, 
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Удаление reference (soft delete по умолчанию).
    """
    request_id = str(uuid.uuid4())
    
    try:
        reference_service = ReferenceService(db)

        # Удаление через сервис
        await reference_service.delete_reference(
            reference_id=reference_id,
            soft_delete=True,  # Можно параметризовать через query param
        )

        logger.info(f"Reference deleted: {reference_id}")

        return BaseResponse(
            success=True,
            message=f"Reference {reference_id} deleted successfully",
            request_id=request_id,
        )

    except NotFoundError as e:
        logger.warning(f"Reference not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error deleting reference: {e}")
        raise


# ======================================================================
# COMPARE
# ======================================================================

@router.post("/compare", response_model=dict)
async def compare_with_references(
    request: ReferenceCompare,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Сравнение лица с несколькими references.
    """
    request_id = str(uuid.uuid4())
    
    try:
        reference_service = ReferenceService(db)

        # Сравнение через сервис
        results = await reference_service.compare_with_references(
            image_data=request.image_data,
            reference_ids=request.reference_ids,
            user_id=request.user_id,
            threshold=request.threshold,
            max_results=request.max_results,
        )

        logger.info(f"Comparison completed: {len(results)} matches")

        return {
            "success": True,
            "request_id": request_id,
            "results": results,
            "total_matches": len(results),
            "timestamp": http_request.state.timestamp if hasattr(http_request.state, 'timestamp') else None,
        }

    except ValidationError as e:
        logger.warning(f"Validation error in comparison: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        raise


# ======================================================================
# STATISTICS (BONUS)
# ======================================================================

@router.get("/reference/stats/{user_id}", response_model=dict)
async def get_reference_statistics(
    user_id: str,
    http_request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Получение статистики по references пользователя.
    """
    request_id = str(uuid.uuid4())
    
    try:
        reference_service = ReferenceService(db)
        stats = await reference_service.get_reference_statistics(user_id)

        return {
            "success": True,
            "user_id": user_id,
            "statistics": stats,
            "request_id": request_id,
        }

    except Exception as e:
        logger.error(f"Error getting statistics for user {user_id}: {e}")
        raise
