from fastapi import APIRouter, Request, Query
from datetime import datetime, timezone
import uuid
import time
import hashlib
from typing import List, Optional

from ..config import settings
from ..db.models import Reference
from ..models.request import ReferenceCreateRequest, ReferenceUpdateRequest
from ..models.response import (
    ReferenceResponse,
    ReferenceListResponse,
    BaseResponse,
)
from ..models.reference import ReferenceCompare

# Опционально: модели для face analysis (используются в enhanced endpoints)
# from ..models.face import (
#     FaceEmbedding,
#     FaceEmbeddingComparison,
#     FaceQualityAssessment,
#     ComprehensiveFaceAnalysis,
# )
from ..services.storage_service import StorageService
from ..services.ml_service import MLService
from ..services.encryption_service import EncryptionService
from ..services.validation_service import ValidationService
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError

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
):
    request_id = str(uuid.uuid4())

    from app.db.crud import ReferenceCRUD
    from app.db.database import get_async_db_manager

    allowed_sort_fields = {
        "created_at",
        "updated_at",
        "quality_score",
        "usage_count",
        "label",
    }

    if sort_by not in allowed_sort_fields:
        raise ValidationError(f"Invalid sort field: {sort_by}")

    if sort_order not in {"asc", "desc"}:
        raise ValidationError("sort_order must be asc or desc")

    async with get_async_db_manager().get_session() as db:
        # get_all_references требует user_id, поэтому если не указан, получаем все через другой способ
        if user_id:
            references = await ReferenceCRUD.get_all_references(db, user_id)
        else:
            # Если user_id не указан, получаем все эталоны через запрос
            from sqlalchemy import select
            result = await db.execute(select(Reference))
            references = list(result.scalars().all())

    def apply_filters(ref: Reference) -> bool:
        if label and label not in (ref.label or ""):
            return False
        if is_active is not None and ref.is_active != is_active:
            return False
        if quality_min is not None and (ref.quality_score or 0) < quality_min:
            return False
        if quality_max is not None and (ref.quality_score or 0) > quality_max:
            return False
        return True

    filtered = list(filter(apply_filters, references))
    reverse = sort_order == "desc"

    def get_sort_key(ref: Reference):
        if sort_by == "created_at":
            return ref.created_at or datetime(1970, 1, 1, tzinfo=timezone.utc)
        elif sort_by == "updated_at":
            return ref.updated_at or datetime(1970, 1, 1, tzinfo=timezone.utc)
        elif sort_by == "quality_score":
            return ref.quality_score or 0.0
        elif sort_by == "usage_count":
            return ref.usage_count or 0
        elif sort_by == "label":
            return ref.label or ""
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    filtered.sort(key=get_sort_key, reverse=reverse)

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = filtered[start:end]

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


# ======================================================================
# GET ONE
# ======================================================================

@router.get("/reference/{reference_id}", response_model=ReferenceResponse)
async def get_reference(reference_id: str, http_request: Request):
    request_id = str(uuid.uuid4())

    from app.db.crud import ReferenceCRUD
    from app.db.database import get_async_db_manager

    async with get_async_db_manager().get_session() as db:
        ref = await ReferenceCRUD.get_reference_by_id(db, reference_id)

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


# ======================================================================
# CREATE
# ======================================================================

@router.post("/reference", response_model=ReferenceResponse)
async def create_reference(request: ReferenceCreateRequest, http_request: Request):
    request_id = str(uuid.uuid4())

    from app.db.crud import ReferenceCRUD
    from app.db.database import get_async_db_manager

    validation = ValidationService()
    storage = StorageService()
    ml = MLService()
    crypto = EncryptionService()

    validated = await validation.validate_image(
        request.image_data,
        settings.MAX_UPLOAD_SIZE,
        settings.ALLOWED_IMAGE_FORMATS,
    )

    if not validated.is_valid:
        raise ValidationError(validated.error_message)

    upload = (
        await storage.upload_image(
            image_data=validated.image_data,
            metadata={"request_id": request_id}
        )
        if settings.STORE_ORIGINAL_IMAGES
        else None
    )

    embedding_result = await ml.generate_embedding(validated.image_data)
    if not embedding_result["success"]:
        raise ProcessingError("Embedding generation failed")

    if embedding_result["quality_score"] < request.quality_threshold:
        raise ValidationError("Image quality below threshold")

    encrypted = await crypto.encrypt_embedding(embedding_result["embedding"])

    # Normalize encrypted bytes and hash
    embedding_encrypted = encrypted if isinstance(encrypted, bytes) else encrypted.encode()
    embedding_hash = hashlib.sha256(embedding_encrypted).hexdigest()

    async with get_async_db_manager().get_session() as db:
        # Генерируем уникальный ID для reference
        ref_id = str(uuid.uuid4())
        
        # Получаем версию для нового reference
        from sqlalchemy import select, func, desc
        latest_ref_result = await db.execute(
            select(Reference)
            .where(Reference.user_id == request.user_id)
            .order_by(desc(Reference.version))
            .limit(1)
        )
        latest_ref = latest_ref_result.scalar_one_or_none()
        version = (latest_ref.version + 1) if latest_ref else 1

        ref = await ReferenceCRUD.create_reference(
            db=db,
            user_id=request.user_id,
            embedding_encrypted=embedding_encrypted,
            embedding_hash=embedding_hash,
            quality_score=embedding_result["quality_score"],
            image_filename=f"reference_{ref_id}.jpg",
            image_size_mb=len(validated.image_data) / (1024 * 1024),
            image_format=validated.image_format,
            file_url=upload.get("file_url") if upload else None,
            face_landmarks=embedding_result.get("landmarks"),
        )

    return ReferenceResponse(
        success=True,
        reference_id=ref.id,
        user_id=ref.user_id,
        label=request.label,
        file_url=ref.file_url,
        created_at=ref.created_at,
        quality_score=ref.quality_score,
        metadata=request.metadata,
        request_id=request_id,
    )


# ======================================================================
# UPDATE
# ======================================================================

@router.put("/reference/{reference_id}", response_model=ReferenceResponse)
async def update_reference(
    reference_id: str,
    request: ReferenceUpdateRequest,
    http_request: Request,
):
    request_id = str(uuid.uuid4())

    from app.db.crud import ReferenceCRUD
    from app.db.database import get_async_db_manager

    async with get_async_db_manager().get_session() as db:
        ref = await ReferenceCRUD.get_reference_by_id(db, reference_id)
        if not ref:
            raise NotFoundError("Reference not found")

        updated = await ReferenceCRUD.update_reference(
            db,
            reference_id,
            label=request.label,
            metadata=request.metadata,
            is_active=request.is_active,
        )

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


# ======================================================================
# DELETE
# ======================================================================

@router.delete("/reference/{reference_id}", response_model=BaseResponse)
async def delete_reference(reference_id: str, http_request: Request):
    request_id = str(uuid.uuid4())

    from app.db.crud import ReferenceCRUD
    from app.db.database import get_async_db_manager

    async with get_async_db_manager().get_session() as db:
        ref = await ReferenceCRUD.get_reference_by_id(db, reference_id)
        if not ref:
            raise NotFoundError("Reference not found")

        await ReferenceCRUD.delete_reference(db, reference_id)

    return BaseResponse(
        success=True,
        message=f"Reference {reference_id} deleted",
        request_id=request_id,
    )


# ======================================================================
# COMPARE
# ======================================================================

@router.post("/compare", response_model=dict)
async def compare_with_references(
    request: ReferenceCompare,
    http_request: Request,
):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    from app.db.crud import ReferenceCRUD
    from app.db.database import get_async_db_manager

    validation = ValidationService()
    ml = MLService()
    crypto = EncryptionService()

    validated = await validation.validate_image(
        request.image_data,
        settings.MAX_UPLOAD_SIZE,
        settings.ALLOWED_IMAGE_FORMATS,
    )

    if not validated.is_valid:
        raise ValidationError(validated.error_message)

    async with get_async_db_manager().get_session() as db:
        if request.reference_ids:
            refs = [
                await ReferenceCRUD.get_reference_by_id(db, rid)
                for rid in request.reference_ids
            ]
            # Фильтруем None значения
            refs = [ref for ref in refs if ref is not None]
        elif request.user_id:
            refs = await ReferenceCRUD.get_all_references(db, request.user_id)
        else:
            # Если не указаны ни reference_ids, ни user_id, получаем все через запрос
            from sqlalchemy import select
            result = await db.execute(select(Reference))
            refs = list(result.scalars().all())

    # Валидация количества найденных эталонов
    if len(refs) > 100:
        raise ValidationError(f"Too many references found: {len(refs)}. Maximum allowed is 100.")

    results = []

    # Дополнительная проверка - если мы дошли до сюда, значит ValidationError не была выброшена
    print(f"DEBUG: Validation passed. Processing {len(refs)} references.")

    for ref in refs[: request.max_results]:
        if ref is None:
            continue
        # Проверяем, что у reference есть embedding_encrypted
        if not hasattr(ref, 'embedding_encrypted') or ref.embedding_encrypted is None:
            continue
        embedding = await crypto.decrypt_embedding(ref.embedding_encrypted)
        compare = await ml.compare_faces(
            validated.image_data,
            embedding,
            request.threshold,
        )
        if compare["success"]:
            results.append(
                {
                    "reference_id": ref.id,
                    "similarity_score": compare["similarity_score"],
                    "distance": compare["distance"],
                    "label": ref.label,
                }
            )

    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    return {
        "success": True,
        "request_id": request_id,
        "processing_time": time.time() - start_time,
        "results": results,
        "timestamp": datetime.now(timezone.utc),
    }