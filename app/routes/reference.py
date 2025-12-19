"""API роуты для эталонных изображений."""

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import uuid
import time
from ..config import settings

from ..models.request import ReferenceCreateRequest, ReferenceUpdateRequest
from ..models.response import ReferenceResponse, ReferenceListResponse, BaseResponse
from ..models.reference import ReferenceSearch, ReferenceCompare, ReferenceCompareResult
from ..services.database_service import DatabaseService
from ..services.storage_service import StorageService
from ..services.ml_service import MLService
from ..services.encryption_service import EncryptionService
from ..services.validation_service import ValidationService
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError

router = APIRouter(prefix="/api/v1", tags=["Reference"])
logger = get_logger(__name__)


@router.get("/reference", response_model=ReferenceListResponse)
async def get_references(
    user_id: str = Query(None, description="ID пользователя для фильтрации"),
    label: str = Query(None, description="Метка эталона для поиска"),
    is_active: bool = Query(None, description="Статус активности"),
    quality_min: float = Query(
        None, ge=0.0, le=1.0, description="Минимальное качество"
    ),
    quality_max: float = Query(
        None, ge=0.0, le=1.0, description="Максимальное качество"
    ),
    page: int = Query(1, ge=1, description="Номер страницы"),
    per_page: int = Query(
        20, ge=1, le=100, description="Количество элементов на странице"
    ),
    sort_by: str = Query("created_at", description="Поле для сортировки"),
    sort_order: str = Query("desc", description="Порядок сортировки: asc или desc"),
    http_request: Request = None,
):
    """
    Получение списка эталонных изображений.

    Args:
        user_id: ID пользователя для фильтрации
        label: Метка эталона для поиска
        is_active: Статус активности
        quality_min: Минимальное качество
        quality_max: Максимальное качество
        page: Номер страницы
        per_page: Количество элементов на странице
        sort_by: Поле для сортировки
        sort_order: Порядок сортировки
        http_request: HTTP запрос

    Returns:
        ReferenceListResponse: Список эталонных изображений
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Getting references with filters, request {request_id}")

        # Инициализация сервисов
        from app.db.crud import ReferenceCRUD
        from app.db.database import get_async_db_manager

        # Валидация параметров сортировки
        allowed_sort_fields = [
            "created_at",
            "updated_at",
            "quality_score",
            "usage_count",
            "label",
        ]
        if sort_by not in allowed_sort_fields:
            raise ValidationError(
                f"Invalid sort_by field. Allowed: {allowed_sort_fields}"
            )

        if sort_order not in ["asc", "desc"]:
            raise ValidationError("sort_order must be 'asc' or 'desc'")

        # Получение эталонов из БД
        all_references = []
        async with get_async_db_manager().get_session() as db:
            # Если указан user_id, получаем только его эталоны
            if user_id:
                all_references = await ReferenceCRUD.get_all_references(db, user_id)
            else:
                # Если user_id не указан, получаем все эталоны (упрощенная реализация)
                # В реальном проекте здесь должен быть более эффективный запрос
                from sqlalchemy import select
                result = await db.execute(select(Reference))
                all_references = list(result.scalars().all())

        # Применение фильтров
        filtered_references = []
        for ref in all_references:
            # Фильтр по label
            if label and (not ref.label or label not in ref.label):
                continue
            # Фильтр по is_active
            if is_active is not None and ref.is_active != is_active:
                continue
            # Фильтр по quality_min
            if quality_min is not None and (ref.quality_score or 0) < quality_min:
                continue
            # Фильтр по quality_max
            if quality_max is not None and (ref.quality_score or 0) > quality_max:
                continue
            filtered_references.append(ref)

        # Сортировка
        reverse = sort_order == "desc"
        if sort_by == "created_at":
            filtered_references.sort(key=lambda x: x.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            filtered_references.sort(key=lambda x: x.updated_at or datetime.min, reverse=reverse)
        elif sort_by == "quality_score":
            filtered_references.sort(key=lambda x: x.quality_score or 0, reverse=reverse)
        elif sort_by == "usage_count":
            filtered_references.sort(key=lambda x: x.usage_count or 0, reverse=reverse)
        elif sort_by == "label":
            filtered_references.sort(key=lambda x: x.label or "", reverse=reverse)

        # Пагинация
        total_count = len(filtered_references)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_references = filtered_references[start_idx:end_idx]

        # Создание ответов
        references = []
        for ref in paginated_references:
            ref_data = {
                "success": True,
                "reference_id": ref.id,
                "user_id": ref.user_id,
                "label": ref.label,
                "file_url": ref.file_url,
                "created_at": ref.created_at,
                "updated_at": ref.updated_at,
                "quality_score": ref.quality_score,
                "usage_count": ref.usage_count or 0,
                "last_used": ref.last_used,
                "metadata": ref.metadata,
            }
            references.append(ReferenceResponse(**ref_data))

        # Вычисление пагинации
        has_next = end_idx < total_count
        has_prev = start_idx > 0

        response = ReferenceListResponse(
            success=True,
            references=references,
            total_count=total_count,
            page=page,
            per_page=per_page,
            has_next=has_next,
            has_prev=has_prev,
            filters_applied={
                "user_id": user_id,
                "label": label,
                "is_active": is_active,
                "quality_min": quality_min,
                "quality_max": quality_max,
            },
            request_id=request_id,
        )

        logger.info(f"Retrieved {len(references)} references, request {request_id}")
        return response

    except ValidationError as e:
        logger.warning(
            f"Validation error getting references, request {request_id}: {str(e)}"
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
        logger.error(f"Error getting references, request {request_id}: {str(e)}")
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


@router.get("/reference/{reference_id}", response_model=ReferenceResponse)
async def get_reference(reference_id: str, http_request: Request):
    """
    Получение информации о конкретном эталонном изображении.

    Args:
        reference_id: ID эталона
        http_request: HTTP запрос

    Returns:
        ReferenceResponse: Информация об эталоне
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Getting reference {reference_id}, request {request_id}")

        # Инициализация сервисов
        from app.db.crud import ReferenceCRUD
        from app.db.database import get_async_db_manager

        # Получение эталона из БД
        async with get_async_db_manager().get_session() as db:
            reference_data = await ReferenceCRUD.get_reference_by_id(db, reference_id)

        if not reference_data:
            raise NotFoundError(f"Reference {reference_id} not found")

        # Удаляем эмбеддинг из ответа
        reference_data.pop("embedding", None)

        response = ReferenceResponse(**reference_data)
        response.request_id = request_id

        logger.info(
            f"Reference {reference_id} retrieved successfully, request {request_id}"
        )
        return response

    except NotFoundError as e:
        logger.warning(
            f"Reference {reference_id} not found, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": "REFERENCE_NOT_FOUND",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(
            f"Error getting reference {reference_id}, request {request_id}: {str(e)}"
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


@router.post("/reference", response_model=ReferenceResponse)
async def create_reference(request: ReferenceCreateRequest, http_request: Request):
    """
    Создание нового эталонного изображения.

    Args:
        request: Данные для создания эталона
        http_request: HTTP запрос

    Returns:
        ReferenceResponse: Созданный эталон
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            f"Creating reference for user {request.user_id}, request {request_id}"
        )

        # Инициализация сервисов
        validation_service = ValidationService()
        storage_service = StorageService()
        ml_service = MLService()
        encryption_service = EncryptionService()
        db_service = DatabaseService()

        # Валидация изображения
        validation_result = await validation_service.validate_image(
            request.image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # Загрузка изображения в хранилище (при необходимости)
        upload_result = None
        if settings.STORE_ORIGINAL_IMAGES:
            upload_result = await storage_service.upload_image(
                image_data=validation_result.image_data,
                metadata={
                    "request_id": request_id,
                    "user_id": request.user_id,
                    "type": "reference",
                    "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Генерация эмбеддинга
        embedding_result = await ml_service.generate_embedding(
            validation_result.image_data
        )

        if not embedding_result.get("success"):
            raise ProcessingError("Failed to generate embedding for reference")

        # Шифрование эмбеддинга
        encrypted_embedding = await encryption_service.encrypt_embedding(
            embedding_result["embedding"]
        )

        # Проверка качества
        quality_score = embedding_result.get("quality_score", 0.0)
        if quality_score < request.quality_threshold:
            raise ValidationError(
                f"Image quality {quality_score} is below threshold {request.quality_threshold}"
            )

        # Создание записи в БД
        async with get_async_db_manager().get_session() as db:
            created_reference = await ReferenceCRUD.create_reference(
                db,
                user_id=request.user_id,
                embedding=str(encrypted_embedding),  # Конвертируем bytes в строку для БД
                embedding_encrypted=encrypted_embedding,
                embedding_hash=str(hash(encrypted_embedding)),  # Простой хеш
                quality_score=quality_score,
                image_filename=upload_result.get("file_url") if upload_result else f"reference_{request_id}.jpg",
                image_size_mb=(
                    upload_result.get("file_size", len(validation_result.image_data)) / (1024 * 1024)
                    if upload_result
                    else len(validation_result.image_data) / (1024 * 1024)
                ),
                image_format=validation_result.image_format,
                file_url=upload_result.get("file_url") if upload_result else None
            )

        # Удаляем исходный файл, если он был сохранен и включено авто-удаление
        if (
            settings.STORE_ORIGINAL_IMAGES
            and settings.DELETE_SOURCE_AFTER_PROCESSING
            and upload_result
        ):
            try:
                await storage_service.delete_image(upload_result.get("image_id"))
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to delete reference source image {upload_result.get('image_id')}: {cleanup_error}"
                )

        # Удаляем эмбеддинг из ответа
        created_reference.pop("embedding", None)

        response = ReferenceResponse(**created_reference)
        response.request_id = request_id

        logger.info(
            f"Reference created successfully: {created_reference['id']}, request {request_id}"
        )
        return response

    except ValidationError as e:
        logger.warning(
            f"Validation error creating reference, request {request_id}: {str(e)}"
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
            f"Processing error creating reference, request {request_id}: {str(e)}"
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
            f"Unexpected error creating reference, request {request_id}: {str(e)}"
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


@router.put("/reference/{reference_id}", response_model=ReferenceResponse)
async def update_reference(
    reference_id: str, request: ReferenceUpdateRequest, http_request: Request
):
    """
    Обновление эталонного изображения.

    Args:
        reference_id: ID эталона для обновления
        request: Данные для обновления
        http_request: HTTP запрос

    Returns:
        ReferenceResponse: Обновленный эталон
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Updating reference {reference_id}, request {request_id}")

        # Инициализация сервисов
        from app.db.crud import ReferenceCRUD
        from app.db.database import get_async_db_manager
        validation_service = ValidationService()
        storage_service = StorageService()
        ml_service = MLService()
        encryption_service = EncryptionService()

        # Получение существующего эталона
        async with get_async_db_manager().get_session() as db:
            existing_reference = await ReferenceCRUD.get_reference_by_id(db, reference_id)

        if not existing_reference:
            raise NotFoundError(f"Reference {reference_id} not found")

        # Подготовка данных для обновления
        update_data = {}

        # Обновление метки
        if request.label is not None:
            update_data["label"] = request.label

        # Обновление метаданных
        if request.metadata is not None:
            existing_metadata = existing_reference.get("metadata", {})
            existing_metadata.update(request.metadata)
            update_data["metadata"] = existing_metadata

        # Обновление порога качества
        if request.quality_threshold is not None:
            update_data["quality_threshold"] = request.quality_threshold

        # Обновление статуса активности
        if request.is_active is not None:
            update_data["is_active"] = request.is_active

        # Обновление изображения (если предоставлено)
        if request.image_data:
            # Валидация нового изображения
            validation_result = await validation_service.validate_image(
                request.image_data,
                max_size=settings.MAX_UPLOAD_SIZE,
                allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
            )

            if not validation_result.is_valid:
                raise ValidationError(
                    f"New image validation failed: {validation_result.error_message}"
                )

            upload_result = None
            if settings.STORE_ORIGINAL_IMAGES:
                upload_result = await storage_service.upload_image(
                    image_data=validation_result.image_data,
                    metadata={
                        "request_id": request_id,
                        "reference_id": reference_id,
                        "type": "reference_update",
                        "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            # Генерация нового эмбеддинга
            embedding_result = await ml_service.generate_embedding(
                validation_result.image_data
            )

            if not embedding_result.get("success"):
                raise ProcessingError(
                    "Failed to generate embedding for updated reference"
                )

            # Шифрование нового эмбеддинга
            encrypted_embedding = await encryption_service.encrypt_embedding(
                embedding_result["embedding"]
            )

            # Проверка качества нового изображения
            quality_score = embedding_result.get("quality_score", 0.0)
            quality_threshold = request.quality_threshold or existing_reference.get(
                "quality_threshold", 0.8
            )

            if quality_score < quality_threshold:
                raise ValidationError(
                    f"New image quality {quality_score} is below threshold {quality_threshold}"
                )

            # Обновление данных изображения
            update_data.update(
                {
                    "file_url": (
                        upload_result.get("file_url")
                        if upload_result
                        else existing_reference.get("file_url")
                    ),
                    "file_size": (
                        upload_result.get("file_size")
                        if upload_result
                        else len(validation_result.image_data)
                    ),
                    "image_format": validation_result.image_format,
                    "image_dimensions": validation_result.dimensions,
                    "embedding": encrypted_embedding,
                    "quality_score": quality_score,
                }
            )

            if (
                settings.STORE_ORIGINAL_IMAGES
                and settings.DELETE_SOURCE_AFTER_PROCESSING
                and upload_result
            ):
                try:
                    await storage_service.delete_image(upload_result.get("image_id"))
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to delete updated reference source image {upload_result.get('image_id')}: {cleanup_error}"
                    )

        # Обновление в БД
        async with get_db_session() as db:
            updated_reference = await ReferenceCRUD.update_reference(db, reference_id, **update_data)

        # Удаляем эмбеддинг из ответа
        updated_reference.pop("embedding", None)

        response = ReferenceResponse(**updated_reference)
        response.request_id = request_id

        logger.info(
            f"Reference {reference_id} updated successfully, request {request_id}"
        )
        return response

    except (ValidationError, NotFoundError) as e:
        logger.warning(
            f"Validation/NotFound error updating reference {reference_id}, request {request_id}: {str(e)}"
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
            f"Processing error updating reference {reference_id}, request {request_id}: {str(e)}"
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
            f"Unexpected error updating reference {reference_id}, request {request_id}: {str(e)}"
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


# Alias endpoint to meet /update-reference contract
@router.put("/update-reference", response_model=ReferenceResponse)
async def update_reference_alias(
    reference_id: str, request: ReferenceUpdateRequest, http_request: Request
):
    """
    Алиас для обновления эталонного изображения по требованиям ТЗ.
    Маршрут перенаправляет на основной обработчик.
    """
    return await update_reference(reference_id, request, http_request)


@router.delete("/reference/{reference_id}", response_model=BaseResponse)
async def delete_reference(reference_id: str, http_request: Request):
    """
    Удаление эталонного изображения.

    Args:
        reference_id: ID эталона для удаления
        http_request: HTTP запрос

    Returns:
        BaseResponse: Результат удаления
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Deleting reference {reference_id}, request {request_id}")

        # Инициализация сервисов
        from app.db.crud import ReferenceCRUD
        from app.db.database import get_async_db_manager
        storage_service = StorageService()

        # Получение информации о эталоне для удаления файла
        async with get_async_db_manager().get_session() as db:
            reference_data = await ReferenceCRUD.get_reference_by_id(db, reference_id)

            if not reference_data:
                raise NotFoundError(f"Reference {reference_id} not found")

            # Удаление из БД
            await ReferenceCRUD.delete_reference(db, reference_id)

        # Удаление файла из хранилища
        if reference_data.get("file_url"):
            await storage_service.delete_image_by_url(reference_data["file_url"])

        response = BaseResponse(
            success=True,
            message=f"Reference {reference_id} deleted successfully",
            request_id=request_id,
        )

        logger.info(
            f"Reference {reference_id} deleted successfully, request {request_id}"
        )
        return response

    except NotFoundError as e:
        logger.warning(
            f"Reference {reference_id} not found, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": "REFERENCE_NOT_FOUND",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(
            f"Error deleting reference {reference_id}, request {request_id}: {str(e)}"
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


@router.post("/reference/compare", response_model=dict)
async def compare_with_references(request: ReferenceCompare, http_request: Request):
    """
    Сравнение изображения с эталонными изображениями.

    Args:
        request: Данные для сравнения
        http_request: HTTP запрос

    Returns:
        dict: Результаты сравнения
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting reference comparison, request {request_id}")

        # Инициализация сервисов
        from app.db.crud import ReferenceCRUD
        from app.db.database import get_async_db_manager
        validation_service = ValidationService()
        ml_service = MLService()
        encryption_service = EncryptionService()

        # Валидация изображения
        validation_result = await validation_service.validate_image(
            request.image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # Получение эталонов для сравнения
        references = []
        async with get_async_db_manager().get_session() as db:
            if request.reference_ids:
                # Используем указанные ID эталонов
                for ref_id in request.reference_ids:
                    ref_data = await ReferenceCRUD.get_reference_by_id(db, ref_id)
                    if ref_data and ref_data.get("is_active", True):
                        references.append(ref_data)
            else:
                # Получаем все активные эталоны пользователя
                if not request.user_id:
                    raise ValidationError(
                        "Either reference_ids or user_id must be provided"
                    )

                # Получаем все эталоны пользователя и фильтруем активные
                all_references = await ReferenceCRUD.get_all_references(db, request.user_id)
                references = [ref for ref in all_references if ref.is_active]

        if not references:
            raise NotFoundError("No active references found for comparison")

        # Ограничиваем количество эталонов для сравнения
        if len(references) > 100:
            raise ValidationError("Too many references for comparison (max 100)")

        # Выполнение сравнения
        results = []

        for reference in references[: request.max_results]:
            try:
                # Дешифрация эмбеддинга эталона
                if reference.get("embedding"):
                    reference_embedding = await encryption_service.decrypt_embedding(
                        reference["embedding"]
                    )

                    # Сравнение с ML сервисом
                    compare_result = await ml_service.compare_faces(
                        image_data=validation_result.image_data,
                        reference_embedding=reference_embedding,
                        threshold=request.threshold,
                    )

                    if compare_result.get("success"):
                        results.append(
                            {
                                "reference_id": reference["id"],
                                "label": reference.get("label"),
                                "user_id": reference["user_id"],
                                "similarity_score": compare_result.get(
                                    "similarity_score", 0.0
                                ),
                                "quality_score": reference.get("quality_score"),
                                "distance": compare_result.get("distance"),
                                "processing_time": compare_result.get(
                                    "processing_time", 0.0
                                ),
                                "metadata": (
                                    reference.get("metadata")
                                    if request.include_metadata
                                    else None
                                ),
                            }
                        )
            except Exception as e:
                logger.warning(
                    f"Error comparing with reference {reference['id']}: {str(e)}"
                )
                continue

        # Сортировка результатов по убыванию схожести
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        processing_time = time.time() - start_time

        response = {
            "success": True,
            "request_id": request_id,
            "total_references_compared": len(references),
            "successful_comparisons": len(results),
            "threshold_used": request.threshold,
            "processing_time": processing_time,
            "results": results,
            "timestamp": datetime.now(timezone.utc),
        }

        logger.info(
            f"Reference comparison completed: {len(results)} matches found, request {request_id}"
        )
        return response

    except ValidationError as e:
        logger.warning(
            f"Validation error in reference comparison, request {request_id}: {str(e)}"
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

    except NotFoundError as e:
        logger.warning(
            f"NotFound error in reference comparison, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": "NOT_FOUND",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(
            f"Unexpected error in reference comparison, request {request_id}: {str(e)}"
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
