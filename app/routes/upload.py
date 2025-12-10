"""
API роуты для загрузки изображений.
Endpoint для загрузки и обработки изображений.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import uuid
import time
from ..config import settings

from ..models.request import UploadRequest
from ..models.response import UploadResponse, BaseResponse
from ..services.storage_service import StorageService
from ..services.ml_service import MLService
from ..services.database_service import DatabaseService
from ..services.cache_service import CacheService
from ..services.validation_service import ValidationService
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError

router = APIRouter()
logger = get_logger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_image(request: UploadRequest, http_request: Request):
    """
    Загрузка и обработка изображения.
    
    Args:
        request: Данные запроса с изображением
        http_request: HTTP запрос для получения метаданных
        
    Returns:
        UploadResponse: Результат загрузки и обработки
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting image upload request {request_id}")
        
        # Инициализация сервисов
        storage_service = StorageService()
        ml_service = MLService()
        validation_service = ValidationService()
        
        # Валидация изображения
        logger.info(f"Validating image for request {request_id}")
        validation_result = await validation_service.validate_image(
            request.image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.ALLOWED_IMAGE_FORMATS
        )
        
        if not validation_result.is_valid:
            raise ValidationError(f"Image validation failed: {validation_result.error_message}")
        
        upload_result = {
            "image_id": request_id,
            "file_url": None,
            "file_size": len(validation_result.image_data),
            "image_format": validation_result.image_format,
        }

        if settings.STORE_ORIGINAL_IMAGES:
            # Загрузка в хранилище (если разрешено)
            logger.info(f"Uploading image to storage for request {request_id}")
            upload_result = await storage_service.upload_image(
                image_data=validation_result.image_data,
                metadata={
                    "request_id": request_id,
                    "user_id": request.user_id,
                    "original_metadata": request.metadata,
                    "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                    "client_ip": http_request.client.host if http_request.client else None,
                    "user_agent": http_request.headers.get("user-agent")
                }
            )
        
        # Обработка ML сервисом (генерация эмбеддинга, анализ качества)
        logger.info(f"Processing image with ML service for request {request_id}")
        ml_result = await ml_service.process_image(
            image_data=validation_result.image_data,
            image_url=upload_result.get("file_url")
        )
        
        # Сохранение информации в БД (опционально)
        if request.user_id:
            logger.info(f"Saving image metadata to database for request {request_id}")
            # TODO: Сохранить метаданные изображения в БД
        
        processing_time = time.time() - start_time
        
        # Формирование ответа
        response = UploadResponse(
            success=True,
            image_id=upload_result.get("image_id", request_id),
            file_url=upload_result.get("file_url"),
            file_size=upload_result.get("file_size"),
            image_format=validation_result.image_format,
            image_dimensions=validation_result.dimensions,
            processing_time=processing_time,
            quality_score=ml_result.get("quality_score") if ml_result else None,
            request_id=request_id
        )
        
        # Удаляем исходный файл, если он был сохранен и включено авто-удаление
        if settings.STORE_ORIGINAL_IMAGES and settings.DELETE_SOURCE_AFTER_PROCESSING:
            try:
                await storage_service.delete_image(upload_result.get("image_id"))
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete source image {upload_result.get('image_id')}: {cleanup_error}")

        logger.info(f"Image upload completed successfully for request {request_id}")
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"validation_error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
    except ProcessingError as e:
        logger.error(f"Processing error for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_code": "PROCESSING_ERROR",
                "error_details": {"processing_error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc)
            }
        )


@router.post("/upload/batch", response_model=dict)
async def upload_images_batch(request: dict, http_request: Request):
    """
    Пакетная загрузка изображений.
    
    Args:
        request: Словарь с массивом изображений и метаданными
        http_request: HTTP запрос
        
    Returns:
        dict: Результаты пакетной загрузки
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Парсинг запроса
        images_data = request.get("images", [])
        metadata = request.get("metadata", {})
        
        if not images_data:
            raise ValidationError("No images provided in batch request")
        
        if len(images_data) > 50:  # Лимит пакетной загрузки
            raise ValidationError("Too many images in batch (max 50)")
        
        logger.info(f"Starting batch image upload with {len(images_data)} images, request {request_id}")
        
        # Инициализация сервисов
        storage_service = StorageService()
        ml_service = MLService()
        validation_service = ValidationService()
        
        results = []
        errors = []
        
        for i, image_data in enumerate(images_data):
            try:
                # Валидация изображения
                validation_result = await validation_service.validate_image(
                    image_data,
                    max_size=settings.MAX_UPLOAD_SIZE,
                    allowed_formats=settings.ALLOWED_IMAGE_FORMATS
                )
                
                if not validation_result.is_valid:
                    errors.append({
                        "index": i,
                        "error": validation_result.error_message,
                        "image_id": None
                    })
                    continue
                
                upload_result = {
                    "image_id": f"{request_id}-{i}",
                    "file_url": None,
                    "file_size": len(validation_result.image_data),
                    "image_format": validation_result.image_format,
                }

                if settings.STORE_ORIGINAL_IMAGES:
                    upload_result = await storage_service.upload_image(
                        image_data=validation_result.image_data,
                        metadata={
                            "batch_request_id": request_id,
                            "batch_index": i,
                            "batch_metadata": metadata,
                            "upload_timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                
                # Обработка ML сервисом
                ml_result = await ml_service.process_image(
                    image_data=validation_result.image_data,
                    image_url=upload_result.get("file_url")
                )
                
                results.append({
                    "index": i,
                    "image_id": upload_result.get("image_id"),
                    "file_url": upload_result.get("file_url"),
                    "file_size": upload_result.get("file_size"),
                    "image_format": validation_result.image_format,
                    "image_dimensions": validation_result.dimensions,
                    "quality_score": ml_result.get("quality_score") if ml_result else None,
                    "success": True
                })

                if settings.STORE_ORIGINAL_IMAGES and settings.DELETE_SOURCE_AFTER_PROCESSING:
                    try:
                        await storage_service.delete_image(upload_result.get("image_id"))
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to delete source image {upload_result.get('image_id')}: {cleanup_error}")
                
            except Exception as e:
                logger.error(f"Error processing image {i} in batch {request_id}: {str(e)}")
                errors.append({
                    "index": i,
                    "error": str(e),
                    "image_id": None
                })
        
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "request_id": request_id,
            "batch_id": str(uuid.uuid4()),
            "total_images": len(images_data),
            "successful_uploads": len(results),
            "failed_uploads": len(errors),
            "processing_time": processing_time,
            "results": results,
            "errors": errors,
            "timestamp": datetime.now(timezone.utc)
        }
        
        logger.info(f"Batch upload completed: {len(results)} successful, {len(errors)} failed, request {request_id}")
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error in batch request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"validation_error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in batch request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc)
            }
        )


@router.delete("/upload/{image_id}", response_model=BaseResponse)
async def delete_uploaded_image(image_id: str, http_request: Request):
    """
    Удаление загруженного изображения.
    
    Args:
        image_id: ID изображения для удаления
        http_request: HTTP запрос
        
    Returns:
        BaseResponse: Результат удаления
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting image deletion for image_id {image_id}, request {request_id}")
        
        # Инициализация сервисов
        storage_service = StorageService()
        
        # Удаление из хранилища
        await storage_service.delete_image(image_id)
        
        # Удаление из БД (если есть)
        # TODO: Удалить метаданные изображения из БД
        
        response = BaseResponse(
            success=True,
            message=f"Image {image_id} deleted successfully",
            request_id=request_id
        )
        
        logger.info(f"Image {image_id} deleted successfully, request {request_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error deleting image {image_id}, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "DELETE_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc)
            }
        )


@router.get("/upload/{image_id}", response_model=dict)
async def get_uploaded_image_info(image_id: str, http_request: Request):
    """
    Получение информации о загруженном изображении.
    
    Args:
        image_id: ID изображения
        http_request: HTTP запрос
        
    Returns:
        dict: Информация об изображении
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Getting image info for {image_id}, request {request_id}")
        
        # Инициализация сервисов
        storage_service = StorageService()
        
        # Получение информации из хранилища
        image_info = await storage_service.get_image_info(image_id)
        
        if not image_info:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error_code": "IMAGE_NOT_FOUND",
                    "error_details": {"image_id": image_id},
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc)
                }
            )
        
        response = {
            "success": True,
            "image_id": image_id,
            "file_url": image_info.get("file_url"),
            "file_size": image_info.get("file_size"),
            "image_format": image_info.get("image_format"),
            "image_dimensions": image_info.get("image_dimensions"),
            "created_at": image_info.get("created_at"),
            "metadata": image_info.get("metadata"),
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc)
        }
        
        logger.info(f"Image info retrieved successfully for {image_id}, request {request_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image info for {image_id}, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc)
            }
        )