"""
API роуты для загрузки изображений.
Реализует систему загрузки с валидацией и поддержкой HEIC.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.db.database import get_async_db_manager
from app.routes.auth import get_current_user
from app.services.session_service import SessionService
from app.services.storage_service import StorageService
from app.services.validation_service import ValidationService
from app.tasks.cleanup import CleanupTasks
from app.utils.file_utils import ImageFileHandler, FileUtils, ImageValidator
from app.utils.logger import get_logger
from app.utils.validators import ValidationError

logger = get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])


def get_storage_service() -> StorageService:
    """Фабричная функция для создания StorageService."""
    return StorageService()


# =============================================================================
# Image Validation Endpoints (NEW - HEIC Support)
# =============================================================================


@router.post("/validate")
async def validate_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Валидация загруженного изображения.
    Поддерживаемые форматы: JPG, PNG, HEIC, HEIF, WebP

    Args:
        file: Файл изображения

    Returns:
        Информация об изображении и результат валидации
    """
    try:
        # Чтение файла
        file_data = await file.read()

        # Валидация с использованием ValidationService
        image, image_info = ValidationService().validate_uploaded_image(
            file_data=file_data,
            filename=file.filename,
            check_face=False,
        )

        # Определяем, было ли сконвертировано из HEIC
        original_format = image_info.get("format", "").upper()
        converted_from_heic = original_format in ["HEIC", "HEIF"]

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Изображение валидно",
                "image_info": {
                    "width": image_info["width"],
                    "height": image_info["height"],
                    "format": original_format,
                    "size_bytes": image_info["size_bytes"],
                    "size_mb": round(image_info["size_mb"], 2),
                    "mime_type": image_info["mime_type"],
                },
                "converted_from_heic": converted_from_heic,
                "dimensions": f"{image_info['width']}x{image_info['height']}",
            },
        )

    except ValidationError as e:
        logger.warning(f"Валидация не пройдена: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка валидации изображения: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера",
        )


@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, Any]:
    """
    Получение списка поддерживаемых форматов изображений.

    Returns:
        Информация о поддерживаемых форматах
    """
    return ValidationService().get_supported_formats_info()


@router.post("/convert-heic")
async def convert_heic_to_jpeg(file: UploadFile = File(...)) -> JSONResponse:
    """
    Конвертация HEIC/HEIF изображения в JPEG.

    Args:
        file: Файл изображения (HEIC/HEIF)

    Returns:
        Информация о конвертации и base64 JPEG
    """
    try:
        file_data = await file.read()

        # Проверяем, что это HEIC
        if not ImageFileHandler.is_heic_format(file_data, file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Файл не является HEIC/HEIF форматом",
            )

        # Конвертируем
        jpeg_data = ImageFileHandler.convert_heic_to_jpeg(file_data, quality=95)

        # Получаем информацию
        image_info = ImageFileHandler.get_image_info(jpeg_data, "converted.jpg")

        import base64

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Конвертация завершена",
                "jpeg_size_bytes": len(jpeg_data),
                "jpeg_size_mb": round(len(jpeg_data) / 1024 / 1024, 2),
                "dimensions": f"{image_info['width']}x{image_info['height']}",
                "jpeg_base64": base64.b64encode(jpeg_data).decode("utf-8"),
            },
        )

    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка конвертации HEIC: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка конвертации изображения",
        )


# =============================================================================
# Upload Session Endpoints (EXISTING)
# =============================================================================


@router.post("/", response_model=Dict[str, Any])
async def create_upload_session(current_user_id: str = Depends(get_current_user)):
    """
    Создание сессии загрузки

    Returns:
        Dict[str, Any]: {
            "session_id": str,
            "expires_at": datetime,
            "max_file_size_mb": float
        }
    """
    try:
        session = await SessionService.create_session(current_user_id)

        # Логирование действия
        async with get_async_db_manager().get_session() as db:
            from app.db.crud import AuditLogCRUD

            await AuditLogCRUD.log_action(
                db,
                action="upload_session_created",
                resource_type="upload_session",
                resource_id=session.session_id,
                user_id=current_user_id,
                description=f"Создана сессия загрузки: {session.session_id}",
            )

        logger.info(f"Сессия загрузки создана: {session.session_id}")

        return {
            "session_id": session.session_id,
            "expires_at": session.expiration_at,
            "max_file_size_mb": FileUtils.MAX_FILE_SIZE_MB,
        }

    except Exception as e:
        logger.error(f"Ошибка создания сессии загрузки: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось создать сессию загрузки",
        )


@router.post("/{session_id}/file", response_model=Dict[str, Any])
async def upload_file_to_session(
    session_id: str,
    file: UploadFile = File(...),
    current_user_id: str = Depends(get_current_user),
):
    """
    Загрузка файла в сессию

    Args:
        session_id: ID сессии загрузки
        file: Файл для загрузки

    Returns:
        Dict[str, Any]: {
            "file_key": str,
            "file_url": str,
            "file_size_mb": float,
            "file_hash": str,
            "session_id": str
        }
    """
    try:
        # Валидация сессии
        if not await SessionService.validate_session(session_id, current_user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недействительная или истекшая сессия загрузки",
            )

        # Проверяем размер ДО чтения
        if file.size and file.size > FileUtils.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Файл слишком большой. Максимум: {FileUtils.MAX_FILE_SIZE_MB}MB",
            )

        # Чтение содержимого файла
        file_content = await file.read()
        original_filename = file.filename

        # Валидация файла
        is_valid, error_msg = ImageValidator.validate_image(
            file_content, original_filename
        )

        if not is_valid:
            logger.warning(f"Валидация файла не пройдена: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=error_msg
            )

        # Конвертация в JPG если необходимо (включая HEIC)
        processed_content = file_content
        processed_filename = original_filename

        # Проверяем расширение и конвертируем если нужно
        ext = FileUtils.get_file_extension(original_filename).lower()
        if ext in [".heic", ".heif"]:
            processed_content = ImageFileHandler.convert_heic_to_jpeg(file_content)
            processed_filename = f"{original_filename.rsplit('.', 1)[0]}.jpg"
        elif ext != ".jpg":
            processed_content, processed_filename = FileUtils.convert_to_jpeg(
                file_content, original_filename
            )

        # Изменение размера если необходимо
        max_width, max_height = 1024, 1024
        dimensions = FileUtils.get_image_dimensions(processed_content)
        if dimensions[0] > max_width or dimensions[1] > max_height:
            processed_content = FileUtils.resize_if_needed(
                processed_content, max_width, max_height
            )

        # Загрузка в хранилище
        storage = get_storage_service()
        result = await storage.upload_image(
            image_data=processed_content,
            key=None,
            metadata={
                "user_id": current_user_id,
                "session_id": session_id,
                "original_name": original_filename,
                "processed_name": processed_filename,
                "file_hash": FileUtils.calculate_file_hash(processed_content),
                "upload_timestamp": datetime.utcnow().isoformat(),
            },
        )

        file_key = result["key"]
        file_url = result["file_url"]
        file_size_mb = FileUtils.get_file_size_mb(processed_content)
        file_hash = result["metadata"]["file_hash"]

        # Обновление сессии
        await SessionService.attach_file_to_session(
            session_id,
            user_id=current_user_id,
            file_key=file_key,
            file_size=file_size_mb,
            file_hash=file_hash,
        )

        logger.info(f"Файл загружен: {file_key} ({file_size_mb:.1f}МБ)")

        return {
            "file_key": file_key,
            "file_url": file_url,
            "file_size_mb": file_size_mb,
            "file_hash": file_hash,
            "session_id": session_id,
            "original_filename": original_filename,
            "processed_filename": processed_filename,
        }

    except HTTPException:
        raise
    except ValidationError as e:
        logger.warning(f"Ошибка валидации: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось загрузить файл",
        )


@router.get("/{session_id}", response_model=Dict[str, Any])
async def get_upload_status(
    session_id: str, current_user_id: str = Depends(get_current_user)
):
    """
    Получение статуса сессии загрузки

    Args:
        session_id: ID сессии

    Returns:
        Dict[str, Any]: Информация о сессии
    """
    try:
        session = await SessionService.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Сессия загрузки не найдена",
            )

        if session.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Неавторизованный доступ"
            )

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "file_key": session.file_key,
            "file_size_mb": session.file_size,
            "created_at": session.created_at,
            "expires_at": session.expiration_at,
            "is_expired": session.is_expired(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения статуса загрузки: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить статус загрузки",
        )


@router.delete("/{session_id}")
async def delete_upload_session(
    session_id: str, current_user_id: str = Depends(get_current_user)
):
    """
    Удаление сессии загрузки и файла

    Args:
        session_id: ID сессии

    Returns:
        Dict[str, str]: {"message": "Сессия загрузки удалена"}
    """
    try:
        session = await SessionService.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Сессия загрузки не найдена",
            )

        if session.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Неавторизованный доступ"
            )

        # Удаление файла из MinIO
        if session.file_key:
            storage = get_storage_service()
            await storage.delete_image(session.file_key)

        # Удаление сессии
        await SessionService.delete_session(session_id)

        # Логирование
        async with get_async_db_manager().get_session() as db:
            from app.db.crud import AuditLogCRUD

            await AuditLogCRUD.log_action(
                db,
                action="upload_session_deleted",
                resource_type="upload_session",
                resource_id=session_id,
                user_id=current_user_id,
                description=f"Сессия загрузки удалена: {session_id}",
            )

        logger.info(f"Сессия загрузки удалена: {session_id}")

        return {"message": "Сессия загрузки удалена"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка удаления сессии загрузки: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось удалить сессию загрузки",
        )


# =============================================================================
# Utility Endpoints
# =============================================================================


@router.get("/sessions/active", response_model=Dict[str, Any])
async def get_active_sessions(current_user_id: str = Depends(get_current_user)):
    """
    Получение активных сессий пользователя

    Returns:
        Dict[str, Any]: Список активных сессий
    """
    try:
        user_sessions = await SessionService.get_user_sessions(current_user_id)

        sessions_data = []
        for session in user_sessions:
            sessions_data.append(
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "expires_at": session.expiration_at,
                    "has_file": session.file_key is not None,
                    "file_size_mb": session.file_size,
                }
            )

        return {
            "user_id": current_user_id,
            "active_sessions_count": len(sessions_data),
            "sessions": sessions_data,
        }

    except Exception as e:
        logger.error(f"Ошибка получения активных сессий: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить активные сессии",
        )


@router.post("/cleanup")
async def cleanup_expired_sessions(current_user_id: str = Depends(get_current_user)):
    """
    Принудительная очистка истекших сессий (только для админов)

    Returns:
        Dict[str, Any]: Результат очистки
    """
    try:
        deleted_count = await CleanupTasks.cleanup_expired_upload_sessions()

        logger.info(f"Принудительная очистка: удалено {deleted_count} истекших сессий")

        return {
            "message": f"Очищено {deleted_count} истекших сессий",
            "deleted_sessions": deleted_count,
        }

    except Exception as e:
        logger.error(f"Ошибка принудительной очистки: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось выполнить очистку",
        )