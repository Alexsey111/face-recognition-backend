"""
Middleware для обработки ошибок.
Централизованная обработка исключений и форматирование ответов с ошибками.
"""

from typing import Any, Optional, Dict, Union, List
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import traceback
from datetime import datetime, timezone
import uuid

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import (
    ValidationError,
    ProcessingError, 
    DatabaseError,
    NotFoundError,
    UnauthorizedError,
    EncryptionError,
    StorageError,
    CacheError,
    MLServiceError,
    WebhookError
)

logger = get_logger(__name__)


class ErrorHandlerMiddleware:
    """
    Middleware для централизованной обработки ошибок.
    """
    
    def __init__(self):
        self.error_mapping = {
            # Кастомные исключения
            ValidationError: {"status_code": 400, "error_code": "VALIDATION_ERROR"},
            ProcessingError: {"status_code": 422, "error_code": "PROCESSING_ERROR"},
            DatabaseError: {"status_code": 500, "error_code": "DATABASE_ERROR"},
            NotFoundError: {"status_code": 404, "error_code": "NOT_FOUND"},
            UnauthorizedError: {"status_code": 401, "error_code": "UNAUTHORIZED"},
            EncryptionError: {"status_code": 500, "error_code": "ENCRYPTION_ERROR"},
            StorageError: {"status_code": 500, "error_code": "STORAGE_ERROR"},
            CacheError: {"status_code": 500, "error_code": "CACHE_ERROR"},
            MLServiceError: {"status_code": 502, "error_code": "ML_SERVICE_ERROR"},
            WebhookError: {"status_code": 502, "error_code": "WEBHOOK_ERROR"},
            
            # Стандартные исключения
            ValueError: {"status_code": 400, "error_code": "VALUE_ERROR"},
            TypeError: {"status_code": 400, "error_code": "TYPE_ERROR"},
            KeyError: {"status_code": 400, "error_code": "KEY_ERROR"},
            AttributeError: {"status_code": 500, "error_code": "ATTRIBUTE_ERROR"},
            FileNotFoundError: {"status_code": 404, "error_code": "FILE_NOT_FOUND"},
            PermissionError: {"status_code": 403, "error_code": "PERMISSION_ERROR"},
            ConnectionError: {"status_code": 503, "error_code": "CONNECTION_ERROR"},
            TimeoutError: {"status_code": 504, "error_code": "TIMEOUT_ERROR"},
        }
    
    async def __call__(self, request: Request, call_next):
        """
        Обработка запроса с перехватом ошибок.
        
        Args:
            request: HTTP запрос
            call_next: Следующий обработчик
            
        Returns:
            Response: HTTP ответ
        """
        try:
            return await call_next(request)
            
        except HTTPException:
            # FastAPI исключения обрабатываем отдельно
            raise
            
        except Exception as exc:
            # Все остальные исключения обрабатываем централизованно
            return await self._handle_exception(request, exc)
    
    async def _handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """
        Обработка исключения и формирование ответа.
        
        Args:
            request: HTTP запрос
            exc: Исключение
            
        Returns:
            JSONResponse: Ответ с ошибкой
        """
        # Получаем информацию об ошибке
        error_info = self._get_error_info(exc)
        
        # Генерируем ID для отслеживания
        error_id = str(uuid.uuid4())
        
        # Подготавливаем данные для логирования
        log_data = self._prepare_error_log_data(request, exc, error_info, error_id)
        
        # Логируем ошибку
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
            extra=log_data,
            exc_info=True
        )
        
        # Формируем ответ
        error_response = self._create_error_response(exc, error_info, error_id, request)
        
        return JSONResponse(
            status_code=error_info["status_code"],
            content=jsonable_encoder(error_response),
            headers={"X-Error-ID": error_id}
        )
    
    def _get_error_info(self, exc: Exception) -> Dict[str, Any]:
        """
        Получение информации об ошибке.
        
        Args:
            exc: Исключение
            
        Returns:
            Dict[str, Any]: Информация об ошибке
        """
        # Ищем точное совпадение типа исключения
        if type(exc) in self.error_mapping:
            return self.error_mapping[type(exc)]
        
        # Ищем базовый класс исключения
        for exc_type, info in self.error_mapping.items():
            if isinstance(exc, exc_type):
                return info
        
        # По умолчанию - внутренняя ошибка сервера
        return {"status_code": 500, "error_code": "INTERNAL_ERROR"}
    
    def _prepare_error_log_data(self, request: Request, exc: Exception, error_info: Dict[str, Any], error_id: str) -> Dict[str, Any]:
        """
        Подготовка данных для логирования ошибки.
        
        Args:
            request: HTTP запрос
            exc: Исключение
            error_info: Информация об ошибке
            error_id: ID ошибки
            
        Returns:
            Dict[str, Any]: Данные для логирования
        """
        log_data = {
            "error_id": error_id,
            "error_type": type(exc).__name__,
            "error_code": error_info["error_code"],
            "status_code": error_info["status_code"],
            "request_method": request.method,
            "request_path": request.url.path,
            "request_query": str(request.query_params) if request.query_params else None,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Добавляем информацию о пользователе если есть
        if hasattr(request.state, 'user_id'):
            log_data["user_id"] = request.state.user_id
        
        # Добавляем stack trace только в debug режиме
        if settings.DEBUG:
            log_data["stack_trace"] = traceback.format_exc()
        
        return log_data
    
    def _create_error_response(self, exc: Exception, error_info: Dict[str, Any], error_id: str, request: Request) -> Dict[str, Any]:
        """
        Создание ответа с ошибкой.
        
        Args:
            exc: Исключение
            error_info: Информация об ошибке
            error_id: ID ошибки
            request: HTTP запрос
            
        Returns:
            Dict[str, Any]: Ответ с ошибкой
        """
        # Базовый ответ
        response = {
            "success": False,
            "error_code": error_info["error_code"],
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": self._get_error_message(exc, error_info),
            "path": request.url.path,
            "method": request.method
        }
        
        # Добавляем детали ошибки в debug режиме
        if settings.DEBUG:
            response["error_details"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc()
            }
        
        # Специфичная обработка для разных типов ошибок
        if isinstance(exc, ValidationError):
            response["validation_errors"] = self._extract_validation_errors(exc)
        elif isinstance(exc, NotFoundError):
            response["resource_type"] = self._extract_resource_type(exc)
        elif isinstance(exc, UnauthorizedError):
            response["auth_required"] = True
        
        return response
    
    def _get_error_message(self, exc: Exception, error_info: Dict[str, Any]) -> str:
        """
        Получение сообщения об ошибке.
        
        Args:
            exc: Исключение
            error_info: Информация об ошибке
            
        Returns:
            str: Сообщение об ошибке
        """
        # Для кастомных исключений используем их сообщения
        if hasattr(exc, 'message'):
            return exc.message
        
        # Для стандартных исключений формируем понятные сообщения
        error_code = error_info["error_code"]
        
        messages = {
            "VALIDATION_ERROR": "Validation failed. Please check your input data.",
            "PROCESSING_ERROR": "Processing failed. Please try again later.",
            "DATABASE_ERROR": "Database operation failed. Please try again later.",
            "NOT_FOUND": "The requested resource was not found.",
            "UNAUTHORIZED": "Authentication required. Please provide valid credentials.",
            "ENCRYPTION_ERROR": "Data encryption/decryption failed.",
            "STORAGE_ERROR": "File storage operation failed.",
            "CACHE_ERROR": "Cache operation failed.",
            "ML_SERVICE_ERROR": "ML service is temporarily unavailable.",
            "WEBHOOK_ERROR": "Webhook delivery failed.",
            "VALUE_ERROR": "Invalid value provided.",
            "TYPE_ERROR": "Invalid data type provided.",
            "KEY_ERROR": "Required key not found in data.",
            "FILE_NOT_FOUND": "The requested file was not found.",
            "PERMISSION_ERROR": "Permission denied.",
            "CONNECTION_ERROR": "Connection to external service failed.",
            "TIMEOUT_ERROR": "Request timed out.",
            "INTERNAL_ERROR": "An internal server error occurred."
        }
        
        return messages.get(error_code, f"An error occurred: {error_code}")
    
    def _extract_validation_errors(self, exc: ValidationError) -> List[Dict[str, Any]]:
        """
        Извлечение ошибок валидации.
        
        Args:
            exc: Исключение валидации
            
        Returns:
            List[Dict[str, Any]]: Список ошибок валидации
        """
        # Если исключение содержит детали валидации
        if hasattr(exc, 'errors') and exc.errors:
            return exc.errors
        
        # По умолчанию возвращаем общую ошибку
        return [{"field": "general", "message": str(exc)}]
    
    def _extract_resource_type(self, exc: NotFoundError) -> str:
        """
        Извлечение типа ресурса из исключения.
        
        Args:
            exc: Исключение NotFoundError
            
        Returns:
            str: Тип ресурса
        """
        # Если исключение содержит тип ресурса
        if hasattr(exc, 'resource_type'):
            return exc.resource_type
        
        # По умолчанию
        return "resource"
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Получение IP адреса клиента.
        
        Args:
            request: HTTP запрос
            
        Returns:
            str: IP адрес клиента
        """
        # Проверяем различные заголовки для получения реального IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Используем IP из соединения
        if request.client:
            return request.client.host
        
        return "unknown"


class GlobalExceptionHandler:
    """
    Глобальный обработчик исключений.
    """
    
    @staticmethod
    async def handle_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        """
        Обработка ошибок валидации.
        
        Args:
            request: HTTP запрос
            exc: Исключение валидации
            
        Returns:
            JSONResponse: Ответ с ошибкой валидации
        """
        error_id = str(uuid.uuid4())
        
        response = {
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Validation failed",
            "validation_errors": GlobalExceptionHandler._extract_validation_errors(exc),
            "path": request.url.path,
            "method": request.method
        }
        
        logger.warning(
            f"Validation error: {str(exc)}",
            extra={
                "error_id": error_id,
                "validation_errors": response["validation_errors"],
                "path": request.url.path,
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=400,
            content=response,
            headers={"X-Error-ID": error_id}
        )
    
    @staticmethod
    async def handle_not_found_error(request: Request, exc: NotFoundError) -> JSONResponse:
        """
        Обработка ошибок "не найдено".
        
        Args:
            request: HTTP запрос
            exc: Исключение NotFoundError
            
        Returns:
            JSONResponse: Ответ с ошибкой "не найдено"
        """
        error_id = str(uuid.uuid4())
        
        response = {
            "success": False,
            "error_code": "NOT_FOUND",
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": f"Resource not found: {str(exc)}",
            "resource_type": getattr(exc, 'resource_type', 'resource'),
            "path": request.url.path,
            "method": request.method
        }
        
        logger.info(
            f"Resource not found: {str(exc)}",
            extra={
                "error_id": error_id,
                "resource_type": response["resource_type"],
                "path": request.url.path,
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=404,
            content=response,
            headers={"X-Error-ID": error_id}
        )
    
    @staticmethod
    async def handle_unauthorized_error(request: Request, exc: UnauthorizedError) -> JSONResponse:
        """
        Обработка ошибок авторизации.
        
        Args:
            request: HTTP запрос
            exc: Исключение UnauthorizedError
            
        Returns:
            JSONResponse: Ответ с ошибкой авторизации
        """
        error_id = str(uuid.uuid4())
        
        response = {
            "success": False,
            "error_code": "UNAUTHORIZED",
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": str(exc) or "Authentication required",
            "auth_required": True,
            "path": request.url.path,
            "method": request.method
        }
        
        logger.warning(
            f"Unauthorized access attempt: {str(exc)}",
            extra={
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "client_ip": GlobalExceptionHandler._get_client_ip(request)
            }
        )
        
        return JSONResponse(
            status_code=401,
            content=response,
            headers={"X-Error-ID": error_id}
        )
    
    @staticmethod
    def _extract_validation_errors(exc: ValidationError) -> List[Dict[str, Any]]:
        """
        Извлечение ошибок валидации.
        
        Args:
            exc: Исключение валидации
            
        Returns:
            List[Dict[str, Any]]: Список ошибок валидации
        """
        if hasattr(exc, 'errors') and exc.errors:
            return exc.errors
        
        return [{"field": "general", "message": str(exc)}]
    
    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Получение IP адреса клиента.
        
        Args:
            request: HTTP запрос
            
        Returns:
            str: IP адрес клиента
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"


# Создаем глобальный экземпляр обработчика ошибок
error_handler = ErrorHandlerMiddleware()
global_exception_handler = GlobalExceptionHandler()