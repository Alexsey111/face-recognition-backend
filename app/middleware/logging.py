"""
Middleware для логирования запросов.
Запись информации о HTTP запросах и ответах без биометрических данных.
"""

from typing import Optional, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import time
import json
import uuid
from datetime import datetime, timezone
import traceback
import asyncio

from ..config import settings
from ..utils.logger import get_logger
from ..services.database_service import DatabaseService

logger = get_logger(__name__)


class LoggingMiddleware:
    """
    Middleware для логирования HTTP запросов и ответов.
    """
    
    def __init__(self):
        self.db_service = DatabaseService()
        self.log_requests = True
        self.log_responses = True
        self.log_request_body = False  # По соображениям безопасности
        self.log_response_body = False  # По соображениям безопасности
        
        # Пути, которые не нужно логировать (избегаем спама)
        self.skip_paths = [
            "/metrics",
            "/health",
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/favicon.ico"
        ]
        
        # Пути, где нужно минимальное логирование
        self.minimal_log_paths = [
            "/metrics",
            "/health",
            "/status"
        ]
    
    async def __call__(self, request: Request, call_next):
        """
        Обработка запроса с логированием.
        
        Args:
            request: HTTP запрос
            call_next: Следующий обработчик
            
        Returns:
            Response: HTTP ответ
        """
        # Генерируем уникальный ID для запроса
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Проверяем, нужно ли логировать этот запрос
        if self._should_skip_logging(request):
            return await call_next(request)
        
        start_time = time.time()
        
        # Подготавливаем данные для логирования
        log_data = await self._prepare_request_log_data(request, request_id)
        
        try:
            # Логируем начало обработки запроса
            logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "event_type": "request_started",
                    **log_data
                }
            )
            
            # Выполняем запрос
            response = await call_next(request)
            
            # Подготавливаем данные ответа
            response_log_data = await self._prepare_response_log_data(response, start_time)
            
            # Объединяем данные запроса и ответа
            full_log_data = {**log_data, **response_log_data}
            
            # Логируем завершение запроса
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "event_type": "request_completed",
                    **full_log_data
                }
            )
            
            # Сохраняем лог в базу данных асинхронно
            asyncio.create_task(self._save_audit_log(request, response, full_log_data))
            
            # Добавляем request_id в заголовки ответа
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Логируем ошибку
            error_log_data = await self._prepare_error_log_data(request, e, start_time)
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "request_id": request_id,
                    "event_type": "request_error",
                    **log_data,
                    **error_log_data
                },
                exc_info=True
            )
            
            # Сохраняем ошибку в базу данных
            asyncio.create_task(self._save_error_log(request, e, {**log_data, **error_log_data}))
            
            # Перебрасываем исключение
            raise
    
    def _should_skip_logging(self, request: Request) -> bool:
        """
        Проверка, нужно ли пропустить логирование запроса.
        
        Args:
            request: HTTP запрос
            
        Returns:
            bool: True если логирование не нужно
        """
        path = request.url.path
        
        # Пропускаем пути из списка
        for skip_path in self.skip_paths:
            if path.startswith(skip_path):
                return True
        
        return False
    
    def _should_minimal_logging(self, request: Request) -> bool:
        """
        Проверка, нужно ли минимальное логирование.
        
        Args:
            request: HTTP запрос
            
        Returns:
            bool: True если нужно минимальное логирование
        """
        path = request.url.path
        
        for minimal_path in self.minimal_log_paths:
            if path.startswith(minimal_path):
                return True
        
        return False
    
    async def _prepare_request_log_data(self, request: Request, request_id: str) -> Dict[str, Any]:
        """
        Подготовка данных для логирования запроса.
        
        Args:
            request: HTTP запрос
            request_id: Уникальный ID запроса
            
        Returns:
            Dict[str, Any]: Данные для логирования
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Добавляем информацию о пользователе (если есть)
        if hasattr(request.state, 'user_id'):
            log_data["user_id"] = request.state.user_id
        
        if hasattr(request.state, 'user_role'):
            log_data["user_role"] = request.state.user_role
        
        # Добавляем заголовки (без конфиденциальных)
        headers = {}
        for key, value in request.headers.items():
            if self._is_safe_header(key):
                headers[key] = value
        
        if headers:
            log_data["headers"] = headers
        
        # Логируем тело запроса только для определенных методов и путей
        if (self.log_request_body and 
            request.method in ["POST", "PUT", "PATCH"] and
            not self._should_minimal_logging(request)):
            
            try:
                # Читаем тело запроса (только первый раз)
                body = await request.body()
                if body:
                    # Логируем только размер и тип контента
                    log_data["body_size"] = len(body)
                    log_data["content_type"] = request.headers.get("Content-Type", "")
                    
                    # Для JSON можно логировать структуру без данных
                    if "application/json" in log_data.get("content_type", ""):
                        try:
                            # Парсим JSON и логируем только ключи
                            body_json = json.loads(body.decode('utf-8'))
                            log_data["body_keys"] = list(body_json.keys()) if isinstance(body_json, dict) else "array"
                        except:
                            log_data["body_keys"] = "invalid_json"
            except Exception:
                # Если не удалось прочитать тело, пропускаем
                pass
        
        return log_data
    
    async def _prepare_response_log_data(self, response: Response, start_time: float) -> Dict[str, Any]:
        """
        Подготовка данных для логирования ответа.
        
        Args:
            response: HTTP ответ
            start_time: Время начала обработки
            
        Returns:
            Dict[str, Any]: Данные для логирования ответа
        """
        end_time = time.time()
        processing_time = end_time - start_time
        
        log_data = {
            "status_code": response.status_code,
            "processing_time": round(processing_time, 3),
            "response_size": self._get_response_size(response),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Добавляем информацию о rate limiting
        if "X-RateLimit-Remaining" in response.headers:
            log_data["rate_limit_remaining"] = response.headers["X-RateLimit-Remaining"]
        
        # Логируем тело ответа только в debug режиме и для ошибок
        if (self.log_response_body and 
            (response.status_code >= 400 or settings.DEBUG)):
            
            try:
                # Получаем тело ответа (это может быть сложно для stream ответов)
                if hasattr(response, 'body'):
                    body = response.body
                    if body:
                        log_data["response_body_size"] = len(body)
                        
                        # Для JSON логируем только структуру
                        content_type = response.headers.get("Content-Type", "")
                        if "application/json" in content_type:
                            try:
                                body_json = json.loads(body.decode('utf-8'))
                                log_data["response_body_keys"] = list(body_json.keys()) if isinstance(body_json, dict) else "array"
                            except:
                                log_data["response_body_keys"] = "invalid_json"
            except Exception:
                # Если не удалось получить тело, пропускаем
                pass
        
        return log_data
    
    async def _prepare_error_log_data(self, request: Request, error: Exception, start_time: float) -> Dict[str, Any]:
        """
        Подготовка данных для логирования ошибки.
        
        Args:
            request: HTTP запрос
            error: Исключение
            start_time: Время начала обработки
            
        Returns:
            Dict[str, Any]: Данные для логирования ошибки
        """
        end_time = time.time()
        processing_time = end_time - start_time
        
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "processing_time": round(processing_time, 3),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Добавляем stack trace только в debug режиме
        if settings.DEBUG:
            log_data["stack_trace"] = traceback.format_exc()
        
        return log_data
    
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
    
    def _is_safe_header(self, header_name: str) -> bool:
        """
        Проверка, является ли заголовок безопасным для логирования.
        
        Args:
            header_name: Название заголовка
            
        Returns:
            bool: True если заголовок безопасен
        """
        unsafe_headers = [
            "authorization",
            "cookie",
            "x-api-key",
            "x-auth-token",
            "proxy-authorization"
        ]
        
        return header_name.lower() not in unsafe_headers
    
    def _get_response_size(self, response: Response) -> Optional[int]:
        """
        Получение размера ответа.
        
        Args:
            response: HTTP ответ
            
        Returns:
            Optional[int]: Размер ответа в байтах
        """
        try:
            # Получаем Content-Length заголовок
            content_length = response.headers.get("Content-Length")
            if content_length:
                return int(content_length)
            
            # Если Content-Length нет, пытаемся получить размер тела
            if hasattr(response, 'body') and response.body:
                return len(response.body)
            
        except Exception:
            pass
        
        return None
    
    async def _save_audit_log(self, request: Request, response: Response, log_data: Dict[str, Any]):
        """
        Сохранение лога аудита в базу данных.
        
        Args:
            request: HTTP запрос
            response: HTTP ответ
            log_data: Данные для сохранения
        """
        try:
            # Создаем запись в журнале аудита
            audit_data = {
                "user_id": log_data.get("user_id"),
                "action": f"{request.method.lower()}_{request.url.path.replace('/', '_').strip('_')}",
                "resource_type": "http_request",
                "resource_id": log_data.get("request_id"),
                "description": f"{request.method} {request.url.path} - {response.status_code}",
                "request_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": log_data.get("query_params", {}),
                    "client_ip": log_data.get("client_ip"),
                    "user_agent": log_data.get("user_agent")
                },
                "response_data": {
                    "status_code": response.status_code,
                    "processing_time": log_data.get("processing_time"),
                    "response_size": log_data.get("response_size")
                },
                "ip_address": log_data.get("client_ip"),
                "user_agent": log_data.get("user_agent")
            }
            
            # Сохраняем в базу данных
            await self.db_service.create_audit_log(audit_data)
            
        except Exception as e:
            logger.error(f"Failed to save audit log: {str(e)}")
    
    async def _save_error_log(self, request: Request, error: Exception, log_data: Dict[str, Any]):
        """
        Сохранение лога ошибки в базу данных.
        
        Args:
            request: HTTP запрос
            error: Исключение
            log_data: Данные для сохранения
        """
        try:
            # Создаем запись об ошибке
            error_data = {
                "user_id": log_data.get("user_id"),
                "action": f"error_{request.method.lower()}_{request.url.path.replace('/', '_').strip('_')}",
                "resource_type": "http_request",
                "resource_id": log_data.get("request_id"),
                "description": f"Error in {request.method} {request.url.path}: {str(error)}",
                "old_values": {
                    "error_type": log_data.get("error_type"),
                    "error_message": log_data.get("error_message")
                },
                "ip_address": log_data.get("client_ip"),
                "user_agent": log_data.get("user_agent")
            }
            
            # Сохраняем в базу данных
            await self.db_service.create_audit_log(error_data)
            
        except Exception as e:
            logger.error(f"Failed to save error log: {str(e)}")


class RequestLogger:
    """
    Утилита для логирования специфичных событий.
    """
    
    @staticmethod
    def log_user_action(user_id: str, action: str, details: Dict[str, Any] = None):
        """
        Логирование действий пользователя.
        
        Args:
            user_id: ID пользователя
            action: Действие
            details: Дополнительные детали
        """
        logger.info(
            f"User action: {action}",
            extra={
                "event_type": "user_action",
                "user_id": user_id,
                "action": action,
                "details": details or {}
            }
        )
    
    @staticmethod
    def log_verification_request(user_id: str, session_id: str, verification_type: str, success: bool):
        """
        Логирование запроса верификации.
        
        Args:
            user_id: ID пользователя
            session_id: ID сессии
            verification_type: Тип верификации
            success: Результат
        """
        logger.info(
            f"Verification request: {verification_type} - {'success' if success else 'failure'}",
            extra={
                "event_type": "verification_request",
                "user_id": user_id,
                "session_id": session_id,
                "verification_type": verification_type,
                "success": success
            }
        )
    
    @staticmethod
    def log_security_event(event_type: str, details: Dict[str, Any]):
        """
        Логирование событий безопасности.
        
        Args:
            event_type: Тип события безопасности
            details: Детали события
        """
        logger.warning(
            f"Security event: {event_type}",
            extra={
                "event_type": "security_event",
                "security_event_type": event_type,
                "details": details
            }
        )
    
    @staticmethod
    def log_performance_metric(metric_name: str, value: float, tags: Dict[str, str] = None):
        """
        Логирование метрик производительности.
        
        Args:
            metric_name: Название метрики
            value: Значение
            tags: Теги для метрики
        """
        logger.info(
            f"Performance metric: {metric_name} = {value}",
            extra={
                "event_type": "performance_metric",
                "metric_name": metric_name,
                "metric_value": value,
                "metric_tags": tags or {}
            }
        )


# Создаем глобальный экземпляр логгера
request_logger = RequestLogger()