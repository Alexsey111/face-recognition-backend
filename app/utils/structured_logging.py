"""
Structured JSON Logging System.
Обеспечивает структурированное логирование в формате JSON с поддержкой:
- Лог уровней (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log rotation (макс. размер файла, количество backup файлов)
- Контекстного логирования с дополнительными полями
- Trace request_id через все логи
- Audit trail для всех операций
"""

import logging
import sys
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import lru_cache
from io import StringIO
import threading
import traceback as tb_lib

from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler

from ..config import settings


# ============================================================================
# Log Levels
# ============================================================================

class LogLevel:
    """Константы уровней логирования."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @classmethod
    def from_string(cls, level: str) -> int:
        """Преобразование строки в уровень логирования."""
        return getattr(cls, level.upper(), logging.INFO)


# ============================================================================
# Sensitive Data Redaction
# ============================================================================

# Поля, которые никогда не должны попадать в логи
SENSITIVE_FIELDS = {
    # Authentication
    "password", "token", "access_token", "refresh_token", "api_key", "secret",
    "jwt", "credential", "passcode", "pin", "otp", "two_factor_code",
    # Financial
    "credit_card", "card_number", "cvv", "ssn", "bank_account", "routing_number",
    # Personal
    "social_security", "passport", "driver_license", "national_id",
    # Face/Biometric (NEVER log raw biometric data)
    "embedding", "face_embedding", "biometric_data", "face_image", 
    "face_data", "liveness_image", "reference_image", "image_data_base64",
    # Other sensitive
    "private_key", "encryption_key", "master_key", "hmac_secret",
}

# Паттерны для обнаружения чувствительных данных в значениях
SENSITIVE_PATTERNS = [
    r'\b\d{16}\b',  # Credit card numbers
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
    r'\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b',  # JWT tokens
]


def redact_sensitive_data(data: Any, depth: int = 0, max_depth: int = 5) -> Any:
    """
    Рекурсивное удаление чувствительных данных из структуры.
    
    Args:
        data: Данные для обработки
        depth: Текущая глубина рекурсии
        max_depth: Максимальная глубина рекурсии
        
    Returns:
        Очищенные данные
    """
    if depth > max_depth:
        return data
    
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            key_lower = key.lower()
            
            # Проверяем, является ли ключ чувствительным
            if key_lower in SENSITIVE_FIELDS or any(
                sens in key_lower for sens in ["password", "secret", "key", "token"]
            ):
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_sensitive_data(value, depth + 1, max_depth)
        return result
    
    elif isinstance(data, list):
        return [redact_sensitive_data(item, depth + 1, max_depth) for item in data]
    
    elif isinstance(data, str):
        # Проверяем наличие JWT токенов
        for pattern in SENSITIVE_PATTERNS:
            import re
            data = re.sub(pattern, "[REDACTED]", data)
        return data
    
    return data


# ============================================================================
# Custom JSON Formatter
# ============================================================================

class StructuredJsonFormatter(jsonlogger.JsonFormatter):
    """
    Кастомный JSON форматтер для структурированного логирования.
    
    Добавляет стандартные поля:
    - timestamp: ISO формат с UTC timezone
    - level: Уровень логирования
    - logger: Имя логгера
    - message: Сообщение
    - request_id: ID запроса (если есть)
    - user_id: ID пользователя (если есть)
    """
    
    def __init__(
        self,
        *args,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.request_id = request_id
        self.user_id = user_id
        self.extra_fields = extra_fields or {}
    
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any]
    ):
        super().add_fields(log_record, record, message_dict)
        
        # Timestamp (ISO 8601 UTC)
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"
        
        # Уровень в верхнем регистре
        log_record["level"] = record.levelname
        
        # Имя логгера
        log_record["logger"] = record.name
        
        # Request ID
        if self.request_id:
            log_record["request_id"] = self.request_id
        
        # User ID
        if self.user_id:
            log_record["user_id"] = self.user_id
        
        # Дополнительные поля
        for key, value in self.extra_fields.items():
            if key not in log_record:
                log_record[key] = value
        
        # Добавляем функцию/модуль для отладки
        if record.funcName and record.funcName != "<module>":
            log_record["function"] = record.funcName
            log_record["file"] = f"{record.filename}:{record.lineno}"
        
        # Обработка исключений
        if record.exc_info:
            log_record["exception"] = self.format_exception(record.exc_info)
        
        # Redact sensitive data
        log_record = redact_sensitive_data(log_record)
    
    def format_exception(self, exc_info) -> Optional[Dict[str, Any]]:
        """Форматирование исключения."""
        if not exc_info:
            return None
        
        return {
            "type": exc_info[0].__name__,
            "message": str(exc_info[1]),
            "traceback": "".join(tb_lib.format_exception(*exc_info))
        }


# ============================================================================
# Log Context
# ============================================================================

class LogContext:
    """
    Контекстный менеджер для логирования.
    Позволяет добавлять request_id и user_id ко всем логам в контексте.
    """
    
    _context_local = threading.local()
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.extra = extra or {}
        self.previous_context = None
    
    def __enter__(self):
        """Вход в контекст."""
        self.previous_context = getattr(self._context_local, "context", None)
        self._context_local.context = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекста."""
        if self.previous_context:
            self._context_local.context = self.previous_context
        else:
            self._context_local.context = None
        return False
    
    @classmethod
    def get_current(cls) -> Optional["LogContext"]:
        """Получение текущего контекста."""
        return getattr(cls._context_local, "context", None)
    
    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """Получение request_id из текущего контекста."""
        ctx = cls.get_current()
        return ctx.request_id if ctx else None
    
    @classmethod
    def get_user_id(cls) -> Optional[str]:
        """Получение user_id из текущего контекста."""
        ctx = cls.get_current()
        return ctx.user_id if ctx else None


# ============================================================================
# Logger Factory
# ============================================================================

class LoggerFactory:
    """
    Фабрика для создания настроенных логгеров.
    Поддерживает:
    - Console logging (JSON или text)
    - File logging с rotation
    - Контекстное логирование
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _handlers: Dict[str, list] = {}
    _lock = threading.Lock()
    
    @classmethod
    def setup_logger(
        cls,
        name: str,
        level: Optional[Union[str, int]] = None,
        log_file: Optional[str] = None,
        log_format: str = "json",
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10,
        propagate: bool = False
    ) -> logging.Logger:
        """
        Создание и настройка логгера.
        
        Args:
            name: Имя логгера
            level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Путь к файлу логов (опционально)
            log_format: Формат логов ("json" или "text")
            max_bytes: Максимальный размер файла в байтах
            backup_count: Количество backup файлов
            propagate: Пропагать ли логи родительскому логгеру
            
        Returns:
            Настроенный логгер
        """
        with cls._lock:
            # Проверяем, существует ли уже логгер
            if name in cls._loggers:
                return cls._loggers[name]
            
            # Получаем или создаём логгер
            logger = logging.getLogger(name)
            
            # Очищаем существующие handlers
            logger.handlers.clear()
            
            # Устанавливаем уровень
            log_level = level or settings.LOG_LEVEL
            if isinstance(log_level, str):
                log_level = LogLevel.from_string(log_level)
            logger.setLevel(log_level)
            
            # Получаем контекст
            ctx = LogContext.get_current()
            request_id = ctx.request_id if ctx else None
            user_id = ctx.user_id if ctx else None
            extra = ctx.extra if ctx else {}
            
            # Создаём formatter
            if log_format == "json":
                formatter = StructuredJsonFormatter(
                    request_id=request_id,
                    user_id=user_id,
                    extra=extra
                )
            else:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler с rotation (если указан log_file)
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8"
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            
            logger.propagate = propagate
            cls._loggers[name] = logger
            cls._handlers[name] = logger.handlers
            
            return logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Получение существующего логгера.
        Если логгер не существует, создаёт новый с настройками по умолчанию.
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        return cls.setup_logger(
            name=name,
            level=settings.LOG_LEVEL,
            log_file=settings.LOG_FILE_PATH,
            log_format=settings.LOG_FORMAT,
            max_bytes=settings.LOG_MAX_SIZE,
            backup_count=settings.LOG_BACKUP_COUNT
        )
    
    @classmethod
    def clear_loggers(cls):
        """Очистка всех логгеров (для тестирования)."""
        with cls._lock:
            for name, logger in cls._loggers.items():
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
            cls._loggers.clear()
            cls._handlers.clear()


# ============================================================================
# Main Logger Functions
# ============================================================================

@lru_cache(maxsize=128)
def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера по имени.
    Кэширует результат для производительности.
    
    Args:
        name: Имя логгера (обычно __name__)
        
    Returns:
        Логгер
    """
    return LoggerFactory.get_logger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    duration_ms: Optional[float] = None,
    tags: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Логирование с контекстом.
    
    Args:
        logger: Логгер для записи
        level: Уровень логирования
        message: Сообщение
        request_id: ID запроса
        user_id: ID пользователя
        ip_address: IP адрес клиента
        duration_ms: Время выполнения в миллисекундах
        tags: Теги для классификации
        extra: Дополнительные поля
        **kwargs: Дополнительные поля для лога
    """
    # Получаем контекст, если не передан явно
    if not request_id:
        request_id = LogContext.get_request_id()
    if not user_id:
        user_id = LogContext.get_user_id()
    
    # Формируем дополнительные поля
    log_extra = {
        "request_id": request_id,
        "user_id": user_id,
        "ip_address": ip_address,
        "duration_ms": duration_ms,
        "tags": tags or [],
        **(extra or {}),
        **kwargs
    }
    
    # Удаляем None значения
    log_extra = {k: v for k, v in log_extra.items() if v is not None}
    
    logger.log(level, message, extra=log_extra)


def log_debug(
    logger: logging.Logger,
    message: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> None:
    """Логирование на уровне DEBUG."""
    log_with_context(logger, logging.DEBUG, message, request_id, user_id, **kwargs)


def log_info(
    logger: logging.Logger,
    message: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> None:
    """Логирование на уровне INFO."""
    log_with_context(logger, logging.INFO, message, request_id, user_id, **kwargs)


def log_warning(
    logger: logging.Logger,
    message: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> None:
    """Логирование на уровне WARNING."""
    log_with_context(logger, logging.WARNING, message, request_id, user_id, **kwargs)


def log_error(
    logger: logging.Logger,
    message: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    error: Optional[Exception] = None,
    error_type: Optional[str] = None,
    **kwargs
) -> None:
    """Логирование на уровне ERROR."""
    extra = kwargs.pop("extra", {})
    
    if error:
        extra["error"] = str(error)
        if hasattr(error, "__class__"):
            extra["error_type"] = error.__class__.__name__
        else:
            extra["error_type"] = error_type or "UnknownError"
    
    log_with_context(logger, logging.ERROR, message, request_id, user_id, extra=extra, **kwargs)


def log_critical(
    logger: logging.Logger,
    message: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    error: Optional[Exception] = None,
    **kwargs
) -> None:
    """Логирование на уровне CRITICAL."""
    extra = kwargs.pop("extra", {})
    
    if error:
        extra["error"] = str(error)
        extra["error_type"] = error.__class__.__name__ if hasattr(error, "__class__") else "UnknownError"
    
    log_with_context(logger, logging.CRITICAL, message, request_id, user_id, extra=extra, **kwargs)


# ============================================================================
# Audit Logger
# ============================================================================

class AuditLogger:
    """
    Специализированный логгер для аудита операций.
    Записывает все важные действия пользователей и системы.
    """
    
    def __init__(self, name: str = "audit"):
        self.logger = get_logger(name)
        self.enabled = settings.AUDIT_LOG_ENABLED
    
    def log(
        self,
        action: str,
        user_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Запись аудит лога.
        
        Args:
            action: Действие (login, logout, verification, create, update, delete, etc.)
            user_id: ID пользователя, совершающего действие
            operator_id: ID оператора (для admin действий)
            resource_type: Тип ресурса (user, reference, session, etc.)
            resource_id: ID ресурса
            old_values: Предыдущие значения (для update/delete)
            new_values: Новые значения (для create/update)
            ip_address: IP адрес клиента
            user_agent: User Agent браузера/клиента
            success: Успешно ли выполнено действие
            error_message: Сообщение об ошибке (если есть)
            request_id: ID запроса
            extra: Дополнительные поля
        """
        if not self.enabled:
            return
        
        # Формируем payload
        audit_entry = {
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "old_values": old_values,
            "new_values": new_values,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "success": success,
            "error_message": error_message,
            "operator_id": operator_id,
            "audit_type": "operation",  # operation, security, system
            **(extra or {}),
            **kwargs
        }
        
        # Удаляем None значения
        audit_entry = {k: v for k, v in audit_entry.items() if v is not None}
        
        # Redact sensitive data
        audit_entry = redact_sensitive_data(audit_entry)
        
        # Выбираем уровень логирования
        if not success:
            level = logging.WARNING
            message = f"Audit: {action} failed"
        else:
            level = logging.INFO
            message = f"Audit: {action} completed"
        
        log_with_context(
            self.logger,
            level,
            message,
            request_id=request_id,
            user_id=user_id,
            extra=audit_entry
        )
    
    # Convenience methods
    def auth_success(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование успешной аутентификации."""
        self.log(
            action="user_login",
            user_id=user_id,
            resource_type="user",
            resource_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            request_id=request_id,
            **kwargs
        )
    
    def auth_failed(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование неудачной аутентификации."""
        self.log(
            action="user_login_failed",
            user_id=user_id,
            resource_type="user",
            resource_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            error_message=error_message,
            request_id=request_id,
            **kwargs
        )
    
    def verification_started(
        self,
        user_id: str,
        session_id: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование начала верификации."""
        self.log(
            action="verification_started",
            user_id=user_id,
            resource_type="verification_session",
            resource_id=session_id,
            ip_address=ip_address,
            request_id=request_id,
            **kwargs
        )
    
    def verification_completed(
        self,
        user_id: str,
        session_id: str,
        is_match: bool,
        similarity_score: Optional[float] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование завершения верификации."""
        self.log(
            action="verification_completed",
            user_id=user_id,
            resource_type="verification_session",
            resource_id=session_id,
            new_values={"is_match": is_match, "similarity_score": similarity_score},
            ip_address=ip_address,
            success=is_match,
            request_id=request_id,
            **kwargs
        )
    
    def reference_created(
        self,
        user_id: str,
        reference_id: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование создания эталонного изображения."""
        self.log(
            action="reference_created",
            user_id=user_id,
            resource_type="reference",
            resource_id=reference_id,
            ip_address=ip_address,
            request_id=request_id,
            **kwargs
        )
    
    def reference_deleted(
        self,
        user_id: str,
        reference_id: str,
        old_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        operator_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование удаления эталонного изображения."""
        self.log(
            action="reference_deleted",
            user_id=user_id,
            operator_id=operator_id,
            resource_type="reference",
            resource_id=reference_id,
            old_values=old_values,
            ip_address=ip_address,
            request_id=request_id,
            **kwargs
        )
    
    def file_uploaded(
        self,
        user_id: str,
        file_name: str,
        file_size: int,
        file_type: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование загрузки файла."""
        self.log(
            action="file_uploaded",
            user_id=user_id,
            resource_type="file",
            resource_id=file_name,
            new_values={"file_name": file_name, "file_size": file_size, "file_type": file_type},
            ip_address=ip_address,
            request_id=request_id,
            **kwargs
        )
    
    def admin_action(
        self,
        operator_id: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование действий администратора."""
        self.log(
            action=action,
            user_id=operator_id,
            operator_id=operator_id,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            ip_address=ip_address,
            success=success,
            error_message=error_message,
            request_id=request_id,
            extra={"audit_type": "admin"},
            **kwargs
        )
    
    def liveness_check(
        self,
        user_id: str,
        session_id: str,
        passed: bool,
        score: Optional[float] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование проверки liveness."""
        self.log(
            action="liveness_check",
            user_id=user_id,
            resource_type="verification_session",
            resource_id=session_id,
            new_values={"passed": passed, "score": score},
            ip_address=ip_address,
            success=passed,
            request_id=request_id,
            **kwargs
        )
    
    def account_changed(
        self,
        user_id: str,
        changes: Dict[str, Any],
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование изменений аккаунта."""
        self.log(
            action="account_changed",
            user_id=user_id,
            resource_type="user",
            resource_id=user_id,
            old_values=changes.get("old"),
            new_values=changes.get("new"),
            ip_address=ip_address,
            request_id=request_id,
            **kwargs
        )
    
    def api_error(
        self,
        user_id: Optional[str] = None,
        error_type: str = "UnknownError",
        error_message: Optional[str] = None,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование API ошибки."""
        self.log(
            action="api_error",
            user_id=user_id,
            resource_type="api",
            resource_id=endpoint,
            new_values={"error_type": error_type, "error_message": error_message},
            ip_address=ip_address,
            success=False,
            error_message=error_message,
            request_id=request_id,
            **kwargs
        )


# Функция для получения audit логгера
@lru_cache(maxsize=1)
def get_audit_logger() -> AuditLogger:
    """Получение audit логгера."""
    return AuditLogger()


# ============================================================================
# Log Entry Builder
# ============================================================================

class LogEntry:
    """
    Builder для создания структурированных лог записей.
    Позволяет создавать лог записи с цепочкой вызовов.
    """
    
    def __init__(self, message: str, level: int = logging.INFO):
        self.message = message
        self.level = level
        self.request_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.ip_address: Optional[str] = None
        self.duration_ms: Optional[float] = None
        self.tags: list = []
        self.extra: Dict[str, Any] = {}
        self.logger: Optional[logging.Logger] = None
    
    def with_request_id(self, request_id: str) -> "LogEntry":
        """Установка request_id."""
        self.request_id = request_id
        return self
    
    def with_user_id(self, user_id: str) -> "LogEntry":
        """Установка user_id."""
        self.user_id = user_id
        return self
    
    def with_ip_address(self, ip_address: str) -> "LogEntry":
        """Установка ip_address."""
        self.ip_address = ip_address
        return self
    
    def with_duration(self, duration_ms: float) -> "LogEntry":
        """Установка duration_ms."""
        self.duration_ms = duration_ms
        return self
    
    def with_tags(self, tags: list) -> "LogEntry":
        """Добавление тегов."""
        self.tags.extend(tags)
        return self
    
    def with_extra(self, **kwargs) -> "LogEntry":
        """Добавление дополнительных полей."""
        self.extra.update(kwargs)
        return self
    
    def with_logger(self, logger: logging.Logger) -> "LogEntry":
        """Установка логгера."""
        self.logger = logger
        return self
    
    def debug(self) -> None:
        """Логирование на уровне DEBUG."""
        self.level = logging.DEBUG
        self._log()
    
    def info(self) -> None:
        """Логирование на уровне INFO."""
        self.level = logging.INFO
        self._log()
    
    def warning(self) -> None:
        """Логирование на уровне WARNING."""
        self.level = logging.WARNING
        self._log()
    
    def error(self, error: Optional[Exception] = None) -> None:
        """Логирование на уровне ERROR."""
        self.level = logging.ERROR
        if error:
            self.extra["error"] = str(error)
            self.extra["error_type"] = error.__class__.__name__
        self._log()
    
    def critical(self, error: Optional[Exception] = None) -> None:
        """Логирование на уровне CRITICAL."""
        self.level = logging.CRITICAL
        if error:
            self.extra["error"] = str(error)
            self.extra["error_type"] = error.__class__.__name__
        self._log()
    
    def _log(self) -> None:
        """Выполнение логирования."""
        if not self.logger:
            self.logger = get_logger(__name__)
        
        log_with_context(
            self.logger,
            self.level,
            self.message,
            request_id=self.request_id,
            user_id=self.user_id,
            ip_address=self.ip_address,
            duration_ms=self.duration_ms,
            tags=self.tags,
            extra=self.extra
        )


def create_log(message: str, level: int = logging.INFO) -> LogEntry:
    """Создание новой лог записи."""
    return LogEntry(message, level)
