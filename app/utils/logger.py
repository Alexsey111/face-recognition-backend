"""
Настройка логгера.
Конфигурация системы логирования для приложения.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import uuid
from datetime import datetime

from ..config import settings
from .constants import LOG_LEVELS, TIME_FORMATS

# Глобальная переменная для хранения логгера
_logger = None


class ColoredFormatter(logging.Formatter):
    """
    Цветной форматтер для консольного вывода.
    """
    
    # Цветовые коды для разных уровней логирования
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Добавляем цвет к уровню логирования
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """
    JSON форматтер для структурированного логирования.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Базовые данные
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Добавляем request_id если есть
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        # Добавляем user_id если есть
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        
        # Добавляем event_type если есть
        if hasattr(record, 'event_type'):
            log_data["event_type"] = record.event_type
        
        # Добавляем дополнительные данные из extra
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Добавляем исключение если есть
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class RequestContextFilter(logging.Filter):
    """
    Фильтр для добавления контекста запроса в логи.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Инициализируем атрибуты
        if not hasattr(record, 'request_id'):
            record.request_id = None
        if not hasattr(record, 'user_id'):
            record.user_id = None
        if not hasattr(record, 'event_type'):
            record.event_type = None
        
        return True


def setup_logger(
    name: str = "face_recognition_service",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
    json_format: bool = False
) -> logging.Logger:
    """
    Настройка системы логирования.
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу логов
        max_file_size: Максимальный размер файла логов
        backup_count: Количество backup файлов
        json_format: Использовать ли JSON формат
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    global _logger
    
    # Используем уровень из настроек если не указан
    if level is None:
        level = settings.LOG_LEVEL.upper()
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level, logging.INFO))
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Создаем форматтер
    if json_format:
        formatter = JSONFormatter()
    else:
        # Определяем формат сообщения
        if settings.DEBUG:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[%(filename)s:%(lineno)d] - %(message)s"
            )
        else:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        formatter = logging.Formatter(format_string)
    
    # Добавляем консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVELS.get(level, logging.INFO))
    
    if settings.DEBUG and not json_format:
        # В debug режиме используем цветной форматтер для консоли
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = formatter
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Добавляем файловый обработчик если указан файл
    if log_file or settings.LOG_FILE_PATH:
        log_file_path = log_file or settings.LOG_FILE_PATH
        
        # Создаем директорию если не существует
        log_dir = Path(log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Ротирующий файловый обработчик
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(LOG_LEVELS.get(level, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Добавляем фильтр контекста
    context_filter = RequestContextFilter()
    logger.addFilter(context_filter)
    
    # Предотвращаем дублирование логов
    logger.propagate = False
    
    # Сохраняем логгер в глобальной переменной
    _logger = logger
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Получение логгера.
    
    Args:
        name: Имя логгера
        
    Returns:
        logging.Logger: Логгер
    """
    global _logger
    
    if _logger is None:
        # Создаем логгер если еще не создан
        _logger = setup_logger()
    
    if name:
        return logging.getLogger(name)
    return _logger


class LoggerMixin:
    """
    Миксин для добавления логгера в классы.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Логгер для класса."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


class StructuredLogger:
    """
    Структурированный логгер для удобного логирования событий.
    """
    
    def __init__(self, name: str = None):
        self.logger = get_logger(name)
    
    def log_event(
        self,
        event_type: str,
        message: str,
        level: str = "INFO",
        **kwargs
    ):
        """
        Логирование события с дополнительными данными.
        
        Args:
            event_type: Тип события
            message: Сообщение
            level: Уровень логирования
            **kwargs: Дополнительные данные
        """
        extra = {
            "event_type": event_type,
            **kwargs
        }
        
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, message, extra=extra)
    
    def log_user_action(self, user_id: str, action: str, **kwargs):
        """Логирование действий пользователя."""
        self.log_event(
            event_type="user_action",
            message=f"User action: {action}",
            user_id=user_id,
            action=action,
            **kwargs
        )
    
    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time: float,
        user_id: str = None,
        request_id: str = None,
        **kwargs
    ):
        """Логирование API запросов."""
        self.log_event(
            event_type="api_request",
            message=f"{method} {path} - {status_code}",
            method=method,
            path=path,
            status_code=status_code,
            response_time=response_time,
            user_id=user_id,
            request_id=request_id,
            **kwargs
        )
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        user_id: str = None,
        request_id: str = None
    ):
        """Логирование ошибок."""
        self.logger.error(
            f"Error: {str(error)}",
            extra={
                "event_type": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "user_id": user_id,
                "request_id": request_id,
            },
            exc_info=True
        )
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        **kwargs
    ):
        """Логирование метрик производительности."""
        self.log_event(
            event_type="performance",
            message=f"Operation {operation} completed in {duration:.3f}s",
            operation=operation,
            duration=duration,
            success=success,
            **kwargs
        )
    
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = "WARNING",
        **kwargs
    ):
        """Логирование событий безопасности."""
        self.log_event(
            event_type="security_event",
            message=f"Security event: {description}",
            security_event_type=event_type,
            description=description,
            severity=severity,
            **kwargs
        )


# Создаем глобальный структурированный логгер
structured_logger = StructuredLogger()


def log_function_call(func):
    """
    Декоратор для логирования вызовов функций.
    
    Args:
        func: Функция для декорирования
        
    Returns:
        Декорированная функция
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.utcnow()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"Function {func.__name__} executed successfully",
                extra={
                    "function": func.__name__,
                    "duration": duration,
                    "success": True
                }
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.error(
                f"Function {func.__name__} failed: {str(e)}",
                extra={
                    "function": func.__name__,
                    "duration": duration,
                    "success": False,
                    "error": str(e)
                },
                exc_info=True
            )
            
            raise
    
    return wrapper


def configure_logging_for_environment():
    """
    Настройка логирования в зависимости от окружения.
    """
    if settings.DEBUG:
        # В debug режиме более подробное логирование
        setup_logger(
            level="DEBUG",
            json_format=False
        )
    elif settings.ENVIRONMENT == "production":
        # В production режиме логируем в JSON формате
        setup_logger(
            level="INFO",
            json_format=True,
            log_file=settings.LOG_FILE_PATH
        )
    else:
        # В других режимах стандартное логирование
        setup_logger(
            level="INFO",
            json_format=False
        )


# Автоматическая настройка при импорте модуля
configure_logging_for_environment()