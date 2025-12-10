"""
Декораторы.
Пользовательские декораторы для функций и методов.
"""

import asyncio
import functools
import time
from typing import Callable, Any, Optional, Dict, List, Union
from datetime import datetime, timezone
import inspect

from .logger import get_logger
from .exceptions import ValidationError, ProcessingError
from .validators import validate_json_schema

logger = get_logger(__name__)


def validate_input(schema: Optional[Dict[str, Any]] = None):
    """
    Декоратор для валидации входных данных.
    
    Args:
        schema: JSON схема для валидации
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if schema:
                # Валидируем kwargs (ключевые аргументы)
                validate_json_schema(kwargs, schema)
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if schema:
                # Валидируем kwargs (ключевые аргументы)
                validate_json_schema(kwargs, schema)
            
            return func(*args, **kwargs)
        
        # Определяем, является ли функция асинхронной
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_request(logger_name: Optional[str] = None):
    """
    Декоратор для логирования запросов.
    
    Args:
        logger_name: Имя логгера
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        decorator_logger = get_logger(logger_name or func.__module__)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = str(datetime.utcnow().timestamp() * 1000000)[:16]  # Простой ID
            
            # Логируем начало выполнения
            decorator_logger.info(
                f"Starting {func.__name__}",
                extra={
                    "request_id": request_id,
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Логируем успешное завершение
                decorator_logger.info(
                    f"Completed {func.__name__} successfully",
                    extra={
                        "request_id": request_id,
                        "function": func.__name__,
                        "duration": duration,
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Логируем ошибку
                decorator_logger.error(
                    f"Failed {func.__name__}: {str(e)}",
                    extra={
                        "request_id": request_id,
                        "function": func.__name__,
                        "duration": duration,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = str(datetime.utcnow().timestamp() * 1000000)[:16]
            
            decorator_logger.info(
                f"Starting {func.__name__}",
                extra={
                    "request_id": request_id,
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                decorator_logger.info(
                    f"Completed {func.__name__} successfully",
                    extra={
                        "request_id": request_id,
                        "function": func.__name__,
                        "duration": duration,
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                decorator_logger.error(
                    f"Failed {func.__name__}: {str(e)}",
                    extra={
                        "request_id": request_id,
                        "function": func.__name__,
                        "duration": duration,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
                
                raise
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def measure_time(threshold: Optional[float] = None, logger_name: Optional[str] = None):
    """
    Декоратор для измерения времени выполнения функции.
    
    Args:
        threshold: Порог в секундах для предупреждений
        logger_name: Имя логгера
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        decorator_logger = get_logger(logger_name or func.__module__)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = await func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            # Логируем время выполнения
            if threshold and duration > threshold:
                decorator_logger.warning(
                    f"Slow operation detected: {func.__name__} took {duration:.3f}s (threshold: {threshold}s)",
                    extra={
                        "function": func.__name__,
                        "duration": duration,
                        "threshold": threshold,
                        "slow": True
                    }
                )
            else:
                decorator_logger.debug(
                    f"Operation completed: {func.__name__} took {duration:.3f}s",
                    extra={
                        "function": func.__name__,
                        "duration": duration
                    }
                )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            if threshold and duration > threshold:
                decorator_logger.warning(
                    f"Slow operation detected: {func.__name__} took {duration:.3f}s (threshold: {threshold}s)",
                    extra={
                        "function": func.__name__,
                        "duration": duration,
                        "threshold": threshold,
                        "slow": True
                    }
                )
            else:
                decorator_logger.debug(
                    f"Operation completed: {func.__name__} took {duration:.3f}s",
                    extra={
                        "function": func.__name__,
                        "duration": duration
                    }
                )
            
            return result
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger_name: Optional[str] = None
):
    """
    Декоратор для повторного выполнения функции при ошибках.
    
    Args:
        max_retries: Максимальное количество попыток
        delay: Задержка между попытками в секундах
        backoff: Множитель для увеличения задержки
        exceptions: Кортеж исключений для повтора
        logger_name: Имя логгера
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        decorator_logger = get_logger(logger_name or func.__module__)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        decorator_logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "attempts": attempt + 1,
                                "max_retries": max_retries,
                                "final_error": str(e)
                            },
                            exc_info=True
                        )
                        break
                    
                    wait_time = delay * (backoff ** attempt)
                    decorator_logger.warning(
                        f"Function {func.__name__} failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {str(e)}",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "wait_time": wait_time,
                            "error": str(e)
                        }
                    )
                    
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        decorator_logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "attempts": attempt + 1,
                                "max_retries": max_retries,
                                "final_error": str(e)
                            },
                            exc_info=True
                        )
                        break
                    
                    wait_time = delay * (backoff ** attempt)
                    decorator_logger.warning(
                        f"Function {func.__name__} failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {str(e)}",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "wait_time": wait_time,
                            "error": str(e)
                        }
                    )
                    
                    time.sleep(wait_time)
            
            raise last_exception
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_result(ttl: int = 300, key_generator: Optional[Callable] = None):
    """
    Декоратор для кэширования результатов функции.
    
    Args:
        ttl: Время жизни кэша в секундах
        key_generator: Функция для генерации ключа кэша
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        # Простая реализация в памяти (в продакшене лучше использовать Redis)
        cache = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Генерируем ключ кэша
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Проверяем кэш
            if cache_key in cache:
                cached_result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return cached_result
            
            # Выполняем функцию и кэшируем результат
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            if cache_key in cache:
                cached_result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return cached_result
            
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            return result
        
        # Добавляем метод для очистки кэша
        def clear_cache():
            cache.clear()
        
        async_wrapper.clear_cache = clear_cache
        sync_wrapper.clear_cache = clear_cache
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def require_auth(required_role: Optional[str] = None):
    """
    Декоратор для проверки аутентификации.
    
    Args:
        required_role: Требуемая роль пользователя
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Ищем request объект в аргументах
            request = None
            for arg in args:
                if hasattr(arg, 'state') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            if not request:
                raise ProcessingError("Request object not found")
            
            # Проверяем аутентификацию
            user_id = getattr(request.state, 'user_id', None)
            if not user_id:
                from .exceptions import UnauthorizedError
                raise UnauthorizedError("Authentication required")
            
            # Проверяем роль если требуется
            if required_role:
                user_role = getattr(request.state, 'user_role', None)
                if user_role != required_role:
                    from .exceptions import ForbiddenError
                    raise ForbiddenError(f"Role '{required_role}' required")
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if hasattr(arg, 'state') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            if not request:
                raise ProcessingError("Request object not found")
            
            user_id = getattr(request.state, 'user_id', None)
            if not user_id:
                from .exceptions import UnauthorizedError
                raise UnauthorizedError("Authentication required")
            
            if required_role:
                user_role = getattr(request.state, 'user_role', None)
                if user_role != required_role:
                    from .exceptions import ForbiddenError
                    raise ForbiddenError(f"Role '{required_role}' required")
            
            return func(*args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def rate_limit(requests_per_minute: int = 60):
    """
    Декоратор для ограничения скорости вызовов функции.
    
    Args:
        requests_per_minute: Максимальное количество вызовов в минуту
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        # Простая реализация в памяти
        call_times = []
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            now = time.time()
            minute_ago = now - 60
            
            # Очищаем старые вызовы
            global call_times
            call_times[:] = [t for t in call_times if t > minute_ago]
            
            # Проверяем лимит
            if len(call_times) >= requests_per_minute:
                from .exceptions import RateLimitError
                raise RateLimitError(
                    f"Rate limit exceeded: {requests_per_minute} requests per minute",
                    limit=requests_per_minute
                )
            
            # Добавляем текущий вызов
            call_times.append(now)
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            now = time.time()
            minute_ago = now - 60
            
            global call_times
            call_times[:] = [t for t in call_times if t > minute_ago]
            
            if len(call_times) >= requests_per_minute:
                from .exceptions import RateLimitError
                raise RateLimitError(
                    f"Rate limit exceeded: {requests_per_minute} requests per minute",
                    limit=requests_per_minute
                )
            
            call_times.append(now)
            
            return func(*args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def deprecated(reason: str = "This function is deprecated"):
    """
    Декоратор для отметки функции как устаревшей.
    
    Args:
        reason: Причина устаревания
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.warning(
                f"Deprecated function {func.__name__} called: {reason}",
                extra={
                    "function": func.__name__,
                    "deprecated": True,
                    "reason": reason
                }
            )
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def memoize(ttl: int = 300):
    """
    Декоратор для мемоизации результатов функции.
    
    Args:
        ttl: Время жизни кэша в секундах
        
    Returns:
        Декоратор функции
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Создаем ключ из аргументов
            key = str(args) + str(sorted(kwargs.items()))
            
            # Проверяем кэш
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Выполняем функцию
            result = await func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        # Добавляем метод для очистки кэша
        def clear_cache():
            cache.clear()
        
        async_wrapper.clear_cache = clear_cache
        sync_wrapper.clear_cache = clear_cache
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator