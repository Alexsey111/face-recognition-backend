"""
Кастомные исключения.
Простые классы исключений для различных типов ошибок.
"""

from typing import Optional, Dict, Any, List


class BaseAppException(Exception):
    """
    Базовый класс для всех исключений приложения.
    """

    def __init__(
        self,
        message: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
        status_code: int = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование исключения в словарь.

        Returns:
            Dict[str, Any]: Данные исключения
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "exception_type": self.__class__.__name__,
        }


class EncryptionError(Exception):
    """Encryption/Decryption errors"""
    pass


class UnauthorizedError(Exception):
    """401 Unauthorized"""
    pass


class ForbiddenError(Exception):
    """403 Forbidden"""
    pass


class ValidationError(Exception):
    """Validation errors"""
    pass


class AuthenticationError(Exception):
    """General authentication errors"""
    pass


# Дополнительные исключения для полноты
class NotFoundError(BaseAppException):
    """Resource not found (404)"""
    pass


class ConflictError(BaseAppException):
    """Resource conflict (409)"""
    pass


class DatabaseError(BaseAppException):
    """Database operation errors"""
    pass


class ValidationError(BaseAppException):
    """
    Исключение для ошибок валидации данных.
    """

    def __init__(
        self,
        message: str,
        field: str = None,
        value: Any = None,
        errors: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, "VALIDATION_ERROR", kwargs.get("details"))
        self.field = field
        self.value = value
        self.errors = errors or []
        self.status_code = 400

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование исключения в словарь."""
        result = super().to_dict()
        result.update(
            {"field": self.field, "value": self.value, "validation_errors": self.errors}
        )
        return result


class ProcessingError(BaseAppException):
    """
    Исключение для ошибок обработки данных.
    """

    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(message, "PROCESSING_ERROR", kwargs.get("details"))
        self.operation = operation
        self.status_code = 422


class DatabaseError(BaseAppException):
    """
    Исключение для ошибок работы с базой данных.
    """

    def __init__(
        self,
        message: str,
        operation: str = None,
        table: str = None,
        query: str = None,
        **kwargs,
    ):
        super().__init__(message, "DATABASE_ERROR", kwargs.get("details"))
        self.operation = operation
        self.table = table
        self.query = query
        self.status_code = 500


class NotFoundError(BaseAppException):
    """
    Исключение для случаев, когда ресурс не найден.
    """

    def __init__(
        self,
        message: str = None,
        resource_type: str = None,
        resource_id: str = None,
        **kwargs,
    ):
        if not message:
            if resource_id:
                message = (
                    f"{resource_type or 'Resource'} with id '{resource_id}' not found"
                )
            else:
                message = f"{resource_type or 'Resource'} not found"

        super().__init__(message, "NOT_FOUND", kwargs.get("details"))
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.status_code = 404


class UnauthorizedError(BaseAppException):
    """
    Исключение для ошибок авторизации.
    """

    def __init__(
        self,
        message: str = "Authentication required",
        auth_method: str = None,
        required_permissions: List[str] = None,
        **kwargs,
    ):
        super().__init__(message, "UNAUTHORIZED", kwargs.get("details"))
        self.auth_method = auth_method
        self.required_permissions = required_permissions
        self.status_code = 401


class ForbiddenError(BaseAppException):
    """
    Исключение для ошибок доступа (403).
    """

    def __init__(
        self,
        message: str = "Access forbidden",
        resource: str = None,
        action: str = None,
        required_permissions: List[str] = None,
        **kwargs,
    ):
        super().__init__(message, "FORBIDDEN", kwargs.get("details"))
        self.resource = resource
        self.action = action
        self.required_permissions = required_permissions
        self.status_code = 403


class ConflictError(BaseAppException):
    """
    Исключение для конфликтов (дублирование данных, etc.).
    """

    def __init__(
        self,
        message: str,
        resource_type: str = None,
        conflict_field: str = None,
        existing_value: Any = None,
        **kwargs,
    ):
        super().__init__(message, "CONFLICT", kwargs.get("details"))
        self.resource_type = resource_type
        self.conflict_field = conflict_field
        self.existing_value = existing_value
        self.status_code = 409


class EncryptionError(BaseAppException):
    """
    Исключение для ошибок шифрования/дешифрования.
    """

    def __init__(
        self, message: str, operation: str = None, key_id: str = None, **kwargs
    ):
        super().__init__(message, "ENCRYPTION_ERROR", kwargs.get("details"))
        self.operation = operation
        self.key_id = key_id
        self.status_code = 500


class StorageError(BaseAppException):
    """
    Исключение для ошибок работы с хранилищем файлов.
    """

    def __init__(
        self,
        message: str,
        storage_type: str = None,
        file_path: str = None,
        operation: str = None,
        **kwargs,
    ):
        super().__init__(message, "STORAGE_ERROR", kwargs.get("details"))
        self.storage_type = storage_type
        self.file_path = file_path
        self.operation = operation
        self.status_code = 500


class CacheError(BaseAppException):
    """
    Исключение для ошибок работы с кэшем.
    """

    def __init__(
        self,
        message: str,
        cache_type: str = None,
        key: str = None,
        operation: str = None,
        **kwargs,
    ):
        super().__init__(message, "CACHE_ERROR", kwargs.get("details"))
        self.cache_type = cache_type
        self.key = key
        self.operation = operation
        self.status_code = 500


class MLServiceError(BaseAppException):
    """
    Исключение для ошибок ML сервиса.
    """

    def __init__(
        self,
        message: str,
        model_name: str = None,
        operation: str = None,
        model_version: str = None,
        **kwargs,
    ):
        super().__init__(message, "ML_SERVICE_ERROR", kwargs.get("details"))
        self.model_name = model_name
        self.operation = operation
        self.model_version = model_version
        self.status_code = 502


class WebhookError(BaseAppException):
    """
    Исключение для ошибок webhook уведомлений.
    """

    def __init__(
        self,
        message: str,
        webhook_url: str = None,
        event_type: str = None,
        response_status: int = None,
        **kwargs,
    ):
        super().__init__(message, "WEBHOOK_ERROR", kwargs.get("details"))
        self.webhook_url = webhook_url
        self.event_type = event_type
        self.response_status = response_status
        self.status_code = 502


class RateLimitError(BaseAppException):
    """
    Исключение для превышения лимитов скорости.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: int = None,
        window: int = None,
        retry_after: int = None,
        **kwargs,
    ):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", kwargs.get("details"))
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        self.status_code = 429


class ConfigurationError(BaseAppException):
    """
    Исключение для ошибок конфигурации.
    """

    def __init__(
        self, message: str, config_key: str = None, config_value: Any = None, **kwargs
    ):
        super().__init__(message, "CONFIGURATION_ERROR", kwargs.get("details"))
        self.config_key = config_key
        self.config_value = config_value
        self.status_code = 500


class ExternalServiceError(BaseAppException):
    """
    Исключение для ошибок внешних сервисов.
    """

    def __init__(
        self,
        message: str,
        service_name: str = None,
        endpoint: str = None,
        status_code: int = None,
        response_body: str = None,
        **kwargs,
    ):
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", kwargs.get("details"))
        self.service_name = service_name
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_body = response_body
        self.status_code = 502


class BusinessLogicError(BaseAppException):
    """
    Исключение для ошибок бизнес-логики.
    """

    def __init__(
        self,
        message: str,
        business_rule: str = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(message, "BUSINESS_LOGIC_ERROR", kwargs.get("details"))
        self.business_rule = business_rule
        self.context = context
        self.status_code = 422


class QuotaExceededError(BaseAppException):
    """
    Исключение для превышения квот.
    """

    def __init__(
        self,
        message: str,
        quota_type: str = None,
        current_usage: int = None,
        quota_limit: int = None,
        reset_time: str = None,
        **kwargs,
    ):
        super().__init__(message, "QUOTA_EXCEEDED", kwargs.get("details"))
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit
        self.reset_time = reset_time
        self.status_code = 429


class MaintenanceModeError(BaseAppException):
    """
    Исключение для режима обслуживания.
    """

    def __init__(
        self,
        message: str = "Service is under maintenance",
        estimated_downtime: str = None,
        maintenance_info: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(message, "MAINTENANCE_MODE", kwargs.get("details"))
        self.estimated_downtime = estimated_downtime
        self.maintenance_info = maintenance_info
        self.status_code = 503


class TimeoutError(BaseAppException):
    """
    Исключение для превышения времени ожидания.
    """

    def __init__(
        self, message: str, operation: str = None, timeout_seconds: int = None, **kwargs
    ):
        super().__init__(message, "TIMEOUT_ERROR", kwargs.get("details"))
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.status_code = 504


class UnsupportedOperationError(BaseAppException):
    """
    Исключение для неподдерживаемых операций.
    """

    def __init__(
        self,
        message: str,
        operation: str = None,
        supported_operations: List[str] = None,
        **kwargs,
    ):
        super().__init__(message, "UNSUPPORTED_OPERATION", kwargs.get("details"))
        self.operation = operation
        self.supported_operations = supported_operations
        self.status_code = 501


class RetryExhaustedError(BaseAppException):
    """
    Исключение для случаев, когда исчерпаны все попытки повтора.
    """

    def __init__(
        self,
        message: str = "All retry attempts exhausted",
        operation: str = None,
        max_retries: int = None,
        retry_count: int = None,
        **kwargs,
    ):
        super().__init__(message, "RETRY_EXHAUSTED", kwargs.get("details"))
        self.operation = operation
        self.max_retries = max_retries
        self.retry_count = retry_count
        self.status_code = 502


class DataIntegrityError(BaseAppException):
    """
    Исключение для ошибок целостности данных.
    """

    def __init__(
        self,
        message: str,
        table: str = None,
        constraint: str = None,
        conflicting_data: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(message, "DATA_INTEGRITY_ERROR", kwargs.get("details"))
        self.table = table
        self.constraint = constraint
        self.conflicting_data = conflicting_data
        self.status_code = 409


class AuthenticationError(BaseAppException):
    """
    Исключение для ошибок аутентификации.
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        auth_method: str = None,
        token_info: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(message, "AUTHENTICATION_ERROR", kwargs.get("details"))
        self.auth_method = auth_method
        self.token_info = token_info
        self.status_code = 401


# Функции для работы с исключениями


def handle_exception(exception: Exception) -> BaseAppException:
    """
    Преобразование стандартных исключений в кастомные.

    Args:
        exception: Стандартное исключение

    Returns:
        BaseAppException: Кастомное исключение
    """
    # Если исключение уже является кастомным, возвращаем как есть
    if isinstance(exception, BaseAppException):
        return exception

    # Маппинг стандартных исключений на кастомные
    exception_mapping = {
        ValueError: lambda e: ValidationError(str(e)),
        TypeError: lambda e: ValidationError(f"Type error: {str(e)}"),
        KeyError: lambda e: NotFoundError(f"Key not found: {str(e)}"),
        FileNotFoundError: lambda e: NotFoundError(f"File not found: {str(e)}"),
        PermissionError: lambda e: ForbiddenError(str(e)),
        ConnectionError: lambda e: ExternalServiceError(str(e)),
        TimeoutError: lambda e: TimeoutError(str(e)),
        OSError: lambda e: ProcessingError(str(e)),
    }

    # Находим соответствующее исключение
    for exc_type, exc_factory in exception_mapping.items():
        if isinstance(exception, exc_type):
            return exc_factory(exception)

    # По умолчанию возвращаем ProcessingError
    return ProcessingError(f"Unexpected error: {str(exception)}")


def create_validation_error(
    message: str,
    field: str = None,
    value: Any = None,
    errors: List[Dict[str, Any]] = None,
) -> ValidationError:
    """
    Создание ошибки валидации.

    Args:
        message: Сообщение об ошибке
        field: Поле, в котором произошла ошибка
        value: Некорректное значение
        errors: Список ошибок

    Returns:
        ValidationError: Исключение валидации
    """
    return ValidationError(
        message=message, field=field, value=value, errors=errors or []
    )


def create_not_found_error(
    resource_type: str, resource_id: str = None, message: str = None
) -> NotFoundError:
    """
    Создание ошибки "не найдено".

    Args:
        resource_type: Тип ресурса
        resource_id: ID ресурса
        message: Сообщение об ошибке

    Returns:
        NotFoundError: Исключение "не найдено"
    """
    return NotFoundError(
        message=message, resource_type=resource_type, resource_id=resource_id
    )


def create_unauthorized_error(
    message: str = "Authentication required",
    auth_method: str = None,
    required_permissions: List[str] = None,
) -> UnauthorizedError:
    """
    Создание ошибки авторизации.

    Args:
        message: Сообщение об ошибке
        auth_method: Метод аутентификации
        required_permissions: Требуемые права доступа

    Returns:
        UnauthorizedError: Исключение авторизации
    """
    return UnauthorizedError(
        message=message,
        auth_method=auth_method,
        required_permissions=required_permissions,
    )
