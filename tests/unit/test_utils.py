import pytest
import logging
from unittest.mock import Mock, patch
from app.utils.logger import setup_logger, get_logger
from app.utils.constants import (
    IMAGE_FORMATS,
    PAGINATION,
    API_LIMITS,
    THRESHOLDS,
    SECURITY,
    TIME_PERIODS,
    USER_ROLES
)
from app.utils.decorators import (
    retry_on_failure,
    validate_input,
    log_execution_time
)


class TestLogger:
    """Тесты для системы логирования"""
    
    def test_setup_logger(self):
        """Тест настройки логгера"""
        logger = setup_logger("test_logger", level=logging.INFO)
        
        assert logger is not None
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
    
    def test_get_logger(self):
        """Тест получения логгера"""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert logger.name == "test_module"
    
    def test_logger_configuration(self):
        """Тест конфигурации логгера"""
        with patch('app.utils.logger.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            logger = setup_logger("test_config", level=logging.DEBUG)
            
            assert logger is not None


class TestConstants:
    """Тесты для констант"""
    
    def test_image_formats(self):
        """Тест форматов изображений"""
        assert isinstance(IMAGE_FORMATS, list)
        assert len(IMAGE_FORMATS) > 0
        assert "JPEG" in IMAGE_FORMATS
        assert "PNG" in IMAGE_FORMATS

    def test_pagination_constants(self):
        """Тест констант пагинации"""
        assert "default_page_size" in PAGINATION
        assert "max_page_size" in PAGINATION
        assert "min_page_size" in PAGINATION
        assert PAGINATION["default_page_size"] == 20
        assert PAGINATION["max_page_size"] == 100
    
    def test_api_limits(self):
        """Тест лимитов API"""
        assert isinstance(API_LIMITS, dict)
        assert "default_requests_per_minute" in API_LIMITS
        assert "verify_requests_per_minute" in API_LIMITS
        assert API_LIMITS["default_requests_per_minute"] == 60
    
    def test_thresholds(self):
        """Тест пороговых значений"""
        assert "verification" in THRESHOLDS
        assert "liveness" in THRESHOLDS
        assert THRESHOLDS["verification"]["default"] == 0.8
        assert THRESHOLDS["liveness"]["default"] == 0.8
    
    def test_security_constants(self):
        """Тест констант безопасности"""
        assert "password_min_length" in SECURITY
        assert "max_login_attempts" in SECURITY
        assert SECURITY["password_min_length"] == 8
        assert SECURITY["max_login_attempts"] == 5
    
    def test_time_constants(self):
        """Тест временных констант"""
        assert "minute" in TIME_PERIODS
        assert "hour" in TIME_PERIODS
        assert "day" in TIME_PERIODS
        assert TIME_PERIODS["minute"] == 60
        assert TIME_PERIODS["hour"] == 3600
    
    def test_user_roles(self):
        """Тест ролей пользователей"""
        assert "USER" in USER_ROLES
        assert "ADMIN" in USER_ROLES
        assert USER_ROLES["USER"] == "user"
        assert USER_ROLES["ADMIN"] == "admin"


class TestDecorators:
    """Тесты для декораторов"""
    
    def test_retry_on_failure_success(self):
        """Тест декоратора повторных попыток при успехе"""
        @retry_on_failure(max_retries=3, delay=1)
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_retry_on_failure_failure(self):
        """Тест декоратора повторных попыток при неудаче"""
        @retry_on_failure(max_retries=2, delay=0.1)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
    
    def test_validate_input_valid(self):
        """Тест валидации входных данных с валидными данными"""
        @validate_input()
        def test_function(data: dict, count: int):
            return f"Processed {count} items: {data}"
        
        result = test_function(data={"key": "value"}, count=5)
        assert result == "Processed 5 items: {'key': 'value'}"
    
    def test_validate_input_invalid(self):
        """Тест валидации входных данных с невалидными данными"""
        # Схема валидации для проверки типов аргументов
        validation_schema = {
            "required": ["data", "count"],
            "properties": {
                "data": {"type": "object"},
                "count": {"type": "integer"}
            }
        }
        
        @validate_input(validation_schema)
        def test_function(data: dict, count: int):
            return f"Processed {count} items: {data}"
        
        # Невалидный тип для count
        with pytest.raises(Exception):  # ValidationError
            test_function({"key": "value"}, "not_a_number")
        
        # Невалидный тип для data
        with pytest.raises(Exception):  # ValidationError
            test_function("not_a_dict", 5)
    
    def test_log_execution_time(self):
        """Тест декоратора логирования времени выполнения"""
        @log_execution_time()
        def slow_function():
            import time
            time.sleep(0.1)  # Имитация медленной операции
            return "completed"
        
        # Просто проверяем, что декоратор применяется и функция работает
        result = slow_function()
        assert result == "completed"
        
        # Проверяем, что функция действительно декорирована
        assert hasattr(slow_function, '__wrapped__')


class TestExceptions:
    """Тесты для исключений"""
    
    def test_validation_error(self):
        """Тест исключения валидации"""
        from app.utils.exceptions import ValidationError
        
        error_msg = "Invalid input data"
        error = ValidationError(error_msg)
        
        assert str(error) == error_msg
    
    def test_validation_error_inheritance(self):
        """Тест наследования исключений валидации"""
        from app.utils.exceptions import ValidationError
        
        # Проверяем, что ValidationError наследуется от Exception
        assert issubclass(ValidationError, Exception)
        
        # Создаем экземпляр и проверяем, что это Exception
        error = ValidationError("test")
        assert isinstance(error, Exception)