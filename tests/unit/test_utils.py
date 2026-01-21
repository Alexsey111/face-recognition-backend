import pytest
import logging
from unittest.mock import Mock, patch
from app.utils.logger import setup_logger, get_logger
from app.utils.constants import (
    IMAGE_FORMATS,
    IMAGE_FORMAT_ALIASES,
    FILE_LIMITS,
    SIMILARITY_LIMITS,
    CONFIDENCE_LEVELS,
    USER_ROLES,
    TIME_PERIODS,
    EMAIL_REGEX,
    USERNAME_REGEX,
    PASSWORD_REGEX,
)
from app.utils.decorators import retry_on_failure, validate_input, log_execution_time


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
        with patch("app.utils.logger.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            logger = setup_logger("test_config", level=logging.DEBUG)

            assert logger is not None


class TestConstants:
    """Тесты для констант"""

    def test_image_formats(self):
        """Тест форматов изображений"""
        assert isinstance(IMAGE_FORMATS, set)
        assert len(IMAGE_FORMATS) > 0
        assert "JPEG" in IMAGE_FORMATS
        assert "PNG" in IMAGE_FORMATS

    def test_image_format_aliases(self):
        """Тест алиасов форматов изображений"""
        assert isinstance(IMAGE_FORMAT_ALIASES, dict)
        assert "JPG" in IMAGE_FORMAT_ALIASES
        assert IMAGE_FORMAT_ALIASES["JPG"] == "JPEG"
        assert "TIF" in IMAGE_FORMAT_ALIASES
        assert IMAGE_FORMAT_ALIASES["TIF"] == "TIFF"

    def test_file_limits(self):
        """Тест лимитов файлов"""
        assert isinstance(FILE_LIMITS, dict)
        assert "max_image_size" in FILE_LIMITS
        assert "max_filename_length" in FILE_LIMITS
        assert FILE_LIMITS["max_image_size"] == 10 * 1024 * 1024  # 10 MB
        assert FILE_LIMITS["max_filename_length"] == 255

    def test_similarity_limits(self):
        """Тест лимитов схожести"""
        assert isinstance(SIMILARITY_LIMITS, dict)
        assert "min_threshold" in SIMILARITY_LIMITS
        assert "max_threshold" in SIMILARITY_LIMITS
        assert "default_threshold" in SIMILARITY_LIMITS
        assert SIMILARITY_LIMITS["min_threshold"] == 0.0
        assert SIMILARITY_LIMITS["max_threshold"] == 1.0
        assert SIMILARITY_LIMITS["default_threshold"] == 0.6

    def test_confidence_levels(self):
        """Тест уровней уверенности"""
        assert isinstance(CONFIDENCE_LEVELS, dict)
        assert "very_high" in CONFIDENCE_LEVELS
        assert "high" in CONFIDENCE_LEVELS
        assert "medium" in CONFIDENCE_LEVELS
        assert "low" in CONFIDENCE_LEVELS
        assert "very_low" in CONFIDENCE_LEVELS
        assert CONFIDENCE_LEVELS["very_high"] == 0.85
        assert CONFIDENCE_LEVELS["high"] == 0.75

    def test_user_roles(self):
        """Тест ролей пользователей"""
        assert isinstance(USER_ROLES, dict)
        assert "USER" in USER_ROLES
        assert "ADMIN" in USER_ROLES
        assert "SYSTEM" in USER_ROLES
        assert USER_ROLES["USER"] == "user"
        assert USER_ROLES["ADMIN"] == "admin"

    def test_time_constants(self):
        """Тест временных констант"""
        assert isinstance(TIME_PERIODS, dict)
        assert "second" in TIME_PERIODS
        assert "minute" in TIME_PERIODS
        assert "hour" in TIME_PERIODS
        assert "day" in TIME_PERIODS
        assert TIME_PERIODS["second"] == 1
        assert TIME_PERIODS["minute"] == 60
        assert TIME_PERIODS["hour"] == 3600

    def test_email_regex(self):
        """Тест регулярного выражения для email"""
        assert EMAIL_REGEX is not None
        assert EMAIL_REGEX.match("test@example.com") is not None
        assert EMAIL_REGEX.match("invalid-email") is None

    def test_username_regex(self):
        """Тест регулярного выражения для username"""
        assert USERNAME_REGEX is not None
        assert USERNAME_REGEX.match("john_doe") is not None
        assert USERNAME_REGEX.match("ab") is None  # too short

    def test_password_regex(self):
        """Тест регулярного выражения для пароля"""
        assert PASSWORD_REGEX is not None
        # Valid password: 8+ chars, uppercase, lowercase, digit, special char
        assert PASSWORD_REGEX.match("Password1!") is not None
        # Invalid: no uppercase
        assert PASSWORD_REGEX.match("password1!") is None


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
            "properties": {"data": {"type": "object"}, "count": {"type": "integer"}},
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
        assert hasattr(slow_function, "__wrapped__")


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
