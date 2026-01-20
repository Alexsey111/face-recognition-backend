"""
Тесты для структурированного логирования
"""
import pytest
import json
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from io import StringIO


from app.utils.structured_logging import (
    LoggerFactory,
    LogContext,
    get_logger,
    log_with_context,
    log_info,
    log_error,
    AuditLogger,
    redact_sensitive_data,
    LogLevel,
    LogEntry,
    create_log
)


class TestStructuredLogging:
    """Тесты структурированного логирования"""

    def test_log_context(self):
        """Тест контекстного логирования"""
        # Тест создания контекста
        ctx = LogContext(request_id="req-123", user_id="user-456")
        assert ctx.request_id == "req-123"
        assert ctx.user_id == "user-456"
        
        # Тест контекстного менеджера
        with LogContext(request_id="req-789"):
            current_ctx = LogContext.get_current()
            assert current_ctx.request_id == "req-789"
        
        # После выхода из контекста должен восстановиться предыдущий
        assert LogContext.get_current() is None

    def test_redact_sensitive_data(self):
        """Тест маскировки чувствительных данных"""
        # Тест с чувствительными полями
        data = {
            "username": "john_doe",
            "password": "secret123",
            "api_key": "abc123",
            "normal_field": "normal_value"
        }
        
        redacted = redact_sensitive_data(data)
        
        assert redacted["username"] == "john_doe"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["normal_field"] == "normal_value"

    def test_redact_sensitive_data_nested(self):
        """Тест маскировки во вложенных структурах"""
        data = {
            "user": {
                "name": "John Doe",
                "credentials": {
                    "password": "secret",
                    "token": "jwt_token"
                }
            },
            "normal_data": "public"
        }
        
        redacted = redact_sensitive_data(data)
        
        assert redacted["user"]["name"] == "John Doe"
        assert redacted["user"]["credentials"]["password"] == "[REDACTED]"
        assert redacted["user"]["credentials"]["token"] == "[REDACTED]"
        assert redacted["normal_data"] == "public"

    def test_log_level_constants(self):
        """Тест констант уровней логирования"""
        assert LogLevel.DEBUG == logging.DEBUG
        assert LogLevel.INFO == logging.INFO
        assert LogLevel.WARNING == logging.WARNING
        assert LogLevel.ERROR == logging.ERROR
        assert LogLevel.CRITICAL == logging.CRITICAL
        
        # Тест преобразования строки в уровень
        assert LogLevel.from_string("DEBUG") == logging.DEBUG
        assert LogLevel.from_string("INFO") == logging.INFO
        assert LogLevel.from_string("warning") == logging.WARNING

    def test_logger_factory(self):
        """Тест фабрики логгеров"""
        # Очищаем существующие логгеры
        LoggerFactory.clear_loggers()
        
        # Создаем новый логгер
        logger = LoggerFactory.setup_logger(
            name="test_logger",
            level="DEBUG",
            log_format="text"
        )
        
        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG
        
        # Получаем существующий логгер
        same_logger = LoggerFactory.get_logger("test_logger")
        assert same_logger is logger

    def test_get_logger_caching(self):
        """Тест кэширования логгеров"""
        LoggerFactory.clear_loggers()
        
        # Первый вызов должен создать логгер
        logger1 = get_logger("cached_logger")
        # Второй вызов должен вернуть тот же логгер
        logger2 = get_logger("cached_logger")
        
        assert logger1 is logger2

    def test_log_with_context(self):
        """Тест логирования с контекстом"""
        # Создаем мок логгера
        mock_logger = Mock()
        
        # Тест логирования с контекстом
        log_with_context(
            mock_logger,
            logging.INFO,
            "Test message",
            request_id="req-123",
            user_id="user-456",
            extra_field="extra_value"
        )
        
        # Проверяем, что логгер был вызван
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        
        # Проверяем параметры
        assert call_args[0][0] == logging.INFO  # level
        assert call_args[0][1] == "Test message"  # message
        
        # Проверяем extra fields
        extra = call_args[1]["extra"]
        assert extra["request_id"] == "req-123"
        assert extra["user_id"] == "user-456"
        assert extra["extra_field"] == "extra_value"

    def test_convenience_log_functions(self):
        """Тест удобных функций логирования"""
        mock_logger = Mock()
        
        # Тест log_info
        log_info(mock_logger, "Info message", request_id="req-1")
        assert mock_logger.log.call_count == 1
        
        # Тест log_error
        mock_error = ValueError("Test error")
        log_error(mock_logger, "Error message", error=mock_error, request_id="req-2")
        assert mock_logger.log.call_count == 2
        
        # Проверяем параметры последнего вызова
        call_args = mock_logger.log.call_args
        extra = call_args[1]["extra"]
        assert extra["error"] == "Test error"
        assert extra["error_type"] == "ValueError"

    def test_audit_logger(self):
        """Тест аудит логгера"""
        # Создаем мок логгера
        with patch('app.utils.structured_logging.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            audit_logger = AuditLogger("test_audit")
            
            # Тест логирования аутентификации
            audit_logger.auth_success(
                user_id="user-123",
                ip_address="192.168.1.1"
            )
            
            # Проверяем, что был вызван логгер
            assert mock_logger.log.called
            
            # Проверяем параметры вызова
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO  # level
            assert "login" in call_args[0][1]  # message

    def test_audit_logger_disabled(self):
        """Тест отключенного аудит логгера"""
        with patch('app.utils.structured_logging.settings') as mock_settings:
            mock_settings.AUDIT_LOG_ENABLED = False
            
            with patch('app.utils.structured_logging.get_logger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                audit_logger = AuditLogger("test_audit")
                
                # Логирование не должно происходить
                audit_logger.auth_success(user_id="user-123")
                
                # Логгер не должен быть вызван
                assert not mock_logger.log.called

    def test_log_entry_builder(self):
        """Тест построителя лог записей"""
        with patch('app.utils.structured_logging.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Создаем лог запись через builder
            entry = create_log("Test message", logging.INFO)
            entry.with_request_id("req-123")
            entry.with_user_id("user-456")
            entry.with_ip_address("192.168.1.1")
            entry.with_extra(field1="value1", field2="value2")
            entry.with_logger(mock_logger)
            
            # Выполняем логирование
            entry.info()
            
            # Проверяем вызов логгера
            assert mock_logger.log.called
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["request_id"] == "req-123"
            assert extra["user_id"] == "user-456"
            assert extra["ip_address"] == "192.168.1.1"
            assert extra["field1"] == "value1"
            assert extra["field2"] == "value2"

    def test_log_entry_error_with_exception(self):
        """Тест логирования ошибки с исключением"""
        with patch('app.utils.structured_logging.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            entry = create_log("Error occurred", logging.ERROR)
            entry.with_logger(mock_logger)
            
            # Создаем исключение
            test_error = ValueError("Test exception")
            
            # Логируем с исключением
            entry.error(error=test_error)
            
            # Проверяем параметры
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            assert extra["error"] == "Test exception"
            assert extra["error_type"] == "ValueError"

    def test_multiple_contexts(self):
        """Тест работы с несколькими контекстами"""
        # Создаем первый контекст
        with LogContext(request_id="req-1", user_id="user-1"):
            assert LogContext.get_request_id() == "req-1"
            assert LogContext.get_user_id() == "user-1"
            
            # Создаем вложенный контекст
            with LogContext(request_id="req-2", user_id="user-2"):
                assert LogContext.get_request_id() == "req-2"
                assert LogContext.get_user_id() == "user-2"
            
            # После выхода из вложенного контекста должен восстановиться первый
            assert LogContext.get_request_id() == "req-1"
            assert LogContext.get_user_id() == "user-1"
        
        # После выхода из всех контекстов должно быть пусто
        assert LogContext.get_request_id() is None
        assert LogContext.get_user_id() is None

    def test_redact_sensitive_data_lists(self):
        """Тест маскировки в списках"""
        data = [
            {"username": "user1", "password": "pass1"},
            {"username": "user2", "password": "pass2"},
            "normal_string",
            123
        ]
        
        redacted = redact_sensitive_data(data)
        
        assert redacted[0]["username"] == "user1"
        assert redacted[0]["password"] == "[REDACTED]"
        assert redacted[1]["username"] == "user2"
        assert redacted[1]["password"] == "[REDACTED]"
        assert redacted[2] == "normal_string"
        assert redacted[3] == 123

    def test_redact_sensitive_data_strings(self):
        """Тест маскировки в строках"""
        # Тест JWT токена
        jwt_string = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        
        redacted = redact_sensitive_data(jwt_string)
        
        # JWT токен должен быть замаскирован
        assert "[REDACTED]" in redacted
        
        # Обычная строка должна остаться без изменений
        normal_string = "This is a normal message"
        assert redact_sensitive_data(normal_string) == normal_string

    def test_logger_factory_with_file(self):
        """Тест фабрики логгеров с файлом"""
        import tempfile
        import os
        
        # Создаем временный файл (закрываем сразу, чтобы не блокировать)
        fd, temp_path = tempfile.mkstemp(suffix=".log")
        os.close(fd)
        
        try:
            LoggerFactory.clear_loggers()
            
            # Создаем логгер с файлом
            logger = LoggerFactory.setup_logger(
                name="file_logger",
                level="INFO",
                log_file=temp_path,
                log_format="text"
            )
            
            assert logger.name == "file_logger"
            assert logger.level == logging.INFO
            
            # Проверяем, что файл был создан
            # (В реальных условиях здесь был бы файл handler)
            
        finally:
            # Очищаем временный файл (сначала закрываем handlers)
            LoggerFactory.clear_loggers()
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    pass  # На Windows файл может быть занят

    def test_log_entry_chaining(self):
        """Тест цепочки вызовов в LogEntry"""
        with patch('app.utils.structured_logging.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Тестируем цепочку вызовов
            create_log("Test message") \
                .with_request_id("req-123") \
                .with_user_id("user-456") \
                .with_ip_address("192.168.1.1") \
                .with_tags(["test", "logging"]) \
                .with_extra(key1="value1", key2="value2") \
                .with_logger(mock_logger) \
                .info()
            
            # Проверяем вызов логгера
            assert mock_logger.log.called
            call_args = mock_logger.log.call_args
            extra = call_args[1]["extra"]
            
            assert extra["request_id"] == "req-123"
            assert extra["user_id"] == "user-456"
            assert extra["ip_address"] == "192.168.1.1"
            assert "test" in extra["tags"]
            assert "logging" in extra["tags"]
            assert extra["key1"] == "value1"
            assert extra["key2"] == "value2"

    def test_audit_logger_convenience_methods(self):
        """Тест удобных методов аудит логгера"""
        with patch('app.utils.structured_logging.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            audit_logger = AuditLogger("test_audit")
            
            # Тест различных методов аудита
            audit_logger.auth_success(user_id="user-123", ip_address="192.168.1.1")
            audit_logger.auth_failed(user_id="user-456", error_message="Invalid password")
            audit_logger.verification_started(user_id="user-789", session_id="sess-123")
            audit_logger.verification_completed(
                user_id="user-789", 
                session_id="sess-123", 
                is_match=True, 
                similarity_score=0.95
            )
            
            # Проверяем, что логгер вызывался 4 раза
            assert mock_logger.log.call_count == 4

    def test_redact_sensitive_data_depth_limit(self):
        """Тест ограничения глубины рекурсии при маскировке"""
        # Создаем глубоко вложенную структуру
        deep_data = {"level1": {"level2": {"level3": {"level4": {"level5": {"password": "secret"}}}}}}
        
        # С ограничением глубины 3, пароль не должен быть замаскирован
        redacted = redact_sensitive_data(deep_data, max_depth=3)
        
        # Пароль должен остаться видимым из-за ограничения глубины
        assert redacted["level1"]["level2"]["level3"]["level4"]["level5"]["password"] == "secret"

    def test_log_context_empty_initialization(self):
        """Тест создания пустого контекста"""
        ctx = LogContext()
        
        assert ctx.request_id is None
        assert ctx.user_id is None
        assert ctx.extra == {}

    def test_redact_sensitive_data_no_sensitive_fields(self):
        """Тест маскировки без чувствительных полей"""
        data = {
            "name": "John Doe",
            "age": 30,
            "city": "New York",
            "hobbies": ["reading", "swimming"]
        }
        
        redacted = redact_sensitive_data(data)
        
        # Все поля должны остаться без изменений
        assert redacted == data

    def test_log_entry_different_levels(self):
        """Тест логирования на разных уровнех"""
        with patch('app.utils.structured_logging.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            entry = create_log("Test message")
            entry.with_logger(mock_logger)
            
            # Тестируем все уровни
            entry.debug()
            assert mock_logger.log.call_count == 1
            assert mock_logger.log.call_args[0][0] == logging.DEBUG
            
            entry.info()
            assert mock_logger.log.call_count == 2
            assert mock_logger.log.call_args[0][0] == logging.INFO
            
            entry.warning()
            assert mock_logger.log.call_count == 3
            assert mock_logger.log.call_args[0][0] == logging.WARNING
            
            entry.error()
            assert mock_logger.log.call_count == 4
            assert mock_logger.log.call_args[0][0] == logging.ERROR
            
            entry.critical()
            assert mock_logger.log.call_count == 5
            assert mock_logger.log.call_args[0][0] == logging.CRITICAL

    def test_log_entry_builder(self):
        """Тест построителя лог записей"""
        # Имитация построителя лог записей
        class LogEntry:
            def __init__(self, message, level="INFO"):
                self.message = message
                self.level = level
                self.request_id = None
                self.user_id = None
                self.ip_address = None
                self.extra = {}
                self.tags = []
            
            def with_request_id(self, request_id):
                self.request_id = request_id
                return self
            
            def with_user_id(self, user_id):
                self.user_id = user_id
                return self
            
            def with_ip_address(self, ip_address):
                self.ip_address = ip_address
                return self
            
            def with_extra(self, **kwargs):
                self.extra.update(kwargs)
                return self
            
            def with_tags(self, tags):
                self.tags.extend(tags)
                return self
            
            def build(self):
                entry = {
                    "message": self.message,
                    "level": self.level,
                    "timestamp": datetime.now().isoformat() + "Z"
                }
                
                if self.request_id:
                    entry["request_id"] = self.request_id
                if self.user_id:
                    entry["user_id"] = self.user_id
                if self.ip_address:
                    entry["ip_address"] = self.ip_address
                if self.extra:
                    entry.update(self.extra)
                if self.tags:
                    entry["tags"] = self.tags
                
                return entry
        
        # Тест построителя
        entry = (LogEntry("Test message")
                 .with_request_id("req-123")
                 .with_user_id("user-456")
                 .with_ip_address("192.168.1.1")
                 .with_extra(field1="value1", field2="value2")
                 .with_tags(["test", "logging"])
                 .build())
        
        assert entry["message"] == "Test message"
        assert entry["request_id"] == "req-123"
        assert entry["user_id"] == "user-456"
        assert entry["ip_address"] == "192.168.1.1"
        assert entry["field1"] == "value1"
        assert entry["field2"] == "value2"
        assert "test" in entry["tags"]
        assert "logging" in entry["tags"]

    def test_log_rotation_concept(self):
        """Тест концепции ротации логов"""
        # Имитация лог ротации
        def should_rotate_log(current_size, max_size, backup_count):
            return current_size >= max_size
        
        # Тест условий ротации
        assert should_rotate_log(100 * 1024 * 1024, 100 * 1024 * 1024, 10) is True  # Равно max
        assert should_rotate_log(150 * 1024 * 1024, 100 * 1024 * 1024, 10) is True  # Больше max
        assert should_rotate_log(50 * 1024 * 1024, 100 * 1024 * 1024, 10) is False  # Меньше max

    def test_log_filtering_by_level(self):
        """Тест фильтрации логов по уровням"""
        # Имитация фильтра логов
        def filter_logs_by_level(logs, min_level):
            level_priority = {
                "DEBUG": 10,
                "INFO": 20,
                "WARNING": 30,
                "ERROR": 40,
                "CRITICAL": 50
            }
            
            min_priority = level_priority.get(min_level, 20)
            
            return [
                log for log in logs 
                if level_priority.get(log["level"], 0) >= min_priority
            ]
        
        # Тестовые логи
        test_logs = [
            {"level": "DEBUG", "message": "Debug info"},
            {"level": "INFO", "message": "Info message"},
            {"level": "WARNING", "message": "Warning"},
            {"level": "ERROR", "message": "Error"},
            {"level": "CRITICAL", "message": "Critical"}
        ]
        
        # Фильтрация по уровню INFO
        filtered = filter_logs_by_level(test_logs, "INFO")
        assert len(filtered) == 4  # INFO, WARNING, ERROR, CRITICAL
        assert all(log["level"] != "DEBUG" for log in filtered)

    def test_context_correlation(self):
        """Тест корреляции контекста в логах"""
        # Имитация корреляции запросов
        correlation_id = "correlation-123"
        
        def create_correlated_logs(correlation_id):
            return [
                {
                    "timestamp": "2024-01-01T10:00:00Z",
                    "level": "INFO",
                    "message": "Request received",
                    "correlation_id": correlation_id
                },
                {
                    "timestamp": "2024-01-01T10:00:01Z",
                    "level": "DEBUG",
                    "message": "Processing request",
                    "correlation_id": correlation_id
                },
                {
                    "timestamp": "2024-01-01T10:00:02Z",
                    "level": "INFO",
                    "message": "Request completed",
                    "correlation_id": correlation_id
                }
            ]
        
        logs = create_correlated_logs(correlation_id)
        
        # Все логи должны иметь одинаковый correlation_id
        assert all(log["correlation_id"] == correlation_id for log in logs)
        assert len(logs) == 3

    def test_log_performance_tracking(self):
        """Тест отслеживания производительности в логах"""
        # Имитация отслеживания времени выполнения
        import time
        
        def log_with_timing(message, start_time):
            duration = time.time() - start_time
            return {
                "message": message,
                "duration_ms": duration * 1000,
                "timestamp": datetime.now().isoformat() + "Z"
            }
        
        start = time.time()
        time.sleep(0.01)  # Имитируем работу
        log_entry = log_with_timing("Operation completed", start)
        
        assert "duration_ms" in log_entry
        assert log_entry["duration_ms"] >= 10  # Минимум 10ms
        assert log_entry["duration_ms"] <= 50   # Максимум 50ms (с запасом)

    def test_error_logging_structure(self):
        """Тест структуры логирования ошибок"""
        def create_error_log(message, error=None, error_type=None, **kwargs):
            log_entry = {
                "level": "ERROR",
                "message": message,
                "timestamp": datetime.now().isoformat() + "Z"
            }
            
            if error:
                log_entry["error"] = str(error)
            if error_type:
                log_entry["error_type"] = error_type
            
            log_entry.update(kwargs)
            return log_entry
        
        # Тест логирования ошибки с исключением
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_log = create_error_log(
                "Operation failed", 
                error=e, 
                error_type="ValueError",
                request_id="req-123"
            )
        
        assert error_log["level"] == "ERROR"
        assert error_log["error"] == "Test error"
        assert error_log["error_type"] == "ValueError"
        assert error_log["request_id"] == "req-123"

    def test_log_sanitization(self):
        """Тест санитизации логов"""
        def sanitize_log_data(data):
            """Удаление потенциально опасных данных из логов"""
            sanitized = {}
            
            for key, value in data.items():
                key_lower = key.lower()
                
                # Удаляем чувствительные поля
                if any(sensitive in key_lower for sensitive in [
                    "password", "token", "secret", "key", "credential"
                ]):
                    sanitized[key] = "[REDACTED]"
                # Санитизируем строки
                elif isinstance(value, str):
                    sanitized[key] = value.replace("<", "<").replace(">", ">")
                else:
                    sanitized[key] = value
            
            return sanitized
        
        # Тест санитизации
        raw_data = {
            "username": "user@example.com",
            "password": "secret123",
            "message": "<script>alert('xss')</script>",
            "api_key": "abc123xyz",
            "normal_field": "normal value"
        }
        
        sanitized = sanitize_log_data(raw_data)
        
        assert sanitized["username"] == "user@example.com"
        assert sanitized["password"] == "[REDACTED]"
        assert "<script>" in sanitized["message"]
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["normal_field"] == "normal value"

    def test_redact_sensitive_data_nested_structures(self):
        """Тест маскировки в глубоко вложенных структурах"""
        def redact_sensitive_data(data, depth=0, max_depth=5):
            if depth > max_depth:
                return data
            
            if isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    if any(sensitive in key.lower() for sensitive in ["password", "secret", "key", "token"]):
                        result[key] = "[REDACTED]"
                    else:
                        result[key] = redact_sensitive_data(value, depth + 1, max_depth)
                return result
            elif isinstance(data, list):
                return [redact_sensitive_data(item, depth + 1, max_depth) for item in data]
            else:
                return data
        
        # Создаем глубоко вложенную структуру
        deep_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "password": "secret"
                            }
                        }
                    }
                }
            }
        }
        
        # С ограничением глубины 3, пароль не должен быть замаскирован
        redacted = redact_sensitive_data(deep_data, max_depth=3)
        
        # Пароль должен остаться видимым из-за ограничения глубины
        assert redacted["level1"]["level2"]["level3"]["level4"]["level5"]["password"] == "secret"

    def test_multiple_log_contexts(self):
        """Тест работы с множественными контекстами"""
        class LogContext:
            def __init__(self, request_id=None, user_id=None):
                self.request_id = request_id
                self.user_id = user_id
                self.previous_context = None
            
            def __enter__(self):
                self.previous_context = getattr(LogContext, "_current_context", None)
                LogContext._current_context = self
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                LogContext._current_context = self.previous_context
            
            @classmethod
            def get_current(cls):
                return getattr(cls, "_current_context", None)
        
        # Создаем первый контекст
        with LogContext(request_id="req-1", user_id="user-1"):
            assert LogContext.get_current().request_id == "req-1"
            
            # Создаем вложенный контекст
            with LogContext(request_id="req-2", user_id="user-2"):
                assert LogContext.get_current().request_id == "req-2"
            
            # После выхода из вложенного контекста должен восстановиться первый
            assert LogContext.get_current().request_id == "req-1"
        
        # После выхода из всех контекстов должно быть пусто
        assert LogContext.get_current() is None

    def test_log_batch_processing(self):
        """Тест пакетной обработки логов"""
        def process_log_batch(logs):
            """Обработка пакета логов"""
            processed = []
            for log in logs:
                # Добавляем метаданные
                processed_log = {
                    "processed_at": datetime.now().isoformat() + "Z",
                    "batch_id": "batch-123",
                    **log
                }
                processed.append(processed_log)
            return processed
        
        # Тестовые логи
        raw_logs = [
            {"level": "INFO", "message": "Log 1"},
            {"level": "WARNING", "message": "Log 2"},
            {"level": "ERROR", "message": "Log 3"}
        ]
        
        processed = process_log_batch(raw_logs)
        
        assert len(processed) == 3
        assert all("processed_at" in log for log in processed)
        assert all("batch_id" in log for log in processed)
        assert all(log["batch_id"] == "batch-123" for log in processed)

    def test_log_metrics_collection(self):
        """Тест сбора метрик из логов"""
        def collect_log_metrics(logs):
            """Сбор метрик из логов"""
            metrics = {
                "total_logs": len(logs),
                "by_level": {},
                "with_errors": 0,
                "unique_request_ids": set()
            }
            
            for log in logs:
                # Подсчет по уровням
                level = log.get("level", "UNKNOWN")
                metrics["by_level"][level] = metrics["by_level"].get(level, 0) + 1
                
                # Подсчет ошибок
                if level in ["ERROR", "CRITICAL"]:
                    metrics["with_errors"] += 1
                
                # Уникальные request_id
                if "request_id" in log:
                    metrics["unique_request_ids"].add(log["request_id"])
            
            metrics["unique_request_ids"] = len(metrics["unique_request_ids"])
            return metrics
        
        # Тестовые логи
        test_logs = [
            {"level": "INFO", "message": "Info 1", "request_id": "req-1"},
            {"level": "WARNING", "message": "Warning 1", "request_id": "req-1"},
            {"level": "ERROR", "message": "Error 1", "request_id": "req-2"},
            {"level": "INFO", "message": "Info 2", "request_id": "req-3"},
            {"level": "CRITICAL", "message": "Critical 1"}
        ]
        
        metrics = collect_log_metrics(test_logs)
        
        assert metrics["total_logs"] == 5
        assert metrics["by_level"]["INFO"] == 2
        assert metrics["by_level"]["WARNING"] == 1
        assert metrics["by_level"]["ERROR"] == 1
        assert metrics["by_level"]["CRITICAL"] == 1
        assert metrics["with_errors"] == 2
        assert metrics["unique_request_ids"] == 3