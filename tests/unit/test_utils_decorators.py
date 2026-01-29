"""
Тесты для утилитных декораторов
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from app.utils.decorators import (
    cache_result,
    deprecated,
    log_request,
    measure_time,
    rate_limit,
    require_auth,
    retry_on_failure,
    validate_input,
)


class TestDecorators:
    """Тесты декораторов"""

    def test_validate_input_decorator(self):
        """Тест декоратора валидации входных данных"""

        @validate_input()
        def test_function(x: int, y: str):
            return f"{y}_{x}"

        # Валидные входные данные
        result = test_function(42, "test")
        assert result == "test_42"

    def test_log_request_decorator(self):
        """Тест декоратора логирования запросов"""
        with patch("app.utils.decorators.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            @log_request()
            def test_function():
                return "success"

            result = test_function()
            assert result == "success"
            # Проверяем, что логирование было вызвано
            assert mock_logger.info.call_count >= 1

    def test_measure_time_decorator(self):
        """Тест декоратора измерения времени"""

        @measure_time()
        def slow_function():
            time.sleep(0.01)
            return "completed"

        result = slow_function()
        assert result == "completed"

    def test_retry_on_failure_decorator(self):
        """Тест декоратора повторных попыток"""
        attempt_count = 0

        @retry_on_failure(max_retries=2, delay=0.01)
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = failing_function()
        assert result == "success"
        assert attempt_count == 3

    def test_cache_result_decorator(self):
        """Тест декоратора кэширования результатов"""
        call_count = 0

        @cache_result(ttl=60)
        def cached_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # Одинаковые аргументы должны возвращать кэшированный результат
        result1 = cached_function(1, 2)
        result2 = cached_function(1, 2)
        result3 = cached_function(2, 3)

        assert result1 == result2 == 3
        assert result3 == 5
        assert call_count == 2  # Только 2 уникальных вызова

    def test_require_auth_decorator(self):
        """Тест декоратора требования аутентификации"""

        @require_auth()
        def protected_function(request):
            return f"user_{request.state.user_id}"

        # Без аутентификации должен вызвать исключение
        mock_request = Mock()
        mock_request.state.user_id = None  # Явно указываем отсутствие user_id
        with pytest.raises(Exception):  # UnauthorizedError
            protected_function(mock_request)

        # С аутентификацией должен работать
        mock_request = Mock()
        mock_request.state.user_id = "test_user"
        result = protected_function(mock_request)
        assert result == "user_test_user"

    def test_rate_limit_decorator(self):
        """Тест декоратора ограничения скорости"""
        call_count = 0

        @rate_limit(requests_per_minute=3)
        def limited_function():
            nonlocal call_count
            call_count += 1
            return f"call_{call_count}"

        # Первые 3 вызова должны пройти
        results = []
        for _ in range(3):
            result = limited_function()
            results.append(result)

        assert len(results) == 3
        assert all("call_" in result for result in results)

        # 4-й вызов должен превысить лимит
        with pytest.raises(Exception):  # RateLimitError
            limited_function()

    def test_deprecated_decorator(self):
        """Тест декоратора устаревших функций"""
        with patch("app.utils.decorators.logger") as mock_logger:

            @deprecated("Use new_function instead")
            def old_function():
                return "old_result"

            result = old_function()
            assert result == "old_result"
            # Проверяем предупреждение о депрекации
            mock_logger.warning.assert_called_once()

    def test_async_decorators(self):
        """Тест асинхронных декораторов"""
        call_count = 0

        @measure_time()
        async def async_function():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return "async_result"

        result = asyncio.run(async_function())
        assert result == "async_result"
        assert call_count == 1

    def test_cache_result_async(self):
        """Тест асинхронного кэширования"""
        call_count = 0

        @cache_result(ttl=60)
        async def async_cached_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        # Первый вызов
        result1 = asyncio.run(async_cached_function(5))
        # Второй вызов с тем же аргументом (должен взять из кэша)
        result2 = asyncio.run(async_cached_function(5))
        # Новый аргумент
        result3 = asyncio.run(async_cached_function(10))

        assert result1 == result2 == 10
        assert result3 == 20
        assert call_count == 2  # Только 2 уникальных вызова

    def test_retry_async(self):
        """Тест асинхронного retry"""
        attempt_count = 0

        @retry_on_failure(max_retries=2, delay=0.01)
        async def async_failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = asyncio.run(async_failing_function())
        assert result == "success"
        assert attempt_count == 3

    def test_rate_limit_async(self):
        """Тест асинхронного rate limiting"""
        call_count = 0

        @rate_limit(requests_per_minute=2)
        async def async_limited_function():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"async_call_{call_count}"

        # Тестируем в синхронном контексте
        async def run_test():
            # Первые 2 вызова должны пройти
            results = []
            for _ in range(2):
                result = await async_limited_function()
                results.append(result)

            assert len(results) == 2

            # 3-й вызов должен превысить лимит
            with pytest.raises(Exception):  # RateLimitError
                await async_limited_function()

        asyncio.run(run_test())

    def test_log_request_async(self):
        """Тест асинхронного логирования"""
        with patch("app.utils.decorators.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            @log_request()
            async def async_test_function():
                await asyncio.sleep(0.01)
                return "async_success"

            result = asyncio.run(async_test_function())
            assert result == "async_success"
            # Проверяем, что логирование было вызвано
            assert mock_logger.info.call_count >= 1

    def test_measure_time_with_threshold(self):
        """Тест измерения времени с порогом"""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            @measure_time(threshold=0.05)
            def slow_function():
                time.sleep(0.06)  # Больше порога
                return "slow_result"

            @measure_time(threshold=0.05)
            def fast_function():
                time.sleep(0.02)  # Меньше порога
                return "fast_result"

            slow_result = slow_function()
            fast_result = fast_function()

            assert slow_result == "slow_result"
            assert fast_result == "fast_result"

            # Проверяем, что для медленной функции было предупреждение
            mock_logger.warning.assert_called()
            # А для быстрой - только debug
            mock_logger.debug.assert_called()

    def test_retry_with_specific_exceptions(self):
        """Тест retry с конкретными типами исключений"""
        attempt_count = 0

        @retry_on_failure(max_retries=2, delay=0.01, exceptions=(ValueError, TypeError))
        def function_with_different_exceptions():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise KeyError("This should not be retried")  # Не в списке исключений
            elif attempt_count == 2:
                raise ValueError("This should be retried")  # В списке исключений
            return "success"

        # Первое исключение не должно повторяться
        with pytest.raises(KeyError):
            function_with_different_exceptions()

        assert attempt_count == 1

    def test_multiple_decorators_combination(self):
        """Тест комбинации нескольких декораторов"""
        call_count = 0

        @log_request()
        @measure_time()
        @cache_result(ttl=60)
        def multi_decorated_function(x: int):
            nonlocal call_count
            call_count += 1
            return x * 3

        # Первый вызов
        result1 = multi_decorated_function(5)
        # Второй вызов с тем же аргументом (должен взять из кэша)
        result2 = multi_decorated_function(5)

        assert result1 == result2 == 15
        assert call_count == 1  # Только один реальный вызов

    def test_cache_key_generation(self):
        """Тест генерации ключей кэша"""
        call_count = 0

        @cache_result(ttl=60)
        def function_with_different_args(a, b, c=10):
            nonlocal call_count
            call_count += 1
            return a + b + c

        # Разные вызовы должны создавать разные ключи
        function_with_different_args(1, 2, 3)
        function_with_different_args(1, 2, 4)  # Другое значение c
        function_with_different_args(1, 2, c=3)  # Именованный аргумент

        assert call_count == 3

    def test_async_retry_decorator(self):
        """Тест асинхронного декоратора retry с исключениями"""
        attempt_count = 0

        @retry_on_failure(max_retries=2, delay=0.01)
        async def async_function_with_exceptions():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = asyncio.run(async_function_with_exceptions())
        assert result == "success"
        assert attempt_count == 3

    def test_async_measure_time_decorator(self):
        """Тест асинхронного декоратора измерения времени"""

        @measure_time(threshold=0.02)
        async def async_slow_function():
            await asyncio.sleep(0.03)  # Превышает порог
            return "slow_async_result"

        @measure_time(threshold=0.02)
        async def async_fast_function():
            await asyncio.sleep(0.01)  # В пределах порога
            return "fast_async_result"

        slow_result = asyncio.run(async_slow_function())
        fast_result = asyncio.run(async_fast_function())

        assert slow_result == "slow_async_result"
        assert fast_result == "fast_async_result"

    def test_rate_limit_different_functions(self):
        """Тест rate limiting для разных функций"""
        call_count_1 = 0
        call_count_2 = 0

        @rate_limit(requests_per_minute=2)
        def limited_function_1():
            nonlocal call_count_1
            call_count_1 += 1
            return f"func1_{call_count_1}"

        @rate_limit(requests_per_minute=1)
        def limited_function_2():
            nonlocal call_count_2
            call_count_2 += 1
            return f"func2_{call_count_2}"

        # Функция 1 - первые 2 вызова должны пройти
        results_1 = []
        for _ in range(2):
            result = limited_function_1()
            results_1.append(result)

        assert len(results_1) == 2

        # Функция 2 - только первый вызов должен пройти
        result_2_1 = limited_function_2()
        assert result_2_1 == "func2_1"

        # Второй вызов функции 2 должен превысить лимит
        with pytest.raises(Exception):
            limited_function_2()
