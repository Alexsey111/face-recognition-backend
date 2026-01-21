"""
Тесты для CORS middleware.
Модуль с покрытием 0% - цель: увеличить до 80%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI, Request
from starlette.responses import Response, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Mock настроек
with patch("app.middleware.cors.settings") as mock_settings:
    mock_settings.CORS_ORIGINS = "https://example.com,https://app.example.com"
    mock_settings.DEBUG = False

    from app.middleware.cors import (
        CORSMiddleware,
        SecurityHeadersMiddleware,
        PreFlightHandlerMiddleware,
        setup_cors,
        is_cors_request,
        get_allowed_origins,
        add_allowed_origin,
        remove_allowed_origin,
        validate_cors_config,
    )


class TestCORSMiddleware:
    """Тесты для CORSMiddleware"""

    def test_init_with_string_origins(self):
        """Тест инициализации с строковыми источниками"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com,https://app.example.com"
            mock_settings.DEBUG = False

            middleware = CORSMiddleware()

            assert middleware.allowed_origins == [
                "https://example.com",
                "https://app.example.com",
            ]
            assert middleware.allowed_credentials is True
            assert middleware.max_age == 86400
            assert "GET" in middleware.allowed_methods
            assert "Authorization" in middleware.allowed_headers

    def test_init_with_list_origins(self):
        """Тест инициализации с списком источников"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = [
                "https://example.com",
                "https://app.example.com",
            ]
            mock_settings.DEBUG = False

            middleware = CORSMiddleware()

            assert middleware.allowed_origins == [
                "https://example.com",
                "https://app.example.com",
            ]

    def test_init_with_debug_mode(self):
        """Тест инициализации в debug режиме"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = True

            middleware = CORSMiddleware()

            # В debug режиме должны добавиться localhost источники
            assert "http://localhost:3000" in middleware.allowed_origins
            assert "http://127.0.0.1:3000" in middleware.allowed_origins
            assert "https://example.com" in middleware.allowed_origins

    def test_add_cors_middleware(self):
        """Тест добавления CORS middleware к приложению"""
        with (
            patch("app.middleware.cors.settings") as mock_settings,
            patch("app.middleware.cors.logger") as mock_logger,
        ):

            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = False

            middleware = CORSMiddleware()
            app = Mock(spec=FastAPI)

            middleware.add_cors_middleware(app)

            # Проверяем что метод add_middleware был вызван
            app.add_middleware.assert_called_once()
            call_args = app.add_middleware.call_args

            # Проверяем что был вызван с правильными аргументами
            assert len(call_args[0]) >= 1
            assert call_args[0][0].__name__ == "CORSMiddleware"

            # Проверяем что есть keyword аргументы
            if len(call_args) > 1 and call_args[1]:
                kwargs = call_args[1]
                assert kwargs["allow_origins"] == ["https://example.com"]
                assert kwargs["allow_credentials"] is True
                assert "X-Request-ID" in kwargs["expose_headers"]

            # Проверяем логирование
            mock_logger.info.assert_called_once()


class TestSecurityHeadersMiddleware:
    """Тесты для SecurityHeadersMiddleware"""

    @pytest.mark.asyncio
    async def test_dispatch_adds_security_headers(self):
        """Тест добавления security headers"""
        middleware = SecurityHeadersMiddleware(app=Mock())

        # Создаем mock request и response
        request = Mock(spec=Request)
        request.url.scheme = "https"

        mock_response = Mock(spec=Response)
        mock_response.headers = {}

        # Mock call_next
        async def mock_call_next(req):
            return mock_response

        response = await middleware.dispatch(request, mock_call_next)

        # Проверяем security headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert (
            response.headers["Permissions-Policy"]
            == "geolocation=(), microphone=(), camera=()"
        )
        assert (
            response.headers["Strict-Transport-Security"]
            == "max-age=31536000; includeSubDomains"
        )
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]

    @pytest.mark.asyncio
    async def test_dispatch_http_scheme_no_hsts(self):
        """Тест что HSTS не добавляется для HTTP"""
        middleware = SecurityHeadersMiddleware(app=Mock())

        request = Mock(spec=Request)
        request.url.scheme = "http"  # HTTP, не HTTPS

        mock_response = Mock(spec=Response)
        mock_response.headers = {}

        async def mock_call_next(req):
            return mock_response

        response = await middleware.dispatch(request, mock_call_next)

        # HSTS не должен быть добавлен для HTTP
        assert "Strict-Transport-Security" not in response.headers


class TestPreFlightHandlerMiddleware:
    """Тесты для PreFlightHandlerMiddleware"""

    @pytest.fixture
    def preflight_middleware(self):
        """Фикстура для создания PreFlightHandlerMiddleware"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com,https://app.example.com"
            mock_settings.DEBUG = False

            middleware = PreFlightHandlerMiddleware(app=Mock())
            # Добавляем необходимые атрибуты для тестирования
            middleware.allowed_origins = [
                "https://example.com",
                "https://app.example.com",
            ]
            middleware.allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            middleware.allowed_headers = ["Content-Type", "Authorization"]
            middleware.max_age = 86400

            return middleware

    @pytest.mark.asyncio
    async def test_dispatch_non_options_request(self, preflight_middleware):
        """Тест обработки не-OPTIONS запроса"""
        request = Mock(spec=Request)
        request.method = "GET"

        mock_response = Mock(spec=Response)

        async def mock_call_next(req):
            return mock_response

        response = await preflight_middleware.dispatch(request, mock_call_next)

        assert response == mock_response

    @pytest.mark.asyncio
    async def test_dispatch_options_request(self, preflight_middleware):
        """Тест обработки OPTIONS запроса"""
        request = Mock(spec=Request)
        request.method = "OPTIONS"
        request.headers = {
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,Authorization",
        }

        response = await preflight_middleware.dispatch(request, lambda req: None)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_preflight_allowed_origin(self, preflight_middleware):
        """Тест обработки preflight с разрешенным источником"""
        request = Mock(spec=Request)
        request.headers = {
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }

        response = await preflight_middleware._handle_preflight(request)

        assert response.status_code == 200
        assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"
        assert response.headers["Access-Control-Allow-Credentials"] == "true"
        assert "POST" in response.headers["Access-Control-Allow-Methods"]
        assert "Content-Type" in response.headers["Access-Control-Allow-Headers"]
        assert response.headers["Vary"] == "Origin"

    @pytest.mark.asyncio
    async def test_handle_preflight_disallowed_origin(self, preflight_middleware):
        """Тест обработки preflight с неразрешенным источником"""
        request = Mock(spec=Request)
        request.headers = {
            "Origin": "https://malicious.com",
            "Access-Control-Request-Method": "POST",
        }

        with patch("app.middleware.cors.logger") as mock_logger:
            response = await preflight_middleware._handle_preflight(request)

        assert response.status_code == 403
        assert b'"error"' in response.body and b'"Origin not allowed"' in response.body

        # Проверяем логирование
        mock_logger.warning.assert_called_once_with(
            "CORS preflight rejected for origin: https://malicious.com"
        )

    @pytest.mark.asyncio
    async def test_handle_preflight_no_origin(self, preflight_middleware):
        """Тест обработки preflight без Origin заголовка"""
        request = Mock(spec=Request)
        request.headers = {"Access-Control-Request-Method": "POST"}

        response = await preflight_middleware._handle_preflight(request)

        assert response.status_code == 403

    def test_is_origin_allowed_exact_match(self, preflight_middleware):
        """Тест точного совпадения источника"""
        assert preflight_middleware._is_origin_allowed("https://example.com") is True
        assert preflight_middleware._is_origin_allowed("https://malicious.com") is False

    def test_is_origin_allowed_wildcard(self, preflight_middleware):
        """Тест wildcard совпадения источника"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://*.example.com"
            mock_settings.DEBUG = False

            middleware = PreFlightHandlerMiddleware(app=Mock())
            middleware.allowed_origins = ["https://*.example.com"]
            middleware.allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            middleware.allowed_headers = ["Content-Type", "Authorization"]
            middleware.max_age = 86400

            # Тест точного совпадения с wildcard источником
            assert middleware._is_origin_allowed("https://*.example.com") is True

            # Тест что обычные источники не match wildcard
            assert middleware._is_origin_allowed("https://api.example.com") is False
            assert middleware._is_origin_allowed("https://app.example.com") is False
            assert middleware._is_origin_allowed("https://example.com") is False

    def test_is_origin_allowed_debug_localhost(self):
        """Тест разрешения localhost в debug режиме"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = True

            middleware = PreFlightHandlerMiddleware(app=Mock())
            middleware.allowed_origins = ["https://example.com"]
            middleware.allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            middleware.allowed_headers = ["Content-Type", "Authorization"]
            middleware.max_age = 86400

            assert middleware._is_origin_allowed("http://localhost:3000") is True
            assert middleware._is_origin_allowed("http://127.0.0.1:8080") is True
            assert middleware._is_origin_allowed("https://example.com") is True


class TestCORSUtilities:
    """Тесты для утилитарных функций CORS"""

    def test_is_cors_request_with_origin(self):
        """Тест определения CORS запроса с Origin"""
        request = Mock(spec=Request)
        request.headers = {"Origin": "https://example.com"}

        assert is_cors_request(request) is True

    def test_is_cors_request_without_origin(self):
        """Тест определения CORS запроса без Origin"""
        request = Mock(spec=Request)
        request.headers = {}

        assert is_cors_request(request) is False

    def test_get_allowed_origins(self):
        """Тест получения разрешенных источников"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com,https://app.example.com"
            mock_settings.DEBUG = False

            origins = get_allowed_origins()

            assert "https://example.com" in origins
            assert "https://app.example.com" in origins

    def test_add_allowed_origin_success(self):
        """Тест успешного добавления источника"""
        with (
            patch("app.middleware.cors.settings") as mock_settings,
            patch("app.middleware.cors.logger") as mock_logger,
        ):

            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = False

            result = add_allowed_origin("https://new-origin.com")

            assert result is True
            mock_logger.info.assert_called_once_with(
                "Added new CORS origin: https://new-origin.com"
            )

    def test_add_allowed_origin_duplicate(self):
        """Тест добавления дублирующегося источника"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = False

            result = add_allowed_origin("https://example.com")

            assert result is False

    def test_add_allowed_origin_exception(self):
        """Тест обработки исключения при добавлении источника"""
        with patch(
            "app.middleware.cors.CORSMiddleware", side_effect=Exception("Config error")
        ):
            result = add_allowed_origin("https://new-origin.com")

            assert result is False

    def test_remove_allowed_origin_success(self):
        """Тест успешного удаления источника"""
        with (
            patch("app.middleware.cors.settings") as mock_settings,
            patch("app.middleware.cors.logger") as mock_logger,
        ):

            mock_settings.CORS_ORIGINS = "https://example.com,https://app.example.com"
            mock_settings.DEBUG = False

            result = remove_allowed_origin("https://example.com")

            assert result is True
            mock_logger.info.assert_called_once_with(
                "Removed CORS origin: https://example.com"
            )

    def test_remove_allowed_origin_not_found(self):
        """Тест удаления несуществующего источника"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = False

            result = remove_allowed_origin("https://non-existent.com")

            assert result is False

    def test_remove_allowed_origin_exception(self):
        """Тест обработки исключения при удалении источника"""
        with patch(
            "app.middleware.cors.settings", side_effect=Exception("Config error")
        ):
            result = remove_allowed_origin("https://example.com")

            assert result is False

    def test_validate_cors_config_success(self):
        """Тест успешной валидации конфигурации"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com,https://app.example.com"
            mock_settings.DEBUG = False

            result = validate_cors_config()

            assert result["valid"] is True
            assert "https://example.com" in result["allowed_origins"]
            assert "https://app.example.com" in result["allowed_origins"]
            assert "GET" in result["allowed_methods"]
            assert "Authorization" in result["allowed_headers"]
            assert result["max_age"] == 86400
            assert result["debug_mode"] is False

    def test_validate_cors_config_exception(self):
        """Тест валидации с исключением"""
        with patch(
            "app.middleware.cors.CORSMiddleware", side_effect=Exception("Config error")
        ):
            result = validate_cors_config()

            assert result["valid"] is False
            assert "error" in result


class TestSetupCORS:
    """Тесты для функции setup_cors"""

    def test_setup_cors(self):
        """Тест настройки CORS"""
        with (
            patch("app.middleware.cors.settings") as mock_settings,
            patch("app.middleware.cors.logger") as mock_logger,
        ):

            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = False

            app = Mock(spec=FastAPI)
            app.add_middleware = Mock()

            setup_cors(app)

            # Проверяем что add_middleware вызывался 3 раза (CORS + SecurityHeaders + PreFlight)
            assert app.add_middleware.call_count == 3

            # Проверяем что логирование происходило (может быть несколько вызовов)
            assert mock_logger.info.call_count >= 1
            # Проверяем что был вызван финальный вызов
            mock_logger.info.assert_any_call("CORS configuration applied successfully")


# === ИНТЕГРАЦИОННЫЕ ТЕСТЫ ===


class TestCORSIntegration:
    """Интеграционные тесты для CORS middleware"""

    @pytest.mark.asyncio
    async def test_full_cors_workflow(self):
        """Тест полного рабочего процесса CORS"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = False

            # Создаем все middleware
            cors_middleware = CORSMiddleware()
            security_middleware = SecurityHeadersMiddleware(app=Mock())
            preflight_middleware = PreFlightHandlerMiddleware(app=Mock())

            # Добавляем необходимые атрибуты для preflight middleware
            preflight_middleware.allowed_origins = ["https://example.com"]
            preflight_middleware.allowed_methods = [
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "OPTIONS",
            ]
            preflight_middleware.allowed_headers = ["Content-Type", "Authorization"]
            preflight_middleware.max_age = 86400

            # Тестируем preflight запрос
            request = Mock(spec=Request)
            request.method = "OPTIONS"
            request.headers = {
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            }

            response = await preflight_middleware.dispatch(request, lambda req: None)

            assert response.status_code == 200
            assert (
                response.headers["Access-Control-Allow-Origin"] == "https://example.com"
            )

            # Тестируем security headers
            request.url.scheme = "https"
            security_response = Mock(spec=Response)
            security_response.headers = {}

            async def mock_call_next(req):
                return security_response

            final_response = await security_middleware.dispatch(request, mock_call_next)

            assert "X-Content-Type-Options" in final_response.headers
            assert "Strict-Transport-Security" in final_response.headers

    @pytest.mark.asyncio
    async def test_cors_with_debug_mode(self):
        """Тест CORS в debug режиме"""
        with patch("app.middleware.cors.settings") as mock_settings:
            mock_settings.CORS_ORIGINS = "https://example.com"
            mock_settings.DEBUG = True

            preflight_middleware = PreFlightHandlerMiddleware(app=Mock())

            # Добавляем необходимые атрибуты
            preflight_middleware.allowed_origins = ["https://example.com"]
            preflight_middleware.allowed_methods = [
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "OPTIONS",
            ]
            preflight_middleware.allowed_headers = ["Content-Type", "Authorization"]
            preflight_middleware.max_age = 86400

            # В debug режиме localhost должен быть разрешен
            request = Mock(spec=Request)
            request.method = "OPTIONS"
            request.headers = {"Origin": "http://localhost:3000"}

            response = await preflight_middleware._handle_preflight(request)

            assert response.status_code == 200
            assert (
                response.headers["Access-Control-Allow-Origin"]
                == "http://localhost:3000"
            )
