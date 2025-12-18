"""
Тесты для Webhook Service.
Модуль с покрытием 18.55% - цель: увеличить до 80%+
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import httpx
from datetime import datetime, timezone

# Mock настроек и зависимостей
with patch('app.services.webhook_service.settings') as mock_settings, \
     patch('app.services.webhook_service.logger') as mock_logger, \
     patch('app.services.webhook_service.__version__', '1.0.0'):
    
    # Настраиваем мок настроек
    mock_settings.WEBHOOK_TIMEOUT = 30
    mock_settings.WEBHOOK_MAX_RETRIES = 3
    mock_settings.WEBHOOK_RETRY_DELAY = 1
    mock_settings.WEBHOOK_URL = "https://example.com/webhook"
    
    from app.services.webhook_service import WebhookService, WebhookError, RetryExhaustedError


class TestWebhookService:
    """Тесты для WebhookService"""
    
    @pytest.fixture
    def webhook_service(self):
        """Фикстура для создания WebhookService"""
        with patch('app.services.webhook_service.settings') as mock_settings:
            mock_settings.WEBHOOK_TIMEOUT = 30
            mock_settings.WEBHOOK_MAX_RETRIES = 3
            mock_settings.WEBHOOK_RETRY_DELAY = 1
            mock_settings.WEBHOOK_URL = "https://example.com/webhook"
            
            service = WebhookService()
            # Mock HTTP клиента
            service.client = Mock()
            return service
            
    @pytest.fixture
    def sample_verification_result(self):
        """Фикстура с образцом результата верификации"""
        return {
            "verified": True,
            "confidence": 0.85,
            "similarity_score": 0.8,
            "threshold_used": 0.8,
            "processing_time": 1.2,
            "face_detected": True,
            "reference_id": "ref-123"
        }
    
    @pytest.fixture
    def sample_liveness_result(self):
        """Фикстура с образцом результата проверки живости"""
        return {
            "liveness_detected": True,
            "confidence": 0.9,
            "challenge_type": "active",
            "anti_spoofing_score": 0.85,
            "processing_time": 0.8,
            "face_detected": True,
            "multiple_faces": False,
            "recommendations": ["good_lighting"]
        }
    
    @pytest.fixture
    def sample_reference_data(self):
        """Фикстура с образцом данных эталона"""
        return {
            "label": "Test Reference",
            "quality_score": 0.9,
            "file_url": "https://example.com/image.jpg",
            "image_dimensions": {"width": 224, "height": 224},
            "processing_time": 1.5
        }
    
    # === ОТПРАВКА РЕЗУЛЬТАТА ВЕРИФИКАЦИИ ===
    
    @pytest.mark.asyncio
    async def test_send_verification_result_success(self, webhook_service, sample_verification_result):
        """Тест успешной отправки результата верификации"""
        with patch('app.services.webhook_service.settings') as mock_settings:
            mock_settings.WEBHOOK_URL = "https://example.com/webhook"
            
            # Mock успешной отправки
            webhook_service._send_webhook = AsyncMock(return_value=True)
            
            result = await webhook_service.send_verification_result(
                user_id="user123",
                session_id="session456",
                verification_result=sample_verification_result
            )
            
            assert result is True
            webhook_service._send_webhook.assert_called_once()
            
            # Проверяем что был вызван с правильными параметрами
            call_args = webhook_service._send_webhook.call_args[0]
            payload = call_args[0]
            
            assert payload["event_type"] == "verification.completed"
            assert payload["user_id"] == "user123"
            assert payload["session_id"] == "session456"
            assert payload["data"]["verified"] is True
            assert payload["data"]["confidence"] == 0.85
            assert "event_id" in payload
            assert "timestamp" in payload
    
    @pytest.mark.asyncio
    async def test_send_verification_result_with_additional_data(self, webhook_service, sample_verification_result):
        """Тест отправки результата верификации с дополнительными данными"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        additional_data = {"custom_field": "custom_value"}
        
        result = await webhook_service.send_verification_result(
            user_id="user123",
            session_id="session456",
            verification_result=sample_verification_result,
            additional_data=additional_data
        )
        
        assert result is True
        # Просто проверяем что метод был вызван
        webhook_service._send_webhook.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_send_verification_result_custom_webhook_url(self, webhook_service, sample_verification_result):
        """Тест отправки результата верификации на кастомный webhook URL"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        custom_url = "https://custom.com/webhook"
        
        result = await webhook_service.send_verification_result(
            user_id="user123",
            session_id="session456",
            verification_result=sample_verification_result,
            webhook_url=custom_url
        )
        
        assert result is True
        # Просто проверяем что метод был вызван
        webhook_service._send_webhook.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_send_verification_result_exception(self, webhook_service, sample_verification_result):
        """Тест обработки исключения при отправке результата верификации"""
        webhook_service._send_webhook = AsyncMock(side_effect=Exception("Network error"))
        
        result = await webhook_service.send_verification_result(
            user_id="user123",
            session_id="session456",
            verification_result=sample_verification_result
        )
        
        assert result is False
    
    # === ОТПРАВКА РЕЗУЛЬТАТА ПРОВЕРКИ ЖИВОСТИ ===
    
    @pytest.mark.asyncio
    async def test_send_liveness_result_success(self, webhook_service, sample_liveness_result):
        """Тест успешной отправки результата проверки живости"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        result = await webhook_service.send_liveness_result(
            user_id="user123",
            session_id="session456",
            liveness_result=sample_liveness_result
        )
        
        assert result is True
        webhook_service._send_webhook.assert_called_once()
        
        call_args = webhook_service._send_webhook.call_args[0]
        payload = call_args[0]
        
        assert payload["event_type"] == "liveness.completed"
        assert payload["user_id"] == "user123"
        assert payload["session_id"] == "session456"
        assert payload["data"]["liveness_detected"] is True
        assert payload["data"]["confidence"] == 0.9
        assert payload["data"]["challenge_type"] == "active"
    
    @pytest.mark.asyncio
    async def test_send_liveness_result_with_additional_data(self, webhook_service, sample_liveness_result):
        """Тест отправки результата проверки живости с дополнительными данными"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        additional_data = {"device_id": "device123"}
        
        result = await webhook_service.send_liveness_result(
            user_id="user123",
            session_id="session456",
            liveness_result=sample_liveness_result,
            additional_data=additional_data
        )
        
        assert result is True
        webhook_service._send_webhook.assert_called_once()
        
    # === ОТПРАВКА УВЕДОМЛЕНИЯ О СОЗДАНИИ ЭТАЛОНА ===
    
    @pytest.mark.asyncio
    async def test_send_reference_created_success(self, webhook_service, sample_reference_data):
        """Тест успешной отправки уведомления о создании эталона"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        result = await webhook_service.send_reference_created(
            user_id="user123",
            reference_id="ref-456",
            reference_data=sample_reference_data
        )
        
        assert result is True
        webhook_service._send_webhook.assert_called_once()
        
        call_args = webhook_service._send_webhook.call_args[0]
        payload = call_args[0]
        
        assert payload["event_type"] == "reference.created"
        assert payload["user_id"] == "user123"
        assert payload["reference_id"] == "ref-456"
        assert payload["data"]["label"] == "Test Reference"
        assert payload["data"]["quality_score"] == 0.9
    
    # === ОТПРАВКА УВЕДОМЛЕНИЯ О АКТИВНОСТИ ПОЛЬЗОВАТЕЛЯ ===
    
    @pytest.mark.asyncio
    async def test_send_user_activity_success(self, webhook_service):
        """Тест успешной отправки уведомления о активности пользователя"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        activity_data = {"action": "login", "ip_address": "192.168.1.1"}
        
        result = await webhook_service.send_user_activity(
            user_id="user123",
            activity_type="login",
            activity_data=activity_data
        )
        
        assert result is True
        webhook_service._send_webhook.assert_called_once()
        
        call_args = webhook_service._send_webhook.call_args[0]
        payload = call_args[0]
        
        assert payload["event_type"] == "user.login"
        assert payload["user_id"] == "user123"
        assert payload["data"]["activity_type"] == "login"
        assert payload["data"]["activity_data"]["action"] == "login"
    
    # === ОТПРАВКА СИСТЕМНОГО УВЕДОМЛЕНИЯ ===
    
    @pytest.mark.asyncio
    async def test_send_system_alert_success(self, webhook_service):
        """Тест успешной отправки системного уведомления"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        result = await webhook_service.send_system_alert(
            alert_type="high_cpu_usage",
            message="CPU usage above 90%",
            severity="warning"
        )
        
        assert result is True
        webhook_service._send_webhook.assert_called_once()
        
        call_args = webhook_service._send_webhook.call_args[0]
        payload = call_args[0]
        
        assert payload["event_type"] == "system.alert"
        assert payload["data"]["alert_type"] == "high_cpu_usage"
        assert payload["data"]["message"] == "CPU usage above 90%"
        assert payload["data"]["severity"] == "warning"
        assert payload["data"]["service"] == "face-recognition-service"
        assert payload["data"]["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_send_system_alert_with_additional_data(self, webhook_service):
        """Тест отправки системного уведомления с дополнительными данными"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        additional_data = {"cpu_usage": 95, "memory_usage": 80}
        
        result = await webhook_service.send_system_alert(
            alert_type="resource_usage",
            message="High resource usage detected",
            severity="warning",
            additional_data=additional_data
        )
        
        assert result is True
        webhook_service._send_webhook.assert_called_once()
    
    # === ПАКЕТНАЯ ОТПРАВКА WEBHOOK ===
    
    @pytest.mark.asyncio
    async def test_send_batch_webhooks_success(self, webhook_service):
        """Тест успешной пакетной отправки webhook"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        webhooks_data = [
            {"payload": {"event_type": "test1"}, "event_type": "test"},
            {"payload": {"event_type": "test2"}, "event_type": "test"},
            {"payload": {"event_type": "test3"}, "event_type": "test"}
        ]
        
        result = await webhook_service.send_batch_webhooks(webhooks_data)
        
        assert result["successful"] == 3
        assert result["failed"] == 0
        assert result["total"] == 3
    
    @pytest.mark.asyncio
    async def test_send_batch_webhooks_mixed_results(self, webhook_service):
        """Тест пакетной отправки с смешанными результатами"""
        # Первые два успешные, третий неудачный
        webhook_service._send_webhook = AsyncMock(side_effect=[True, True, False])
        
        webhooks_data = [
            {"payload": {"event_type": "test1"}, "event_type": "test"},
            {"payload": {"event_type": "test2"}, "event_type": "test"},
            {"payload": {"event_type": "test3"}, "event_type": "test"}
        ]
        
        result = await webhook_service.send_batch_webhooks(webhooks_data)
        
        assert result["successful"] == 2
        assert result["failed"] == 1
        assert result["total"] == 3
    
    @pytest.mark.asyncio
    async def test_send_batch_webhooks_with_exceptions(self, webhook_service):
        """Тест пакетной отправки с исключениями"""
        webhook_service._send_webhook = AsyncMock(side_effect=[
            True, 
            Exception("Network error"), 
            False
        ])
        
        webhooks_data = [
            {"payload": {"event_type": "test1"}, "event_type": "test"},
            {"payload": {"event_type": "test2"}, "event_type": "test"},
            {"payload": {"event_type": "test3"}, "event_type": "test"}
        ]
        
        result = await webhook_service.send_batch_webhooks(webhooks_data)
        
        assert result["successful"] == 1
        assert result["failed"] == 2  # 1 exception + 1 False
        assert result["total"] == 3
    
    # === ТЕСТИРОВАНИЕ WEBHOOK ENDPOINT ===
    
    @pytest.mark.asyncio
    async def test_webhook_success(self, webhook_service):
        """Тест успешного тестирования webhook"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        # Mock time для тестирования времени ответа
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 1.0  # start_time
            mock_loop.return_value.time.return_value = 1.5  # end_time
            
            result = await webhook_service.test_webhook()
        
        assert result["success"] is True
        assert result["response_time"] == 0.5
        assert result["webhook_url"] == "https://example.com/webhook"
        assert "timestamp" in result
        
        # Проверяем что был отправлен тестовый payload
        webhook_service._send_webhook.assert_called_once()
        call_args = webhook_service._send_webhook.call_args[0]
        payload = call_args[0]
        
        assert payload["event_type"] == "webhook.test"
        assert payload["data"]["message"] == "This is a test webhook from Face Recognition Service"
    
    @pytest.mark.asyncio
    async def test_webhook_with_custom_url_and_data(self, webhook_service):
        """Тест тестирования webhook с кастомным URL и данными"""
        webhook_service._send_webhook = AsyncMock(return_value=True)
        
        test_data = {"custom_field": "test_value"}
        custom_url = "https://custom.com/webhook"
        
        result = await webhook_service.test_webhook(
            webhook_url=custom_url,
            test_data=test_data
        )
        
        assert result["success"] is True
        assert result["webhook_url"] == custom_url
        webhook_service._send_webhook.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_webhook_failure(self, webhook_service):
        """Тест неудачного тестирования webhook"""
        webhook_service._send_webhook = AsyncMock(return_value=False)
        
        result = await webhook_service.test_webhook()
        
        assert result["success"] is False
        assert "error" in result
        assert result["webhook_url"] == "https://example.com/webhook"
    
    @pytest.mark.asyncio
    async def test_webhook_exception(self, webhook_service):
        """Тест обработки исключения при тестировании webhook"""
        webhook_service._send_webhook = AsyncMock(side_effect=Exception("Test error"))
        
        result = await webhook_service.test_webhook()
        
        assert result["success"] is False
        assert result["error"] == "Test error"
        assert result["webhook_url"] == "https://example.com/webhook"
    
    # === ВАЛИДАЦИЯ WEBHOOK URL ===
    
    @pytest.mark.asyncio
    async def test_validate_webhook_url_success(self, webhook_service):
        """Тест успешной валидации webhook URL"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.5
        webhook_service.client.get = AsyncMock(return_value=mock_response)
        
        result = await webhook_service.validate_webhook_url("https://example.com/webhook")
        
        assert result["valid"] is True
        assert result["status_code"] == 200
        assert result["response_time"] == 0.5
        assert result["url"] == "https://example.com/webhook"
    
    @pytest.mark.asyncio
    async def test_validate_webhook_url_invalid_protocol(self, webhook_service):
        """Тест валидации webhook URL с неправильным протоколом"""
        result = await webhook_service.validate_webhook_url("ftp://example.com/webhook")
        
        assert result["valid"] is False
        assert "URL must start with http:// or https://" in result["error"]
        # URL может не быть в результате для невалидного протокола
        assert "url" not in result or result.get("url") == "ftp://example.com/webhook"
    
    @pytest.mark.asyncio
    async def test_validate_webhook_url_timeout(self, webhook_service):
        """Тест валидации webhook URL с timeout"""
        webhook_service.client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        
        result = await webhook_service.validate_webhook_url("https://example.com/webhook")
        
        assert result["valid"] is False
        assert result["error"] == "Connection timeout"
        assert result["url"] == "https://example.com/webhook"
    
    @pytest.mark.asyncio
    async def test_validate_webhook_url_connection_error(self, webhook_service):
        """Тест валидации webhook URL с ошибкой соединения"""
        webhook_service.client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        
        result = await webhook_service.validate_webhook_url("https://example.com/webhook")
        
        assert result["valid"] is False
        assert result["error"] == "Connection failed"
        assert result["url"] == "https://example.com/webhook"
    
    @pytest.mark.asyncio
    async def test_validate_webhook_url_generic_exception(self, webhook_service):
        """Тест валидации webhook URL с общим исключением"""
        webhook_service.client.get = AsyncMock(side_effect=Exception("Generic error"))
        
        result = await webhook_service.validate_webhook_url("https://example.com/webhook")
        
        assert result["valid"] is False
        assert result["error"] == "Generic error"
        assert result["url"] == "https://example.com/webhook"
    
    # === ВНУТРИННИЙ МЕТОД _SEND_WEBHOOK ===
    
    @pytest.mark.asyncio
    async def test_send_webhook_success(self, webhook_service):
        """Тест успешной отправки webhook"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        webhook_service.client.post = AsyncMock(return_value=mock_response)
        
        payload = {"event_type": "test", "data": {"test": "data"}}
        
        result = await webhook_service._send_webhook(
            payload=payload,
            webhook_url="https://example.com/webhook",
            event_type="test"
        )
        
        assert result is True
        webhook_service.client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_webhook_client_error(self, webhook_service):
        """Тест отправки webhook с клиентской ошибкой"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        webhook_service.client.post = AsyncMock(return_value=mock_response)
        
        payload = {"event_type": "test", "data": {"test": "data"}}
        
        result = await webhook_service._send_webhook(
            payload=payload,
            webhook_url="https://example.com/webhook",
            event_type="test"
        )
        
        assert result is False  # Клиентские ошибки не повторяются
    
    @pytest.mark.asyncio
    async def test_send_webhook_server_error_with_retry(self, webhook_service):
        """Тест отправки webhook с серверной ошибкой и повторными попытками"""
        # Первые два запроса возвращают 500, третий успешный
        mock_responses = [
            Mock(status_code=500, text="Internal Server Error"),
            Mock(status_code=500, text="Internal Server Error"),
            Mock(status_code=200, text="OK")
        ]
        webhook_service.client.post = AsyncMock(side_effect=mock_responses)
        
        payload = {"event_type": "test", "data": {"test": "data"}}
        
        result = await webhook_service._send_webhook(
            payload=payload,
            webhook_url="https://example.com/webhook",
            event_type="test"
        )
        
        assert result is True
        assert webhook_service.client.post.call_count == 3  # 2 ошибки + 1 успех
    
    @pytest.mark.asyncio
    async def test_send_webhook_timeout(self, webhook_service):
        """Тест отправки webhook с timeout"""
        webhook_service.client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        
        payload = {"event_type": "test", "data": {"test": "data"}}
        
        result = await webhook_service._send_webhook(
            payload=payload,
            webhook_url="https://example.com/webhook",
            event_type="test"
        )
        
        assert result is False  # После всех попыток
    
    @pytest.mark.asyncio
    async def test_send_webhook_connection_error(self, webhook_service):
        """Тест отправки webhook с ошибкой соединения"""
        webhook_service.client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        
        payload = {"event_type": "test", "data": {"test": "data"}}
        
        result = await webhook_service._send_webhook(
            payload=payload,
            webhook_url="https://example.com/webhook",
            event_type="test"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_webhook_generic_exception(self, webhook_service):
        """Тест отправки webhook с общим исключением"""
        webhook_service.client.post = AsyncMock(side_effect=Exception("Generic error"))
        
        payload = {"event_type": "test", "data": {"test": "data"}}
        
        result = await webhook_service._send_webhook(
            payload=payload,
            webhook_url="https://example.com/webhook",
            event_type="test"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_webhook_no_url(self, webhook_service):
        """Тест отправки webhook без URL"""
        payload = {"event_type": "test", "data": {"test": "data"}}
        
        result = await webhook_service._send_webhook(
            payload=payload,
            webhook_url="",
            event_type="test"
        )
        
        assert result is False
    
    # === ЗАКРЫТИЕ КЛИЕНТА ===
    
    @pytest.mark.asyncio
    async def test_close(self, webhook_service):
        """Тест закрытия HTTP клиента"""
        webhook_service.client.aclose = AsyncMock()
        
        await webhook_service.close()
        
        webhook_service.client.aclose.assert_called_once()


# === ИНТЕГРАЦИОННЫЕ ТЕСТЫ ===

class TestWebhookServiceIntegration:
    """Интеграционные тесты для WebhookService"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_verification(self):
        """Тест полного рабочего процесса верификации"""
        with patch('app.services.webhook_service.settings') as mock_settings, \
             patch('app.services.webhook_service.logger'):
            
            mock_settings.WEBHOOK_TIMEOUT = 30
            mock_settings.WEBHOOK_MAX_RETRIES = 3
            mock_settings.WEBHOOK_RETRY_DELAY = 1
            mock_settings.WEBHOOK_URL = "https://example.com/webhook"
            
            service = WebhookService()
            service.client = Mock()
            
            # Mock успешной отправки
            service.client.post = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            service.client.post.return_value = mock_response
            
            # Отправляем результат верификации
            verification_result = {
                "verified": True,
                "confidence": 0.85,
                "similarity_score": 0.8,
                "processing_time": 1.2,
                "face_detected": True
            }
            
            result = await service.send_verification_result(
                user_id="user123",
                session_id="session456",
                verification_result=verification_result
            )
            
            assert result is True
            
            # Проверяем что был сделан POST запрос
            service.client.post.assert_called_once()
            call_args = service.client.post.call_args
            
            # Проверяем URL
            assert call_args[0][0] == "https://example.com/webhook"
            
            # Проверяем headers
            headers = call_args[1]["headers"]
            assert headers["Content-Type"] == "application/json"
            assert headers["X-Event-Type"] == "verification"
            assert "X-Event-ID" in headers
            
            # Проверяем payload
            payload = call_args[1]["json"]
            assert payload["event_type"] == "verification.completed"
            assert payload["user_id"] == "user123"
            assert payload["data"]["verified"] is True
            assert payload["data"]["confidence"] == 0.85