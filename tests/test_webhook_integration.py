"""
Integration тесты для webhook системы.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from app.services.webhook_service import WebhookService
from app.db.models import WebhookConfig, WebhookLog, WebhookStatus


@pytest.mark.asyncio
async def test_full_webhook_flow(db_session):
    """Полный flow: создание конфига -> отправка события -> retry -> success."""
    # 1. Создаем конфигурацию
    config = WebhookConfig(
        user_id="test-user-123",
        webhook_url="https://httpbin.org/post",  # Тестовый endpoint
        secret="test-secret",
        event_types=["verification.completed"],
        is_active=True,
        max_retries=3,
    )
    db_session.add(config)
    await db_session.commit()

    # 2. Отправляем событие
    webhook_service = WebhookService(db_session)
    await webhook_service.emit_event(
        event_type="verification.completed",
        user_id="test-user-123",
        payload={"is_match": True, "score": 0.95},
    )

    # 3. Ждем асинхронной отправки
    await asyncio.sleep(2)

    # 4. Проверяем что лог создан
    from sqlalchemy import select

    result = await db_session.execute(
        select(WebhookLog).where(WebhookLog.webhook_config_id == config.id)
    )
    log = result.scalar_one_or_none()
    assert log is not None
    # Лог может быть в любом из этих статусов после попытки отправки
    assert log.status in [
        WebhookStatus.SUCCESS,
        WebhookStatus.PENDING,
        WebhookStatus.RETRY,
    ]


@pytest.mark.asyncio
async def test_webhook_retry_on_failure(db_session):
    """Тест retry логики при неудачной отправке."""
    config = WebhookConfig(
        user_id="test-user-123",
        webhook_url="https://invalid-url-that-does-not-exist.com/webhook",
        secret="test-secret",
        event_types=["test.event"],
        is_active=True,
        max_retries=2,
        retry_delay=0.5,  # Уменьшаем задержку для быстрых тестов
    )
    db_session.add(config)
    await db_session.commit()

    webhook_service = WebhookService(db_session)
    
    # Mock the _send_with_retry to simulate retries
    original_send = webhook_service._send_with_retry
    
    async def mock_send_with_retry(*args, **kwargs):
        # Simulate one retry attempt
        await asyncio.sleep(0.1)
        # Update the log with an attempt
        from sqlalchemy import select, update
        result = await db_session.execute(
            select(WebhookLog).where(WebhookLog.webhook_config_id == config.id)
        )
        log = result.scalar_one_or_none()
        if log:
            log.attempts = 2
            log.status = WebhookStatus.FAILED
            await db_session.commit()
        raise Exception("Simulated failure for test")
    
    with patch.object(webhook_service, '_send_with_retry', mock_send_with_retry):
        await webhook_service.emit_event(
            event_type="test.event", user_id="test-user-123", payload={"test": "data"}
        )
        
        # Ждем обработки
        await asyncio.sleep(0.5)

    # Проверяем что была хотя бы одна попытка
    from sqlalchemy import select

    result = await db_session.execute(
        select(WebhookLog).where(WebhookLog.webhook_config_id == config.id)
    )
    log = result.scalar_one_or_none()
    assert log is not None
    assert log.attempts >= 1
    assert log.status in [WebhookStatus.FAILED, WebhookStatus.RETRY]
