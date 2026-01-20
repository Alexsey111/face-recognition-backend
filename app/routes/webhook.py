"""
Routes для управления webhook конфигурациями.
"""

import asyncio
import uuid
import ipaddress
import re
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from ..db.database import get_async_db
from ..db.models import WebhookConfig, WebhookLog, WebhookEventType, WebhookStatus, User
from ..models.webhook import (
    WebhookConfigCreate,
    WebhookConfigUpdate, 
    WebhookConfig as WebhookConfigResponse,
    WebhookLog as WebhookLogResponse,
    WebhookTestRequest,
    WebhookTestResponse,
    WebhookStatistics,
    WebhookRetryRequest,
    WebhookBulkAction
)
from ..services.webhook_service import WebhookService
from ..utils.logger import get_logger
from ..routes.auth import get_current_user
from ..models.user import UserModel

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])


# Helper functions for model mapping
def to_webhook_config_response(config: WebhookConfig) -> WebhookConfigResponse:
    """Helper function to map WebhookConfig model to response model."""
    return WebhookConfigResponse(
        id=config.id,
        user_id=config.user_id,
        webhook_url=config.webhook_url,
        event_types=[WebhookEventType(event_type) for event_type in config.event_types],
        is_active=config.is_active,
        timeout=config.timeout,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
        created_at=config.created_at,
        updated_at=config.updated_at
    )


def to_webhook_log_response(log: WebhookLog, config: WebhookConfig) -> WebhookLogResponse:
    """Helper function to map WebhookLog model to response model."""
    return WebhookLogResponse(
        id=log.id,
        webhook_config_id=log.webhook_config_id,
        event_type=WebhookEventType(log.event_type),
        payload=log.payload,
        payload_hash=log.payload_hash,
        attempts=log.attempts,
        last_attempt_at=log.last_attempt_at,
        next_retry_at=log.next_retry_at,
        status=WebhookStatus(log.status),
        http_status=log.http_status,
        response_body=log.response_body,
        error_message=log.error_message,
        signature=log.signature,
        created_at=log.created_at
    )


# Security validation functions
def validate_webhook_url_security(webhook_url: str) -> tuple[bool, str]:
    """
    Validate webhook URL for security issues.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check URL format
        if not webhook_url.startswith(('http://', 'https://')):
            return False, "URL must start with http:// or https://"
        
        # Check URL length
        if len(webhook_url) > 2048:
            return False, "URL too long (max 2048 characters)"
        
        # Parse URL
        from urllib.parse import urlparse
        parsed = urlparse(webhook_url)
        
        # Check for localhost and private IP ranges
        hostname = parsed.hostname.lower()
        
        # Block localhost variations
        localhost_hosts = {'localhost', 'localhost.localdomain', 'localhost6.localdomain6'}
        if hostname in localhost_hosts:
            return False, "Localhost URLs are not allowed"
        
        # Check for private IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return False, "Private/loopback IP addresses are not allowed"
        except ValueError:
            # Not an IP address, might be a domain name
            pass
        
        # Check for suspicious patterns
        if re.search(r'[\x00-\x1f\x7f]', webhook_url):
            return False, "URL contains control characters"
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


@router.post("/configs", response_model=WebhookConfigResponse)
async def create_webhook_config(
    config_data: WebhookConfigCreate,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Создание новой конфигурации webhook.
    
    - **user_id**: ID пользователя (должен совпадать с текущим пользователем)
    - **webhook_url**: URL endpoint для отправки webhook
    - **secret**: Секретный ключ для HMAC подписи
    - **event_types**: Типы событий для отправки
    - **timeout**: Таймаут запроса в секундах (по умолчанию 10)
    - **max_retries**: Максимальное количество попыток (по умолчанию 3)
    - **retry_delay**: Базовая задержка retry в секундах (по умолчанию 1)
    """
    try:
        # Проверяем, что пользователь может создавать webhook для себя
        if config_data.user_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="You can only create webhook configurations for your own user ID"
            )
        
        # Проверяем безопасность webhook URL
        is_secure, security_error = validate_webhook_url_security(config_data.webhook_url)
        if not is_secure:
            raise HTTPException(
                status_code=400,
                detail=f"Webhook URL security validation failed: {security_error}"
            )
        
        # Проверяем валидность webhook URL через сервис
        webhook_service = WebhookService(db)
        validation_result = await webhook_service.validate_webhook_url(config_data.webhook_url)
        
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid webhook URL: {validation_result.get('error', 'Unknown error')}"
            )
        
        # Создаем конфигурацию
        webhook_config = WebhookConfig(
            user_id=config_data.user_id,
            webhook_url=config_data.webhook_url,
            secret=config_data.secret,
            event_types=[event_type.value for event_type in config_data.event_types],
            is_active=config_data.is_active,
            timeout=config_data.timeout,
            max_retries=config_data.max_retries,
            retry_delay=config_data.retry_delay
        )
        
        db.add(webhook_config)
        await db.commit()
        await db.refresh(webhook_config)
        
        logger.info(f"Webhook config created for user {current_user.id}: {webhook_config.id}")
        
        return to_webhook_config_response(webhook_config)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating webhook config: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/configs", response_model=List[WebhookConfigResponse])
async def get_webhook_configs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Получение списка webhook конфигураций.

    Пользователи могут видеть только свои конфигурации, админы - все.
    """
    try:
        # Формируем базовый запрос
        query = select(WebhookConfig)
        
        # Применяем фильтры
        if current_user.role != "admin":
            # Обычные пользователи видят только свои конфигурации
            query = query.where(WebhookConfig.user_id == current_user.id)
        elif user_id:
            # Админы могут фильтровать по user_id
            query = query.where(WebhookConfig.user_id == user_id)
        
        if is_active is not None:
            query = query.where(WebhookConfig.is_active == is_active)
        
        # Добавляем сортировку и пагинацию
        query = query.order_by(desc(WebhookConfig.created_at))
        query = query.offset((page - 1) * per_page).limit(per_page)
        
        result = await db.execute(query)
        configs = result.scalars().all()
        
        # Формируем ответ с помощью helper функции
        return [to_webhook_config_response(config) for config in configs]
        
    except Exception as e:
        logger.error(f"Error getting webhook configs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/configs/{config_id}", response_model=WebhookConfigResponse)
async def get_webhook_config(
    config_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Получение webhook конфигурации по ID."""
    try:
        result = await db.execute(select(WebhookConfig).where(WebhookConfig.id == config_id))
        config = result.scalar_one_or_none()
        
        if not config:
            raise HTTPException(status_code=404, detail="Webhook config not found")
        
        # Проверяем права доступа
        if current_user.role != "admin" and config.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return to_webhook_config_response(config)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting webhook config {config_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/configs/{config_id}", response_model=WebhookConfigResponse)
async def update_webhook_config(
    config_id: str,
    config_update: WebhookConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Обновление webhook конфигурации."""
    try:
        result = await db.execute(select(WebhookConfig).where(WebhookConfig.id == config_id))
        config = result.scalar_one_or_none()
        
        if not config:
            raise HTTPException(status_code=404, detail="Webhook config not found")
        
        # Проверяем права доступа
        if current_user.role != "admin" and config.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Обновляем поля
        update_data = config_update.dict(exclude_unset=True)
        
        if "event_types" in update_data:
            update_data["event_types"] = [event_type.value for event_type in update_data["event_types"]]
        
        for field, value in update_data.items():
            setattr(config, field, value)
        
        await db.commit()
        await db.refresh(config)
        
        logger.info(f"Webhook config updated: {config_id}")
        
        return to_webhook_config_response(config)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating webhook config {config_id}: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/configs/{config_id}")
async def delete_webhook_config(
    config_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Удаление webhook конфигурации."""
    try:
        result = await db.execute(select(WebhookConfig).where(WebhookConfig.id == config_id))
        config = result.scalar_one_or_none()
        
        if not config:
            raise HTTPException(status_code=404, detail="Webhook config not found")
        
        # Проверяем права доступа
        if current_user.role != "admin" and config.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Удаляем конфигурацию (каскадное удаление логов)
        await db.delete(config)
        await db.commit()
        
        logger.info(f"Webhook config deleted: {config_id}")
        
        return {"message": "Webhook config deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting webhook config {config_id}: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/test", response_model=WebhookTestResponse)
async def test_webhook(
    test_request: WebhookTestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Тестирование webhook endpoint.
    
    Если webhook_url не указан, используется активная конфигурация пользователя.
    """
    try:
        webhook_service = WebhookService(db)
        
        # Если URL не указан, ищем активную конфигурацию
        webhook_url = test_request.webhook_url
        if not webhook_url:
            result = await db.execute(
                select(WebhookConfig).where(
                    and_(
                        WebhookConfig.user_id == current_user.id,
                        WebhookConfig.is_active == True,
                        WebhookConfig.event_types.contains([WebhookEventType.WEBHOOK_TEST.value])
                    )
                ).order_by(desc(WebhookConfig.created_at)).limit(1)
            )
            config = result.scalar_one_or_none()
            
            if not config:
                raise HTTPException(
                    status_code=404, 
                    detail="No active webhook configuration found for testing"
                )
            
            webhook_url = config.webhook_url
            secret = config.secret
        else:
            # Проверяем безопасность указанного URL
            is_secure, security_error = validate_webhook_url_security(webhook_url)
            if not is_secure:
                raise HTTPException(
                    status_code=400,
                    detail=f"Webhook URL security validation failed: {security_error}"
                )
            
            # Используем тестовый секрет для указанного URL
            secret = "test-secret-key"
        
        # Создаем тестовый payload
        payload = webhook_service.create_webhook_payload(
            event_type=test_request.event_type.value,
            data={
                "message": "This is a test webhook",
                "user_id": current_user.id,
                **(test_request.custom_data or {})
            },
            user_id=current_user.id
        )
        
        # Вычисляем подпись
        signature = webhook_service.compute_hmac_signature(payload, secret)
        
        # Добавляем подпись в payload
        payload["signature"] = signature
        
        # Отправляем тестовый webhook
        success = await webhook_service._send_webhook(
            payload=payload,
            webhook_url=webhook_url,
            event_type="test"
        )
        
        return WebhookTestResponse(
            success=success,
            webhook_url=webhook_url,
            response_time=0.0,  # Будет заполнено в реальном запросе
            status_code=200 if success else 500,
            error=None if success else "Test webhook failed",
            timestamp=datetime.now(timezone.utc).isoformat(),
            signature=signature
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing webhook: {str(e)}")
        return WebhookTestResponse(
            success=False,
            webhook_url=test_request.webhook_url or "default",
            response_time=0.0,
            error=str(e),
            timestamp=datetime.now(timezone.utc).isoformat()
        )


@router.get("/logs", response_model=List[WebhookLogResponse])
async def get_webhook_logs(
    config_id: Optional[str] = Query(None, description="Filter by config ID"),
    event_type: Optional[WebhookEventType] = Query(None, description="Filter by event type"),
    status: Optional[WebhookStatus] = Query(None, description="Filter by status"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Получение логов webhook.
    
    Пользователи могут видеть только логи своих webhook, админы - все.
    """
    try:
        # Формируем базовый запрос с join к конфигурации
        query = select(WebhookLog, WebhookConfig).join(
            WebhookConfig, WebhookConfig.id == WebhookLog.webhook_config_id
        )
        
        # Применяем фильтры
        if current_user.role != "admin":
            # Обычные пользователи видят только свои логи
            query = query.where(WebhookConfig.user_id == current_user.id)
        elif user_id:
            # Админы могут фильтровать по user_id
            query = query.where(WebhookConfig.user_id == user_id)
        
        if config_id:
            query = query.where(WebhookLog.webhook_config_id == config_id)
        
        if event_type:
            query = query.where(WebhookLog.event_type == event_type.value)
        
        if status:
            query = query.where(WebhookLog.status == status)
        
        # Добавляем сортировку и пагинацию
        query = query.order_by(desc(WebhookLog.created_at))
        query = query.offset((page - 1) * per_page).limit(per_page)
        
        result = await db.execute(query)
        logs = result.all()
        
        # Формируем ответ с помощью helper функции
        return [to_webhook_log_response(log, config) for log, config in logs]
        
    except Exception as e:
        logger.error(f"Error getting webhook logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/statistics", response_model=WebhookStatistics)
async def get_webhook_statistics(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Получение статистики webhook."""
    try:
        # Формируем базовый запрос
        query = select(WebhookLog, WebhookConfig).join(
            WebhookConfig, WebhookConfig.id == WebhookLog.webhook_config_id
        )

        # Применяем фильтры
        if current_user.role != "admin":
            query = query.where(WebhookConfig.user_id == current_user.id)
        elif user_id:
            query = query.where(WebhookConfig.user_id == user_id)
        
        result = await db.execute(query)
        logs = result.all()
        
        # Вычисляем статистику
        total_webhooks = len(logs)
        successful_webhooks = len([log for log, config in logs if log.status == WebhookStatus.SUCCESS])
        failed_webhooks = len([log for log, config in logs if log.status == WebhookStatus.FAILED])
        pending_webhooks = len([log for log, config in logs if log.status == WebhookStatus.PENDING])
        retry_webhooks = len([log for log, config in logs if log.status == WebhookStatus.RETRY])
        
        success_rate = (successful_webhooks / total_webhooks * 100) if total_webhooks > 0 else 0
        
        # Вычисляем среднее время ответа
        response_times = [log.processing_time for log, config in logs if log.processing_time]
        average_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Находим время последнего webhook
        last_webhook = max([log.created_at for log, config in logs]) if logs else None
        
        return WebhookStatistics(
            total_webhooks=total_webhooks,
            successful_webhooks=successful_webhooks,
            failed_webhooks=failed_webhooks,
            pending_webhooks=pending_webhooks,
            retry_webhooks=retry_webhooks,
            success_rate=success_rate,
            average_response_time=average_response_time,
            last_webhook_at=last_webhook
        )
        
    except Exception as e:
        logger.error(f"Error getting webhook statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/retry/{log_id}")
async def retry_webhook(
    log_id: str,
    retry_request: WebhookRetryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Повторная отправка webhook."""
    try:
        result = await db.execute(
            select(WebhookLog, WebhookConfig).join(
                WebhookConfig, WebhookConfig.id == WebhookLog.webhook_config_id
            ).where(WebhookLog.id == log_id)
        )
        log_data = result.one_or_none()
        
        if not log_data:
            raise HTTPException(status_code=404, detail="Webhook log not found")
        
        log, config = log_data
        
        # Проверяем права доступа
        if current_user.role != "admin" and config.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Проверяем статус webhook
        if log.status not in [WebhookStatus.FAILED, WebhookStatus.EXPIRED] and not retry_request.force:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot retry webhook with status {log.status}"
            )
        
        # Обновляем статус и планируем отправку
        # NOTE: Использование внутреннего метода _send_webhook_with_config осознанное,
        # так как это часть внутреннего API сервиса для ручного retry
        log.status = WebhookStatus.PENDING
        log.attempts = 0  # Сбрасываем счетчик для новой попытки
        log.error_message = None
        log.next_retry_at = None
        
        await db.commit()
        
        # Асинхронно отправляем webhook
        from ..services.webhook_service import WebhookService
        webhook_service = WebhookService(db)
        
        # Запускаем отправку в фоне, чтобы не блокировать ответ
        asyncio.create_task(
            webhook_service._send_webhook_with_config(log.payload, config, log.id)
        )
        
        logger.info(f"Webhook {log_id} queued for manual retry")
        
        return {"message": "Webhook queued for retry", "log_id": log_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying webhook {log_id}: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/bulk-action")
async def bulk_webhook_action(
    action: WebhookBulkAction,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Массовые операции с webhook конфигурациями."""
    try:
        # Проверяем права доступа
        if current_user.role != "admin":
            # Обычные пользователи могут выполнять bulk операции только со своими конфигурациями
            result = await db.execute(
                select(WebhookConfig).where(
                    and_(
                        WebhookConfig.id.in_(action.webhook_config_ids),
                        WebhookConfig.user_id == current_user.id
                    )
                )
            )
        else:
            # Админы могут работать с любыми конфигурациями
            result = await db.execute(
                select(WebhookConfig).where(WebhookConfig.id.in_(action.webhook_config_ids))
            )
        
        configs = result.scalars().all()
        
        if len(configs) != len(action.webhook_config_ids):
            raise HTTPException(status_code=404, detail="Some webhook configs not found")
        
        # Выполняем действие
        updated_count = 0
        if action.action == "activate":
            for config in configs:
                config.is_active = True
                updated_count += 1
        elif action.action == "deactivate":
            for config in configs:
                config.is_active = False
                updated_count += 1
        elif action.action == "delete":
            for config in configs:
                await db.delete(config)
                updated_count += 1
        elif action.action == "test":
            # Запускаем тесты для указанных конфигураций
            # NOTE: Использование внутреннего метода _send_webhook_with_config осознанное,
            # так как это часть внутреннего API сервиса для массового тестирования
            from ..services.webhook_service import WebhookService
            webhook_service = WebhookService(db)
            
            for config in configs:
                payload = webhook_service.create_webhook_payload(
                    event_type="bulk.test",
                    data={"message": "Bulk test", "config_id": config.id}
                )
                # Запускаем тестирование в фоне для каждой конфигурации
                asyncio.create_task(
                    webhook_service._send_webhook_with_config(payload, config, str(uuid.uuid4()))
                )
                updated_count += 1
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action.action}")
        
        await db.commit()
        
        logger.info(f"Bulk action '{action.action}' performed on {updated_count} webhook configs")
        
        return {
            "message": f"Bulk action '{action.action}' completed",
            "affected_count": updated_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing bulk webhook action: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


