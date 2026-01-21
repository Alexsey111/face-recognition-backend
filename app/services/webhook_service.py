"""Сервис webhook-уведомлений."""

import asyncio
import json
import hmac
import hashlib
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, JSON
from sqlalchemy.dialects.postgresql import insert

from ..db.models import WebhookConfig, WebhookLog, WebhookStatus
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Константы
MAX_RESPONSE_LENGTH = 1000
MAX_ERROR_LENGTH = 1000
DEFAULT_TIMEOUT = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1


class WebhookService:
    """
    Сервис webhook-уведомлений.

    ЗАДАЧИ СЕРВИСА:
    - формирование payload
    - подпись HMAC
    - отправка webhook
    - retry с backoff
    - обновление логов
    - дедупликация событий

    ROUTES:
    - НИЧЕГО не знают о retry
    - НИЧЕГО не знают о логах
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_agent = "FaceVerify-Service/2.0"
        self.default_timeout = DEFAULT_TIMEOUT
        self.default_max_retries = DEFAULT_MAX_RETRIES

    # ------------------------------------------------------------------
    # PAYLOAD + SIGNATURE
    # ------------------------------------------------------------------

    def create_webhook_payload(
        self,
        *,
        event_type: str,
        user_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Создание стандартизированного payload для webhook.

        Args:
            event_type: Тип события (например, "liveness.completed")
            user_id: ID пользователя
            data: Данные события

        Returns:
            Сформированный payload
        """
        return {
            "event": event_type,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": data,
            "version": "1.0",
        }

    def compute_hmac_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """
        Вычисление HMAC подписи для payload.

        Args:
            payload: Данные для подписи
            secret: Секретный ключ

        Returns:
            HMAC подпись в формате "sha256=<digest>"
        """
        message = json.dumps(payload, sort_keys=True).encode("utf-8")
        digest = hmac.new(
            secret.encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={digest}"

    def compute_payload_hash(self, payload: Dict[str, Any]) -> str:
        """
        Вычисление хеша payload для дедупликации.

        Args:
            payload: Данные события

        Returns:
            SHA256 хеш payload
        """
        message = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(message).hexdigest()

    # ------------------------------------------------------------------
    # PUBLIC API — ЕДИНСТВЕННАЯ ТОЧКА ВХОДА
    # ------------------------------------------------------------------

    async def emit_event(
        self,
        *,
        event_type: str,
        user_id: str,
        payload: Dict[str, Any],
        skip_duplicates: bool = True,
        max_retries: Optional[int] = None,
    ) -> None:
        """
        ЕДИНСТВЕННЫЙ публичный метод для отправки webhook событий.

        Используется:
        - верификацией
        - liveness detection
        - тестами
        - bulk retry

        Args:
            event_type: Тип события
            user_id: ID пользователя
            payload: Данные события
            skip_duplicates: Пропускать дубликаты (по умолчанию True)
            max_retries: Переопределить количество retry (опционально)
        """
        # Получаем активные конфигурации для данного пользователя
        # event_types хранится как JSON массив, фильтруем в Python т.к. JSON @> JSON не поддерживается
        result = await self.db.execute(
            select(WebhookConfig)
            .where(WebhookConfig.user_id == user_id)
            .where(WebhookConfig.is_active.is_(True))
        )

        configs = result.scalars().all()

        # Фильтруем конфигурации по типу события
        configs = [c for c in configs if event_type in (c.event_types or [])]

        if not configs:
            logger.debug(
                f"No active webhook configs found for user {user_id} and event {event_type}"
            )
            return

        # Вычисляем хеш payload для дедупликации
        payload_hash = self.compute_payload_hash(payload) if skip_duplicates else None

        for config in configs:
            # Проверка на дубликаты (если включено)
            if skip_duplicates and payload_hash:
                duplicate_check = await self.db.execute(
                    select(WebhookLog)
                    .where(
                        WebhookLog.webhook_config_id == config.id,
                        WebhookLog.payload_hash == payload_hash,
                        WebhookLog.created_at
                        >= datetime.now(timezone.utc) - timedelta(hours=1),
                    )
                    .limit(1)
                )

                if duplicate_check.scalar_one_or_none():
                    logger.info(
                        f"Skipping duplicate webhook for config {config.id}, "
                        f"event {event_type}, hash {payload_hash[:8]}..."
                    )
                    continue

            log_id = uuid.uuid4()

            # Создаем лог с подписью
            signature = self.compute_hmac_signature(payload, config.secret)

            log = WebhookLog(
                id=str(log_id),  # UUID -> String для VARCHAR(36) колонки
                webhook_config_id=config.id,
                event_type=event_type,
                payload=payload,
                payload_hash=payload_hash,
                signature=signature,
                attempts=0,
                status=WebhookStatus.PENDING,
                created_at=datetime.now(timezone.utc),
            )

            self.db.add(log)
            await self.db.commit()

            # Запускаем отправку асинхронно
            asyncio.create_task(
                self._send_with_retry(
                    payload=payload,
                    config=config,
                    log_id=log_id,
                    signature=signature,
                    max_retries_override=max_retries,
                )
            )

            logger.info(
                f"Webhook queued: config={config.id}, event={event_type}, "
                f"log={log_id}, user={user_id}"
            )

    # ------------------------------------------------------------------
    # RETRY LOGIC
    # ------------------------------------------------------------------

    async def _send_with_retry(
        self,
        *,
        payload: Dict[str, Any],
        config: WebhookConfig,
        log_id: uuid.UUID,
        signature: str,
        max_retries_override: Optional[int] = None,
    ) -> None:
        """
        Отправка webhook с retry логикой.

        Args:
            payload: Данные для отправки
            config: Конфигурация webhook
            log_id: ID лога
            signature: HMAC подпись
            max_retries_override: Переопределить max_retries из конфига
        """
        max_retries = (
            max_retries_override or config.max_retries or self.default_max_retries
        )

        for attempt in range(1, max_retries + 1):
            start_time = time.time()

            try:
                success, status_code, response_text = await self._send_once(
                    payload=payload,
                    config=config,
                    signature=signature,
                )

                processing_time = time.time() - start_time

                if success:
                    await self._update_log_success(
                        log_id=log_id,
                        status_code=status_code,
                        response=response_text,
                        processing_time=processing_time,
                        attempt=attempt,
                    )
                    logger.info(
                        f"Webhook delivered successfully: log={log_id}, "
                        f"attempt={attempt}/{max_retries}, time={processing_time:.3f}s"
                    )
                    return

                await self._update_log_retry(
                    log_id=log_id,
                    attempt=attempt,
                    status_code=status_code,
                    error=response_text,
                    max_retries=max_retries,
                )

                logger.warning(
                    f"Webhook delivery failed: log={log_id}, "
                    f"attempt={attempt}/{max_retries}, status={status_code}"
                )

            except Exception as exc:
                processing_time = time.time() - start_time
                await self._update_log_retry(
                    log_id=log_id,
                    attempt=attempt,
                    status_code=0,
                    error=str(exc),
                    max_retries=max_retries,
                )

                logger.error(
                    f"Webhook delivery exception: log={log_id}, "
                    f"attempt={attempt}/{max_retries}, error={str(exc)}"
                )

            # Exponential backoff для следующей попытки
            if attempt < max_retries:
                delay_base = config.retry_delay or DEFAULT_RETRY_DELAY
                delay = delay_base * (2 ** (attempt - 1))
                logger.debug(f"Waiting {delay}s before retry {attempt + 1}")
                await asyncio.sleep(delay)

        # Все попытки исчерпаны
        await self._mark_log_failed(log_id)
        logger.error(
            f"Webhook delivery failed after {max_retries} attempts: log={log_id}"
        )

    # ------------------------------------------------------------------
    # SINGLE SEND
    # ------------------------------------------------------------------

    async def _send_once(
        self,
        *,
        payload: Dict[str, Any],
        config: WebhookConfig,
        signature: str,
    ) -> tuple[bool, int, str]:
        """
        Однократная отправка webhook.

        Args:
            payload: Данные для отправки
            config: Конфигурация webhook
            signature: HMAC подпись

        Returns:
            Tuple (success, status_code, response_text)
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            "X-Webhook-Signature": signature,
            "X-Webhook-Event": payload.get("event"),
            "X-Webhook-Delivery": str(uuid.uuid4()),  # Уникальный ID доставки
        }

        timeout = aiohttp.ClientTimeout(total=config.timeout or self.default_timeout)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    config.webhook_url,
                    json=payload,
                    headers=headers,
                ) as response:
                    text = await response.text()
                    success = 200 <= response.status < 300
                    return success, response.status, text[:MAX_RESPONSE_LENGTH]

        except aiohttp.ClientError as e:
            return False, 0, f"Client error: {str(e)}"[:MAX_ERROR_LENGTH]
        except asyncio.TimeoutError:
            return False, 0, "Request timeout"
        except Exception as e:
            return False, 0, f"Unexpected error: {str(e)}"[:MAX_ERROR_LENGTH]

    # ------------------------------------------------------------------
    # LOG UPDATES
    # ------------------------------------------------------------------

    async def _update_log_success(
        self,
        *,
        log_id: uuid.UUID,
        status_code: int,
        response: str,
        processing_time: float,
        attempt: int,
    ) -> None:
        """Обновление лога при успешной доставке."""
        await self.db.execute(
            update(WebhookLog)
            .where(WebhookLog.id == str(log_id))
            .values(
                status=WebhookStatus.SUCCESS,
                attempts=attempt,
                http_status=status_code,
                response_body=response[:MAX_RESPONSE_LENGTH],
                processing_time=processing_time,
                last_attempt_at=datetime.now(timezone.utc),
                next_retry_at=None,
                error_message=None,
            )
        )
        await self.db.commit()

    async def _update_log_retry(
        self,
        *,
        log_id: uuid.UUID,
        attempt: int,
        status_code: int,
        error: str,
        max_retries: int,
    ) -> None:
        """Обновление лога при неудачной попытке."""
        # Вычисляем время следующей попытки
        next_retry_at = None
        if attempt < max_retries:
            # Получаем конфигурацию для вычисления задержки
            result = await self.db.execute(
                select(WebhookConfig)
                .join(WebhookLog, WebhookLog.webhook_config_id == WebhookConfig.id)
                .where(WebhookLog.id == str(log_id))
            )
            config = result.scalar_one_or_none()

            if config:
                delay_base = config.retry_delay or DEFAULT_RETRY_DELAY
                delay_seconds = delay_base * (2**attempt)
                next_retry_at = datetime.now(timezone.utc) + timedelta(
                    seconds=delay_seconds
                )

        await self.db.execute(
            update(WebhookLog)
            .where(WebhookLog.id == str(log_id))
            .values(
                status=WebhookStatus.RETRY,
                attempts=attempt,
                http_status=status_code,
                error_message=error[:MAX_ERROR_LENGTH],
                last_attempt_at=datetime.now(timezone.utc),
                next_retry_at=next_retry_at,
            )
        )
        await self.db.commit()

    async def _mark_log_failed(self, log_id: uuid.UUID) -> None:
        """Пометка лога как окончательно неудавшегося."""
        await self.db.execute(
            update(WebhookLog)
            .where(WebhookLog.id == str(log_id))
            .values(
                status=WebhookStatus.FAILED,
                last_attempt_at=datetime.now(timezone.utc),
                next_retry_at=None,
            )
        )
        await self.db.commit()

    # ------------------------------------------------------------------
    # URL VALIDATION (used in routes)
    # ------------------------------------------------------------------

    async def validate_webhook_url(self, url: str) -> Dict[str, Any]:
        """
        Валидация webhook URL путем тестовой отправки HEAD запроса.

        Args:
            url: URL для проверки

        Returns:
            Dict с результатом валидации
        """
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(url, allow_redirects=True) as response:
                    return {"valid": True, "status": response.status, "reachable": True}
        except aiohttp.ClientError as e:
            return {
                "valid": False,
                "error": f"Connection error: {str(e)}",
                "reachable": False,
            }
        except asyncio.TimeoutError:
            return {"valid": False, "error": "Connection timeout", "reachable": False}
        except Exception as exc:
            return {"valid": False, "error": str(exc), "reachable": False}

    # ------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------

    async def close(self):
        """
        Закрытие сервиса (если требуется cleanup).
        Вызывается при shutdown приложения.
        """
        # В текущей реализации нет ресурсов для закрытия,
        # но метод оставлен для возможного расширения
        logger.info("WebhookService closed")

    # ------------------------------------------------------------------
    # INTERNAL HELPERS FOR ROUTES
    # ------------------------------------------------------------------

    async def _send_webhook_with_config(
        self,
        payload: Dict[str, Any],
        config: WebhookConfig,
        log_id: str,
    ) -> None:
        """
        Вспомогательный метод для отправки webhook с существующей конфигурацией.
        Используется для retry и bulk операций.
        """
        signature = self.compute_hmac_signature(payload, config.secret)
        await self._send_with_retry(
            payload=payload,
            config=config,
            log_id=uuid.UUID(log_id),
            signature=signature,
            max_retries_override=None,
        )
