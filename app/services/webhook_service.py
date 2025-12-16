"""
Сервис вебхуков.
Отправка уведомлений и событий в внешние системы через webhook-и.
"""

import json
import asyncio
from typing import Optional, Dict, Any, List
import httpx
from datetime import datetime, timezone
import uuid

from .. import __version__
from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import WebhookError, RetryExhaustedError

logger = get_logger(__name__)


class WebhookService:
    """
    Сервис для отправки webhook уведомлений.
    """

    def __init__(self):
        self.timeout = settings.WEBHOOK_TIMEOUT
        self.max_retries = settings.WEBHOOK_MAX_RETRIES
        self.retry_delay = settings.WEBHOOK_RETRY_DELAY
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def send_verification_result(
        self,
        user_id: str,
        session_id: str,
        verification_result: Dict[str, Any],
        webhook_url: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Отправка результата верификации.

        Args:
            user_id: ID пользователя
            session_id: ID сессии
            verification_result: Результат верификации
            webhook_url: URL webhook (если None, используется стандартный)
            additional_data: Дополнительные данные

        Returns:
            bool: True если webhook отправлен успешно
        """
        try:
            payload = {
                "event_type": "verification.completed",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "session_id": session_id,
                "data": {
                    "verified": verification_result.get("verified", False),
                    "confidence": verification_result.get("confidence", 0.0),
                    "similarity_score": verification_result.get(
                        "similarity_score", 0.0
                    ),
                    "threshold_used": verification_result.get("threshold_used", 0.8),
                    "processing_time": verification_result.get("processing_time", 0.0),
                    "face_detected": verification_result.get("face_detected", False),
                    "reference_id": verification_result.get("reference_id"),
                    **(additional_data or {}),
                },
            }

            return await self._send_webhook(
                payload=payload,
                webhook_url=webhook_url or settings.WEBHOOK_URL,
                event_type="verification",
            )

        except Exception as e:
            logger.error(f"Failed to send verification webhook: {str(e)}")
            return False

    async def send_liveness_result(
        self,
        user_id: str,
        session_id: str,
        liveness_result: Dict[str, Any],
        webhook_url: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Отправка результата проверки живости.

        Args:
            user_id: ID пользователя
            session_id: ID сессии
            liveness_result: Результат проверки живости
            webhook_url: URL webhook (если None, используется стандартный)
            additional_data: Дополнительные данные

        Returns:
            bool: True если webhook отправлен успешно
        """
        try:
            payload = {
                "event_type": "liveness.completed",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "session_id": session_id,
                "data": {
                    "liveness_detected": liveness_result.get(
                        "liveness_detected", False
                    ),
                    "confidence": liveness_result.get("confidence", 0.0),
                    "challenge_type": liveness_result.get("challenge_type", "passive"),
                    "anti_spoofing_score": liveness_result.get("anti_spoofing_score"),
                    "processing_time": liveness_result.get("processing_time", 0.0),
                    "face_detected": liveness_result.get("face_detected", False),
                    "multiple_faces": liveness_result.get("multiple_faces", False),
                    "recommendations": liveness_result.get("recommendations", []),
                    **(additional_data or {}),
                },
            }

            return await self._send_webhook(
                payload=payload,
                webhook_url=webhook_url or settings.WEBHOOK_URL,
                event_type="liveness",
            )

        except Exception as e:
            logger.error(f"Failed to send liveness webhook: {str(e)}")
            return False

    async def send_reference_created(
        self,
        user_id: str,
        reference_id: str,
        reference_data: Dict[str, Any],
        webhook_url: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Отправка уведомления о создании эталона.

        Args:
            user_id: ID пользователя
            reference_id: ID эталона
            reference_data: Данные эталона
            webhook_url: URL webhook (если None, используется стандартный)
            additional_data: Дополнительные данные

        Returns:
            bool: True если webhook отправлен успешно
        """
        try:
            payload = {
                "event_type": "reference.created",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "reference_id": reference_id,
                "data": {
                    "label": reference_data.get("label"),
                    "quality_score": reference_data.get("quality_score"),
                    "file_url": reference_data.get("file_url"),
                    "image_dimensions": reference_data.get("image_dimensions"),
                    "processing_time": reference_data.get("processing_time"),
                    **(additional_data or {}),
                },
            }

            return await self._send_webhook(
                payload=payload,
                webhook_url=webhook_url or settings.WEBHOOK_URL,
                event_type="reference",
            )

        except Exception as e:
            logger.error(f"Failed to send reference webhook: {str(e)}")
            return False

    async def send_user_activity(
        self,
        user_id: str,
        activity_type: str,
        activity_data: Dict[str, Any],
        webhook_url: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Отправка уведомления о активности пользователя.

        Args:
            user_id: ID пользователя
            activity_type: Тип активности (login, logout, upload, etc.)
            activity_data: Данные активности
            webhook_url: URL webhook (если None, используется стандартный)
            additional_data: Дополнительные данные

        Returns:
            bool: True если webhook отправлен успешно
        """
        try:
            payload = {
                "event_type": f"user.{activity_type}",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "data": {
                    "activity_type": activity_type,
                    "activity_data": activity_data,
                    **(additional_data or {}),
                },
            }

            return await self._send_webhook(
                payload=payload,
                webhook_url=webhook_url or settings.WEBHOOK_URL,
                event_type="user_activity",
            )

        except Exception as e:
            logger.error(f"Failed to send user activity webhook: {str(e)}")
            return False

    async def send_system_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "info",
        additional_data: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
    ) -> bool:
        """
        Отправка системного уведомления.

        Args:
            alert_type: Тип предупреждения
            message: Сообщение
            severity: Серьезность (info, warning, error, critical)
            additional_data: Дополнительные данные
            webhook_url: URL webhook (если None, используется стандартный)

        Returns:
            bool: True если webhook отправлен успешно
        """
        try:
            payload = {
                "event_type": "system.alert",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "alert_type": alert_type,
                    "message": message,
                    "severity": severity,
                    "service": "face-recognition-service",
                    "version": __version__,
                    **(additional_data or {}),
                },
            }

            return await self._send_webhook(
                payload=payload,
                webhook_url=webhook_url or settings.WEBHOOK_URL,
                event_type="system_alert",
            )

        except Exception as e:
            logger.error(f"Failed to send system alert webhook: {str(e)}")
            return False

    async def send_batch_webhooks(
        self, webhooks_data: List[Dict[str, Any]], webhook_url: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Отправка пакетных webhook уведомлений.

        Args:
            webhooks_data: Список данных для webhook-ов
            webhook_url: URL webhook (если None, используется стандартный)

        Returns:
            Dict[str, int]: Статистика отправки (successful, failed)
        """
        successful = 0
        failed = 0

        # Отправляем webhook-и параллельно
        tasks = []
        for webhook_data in webhooks_data:
            task = asyncio.create_task(
                self._send_webhook(
                    payload=webhook_data.get("payload"),
                    webhook_url=webhook_url
                    or webhook_data.get("webhook_url")
                    or settings.WEBHOOK_URL,
                    event_type=webhook_data.get("event_type", "batch"),
                )
            )
            tasks.append(task)

        # Ждем завершения всех задач
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Подсчитываем результаты
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch webhook failed: {str(result)}")
                failed += 1
            elif result:
                successful += 1
            else:
                failed += 1

        logger.info(f"Batch webhooks sent: {successful} successful, {failed} failed")

        return {"successful": successful, "failed": failed, "total": len(webhooks_data)}

    async def test_webhook(
        self,
        webhook_url: Optional[str] = None,
        test_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Тестирование webhook endpoint.

        Args:
            webhook_url: URL webhook для тестирования
            test_data: Тестовые данные

        Returns:
            Dict[str, Any]: Результат тестирования
        """
        try:
            payload = {
                "event_type": "webhook.test",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "message": "This is a test webhook from Face Recognition Service",
                    "service": "face-recognition-service",
                    "version": __version__,
                    **(test_data or {}),
                },
            }

            start_time = asyncio.get_event_loop().time()
            success = await self._send_webhook(
                payload=payload,
                webhook_url=webhook_url or settings.WEBHOOK_URL,
                event_type="test",
            )
            end_time = asyncio.get_event_loop().time()

            return {
                "success": success,
                "response_time": end_time - start_time,
                "webhook_url": webhook_url or settings.WEBHOOK_URL,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Webhook test failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "webhook_url": webhook_url or settings.WEBHOOK_URL,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _send_webhook(
        self, payload: Dict[str, Any], webhook_url: str, event_type: str
    ) -> bool:
        """
        Отправка webhook с повторными попытками.

        Args:
            payload: Данные для отправки
            webhook_url: URL webhook
            event_type: Тип события

        Returns:
            bool: True если отправлен успешно
        """
        if not webhook_url:
            logger.warning(f"No webhook URL provided for {event_type}")
            return False

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Sending webhook {event_type} (attempt {attempt + 1}) to {webhook_url}"
                )

                # Подготавливаем заголовки
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "FaceRecognitionService/1.0",
                    "X-Event-Type": event_type,
                    "X-Event-ID": payload.get("event_id", str(uuid.uuid4())),
                    "X-Timestamp": payload.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                }

                # Отправляем запрос
                response = await self.client.post(
                    webhook_url, json=payload, headers=headers
                )

                # Проверяем ответ
                if response.status_code >= 200 and response.status_code < 300:
                    logger.info(
                        f"Webhook {event_type} sent successfully (status: {response.status_code})"
                    )
                    return True
                elif response.status_code >= 400 and response.status_code < 500:
                    # Клиентские ошибки - не повторяем
                    logger.warning(
                        f"Webhook {event_type} failed with client error {response.status_code}: {response.text}"
                    )
                    return False
                else:
                    # Серверные ошибки - повторяем
                    logger.warning(
                        f"Webhook {event_type} failed with server error {response.status_code}: {response.text}"
                    )
                    last_error = f"HTTP {response.status_code}: {response.text}"

            except httpx.TimeoutException:
                logger.warning(f"Webhook {event_type} timeout (attempt {attempt + 1})")
                last_error = "Timeout"

            except httpx.ConnectError as e:
                logger.warning(
                    f"Webhook {event_type} connection error (attempt {attempt + 1}): {str(e)}"
                )
                last_error = f"Connection error: {str(e)}"

            except Exception as e:
                logger.warning(
                    f"Webhook {event_type} unexpected error (attempt {attempt + 1}): {str(e)}"
                )
                last_error = str(e)

            # Если это не последняя попытка, ждем перед повтором
            if attempt < self.max_retries:
                await asyncio.sleep(
                    self.retry_delay * (attempt + 1)
                )  # Экспоненциальная задержка

        # Все попытки исчерпаны
        logger.error(
            f"Webhook {event_type} failed after {self.max_retries + 1} attempts: {last_error}"
        )
        return False

    async def validate_webhook_url(self, webhook_url: str) -> Dict[str, Any]:
        """
        Валидация webhook URL.

        Args:
            webhook_url: URL для валидации

        Returns:
            Dict[str, Any]: Результат валидации
        """
        try:
            # Проверяем формат URL
            if not webhook_url.startswith(("http://", "https://")):
                return {
                    "valid": False,
                    "error": "URL must start with http:// or https://",
                }

            # Пробуем подключиться к endpoint
            response = await self.client.get(
                webhook_url, headers={"User-Agent": "FaceRecognitionService/1.0"}
            )

            return {
                "valid": True,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "url": webhook_url,
            }

        except httpx.TimeoutException:
            return {"valid": False, "error": "Connection timeout", "url": webhook_url}
        except httpx.ConnectError:
            return {"valid": False, "error": "Connection failed", "url": webhook_url}
        except Exception as e:
            return {"valid": False, "error": str(e), "url": webhook_url}

    async def close(self):
        """
        Закрытие HTTP клиента.
        """
        await self.client.aclose()
