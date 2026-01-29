"""
API эндпоинты для просмотра метрик
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_current_user
from app.middleware.metrics import get_metrics, get_metrics_content_type
from app.models.response import BaseResponse
from app.services.metrics_service import MetricsService, get_metrics_service
from app.utils.logger import get_logger

router = APIRouter(prefix="/metrics", tags=["metrics"])
logger = get_logger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================


@router.get("/prometheus")
async def get_prometheus_metrics() -> Dict[str, str]:
    """
    Эндпоинт для Prometheus scraping.

    Возвращает метрики в формате Prometheus text format.
    """
    metrics = get_metrics()
    content_type = get_metrics_content_type()

    # FastAPI ожидает Response объект для установки заголовков
    from fastapi import Response

    return Response(
        content=metrics, media_type=content_type, headers={"Content-Type": content_type}
    )


# ============================================================================
# Biometric Metrics (FAR/FRR/EER)
# ============================================================================


@router.get("/biometric", response_model=BaseResponse)
async def get_biometric_metrics(
    user: Dict[str, Any] = Depends(get_current_user),
) -> BaseResponse:
    """
    **Текущие биометрические метрики (FAR/FRR/EER).**

    Требования ТЗ:
    - FAR < 0.1% (False Accept Rate)
    - FRR < 1-3% (False Reject Rate)
    - Accuracy > 99%

    Returns:
    - FAR, FRR, EER текущие значения
    - Compliance статус
    - Статистика верификаций
    - Распределение скоров

    **Доступ:** Аутентифицированные пользователи
    """
    try:
        metrics_service = await get_metrics_service()
        metrics = await metrics_service.get_current_metrics()

        return BaseResponse(
            success=True,
            message="Biometric metrics retrieved successfully",
            data={
                "timestamp": metrics["timestamp"],
                "metrics": {
                    "far_percent": f"{metrics['far']:.4f}%",
                    "frr_percent": f"{metrics['frr']:.4f}%",
                    "eer_percent": f"{metrics['eer']:.4f}%",
                },
                "compliance": metrics["compliance"],
                "statistics": {
                    "genuine_count": metrics["genuine_count"],
                    "impostor_count": metrics["impostor_count"],
                    "total_verifications": (
                        metrics["genuine_count"] + metrics["impostor_count"]
                    ),
                },
                "distributions": metrics["distributions"],
                "window_size": metrics["window_size"],
            },
        )

    except Exception as e:
        logger.error(f"Failed to get biometric metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve biometric metrics: {str(e)}",
        )


@router.get("/biometric/history", response_model=BaseResponse)
async def get_biometric_metrics_history(
    days: int = Query(
        default=7,
        ge=1,
        le=90,
        description="Количество дней истории",
    ),
    user: Dict[str, Any] = Depends(get_current_user),
) -> BaseResponse:
    """
    **Исторические биометрические метрики.**

    Получение временного ряда метрик FAR/FRR/EER из базы данных.

    Args:
        days: Количество дней истории (1-90)

    Returns:
    - Временной ряд метрик
    - Статистика по периодам

    **Доступ:** Аутентифицированные пользователи
    """
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timezone.timedelta(days=days)

        metrics_service = await get_metrics_service()
        history = await metrics_service.get_historical_metrics(start_date, end_date)

        return BaseResponse(
            success=True,
            message="Historical metrics retrieved successfully",
            data={
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days,
                },
                "data_points": len(history),
                "metrics": history,
            },
        )

    except Exception as e:
        logger.error(f"Failed to get metrics history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics history: {str(e)}",
        )


# ============================================================================
# Service Status
# ============================================================================


@router.get("/health", response_model=BaseResponse)
async def metrics_health(
    user: Dict[str, Any] = Depends(get_current_user),
) -> BaseResponse:
    """
    **Healthcheck сервиса метрик.**

    Returns:
    - Статус сервиса
    - Состояние буферов
    - Информация о фоновых задачах

    **Доступ:** Аутентифицированные пользователи
    """
    try:
        metrics_service = await get_metrics_service()
        status = metrics_service.get_status()

        return BaseResponse(
            success=True,
            message="Metrics service status retrieved",
            data={
                "status": "healthy" if status["running"] else "stopped",
                "service": "metrics",
                "details": status,
            },
        )

    except Exception as e:
        logger.error(f"Failed to get metrics health: {str(e)}", exc_info=True)
        return BaseResponse(
            success=False,
            message="Metrics service unhealthy",
            data={
                "status": "error",
                "error": str(e),
            },
        )


# ============================================================================
# Service Status (Public)
# ============================================================================


@router.get("/status")
async def metrics_status_public() -> Dict[str, Any]:
    """
    **Public healthcheck сервиса метрик.**

    Не требует аутентификации.

    Returns:
    - Статус сервиса
    """
    try:
        metrics_service = await get_metrics_service()
        status = metrics_service.get_status()

        return {
            "status": "healthy" if status["running"] else "degraded",
            "service": "metrics",
        }

    except Exception:
        return {
            "status": "unhealthy",
            "service": "metrics",
        }
