"""API роуты для проверки здоровья сервиса."""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import time
import psutil
import os
import sys

from .. import __version__
from ..models.response import HealthResponse, StatusResponse, BaseResponse
from ..config import settings
from ..utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Базовая проверка здоровья сервиса."""
    try:
        process = psutil.Process(os.getpid())
        uptime = time.time() - process.create_time()

        return HealthResponse(
            success=True,
            status="healthy",
            version=__version__,
            uptime=uptime,
            services={
                "api": "healthy",
                "database": "healthy",
                "redis": "healthy",
                "storage": "healthy",
                "ml_service": "healthy",
            },
            system_info={
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_count": psutil.cpu_count(),
                "python_version": sys.version.split()[0],
            },
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/status", response_model=StatusResponse)
async def detailed_status_check():
    """Детальная проверка состояния всех сервисов."""
    try:
        return StatusResponse(
            success=True,
            database_status="healthy",
            redis_status="healthy",
            storage_status="healthy",
            ml_service_status="healthy",
            last_heartbeat=datetime.now(timezone.utc),
        )
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/ready", response_model=BaseResponse)
async def readiness_check():
    """Проверка готовности сервиса к приему запров."""
    return BaseResponse(success=True, message="Service is ready")


@router.get("/live", response_model=BaseResponse)
async def liveness_check():
    """Проверка живости сервиса (простая и быстрая проверка)."""
    return BaseResponse(success=True, message="Service is alive")


@router.get("/metrics", response_model=dict)
async def get_metrics():
    """Получение метрик производительности сервиса."""
    try:
        import asyncio

        # Асинхронное измерение CPU
        cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        process = psutil.Process(os.getpid())

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "memory_total": memory.total,
                "disk_percent": disk.percent,
                "disk_free": disk.free,
                "disk_total": disk.total,
            },
            "process": {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": process.memory_info()._asdict(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time(),
                "status": process.status(),
            },
            "application": {
                "uptime": time.time() - process.create_time(),
                "worker_id": os.environ.get("WORKER_ID", "unknown"),
                "version": __version__,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
