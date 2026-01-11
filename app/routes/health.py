"""API роуты для проверки здоровья сервиса."""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import time
import psutil
import os
import sys
import asyncio
import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .. import __version__
from ..models.response import HealthResponse, StatusResponse, BaseResponse
from ..config import settings
from ..utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


async def check_database_connection() -> tuple[bool, str]:
    """Проверка подключения к базе данных."""
    try:
        # Пытаемся подключиться к PostgreSQL
        conn = await asyncpg.connect(settings.DATABASE_URL)
        await conn.execute('SELECT 1')
        await conn.close()
        return True, "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False, str(e)


async def check_redis_connection() -> tuple[bool, str]:
    """Проверка подключения к Redis."""
    try:
        r = redis.from_url(settings.redis_url_with_auth, decode_responses=True)
        await r.ping()
        await r.close()
        return True, "connected"
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        return False, str(e)


async def check_storage_connection() -> tuple[bool, str]:
    """Проверка подключения к хранилищу (S3/MinIO)."""
    try:
        import aioboto3
        from botocore.config import Config
        
        session = aioboto3.Session()
        async with session.client(
            's3',
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
            config=Config(signature_version='s3v4'),
            use_ssl=settings.S3_USE_SSL
        ) as s3_client:
            # Проверяем существование бакета
            await s3_client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
            return True, "connected"
    except Exception as e:
        logger.error(f"Storage health check failed: {str(e)}")
        return False, str(e)


async def check_ml_service_connection() -> tuple[bool, str]:
    """Проверка подключения к ML сервису."""
    try:
        if not settings.USE_LOCAL_ML_SERVICE:
            # Проверка внешнего ML сервиса
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{settings.ML_SERVICE_URL}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return True, "connected"
                    return False, f"status_code: {response.status}"
        else:
            # Локальный ML сервис - всегда считаем недоступным пока не загружена модель
            return False, "model_not_loaded"
    except Exception as e:
        logger.error(f"ML service health check failed: {str(e)}")
        return False, str(e)


async def check_all_services() -> dict[str, tuple[bool, str]]:
    """Проверка всех сервисов."""
    tasks = {
        "database": check_database_connection(),
        "redis": check_redis_connection(),
        "storage": check_storage_connection(),
        "ml_service": check_ml_service_connection(),
    }
    
    results = {}
    for service_name, task in tasks.items():
        try:
            is_healthy, status = await task
            results[service_name] = (is_healthy, status)
        except Exception as e:
            logger.error(f"Health check for {service_name} failed: {str(e)}")
            results[service_name] = (False, str(e))
    
    return results


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Базовая проверка здоровья сервиса."""
    try:
        process = psutil.Process(os.getpid())
        uptime = time.time() - process.create_time()

        # Проверяем все сервисы
        service_results = await check_all_services()
        
        # Формируем статусы сервисов
        services_status = {}
        for service_name, (is_healthy, status) in service_results.items():
            services_status[service_name] = "healthy" if is_healthy else f"unhealthy: {status}"
        
        # API всегда здоров
        services_status["api"] = "healthy"
        
        # Проверяем, все ли критичные сервисы здоровы
        critical_services = ["database", "storage"]
        all_healthy = all(
            service_results[service][0] for service in critical_services 
            if service in service_results
        )
        
        overall_status = "healthy" if all_healthy else "degraded"
        
        return HealthResponse(
            success=all_healthy,
            status=overall_status,
            version=__version__,
            uptime=uptime,
            services=services_status,
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
        # Проверяем все сервисы
        service_results = await check_all_services()
        
        # Формируем статусы сервисов
        database_status = "healthy" if service_results.get("database", (False, ""))[0] else f"unhealthy: {service_results.get('database', (False, 'not checked'))[1]}"
        redis_status = "healthy" if service_results.get("redis", (False, ""))[0] else f"unhealthy: {service_results.get('redis', (False, 'not checked'))[1]}"
        storage_status = "healthy" if service_results.get("storage", (False, ""))[0] else f"unhealthy: {service_results.get('storage', (False, 'not checked'))[1]}"
        ml_service_status = "healthy" if service_results.get("ml_service", (False, ""))[0] else f"unhealthy: {service_results.get('ml_service', (False, 'not checked'))[1]}"
        
        # Проверяем общий статус
        critical_services = ["database", "storage"]
        all_critical_healthy = all(
            service_results[service][0] for service in critical_services 
            if service in service_results
        )
        
        return StatusResponse(
            success=all_critical_healthy,
            database_status=database_status,
            redis_status=redis_status,
            storage_status=storage_status,
            ml_service_status=ml_service_status,
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
