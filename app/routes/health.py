"""
API роуты для проверки здоровья сервиса.
Эндпоинты для мониторинга состояния приложения.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import time
import psutil
import os

from ..models.response import HealthResponse, StatusResponse, BaseResponse
from ..config import settings
from ..services.database_service import DatabaseService
from ..services.cache_service import CacheService
from ..services.storage_service import StorageService
from ..services.ml_service import MLService
from ..utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Базовая проверка здоровья сервиса.
    
    Returns:
        HealthResponse: Информация о состоянии сервиса
    """
    start_time = time.time()
    
    try:
        # Получаем uptime процесса
        process = psutil.Process(os.getpid())
        uptime = time.time() - process.create_time()
        
        # Проверяем статус внешних сервисов
        services_status = await check_services_health()
        
        # Информация о системе
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "python_version": os.sys.version,
            "process_id": os.getpid(),
            "worker_id": os.environ.get("WORKER_ID", "unknown")
        }
        
        response = HealthResponse(
            success=True,
            status="healthy",
            version="1.0.0",
            uptime=uptime,
            services=services_status,
            system_info=system_info
        )
        
        logger.info("Health check completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc)
            }
        )


@router.get("/status", response_model=StatusResponse)
async def detailed_status_check(request: Request):
    """
    Детальная проверка состояния всех сервисов.
    
    Returns:
        StatusResponse: Подробная информация о состоянии сервисов
    """
    start_time = time.time()
    
    try:
        # Проверяем состояние всех внешних сервисов
        database_status = await check_database_status()
        redis_status = await check_redis_status()
        storage_status = await check_storage_status()
        ml_service_status = await check_ml_service_status()
        
        response = StatusResponse(
            success=True,
            database_status=database_status,
            redis_status=redis_status,
            storage_status=storage_status,
            ml_service_status=ml_service_status,
            last_heartbeat=datetime.now(timezone.utc)
        )
        
        logger.info("Detailed status check completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Detailed status check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc)
            }
        )


@router.get("/ready", response_model=BaseResponse)
async def readiness_check(request: Request):
    """
    Проверка готовности сервиса к приему запросов.
    
    Returns:
        BaseResponse: Статус готовности
    """
    try:
        # Проверяем критически важные сервисы
        db_ready = await check_database_status()
        redis_ready = await check_redis_status()
        ml_ready = await check_ml_service_status()
        
        if db_ready != "healthy" or redis_ready != "healthy" or ml_ready != "healthy":
            raise HTTPException(
                status_code=503,
                detail={
                    "success": False,
                    "message": "Service not ready",
                    "database_status": db_ready,
                    "redis_status": redis_ready,
                    "ml_service_status": ml_ready
                }
            )
        
        return BaseResponse(
            success=True,
            message="Service is ready"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "message": "Readiness check failed",
                "error": str(e)
            }
        )


@router.get("/live", response_model=BaseResponse)
async def liveness_check(request: Request):
    """
    Проверка живости сервиса (простая проверка).
    
    Returns:
        BaseResponse: Статус живости
    """
    return BaseResponse(
        success=True,
        message="Service is alive"
    )


async def check_services_health() -> dict:
    """
    Проверка состояния всех внешних сервисов.
    
    Returns:
        dict: Статус каждого сервиса
    """
    services = {}
    
    # Проверяем БД
    services["database"] = await check_database_status()
    
    # Проверяем Redis
    services["redis"] = await check_redis_status()
    
    # Проверяем хранилище файлов
    services["storage"] = await check_storage_status()
    
    # Проверяем ML сервис
    services["ml_service"] = await check_ml_service_status()
    
    return services


async def check_database_status() -> str:
    """
    Проверка состояния подключения к базе данных.
    
    Returns:
        str: Статус подключения
    """
    try:
        db_service = DatabaseService()
        await db_service.health_check()
        return "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return "unhealthy"


async def check_redis_status() -> str:
    """
    Проверка состояния подключения к Redis.
    
    Returns:
        str: Статус подключения
    """
    try:
        cache_service = CacheService()
        await cache_service.health_check()
        return "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        return "unhealthy"


async def check_storage_status() -> str:
    """
    Проверка состояния хранилища файлов.
    
    Returns:
        str: Статус хранилища
    """
    try:
        storage_service = StorageService()
        await storage_service.health_check()
        return "healthy"
    except Exception as e:
        logger.error(f"Storage health check failed: {str(e)}")
        return "unhealthy"


async def check_ml_service_status() -> str:
    """
    Проверка состояния ML сервиса.
    
    Returns:
        str: Статус ML сервиса
    """
    try:
        ml_service = MLService()
        await ml_service.health_check()
        return "healthy"
    except Exception as e:
        logger.error(f"ML service health check failed: {str(e)}")
        return "unhealthy"


@router.get("/metrics", response_model=dict)
async def get_metrics(request: Request):
    """
    Получение метрик производительности сервиса.
    
    Returns:
        dict: Метрики производительности
    """
    try:
        # Базовые системные метрики
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Метрики процесса
        process = psutil.Process(os.getpid())
        
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "memory_total": memory.total,
                "disk_percent": disk.percent,
                "disk_free": disk.free,
                "disk_total": disk.total
            },
            "process": {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": process.memory_info()._asdict(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time(),
                "status": process.status()
            },
            "application": {
                "uptime": time.time() - process.create_time(),
                "worker_id": os.environ.get("WORKER_ID", "unknown"),
                "version": "1.0.0"
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc)
            }
        )