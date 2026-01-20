# app/routes/metrics.py (СОЗДАТЬ НОВЫЙ)

"""
Metrics endpoint for cache and database monitoring
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any, AsyncGenerator
from sqlalchemy.orm import Session

from app.db.database import get_db, engine
from app.services.cache_service import CacheService
from app.utils.logger import get_logger

router = APIRouter(tags=["Metrics"])
logger = get_logger(__name__)


async def get_cache() -> AsyncGenerator[CacheService, None]:
    """Dependency for cache service"""
    cache = CacheService()
    try:
        yield cache
    finally:
        await cache.close()


@router.get("/metrics/cache")
async def get_cache_metrics(
    cache: CacheService = Depends(get_cache)
) -> Dict[str, Any]:
    """
    Get cache performance metrics
    
    Returns:
        - Hit rate (target: > 80%)
        - Total hits/misses
        - Error count
        - Health status
    """
    try:
        stats = await cache.get_cache_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/metrics/database")
async def get_database_metrics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get database connection pool metrics
    
    Returns:
        - Pool size
        - Active connections
        - Utilization percentage
    """
    try:
        pool = engine.pool
        
        pool_size = pool.size()
        checked_out = pool.checkedout()
        overflow = pool.overflow()
        
        utilization = (checked_out / pool_size * 100) if pool_size > 0 else 0
        
        return {
            "status": "success",
            "data": {
                "pool_size": pool_size,
                "active_connections": checked_out,
                "overflow": overflow,
                "utilization_percent": round(utilization, 2),
                "status": "healthy" if utilization < 80 else "high_load"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get database metrics: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/metrics/system")
async def get_system_metrics(
    cache: CacheService = Depends(get_cache),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Comprehensive system health check
    
    Returns:
        - Cache metrics
        - Database metrics
        - Overall health status
    """
    try:
        # Cache stats
        cache_stats = await cache.get_cache_stats()
        
        # Database stats
        pool = engine.pool
        db_metrics = {
            "pool_size": pool.size(),
            "active_connections": pool.checkedout(),
            "utilization_percent": round(
                (pool.checkedout() / pool.size() * 100) if pool.size() > 0 else 0,
                2
            )
        }
        
        # Redis health
        redis_healthy = await cache.health_check()
        
        # Overall status
        overall_status = "healthy"
        
        if not redis_healthy:
            overall_status = "degraded"
        elif cache_stats.get("hit_rate_percent", 0) < 70:
            overall_status = "warning"
        elif db_metrics["utilization_percent"] > 80:
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "cache": cache_stats,
            "database": db_metrics,
            "redis_connection": "up" if redis_healthy else "down"
        }
    
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
