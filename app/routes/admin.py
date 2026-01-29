"""
API роуты для администрирования.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, List, Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from ..dependencies import get_current_user
from ..middleware.metrics import record_business_error, track_processing
from ..models.reference import ReferenceStats
from ..models.response import BaseResponse
from ..models.user import UserListResponse, UserUpdate
from ..models.verification import SessionStats
from ..services.cache_service import CacheService
from ..services.database_service import DatabaseService
from ..services.ml_service import MLService
from ..services.storage_service import StorageService
from ..utils.exceptions import NotFoundError, ValidationError
from ..utils.logger import audit_event, get_logger

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = get_logger(__name__)


class AdminStatsResponse(BaseModel):
    success: bool
    request_id: str
    total_users: int
    active_sessions: int
    pending_verifications: int
    total_references: int
    verification_stats: Optional[dict] = None
    timestamp: datetime


class UserActivityResponse(BaseModel):
    success: bool
    request_id: str
    users: List[dict]
    total: int
    page: int
    page_size: int
    timestamp: datetime


class AuditLogResponse(BaseModel):
    success: bool
    request_id: str
    audit_logs: List[dict]
    total_count: int
    limit: int
    offset: int
    has_next: bool
    timestamp: datetime


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def get_performance_metrics() -> dict:
    """Get system performance metrics."""
    try:
        process = psutil.Process()
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": mem.percent,
            "memory_available": mem.available,
            "disk_usage": disk.percent,
            "disk_free": disk.free,
            "process_memory": process.memory_info().rss,
            "process_cpu": process.cpu_percent(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return {}


@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    date_from: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    include_user_stats: bool = Query(False),
    include_performance: bool = Query(True),
    http_request: Request = None,
    current_user: dict = Depends(get_current_user),
):
    request_id = str(uuid.uuid4())

    try:
        with track_processing("admin_stats"):
            logger.info(f"Getting admin stats, request {request_id}")

            db_service = DatabaseService()

            # Get total counts
            total_users = await db_service.get_total_users()
            active_sessions = await db_service.get_active_sessions()
            pending_verifications = await db_service.get_pending_verifications()
            total_references = await db_service.get_total_references()

            # Get verification stats
            verification_stats = await db_service.get_verification_stats()
            user_activity = await db_service.get_user_activity(
                date_from=date_from, date_to=date_to
            )

            performance_metrics = None
            if include_performance:
                performance_metrics = await get_performance_metrics()

            # Audit log
            audit_event(
                action="admin_stats_viewed",
                target_type="admin",
                target_id=current_user.get("user_id"),
                admin_id=current_user.get("user_id"),
                details={
                    "date_from": date_from,
                    "date_to": date_to,
                    "include_user_stats": include_user_stats,
                    "include_performance": include_performance,
                },
                ip_address=get_client_ip(http_request),
                user_agent=http_request.headers.get("user-agent"),
                success=True,
            )

            return AdminStatsResponse(
                success=True,
                request_id=request_id,
                total_users=total_users,
                active_sessions=active_sessions,
                pending_verifications=pending_verifications,
                total_references=total_references,
                verification_stats=verification_stats,
                timestamp=datetime.now(timezone.utc),
            )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        record_business_error("admin_stats_error")
        logger.exception("Admin stats error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit-logs")
async def get_audit_logs(
    action: str | None = Query(None),
    user_id: str | None = Query(None),
    admin_id: str | None = Query(None),
    target_type: str | None = Query(None),
    date_from: str | None = Query(None, description="YYYY-MM-DD HH:MM:SS"),
    date_to: str | None = Query(None, description="YYYY-MM-DD HH:MM:SS"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    http_request: Request = None,
    current_user: dict = Depends(get_current_user),
):
    request_id = str(uuid.uuid4())

    try:
        with track_processing("admin_audit_logs"):
            # Parse dates
            parsed_date_from = (
                datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S") if date_from else None
            )
            parsed_date_to = (
                datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S") if date_to else None
            )

            filters = {
                k: v
                for k, v in {
                    "action": action,
                    "user_id": user_id,
                    "admin_id": admin_id,
                    "target_type": target_type,
                }.items()
                if v
            }

            db_service = DatabaseService()

            logs = await db_service.get_audit_logs(
                filters=filters,
                date_from=parsed_date_from,
                date_to=parsed_date_to,
                limit=limit,
                offset=offset,
            )

            total = await db_service.get_audit_logs_count(
                filters=filters,
                date_from=parsed_date_from,
                date_to=parsed_date_to,
            )

            # Audit log
            audit_event(
                action="admin_audit_logs_viewed",
                target_type="admin",
                target_id=current_user.get("user_id"),
                admin_id=current_user.get("user_id"),
                details={"filters": filters, "limit": limit, "offset": offset},
                ip_address=get_client_ip(http_request),
                user_agent=http_request.headers.get("user-agent"),
                success=True,
            )

            return {
                "success": True,
                "request_id": request_id,
                "audit_logs": logs,
                "total_count": total,
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total,
                "timestamp": datetime.now(timezone.utc),
            }

    except ValueError:
        raise ValidationError("Invalid datetime format")

    except Exception as e:
        record_business_error("admin_audit_logs_error")
        logger.exception("Audit logs error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health")
async def get_system_health(
    http_request: Request,
    current_user: dict = Depends(get_current_user),
):
    request_id = str(uuid.uuid4())

    try:
        with track_processing("admin_health"):
            services_status = {}

            async def check(name, coro):
                try:
                    start = time.time()
                    await coro()
                    return {
                        "status": "healthy",
                        "response_time": round(time.time() - start, 4),
                    }
                except Exception as e:
                    return {"status": "unhealthy", "error": str(e)}

            services_status["database"] = await check(
                "database", DatabaseService().health_check
            )
            services_status["redis"] = await check("redis", CacheService().health_check)
            services_status["storage"] = await check(
                "storage", StorageService().health_check
            )
            services_status["ml_service"] = await check(
                "ml_service", MLService().health_check
            )

            system_info = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "process_count": len(psutil.pids()),
                "boot_time": psutil.boot_time(),
            }

            # Audit log
            audit_event(
                action="admin_health_viewed",
                target_type="admin",
                target_id=current_user.get("user_id"),
                admin_id=current_user.get("user_id"),
                details={"services": list(services_status.keys())},
                ip_address=get_client_ip(http_request),
                user_agent=http_request.headers.get("user-agent"),
                success=True,
            )

            return {
                "success": True,
                "request_id": request_id,
                "overall_status": (
                    "healthy"
                    if all(s["status"] == "healthy" for s in services_status.values())
                    else "unhealthy"
                ),
                "services": services_status,
                "system": system_info,
                "timestamp": datetime.now(timezone.utc),
            }

    except Exception as e:
        record_business_error("admin_health_error")
        logger.exception("Health check error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors")
async def get_errors(
    limit: int = Query(100, ge=1, le=1000),
    error_type: str | None = Query(None),
    severity: str | None = Query(None, regex="^(error|critical|warning)$"),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Получение последних ошибок системы из audit_logs.
    Фильтры:
    - error_type: Тип ошибки (ValidationError, ProcessingError, etc.)
    - severity: Уровень критичности (error, critical, warning)
    - date_from/date_to: Временной диапазон
    """
    request_id = str(uuid.uuid4())
    try:
        with track_processing("admin_errors"):
            # Парсинг дат
            parsed_date_from = None
            parsed_date_to = None
            if date_from:
                try:
                    parsed_date_from = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    raise ValidationError(
                        "Invalid date_from format. Expected: YYYY-MM-DD HH:MM:SS"
                    )
            if date_to:
                try:
                    parsed_date_to = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    raise ValidationError(
                        "Invalid date_to format. Expected: YYYY-MM-DD HH:MM:SS"
                    )
            # Построение фильтров
            filters = {}
            if error_type:
                filters["error_type"] = error_type
            if severity:
                filters["severity"] = severity
            # Запрос к БД
            db_service = DatabaseService()
            # Получаем ошибки из audit_logs (где success=False)
            error_logs = await db_service.get_error_logs(
                filters=filters,
                date_from=parsed_date_from,
                date_to=parsed_date_to,
                limit=limit,
            )
            total = await db_service.get_error_logs_count(
                filters=filters,
                date_from=parsed_date_from,
                date_to=parsed_date_to,
            )
            # Группировка по типу ошибки для статистики
            error_stats = {}
            for log in error_logs:
                error_type = log.get("error_type", "unknown")
                if error_type not in error_stats:
                    error_stats[error_type] = {
                        "count": 0,
                        "last_occurred": None,
                    }
                error_stats[error_type]["count"] += 1
                # Обновляем last_occurred
                occurred_at = log.get("created_at")
                if occurred_at:
                    if (
                        not error_stats[error_type]["last_occurred"]
                        or occurred_at > error_stats[error_type]["last_occurred"]
                    ):
                        error_stats[error_type]["last_occurred"] = occurred_at
            # Audit log
            audit_event(
                action="admin_errors_viewed",
                target_type="admin",
                target_id=current_user.get("user_id"),
                admin_id=current_user.get("user_id"),
                details={"filters": filters, "limit": limit},
                success=True,
            )
            return {
                "success": True,
                "request_id": request_id,
                "errors": error_logs,
                "error_stats": error_stats,
                "total_count": total,
                "limit": limit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        record_business_error("admin_errors_error")
        logger.exception("Error fetching error logs")
        raise HTTPException(status_code=500, detail=str(e))
