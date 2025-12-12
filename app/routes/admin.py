"""API роуты для администрирования."""

from fastapi import APIRouter, HTTPException, Request, Query
from datetime import datetime, timezone, timedelta
import uuid
import psutil

from ..models.response import AdminStatsResponse, BaseResponse
from ..models.user import UserListResponse, UserUpdate
from ..models.reference import ReferenceStats
from ..models.verification import SessionStats
from ..services.database_service import DatabaseService
from ..services.cache_service import CacheService
from ..services.storage_service import StorageService
from ..services.ml_service import MLService
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, NotFoundError

router = APIRouter(prefix="/api/v1", tags=["Admin"])

logger = get_logger(__name__)


@router.get("/admin/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    date_from: str = Query(None, description="Начальная дата (YYYY-MM-DD)"),
    date_to: str = Query(None, description="Конечная дата (YYYY-MM-DD)"),
    include_user_stats: bool = Query(
        False, description="Включить статистику по пользователям"
    ),
    include_performance: bool = Query(
        True, description="Включить метрики производительности"
    ),
    http_request: Request = None,
):
    """
    Получение общей статистики сервиса.

    Args:
        date_from: Начальная дата периода
        date_to: Конечная дата периода
        include_user_stats: Включить статистику по пользователям
        include_performance: Включить метрики производительности
        http_request: HTTP запрос

    Returns:
        AdminStatsResponse: Общая статистика сервиса
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Getting admin stats, request {request_id}")

        # Валидация дат
        if date_from:
            try:
                datetime.strptime(date_from, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid date_from format. Expected YYYY-MM-DD")

        if date_to:
            try:
                datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                raise ValidationError("Invalid date_to format. Expected YYYY-MM-DD")

        # Инициализация сервисов
        db_service = DatabaseService()

        # Определение периода
        if not date_from and not date_to:
            # По умолчанию - последние 30 дней
            date_to = datetime.now(timezone.utc).date()
            date_from = date_to - timedelta(days=30)
        elif date_from and not date_to:
            date_to = datetime.now(timezone.utc).date()
        elif not date_from and date_to:
            date_from = date_to - timedelta(days=30)

        # Получение базовой статистики
        total_requests = await db_service.get_total_requests_count(date_from, date_to)
        successful_requests = await db_service.get_successful_requests_count(
            date_from, date_to
        )
        failed_requests = await db_service.get_failed_requests_count(date_from, date_to)
        average_response_time = await db_service.get_average_response_time(
            date_from, date_to
        )

        # Статистика верификации
        verification_stats = await db_service.get_verification_stats(date_from, date_to)

        # Статистика проверки живости
        liveness_stats = await db_service.get_liveness_stats(date_from, date_to)

        # Статистика по пользователям (если запрошена)
        user_stats = None
        if include_user_stats:
            user_stats = await db_service.get_user_stats(date_from, date_to)

        # Метрики производительности (если запрошены)
        performance_metrics = None
        if include_performance:
            performance_metrics = await get_performance_metrics()

        response = AdminStatsResponse(
            success=True,
            period={"from": date_from.isoformat(), "to": date_to.isoformat()},
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=average_response_time,
            verification_stats=verification_stats,
            liveness_stats=liveness_stats,
            user_stats=user_stats,
            performance_metrics=performance_metrics,
            request_id=request_id,
        )

        logger.info(f"Admin stats retrieved successfully, request {request_id}")
        return response

    except ValidationError as e:
        logger.warning(
            f"Validation error getting admin stats, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(f"Error getting admin stats, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.get("/admin/logs", response_model=dict)
async def get_logs(
    level: str = Query("INFO", description="Уровень логирования"),
    service: str = Query(None, description="Фильтр по сервису"),
    date_from: str = Query(None, description="Начальная дата (YYYY-MM-DD HH:MM:SS)"),
    date_to: str = Query(None, description="Конечная дата (YYYY-MM-DD HH:MM:SS)"),
    limit: int = Query(
        100, ge=1, le=1000, description="Максимальное количество записей"
    ),
    http_request: Request = None,
):
    """
    Получение логов системы.

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service: Фильтр по сервису
        date_from: Начальная дата
        date_to: Конечная дата
        limit: Максимальное количество записей
        http_request: HTTP запрос

    Returns:
        dict: Логи системы
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Getting logs, request {request_id}")

        # Валидация уровня логирования
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level not in allowed_levels:
            raise ValidationError(f"Invalid log level. Allowed: {allowed_levels}")

        # Инициализация сервисов
        db_service = DatabaseService()

        # Парсинг дат
        parsed_date_from = None
        parsed_date_to = None

        if date_from:
            try:
                parsed_date_from = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValidationError(
                    "Invalid date_from format. Expected YYYY-MM-DD HH:MM:SS"
                )

        if date_to:
            try:
                parsed_date_to = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValidationError(
                    "Invalid date_to format. Expected YYYY-MM-DD HH:MM:SS"
                )

        # Получение логов из БД
        logs = await db_service.get_logs(
            level=level,
            service=service,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            limit=limit,
        )

        response = {
            "success": True,
            "request_id": request_id,
            "logs": logs,
            "total_count": len(logs),
            "filters_applied": {
                "level": level,
                "service": service,
                "date_from": date_from,
                "date_to": date_to,
                "limit": limit,
            },
            "timestamp": datetime.now(timezone.utc),
        }

        logger.info(f"Retrieved {len(logs)} log entries, request {request_id}")
        return response

    except ValidationError as e:
        logger.warning(f"Validation error getting logs, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(f"Error getting logs, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.get("/admin/users", response_model=UserListResponse)
async def get_all_users(
    page: int = Query(1, ge=1, description="Номер страницы"),
    per_page: int = Query(
        20, ge=1, le=100, description="Количество пользователей на странице"
    ),
    role: str = Query(None, description="Фильтр по роли"),
    is_active: bool = Query(None, description="Фильтр по статусу активности"),
    search: str = Query(None, description="Поиск по имени или email"),
    sort_by: str = Query("created_at", description="Поле для сортировки"),
    sort_order: str = Query("desc", description="Порядок сортировки"),
    http_request: Request = None,
):
    """
    Получение списка всех пользователей (админ функция).

    Args:
        page: Номер страницы
        per_page: Количество пользователей на странице
        role: Фильтр по роли
        is_active: Фильтр по статусу активности
        search: Поиск по имени или email
        sort_by: Поле для сортировки
        sort_order: Порядок сортировки
        http_request: HTTP запрос

    Returns:
        UserListResponse: Список пользователей
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Getting all users, request {request_id}")

        # Инициализация сервисов
        db_service = DatabaseService()

        # Валидация параметров
        allowed_sort_fields = [
            "created_at",
            "updated_at",
            "username",
            "email",
            "last_login",
        ]
        if sort_by not in allowed_sort_fields:
            raise ValidationError(
                f"Invalid sort_by field. Allowed: {allowed_sort_fields}"
            )

        if sort_order not in ["asc", "desc"]:
            raise ValidationError("sort_order must be 'asc' or 'desc'")

        # Построение фильтров
        filters = {}
        if role:
            filters["role"] = role
        if is_active is not None:
            filters["is_active"] = is_active
        if search:
            filters["search"] = search

        # Получение пользователей
        result = await db_service.get_all_users(
            filters=filters,
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        response = UserListResponse(
            success=True,
            users=result["users"],
            total_count=result["total_count"],
            page=page,
            per_page=per_page,
            has_next=result["has_next"],
            has_prev=result["has_prev"],
            request_id=request_id,
        )

        logger.info(f"Retrieved {len(result['users'])} users, request {request_id}")
        return response

    except ValidationError as e:
        logger.warning(
            f"Validation error getting users, request {request_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(f"Error getting users, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.put("/admin/users/{user_id}", response_model=dict)
async def update_user(user_id: str, request: UserUpdate, http_request: Request = None):
    """
    Обновление пользователя (админ функция).

    Args:
        user_id: ID пользователя
        request: Данные для обновления
        http_request: HTTP запрос

    Returns:
        dict: Обновленный пользователь
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Updating user {user_id}, request {request_id}")

        # Инициализация сервисов
        db_service = DatabaseService()

        # Проверка существования пользователя
        existing_user = await db_service.get_user_by_id(user_id)
        if not existing_user:
            raise NotFoundError(f"User {user_id} not found")

        # Обновление пользователя
        updated_user = await db_service.update_user(user_id, request.dict())

        # Логирование действия администратора
        await db_service.log_admin_action(
            admin_id="current_admin_id",  # TODO: Получить из контекста аутентификации
            action="update_user",
            target_type="user",
            target_id=user_id,
            details={"updated_fields": request.dict()},
            ip_address=http_request.client.host if http_request.client else None,
        )

        response = {
            "success": True,
            "user": updated_user,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc),
        }

        logger.info(f"User {user_id} updated successfully, request {request_id}")
        return response

    except NotFoundError as e:
        logger.warning(f"User {user_id} not found, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error_code": "USER_NOT_FOUND",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )

    except Exception as e:
        logger.error(f"Error updating user {user_id}, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.get("/admin/references/stats", response_model=ReferenceStats)
async def get_references_stats(http_request: Request):
    """
    Получение статистики по эталонным изображениям.

    Args:
        http_request: HTTP запрос

    Returns:
        ReferenceStats: Статистика по эталонам
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Getting references stats, request {request_id}")

        # Инициализация сервисов
        db_service = DatabaseService()

        # Получение статистики
        stats = await db_service.get_references_statistics()

        response = ReferenceStats(**stats)
        response.request_id = request_id

        logger.info(f"References stats retrieved successfully, request {request_id}")
        return response

    except Exception as e:
        logger.error(f"Error getting references stats, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.get("/admin/sessions/stats", response_model=SessionStats)
async def get_sessions_stats(http_request: Request):
    """
    Получение статистики по сессиям верификации.

    Args:
        http_request: HTTP запрос

    Returns:
        SessionStats: Статистика по сессиям
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Getting sessions stats, request {request_id}")

        # Инициализация сервисов
        db_service = DatabaseService()

        # Получение статистики
        stats = await db_service.get_sessions_statistics()

        response = SessionStats(**stats)
        response.request_id = request_id

        logger.info(f"Sessions stats retrieved successfully, request {request_id}")
        return response

    except Exception as e:
        logger.error(f"Error getting sessions stats, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.get("/admin/system/health", response_model=dict)
async def get_system_health(http_request: Request):
    """
    Получение детальной информации о состоянии системы.

    Args:
        http_request: HTTP запрос

    Returns:
        dict: Информация о состоянии системы
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Getting system health, request {request_id}")

        # Проверка всех сервисов
        services_status = {}

        # Проверка БД
        try:
            db_service = DatabaseService()
            await db_service.health_check()
            services_status["database"] = {"status": "healthy", "response_time": None}
        except Exception as e:
            services_status["database"] = {"status": "unhealthy", "error": str(e)}

        # Проверка Redis
        try:
            cache_service = CacheService()
            await cache_service.health_check()
            services_status["redis"] = {"status": "healthy", "response_time": None}
        except Exception as e:
            services_status["redis"] = {"status": "unhealthy", "error": str(e)}

        # Проверка хранилища
        try:
            storage_service = StorageService()
            await storage_service.health_check()
            services_status["storage"] = {"status": "healthy", "response_time": None}
        except Exception as e:
            services_status["storage"] = {"status": "unhealthy", "error": str(e)}

        # Проверка ML сервиса
        try:
            ml_service = MLService()
            await ml_service.health_check()
            services_status["ml_service"] = {"status": "healthy", "response_time": None}
        except Exception as e:
            services_status["ml_service"] = {"status": "unhealthy", "error": str(e)}

        # Системная информация
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "load_average": (
                psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            ),
            "boot_time": psutil.boot_time(),
            "python_version": psutil.sys.version,
            "process_count": len(psutil.pids()),
        }

        response = {
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

        logger.info(f"System health retrieved successfully, request {request_id}")
        return response

    except Exception as e:
        logger.error(f"Error getting system health, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


@router.post("/admin/maintenance", response_model=BaseResponse)
async def toggle_maintenance_mode(
    enabled: bool = Query(..., description="Включить режим обслуживания"),
    message: str = Query(None, description="Сообщение о режиме обслуживания"),
    http_request: Request = None,
):
    """
    Включение/выключение режима обслуживания.

    Args:
        enabled: Включить режим обслуживания
        message: Сообщение о режиме обслуживания
        http_request: HTTP запрос

    Returns:
        BaseResponse: Результат операции
    """
    request_id = str(uuid.uuid4())

    try:
        logger.info(f"Toggling maintenance mode to {enabled}, request {request_id}")

        # Инициализация сервисов
        cache_service = CacheService()

        # Сохранение статуса режима обслуживания
        maintenance_data = {
            "enabled": enabled,
            "message": message or "System is under maintenance",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "admin_id": "current_admin_id",  # TODO: Получить из контекста
        }

        await cache_service.set_maintenance_mode(maintenance_data)

        # Логирование действия
        mode_text = "enabled" if enabled else "disabled"
        logger.info(f"Maintenance mode {mode_text} by admin, request {request_id}")

        response = BaseResponse(
            success=True,
            message=f"Maintenance mode {mode_text} successfully",
            request_id=request_id,
        )

        logger.info(f"Maintenance mode toggled successfully, request {request_id}")
        return response

    except Exception as e:
        logger.error(f"Error toggling maintenance mode, request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_details": {"error": str(e)},
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
            },
        )


async def get_performance_metrics() -> dict:
    """
    Получение метрик производительности системы.

    Returns:
        dict: Метрики производительности
    """
    try:
        # Базовые системные метрики
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Метрики процесса
        process = psutil.Process()

        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available": memory.available,
            "disk_usage": disk.percent,
            "disk_free": disk.free,
            "process_memory": process.memory_info().rss,
            "process_cpu": process.cpu_percent(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return {}
