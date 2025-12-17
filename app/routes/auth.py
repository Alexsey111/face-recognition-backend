"""
Маршруты аутентификации и авторизации.
Login, logout, refresh tokens и управление сессиями.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, List

from ..services.auth_service import AuthService
from ..services.database_service import DatabaseService
from ..utils.validators import validate_email, validate_password
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, UnauthorizedError, NotFoundError
from ..middleware.auth import RequireAuth, RequireRole

logger = get_logger(__name__)

# Схема безопасности для Bearer токенов
security = HTTPBearer(auto_error=False)


# =============================================================================
# Зависимости FastAPI
# =============================================================================

async def get_current_user(request: Request) -> str:
    """
    Зависимость FastAPI для получения текущего пользователя из JWT токена.
    
    Используется как Depends(get_current_user) в endpoints.
    
    Returns:
        str: ID пользователя
        
    Raises:
        HTTPException: Если токен недействителен или отсутствует
    """
    try:
        # Получаем токен из заголовка Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing or invalid"
            )
        
        token = auth_header.split(" ", 1)[1]
        
        # Верифицируем токен и получаем информацию о пользователе
        user_info = await auth_service.get_user_info_from_token(token)
        
        if not user_info or "user_id" not in user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Проверяем, что пользователь существует и активен
        user = await db_service.get_user(user_info["user_id"])
        if not user or not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        return user_info["user_id"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

# Инициализация сервисов
auth_service = AuthService()
db_service = DatabaseService()

# Создание роутера
router = APIRouter(prefix="/auth", tags=["authentication"])

# Pydantic модели для запросов и ответов


class LoginRequest(BaseModel):
    """Запрос на вход в систему."""
    email: EmailStr = Field(..., description="Email пользователя")
    password: str = Field(..., min_length=8, description="Пароль пользователя")


class RegisterRequest(BaseModel):
    """Запрос на регистрацию."""
    email: EmailStr = Field(..., description="Email пользователя")
    username: str = Field(..., min_length=3, max_length=50, description="Имя пользователя")
    password: str = Field(..., min_length=8, description="Пароль пользователя")
    full_name: Optional[str] = Field(None, max_length=100, description="Полное имя")


class TokenResponse(BaseModel):
    """Ответ с токенами."""
    access_token: str = Field(..., description="Access токен")
    refresh_token: str = Field(..., description="Refresh токен")
    token_type: str = Field(default="bearer", description="Тип токена")
    expires_in: int = Field(..., description="Время жизни токена в секундах")


class RefreshTokenRequest(BaseModel):
    """Запрос на обновление токена."""
    refresh_token: str = Field(..., description="Refresh токен")


class UserInfo(BaseModel):
    """Информация о пользователе."""
    user_id: str = Field(..., description="ID пользователя")
    email: str = Field(..., description="Email пользователя")
    username: str = Field(..., description="Имя пользователя")
    role: str = Field(..., description="Роль пользователя")
    permissions: List[str] = Field(default_factory=list, description="Разрешения пользователя")
    is_active: bool = Field(..., description="Активен ли пользователь")
    created_at: str = Field(..., description="Дата создания аккаунта")


class LoginResponse(BaseModel):
    """Ответ на успешный вход."""
    success: bool = Field(default=True, description="Успешность операции")
    message: str = Field(default="Login successful", description="Сообщение")
    user: UserInfo = Field(..., description="Информация о пользователе")
    tokens: TokenResponse = Field(..., description="Токены доступа")


class LogoutResponse(BaseModel):
    """Ответ на выход из системы."""
    success: bool = Field(default=True, description="Успешность операции")
    message: str = Field(default="Logout successful", description="Сообщение")


class VerifyTokenResponse(BaseModel):
    """Ответ верификации токена."""
    valid: bool = Field(..., description="Валиден ли токен")
    user_info: Optional[UserInfo] = Field(None, description="Информация о пользователе")
    token_info: Optional[Dict[str, Any]] = Field(None, description="Информация о токене")


class ChangePasswordRequest(BaseModel):
    """Запрос на смену пароля."""
    current_password: str = Field(..., description="Текущий пароль")
    new_password: str = Field(..., min_length=8, description="Новый пароль")


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    error: str = Field(..., description="Код ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Any]] = Field(None, description="Дополнительные детали")


# Эндпоинты


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Неверные данные запроса"},
        401: {"model": ErrorResponse, "description": "Неверные учетные данные"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации"}
    }
)
async def login(request: LoginRequest, http_request: Request):
    """
    Вход в систему с email и паролем.
    
    - **email**: Email пользователя
    - **password**: Пароль пользователя
    """
    try:
        logger.info(f"Login attempt for email: {request.email}")
        
        # Валидация входных данных
        validate_email(request.email)
        validate_password(request.password)
        
        # Поиск пользователя в базе данных
        user = await db_service.get_user_by_email(request.email)
        if not user:
            logger.warning(f"Login failed - user not found: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Проверка активности пользователя
        if not user.get("is_active", True):
            logger.warning(f"Login failed - user inactive: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Проверка пароля
        stored_password_hash = user.get("password_hash")
        if not stored_password_hash:
            logger.error(f"Login failed - no password hash for user: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Верификация пароля
        is_valid_password = await auth_service.verify_password(
            request.password, 
            stored_password_hash
        )
        
        if not is_valid_password:
            logger.warning(f"Login failed - invalid password for: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Получаем информацию о запросе для логирования
        user_agent = http_request.headers.get("user-agent")
        client_ip = http_request.client.host
        
        # Создаем сессию с токенами
        session_tokens = await auth_service.create_user_session(
            user_id=user["id"],
            user_agent=user_agent,
            ip_address=client_ip
        )
        
        # Формируем информацию о пользователе
        user_info = UserInfo(
            user_id=user["id"],
            email=user["email"],
            username=user["username"],
            role=user.get("role", "user"),
            permissions=user.get("permissions", []),
            is_active=user.get("is_active", True),
            created_at=user.get("created_at", "")
        )
        
        logger.info(f"Login successful for user: {user['id']}")
        
        return LoginResponse(
            user=user_info,
            tokens=TokenResponse(**session_tokens)
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Validation error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/register",
    response_model=LoginResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Неверные данные запроса"},
        409: {"model": ErrorResponse, "description": "Пользователь уже существует"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации"}
    }
)
async def register(request: RegisterRequest):
    """
    Регистрация нового пользователя.
    
    - **email**: Email пользователя
    - **username**: Имя пользователя
    - **password**: Пароль пользователя
    - **full_name**: Полное имя (опционально)
    """
    try:
        logger.info(f"Registration attempt for email: {request.email}")
        
        # Валидация входных данных
        validate_email(request.email)
        validate_password(request.password)
        
        # Проверяем, не существует ли пользователь с таким email
        existing_user = await db_service.get_user_by_email(request.email)
        if existing_user:
            logger.warning(f"Registration failed - user already exists: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )
        
        # Проверяем username
        existing_username = await db_service.get_user_by_username(request.username)
        if existing_username:
            logger.warning(f"Registration failed - username taken: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username is already taken"
            )
        
        # Хешируем пароль
        password_hash = await auth_service.hash_password(request.password)
        
        # Создаем пользователя
        user_data = {
            "email": request.email,
            "username": request.username,
            "password_hash": password_hash,
            "full_name": request.full_name,
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True
        }
        
        new_user = await db_service.create_user(user_data)
        
        # Создаем сессию с токенами
        session_tokens = await auth_service.create_user_session(
            user_id=new_user["id"]
        )
        
        # Формируем информацию о пользователе
        user_info = UserInfo(
            user_id=new_user["id"],
            email=new_user["email"],
            username=new_user["username"],
            role=new_user.get("role", "user"),
            permissions=new_user.get("permissions", []),
            is_active=new_user.get("is_active", True),
            created_at=new_user.get("created_at", "")
        )
        
        logger.info(f"Registration successful for user: {new_user['id']}")
        
        return LoginResponse(
            user=user_info,
            tokens=TokenResponse(**session_tokens)
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Validation error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(RequireAuth())],
    responses={
        401: {"model": ErrorResponse, "description": "Требуется аутентификация"}
    }
)
async def logout(request: Request):
    """
    Выход из системы и отзыв токенов.
    """
    try:
        # Получаем токен из заголовка Authorization
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            
            # Отзываем токен
            await auth_service.revoke_token(token)
        
        logger.info(f"Logout successful for user: {getattr(request.state, 'user_id', 'unknown')}")
        
        return LogoutResponse()
        
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        # Всегда возвращаем успех для logout
        return LogoutResponse()


@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Неверный refresh токен"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации"}
    }
)
async def refresh_token(request: RefreshTokenRequest):
    """
    Обновление access токена с помощью refresh токена.
    
    - **refresh_token**: Refresh токен
    """
    try:
        logger.info("Token refresh attempt")
        
        # Валидация refresh токена
        if not request.refresh_token:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Refresh token is required"
            )
        
        # ✅ Используем token rotation (возвращает новые access И refresh)
        tokens = await auth_service.refresh_access_token(request.refresh_token)
        logger.info(f"Token rotation successful")
        return TokenResponse(**tokens)  # ✅ Оба токена новые
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during token refresh: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.get(
    "/me",
    response_model=UserInfo,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(RequireAuth())],
    responses={
        401: {"model": ErrorResponse, "description": "Требуется аутентификация"},
        404: {"model": ErrorResponse, "description": "Пользователь не найден"}
    }
)
async def get_current_user_info(request: Request):
    """
    Получение информации о текущем аутентифицированном пользователе.
    """
    try:
        user_id = request.state.user_id
        
        # Получаем пользователя из базы данных
        user = await db_service.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Проверяем активность пользователя
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        user_info = UserInfo(
            user_id=user["id"],
            email=user["email"],
            username=user["username"],
            role=user.get("role", "user"),
            permissions=user.get("permissions", []),
            is_active=user.get("is_active", True),
            created_at=user.get("created_at", "")
        )
        
        logger.debug(f"User info retrieved for: {user_id}")
        
        return user_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/verify",
    response_model=VerifyTokenResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Неверный токен"}
    }
)
async def verify_token(request: Request):
    """
    Верификация JWT токена и получение информации о пользователе.
    """
    try:
        # Получаем токен из заголовка Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return VerifyTokenResponse(valid=False)
        
        token = auth_header.split(" ", 1)[1]
        
        # Верифицируем токен
        user_info = await auth_service.get_user_info_from_token(token)
        token_info = auth_service.get_token_info(token)
        
        # Получаем полную информацию о пользователе
        user = await db_service.get_user(user_info["user_id"])
        if not user:
            return VerifyTokenResponse(valid=False)
        
        user_response = UserInfo(
            user_id=user["id"],
            email=user["email"],
            username=user["username"],
            role=user.get("role", "user"),
            permissions=user.get("permissions", []),
            is_active=user.get("is_active", True),
            created_at=user.get("created_at", "")
        )
        
        return VerifyTokenResponse(
            valid=True,
            user_info=user_response,
            token_info=token_info
        )
        
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        return VerifyTokenResponse(valid=False)


@router.post(
    "/change-password",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(RequireAuth())],
    responses={
        400: {"model": ErrorResponse, "description": "Неверный текущий пароль"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации"}
    }
)
async def change_password(request: ChangePasswordRequest, request_state: Request):
    """
    Смена пароля аутентифицированного пользователя.
    
    - **current_password**: Текущий пароль
    - **new_password**: Новый пароль
    """
    try:
        user_id = request_state.state.user_id
        
        # Получаем пользователя
        user = await db_service.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Проверяем текущий пароль
        stored_password_hash = user.get("password_hash")
        if not stored_password_hash:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password hash not found"
            )
        
        is_valid_password = await auth_service.verify_password(
            request.current_password,
            stored_password_hash
        )
        
        if not is_valid_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Валидируем новый пароль
        validate_password(request.new_password)
        
        # Хешируем новый пароль
        new_password_hash = await auth_service.hash_password(request.new_password)
        
        # Обновляем пароль в базе данных
        await db_service.update_user(user_id, {"password_hash": new_password_hash})
        
        logger.info(f"Password changed successfully for user: {user_id}")
        
        return {"success": True, "message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )