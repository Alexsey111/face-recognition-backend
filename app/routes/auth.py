"""
Маршруты аутентификации и авторизации.
Login, logout, refresh tokens и управление сессиями.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, List, AsyncGenerator

from ..services.auth_service import AuthService
from ..services.database_service import BiometricService
from ..utils.validators import validate_email, validate_password
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, UnauthorizedError, NotFoundError
from ..db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

# Схема безопасности для Bearer токенов
security = HTTPBearer(auto_error=False)

# Создание роутера
router = APIRouter(prefix="/auth", tags=["authentication"])


# =============================================================================
# Dependency Functions
# =============================================================================


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency для получения БД сессии.
    """
    from ..db.database import db_manager

    async with db_manager.get_session() as session:
        yield session


def get_auth_service(db: AsyncSession = Depends(get_db_session)) -> AuthService:
    """
    Dependency для получения AuthService с DB session.
    """
    return AuthService(db=db)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session),
) -> str:
    """
    Dependency to extract current user from JWT token.
    Returns user_id.
    """
    # Создаём auth_service БЕЗ db (не нужен для verify_token)
    auth_service = AuthService()
    token = credentials.credentials

    try:
        user_info = await auth_service.get_user_info_from_token(token)
        user_id = user_info.get("user_id")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        # Verify user exists
        db_service = BiometricService(db)
        user = await db_service.get_user(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )

        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive",
            )

        return user_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


# =============================================================================
# Pydantic модели для запросов и ответов
# =============================================================================


class LoginRequest(BaseModel):
    """Запрос на вход в систему."""

    email: EmailStr = Field(..., description="Email пользователя")
    password: str = Field(..., min_length=8, description="Пароль пользователя")


class RegisterRequest(BaseModel):
    """Запрос на регистрацию."""

    email: EmailStr = Field(..., description="Email пользователя")
    password: str = Field(..., min_length=8, description="Пароль пользователя")
    full_name: Optional[str] = Field(None, max_length=100, description="Полное имя")
    phone: Optional[str] = Field(None, description="Номер телефона")


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
    full_name: Optional[str] = Field(None, description="Полное имя")
    role: str = Field(..., description="Роль пользователя")
    permissions: List[str] = Field(
        default_factory=list, description="Разрешения пользователя"
    )
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
    token_info: Optional[Dict[str, Any]] = Field(
        None, description="Информация о токене"
    )


class ChangePasswordRequest(BaseModel):
    """Запрос на смену пароля."""

    current_password: str = Field(..., description="Текущий пароль")
    new_password: str = Field(..., min_length=8, description="Новый пароль")


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""

    error: str = Field(..., description="Код ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Any]] = Field(None, description="Дополнительные детали")


# =============================================================================
# Эндпоинты
# =============================================================================


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Неверные данные запроса"},
        401: {"model": ErrorResponse, "description": "Неверные учетные данные"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации"},
    },
)
async def login(
    request: LoginRequest,
    http_request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Вход в систему с email и паролем.

    - **email**: Email пользователя
    - **password**: Пароль пользователя
    """
    db_service = BiometricService(db)

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
                detail="Invalid email or password",
            )

        # Проверка активности пользователя
        if not user.is_active:
            logger.warning(f"Login failed - user inactive: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated",
            )

        # Проверка пароля
        stored_password_hash = user.password_hash
        if not stored_password_hash:
            logger.error(f"Login failed - no password hash for user: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Верификация пароля
        is_valid_password = await auth_service.verify_password(
            request.password, stored_password_hash
        )

        if not is_valid_password:
            logger.warning(f"Login failed - invalid password for: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Получаем информацию о запросе для логирования
        user_agent = http_request.headers.get("user-agent")
        client_ip = http_request.client.host

        # Создаем сессию с токенами (async)
        session_tokens = await auth_service.create_user_session(
            user_id=user.id, user_agent=user_agent, ip_address=client_ip
        )

        # Формируем информацию о пользователе
        user_info = UserInfo(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role="user",
            permissions=["read_own_data"],
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
        )

        logger.info(f"Login successful for user: {user.id}")

        return LoginResponse(user=user_info, tokens=TokenResponse(**session_tokens))

    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Validation error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/register",
    response_model=LoginResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Неверные данные запроса"},
        409: {"model": ErrorResponse, "description": "Пользователь уже существует"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации"},
    },
)
async def register(
    request: RegisterRequest,
    http_request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Регистрация нового пользователя.

    - **email**: Email пользователя
    - **password**: Пароль пользователя
    - **full_name**: Полное имя (опционально)
    - **phone**: Номер телефона (опционально)
    """
    from ..models.user import UserCreate

    db_service = BiometricService(db)

    try:
        logger.info(f"Registration attempt for email: {request.email}")

        # Валидация входных данных
        validate_email(request.email)
        validate_password(request.password)

        # Проверяем, не существует ли пользователь с таким email
        existing_user = await db_service.get_user_by_email(request.email)
        if existing_user:
            logger.warning(
                f"Registration failed - user already exists: {request.email}"
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists",
            )

        # Хешируем пароль (async для production)
        password_hash = await auth_service.hash_password(request.password)

        # Создаём объект UserCreate
        user_create = UserCreate(
            email=request.email,
            password=request.password,  # UserCreate.password - открытый пароль
            full_name=request.full_name,
            phone=request.phone,
        )

        # Получаем IP и User-Agent для аудита
        user_agent = http_request.headers.get("user-agent")
        client_ip = http_request.client.host

        # Создаём пользователя с аудитом
        new_user = await db_service.create_user_with_audit(
            user=user_create, ip_address=client_ip, user_agent=user_agent
        )

        # Создаем сессию с токенами (async)
        session_tokens = await auth_service.create_user_session(
            user_id=new_user.id, user_agent=user_agent, ip_address=client_ip
        )

        # Формируем информацию о пользователе
        user_info = UserInfo(
            user_id=new_user.id,
            email=new_user.email,
            full_name=new_user.full_name,
            role="user",
            permissions=["read_own_data"],
            is_active=new_user.is_active,
            created_at=new_user.created_at.isoformat(),
        )

        logger.info(f"Registration successful for user: {new_user.id}")

        return LoginResponse(user=user_info, tokens=TokenResponse(**session_tokens))

    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Validation error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Требуется аутентификация"}
    },
)
async def logout(
    request: Request,
    current_user_id: str = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Выход из системы и отзыв токенов.
    """
    try:
        # Получаем токен из заголовка Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return LogoutResponse()

        token = auth_header.split(" ", 1)[1]

        # Проверяем, является ли токен валидным JWT (содержит точки)
        if token and "." in token and len(token.split(".")) == 3:
            try:
                # Отзываем только валидные JWT токены
                await auth_service.revoke_token(token)
            except Exception as e:
                # Логируем ошибку, но не прерываем процесс logout
                logger.warning(f"Error revoking token: {str(e)}")

        logger.info(f"Logout successful for user: {current_user_id}")
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
        422: {"model": ErrorResponse, "description": "Ошибка валидации"},
    },
)
async def refresh_token(
    request: RefreshTokenRequest, auth_service: AuthService = Depends(get_auth_service)
):
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
                detail="Refresh token is required",
            )

        # Используем token rotation (возвращает новые access И refresh)
        tokens = await auth_service.refresh_access_token(request.refresh_token)
        logger.info("Token rotation successful")
        return TokenResponse(**tokens)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during token refresh: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )


@router.get(
    "/me",
    response_model=UserInfo,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Требуется аутентификация"},
        404: {"model": ErrorResponse, "description": "Пользователь не найден"},
    },
)
async def get_current_user_info(
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> UserInfo:
    """
    Получение информации о текущем аутентифицированном пользователе.
    """
    db_service = BiometricService(db)

    try:
        # Получаем пользователя из базы данных
        user = await db_service.get_user(current_user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Проверяем активность пользователя
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated",
            )

        user_response = UserInfo(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role="user",
            permissions=["read_own_data"],
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
        )

        logger.debug(f"User info retrieved for: {current_user_id}")

        return user_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/verify",
    response_model=VerifyTokenResponse,
    status_code=status.HTTP_200_OK,
    responses={401: {"model": ErrorResponse, "description": "Неверный токен"}},
)
async def verify_token(request: Request, db: AsyncSession = Depends(get_db_session)):
    """
    Верификация JWT токена и получение информации о пользователе.
    """
    auth_service = AuthService()
    db_service = BiometricService(db)

    try:
        # Получаем токен из заголовка Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return VerifyTokenResponse(valid=False)

        token = auth_header.split(" ", 1)[1]

        # Проверяем, является ли токен валидным JWT (содержит точки и имеет 3 сегмента)
        if not token or "." not in token or len(token.split(".")) != 3:
            return VerifyTokenResponse(valid=False)

        try:
            # Верифицируем токен
            user_info = await auth_service.get_user_info_from_token(token)
            token_info = auth_service.get_token_info(token)

            # Получаем полную информацию о пользователе
            user = await db_service.get_user(user_info["user_id"])
            if not user:
                return VerifyTokenResponse(valid=False)

            user_response = UserInfo(
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                role="user",
                permissions=["read_own_data"],
                is_active=user.is_active,
                created_at=user.created_at.isoformat(),
            )

            return VerifyTokenResponse(
                valid=True, user_info=user_response, token_info=token_info
            )
        except Exception as e:
            # Логируем ошибку, но возвращаем valid=False
            logger.warning(f"Token verification failed: {str(e)}")
            return VerifyTokenResponse(valid=False)

    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        return VerifyTokenResponse(valid=False)


@router.post(
    "/change-password",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Неверный текущий пароль"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации"},
    },
)
async def change_password(
    request: ChangePasswordRequest,
    current_user_id: str = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Смена пароля аутентифицированного пользователя.

    - **current_password**: Текущий пароль
    - **new_password**: Новый пароль
    """
    from ..models.user import UserUpdate

    db_service = BiometricService(db)

    try:
        # Получаем пользователя
        user = await db_service.get_user(current_user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Проверяем текущий пароль
        stored_password_hash = user.password_hash
        if not stored_password_hash:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password hash not found",
            )

        is_valid_password = await auth_service.verify_password(
            request.current_password, stored_password_hash
        )

        if not is_valid_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )

        # Валидируем новый пароль
        validate_password(request.new_password)

        # Хешируем новый пароль (async для production)
        new_password_hash = await auth_service.hash_password(request.new_password)

        # Обновляем пароль в базе данных (нужно добавить метод update_user_with_audit)
        # Пока используем прямое обновление
        user.password_hash = new_password_hash
        db.add(user)
        await db.commit()

        logger.info(f"Password changed successfully for user: {current_user_id}")

        return {"success": True, "message": "Password changed successfully"}

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
