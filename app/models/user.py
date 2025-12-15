"""
Pydantic модели для работы с пользователями.
Модели для создания, обновления и представления пользователей.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, validator
from datetime import datetime
import uuid


class UserModel(BaseModel):
    """
    Базовая модель пользователя.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Уникальный ID пользователя",
    )
    email: EmailStr = Field(..., description="Email адрес")
    phone: Optional[str] = Field(None, description="Телефон")
    full_name: Optional[str] = Field(None, max_length=255, description="Полное имя")
    is_active: bool = Field(default=True, description="Активен ли пользователь")
    is_verified: bool = Field(
        default=False, description="Верифицирован ли пользователь"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Дата создания"
    )
    updated_at: Optional[datetime] = Field(
        None, description="Дата последнего обновления"
    )
    last_verified_at: Optional[datetime] = Field(None, description="Последняя верификация")

    # Статистика использования
    total_uploads: int = Field(default=0, description="Общее количество загрузок")
    total_verifications: int = Field(
        default=0, description="Общее количество верификаций"
    )
    successful_verifications: int = Field(default=0, description="Успешные верификации")

    # Настройки пользователя
    settings: Optional[Dict[str, Any]] = Field(
        None, description="Настройки пользователя"
    )

    class Config:
        orm_mode = True


class UserCreate(BaseModel):
    """
    Модель для создания пользователя.
    """

    email: EmailStr = Field(..., description="Email адрес")
    phone: Optional[str] = Field(None, description="Телефон")
    full_name: Optional[str] = Field(None, max_length=255, description="Полное имя")


class UserUpdate(BaseModel):
    """
    Модель для обновления пользователя.
    """

    email: Optional[EmailStr] = Field(None, description="Email адрес")
    phone: Optional[str] = Field(None, description="Телефон")
    full_name: Optional[str] = Field(None, max_length=255, description="Полное имя")
    is_active: Optional[bool] = Field(None, description="Активен ли пользователь")
    is_verified: Optional[bool] = Field(
        None, description="Верифицирован ли пользователь"
    )
    settings: Optional[Dict[str, Any]] = Field(
        None, description="Настройки пользователя"
    )


class UserLogin(BaseModel):
    """
    Модель для входа пользователя.
    """

    email: EmailStr = Field(..., description="Email адрес")
    password: str = Field(..., description="Пароль")
    remember_me: bool = Field(default=False, description="Запомнить меня")


class UserPasswordChange(BaseModel):
    """
    Модель для смены пароля.
    """

    current_password: str = Field(..., description="Текущий пароль")
    new_password: str = Field(..., min_length=8, description="Новый пароль")
    confirm_password: str = Field(..., description="Подтверждение нового пароля")

    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Проверка совпадения паролей."""
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v

    @validator("new_password")
    def validate_new_password(cls, v):
        """Валидация нового пароля."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # Проверяем наличие разных типов символов
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)

        if not (has_upper and has_lower and has_digit):
            raise ValueError(
                "Password must contain at least one uppercase letter, one lowercase letter, and one digit"
            )

        return v


class UserPasswordReset(BaseModel):
    """
    Модель для сброса пароля.
    """

    email: EmailStr = Field(..., description="Email адрес")


class UserPasswordResetConfirm(BaseModel):
    """
    Модель для подтверждения сброса пароля.
    """

    token: str = Field(..., description="Токен сброса пароля")
    new_password: str = Field(..., min_length=8, description="Новый пароль")
    confirm_password: str = Field(..., description="Подтверждение нового пароля")

    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Проверка совпадения паролей."""
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v

    @validator("new_password")
    def validate_new_password(cls, v):
        """Валидация нового пароля."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # Проверяем наличие разных типов символов
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)

        if not (has_upper and has_lower and has_digit):
            raise ValueError(
                "Password must contain at least one uppercase letter, one lowercase letter, and one digit"
            )

        return v


class UserStats(BaseModel):
    """
    Модель для статистики пользователя.
    """

    user_id: str = Field(..., description="ID пользователя")
    total_uploads: int = Field(..., description="Общее количество загрузок")
    total_verifications: int = Field(..., description="Общее количество верификаций")
    successful_verifications: int = Field(..., description="Успешные верификации")
    verification_success_rate: float = Field(
        ..., description="Процент успешных верификаций"
    )
    last_upload: Optional[datetime] = Field(None, description="Последняя загрузка")
    last_verification: Optional[datetime] = Field(
        None, description="Последняя верификация"
    )
    average_response_time: float = Field(..., description="Среднее время ответа")
    total_references: int = Field(..., description="Общее количество эталонов")


class UserProfile(BaseModel):
    """
    Модель для профиля пользователя (публичная информация).
    """

    id: str = Field(..., description="ID пользователя")
    email: EmailStr = Field(..., description="Email адрес")
    full_name: Optional[str] = Field(None, description="Полное имя")
    phone: Optional[str] = Field(None, description="Телефон")
    created_at: datetime = Field(..., description="Дата регистрации")
    is_verified: bool = Field(..., description="Верифицирован ли пользователь")
    stats: Optional[UserStats] = Field(None, description="Статистика пользователя")


class UserListResponse(BaseModel):
    """
    Модель для списка пользователей.
    """

    users: List[UserProfile] = Field(..., description="Список пользователей")
    total_count: int = Field(..., description="Общее количество пользователей")
    page: int = Field(..., description="Номер текущей страницы")
    per_page: int = Field(..., description="Количество пользователей на странице")
    has_next: bool = Field(..., description="Есть ли следующая страница")
    has_prev: bool = Field(..., description="Есть ли предыдущая страница")
