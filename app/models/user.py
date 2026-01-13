"""
Pydantic модели для работы с пользователями.
Модели для создания, обновления и представления пользователей.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict
from datetime import datetime, timezone
import uuid


class UserModel(BaseModel):
    """
    Базовая модель пользователя.
    """
    model_config = ConfigDict(from_attributes=True)

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
        default_factory=lambda: datetime.now(timezone.utc), description="Дата создания"
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


class UserCreate(BaseModel):
    """
    Модель для создания пользователя.
    """
    email: EmailStr = Field(..., description="Email адрес")
    phone: Optional[str] = Field(None, description="Телефон")
    full_name: Optional[str] = Field(None, max_length=255, description="Полное имя")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        return v


class UserUpdate(BaseModel):
    """
    Модель для обновления пользователя.
    """
    phone: Optional[str] = Field(None, description="Телефон")
    full_name: Optional[str] = Field(None, max_length=255, description="Полное имя")
    is_active: Optional[bool] = Field(None, description="Активен ли пользователь")
    settings: Optional[Dict[str, Any]] = Field(None, description="Настройки пользователя")


class UserResponse(BaseModel):
    """
    Модель ответа с данными пользователя.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Уникальный ID пользователя")
    email: EmailStr = Field(..., description="Email адрес")
    phone: Optional[str] = Field(None, description="Телефон")
    full_name: Optional[str] = Field(None, description="Полное имя")
    is_active: bool = Field(..., description="Активен ли пользователь")
    is_verified: bool = Field(..., description="Верифицирован ли пользователь")
    created_at: datetime = Field(..., description="Дата создания")
    updated_at: Optional[datetime] = Field(None, description="Дата обновления")
    last_verified_at: Optional[datetime] = Field(None, description="Последняя верификация")
    total_uploads: int = Field(..., description="Общее количество загрузок")
    total_verifications: int = Field(..., description="Общее количество верификаций")
    successful_verifications: int = Field(..., description="Успешные верификации")
    settings: Optional[Dict[str, Any]] = Field(None, description="Настройки пользователя")


class UserListResponse(BaseModel):
    """
    Модель для списка пользователей.
    """
    users: List[UserModel] = Field(..., description="Список пользователей")
    total_count: int = Field(..., description="Общее количество пользователей")
    page: int = Field(..., description="Номер страницы")
    per_page: int = Field(..., description="Пользователей на странице")
    has_next: bool = Field(..., description="Есть ли следующая страница")
    has_prev: bool = Field(..., description="Есть ли предыдущая страница")
