import pytest
from pydantic import ValidationError
from app.models.user import (
    UserModel, UserCreate, UserUpdate, UserLogin, 
    UserPasswordChange, UserPasswordReset, UserPasswordResetConfirm,
    UserStats, UserProfile, UserListResponse
)
from app.models.reference import ReferenceCreate
from app.models.verification import VerificationRequest
from app.models.response import ReferenceResponse, VerifyResponse, BaseResponse


class TestUserModels:
    """Тесты для пользовательских моделей"""
    
    def test_user_model_creation(self):
        """Тест создания базовой модели пользователя"""
        user_data = {
            "email": "test@example.com",
            "full_name": "Test User"
        }
        
        user = UserModel(**user_data)
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.is_active is True  # Значение по умолчанию
        assert user.is_verified is False  # Значение по умолчанию
        assert user.total_uploads == 0  # Значение по умолчанию
        assert user.total_verifications == 0  # Значение по умолчанию
        assert user.successful_verifications == 0  # Значение по умолчанию
    
    def test_user_model_default_values(self):
        """Тест значений по умолчанию в модели пользователя"""
        user = UserModel(email="test@example.com")
        
        assert user.is_active is True
        assert user.is_verified is False
        assert user.total_uploads == 0
        assert user.total_verifications == 0
        assert user.successful_verifications == 0
        assert user.id is not None
        assert user.created_at is not None
        assert user.settings is None  # Опциональное поле
    
    def test_user_create_valid(self):
        """Тест создания пользователя с валидными данными"""
        user_data = {
            "email": "test@example.com",
            "full_name": "Test User"
        }
        
        user = UserCreate(**user_data)
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        
    def test_user_create_invalid_email(self):
        """Тест создания пользователя с невалидным email"""
        user_data = {
            "email": "invalid_email"
        }
        
        with pytest.raises(ValidationError):
            UserCreate(**user_data)
    
    def test_user_create_email_validation(self):
        """Тест валидации email при создании пользователя"""
        # Невалидный email
        user_data = {
            "email": "invalid_email"
        }
        
        with pytest.raises(ValidationError):
            UserCreate(**user_data)
        
        # Пустой email
        user_data = {
            "email": ""
        }
        
        with pytest.raises(ValidationError):
            UserCreate(**user_data)
        
        # Слишком длинное имя
        user_data = {
            "email": "test@example.com",
            "full_name": "a" * 256  # Слишком длинное имя
        }
        
        with pytest.raises(ValidationError):
            UserCreate(**user_data)
    
    def test_user_create_phone_validation(self):
        """Тест валидации телефона при создании пользователя"""
        # Валидный телефон (опциональное поле)
        user_data = {
            "email": "test@example.com",
            "phone": "+1234567890"
        }
        
        user = UserCreate(**user_data)
        assert user.phone == "+1234567890"
        
        # Пользователь может быть создан без телефона
        user_data_no_phone = {
            "email": "test2@example.com"
        }
        
        user_no_phone = UserCreate(**user_data_no_phone)
        assert user_no_phone.phone is None
    
    def test_user_update_partial(self):
        """Тест частичного обновления пользователя"""
        user_data = {
            "email": "newemail@example.com",
            "full_name": "New Full Name",
            "is_active": False
        }
        
        user_update = UserUpdate(**user_data)
        
        assert user_update.email == "newemail@example.com"
        assert user_update.full_name == "New Full Name"
        assert user_update.is_active is False
        assert user_update.phone is None  # Не обновлялось
        assert user_update.is_verified is None  # Не обновлялось
    
    def test_user_login(self):
        """Тест модели входа пользователя"""
        login_data = {
            "email": "test@example.com",
            "password": "password123",
            "remember_me": True
        }
        
        login = UserLogin(**login_data)
        
        assert login.email == "test@example.com"
        assert login.password == "password123"
        assert login.remember_me is True
    
    def test_user_password_change(self):
        """Тест модели смены пароля"""
        password_data = {
            "current_password": "oldpassword",
            "new_password": "NewSecure123!",
            "confirm_password": "NewSecure123!"
        }
        
        password_change = UserPasswordChange(**password_data)
        
        assert password_change.current_password == "oldpassword"
        assert password_change.new_password == "NewSecure123!"
        assert password_change.confirm_password == "NewSecure123!"
    
    def test_user_password_change_mismatch(self):
        """Тест несовпадения паролей при смене"""
        password_data = {
            "current_password": "oldpassword",
            "new_password": "NewSecure123!",
            "confirm_password": "DifferentPassword123!"
        }
        
        with pytest.raises(ValidationError):
            UserPasswordChange(**password_data)
    
    def test_user_password_reset(self):
        """Тест модели сброса пароля"""
        reset_data = {
            "email": "test@example.com"
        }
        
        password_reset = UserPasswordReset(**reset_data)
        
        assert password_reset.email == "test@example.com"
    
    def test_user_password_reset_confirm(self):
        """Тест модели подтверждения сброса пароля"""
        confirm_data = {
            "token": "reset_token_123",
            "new_password": "NewSecure123!",
            "confirm_password": "NewSecure123!"
        }
        
        reset_confirm = UserPasswordResetConfirm(**confirm_data)
        
        assert reset_confirm.token == "reset_token_123"
        assert reset_confirm.new_password == "NewSecure123!"
    
    def test_user_stats(self):
        """Тест модели статистики пользователя"""
        stats_data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "total_uploads": 100,
            "total_verifications": 500,
            "successful_verifications": 450,
            "verification_success_rate": 0.9,
            "average_response_time": 0.5,
            "total_references": 10
        }
        
        user_stats = UserStats(**stats_data)
        
        assert user_stats.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert user_stats.total_uploads == 100
        assert user_stats.total_verifications == 500
        assert user_stats.successful_verifications == 450
        assert user_stats.verification_success_rate == 0.9
        assert user_stats.average_response_time == 0.5
        assert user_stats.total_references == 10
    
    def test_user_profile(self):
        """Тест модели профиля пользователя"""
        profile_data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "test@example.com",
            "full_name": "Test User",
            "phone": "+1234567890",
            "created_at": "2024-01-01T00:00:00",
            "is_verified": True
        }
        
        user_profile = UserProfile(**profile_data)
        
        assert user_profile.id == "550e8400-e29b-41d4-a716-446655440000"
        assert user_profile.email == "test@example.com"
        assert user_profile.full_name == "Test User"
        assert user_profile.phone == "+1234567890"
        assert user_profile.is_verified is True
        assert user_profile.stats is None  # Опциональное поле
    
    def test_user_list_response(self):
        """Тест модели списка пользователей"""
        users = [
            UserProfile(
                id="550e8400-e29b-41d4-a716-446655440000",
                email="user1@example.com",
                full_name="User One",
                created_at="2024-01-01T00:00:00",
                is_verified=True
            ),
            UserProfile(
                id="550e8400-e29b-41d4-a716-446655440001",
                email="user2@example.com",
                full_name="User Two",
                created_at="2024-01-02T00:00:00",
                is_verified=False
            )
        ]
        
        list_response = UserListResponse(
            users=users,
            total_count=2,
            page=1,
            per_page=10,
            has_next=False,
            has_prev=False
        )
        
        assert len(list_response.users) == 2
        assert list_response.total_count == 2
        assert list_response.page == 1
        assert list_response.per_page == 10
        assert list_response.has_next is False
        assert list_response.has_prev is False


class TestReferenceModels:
    """Тесты для моделей эталонных изображений"""
    
    def test_reference_create_valid(self):
        """Тест создания эталонного изображения"""
        ref_data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "label": "John Doe",
            "image_data": "base64_encoded_image",
            "metadata": {"source": "upload"}
        }
        
        reference = ReferenceCreate(**ref_data)
        
        assert reference.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert reference.label == "John Doe"
        assert reference.image_data == "base64_encoded_image"
        assert reference.metadata["source"] == "upload"
    
    def test_reference_create_minimal(self):
        """Тест создания эталонного изображения с минимальными данными"""
        ref_data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD",
            "label": "John Doe"
        }
        
        reference = ReferenceCreate(**ref_data)
        
        assert reference.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert reference.label == "John Doe"
        assert reference.image_data == "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
        assert reference.metadata is None  # Опциональное поле
    
    def test_reference_response(self):
        """Тест ответа эталонного изображения"""
        ref_response_data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "label": "John Doe",
            "created_at": "2024-01-01T00:00:00Z",
            "is_active": True,
            "success": True,
            "reference_id": "550e8400-e29b-41d4-a716-446655440000"
        }
        
        reference_response = ReferenceResponse(**ref_response_data)
        
        assert reference_response.id == "550e8400-e29b-41d4-a716-446655440000"
        assert reference_response.label == "John Doe"
        assert reference_response.is_active is True


class TestVerificationModels:
    """Тесты для моделей верификации"""
    
    def test_verification_request_valid(self):
        """Тест запроса верификации"""
        verification_data = {
            "image_data": "base64_encoded_test_image",
            "reference_id": "550e8400-e29b-41d4-a716-446655440000",
            "threshold": 0.8,
            "session_id": "550e8400-e29b-41d4-a716-446655440000"
        }
        
        verification_request = VerificationRequest(**verification_data)
        
        assert verification_request.image_data == "base64_encoded_test_image"
        assert verification_request.reference_id == "550e8400-e29b-41d4-a716-446655440000"
        assert verification_request.threshold == 0.8
        assert verification_request.session_id == "550e8400-e29b-41d4-a716-446655440000"
    
    def test_verification_request_default_threshold(self):
        """Тест запроса верификации с порогом по умолчанию"""
        verification_data = {
            "image_data": "base64_encoded_test_image",
            "reference_id": "550e8400-e29b-41d4-a716-446655440000"
        }
        
        verification_request = VerificationRequest(**verification_data)
        
        assert verification_request.threshold is None  # Порог по умолчанию будет установлен в сервисе
    
    def test_verification_response_match(self):
        """Тест ответа верификации при совпадении"""
        response_data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "verified": True,
            "confidence": 0.95,
            "similarity_score": 0.95,
            "threshold_used": 0.8,
            "processing_time": 0.123,
            "face_detected": True,
            "success": True
        }
        
        verification_response = VerifyResponse(**response_data)
        
        assert verification_response.verified is True
        assert verification_response.confidence == 0.95
        assert verification_response.threshold_used == 0.8
    
    def test_verification_response_no_match(self):
        """Тест ответа верификации при отсутствии совпадения"""
        response_data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "verified": False,
            "confidence": 0.45,
            "similarity_score": 0.45,
            "threshold_used": 0.8,
            "processing_time": 0.098,
            "face_detected": True,
            "success": False
        }
        
        verification_response = VerifyResponse(**response_data)
        
        assert verification_response.verified is False
        assert verification_response.confidence == 0.45


class TestBaseModels:
    """Тесты для базовых моделей"""
    
    def test_upload_request(self):
        """Тест запроса загрузки изображения"""
        from app.models.request import UploadRequest
        
        request_data = {
            "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "metadata": {"source": "mobile_app"}
        }
        
        upload_request = UploadRequest(**request_data)
        
        assert upload_request.image_data == "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
        assert upload_request.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert upload_request.metadata["source"] == "mobile_app"
    
    def test_base_response(self):
        """Тест базового ответа"""
        response_data = {
            "success": True,
            "message": "Operation completed successfully"
        }
        
        base_response = BaseResponse(**response_data)
        
        assert base_response.success is True
        assert base_response.message == "Operation completed successfully"
        assert base_response.request_id is not None  # Auto-generated
        assert base_response.timestamp is not None  # Auto-generated
