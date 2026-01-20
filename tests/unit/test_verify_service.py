"""
Unit-тесты для VerifyService.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta

from app.services.verify_service import VerifyService
from app.db.models import VerificationSession, Reference
from app.utils.exceptions import ValidationError, ProcessingError, NotFoundError


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def verify_service(
    db_session,
    mock_ml_service,
    mock_encryption_service,
    mock_validation_service,
    mock_cache_service,
    mock_webhook_service,
):
    """Создание VerifyService с замоканными зависимостями."""
    service = VerifyService(db_session)
    
    service.ml_service = mock_ml_service
    service.encryption_service = mock_encryption_service
    service.validation_service = mock_validation_service
    service.cache_service = mock_cache_service
    service.webhook_service = mock_webhook_service
    
    return service


@pytest.fixture
def mock_reference(mock_user_id, mock_reference_id, mock_embedding):
    """Mock Reference объект."""
    return Reference(
        id=mock_reference_id,
        user_id=mock_user_id,
        embedding_encrypted=b"encrypted_" + mock_embedding[:20],
        quality_score=0.85,
        version=1,
        is_active=True,
    )


@pytest.fixture
def mock_verification_session(mock_user_id, mock_reference_id):
    """Mock VerificationSession объект."""
    return VerificationSession(
        id="session-123",
        session_id="session-123",
        user_id=mock_user_id,
        reference_id=mock_reference_id,
        session_type="verification",
        status="pending",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
    )


# ======================================================================
# Тесты verify_face
# ======================================================================

@pytest.mark.asyncio
async def test_verify_face_success(
    verify_service,
    mock_user_id,
    mock_image_data,
    mock_reference,
):
    """Тест успешной верификации."""
    
    # Mock получения reference
    async def mock_get_ref(user_id):
        return mock_reference
    
    verify_service._get_user_reference = mock_get_ref
    
    # Mock сохранения результата
    async def mock_save(*args, **kwargs):
        return VerificationSession(
            id="ver-123",
            session_id="session-123",
            user_id=mock_user_id,
        )
    
    verify_service._save_verification = mock_save
    verify_service._cache_verification_result = AsyncMock()
    
    # Верификация
    result = await verify_service.verify_face(
        user_id=mock_user_id,
        image_data=mock_image_data.encode(),
        threshold=0.8,
    )
    
    # Проверки
    assert result["verified"] is True
    assert result["similarity_score"] >= 0.8
    assert result["confidence"] > 0.0
    assert "verification_id" in result
    assert "processing_time" in result


@pytest.mark.asyncio
async def test_verify_face_no_reference(
    verify_service,
    mock_user_id,
    mock_image_data,
):
    """Тест верификации без reference."""
    
    # Mock отсутствия reference
    async def mock_get_ref(user_id):
        return None
    
    verify_service._get_user_reference = mock_get_ref
    
    # Ожидаем NotFoundError
    with pytest.raises(NotFoundError, match="Reference.*not found"):
        await verify_service.verify_face(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
        )


@pytest.mark.asyncio
async def test_verify_face_low_similarity(
    verify_service,
    mock_user_id,
    mock_image_data,
    mock_reference,
):
    """Тест верификации с низким similarity."""
    
    # Устанавливаем низкий similarity
    verify_service.ml_service.verify_face.return_value = {
        "success": True,
        "similarity": 0.4,  # Ниже порога
        "confidence": 0.5,
        "liveness": 0.6,
        "quality_score": 0.7,
        "face_detected": True,
    }
    
    async def mock_get_ref(user_id):
        return mock_reference
    
    verify_service._get_user_reference = mock_get_ref
    
    async def mock_save(*args, **kwargs):
        return VerificationSession(id="ver-123", session_id="session-123")
    
    verify_service._save_verification = mock_save
    verify_service._cache_verification_result = AsyncMock()
    
    result = await verify_service.verify_face(
        user_id=mock_user_id,
        image_data=mock_image_data.encode(),
        threshold=0.8,
    )
    
    # Верификация должна провалиться
    assert result["verified"] is False
    assert result["similarity_score"] < 0.8


@pytest.mark.asyncio
async def test_verify_face_ml_failure(
    verify_service,
    mock_user_id,
    mock_image_data,
    mock_reference,
):
    """Тест верификации с ошибкой ML."""
    
    # Устанавливаем ошибку ML
    verify_service.ml_service.verify_face.return_value = {
        "success": False,
        "error": "Face not detected",
    }
    
    async def mock_get_ref(user_id):
        return mock_reference
    
    verify_service._get_user_reference = mock_get_ref
    
    # Ожидаем ProcessingError
    with pytest.raises(ProcessingError, match="ML verification failed"):
        await verify_service.verify_face(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
        )


# ======================================================================
# Тесты calculate_dynamic_threshold
# ======================================================================

def test_calculate_dynamic_threshold_with_requested(verify_service):
    """Тест динамического порога с явным значением."""
    
    threshold = verify_service.calculate_dynamic_threshold(
        requested_threshold=0.75,
        quality_score=0.8,
    )
    
    # Должен использовать запрошенный порог
    assert threshold == 0.75


def test_calculate_dynamic_threshold_high_quality(verify_service):
    """Тест динамического порога с высоким качеством."""
    
    threshold = verify_service.calculate_dynamic_threshold(
        requested_threshold=None,
        quality_score=0.9,  # Высокое качество
        base_threshold=0.65,
    )
    
    # Порог должен быть немного ниже базового (качество выше 0.5)
    assert threshold < 0.65


def test_calculate_dynamic_threshold_low_quality(verify_service):
    """Тест динамического порога с низким качеством."""
    
    threshold = verify_service.calculate_dynamic_threshold(
        requested_threshold=None,
        quality_score=0.3,  # Низкое качество
        base_threshold=0.65,
    )
    
    # Порог должен быть немного выше базового (качество ниже 0.5)
    assert threshold > 0.65


def test_calculate_dynamic_threshold_clamping(verify_service):
    """Тест ограничения порога в допустимых пределах."""
    
    # Очень высокий запрошенный порог
    threshold_high = verify_service.calculate_dynamic_threshold(
        requested_threshold=0.99,
        quality_score=0.8,
    )
    
    # Не должен превышать MAX
    assert threshold_high <= 0.95
    
    # Очень низкий запрошенный порог
    threshold_low = verify_service.calculate_dynamic_threshold(
        requested_threshold=0.1,
        quality_score=0.8,
    )
    
    # Не должен быть ниже MIN
    assert threshold_low >= 0.5


# ======================================================================
# Тесты determine_confidence_level
# ======================================================================

def test_determine_confidence_level_high(verify_service):
    """Тест определения высокого уровня confidence."""
    
    level = verify_service.determine_confidence_level(
        similarity=0.95,
        threshold=0.8,
        quality_score=0.9,
    )
    
    assert level == "high"


def test_determine_confidence_level_medium(verify_service):
    """Тест определения среднего уровня confidence."""
    
    level = verify_service.determine_confidence_level(
        similarity=0.75,
        threshold=0.7,
        quality_score=0.7,
    )
    
    assert level in ["medium", "high"]


def test_determine_confidence_level_low(verify_service):
    """Тест определения низкого уровня confidence."""
    
    level = verify_service.determine_confidence_level(
        similarity=0.55,
        threshold=0.5,
        quality_score=0.6,
    )
    
    assert level in ["low", "medium"]


def test_determine_confidence_level_downgrade_on_low_quality(verify_service):
    """Тест понижения confidence из-за низкого качества."""
    
    # Высокий similarity, но низкое качество
    level = verify_service.determine_confidence_level(
        similarity=0.95,
        threshold=0.8,
        quality_score=0.3,  # Низкое качество
    )
    
    # Должно быть понижено с high до medium
    assert level in ["medium", "low"]


# ======================================================================
# Тесты create_verification_session
# ======================================================================

@pytest.mark.asyncio
async def test_create_verification_session(
    verify_service,
    mock_user_id,
    mock_reference_id,
):
    """Тест создания сессии верификации."""
    
    with patch("app.services.verify_service.VerificationSessionCRUD") as mock_crud:
        mock_session = VerificationSession(
            session_id="new-session-123",
            user_id=mock_user_id,
            reference_id=mock_reference_id,
            session_type="verification",
            status="pending",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        mock_crud.create_session = AsyncMock(return_value=mock_session)
        
        session = await verify_service.create_verification_session(
            user_id=mock_user_id,
            reference_id=mock_reference_id,
            expires_in_minutes=30,
        )
        
        assert session.user_id == mock_user_id
        assert session.reference_id == mock_reference_id
        assert session.status == "pending"


# ======================================================================
# Тесты get_verification_history
# ======================================================================

@pytest.mark.asyncio
async def test_get_verification_history(
    verify_service,
    mock_user_id,
):
    """Тест получения истории верификаций."""
    
    with patch("app.services.verify_service.VerificationSessionCRUD") as mock_crud:
        # Mock сессий
        sessions = [
            VerificationSession(
                session_id=f"session-{i}",
                user_id=mock_user_id,
                status="completed",
                is_match=(i % 2 == 0),
                created_at=datetime.now(timezone.utc),
            )
            for i in range(10)
        ]
        mock_crud.get_user_sessions = AsyncMock(return_value=sessions)
        
        history = await verify_service.get_verification_history(
            user_id=mock_user_id,
            limit=5,
            offset=0,
        )
        
        assert len(history) <= 5


@pytest.mark.asyncio
async def test_get_verification_history_with_filters(
    verify_service,
    mock_user_id,
):
    """Тест получения истории с фильтрами."""
    
    with patch("app.services.verify_service.VerificationSessionCRUD") as mock_crud:
        sessions = [
            VerificationSession(
                session_id=f"session-{i}",
                user_id=mock_user_id,
                status="completed",
                is_match=True,
                created_at=datetime.now(timezone.utc),
            )
            for i in range(5)
        ]
        mock_crud.get_user_sessions = AsyncMock(return_value=sessions)
        
        history = await verify_service.get_verification_history(
            user_id=mock_user_id,
            filters={"verified": True},
            limit=10,
        )
        
        # Все результаты должны быть verified=True
        for session in history:
            assert session.is_match is True


# ======================================================================
# Тесты send_verification_webhook
# ======================================================================

@pytest.mark.asyncio
async def test_send_verification_webhook(
    verify_service,
    mock_user_id,
):
    """Тест отправки webhook."""
    
    await verify_service.send_verification_webhook(
        user_id=mock_user_id,
        verification_id="ver-123",
        verified=True,
        similarity=0.92,
        confidence=0.88,
    )
    
    # Проверяем, что webhook был вызван
    verify_service.webhook_service.emit_event.assert_called_once()
    
    # Проверяем payload
    call_args = verify_service.webhook_service.emit_event.call_args
    assert call_args[1]["event_type"] == "face.verified"
    assert call_args[1]["user_id"] == mock_user_id
    assert call_args[1]["payload"]["verified"] is True
