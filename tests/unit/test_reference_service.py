"""
Unit-тесты для ReferenceService.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import hashlib
from datetime import datetime, timezone

from app.services.reference_service import ReferenceService
from app.db.models import Reference
from app.utils.exceptions import ValidationError, ProcessingError, NotFoundError


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def reference_service(
    db_session,
    mock_ml_service,
    mock_encryption_service,
    mock_validation_service,
    mock_storage_service,
):
    """Создание ReferenceService с замоканными зависимостями."""
    service = ReferenceService(db_session)
    
    # Патчим зависимости
    service.ml_service = mock_ml_service
    service.encryption_service = mock_encryption_service
    service.validation_service = mock_validation_service
    service.storage_service = mock_storage_service
    
    return service


@pytest.fixture
def mock_reference(mock_user_id, mock_reference_id, mock_embedding):
    """Mock Reference объект."""
    return Reference(
        id=mock_reference_id,
        user_id=mock_user_id,
        embedding_encrypted=b"encrypted_" + mock_embedding[:20],
        embedding_hash=hashlib.sha256(b"encrypted_" + mock_embedding[:20]).hexdigest(),
        quality_score=0.85,
        image_filename="test.jpg",
        image_size_mb=0.5,
        image_format="JPEG",
        file_url="https://storage.example.com/test.jpg",
        face_landmarks=[[100, 100], [200, 100]],
        label="test_reference",
        version=1,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


# ======================================================================
# Тесты create_reference
# ======================================================================

@pytest.mark.asyncio
async def test_create_reference_success(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Тест успешного создания reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        # Mock создания reference - используем AsyncMock для async метода
        mock_crud.create_reference = AsyncMock(return_value=Reference(
            id="new-ref-123",
            user_id=mock_user_id,
            quality_score=0.85,
            version=1,
            is_active=True,
        ))
        
        # Mock получения предыдущего reference для calculate_similarity_with_old
        mock_crud.get_reference_by_id = AsyncMock(return_value=None)
        
        # Создание reference
        result = await reference_service.create_reference(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
            label="test",
            quality_threshold=0.7,
        )
        
        # Проверки
        assert result is not None
        assert result.user_id == mock_user_id
        assert result.quality_score == 0.85
        
        # Проверяем, что ML сервис был вызван
        reference_service.ml_service.generate_embedding.assert_called_once()
        
        # Проверяем, что шифрование было выполнено
        reference_service.encryption_service.encrypt_embedding.assert_called_once()


@pytest.mark.asyncio
async def test_create_reference_low_quality(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Тест отклонения reference из-за низкого качества."""
    
    # Устанавливаем низкий quality_score
    reference_service.ml_service.generate_embedding.return_value = {
        "success": True,
        "embedding": b"fake_embedding",
        "quality_score": 0.3,  # Ниже порога
        "face_detected": True,
    }
    
    # Ожидаем ValidationError
    with pytest.raises(ValidationError, match="quality.*below threshold"):
        await reference_service.create_reference(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
            quality_threshold=0.7,
        )


@pytest.mark.asyncio
async def test_create_reference_embedding_failed(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Тест ошибки генерации embedding."""
    
    # Устанавливаем неудачный результат
    reference_service.ml_service.generate_embedding.return_value = {
        "success": False,
        "error": "Face not detected",
    }
    
    # Ожидаем ProcessingError
    with pytest.raises(ProcessingError, match="Embedding generation failed"):
        await reference_service.create_reference(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
        )


@pytest.mark.asyncio
async def test_create_reference_with_versioning(
    reference_service,
    mock_user_id,
    mock_image_data,
    mock_reference,
):
    """Тест создания reference с версионированием."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        # Mock получения предыдущего reference
        async def mock_get_latest(user_id):
            return mock_reference
        
        reference_service.get_latest_reference = mock_get_latest
        
        # Mock создания нового reference
        mock_crud.create_reference = AsyncMock(return_value=Reference(
            id="new-ref-456",
            user_id=mock_user_id,
            version=2,  # Должна быть версия 2
            previous_reference_id=mock_reference.id,
            is_active=True,
        ))
        
        # Mock get_reference_by_id для calculate_similarity_with_old
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        
        # Создание reference
        result = await reference_service.create_reference(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
        )
        
        # Проверяем версию
        assert result.version == 2
        assert result.previous_reference_id == mock_reference.id


# ======================================================================
# Тесты get_reference
# ======================================================================

@pytest.mark.asyncio
async def test_get_reference_success(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Тест успешного получения reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        
        result = await reference_service.get_reference(mock_reference_id)
        
        assert result is not None
        assert result.id == mock_reference_id


@pytest.mark.asyncio
async def test_get_reference_not_found(
    reference_service,
    mock_reference_id,
):
    """Тест получения несуществующего reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=None)
        
        result = await reference_service.get_reference(mock_reference_id)
        
        assert result is None


# ======================================================================
# Тесты update_reference
# ======================================================================

@pytest.mark.asyncio
async def test_update_reference_success(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Тест успешного обновления reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        
        updated_ref = Reference(
            id=mock_reference.id,
            user_id=mock_reference.user_id,
            label="updated_label",
            quality_score=mock_reference.quality_score,
            version=mock_reference.version,
            is_active=mock_reference.is_active,
        )
        mock_crud.update_reference = AsyncMock(return_value=updated_ref)
        
        result = await reference_service.update_reference(
            reference_id=mock_reference_id,
            label="updated_label",
        )

        assert result.label == "updated_label"


@pytest.mark.asyncio
async def test_update_reference_not_found(
    reference_service,
    mock_reference_id,
):
    """Тест обновления несуществующего reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=None)
        
        with pytest.raises(NotFoundError, match="not found"):
            await reference_service.update_reference(
                reference_id=mock_reference_id,
                label="new_label",
            )


# ======================================================================
# Тесты delete_reference
# ======================================================================

@pytest.mark.asyncio
async def test_delete_reference_soft(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Тест soft delete reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_crud.update_reference = AsyncMock(return_value=mock_reference)
        
        result = await reference_service.delete_reference(
            reference_id=mock_reference_id,
            soft_delete=True,
        )
        
        assert result is True
        mock_crud.update_reference.assert_called_once()


@pytest.mark.asyncio
async def test_delete_reference_hard(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Тест hard delete reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_crud.delete_reference = AsyncMock(return_value=True)
        
        result = await reference_service.delete_reference(
            reference_id=mock_reference_id,
            soft_delete=False,
        )
        
        assert result is True
        mock_crud.delete_reference.assert_called_once()


# ======================================================================
# Тесты compare_with_references
# ======================================================================

@pytest.mark.asyncio
async def test_compare_with_references_success(
    reference_service,
    mock_image_data,
    mock_reference,
):
    """Тест успешного сравнения с references."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        # Mock получения references
        async def mock_get_all(user_id, include_inactive=False):
            return [mock_reference]
        
        reference_service.get_all_references = mock_get_all
        
        # Сравнение
        results = await reference_service.compare_with_references(
            image_data=mock_image_data.encode(),
            user_id=mock_reference.user_id,
            threshold=0.6,
        )
        
        # Проверки
        assert len(results) > 0
        assert results[0]["reference_id"] == mock_reference.id
        assert results[0]["similarity_score"] >= 0.6
        assert results[0]["is_match"] is True


@pytest.mark.asyncio
async def test_compare_with_references_no_match(
    reference_service,
    mock_image_data,
    mock_reference,
):
    """Тест сравнения без совпадений."""
    
    # Устанавливаем низкий similarity
    reference_service.ml_service.compare_faces.return_value = {
        "success": True,
        "similarity_score": 0.3,
        "distance": 0.7,
    }
    
    async def mock_get_all(user_id, include_inactive=False):
        return [mock_reference]
    
    reference_service.get_all_references = mock_get_all
    
    results = await reference_service.compare_with_references(
        image_data=mock_image_data.encode(),
        user_id=mock_reference.user_id,
        threshold=0.6,
    )
    
    # Должен быть результат, но is_match=False
    assert len(results) > 0
    assert results[0]["is_match"] is False


@pytest.mark.asyncio
async def test_compare_with_references_no_user_id_no_reference_ids(
    reference_service,
    mock_image_data,
):
    """Тест сравнения без user_id и reference_ids."""
    
    with pytest.raises(ValidationError, match="user_id.*must be provided"):
        await reference_service.compare_with_references(
            image_data=mock_image_data.encode(),
        )


# ======================================================================
# Тесты calculate_similarity_with_old
# ======================================================================

@pytest.mark.asyncio
async def test_calculate_similarity_with_old_success(
    reference_service,
    mock_reference_id,
    mock_reference,
    mock_embedding,
):
    """Тест расчёта similarity с предыдущим reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        
        similarity = await reference_service.calculate_similarity_with_old(
            new_embedding=mock_embedding,
            old_reference_id=mock_reference_id,
        )
        
        assert similarity > 0.0
        assert similarity <= 1.0


@pytest.mark.asyncio
async def test_calculate_similarity_with_old_not_found(
    reference_service,
    mock_reference_id,
    mock_embedding,
):
    """Тест расчёта similarity с несуществующим reference."""
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=None)
        
        similarity = await reference_service.calculate_similarity_with_old(
            new_embedding=mock_embedding,
            old_reference_id=mock_reference_id,
        )
        
        # Должен вернуть 0.0 при отсутствии reference
        assert similarity == 0.0


# ======================================================================
# Тесты get_reference_statistics
# ======================================================================

@pytest.mark.asyncio
async def test_get_reference_statistics(
    reference_service,
    mock_user_id,
    mock_reference,
):
    """Тест получения статистики по references."""
    
    now = datetime.now(timezone.utc)
    
    # Mock нескольких references
    ref1 = Reference(
        id="ref-1",
        user_id=mock_user_id,
        quality_score=0.85,
        version=1,
        is_active=True,
        created_at=now,
    )
    ref2 = Reference(
        id="ref-2",
        user_id=mock_user_id,
        quality_score=0.75,
        version=1,
        is_active=True,
        created_at=now,
    )
    ref3 = Reference(
        id="ref-3",
        user_id=mock_user_id,
        quality_score=0.90,
        version=1,
        is_active=False,
        created_at=now,
    )
    references = [ref1, ref2, ref3]
    
    async def mock_get_all(user_id, include_inactive=True):
        return references
    
    reference_service.get_all_references = mock_get_all
    
    stats = await reference_service.get_reference_statistics(mock_user_id)
    
    # Проверки
    assert stats["total_references"] == 3
    assert stats["active_references"] == 2
    assert stats["inactive_references"] == 1
    assert stats["average_quality_score"] > 0.0
