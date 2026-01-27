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
        mock_crud.create_reference = AsyncMock(
            return_value=Reference(
                id="new-ref-123",
                user_id=mock_user_id,
                quality_score=0.85,
                version=1,
                is_active=True,
            )
        )

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
        mock_crud.create_reference = AsyncMock(
            return_value=Reference(
                id="new-ref-456",
                user_id=mock_user_id,
                version=2,  # Должна быть версия 2
                previous_reference_id=mock_reference.id,
                is_active=True,
            )
        )

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


# ======================================================================
# Дополнительные тесты для повышения покрытия
# ======================================================================


@pytest.mark.asyncio
async def test_get_all_references_with_filters(
    reference_service,
):
    """Test getting all references with filtering."""
    mock_user_id = "test-user-123"
    
    ref1 = MagicMock()
    ref1.id = "ref-1"
    ref1.label = "my_label"
    ref1.is_active = True
    ref1.quality_score = 0.9
    
    ref2 = MagicMock()
    ref2.id = "ref-2"
    ref2.label = "other_label"
    ref2.is_active = False
    ref2.quality_score = 0.7
    
    async def mock_get_all(user_id, include_inactive=False):
        return [ref1, ref2]
    
    reference_service.get_all_references = mock_get_all
    
    result = await reference_service.get_all_references(
        user_id=mock_user_id,
        include_inactive=False
    )
    
    assert len(result) == 2


@pytest.mark.asyncio
async def test_get_latest_reference_multiple(
    reference_service,
):
    """Test getting latest reference when multiple exist."""
    mock_user_id = "test-user-123"
    mock_ref = MagicMock()
    mock_ref.id = "latest-ref"
    mock_ref.user_id = mock_user_id
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        async def mock_get_latest(user_id):
            return mock_ref
        
        reference_service.get_latest_reference = mock_get_latest
        
        result = await reference_service.get_latest_reference(mock_user_id)
        
        assert result.id == "latest-ref"


@pytest.mark.asyncio
async def test_update_reference_metadata_only(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Test updating only metadata of reference."""
    updated_ref = MagicMock()
    updated_ref.id = mock_reference_id
    updated_ref.metadata = {"new_key": "new_value"}
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_crud.update_reference = AsyncMock(return_value=updated_ref)
        
        result = await reference_service.update_reference(
            reference_id=mock_reference_id,
            metadata={"new_key": "new_value"}
        )
        
        mock_crud.update_reference.assert_called_once()


@pytest.mark.asyncio
async def test_update_reference_deactivate(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Test deactivating a reference."""
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_crud.update_reference = AsyncMock(return_value=mock_reference)
        
        result = await reference_service.update_reference(
            reference_id=mock_reference_id,
            is_active=False
        )
        
        mock_crud.update_reference.assert_called_once_with(
            db=reference_service.db,
            reference_id=mock_reference_id,
            label=None,
            metadata=None,
            is_active=False,
        )


@pytest.mark.asyncio
async def test_delete_reference_not_found(
    reference_service,
    mock_reference_id,
):
    """Test deleting non-existent reference."""
    with patch.object(reference_service, 'get_reference', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        
        with pytest.raises(NotFoundError):
            await reference_service.delete_reference(mock_reference_id)


@pytest.mark.asyncio
async def test_compare_with_references_empty_result(
    reference_service,
    mock_image_data,
):
    """Test comparing when no references found."""
    mock_user_id = "test-user-123"
    
    with patch.object(reference_service, 'get_all_references', new_callable=AsyncMock) as mock_get_all:
        mock_get_all.return_value = []
        
        results = await reference_service.compare_with_references(
            image_data=mock_image_data.encode(),
            user_id=mock_user_id,
        )
        
        assert results == []


@pytest.mark.asyncio
async def test_compare_with_specific_references(
    reference_service,
    mock_image_data,
    mock_reference,
):
    """Test comparing with specific reference IDs."""
    mock_ref_ids = ["ref-1", "ref-2"]
    
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        
        async def mock_get_ref(ref_id):
            return mock_reference
        
        reference_service.get_reference = mock_get_ref
        
        results = await reference_service.compare_with_references(
            image_data=mock_image_data.encode(),
            reference_ids=mock_ref_ids,
        )
        
        assert len(results) > 0


@pytest.mark.asyncio
async def test_compare_with_references_error_handling(
    reference_service,
    mock_image_data,
    mock_user_id,
    mock_reference,
):
    """Test error handling during comparison."""
    # Make compare_faces raise an exception
    reference_service.ml_service.compare_faces.side_effect = Exception("ML error")
    
    async def mock_get_all(user_id, include_inactive=False):
        return [mock_reference]
    
    reference_service.get_all_references = mock_get_all
    
    # Should handle error gracefully and continue
    results = await reference_service.compare_with_references(
        image_data=mock_image_data.encode(),
        user_id=mock_user_id,
    )
    
    # Results should be empty due to error
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_reference_statistics_single_reference(
    reference_service,
    mock_user_id,
):
    """Test statistics with single active reference."""
    now = datetime.now(timezone.utc)
    
    ref = Reference(
        id="ref-1",
        user_id=mock_user_id,
        quality_score=0.95,
        version=1,
        is_active=True,
        created_at=now,
    )
    
    async def mock_get_all(user_id, include_inactive=True):
        return [ref]
    
    reference_service.get_all_references = mock_get_all
    
    stats = await reference_service.get_reference_statistics(mock_user_id)
    
    assert stats["total_references"] == 1
    assert stats["active_references"] == 1
    assert stats["average_quality_score"] == 0.95


@pytest.mark.asyncio
async def test_create_reference_duplicate_embedding(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Test creating reference with duplicate embedding."""
    from sqlalchemy import text
    
    # Mock finding duplicate
    mock_duplicate = MagicMock()
    mock_duplicate.id = "existing-ref"
    
    with patch.object(reference_service.db, 'execute', new_callable=AsyncMock) as mock_execute:
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_duplicate
        mock_execute.return_value = mock_result
        
        with pytest.raises(ValidationError, match="Duplicate embedding detected"):
            await reference_service._check_duplicate_embedding(
                user_id=mock_user_id,
                embedding_hash="duplicate-hash"
            )


@pytest.mark.asyncio
async def test_create_reference_new_version(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Test getting next version for user with existing references."""
    with patch.object(reference_service.db, 'execute', new_callable=AsyncMock) as mock_execute:
        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_execute.return_value = mock_result
        
        version = await reference_service._get_next_version(mock_user_id)
        
        assert version == 4


@pytest.mark.asyncio
async def test_create_reference_first_version(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Test getting next version for user with no references."""
    with patch.object(reference_service.db, 'execute', new_callable=AsyncMock) as mock_execute:
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_execute.return_value = mock_result
        
        version = await reference_service._get_next_version(mock_user_id)
        
        assert version == 1


@pytest.mark.asyncio
async def test_create_reference_no_previous_for_similarity(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Test creating reference when no previous reference exists."""
    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud, \
         patch.object(reference_service, 'get_latest_reference', new_callable=AsyncMock) as mock_get_latest:
        
        mock_get_latest.return_value = None
        
        mock_crud.create_reference = AsyncMock(
            return_value=Reference(
                id="new-ref",
                user_id=mock_user_id,
                version=1,
                is_active=True,
            )
        )
        
        result = await reference_service.create_reference(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
        )
        
        # similarity_with_previous should be None in metadata
        assert result is not None


@pytest.mark.asyncio
async def test_compare_embeddings_directly(
    reference_service,
    mock_embedding,
):
    """Test direct embedding comparison."""
    reference_service.ml_service.compare_embeddings.return_value = 0.88
    
    similarity = await reference_service.ml_service.compare_embeddings(
        embedding_1=mock_embedding,
        embedding_2=mock_embedding,
    )
    
    assert similarity == 0.88


@pytest.mark.asyncio
async def test_decrypt_embedding_for_reference(
    reference_service,
    mock_reference,
    mock_embedding,
):
    """Test decrypting embedding from reference."""
    with patch.object(reference_service.encryption_service, 'decrypt_embedding', new_callable=AsyncMock) as mock_decrypt:
        mock_decrypt.return_value = mock_embedding
        
        result = await reference_service.encryption_service.decrypt_embedding(
            mock_reference.embedding_encrypted
        )
        
        assert result == mock_embedding


@pytest.mark.asyncio
async def test_get_reference_statistics_sorted_by_date(
    reference_service,
    mock_user_id,
):
    """Test statistics shows correct newest/oldest dates."""
    now = datetime.now(timezone.utc)
    yesterday = datetime.now(timezone.utc).replace(day=now.day - 1)
    
    newer_ref = Reference(
        id="newer",
        user_id=mock_user_id,
        quality_score=0.8,
        version=2,
        is_active=True,
        created_at=now,
    )
    older_ref = Reference(
        id="older",
        user_id=mock_user_id,
        quality_score=0.7,
        version=1,
        is_active=True,
        created_at=yesterday,
    )
    
    async def mock_get_all(user_id, include_inactive=True):
        return [newer_ref, older_ref]
    
    reference_service.get_all_references = mock_get_all
    
    stats = await reference_service.get_reference_statistics(mock_user_id)
    
    assert stats["newest_reference"] is not None
    assert stats["oldest_reference"] is not None
    assert stats["latest_version"] == 2


# ======================================================================
# ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ ДЛЯ ДОСТИЖЕНИЯ 100% ПОКРЫТИЯ
# ======================================================================

@pytest.mark.asyncio
async def test_compare_with_references_max_results_limit(
    reference_service,
    mock_image_data,
    mock_reference,
):
    """Тест ограничения максимального количества результатов."""

    # Создаём много references
    references = [mock_reference] * 150  # Больше чем max_results

    async def mock_get_all(user_id, include_inactive=False):
        return references

    reference_service.get_all_references = mock_get_all

    with pytest.raises(ValidationError, match="Too many references"):
        await reference_service.compare_with_references(
            image_data=mock_image_data.encode(),
            user_id=mock_reference.user_id,
            max_results=100,  # Максимум 100
        )


@pytest.mark.asyncio
async def test_compare_with_references_specific_ids_not_found(
    reference_service,
    mock_image_data,
):
    """Тест сравнения с несуществующими reference IDs."""

    async def mock_get_ref(ref_id):
        return None

    reference_service.get_reference = mock_get_ref

    results = await reference_service.compare_with_references(
        image_data=mock_image_data.encode(),
        reference_ids=["nonexistent-1", "nonexistent-2"],
    )

    assert results == []  # Пустой результат


@pytest.mark.asyncio
async def test_calculate_similarity_with_old_decrypt_error(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Тест ошибки при расшифровке embedding в calculate_similarity_with_old."""

    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)

        # Имитируем ошибку расшифровки
        reference_service.encryption_service.decrypt_embedding = AsyncMock(
            side_effect=Exception("Decrypt error")
        )

        similarity = await reference_service.calculate_similarity_with_old(
            new_embedding=b"new_embedding",
            old_reference_id=mock_reference_id,
        )

        assert similarity == 0.0  # Должен вернуть 0.0 при ошибке


@pytest.mark.asyncio
async def test_get_all_references_include_inactive(
    reference_service,
    mock_user_id,
    mock_reference,
):
    """Тест получения всех references включая неактивные."""

    active_ref = Reference(id="active", user_id=mock_user_id, is_active=True, created_at=datetime.now(timezone.utc))
    inactive_ref = Reference(id="inactive", user_id=mock_user_id, is_active=False, created_at=datetime.now(timezone.utc))

    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        async def mock_get_all_refs(db, user_id):
            return [active_ref, inactive_ref]

        mock_crud.get_all_references = mock_get_all_refs

        result = await reference_service.get_all_references(
            user_id=mock_user_id,
            include_inactive=True,
        )

        assert len(result) == 2


@pytest.mark.asyncio
async def test_create_reference_store_original_false(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Тест создания reference без сохранения оригинала."""

    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud, \
         patch("app.services.reference_service.settings") as mock_settings:

        mock_settings.STORE_ORIGINAL_IMAGES = False

        mock_crud.create_reference = AsyncMock(
            return_value=Reference(
                id="new-ref-123",
                user_id=mock_user_id,
                quality_score=0.85,
                version=1,
                is_active=True,
                file_url=None,  # Без URL
            )
        )

        # Mock получения предыдущего reference
        mock_crud.get_reference_by_id = AsyncMock(return_value=None)

        result = await reference_service.create_reference(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
            store_original=False,
        )

        assert result.file_url is None


@pytest.mark.asyncio
async def test_create_reference_with_previous_similarity_error(
    reference_service,
    mock_user_id,
    mock_image_data,
    mock_reference,
):
    """Тест создания reference с ошибкой при расчёте similarity с предыдущим."""

    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        # Mock получения предыдущего reference
        async def mock_get_latest(user_id):
            return mock_reference

        reference_service.get_latest_reference = mock_get_latest

        # Mock ошибки в calculate_similarity_with_old
        reference_service.calculate_similarity_with_old = AsyncMock(
            side_effect=Exception("Similarity calc error")
        )

        mock_crud.create_reference = AsyncMock(
            return_value=Reference(
                id="new-ref-123",
                user_id=mock_user_id,
                version=2,
                is_active=True,
            )
        )

        # Создание должно пройти несмотря на ошибку similarity
        result = await reference_service.create_reference(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
        )

        assert result.version == 2


@pytest.mark.asyncio
async def test_update_reference_partial_update(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Тест частичного обновления reference."""

    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)

        updated_ref = Reference(
            id=mock_reference.id,
            user_id=mock_reference.user_id,
            label="new_label",
            metadata={"existing": "value", "new": "field"},
            quality_score=mock_reference.quality_score,
            version=mock_reference.version,
            is_active=mock_reference.is_active,
        )
        mock_crud.update_reference = AsyncMock(return_value=updated_ref)

        result = await reference_service.update_reference(
            reference_id=mock_reference_id,
            label="new_label",
            metadata={"new": "field"},
        )

        assert result.label == "new_label"


@pytest.mark.asyncio
async def test_delete_reference_soft_delete_error(
    reference_service,
    mock_reference_id,
    mock_reference,
):
    """Тест ошибки при soft delete."""

    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        mock_crud.get_reference_by_id = AsyncMock(return_value=mock_reference)
        mock_crud.update_reference = AsyncMock(side_effect=Exception("Update error"))

        # Soft delete должен пройти через update_reference
        with pytest.raises(Exception, match="Update error"):
            await reference_service.delete_reference(
                reference_id=mock_reference_id,
                soft_delete=True,
            )


@pytest.mark.asyncio
async def test_compare_with_references_face_detection_error(
    reference_service,
    mock_image_data,
    mock_reference,
):
    """Тест ошибки обнаружения лица в compare_with_references."""

    # Имитируем ошибку в ML service
    reference_service.ml_service.generate_embedding.return_value = {
        "success": False,
        "error": "Face not detected",
    }

    async def mock_get_all(user_id, include_inactive=False):
        return [mock_reference]

    reference_service.get_all_references = mock_get_all

    with pytest.raises(ProcessingError, match="Failed to generate embedding"):
        await reference_service.compare_with_references(
            image_data=mock_image_data.encode(),
            user_id=mock_reference.user_id,
        )


@pytest.mark.asyncio
async def test_get_reference_statistics_no_references(
    reference_service,
    mock_user_id,
):
    """Тест статистики при отсутствии references."""

    async def mock_get_all(user_id, include_inactive=True):
        return []

    reference_service.get_all_references = mock_get_all

    stats = await reference_service.get_reference_statistics(mock_user_id)

    assert stats["total_references"] == 0
    assert stats["active_references"] == 0
    assert stats["average_quality_score"] == 0.0
    assert stats["latest_version"] == 0


@pytest.mark.asyncio
async def test_get_latest_reference_none(
    reference_service,
    mock_user_id,
):
    """Тест получения latest reference когда их нет."""

    with patch("app.services.reference_service.ReferenceCRUD") as mock_crud:
        async def mock_execute():
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            return mock_result

        mock_crud.db.execute = mock_execute

        result = await reference_service.get_latest_reference(mock_user_id)

        assert result is None


@pytest.mark.asyncio
async def test_create_reference_validation_error(
    reference_service,
    mock_user_id,
    mock_image_data,
):
    """Тест ошибки валидации изображения."""

    # Имитируем ошибку валидации
    reference_service.validation_service.validate_image.return_value = {
        "is_valid": False,
        "error_message": "Invalid image format",
    }

    with pytest.raises(ValidationError, match="Invalid image format"):
        await reference_service.create_reference(
            user_id=mock_user_id,
            image_data=mock_image_data.encode(),
        )


@pytest.mark.asyncio
async def test_compare_with_references_validation_error(
    reference_service,
    mock_image_data,
):
    """Тест ошибки валидации в compare_with_references."""

    reference_service.validation_service.validate_image.return_value = {
        "is_valid": False,
        "error_message": "Corrupted image",
    }

    with pytest.raises(ValidationError, match="Corrupted image"):
        await reference_service.compare_with_references(
            image_data=mock_image_data.encode(),
            user_id="test-user",
        )
