"""
Unit-тест фикстуры с замоканными сервисами.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


# ======================================================================
# Mock Services
# ======================================================================

@pytest.fixture
def mock_ml_service():
    """Mock MLService."""
    mock = AsyncMock()
    
    # generate_embedding
    mock.generate_embedding.return_value = {
        "success": True,
        "embedding": np.random.rand(512).astype(np.float32).tobytes(),
        "quality_score": 0.85,
        "face_detected": True,
        "landmarks": [[100, 100], [200, 100], [150, 150]],
        "model_version": "v1.0.0",
    }
    
    # verify_face
    mock.verify_face.return_value = {
        "success": True,
        "similarity": 0.92,
        "confidence": 0.88,
        "liveness": 0.95,
        "quality_score": 0.85,
        "face_detected": True,
    }
    
    # compare_faces
    mock.compare_faces.return_value = {
        "success": True,
        "similarity_score": 0.92,
        "distance": 0.08,
    }
    
    # compare_embeddings
    mock.compare_embeddings.return_value = 0.92
    
    # check_liveness
    mock.check_liveness.return_value = {
        "success": True,
        "liveness_detected": True,
        "confidence": 0.88,
        "face_detected": True,
        "multiple_faces": False,
        "image_quality": 0.85,
        "recommendations": [],
    }
    
    # check_active_liveness
    mock.check_active_liveness.return_value = {
        "success": True,
        "liveness_detected": True,
        "confidence": 0.90,
        "face_detected": True,
        "image_quality": 0.85,
        "anti_spoofing_score": 0.92,
        "recommendations": [],
        "challenge_specific_data": {"blink_count": 2},
    }
    
    # analyze_video_liveness
    mock.analyze_video_liveness.return_value = {
        "success": True,
        "liveness_detected": True,
        "confidence": 0.93,
        "frames_processed": 10,
        "face_detected": True,
        "sequence_data": {"motion_detected": True},
        "anti_spoofing_score": 0.94,
        "recommendations": [],
    }
    
    # advanced_anti_spoofing_check
    mock.advanced_anti_spoofing_check.return_value = {
        "success": True,
        "liveness_detected": True,
        "confidence": 0.95,
        "anti_spoofing_score": 0.96,
        "face_detected": True,
        "analysis_results": {
            "depth_analysis": {"score": 0.92},
            "texture_analysis": {"score": 0.94},
            "certified_analysis": {
                "is_certified_passed": True,
                "certification_level": "high",
            },
        },
        "component_scores": {"depth": 0.92, "texture": 0.94},
        "recommendations": [],
    }
    
    # batch_generate_embeddings
    mock.batch_generate_embeddings.return_value = [
        {
            "success": True,
            "embedding": np.random.rand(512).astype(np.float32).tobytes(),
            "quality_score": 0.85,
            "face_detected": True,
        }
        for _ in range(3)
    ]
    
    return mock


@pytest.fixture
def mock_encryption_service():
    """Mock EncryptionService."""
    mock = AsyncMock()
    
    # encrypt_embedding
    async def encrypt_embedding(data):
        return b"encrypted_" + data[:20]
    
    mock.encrypt_embedding.side_effect = encrypt_embedding
    
    # decrypt_embedding
    async def decrypt_embedding(encrypted):
        return encrypted.replace(b"encrypted_", b"") + b"\x00" * 492
    
    mock.decrypt_embedding.side_effect = decrypt_embedding
    
    return mock


@pytest.fixture
def mock_validation_service():
    """Mock ValidationService."""
    mock = AsyncMock()
    
    # validate_image
    class ValidationResult:
        def __init__(self):
            self.is_valid = True
            self.error_message = None
            self.image_data = b"fake_image_data"
            self.image_format = "PNG"
    
    mock.validate_image.return_value = ValidationResult()
    
    return mock


@pytest.fixture
def mock_storage_service():
    """Mock StorageService."""
    mock = AsyncMock()
    
    # upload_image
    mock.upload_image.return_value = {
        "success": True,
        "file_url": "https://storage.example.com/test-image.jpg",
        "file_id": "file-123",
    }
    
    return mock


@pytest.fixture
def mock_cache_service():
    """Mock CacheService."""
    mock = AsyncMock()
    
    # set
    mock.set.return_value = True
    
    # get
    mock.get.return_value = None
    
    # set_verification_session
    mock.set_verification_session.return_value = True
    
    # get_verification_session
    mock.get_verification_session.return_value = None
    
    return mock


@pytest.fixture
def mock_webhook_service():
    """Mock WebhookService."""
    mock = AsyncMock()
    
    # emit_event
    mock.emit_event.return_value = True
    
    return mock


@pytest.fixture
def mock_anti_spoofing_service():
    """Mock AntiSpoofingService."""
    mock = AsyncMock()
    
    # check_liveness
    mock.check_liveness.return_value = {
        "success": True,
        "is_real": True,
        "liveness_score": 0.95,
        "depth_analysis": {"score": 0.92},
    }
    
    return mock
