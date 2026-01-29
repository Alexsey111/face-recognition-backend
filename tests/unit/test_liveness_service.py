"""
Unit-тесты для LivenessService.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.liveness_service import LivenessService
from app.utils.exceptions import ProcessingError, ValidationError

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def liveness_service(
    db_session,
    mock_ml_service,
    mock_anti_spoofing_service,
    mock_validation_service,
    mock_cache_service,
    mock_webhook_service,
):
    """Создание LivenessService с замоканными зависимостями."""
    service = LivenessService(db_session)

    service.ml_service = mock_ml_service
    service.anti_spoofing_service = mock_anti_spoofing_service
    service.validation_service = mock_validation_service
    service.cache_service = mock_cache_service
    service.webhook_service = mock_webhook_service

    return service


# ======================================================================
# Тесты check_liveness
# ======================================================================


@pytest.mark.asyncio
async def test_check_liveness_passive_success(
    liveness_service,
    mock_image_data,
):
    """Тест успешной passive liveness проверки."""

    result = await liveness_service.check_liveness(
        image_data=mock_image_data.encode(),
        challenge_type="passive",
    )

    assert result["success"] is True
    assert result["liveness_detected"] is True
    assert result["confidence"] > 0.0
    assert result["challenge_type"] == "passive"


@pytest.mark.asyncio
async def test_check_liveness_active_success(
    liveness_service,
    mock_image_data,
):
    """Тест успешной active liveness проверки."""

    result = await liveness_service.check_liveness(
        image_data=mock_image_data.encode(),
        challenge_type="blink",
        challenge_data={"expected_blinks": 2},
    )

    assert result["success"] is True
    assert result["liveness_detected"] is True
    assert result["challenge_type"] == "blink"


@pytest.mark.asyncio
async def test_check_liveness_unsupported_challenge(
    liveness_service,
    mock_image_data,
):
    """Тест с неподдерживаемым типом челленджа."""

    with pytest.raises(ValidationError, match="Unsupported challenge_type"):
        await liveness_service.check_liveness(
            image_data=mock_image_data.encode(),
            challenge_type="unsupported_type",
        )


@pytest.mark.asyncio
async def test_check_liveness_missing_challenge_data(
    liveness_service,
    mock_image_data,
):
    """Тест active liveness без challenge_data."""

    with pytest.raises(ValidationError, match="challenge_data is required"):
        await liveness_service.check_liveness(
            image_data=mock_image_data.encode(),
            challenge_type="blink",
            challenge_data=None,
        )


# ======================================================================
# Тесты check_passive_liveness
# ======================================================================


@pytest.mark.asyncio
async def test_check_passive_liveness_with_certified(
    liveness_service,
    mock_image_data,
):
    """Тест passive liveness с certified anti-spoofing."""

    with patch("app.services.liveness_service.settings") as mock_settings:
        mock_settings.USE_CERTIFIED_LIVENESS = True
        mock_settings.LIVENESS_CONFIDENCE_THRESHOLD = 0.7

        result = await liveness_service.check_passive_liveness(
            image_data=mock_image_data.encode(),
        )

        assert "liveness_detected" in result
        assert "confidence" in result
        assert result["liveness_type"] == "passive_certified"


@pytest.mark.asyncio
async def test_check_passive_liveness_without_certified(
    liveness_service,
    mock_image_data,
):
    """Тест passive liveness без certified anti-spoofing."""

    with patch("app.services.liveness_service.settings") as mock_settings:
        mock_settings.USE_CERTIFIED_LIVENESS = False

        result = await liveness_service.check_passive_liveness(
            image_data=mock_image_data.encode(),
        )

        assert result["liveness_type"] == "passive_basic"


# ======================================================================
# Тесты check_active_liveness
# ======================================================================


@pytest.mark.asyncio
async def test_check_active_liveness_blink(
    liveness_service,
    mock_image_data,
):
    """Тест active liveness с blink челленджем."""

    result = await liveness_service.check_active_liveness(
        image_data=mock_image_data.encode(),
        challenge_type="blink",
        challenge_data={"expected_blinks": 2},
    )

    assert result["liveness_type"] == "active_blink"
    assert "challenge_specific_data" in result


@pytest.mark.asyncio
async def test_check_active_liveness_smile(
    liveness_service,
    mock_image_data,
):
    """Тест active liveness с smile челленджем."""

    result = await liveness_service.check_active_liveness(
        image_data=mock_image_data.encode(),
        challenge_type="smile",
        challenge_data={"timeout": 3},
    )

    assert result["liveness_type"] == "active_smile"


# ======================================================================
# Тесты analyze_video_liveness
# ======================================================================


@pytest.mark.asyncio
async def test_analyze_video_liveness_success(
    liveness_service,
):
    """Тест video liveness анализа."""

    # Mock video frames
    video_frames = [b"frame_1", b"frame_2", b"frame_3"]

    result = await liveness_service.analyze_video_liveness(
        video_frames=video_frames,
        challenge_type="video_blink",
    )

    assert result["liveness_detected"] is True
    assert result["frames_processed"] > 0
    assert "sequence_data" in result


@pytest.mark.asyncio
async def test_analyze_video_liveness_empty_frames(
    liveness_service,
):
    """Тест video liveness с пустым списком кадров."""

    with pytest.raises(ValidationError, match="No video frames"):
        await liveness_service.analyze_video_liveness(
            video_frames=[],
        )


# ======================================================================
# Тесты perform_anti_spoofing_check
# ======================================================================


@pytest.mark.asyncio
async def test_perform_anti_spoofing_check_certified(
    liveness_service,
    mock_image_data,
):
    """Тест certified anti-spoofing проверки."""

    result = await liveness_service.perform_anti_spoofing_check(
        image_data=mock_image_data.encode(),
        analysis_type="certified",
    )

    assert result["liveness_detected"] is True
    assert result["anti_spoofing_score"] > 0.0
    assert result["analysis_type"] == "certified"


@pytest.mark.asyncio
async def test_perform_anti_spoofing_check_comprehensive(
    liveness_service,
    mock_image_data,
):
    """Тест comprehensive anti-spoofing проверки."""

    result = await liveness_service.perform_anti_spoofing_check(
        image_data=mock_image_data.encode(),
        analysis_type="comprehensive",
        include_reasoning=True,
    )

    assert "depth_analysis" in result
    assert "texture_analysis" in result
    assert "certified_analysis" in result


# ======================================================================
# Тесты generate_challenge
# ======================================================================


@pytest.mark.asyncio
async def test_generate_challenge_blink(
    liveness_service,
):
    """Тест генерации blink челленджа."""

    challenge = await liveness_service.generate_challenge("blink")

    assert challenge["challenge_type"] == "blink"
    assert "challenge_id" in challenge
    assert "action" in challenge
    assert "expected_blinks" in challenge


@pytest.mark.asyncio
async def test_generate_challenge_smile(
    liveness_service,
):
    """Тест генерации smile челленджа."""

    challenge = await liveness_service.generate_challenge("smile")

    assert challenge["challenge_type"] == "smile"
    assert "timeout_seconds" in challenge


@pytest.mark.asyncio
async def test_generate_challenge_random(
    liveness_service,
):
    """Тест генерации случайного челленджа."""

    challenge = await liveness_service.generate_challenge("random")

    # Должен быть один из поддерживаемых типов
    assert challenge["challenge_type"] in ["blink", "smile", "turn_head"]


# ======================================================================
# Тесты get_supported_challenges
# ======================================================================


def test_get_supported_challenges():
    """Тест получения списка поддерживаемых челленджей."""

    challenges = LivenessService.get_supported_challenges()

    assert isinstance(challenges, dict)
    assert "passive" in challenges
    assert "blink" in challenges
    assert "smile" in challenges
    assert "turn_head" in challenges
