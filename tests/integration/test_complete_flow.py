"""End-to-end tests for complete face verification flow."""

import pytest
import uuid
from io import BytesIO


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for upload tests."""
    import base64

    # Minimal 1x1 PNG image (base64 encoded)
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    return base64.b64decode(png_base64)


@pytest.fixture
def unique_user_data():
    """Generate unique user data for E2E tests."""
    unique_id = uuid.uuid4().hex[:8]
    return {
        "email": f"e2e_{unique_id}@example.com",
        "password": "SecurePass123!",
        "full_name": f"Test User {unique_id}",
    }


@pytest.mark.e2e
class TestCompleteVerificationFlow:
    """End-to-end test for complete face verification flow."""

    @pytest.mark.asyncio
    async def test_complete_verification_flow(
        self,
        async_client,
        auth_headers,
        sample_image_bytes,
    ):
        """
        Complete flow:
        1. Register user (or use existing)
        2. Upload reference image
        3. Create reference
        4. Upload verification image
        5. Verify face
        6. Check results
        """
        unique_suffix = uuid.uuid4().hex[:8]
        headers = auth_headers

        # Step 1: Create upload session for reference
        response = await async_client.post(
            "/api/v1/upload",
            headers=headers,
        )
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        # Step 2: Upload reference file
        files = {"file": (f"reference_{unique_suffix}.png", sample_image_bytes, "image/png")}
        response = await async_client.post(
            f"/api/v1/upload/{session_id}/file",
            headers=headers,
            files=files,
        )
        assert response.status_code == 200
        file_data = response.json()
        file_key = file_data["file_key"]

        # Step 3: Create reference
        response = await async_client.put(
            "/api/v1/reference",
            headers=headers,
            json={
                "file_key": file_key,
                "label": f"Reference {unique_suffix}",
            },
        )
        assert response.status_code == 200
        reference_data = response.json()
        assert "reference_id" in reference_data

        # Step 4: Create verification upload session
        response = await async_client.post(
            "/api/v1/upload",
            headers=headers,
        )
        assert response.status_code == 200
        verify_session_id = response.json()["session_id"]

        # Step 5: Upload verification file
        files = {"file": (f"verify_{unique_suffix}.png", sample_image_bytes, "image/png")}
        response = await async_client.post(
            f"/api/v1/upload/{verify_session_id}/file",
            headers=headers,
            files=files,
        )
        assert response.status_code == 200
        verify_file_key = response.json()["file_key"]

        # Step 6: Perform verification
        response = await async_client.post(
            "/api/v1/verify",
            headers=headers,
            json={"file_key": verify_file_key},
        )
        assert response.status_code == 200
        verify_result = response.json()

        # Verify response structure
        assert "is_match" in verify_result
        assert "similarity_score" in verify_result
        assert "confidence" in verify_result
        assert "verification_id" in verify_result

        # Step 7: Check liveness (optional)
        response = await async_client.post(
            "/api/v1/liveness",
            headers=headers,
            json={"file_key": verify_file_key},
        )
        assert response.status_code == 200
        liveness_result = response.json()
        assert "is_live" in liveness_result
        assert "confidence" in liveness_result
        assert "liveness_score" in liveness_result

    @pytest.mark.asyncio
    async def test_upload_reference_verify_flow(
        self,
        async_client,
        auth_headers,
        sample_image_bytes,
    ):
        """Test simplified upload → reference → verify flow."""
        unique_suffix = uuid.uuid4().hex[:8]
        headers = auth_headers

        # Upload
        response = await async_client.post(
            "/api/v1/upload",
            headers=headers,
        )
        session_id = response.json()["session_id"]

        # Upload file
        files = {"file": (f"test_{unique_suffix}.png", sample_image_bytes, "image/png")}
        response = await async_client.post(
            f"/api/v1/upload/{session_id}/file",
            headers=headers,
            files=files,
        )
        file_key = response.json()["file_key"]

        # Create reference
        response = await async_client.put(
            "/api/v1/reference",
            headers=headers,
            json={"file_key": file_key},
        )
        assert response.status_code == 200

        # Verify
        response = await async_client.post(
            "/api/v1/verify",
            headers=headers,
            json={"file_key": file_key},
        )
        assert response.status_code == 200
        assert response.json()["is_match"] is True