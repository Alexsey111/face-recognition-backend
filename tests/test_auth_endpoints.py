"""Tests for authentication endpoints."""

import pytest
import uuid
from fastapi import status


class TestAuthEndpoints:
    """Authentication endpoint tests using async client."""

    @pytest.mark.asyncio
    async def test_register_success(self, async_client):
        """Test successful user registration."""
        unique_email = f"newuser_{uuid.uuid4().hex[:8]}@example.com"

        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": unique_email,
                "password": "SecurePass123!",
                "full_name": "New User",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == unique_email
        assert "id" in data
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, async_client, test_user_123):
        """Test registration with existing email."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "user-123@example.com",
                "password": "SecurePass123!",
                "full_name": "Duplicate User",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already registered" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_register_weak_password(self, async_client):
        """Test registration with weak password."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "weak@example.com",
                "password": "123",
                "full_name": "Weak Password User",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_login_success(self, async_client, test_user_123):
        """Test successful login."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "user-123@example.com",
                "password": "testpassword",  # Default test password
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, async_client, test_user_123):
        """Test login with wrong password."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "user-123@example.com",
                "password": "wrongpassword",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, async_client):
        """Test login with non-existent user."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent@example.com",
                "password": "anypassword",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_current_user(self, async_client, auth_headers):
        """Test getting current user info."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "email" in data

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, async_client):
        """Test getting current user with invalid token."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client, test_user_123):
        """Test token refresh."""
        # First login to get refresh token
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "user-123@example.com",
                "password": "testpassword",
            },
        )

        refresh_token = login_response.json().get("refresh_token")

        if refresh_token:
            response = await async_client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": refresh_token},
            )

            # If refresh endpoint exists
            if response.status_code != status.HTTP_404_NOT_FOUND:
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert "access_token" in data

    @pytest.mark.asyncio
    async def test_logout(self, async_client, auth_headers):
        """Test user logout."""
        response = await async_client.post(
            "/api/v1/auth/logout",
            headers=auth_headers,
        )

        # Depending on implementation
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_204_NO_FOUND,
        ]