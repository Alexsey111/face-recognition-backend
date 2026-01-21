"""Tests for auth service.
Hybrid sync/async approach:
- Sync methods (no await): create_access_token, create_refresh_token,
  generate_secure_token, validate_user_permissions, get_token_info, needs_password_rehash
- Async methods (with await): verify_token, hash_password, verify_password,
  refresh_access_token, revoke_token, is_token_revoked, check_rate_limit, get_user_info_from_token
"""

import pytest
import jwt
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.services.auth_service import AuthService, AuthenticationError
from app.config import settings


class TestAuthService:
    """Tests for AuthService."""

    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance for tests."""
        return AuthService()

    @pytest.fixture
    def test_user_id(self):
        """Test user ID."""
        return "test-user-123"

    @pytest.fixture
    def test_role(self):
        """Test user role."""
        return "user"

    def test_initialization(self, auth_service):
        """Test service initialization."""
        assert auth_service.jwt_secret_key is not None
        assert auth_service.jwt_algorithm is not None
        assert auth_service.access_token_expire_minutes > 0
        assert auth_service.refresh_token_expire_days > 0

    # =============================================================================
    # SYNC METHOD TESTS (no await)
    # =============================================================================

    def test_create_access_token_success(self, auth_service, test_user_id, test_role):
        """Test sync: successful access token creation (no await)."""
        token = auth_service.create_access_token(test_user_id, test_role)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode token to verify contents
        payload = jwt.decode(
            token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm]
        )

        assert payload["user_id"] == test_user_id
        assert payload["role"] == test_role
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_create_access_token_with_permissions(self, auth_service, test_user_id):
        """Test sync: access token creation with permissions (no await)."""
        permissions = ["read", "write", "delete"]

        token = auth_service.create_access_token(test_user_id, permissions=permissions)

        payload = jwt.decode(
            token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm]
        )

        assert payload["permissions"] == permissions

    def test_create_access_token_with_additional_claims(
        self, auth_service, test_user_id
    ):
        """Test sync: access token creation with additional claims (no await)."""
        additional_claims = {"tenant_id": "tenant-123", "scope": "admin"}

        token = auth_service.create_access_token(
            test_user_id, additional_claims=additional_claims
        )

        payload = jwt.decode(
            token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm]
        )

        assert payload["tenant_id"] == "tenant-123"
        assert payload["scope"] == "admin"

    def test_create_refresh_token_success(self, auth_service, test_user_id):
        """Test sync: successful refresh token creation (no await)."""
        token = auth_service.create_refresh_token(test_user_id)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        payload = jwt.decode(
            token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm]
        )

        assert payload["user_id"] == test_user_id
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_generate_secure_token(self, auth_service):
        """Test sync: secure token generation (no await)."""
        token = auth_service.generate_secure_token(32)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) == 64  # 32 bytes in hex = 64 chars

    def test_validate_user_permissions_admin_role(self, auth_service):
        """Test sync: permission validation for admin role (no await)."""
        result = auth_service.validate_user_permissions(
            "admin", ["read", "write", "delete", "admin"]
        )
        assert result is True

    def test_validate_user_permissions_user_role(self, auth_service):
        """Test sync: permission validation for user role (no await)."""
        result = auth_service.validate_user_permissions("user", ["read_own_data"])
        assert result is True

    def test_validate_user_permissions_insufficient(self, auth_service):
        """Test sync: insufficient permissions (no await)."""
        from app.utils.exceptions import ForbiddenError

        with pytest.raises(ForbiddenError):
            auth_service.validate_user_permissions("user", ["admin_operations"])

    def test_validate_user_permissions_with_user_permissions(self, auth_service):
        """Test sync: permission validation with explicit user permissions (no await)."""
        user_permissions = ["read", "write"]

        result = auth_service.validate_user_permissions(
            "user", ["read"], user_permissions
        )
        assert result is True

        from app.utils.exceptions import ForbiddenError

        with pytest.raises(ForbiddenError):
            auth_service.validate_user_permissions(
                "user", ["read", "write", "delete"], user_permissions
            )

    def test_needs_password_rehash_bcrypt(self, auth_service):
        """Test sync: check if bcrypt password needs rehash (no await)."""
        password = "test_password_123"
        hashed_password = auth_service._pwd_context.hash(password)

        needs_rehash = auth_service.needs_password_rehash(hashed_password)
        assert needs_rehash is False

    def test_needs_password_rehash_legacy(self, auth_service):
        """Test sync: check if legacy PBKDF2 needs rehash (no await)."""
        import hashlib, secrets

        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", b"test_password_123", salt, 100000
        )
        combined = salt + password_hash
        legacy_hash = combined.hex()

        needs_rehash = auth_service.needs_password_rehash(legacy_hash)
        assert needs_rehash is True

    def test_get_token_info_success(self, auth_service, test_user_id):
        """Test sync: get token info without verification (no await)."""
        token = jwt.encode(
            {
                "user_id": test_user_id,
                "type": "access",
                "role": "user",
                "iat": datetime.now(timezone.utc),
                "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            },
            auth_service.jwt_secret_key,
            algorithm=auth_service.jwt_algorithm,
        )

        token_info = auth_service.get_token_info(token)

        assert token_info["type"] == "access"
        assert token_info["user_id"] == test_user_id
        assert token_info["role"] == "user"
        assert "issued_at" in token_info
        assert "expires_at" in token_info
        assert "is_expired" in token_info

    def test_get_token_info_invalid(self, auth_service):
        """Test sync: get info for invalid token (no await)."""
        token_info = auth_service.get_token_info("invalid_token")

        assert "error" in token_info

    @pytest.mark.asyncio
    async def test_create_user_session_success(self, auth_service, test_user_id):
        """Test async: successful user session creation."""
        user_agent = "test-browser"
        ip_address = "127.0.0.1"

        session = await auth_service.create_user_session(
            test_user_id, user_agent, ip_address
        )

        assert "access_token" in session
        assert "refresh_token" in session
        assert "token_type" in session
        assert "expires_in" in session
        assert session["token_type"] == "bearer"
        assert session["expires_in"] > 0

    # =============================================================================
    # ASYNC METHOD TESTS (with await)
    # =============================================================================

    @pytest.mark.asyncio
    async def test_verify_valid_access_token(self, auth_service, test_user_id):
        """Test async: verify valid access token (with await)."""
        token = auth_service.create_access_token(test_user_id)  # sync
        payload = await auth_service.verify_token(token, "access")  # async

        assert payload["user_id"] == test_user_id
        assert payload["type"] == "access"

    @pytest.mark.asyncio
    async def test_verify_valid_refresh_token(self, auth_service, test_user_id):
        """Test async: verify valid refresh token (with await)."""
        token = auth_service.create_refresh_token(test_user_id)  # sync
        payload = await auth_service.verify_token(token, "refresh")  # async

        assert payload["user_id"] == test_user_id
        assert payload["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_verify_expired_token(self, auth_service, test_user_id):
        """Test async: verify expired token (with await)."""
        expire = datetime.now(timezone.utc) - timedelta(seconds=1)

        payload = {
            "user_id": test_user_id,
            "type": "access",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": "test-jti",
        }

        expired_token = jwt.encode(
            payload, auth_service.jwt_secret_key, algorithm=auth_service.jwt_algorithm
        )

        from app.utils.exceptions import UnauthorizedError

        with pytest.raises(UnauthorizedError, match="Token has expired"):
            await auth_service.verify_token(expired_token, "access")

    @pytest.mark.asyncio
    async def test_verify_invalid_token_type(self, auth_service, test_user_id):
        """Test async: verify token with wrong type (with await)."""
        token = auth_service.create_access_token(test_user_id)  # sync

        from app.utils.exceptions import UnauthorizedError

        with pytest.raises(UnauthorizedError, match="Invalid token type"):
            await auth_service.verify_token(token, "refresh")

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self, auth_service, test_user_id):
        """Test async: refresh access token (with await)."""
        refresh_token = auth_service.create_refresh_token(test_user_id)  # sync

        token_response = await auth_service.refresh_access_token(refresh_token)  # async

        assert token_response is not None
        assert isinstance(token_response, dict)
        assert "access_token" in token_response

        new_access_token = token_response["access_token"]
        payload = await auth_service.verify_token(new_access_token, "access")  # async
        assert payload["user_id"] == test_user_id

    @pytest.mark.asyncio
    async def test_refresh_access_token_invalid(self, auth_service):
        """Test async: refresh with invalid token (with await)."""
        from app.utils.exceptions import UnauthorizedError

        with pytest.raises(UnauthorizedError):
            await auth_service.refresh_access_token("invalid_token")

    @pytest.mark.asyncio
    async def test_hash_password_success(self, auth_service):
        """Test async: password hashing (with await)."""
        password = "test_password_123"

        hashed_password = await auth_service.hash_password(password)

        assert hashed_password is not None
        assert isinstance(hashed_password, str)
        assert len(hashed_password) > 0
        assert hashed_password != password
        assert hashed_password.startswith("$pbkdf2-sha256$")

    @pytest.mark.asyncio
    async def test_verify_password_correct(self, auth_service):
        """Test async: verify correct password (with await)."""
        password = "test_password_123"

        hashed_password = await auth_service.hash_password(password)
        is_valid = await auth_service.verify_password(password, hashed_password)

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_password_incorrect(self, auth_service):
        """Test async: verify incorrect password (with await)."""
        correct_password = "test_password_123"
        wrong_password = "wrong_password_456"

        hashed_password = await auth_service.hash_password(correct_password)
        is_valid = await auth_service.verify_password(wrong_password, hashed_password)

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_hash_password_uniqueness(self, auth_service):
        """Test async: password hash uniqueness (with await)."""
        password = "same_password"

        hash1 = await auth_service.hash_password(password)
        hash2 = await auth_service.hash_password(password)

        assert hash1 != hash2  # Different salts
        assert await auth_service.verify_password(password, hash1)
        assert await auth_service.verify_password(password, hash2)

    @pytest.mark.asyncio
    async def test_legacy_pbkdf2_compatibility(self, auth_service):
        """Test async: legacy PBKDF2 compatibility (with await)."""
        import hashlib, secrets

        password = "test_password_123"
        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 100000
        )
        legacy_hash = (salt + password_hash).hex()

        is_valid = await auth_service.verify_password(password, legacy_hash)
        assert is_valid is True

        is_invalid = await auth_service.verify_password("wrong_password", legacy_hash)
        assert is_invalid is False

    @pytest.mark.asyncio
    async def test_password_migration_workflow(self, auth_service):
        """Test async: password migration workflow (with await)."""
        import hashlib, secrets

        password = "migration_test_password"
        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 100000
        )
        old_hash = (salt + password_hash).hex()

        assert await auth_service.verify_password(password, old_hash) is True
        assert auth_service.needs_password_rehash(old_hash) is True

        new_hash = await auth_service.hash_password(password)
        assert auth_service.needs_password_rehash(new_hash) is False
        assert await auth_service.verify_password(password, new_hash) is True
        assert await auth_service.verify_password(password, old_hash) is True

    @pytest.mark.asyncio
    async def test_get_user_info_from_token(self, auth_service, test_user_id):
        """Test async: get user info from token (with await)."""
        token = auth_service.create_access_token(  # sync
            test_user_id, role="premium", permissions=["read", "write"]
        )

        user_info = await auth_service.get_user_info_from_token(token)  # async

        assert user_info["user_id"] == test_user_id
        assert user_info["role"] == "premium"
        assert user_info["permissions"] == ["read", "write"]
        assert user_info["token_type"] == "access"
        assert "issued_at" in user_info
        assert "expires_at" in user_info
        assert "jti" in user_info

    @pytest.mark.asyncio
    async def test_revoke_token_success(self, auth_service, test_user_id):
        """Test async: revoke token (with await)."""
        token = auth_service.create_access_token(test_user_id)  # sync

        result = await auth_service.revoke_token(token)  # async

        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_token_invalid(self, auth_service):
        """Test async: revoke invalid token (with await)."""
        result = await auth_service.revoke_token("invalid_token")
        assert result is False


class TestAuthServiceSecurity:
    """Security tests for authentication."""

    @pytest.fixture
    def auth_service(self):
        return AuthService()

    def test_token_uniqueness(self, auth_service):
        """Test sync: token uniqueness (no await)."""
        user_id = "test_user"

        token1 = auth_service.create_access_token(user_id)
        token2 = auth_service.create_access_token(user_id)

        assert token1 != token2

    @pytest.mark.asyncio
    async def test_token_tampering_detection(self, auth_service):
        """Test async: token tampering detection (with await)."""
        user_id = "test_user"

        token = auth_service.create_access_token(user_id)

        payload = jwt.decode(token, options={"verify_signature": False})
        payload["user_id"] = "different_user"

        tampered_token = jwt.encode(
            payload, "different_secret", algorithm=auth_service.jwt_algorithm
        )

        from app.utils.exceptions import UnauthorizedError

        with pytest.raises(UnauthorizedError):
            await auth_service.verify_token(tampered_token, "access")

    @pytest.mark.asyncio
    async def test_password_security_features(self, auth_service):
        """Test async: password security features (with await)."""
        password = "test_password_123"

        hashed_password = await auth_service.hash_password(password)

        assert hashed_password.startswith("$pbkdf2-sha256$")
        assert len(hashed_password) > 50
        assert hashed_password != password

        hash2 = await auth_service.hash_password(password)
        assert hashed_password != hash2

        assert await auth_service.verify_password(password, hashed_password)
        assert await auth_service.verify_password(password, hash2)
        assert (
            await auth_service.verify_password("wrong_password", hashed_password)
            is False
        )
        assert await auth_service.verify_password("wrong_password", hash2) is False

    def test_jwt_algorithm_security(self, auth_service):
        """Test sync: JWT algorithm security (no await)."""
        assert auth_service.jwt_algorithm in ["HS256", "HS384", "HS512"]
        assert auth_service.jwt_algorithm != "none"

    @pytest.mark.asyncio
    async def test_token_expiration(self, auth_service):
        """Test async: token expiration (with await)."""
        user_id = "test_user"

        token = auth_service.create_access_token(user_id)

        payload = jwt.decode(token, options={"verify_signature": False})
        exp = payload.get("exp")
        assert exp is not None

        exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        assert exp_datetime > now

    @pytest.mark.asyncio
    async def test_privilege_escalation_prevention(self, auth_service):
        """Test async: privilege escalation prevention (with await)."""
        user_id = "test_user"

        token = auth_service.create_access_token(
            user_id, role="user", permissions=["read_own_data"]
        )

        payload = jwt.decode(
            token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm]
        )

        assert payload["role"] == "user"
        assert "admin" not in payload.get("permissions", [])
