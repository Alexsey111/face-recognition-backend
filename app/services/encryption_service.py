import base64
import os
import json
import secrets
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any
import asyncio

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import EncryptionError

logger = get_logger(__name__)


# =============================================================================
# Payload format (authenticated by AES-256-GCM)
# =============================================================================
# {
#   "v": "2.0",
#   "alg": "aes-256-gcm",
#   "ts": "...",
#   "meta": {...},
#   "data": "<base64>"
# }
# =============================================================================


class EncryptionService:
    """
    Cryptographically safe encryption service.
    Uses AES-256-GCM (Galois/Counter Mode) for authenticated encryption.
    Provides 256-bit encryption strength with built-in authentication.
    """

    SUPPORTED_VERSION = "2.0"
    ALGORITHM = "aes-256-gcm"
    NONCE_LENGTH = 12  # GCM recommended nonce size
    KEY_LENGTH = 32  # AES-256 requires 32-byte key
    TAG_LENGTH = 16  # GCM authentication tag length

    def __init__(self) -> None:
        self._key = self._derive_key(settings.ENCRYPTION_KEY)
        self._aesgcm = AESGCM(self._key)

    # =========================================================================
    # Key handling
    # =========================================================================

    def _derive_key(self, key: str) -> bytes:
        """
        Derive a 256-bit key from the provided key string using PBKDF2.
        This allows using a passphrase-like key while getting a fixed-length
        AES-256 key.
        """
        try:
            # Check for empty key
            if not key or not key.strip():
                raise EncryptionError("Encryption key cannot be empty")

            # If the key is already 32 bytes, use it directly
            key_bytes = key.encode("utf-8")
            if len(key_bytes) == self.KEY_LENGTH:
                return key_bytes

            # If the key is a Fernet-style base64 key (32 bytes decoded), use it
            try:
                decoded = base64.urlsafe_b64decode(key_bytes)
                if len(decoded) == self.KEY_LENGTH:
                    return decoded
            except Exception:
                pass

            # Otherwise, derive a key from the passphrase using PBKDF2
            salt = b"face-recognition-salt"  # Fixed salt for consistent derivation
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.KEY_LENGTH,
                salt=salt,
                iterations=480000,  # OWASP recommended minimum
                backend=default_backend(),
            )
            return kdf.derive(key_bytes)

        except Exception as exc:
            raise EncryptionError(f"Invalid encryption key: {exc}") from exc

    @staticmethod
    def generate_key() -> str:
        """
        Generate a valid AES-256 key.
        Returns base64-encoded 32-byte key.
        """
        key = secrets.token_bytes(EncryptionService.KEY_LENGTH)
        return base64.urlsafe_b64encode(key).decode("utf-8")

    @staticmethod
    def verify_key(key: str) -> bool:
        """
        Verify that a key is valid for AES-256 encryption.
        """
        try:
            key_bytes = key.encode("utf-8")
            # Try direct 32-byte key
            if len(key_bytes) == 32:
                return True
            # Try base64 decoding
            decoded = base64.urlsafe_b64decode(key_bytes)
            return len(decoded) == 32
        except Exception:
            return False

    # =========================================================================
    # Payload packing (no encryption here)
    # =========================================================================

    def _pack_payload(
        self, data: bytes, metadata: Optional[Dict[str, Any]], version: str
    ) -> bytes:
        """
        Pack data into JSON payload (WITHOUT nonce).
        """
        payload = {
            "v": version,
            "alg": self.ALGORITHM,
            "ts": datetime.now(timezone.utc).isoformat(),
            "meta": metadata or {},
            "data": base64.b64encode(data).decode("ascii"),
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _unpack_payload(self, decrypted_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Unpack decrypted JSON payload.
        """
        try:
            payload = json.loads(decrypted_bytes.decode("utf-8"))
        except Exception as exc:
            raise EncryptionError("Invalid decrypted payload format") from exc

        if payload.get("alg") != self.ALGORITHM:
            raise EncryptionError("Unsupported encryption algorithm")

        if payload.get("v") != self.SUPPORTED_VERSION:
            raise EncryptionError("Unsupported encryption version")

        try:
            data = base64.b64decode(payload["data"])
        except Exception as exc:
            raise EncryptionError("Corrupted payload data") from exc

        return data, payload

    # =========================================================================
    # Public API
    # =========================================================================

    async def encrypt(
        self,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = SUPPORTED_VERSION,
    ) -> bytes:
        """
        Encrypt data using AES-256-GCM.

        Returns: nonce (12 bytes) + encrypted_payload
        """
        if data is None:
            raise EncryptionError("No data to encrypt")

        try:
            # 1. Pack data into JSON payload
            payload_bytes = self._pack_payload(data, metadata or {}, version)

            # 2. Generate nonce
            nonce = secrets.token_bytes(self.NONCE_LENGTH)

            # 3. Encrypt payload
            encrypted = await asyncio.to_thread(
                self._aesgcm.encrypt, nonce, payload_bytes, None
            )

            # 4. Return: nonce + encrypted_payload
            return nonce + encrypted
        except Exception as exc:
            raise EncryptionError(f"Encryption failed: {exc}") from exc

    async def decrypt(self, token: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Decrypt data using AES-256-GCM.

        Token format: nonce (12 bytes) + encrypted_data
        """
        if not token:
            raise EncryptionError("Empty encrypted token")

        if len(token) < self.NONCE_LENGTH + self.TAG_LENGTH:
            raise EncryptionError("Invalid token format: too short")

        try:
            # 1. Extract nonce and encrypted data
            nonce = token[: self.NONCE_LENGTH]
            encrypted_data = token[self.NONCE_LENGTH :]

            # 2. Decrypt payload
            decrypted_bytes = await asyncio.to_thread(
                self._aesgcm.decrypt, nonce, encrypted_data, None
            )

            # 3. Unpack payload
            data, payload = self._unpack_payload(decrypted_bytes)

            return data, payload
        except Exception as exc:
            raise EncryptionError("Invalid or tampered encrypted data") from exc

    async def encrypt_embedding(self, embedding: bytes) -> bytes:
        return await self.encrypt(
            embedding,
            metadata={"type": "face_embedding"},
        )

    async def decrypt_embedding(self, encrypted_embedding: bytes) -> bytes:
        data, _ = await self.decrypt(encrypted_embedding)
        return data

    def encrypt_data_sync(self, data: bytes) -> bytes:
        """
        Синхронное шифрование данных (без async).
        
        Used by BiometricEncryption for synchronous operations.
        """
        import base64
        import json
        import secrets
        from datetime import datetime, timezone

        # Pack data into JSON payload
        payload = {
            "v": self.SUPPORTED_VERSION,
            "alg": self.ALGORITHM,
            "ts": datetime.now(timezone.utc).isoformat(),
            "meta": {"type": "face_embedding"},
            "data": base64.b64encode(data).decode("ascii"),
        }
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        # Generate nonce
        nonce = secrets.token_bytes(self.NONCE_LENGTH)

        # Encrypt payload
        encrypted = self._aesgcm.encrypt(nonce, payload_bytes, None)

        return nonce + encrypted

    def decrypt_data_sync(self, token: bytes) -> Tuple[bytes, dict]:
        """
        Синхронная расшифровка данных (без async).
        
        Used by BiometricEncryption for synchronous operations.
        """
        import base64
        import json

        if len(token) < self.NONCE_LENGTH + self.TAG_LENGTH:
            raise EncryptionError("Invalid token format: too short")

        # Extract nonce and encrypted data
        nonce = token[: self.NONCE_LENGTH]
        encrypted_data = token[self.NONCE_LENGTH:]

        # Decrypt payload
        decrypted_bytes = self._aesgcm.decrypt(nonce, encrypted_data, None)

        # Unpack payload
        payload = json.loads(decrypted_bytes.decode("utf-8"))
        data = base64.b64decode(payload["data"])

        return data, payload

    async def encrypt_data(
        self, data: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Compatibility wrapper used by tests and other services."""
        return await self.encrypt(data, metadata)

    async def decrypt_data(self, token: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Compatibility wrapper returning (data, payload)."""
        return await self.decrypt(token)

    async def decrypt_file(
        self,
        encrypted_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        # Чтение зашифрованного файла — в отдельный поток
        def _read_encrypted():
            with open(encrypted_path, "rb") as f:
                return f.read()

        token = await asyncio.to_thread(_read_encrypted)

        data, meta = await self.decrypt(token)

        if not output_path:
            # Извлекаем имя файла из metadata, если есть
            output_path = meta.get("meta", {}).get("filename", encrypted_path + ".dec")

        # Запись расшифрованного файла — в отдельный поток
        def _write_decrypted():
            with open(output_path, "wb") as f:
                f.write(data)

        await asyncio.to_thread(_write_decrypted)

        logger.info("File decrypted: %s -> %s", encrypted_path, output_path)
        return output_path

    def get_encryption_info(self) -> Dict[str, Any]:
        return {
            "algorithm": self.ALGORITHM,
            "version": self.SUPPORTED_VERSION,
            "key_valid": True,
            "key_length_bits": self.KEY_LENGTH * 8,
            "authenticated_encryption": True,
            "mode": "GCM",
            "supports_embeddings": True,
            "supports_files": True,
            "supports_versioning": True,
            "key_rotation_ready": True,
        }
