"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import base64
import hashlib
import logging
import os
import secrets
import threading
from pathlib import Path
from typing import Dict, Optional

from cryptography.fernet import Fernet

# Module-level lock for singleton initialization
_singleton_lock = threading.Lock()


class KeyIntegrityError(RuntimeError):
    """Raised when existing key material cannot be decrypted or verified.

    This is deliberately fatal. Silently regenerating a key that failed to
    decrypt would invalidate every artefact signed or encrypted with the
    previous key (backup HMACs, issued JWTs) while reporting success, so
    integrity checks would then "pass" against unverifiable state.

    Recovery is an explicit operator action:
        - restore the correct ``APGI_MASTER_KEY``, or
        - run ``python3 -m utils.key_rotation_manager --rotate`` to mint new
          keys, accepting that artefacts under the old keys become invalid.
    """


class SecureKeyManager:
    """Thread-safe secure key manager with point-of-use retrieval."""

    def __init__(self, keys_dir: str = ".keys"):
        """
        Initialize secure key manager.

        Args:
            keys_dir: Directory to store encrypted key files
        """
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(exist_ok=True)
        self.keys_dir.chmod(0o700)

        # Thread lock for thread-safe operations (RLock for reentrancy)
        self._lock = threading.RLock()

        # Initialize logger
        self.logger = logging.getLogger("secure_key_manager")
        self.logger.setLevel(logging.INFO)

        # Load or create keys on first access
        self._pickle_key_file = self.keys_dir / "pickle_secret_key.enc"
        self._backup_key_file = self.keys_dir / "backup_hmac_key.enc"
        self._jwt_key_file = self.keys_dir / "jwt_secret_key.enc"

        # Cache for frequently accessed keys (cleared on rotation)
        self._key_cache: Dict[str, str] = {}

        # Generate master key on initialization to ensure it's available
        self._get_master_key()

    def _get_master_key(self) -> str:
        """Get or create master encryption key.

        Policy:
        - In normal operation, `APGI_MASTER_KEY` must be explicitly provided.
        - For development/tests, an ephemeral key is allowed only when
          `APGI_ALLOW_EPHEMERAL_MASTER_KEY=1` (or true/yes) is set.
        """
        master_key = os.environ.get("APGI_MASTER_KEY")
        if not master_key:
            allow_ephemeral = os.environ.get("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            if not allow_ephemeral:
                raise RuntimeError(
                    "APGI_MASTER_KEY is not set. Set APGI_MASTER_KEY for persistent key encryption, "
                    "or (dev/tests only) set APGI_ALLOW_EPHEMERAL_MASTER_KEY=1 to allow an ephemeral key."
                )

            master_key = Fernet.generate_key().decode()  # Convert bytes to base64 string
            os.environ["APGI_MASTER_KEY"] = master_key
            self.logger.warning(
                "APGI_MASTER_KEY not set; generated ephemeral key because "
                "APGI_ALLOW_EPHEMERAL_MASTER_KEY is enabled. Previously encrypted keys may not decrypt."
            )
        return master_key

    def _load_encrypted_key(self, key_file: Path) -> str:
        """
        Load and decrypt key from encrypted file.

        Args:
            key_file: Path to encrypted key file

        Returns:
            Decrypted key hex string
        """
        if not key_file.exists():
            raise FileNotFoundError(f"Key file {key_file} does not exist")

        master_key = self._get_master_key()
        # master_key is a base64-encoded string, convert to bytes for Fernet
        master_key_bytes = master_key.encode() if isinstance(master_key, str) else master_key
        fernet = Fernet(master_key_bytes)

        encrypted = key_file.read_bytes()
        try:
            decrypted_b64 = fernet.decrypt(encrypted).decode("utf-8")
        except Exception as decrypt_error:
            # The key file is preserved: it may still be recoverable with the
            # correct master key, and destroying it would be unrecoverable.
            raise KeyIntegrityError(
                f"Key file {key_file.name} could not be decrypted with the current APGI_MASTER_KEY. "
                f"The file has been left intact. Restore the correct master key, or run "
                f"'python3 -m utils.key_rotation_manager --rotate' to mint new keys "
                f"(artefacts signed with the previous key will no longer verify)."
            ) from decrypt_error

        try:
            # Check if content is valid base64 by attempting to decode
            decoded_bytes = base64.b64decode(decrypted_b64)
        except Exception as decode_error:
            raise KeyIntegrityError(f"Invalid key format in {key_file.name}: {decode_error}") from decode_error

        return decoded_bytes.hex()

    def _generate_and_save_key(self, key_file: Path) -> str:
        """
        Generate new key and save encrypted.

        Args:
            key_file: Path where to save encrypted key

        Returns:
            Key fingerprint (first 16 chars of SHA256)
        """
        # Generate cryptographically secure key using secrets module (CSPRNG)
        key_bytes = secrets.token_bytes(32)

        # Calculate fingerprint
        fingerprint = hashlib.sha256(key_bytes).hexdigest()[:16]

        # Save encrypted
        master_key = self._get_master_key()
        # master_key is a base64-encoded string, convert to bytes for Fernet
        master_key_bytes = master_key.encode() if isinstance(master_key, str) else master_key
        fernet = Fernet(master_key_bytes)
        key_b64 = base64.b64encode(key_bytes).decode("utf-8")
        encrypted = fernet.encrypt(key_b64.encode("utf-8"))

        # Write securely with atomic operation
        temp_file = key_file.with_suffix(".tmp")
        fd = os.open(str(temp_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, encrypted)
        finally:
            os.close(fd)

        # Atomic rename
        temp_file.replace(key_file)

        self.logger.info(f"Generated new key with fingerprint {fingerprint}")
        return fingerprint

    def _get_or_create_key(self, cache_key: str, key_file: Path, env_var: str) -> str:
        """Resolve one key: environment override, then file, then first-run mint.

        Bootstrapping a *missing* key file is legitimate. Failing to decrypt an
        *existing* one is not — that propagates as :class:`KeyIntegrityError`
        rather than being papered over with a fresh key.

        Args:
            cache_key: Internal cache slot name.
            key_file: Encrypted key file for this purpose.
            env_var: Environment variable that overrides the file.

        Returns:
            Hex-encoded key.

        Raises:
            KeyIntegrityError: If an existing key file cannot be decrypted.
        """
        with self._lock:
            if cache_key in self._key_cache:
                return self._key_cache[cache_key]

            env_key = os.environ.get(env_var)
            if env_key and self._is_valid_hex_key(env_key):
                self._key_cache[cache_key] = env_key
                return env_key

            if key_file.exists():
                # Any failure here is fatal by design — see KeyIntegrityError.
                key_hex = self._load_encrypted_key(key_file)
            else:
                fingerprint = self._generate_and_save_key(key_file)
                key_hex = self._load_encrypted_key(key_file)
                self.logger.info(f"Generated new {env_var} with fingerprint {fingerprint}")

            self._key_cache[cache_key] = key_hex
            return key_hex

    def get_pickle_secret_key(self) -> str:
        """
        Get PICKLE_SECRET_KEY securely.

        Returns:
            Hex-encoded secret key

        Raises:
            KeyIntegrityError: If the existing key file cannot be decrypted.
        """
        return self._get_or_create_key("pickle_secret", self._pickle_key_file, "PICKLE_SECRET_KEY")

    def _is_valid_hex_key(self, key: str) -> bool:
        """Check if key is valid 64-character hex string."""
        if not key or len(key) != 64:
            return False
        try:
            int(key, 16)
            return True
        except ValueError:
            return False

    def get_backup_hmac_key(self) -> str:
        """
        Get APGI_BACKUP_HMAC_KEY securely.

        Returns:
            Hex-encoded HMAC key

        Raises:
            KeyIntegrityError: If the existing key file cannot be decrypted.
        """
        return self._get_or_create_key("backup_hmac", self._backup_key_file, "APGI_BACKUP_HMAC_KEY")

    def get_jwt_secret_key(self) -> str:
        """
        Get APGI_JWT_SECRET securely.

        Backed by its own key file. JWT signing material is never shared with
        the pickle or backup-HMAC keys: distinct cryptographic purposes get
        distinct keys, so rotating one does not silently invalidate another.

        Returns:
            Hex-encoded JWT secret key

        Raises:
            KeyIntegrityError: If the existing key file cannot be decrypted.
        """
        return self._get_or_create_key("jwt_secret", self._jwt_key_file, "APGI_JWT_SECRET")

    def clear_cache(self) -> None:
        """Clear key cache - call after key rotation."""
        with self._lock:
            self._key_cache.clear()

    def rotate_keys(self) -> dict:
        """
        Rotate all managed keys and return rotation info.

        This is the sanctioned recovery path when key material cannot be
        decrypted. Rotation is explicit and destructive by intent: artefacts
        signed with the previous keys will no longer verify.

        Returns:
            Dictionary mapping each key name to its old and new fingerprints.
        """
        managed = (
            ("pickle_secret_key", "pickle_secret", self._pickle_key_file),
            ("backup_hmac_key", "backup_hmac", self._backup_key_file),
            ("jwt_secret_key", "jwt_secret", self._jwt_key_file),
        )

        with self._lock:
            # Clear cache to force reload of new keys
            self.clear_cache()

            rotation_results: Dict[str, Dict[str, Optional[str]]] = {}
            summary_parts = []

            for label, cache_key, key_file in managed:
                old_fingerprint = None
                if key_file.exists():
                    try:
                        old_key_hex = self._load_encrypted_key(key_file)
                        old_fingerprint = hashlib.sha256(old_key_hex.encode()).hexdigest()[:16]
                    except KeyIntegrityError:
                        # Undecryptable material is precisely what rotation
                        # exists to replace; record it as unknown and continue.
                        old_fingerprint = None
                        self.logger.warning(f"Previous {label} could not be read; rotating to a fresh key")

                new_fingerprint = self._generate_and_save_key(key_file)
                self._key_cache[cache_key] = self._load_encrypted_key(key_file)

                rotation_results[label] = {
                    "old_fingerprint": old_fingerprint,
                    "new_fingerprint": new_fingerprint,
                }
                old_short = f"{old_fingerprint[:8]}..." if old_fingerprint else "none"
                summary_parts.append(f"{label} {old_short} -> {new_fingerprint[:8]}...")

            self.logger.info(f"Key rotation completed: {', '.join(summary_parts)}")

            return rotation_results

    def invalidate_all_references(self) -> None:
        """
        Invalidate all in-flight references by clearing cache and forcing reload.
        This should be called after key rotation to ensure no stale references.
        """
        with self._lock:
            self.clear_cache()

            # Force reload of keys on next access
            self._pickle_key_file = self._pickle_key_file
            self._backup_key_file = self._backup_key_file

            self.logger.info("All key references invalidated - keys will be reloaded on next access")


# Global secure key manager instance
_secure_key_manager: Optional[SecureKeyManager] = None


def get_secure_key_manager() -> SecureKeyManager:
    """Get or create global secure key manager instance (thread-safe)."""
    global _secure_key_manager
    if _secure_key_manager is None:
        with _singleton_lock:
            # Double-check pattern
            if _secure_key_manager is None:
                _secure_key_manager = SecureKeyManager()
    return _secure_key_manager


def get_pickle_secret_key() -> str:
    """Get PICKLE_SECRET_KEY securely."""
    # Check environment variable first for global functions
    env_key = os.environ.get("PICKLE_SECRET_KEY")
    if env_key and len(env_key) == 64 and all(c in "0123456789abcdefABCDEF" for c in env_key):
        return env_key
    return get_secure_key_manager().get_pickle_secret_key()


def get_backup_hmac_key() -> str:
    """Get APGI_BACKUP_HMAC_KEY securely."""
    # Check environment variable first for global functions
    env_key = os.environ.get("APGI_BACKUP_HMAC_KEY")
    if env_key and len(env_key) == 64 and all(c in "0123456789abcdefABCDEF" for c in env_key):
        return env_key
    return get_secure_key_manager().get_backup_hmac_key()


def get_jwt_secret() -> str:
    """Get APGI_JWT_SECRET securely.

    Uses dedicated JWT key material. Reusing the pickle secret here would mean
    a pickle-key rotation silently invalidated every issued token, and would
    give one compromised key authority over two trust domains.
    """
    # Check environment variable first for global functions
    env_key = os.environ.get("APGI_JWT_SECRET")
    if env_key and len(env_key) == 64 and all(c in "0123456789abcdefABCDEF" for c in env_key):
        return env_key
    return get_secure_key_manager().get_jwt_secret_key()


def rotate_keys() -> dict:
    """Rotate both keys securely."""
    return get_secure_key_manager().rotate_keys()


def invalidate_all_key_references() -> None:
    """Invalidate all in-flight key references."""
    get_secure_key_manager().invalidate_all_references()
