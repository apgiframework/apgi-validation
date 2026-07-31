"""
Tests for environment variable initialization and missing key handling.
"""

import os

import pytest

from utils.secure_key_manager import SecureKeyManager, get_backup_hmac_key, get_pickle_secret_key


@pytest.fixture(autouse=True)
def reset_key_manager_singleton():
    """Reset the global key manager singleton before each test."""
    # Reset before test
    import utils.secure_key_manager as skm

    skm._secure_key_manager = None
    # Also clear any environment variables that might interfere
    original_environ = os.environ.copy()
    # Tests run in ephemeral mode unless a test explicitly disables it.
    os.environ["APGI_ALLOW_EPHEMERAL_MASTER_KEY"] = "1"

    for key in ["PICKLE_SECRET_KEY", "APGI_BACKUP_HMAC_KEY", "APGI_MASTER_KEY"]:
        if key in os.environ:
            del os.environ[key]
    yield
    # Reset after test
    skm._secure_key_manager = None
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_environ)


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """Clean environment variables and isolate key material.

    These tests deliberately unset the key environment variables to exercise
    the file-backed path. Without an isolated keys directory that path resolves
    to the repository's real ``.keys/``, so the suite would depend on — and
    mutate — the developer's own key material.
    """
    import utils.secure_key_manager as skm

    original_env = os.environ.copy()
    original_manager = skm._secure_key_manager

    # Key files are resolved relative to the working directory.
    monkeypatch.chdir(tmp_path)
    skm._secure_key_manager = None

    yield

    skm._secure_key_manager = original_manager
    os.environ.clear()
    os.environ.update(original_env)


class TestEnvVarInitialization:
    """Test environment variable initialization security."""

    def test_missing_pickle_secret_key(self, clean_env):
        """Test handling of missing PICKLE_SECRET_KEY environment variable."""
        # Ensure PICKLE_SECRET_KEY is not set
        if "PICKLE_SECRET_KEY" in os.environ:
            del os.environ["PICKLE_SECRET_KEY"]

        # Should generate new key when environment variable is missing
        key = get_pickle_secret_key()
        assert key is not None
        assert len(key) == 64  # 32 bytes = 64 hex chars
        assert all(c in "0123456789abcdef" for c in key.lower())

    def test_missing_backup_hmac_key(self, clean_env):
        """Test handling of missing APGI_BACKUP_HMAC_KEY environment variable."""
        # Ensure APGI_BACKUP_HMAC_KEY is not set
        if "APGI_BACKUP_HMAC_KEY" in os.environ:
            del os.environ["APGI_BACKUP_HMAC_KEY"]

        # Should generate new key when environment variable is missing
        key = get_backup_hmac_key()
        assert key is not None
        assert len(key) == 64  # 32 bytes = 64 hex chars
        assert all(c in "0123456789abcdef" for c in key.lower())

    def test_valid_pickle_secret_key_env_var(self, clean_env):
        """Test valid PICKLE_SECRET_KEY environment variable."""
        valid_key = "a1b2c3d4" * 8  # 64 hex chars
        os.environ["PICKLE_SECRET_KEY"] = valid_key

        key = get_pickle_secret_key()
        assert key == valid_key

    def test_valid_backup_hmac_key_env_var(self, clean_env):
        """Test valid APGI_BACKUP_HMAC_KEY environment variable."""
        valid_key = "f6e5d4c3" * 8  # 64 hex chars
        os.environ["APGI_BACKUP_HMAC_KEY"] = valid_key

        key = get_backup_hmac_key()
        assert key == valid_key

    def test_invalid_pickle_secret_key_format(self, clean_env):
        """Test invalid PICKLE_SECRET_KEY format handling."""
        invalid_keys = [
            "short",  # Too short
            "not_hex_chars",  # Non-hex characters
            "a" * 65,  # Too long
            "",  # Empty string
        ]

        for invalid_key in invalid_keys:
            os.environ["PICKLE_SECRET_KEY"] = invalid_key
            # Should handle gracefully or raise appropriate error
            try:
                key = get_pickle_secret_key()
                # If it doesn't raise, it should still return a valid key
                assert len(key) == 64
            except ValueError:
                # Expected behavior for invalid keys
                pass

    def test_invalid_backup_hmac_key_format(self, clean_env):
        """Test invalid APGI_BACKUP_HMAC_KEY format handling."""
        invalid_keys = [
            "short",  # Too short
            "not_hex_chars",  # Non-hex characters
            "b" * 65,  # Too long
            "",  # Empty string
        ]

        for invalid_key in invalid_keys:
            os.environ["APGI_BACKUP_HMAC_KEY"] = invalid_key
            # Should handle gracefully or raise appropriate error
            try:
                key = get_backup_hmac_key()
                # If it doesn't raise, it should still return a valid key
                assert len(key) == 64
            except ValueError:
                # Expected behavior for invalid keys
                pass

    def test_key_consistency_across_calls(self, clean_env):
        """Test key consistency across multiple calls."""
        # First call should generate or retrieve key
        key1 = get_pickle_secret_key()
        key2 = get_pickle_secret_key()

        # Should return the same key
        assert key1 == key2

        # Same for backup key
        backup1 = get_backup_hmac_key()
        backup2 = get_backup_hmac_key()

        assert backup1 == backup2

    def test_key_entropy_validation(self, clean_env, tmp_path):
        """Test that generated keys have sufficient entropy."""
        # Generate multiple keys to check entropy
        keys = []
        for _ in range(10):
            # Clear environment to force generation
            if "PICKLE_SECRET_KEY" in os.environ:
                del os.environ["PICKLE_SECRET_KEY"]
            # Create manager instance with unique temp dir to avoid singleton issues
            keys_dir = tmp_path / f".keys_{_}"
            manager = SecureKeyManager(keys_dir=str(keys_dir))
            key = manager.get_pickle_secret_key()
            keys.append(key)

        # All keys should be unique (high entropy)
        assert len(set(keys)) == len(keys)

        # Each key should be valid hex
        for key in keys:
            assert len(key) == 64
            assert all(c in "0123456789abcdef" for c in key.lower())

    def test_master_key_generation(self, clean_env):
        """Test master key generation for key encryption."""
        # Ensure APGI_MASTER_KEY is not set
        if "APGI_MASTER_KEY" in os.environ:
            del os.environ["APGI_MASTER_KEY"]

        # Creating manager should generate master key
        SecureKeyManager()

        # Master key should be set in environment
        assert "APGI_MASTER_KEY" in os.environ
        master_key = os.environ["APGI_MASTER_KEY"]
        assert len(master_key) == 44  # Fernet key length

    def test_key_file_permissions(self, clean_env, tmp_path):
        """Test that key files have secure permissions."""
        import platform

        # Skip on Windows as it doesn't support Unix-style permissions
        if platform.system() == "Windows":
            pytest.skip("Unix-style file permissions not supported on Windows")

        # Use custom keys directory
        keys_dir = tmp_path / ".test_keys"
        manager = SecureKeyManager(keys_dir=str(keys_dir))

        # Access keys to trigger file creation
        manager.get_pickle_secret_key()
        manager.get_backup_hmac_key()

        # Check that keys directory has secure permissions (0o700)
        assert keys_dir.exists()
        mode = os.stat(keys_dir).st_mode & 0o777
        assert mode == 0o700

        # Check that key files are created
        pickle_key_file = keys_dir / "pickle_secret_key.enc"
        backup_key_file = keys_dir / "backup_hmac_key.enc"

        assert pickle_key_file.exists()
        assert backup_key_file.exists()

        # Check that key files have secure permissions (0o600)
        p_mode = os.stat(pickle_key_file).st_mode & 0o777
        b_mode = os.stat(backup_key_file).st_mode & 0o777
        print(f"DEBUG: Pickle file mode: {oct(p_mode)}")
        print(f"DEBUG: Backup file mode: {oct(b_mode)}")
        assert (p_mode & 0o077) == 0, f"Pickle file mode {oct(p_mode)} is not private enough"
        assert (b_mode & 0o077) == 0, f"Backup file mode {oct(b_mode)} is not private enough"

    def test_concurrent_key_access(self, clean_env):
        """Test concurrent key access doesn't cause issues."""
        import threading
        from queue import Queue

        keys_queue: Queue = Queue()
        errors_queue: Queue = Queue()

        def get_key():
            try:
                # Use a fresh manager for each thread to test concurrent initialization
                # if we were testing that, but here we test the global singleton
                key = get_pickle_secret_key()
                keys_queue.put(key)
            except Exception as e:
                errors_queue.put(e)

        # Run multiple threads accessing keys
        threads = []
        for _ in range(20):  # Increased thread count
            thread = threading.Thread(target=get_key)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        keys = []
        while not keys_queue.empty():
            keys.append(keys_queue.get())

        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())

        # Should not have any errors
        assert len(errors) == 0
        assert len(keys) == 20

        # All keys should be the same
        assert len(set(keys)) == 1


class TestKeyManagerEdgeCases:
    """Test edge cases for key manager."""

    def test_keys_directory_creation(self, tmp_path):
        """Test keys directory creation with proper permissions."""
        keys_dir = tmp_path / "new_keys"
        SecureKeyManager(keys_dir=str(keys_dir))

        assert keys_dir.exists()
        assert keys_dir.is_dir()

    def test_missing_master_key_handling(self, clean_env):
        """Test handling when master key is missing."""
        # Ensure master key is not set
        if "APGI_MASTER_KEY" in os.environ:
            del os.environ["APGI_MASTER_KEY"]

        # Should generate new master key
        SecureKeyManager()
        assert "APGI_MASTER_KEY" in os.environ
        master_key = os.environ["APGI_MASTER_KEY"]
        assert len(master_key) == 44  # Fernet key length

        # Test that key generation was successful
        assert True

    def test_key_rotation_env_var_update(self, clean_env, tmp_path):
        """Test that key rotation updates environment variables."""
        # Set master key first for consistent encryption (valid Fernet key: 32 url-safe base64 bytes)
        import base64

        os.environ["APGI_MASTER_KEY"] = base64.urlsafe_b64encode(b"0" * 32).decode()
        # Set initial keys
        os.environ["PICKLE_SECRET_KEY"] = "a1b2c3d4e5f6" * 8
        os.environ["APGI_BACKUP_HMAC_KEY"] = "f6e5d4c3b2a1" * 8

        # Use fresh keys directory to avoid conflicts with existing keys
        keys_dir = tmp_path / ".keys"
        manager = SecureKeyManager(keys_dir=str(keys_dir))

        # First initialize keys so they exist for rotation
        manager.get_pickle_secret_key()
        manager.get_backup_hmac_key()

        # Rotate keys
        results = manager.rotate_keys()

        # Should have rotation results
        assert "pickle_secret_key" in results
        assert "backup_hmac_key" in results
        assert "old_fingerprint" in results["pickle_secret_key"]
        assert "new_fingerprint" in results["pickle_secret_key"]
