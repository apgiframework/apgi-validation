"""Security regression tests for SecureKeyManager.

Guards two defects that made key handling fail *open*:

1. On decryption failure the manager logged a warning, deleted the key file and
   minted a replacement. Every backup HMAC and issued JWT under the old key
   became unverifiable while the system reported success.
2. get_jwt_secret() returned the pickle secret, giving one key authority over
   two trust domains and coupling JWT validity to pickle-key rotation.
"""

import hashlib

import pytest
from cryptography.fernet import Fernet

from utils.secure_key_manager import KeyIntegrityError, SecureKeyManager


@pytest.fixture()
def master_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("APGI_MASTER_KEY", key)
    # Ensure file-backed paths are exercised rather than env overrides.
    monkeypatch.delenv("PICKLE_SECRET_KEY", raising=False)
    monkeypatch.delenv("APGI_BACKUP_HMAC_KEY", raising=False)
    monkeypatch.delenv("APGI_JWT_SECRET", raising=False)
    return key


@pytest.fixture()
def manager(tmp_path, master_key):
    return SecureKeyManager(keys_dir=str(tmp_path / "keys"))


def _rotate_master_key(monkeypatch):
    """Simulate a lost/changed master key."""
    monkeypatch.setenv("APGI_MASTER_KEY", Fernet.generate_key().decode())


class TestKeySeparation:
    """Distinct cryptographic purposes must use distinct key material."""

    def test_all_three_keys_are_distinct(self, manager):
        keys = {
            manager.get_pickle_secret_key(),
            manager.get_backup_hmac_key(),
            manager.get_jwt_secret_key(),
        }
        assert len(keys) == 3, "Key material is shared across purposes"

    def test_module_level_jwt_helper_does_not_return_pickle_key(self, tmp_path, master_key, monkeypatch):
        import utils.secure_key_manager as skm

        instance = SecureKeyManager(keys_dir=str(tmp_path / "keys"))
        monkeypatch.setattr(skm, "_secure_key_manager", instance)

        assert skm.get_jwt_secret() != skm.get_pickle_secret_key()
        assert skm.get_jwt_secret() == instance.get_jwt_secret_key()

    def test_keys_are_64_char_hex(self, manager):
        for key in (
            manager.get_pickle_secret_key(),
            manager.get_backup_hmac_key(),
            manager.get_jwt_secret_key(),
        ):
            assert len(key) == 64
            int(key, 16)


class TestFailClosedOnUndecryptableKey:
    """An existing key that will not decrypt must be fatal, never replaced."""

    @pytest.mark.parametrize(
        "accessor",
        ["get_pickle_secret_key", "get_backup_hmac_key", "get_jwt_secret_key"],
    )
    def test_decrypt_failure_raises(self, tmp_path, master_key, monkeypatch, accessor):
        keys_dir = str(tmp_path / "keys")
        SecureKeyManager(keys_dir=keys_dir).__getattribute__(accessor)()

        _rotate_master_key(monkeypatch)
        fresh = SecureKeyManager(keys_dir=keys_dir)

        with pytest.raises(KeyIntegrityError):
            getattr(fresh, accessor)()

    def test_key_file_is_preserved_on_failure(self, tmp_path, master_key, monkeypatch):
        """Deleting the file would destroy material recoverable with the right key."""
        keys_dir = tmp_path / "keys"
        SecureKeyManager(keys_dir=str(keys_dir)).get_pickle_secret_key()
        key_file = keys_dir / "pickle_secret_key.enc"
        original = key_file.read_bytes()

        _rotate_master_key(monkeypatch)
        with pytest.raises(KeyIntegrityError):
            SecureKeyManager(keys_dir=str(keys_dir)).get_pickle_secret_key()

        assert key_file.exists(), "Key file was destroyed on decryption failure"
        assert key_file.read_bytes() == original, "Key file was overwritten"

    def test_error_names_the_recovery_path(self, tmp_path, master_key, monkeypatch):
        keys_dir = str(tmp_path / "keys")
        SecureKeyManager(keys_dir=keys_dir).get_pickle_secret_key()

        _rotate_master_key(monkeypatch)
        with pytest.raises(KeyIntegrityError) as excinfo:
            SecureKeyManager(keys_dir=keys_dir).get_pickle_secret_key()

        message = str(excinfo.value)
        assert "APGI_MASTER_KEY" in message
        assert "rotate" in message.lower()

    def test_corrupted_file_raises(self, tmp_path, master_key):
        keys_dir = tmp_path / "keys"
        manager = SecureKeyManager(keys_dir=str(keys_dir))
        manager.get_pickle_secret_key()

        (keys_dir / "pickle_secret_key.enc").write_bytes(b"not-a-fernet-token")

        with pytest.raises(KeyIntegrityError):
            SecureKeyManager(keys_dir=str(keys_dir)).get_pickle_secret_key()

    def test_restoring_correct_master_key_recovers_the_key(self, tmp_path, master_key, monkeypatch):
        """Because the file survives, the original key is still recoverable."""
        keys_dir = str(tmp_path / "keys")
        original = SecureKeyManager(keys_dir=keys_dir).get_pickle_secret_key()

        _rotate_master_key(monkeypatch)
        with pytest.raises(KeyIntegrityError):
            SecureKeyManager(keys_dir=keys_dir).get_pickle_secret_key()

        monkeypatch.setenv("APGI_MASTER_KEY", master_key)
        assert SecureKeyManager(keys_dir=keys_dir).get_pickle_secret_key() == original


class TestFirstRunBootstrap:
    """Minting a key that does not yet exist remains legitimate."""

    def test_missing_key_file_is_created(self, tmp_path, master_key):
        keys_dir = tmp_path / "keys"
        key = SecureKeyManager(keys_dir=str(keys_dir)).get_pickle_secret_key()

        assert (keys_dir / "pickle_secret_key.enc").exists()
        assert len(key) == 64

    def test_key_is_stable_across_instances(self, tmp_path, master_key):
        keys_dir = str(tmp_path / "keys")
        first = SecureKeyManager(keys_dir=keys_dir).get_pickle_secret_key()
        second = SecureKeyManager(keys_dir=keys_dir).get_pickle_secret_key()
        assert first == second


class TestEnvironmentOverride:
    def test_valid_env_key_takes_precedence(self, tmp_path, master_key, monkeypatch):
        override = "a" * 64
        monkeypatch.setenv("PICKLE_SECRET_KEY", override)
        assert SecureKeyManager(keys_dir=str(tmp_path / "keys")).get_pickle_secret_key() == override

    @pytest.mark.parametrize("bad", ["too-short", "z" * 64, "your_64_character_hex_string_here"])
    def test_invalid_env_key_is_ignored(self, tmp_path, master_key, monkeypatch, bad):
        """A placeholder in .env must not become live signing material."""
        monkeypatch.setenv("APGI_JWT_SECRET", bad)
        resolved = SecureKeyManager(keys_dir=str(tmp_path / "keys")).get_jwt_secret_key()
        assert resolved != bad
        assert len(resolved) == 64


class TestRotation:
    """Rotation is the sanctioned, explicit recovery path."""

    def test_rotation_replaces_all_three_keys(self, tmp_path, master_key):
        keys_dir = str(tmp_path / "keys")
        manager = SecureKeyManager(keys_dir=keys_dir)
        before = (
            manager.get_pickle_secret_key(),
            manager.get_backup_hmac_key(),
            manager.get_jwt_secret_key(),
        )

        results = manager.rotate_keys()

        assert set(results) == {"pickle_secret_key", "backup_hmac_key", "jwt_secret_key"}
        after = (
            manager.get_pickle_secret_key(),
            manager.get_backup_hmac_key(),
            manager.get_jwt_secret_key(),
        )
        assert all(a != b for a, b in zip(before, after))

    def test_rotation_reports_old_and_new_fingerprints(self, tmp_path, master_key):
        manager = SecureKeyManager(keys_dir=str(tmp_path / "keys"))
        original = manager.get_pickle_secret_key()

        entry = manager.rotate_keys()["pickle_secret_key"]

        assert entry["old_fingerprint"] == hashlib.sha256(original.encode()).hexdigest()[:16]
        assert entry["new_fingerprint"] != entry["old_fingerprint"]

    def test_rotation_recovers_from_undecryptable_keys(self, tmp_path, master_key, monkeypatch):
        """Rotation must work even when the old material cannot be read."""
        keys_dir = str(tmp_path / "keys")
        SecureKeyManager(keys_dir=keys_dir).get_pickle_secret_key()

        _rotate_master_key(monkeypatch)
        manager = SecureKeyManager(keys_dir=keys_dir)

        results = manager.rotate_keys()
        assert results["pickle_secret_key"]["old_fingerprint"] is None
        assert results["pickle_secret_key"]["new_fingerprint"]
        assert len(manager.get_pickle_secret_key()) == 64
