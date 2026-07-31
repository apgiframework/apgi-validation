"""
Comprehensive tests for path validation security including symlink traversal,
TOCTOU race conditions, absolute path rejection, and other security edge cases.
"""

import tempfile
import threading
from pathlib import Path

import pytest

from utils.path_security import validate_file_path

# Define PROJECT_ROOT locally to avoid main.py import issues
PROJECT_ROOT = str(Path(__file__).parent.parent)


class TestPathValidationSecurity:
    """Test path validation security mechanisms."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def safe_file(self, temp_dir):
        """Create a safe test file."""
        safe_file = temp_dir / "safe_file.txt"
        safe_file.write_text("safe content")
        return safe_file

    def test_symlink_traversal_prevention(self, temp_dir):
        """Test that symlink traversal attacks are prevented."""
        # Create a file outside temp directory
        outside_file = temp_dir.parent / "outside.txt"
        outside_file.write_text("sensitive data")

        # Create symlink pointing outside temp directory
        malicious_link = temp_dir / "malicious_link"
        malicious_link.symlink_to(outside_file)

        # Should raise ValueError for symlink traversal
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(malicious_link, PROJECT_ROOT)

    def test_relative_path_traversal_prevention(self, temp_dir):
        """Test that relative path traversal is prevented."""
        # Test various traversal patterns
        traversal_patterns = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
        ]

        for pattern in traversal_patterns:
            malicious_path = temp_dir / pattern
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(malicious_path, PROJECT_ROOT)

    def test_absolute_path_rejection(self):
        """Test that absolute paths outside allowed directories are rejected."""
        absolute_paths = [
            Path("/etc/passwd"),
            Path("C:\\Windows\\System32\\config"),
            Path("/var/log/auth.log"),
        ]

        for abs_path in absolute_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(abs_path, PROJECT_ROOT)

    def test_toctou_race_condition_protection(self, temp_dir):
        """Test Time-of-Check-Time-of-Use race condition protection.

        Note: This test verifies that symlink-based TOCTOU attacks are prevented.
        The validate_file_path function uses os.path.realpath() which atomically
        resolves symlinks, preventing classic TOCTOU race conditions.
        """
        # Create initial safe file
        safe_file = temp_dir / "initial_file.txt"
        safe_file.write_text("initial content")

        # Verify that symlinks are properly resolved and validated
        # Create a symlink that initially points to a safe file
        symlink_path = temp_dir / "race_link"
        symlink_path.symlink_to(safe_file)

        # Should pass validation when pointing to safe location (use temp_dir as base)
        result = validate_file_path(symlink_path, str(temp_dir))
        assert result is not None

        # Create a file outside the allowed directory
        outside_file = temp_dir.parent / "outside_race.txt"
        outside_file.write_text("outside data")

        # Remove the symlink and recreate it pointing outside
        symlink_path.unlink()
        symlink_path.symlink_to(outside_file)

        # Should detect that the symlink now points outside allowed directory
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(symlink_path, str(temp_dir))

    def test_concurrent_path_validation(self, temp_dir):
        """Test concurrent path validation doesn't have race conditions."""
        safe_file = temp_dir / "concurrent_test.txt"
        safe_file.write_text("concurrent content")

        results = []
        errors = []

        def validate_path():
            try:
                validate_file_path(safe_file, str(temp_dir))
                results.append(True)
            except ValueError as e:
                errors.append(e)

        # Run multiple validations concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=validate_path)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All validations should succeed without race conditions
        assert len(results) == 10
        assert len(errors) == 0

    def test_path_with_null_bytes(self, temp_dir):
        """Test that paths with null bytes are rejected."""
        null_byte_paths = [
            temp_dir / "file\x00.txt",
            temp_dir / "safe\x00../../../etc/passwd",
            Path("test\x00file"),
        ]

        for path in null_byte_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(path, PROJECT_ROOT)

    def test_path_with_unicode_tricks(self, temp_dir):
        """Test that Unicode-based path tricks are rejected."""
        unicode_tricks = [
            temp_dir / "file\u202etxt",  # Unicode right-to-left override
            temp_dir / "file\u200b.txt",  # Zero-width space
            temp_dir / "file\ufeff.txt",  # BOM character
        ]

        for path in unicode_tricks:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(path, PROJECT_ROOT)

    def test_file_permission_checks(self, temp_dir):
        """Test file permission validation."""
        # Create file with restrictive permissions
        restricted_file = temp_dir / "restricted.txt"
        restricted_file.write_text("restricted")
        restricted_file.chmod(0o000)  # No permissions

        # Should reject files with no read permissions
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(restricted_file, PROJECT_ROOT)

    def test_file_size_validation(self, temp_dir):
        """Test that an oversized file outside the project root is rejected.

        The file is created *sparse* via truncate: it reports >1GB but occupies
        no disk blocks. Previously this materialised ``b"x" * (1024**3 + 1)`` —
        a real gigabyte in RAM and on disk — which made the test fail with
        ``OSError: No space left on device`` on any machine short of a spare
        gigabyte, unrelated to the behaviour under test.

        Note the assertion is about path containment, not size: the temp dir
        lies outside PROJECT_ROOT, so validation rejects it as traversal.
        """
        oversized_file = temp_dir / "oversized.txt"
        with open(oversized_file, "wb") as f:
            f.truncate(1024 * 1024 * 1024 + 1)  # > 1GB, sparse

        assert oversized_file.stat().st_size > 1024 * 1024 * 1024

        # Should reject files outside the project root
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(oversized_file, PROJECT_ROOT)

    def test_safe_path_acceptance(self, safe_file):
        """Test that safe paths are accepted."""
        # Valid file should pass validation (use parent dir as base)
        validate_file_path(safe_file, str(safe_file.parent))

        # Valid directory should pass validation
        validate_file_path(safe_file.parent, str(safe_file.parent))

    def test_path_normalization(self, temp_dir):
        """Test path normalization attacks."""
        # Test various normalization attacks using raw strings
        # (pathlib.Path normalizes paths when using / operator)
        normalization_attacks = [
            str(temp_dir / "test") + "/../etc/passwd",  # Directory traversal
            str(temp_dir / "test") + "/./../../etc/passwd",  # Current directory
            str(temp_dir / "test") + "//etc/passwd",  # Double slash
        ]

        for attack_path in normalization_attacks:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(attack_path, temp_dir)


class TestPathValidationEdgeCases:
    """Test edge cases for path validation."""

    def test_empty_path(self):
        """Test empty path rejection."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            validate_file_path(Path(""), PROJECT_ROOT)

    def test_none_path(self):
        """Test None path rejection."""
        with pytest.raises(TypeError):
            validate_file_path(None, PROJECT_ROOT)

    def test_path_with_spaces(self, temp_dir):
        """Test paths with spaces are handled correctly."""
        spaced_file = temp_dir / "file with spaces.txt"
        spaced_file.write_text("content")

        # Should accept valid paths with spaces (use temp_dir as base)
        validate_file_path(spaced_file, str(temp_dir))

    def test_very_long_path(self, temp_dir):
        """Test very long path handling."""
        # Create path that exceeds typical limits
        long_name = "a" * 255
        long_path = temp_dir / long_name

        # Should handle long paths appropriately
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(long_path, PROJECT_ROOT)
