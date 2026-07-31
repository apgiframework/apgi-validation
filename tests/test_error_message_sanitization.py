"""Security tests for _sanitize_error_message.

Historical defect: the sanitizer only redacted long opaque strings (40+ chars)
and hex blobs, so credentials in ``name=value`` form — ``password=hunter2``,
``token=abc123``, ``api_key=sk-live-1`` — passed through verbatim into logs,
contrary to the function's documented contract that it removes "API keys and
tokens".
"""

import pytest

from main import _sanitize_error_message as sanitize

SECRET_VALUES = ["secret123", "abc123", "hunter2", "sk-live-9999", "xyz789"]


class TestCredentialRedaction:
    @pytest.mark.parametrize(
        "message,secret",
        [
            ("Error: password=secret123", "secret123"),
            ("Error: passwd=secret123", "secret123"),
            ("Error: passphrase=secret123", "secret123"),
            ("token=abc123 rejected", "abc123"),
            ("api_key=sk-live-9999", "sk-live-9999"),
            ("api-key=sk-live-9999", "sk-live-9999"),
            ("apikey=sk-live-9999", "sk-live-9999"),
            ("access_key=AKIAIOSFODNN7", "AKIAIOSFODNN7"),
            ("private_key=MIIEpAIB", "MIIEpAIB"),
            ("session_id=xyz789", "xyz789"),
            ("credential=hunter2", "hunter2"),
            ('{"secret": "hunter2"}', "hunter2"),
            ("secret: hunter2", "hunter2"),
            ("PASSWORD=secret123", "secret123"),
        ],
    )
    def test_secret_value_is_removed(self, message, secret):
        assert secret not in sanitize(message)

    def test_multiple_secrets_in_one_message(self):
        result = sanitize("Error: password=secret123, token=abc123")
        assert "secret123" not in result
        assert "abc123" not in result
        assert result.count("[REDACTED]") >= 2

    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "Basic", "BASIC"])
    def test_authorization_header_value_removed(self, scheme):
        result = sanitize(f"Rejected: {scheme} dXNlcjpwYXNzd29yZA")
        assert "dXNlcjpwYXNzd29yZA" not in result

    def test_authorization_header_with_prefix(self):
        """The header name must not shadow the credential that follows it."""
        result = sanitize("Authorization: Basic dXNlcjpwYXNz")
        assert "dXNlcjpwYXNz" not in result

    def test_jwt_is_redacted(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvZSJ9.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
        assert jwt not in sanitize(f"Token rejected: {jwt}")

    def test_hex_key_is_redacted(self):
        key = "a" * 64
        assert key not in sanitize(f"Key mismatch: {key}")


class TestNonSensitiveDataPreserved:
    """Over-redaction would make error messages useless for debugging."""

    @pytest.mark.parametrize(
        "message",
        [
            "Connection failed on port 5432",
            "n_trials=1000 alpha=0.05",
            "theta_0=0.5",
            "tau_theta=30.0",
            "Protocol VP_02 returned status=FAILED",
            "Expected 3 columns, got 5",
        ],
    )
    def test_benign_message_survives(self, message):
        result = sanitize(message)
        # The informative parts must still be present.
        for token in message.split():
            if not any(s in token for s in ("password", "secret", "token", "key")):
                assert token.split("=")[0] in result


class TestPathSanitization:
    def test_unix_path_redacted(self):
        assert "/home/user" not in sanitize("Error accessing /home/user/secret/file.txt")

    def test_message_prefix_retained(self):
        assert "Error accessing" in sanitize("Error accessing /home/user/secret/file.txt")


class TestMessageLength:
    def test_long_message_truncated(self):
        result = sanitize("x" * 5000)
        assert len(result) <= 203  # 200 chars + "..."

    def test_short_message_not_truncated(self):
        assert sanitize("brief error") == "brief error"

    def test_empty_message(self):
        assert sanitize("") == ""


class TestReturnContract:
    @pytest.mark.parametrize("message", ["", "plain", "password=x", "/a/b/c"])
    def test_always_returns_string(self, message):
        assert isinstance(sanitize(message), str)
