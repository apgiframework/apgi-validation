"""Tests for utils.apgi_api_client."""

from unittest.mock import MagicMock

import pytest
import requests

from utils.apgi_api_client import (
    DEFAULT_API_URL,
    ApgiApiClient,
    ApgiApiError,
    resolve_base_url,
)


def _response(status=200, json_body=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.content = b"x" if json_body is not None or text else b""
    r.text = text
    r.json.return_value = json_body
    return r


def test_default_base_url_is_hosted_api(monkeypatch):
    monkeypatch.delenv("APGI_API_URL", raising=False)
    assert resolve_base_url() == DEFAULT_API_URL == "https://api.apgiframework.com"


def test_env_override_and_trailing_slash(monkeypatch):
    monkeypatch.setenv("APGI_API_URL", "https://staging.apgiframework.com/")
    assert resolve_base_url() == "https://staging.apgiframework.com"


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://localhost:8000"])
def test_loopback_http_allowed(url):
    assert resolve_base_url(url) == url


@pytest.mark.parametrize("url", ["http://api.apgiframework.com", "ftp://x.test", "not a url"])
def test_insecure_or_invalid_url_rejected(url):
    with pytest.raises(ValueError):
        resolve_base_url(url)


def test_get_sends_bearer_token_and_builds_url():
    session = MagicMock()
    session.request.return_value = _response(json_body={"status": "ok"})
    client = ApgiApiClient(token="abc", session=session)
    assert client.health() == {"status": "ok"}
    args, kwargs = session.request.call_args
    assert args == ("GET", "https://api.apgiframework.com/health")
    assert kwargs["headers"]["Authorization"] == "Bearer abc"


def test_login_stores_token():
    session = MagicMock()
    session.request.return_value = _response(json_body={"access_token": "tok"})
    client = ApgiApiClient(token="", session=session)
    assert client.login("u", "p") == "tok"
    assert client.token == "tok"
    assert session.request.call_args.kwargs["json"] == {"username": "u", "password": "p"}


def test_http_error_raises_with_status():
    session = MagicMock()
    session.request.return_value = _response(status=401, text="nope")
    with pytest.raises(ApgiApiError) as exc:
        ApgiApiClient(session=session).meta()
    assert exc.value.status_code == 401


def test_network_failure_wrapped():
    session = MagicMock()
    session.request.side_effect = requests.ConnectionError("down")
    with pytest.raises(ApgiApiError, match="unreachable"):
        ApgiApiClient(session=session).health()
