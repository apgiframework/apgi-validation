#!/usr/bin/env python3
"""
Client for the APGI System API (https://api.apgiframework.com).

Every APGI application talks to the same hosted API. This module is the one
place in apgi-validation that knows where it lives, so protocol and GUI code
never hardcode a URL.

Configuration (environment variables):
    APGI_API_URL      Base URL. Defaults to https://api.apgiframework.com.
                      Plain http is accepted only for loopback hosts, so a
                      local `uvicorn app.main:app` can be used in development.
    APGI_API_TOKEN    Bearer token from POST /v1/auth/login. Optional for
                      public endpoints (/health, /v1/meta, /v1/version).
    APGI_API_TIMEOUT  Per-request timeout in seconds (default 15).
"""

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("apgi_api_client")

DEFAULT_API_URL = "https://api.apgiframework.com"
API_VERSION = "v1"
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ApgiApiError(RuntimeError):
    """Raised when the API is unreachable or returns an error status."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def resolve_base_url(url: Optional[str] = None) -> str:
    """Return a validated base URL without a trailing slash."""
    base = (url or os.getenv("APGI_API_URL") or DEFAULT_API_URL).strip().rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"APGI_API_URL must be an http(s) URL, got {base!r}")
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"Refusing plain-http API URL {base!r}: credentials would travel unencrypted. "
            "Use https, or a loopback host for local development."
        )
    return base


class ApgiApiClient:
    """Thin, retrying wrapper around the APGI System API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = resolve_base_url(base_url)
        self.token = token if token is not None else os.getenv("APGI_API_TOKEN")
        self.timeout = float(timeout if timeout is not None else os.getenv("APGI_API_TIMEOUT", "15"))
        if session is None:
            session = requests.Session()
            # Retry only idempotent reads on transient gateway errors.
            retry = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset({"GET", "HEAD"}),
            )
            session.mount("https://", HTTPAdapter(max_retries=retry))
            session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session = session

    # -- low level ----------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "apgi-validation"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.request(method, url, headers=self._headers(), timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise ApgiApiError(f"APGI API unreachable at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ApgiApiError(
                f"{method} {path} failed with HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ApgiApiError(f"{method} {path} returned non-JSON body") from exc

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params=params or None)

    def post(self, path: str, payload: Dict[str, Any]) -> Any:
        return self.request("POST", path, json=payload)

    # -- endpoints ----------------------------------------------------------

    def health(self) -> Any:
        return self.get("/health")

    def meta(self) -> Any:
        """Citation, licence and release-state identifiers (public)."""
        return self.get(f"/{API_VERSION}/meta")

    def version(self) -> Any:
        return self.get(f"/{API_VERSION}/version")

    def login(self, username: str, password: str) -> str:
        """Authenticate, store the access token on the client and return it."""
        body = self.post(f"/{API_VERSION}/auth/login", {"username": username, "password": password})
        token = (body or {}).get("access_token")
        if not token:
            raise ApgiApiError("login response contained no access_token")
        self.token = token
        return str(token)

    def norms(self, study_id: str, battery_id: str, task_type: str, metric: str, score: float) -> Any:
        """Percentile rank of `score` against the study's reference sample."""
        return self.get(
            f"/{API_VERSION}/norms",
            study_id=study_id,
            battery_id=battery_id,
            task_type=task_type,
            metric=metric,
            score=score,
        )


def get_client() -> ApgiApiClient:
    """Client configured from the environment."""
    return ApgiApiClient()
