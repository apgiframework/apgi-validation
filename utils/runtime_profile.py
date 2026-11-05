"""Runtime security profile shared by every entry point (APGI-IMP-002).

Secure by default:

* Authentication is **enforced** unless ``APGI_ENFORCE_AUTH=0`` is set
  explicitly. An absent variable no longer selects development behaviour.
* ``APGI_ENFORCE_AUTH=0`` is refused when ``APGI_ENV=production``.
* Development bypasses (ephemeral master key, protocol/manifest lock skips)
  are refused when ``APGI_ENV=production``.

GUI and headless modes use the same gate, so ``--headless`` is not a way
around authentication.
"""

from __future__ import annotations

import os
from typing import Optional

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class InsecureConfigurationError(RuntimeError):
    """A development-only setting was used where it is not permitted."""


def environment_name() -> str:
    return os.environ.get("APGI_ENV", "").strip().lower() or "unspecified"


def is_production() -> bool:
    return environment_name() in {"prod", "production"}


def auth_enforced() -> bool:
    """True unless auth was explicitly disabled outside production."""
    raw = os.environ.get("APGI_ENFORCE_AUTH")
    if raw is None or raw.strip().lower() not in _FALSE:
        return True
    if is_production():
        raise InsecureConfigurationError("APGI_ENFORCE_AUTH=0 is not permitted when APGI_ENV=production")
    return False


def require_entry_point_auth(token: Optional[str], entry_point: str) -> Optional[str]:
    """Validate ``token`` for a GUI/headless entry point, or mint a dev token.

    Returns the token to use. Raises ``SystemExit`` when authentication is
    enforced and no token is supplied, and ``PermissionError`` for an invalid
    token or insufficient role.
    """
    from utils.security_gateway import Role, SecurityGateway

    if token:
        SecurityGateway().require_roles(token, [Role.RESEARCHER, Role.ADMIN])
        return token
    if auth_enforced():
        raise SystemExit(
            f"Authentication required for {entry_point}: pass --token. For local development only, "
            "set APGI_ENFORCE_AUTH=0 explicitly (refused when APGI_ENV=production)."
        )
    from utils.auth_adapter import get_auth_manager

    print(f"Development mode ({entry_point}): APGI_ENFORCE_AUTH=0, generated a local 24-hour token")
    return get_auth_manager().generate_token("dev_user", Role.RESEARCHER, 24)


def ephemeral_master_key_allowed() -> bool:
    allowed = os.environ.get("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "").strip().lower() in _TRUE
    if allowed and is_production():
        raise InsecureConfigurationError("APGI_ALLOW_EPHEMERAL_MASTER_KEY is not permitted when APGI_ENV=production")
    return allowed
