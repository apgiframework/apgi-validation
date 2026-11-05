#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import logging
import sys
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

import jwt

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.secure_key_manager import get_jwt_secret

logger = logging.getLogger("auth_adapter")

T = TypeVar("T", bound=Callable[..., Any])


class Role(Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    AUDITOR = "auditor"
    GUEST = "guest"


# Least privilege defaults
DEFAULT_ROLE = Role.GUEST


class AuthSession:
    """Represents an active authenticated session."""

    def __init__(self, user_id: str, role: Role, token: str, expires_at: float):
        self.user_id = user_id
        self.role = role
        self.token = token
        self.expires_at = expires_at

    def is_valid(self) -> bool:
        return time.time() < self.expires_at


class AuthAdapter:
    """Central adapter for AuthN and AuthZ operations."""

    def __init__(self) -> None:
        # Use secure key manager for JWT secret
        try:
            secret = get_jwt_secret()
            if not secret:
                logger.critical("JWT secret could not be loaded from secure key manager.")
                raise RuntimeError("JWT secret not available from secure key manager. Check key configuration.")
            self.jwt_secret: str = secret
            self.algorithm = "HS256"
            self._sessions: Dict[str, AuthSession] = {}
        except Exception as e:
            logger.critical(f"Failed to initialize AuthAdapter: {e}")
            raise RuntimeError(f"Authentication system initialization failed: {e}")

    def generate_token(self, user_id: str, role: Role, expiration_hours: int = 1) -> str:
        """Generate a secure JWT token for a session."""
        payload = {
            "sub": user_id,
            "role": role.value,
            "exp": time.time() + (expiration_hours * 3600),
            "iat": time.time(),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> Optional[AuthSession]:
        """Validate a JWT token and establish/return a session."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.algorithm])
            user_id = cast(str, payload.get("sub", ""))
            role_str = cast(str, payload.get("role", DEFAULT_ROLE.value))

            try:
                role = Role(role_str)
            except ValueError:
                role = DEFAULT_ROLE

            session = AuthSession(
                user_id=user_id,
                role=role,
                token=token,
                expires_at=float(payload.get("exp", 0)),
            )
            self._sessions[token] = session
            return session

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None

    def check_permission(self, token: str, required_roles: List[Role]) -> bool:
        """Check if the given token has one of the required roles."""
        session = self.validate_token(token)
        if not session or not session.is_valid():
            return False

        # No implicit role override (APGI-API-002, docs/Compliance-Matrix.md):
        # an operation that admits administrators lists Role.ADMIN explicitly.
        return session.role in required_roles


# Global Auth Adapter (lazy-initialized to avoid env var requirement at import time)
_auth_manager: Optional[AuthAdapter] = None


def get_auth_manager() -> AuthAdapter:
    """Get or create the global AuthAdapter instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthAdapter()
    return _auth_manager


def require_roles(roles: List[Role]) -> Callable[[T], T]:
    """Decorator to enforce role-based access control."""

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(token: str, *args: Any, **kwargs: Any) -> Any:
            if not get_auth_manager().check_permission(token, roles):
                raise PermissionError(f"Access denied. Requires one of: {[r.value for r in roles]}")
            return func(token, *args, **kwargs)

        return cast(T, wrapper)

    return decorator


if __name__ == "__main__":
    # Demo/test mode
    print("APGI Authentication & Authorization Adapter")
    print("=" * 50)
    try:
        auth_manager = get_auth_manager()
        print("✅ AuthAdapter initialized successfully")
        print(f"   - Available roles: {[r.value for r in Role]}")
        print(f"   - Default role: {DEFAULT_ROLE.value}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
