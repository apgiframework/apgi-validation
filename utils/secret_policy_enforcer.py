#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for standalone execution
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.secure_key_manager import get_secure_key_manager

logger = logging.getLogger("secret_policy")


def enforce_secret_policy():
    """
    Check environment and key status to enforce strict secret policy.
    Aborts application startup if policy is violated.
    """
    logger.info("Enforcing APGI Secret Management Policy...")

    # 1. Mandatory env checks
    master_key = os.environ.get("APGI_MASTER_KEY")
    if not master_key:
        allow_ephemeral = os.environ.get("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not allow_ephemeral:
            logger.error("CRITICAL: APGI_MASTER_KEY is not set in environment.")
            logger.error(
                "Set APGI_MASTER_KEY, or (dev/tests only) set APGI_ALLOW_EPHEMERAL_MASTER_KEY=1. Aborting startup."
            )
            sys.exit(1)
        logger.warning(
            "APGI_MASTER_KEY is not set; ephemeral master key is allowed because "
            "APGI_ALLOW_EPHEMERAL_MASTER_KEY is enabled."
        )

    # 2. Verify Key Manager can load keys (validates encryption).
    # A KeyIntegrityError here means existing key material could not be
    # decrypted; that is fatal by design and must not be papered over.
    try:
        manager = get_secure_key_manager()
        manager.get_pickle_secret_key()
        manager.get_backup_hmac_key()
        manager.get_jwt_secret_key()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load secure keys: {e}")
        logger.error("Key integrity check failed. Aborting startup.")
        sys.exit(1)

    # 3. Reject malformed key overrides.
    #
    # Every managed key must be exactly 64 hex characters. A value that is set
    # but malformed is *silently ignored* by the key manager, which falls back
    # to file-backed material — so an operator who believes they configured a
    # secret has in fact configured nothing. Failing loudly here is the point:
    # the previous check only rejected values under 8 characters or drawn from
    # a four-item denylist, so placeholders like
    # "your_64_character_hex_string_here" sailed through.
    managed_keys = [
        "APGI_JWT_SECRET",
        "PICKLE_SECRET_KEY",
        "APGI_BACKUP_HMAC_KEY",
    ]

    for secret in managed_keys:
        val = os.environ.get(secret)
        if not val:
            continue  # Unset is fine: the key manager supplies it.

        if len(val) != 64 or not all(c in "0123456789abcdefABCDEF" for c in val):
            logger.error(
                f"CRITICAL: {secret} is set but is not a 64-character hex string. "
                f"It would be silently ignored, leaving the key manager to use different material."
            )
            logger.error('Generate one with: python3 -c "import os; print(os.urandom(32).hex())". Aborting startup.')
            sys.exit(1)

    logger.info("Secret Management Policy checks passed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    enforce_secret_policy()
