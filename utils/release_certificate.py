"""Release certificate and certificate-before-deposit gate (APGI-REL-001, APGI-REL-005).

A release certificate names the exact candidate (artifact digests), the
verified protocol lock, the test evidence and the named approvers, and is
signed with Ed25519. :func:`assert_deposit_allowed` must be called before any
archival deposit (OSF, Zenodo, …): it refuses unless every artifact to be
deposited is byte-identical to one in a valid certificate whose approvals are
all recorded. No deposit code in this repository bypasses it — there is
currently no deposit code at all, and new code must route through here.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

PENDING_MARKERS = ("PENDING", "TBD", "")


class CertificationError(PermissionError):
    """The candidate is not certified for promotion or deposit."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(cert: Mapping[str, object]) -> bytes:
    body = {k: v for k, v in cert.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def create_certificate(
    *,
    version: str,
    source_commit: str,
    artifacts: Iterable[Path],
    protocol_lock_sha256: str,
    test_evidence: Mapping[str, str],
    approvers: Mapping[str, str],
) -> Dict[str, object]:
    """Build an unsigned certificate. ``approvers`` maps role -> named person."""
    return {
        "certificate_version": "1.0",
        "release_version": version,
        "source_commit": source_commit,
        "protocol_lock_sha256": protocol_lock_sha256,
        "artifacts": {Path(p).name: _sha256(Path(p)) for p in artifacts},
        "test_evidence": dict(test_evidence),
        "approvers": dict(approvers),
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }


def sign_certificate(cert: Dict[str, object], private_key: bytes) -> Dict[str, object]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    signed = dict(cert)
    signed["signature"] = base64.b64encode(
        Ed25519PrivateKey.from_private_bytes(private_key).sign(_canonical(cert))
    ).decode()
    return signed


def verify_certificate(cert: Mapping[str, object], trusted_public_key: bytes) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    sig = cert.get("signature")
    if not isinstance(sig, str):
        raise CertificationError("certificate is unsigned")
    try:
        Ed25519PublicKey.from_public_bytes(trusted_public_key).verify(base64.b64decode(sig), _canonical(cert))
    except (InvalidSignature, ValueError) as exc:
        raise CertificationError("certificate signature does not verify") from exc
    approvers = cert.get("approvers") or {}
    if not isinstance(approvers, dict) or not approvers:
        raise CertificationError("certificate names no approvers")
    pending = [
        role
        for role, who in approvers.items()
        if str(who).strip().upper() in PENDING_MARKERS or str(who).strip().upper().startswith("PENDING")
    ]
    if pending:
        raise CertificationError(f"approvals not recorded for: {pending}")
    if not cert.get("test_evidence"):
        raise CertificationError("certificate carries no test evidence")


def assert_deposit_allowed(paths: Iterable[Path], cert: Mapping[str, object], trusted_public_key: bytes) -> List[str]:
    """Raise unless every path is a certified artifact; returns the verified names."""
    verify_certificate(cert, trusted_public_key)
    certified: Mapping[str, str] = cert.get("artifacts") or {}  # type: ignore[assignment]
    checked = []
    for path in paths:
        p = Path(path)
        expected = certified.get(p.name)
        if expected is None:
            raise CertificationError(f"{p.name} is not in the certified release set")
        if _sha256(p) != expected:
            raise CertificationError(f"{p.name} differs from the certified artifact")
        checked.append(p.name)
    if not checked:
        raise CertificationError("nothing to deposit")
    return checked
