"""Registered protocol lock verification (APGI-REP-001, APGI-PRO-002, APGI-VAL-001).

Every protocol definition in ``protocols/`` is covered by ``protocols/MANIFEST.json``,
which lists the SHA-256 digest of each released file. A manifest that lives
next to the files it protects cannot detect a coordinated edit of both, so the
digest of the approved manifest itself is pinned separately in
``config/protocol_lock.json`` (the *trust anchor*).

Verification is all-or-nothing:

* the trust anchor must exist and name the manifest digest;
* ``MANIFEST.json`` must hash to that digest and carry the expected release;
* every listed file must exist and match its digest; and
* every protocol definition on disk must be listed.

Any failure raises :class:`ProtocolLockError` before a definition is parsed,
so callers produce no output and keep no partial state. Each verification
attempt is appended to :data:`LOCK_AUDIT_LOG` and logged.

``APGI_SKIP_PROTOCOL_LOCK=1`` disables the check for local development only;
it is refused when ``APGI_ENV`` is ``production``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOLS_DIR = PROJECT_ROOT / "protocols"
DEFAULT_TRUST_ANCHOR = PROJECT_ROOT / "config" / "protocol_lock.json"

SKIP_ENV = "APGI_SKIP_PROTOCOL_LOCK"


class ProtocolLoadError(Exception):
    """Base class for a protocol definition that must not be used."""


class ProtocolLockError(ProtocolLoadError):
    """The protocol lock (trust anchor, manifest or file digest) failed."""


class ProtocolSchemaError(ProtocolLoadError):
    """A protocol definition is not a schema-valid registered definition."""


@dataclass
class LockAuditRecord:
    """One verification attempt, retained as tamper evidence."""

    timestamp: str
    protocols_dir: str
    outcome: str  # "verified" | "rejected" | "skipped_dev"
    manifest_sha256: Optional[str]
    release: Optional[str]
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


LOCK_AUDIT_LOG: List[LockAuditRecord] = []
_AUDIT_LOCK = threading.Lock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_production() -> bool:
    return os.environ.get("APGI_ENV", "").strip().lower() in {"prod", "production"}


def _record(record: LockAuditRecord) -> None:
    with _AUDIT_LOCK:
        LOCK_AUDIT_LOG.append(record)
    level = logging.INFO if record.outcome == "verified" else logging.ERROR
    logger.log(level, "protocol_lock_audit %s", json.dumps(record.to_dict(), sort_keys=True))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class VerifiedLock:
    """Result of a successful verification: file name -> approved digest."""

    release: str
    manifest_sha256: str
    digests: Dict[str, str]

    def is_locked_definition(self, path: Path) -> bool:
        expected = self.digests.get(path.name)
        return expected is not None and sha256_file(path) == expected


def _load_anchor(anchor_path: Path) -> dict:
    try:
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProtocolLockError(f"trust anchor unreadable: {anchor_path}: {exc}") from exc
    if not isinstance(anchor, dict) or not isinstance(anchor.get("manifest_sha256"), str):
        raise ProtocolLockError(f"trust anchor malformed: {anchor_path}")
    return anchor


def verify_protocol_lock(
    protocols_dir: Optional[Path] = None,
    trust_anchor: Optional[Path] = None,
) -> VerifiedLock:
    """Verify the full protocol lock or raise :class:`ProtocolLockError`."""
    folder = Path(protocols_dir) if protocols_dir is not None else DEFAULT_PROTOCOLS_DIR
    anchor_path = Path(trust_anchor) if trust_anchor is not None else DEFAULT_TRUST_ANCHOR

    if os.environ.get("APGI_SKIP_PROTOCOL_LOCK", "0") == "1":
        if is_production():
            err = f"{SKIP_ENV}=1 is not permitted when APGI_ENV=production"
            _record(LockAuditRecord(_now(), str(folder), "rejected", None, None, [err]))
            raise ProtocolLockError(err)
        logger.warning("Protocol lock verification skipped (%s=1, development only)", SKIP_ENV)
        _record(LockAuditRecord(_now(), str(folder), "skipped_dev", None, None, []))
        digests = {p.name: sha256_file(p) for p in folder.glob("protocol_*.json")}
        return VerifiedLock(release="UNVERIFIED-DEV", manifest_sha256="", digests=digests)

    errors: List[str] = []
    manifest_digest: Optional[str] = None
    release: Optional[str] = None
    digests: Dict[str, str] = {}
    try:
        anchor = _load_anchor(anchor_path)
        manifest_path = folder / "MANIFEST.json"
        if not manifest_path.is_file():
            raise ProtocolLockError(f"MANIFEST.json missing in {folder}")
        manifest_digest = sha256_file(manifest_path)
        if manifest_digest != anchor["manifest_sha256"]:
            errors.append("MANIFEST.json digest differs from the trust anchor (manifest was modified)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        release = manifest.get("release")
        if release != anchor.get("release"):
            errors.append(f"manifest release {release!r} differs from anchored release {anchor.get('release')!r}")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            errors.append("MANIFEST.json files must be an array")
            entries = []
        for entry in entries:
            name = entry.get("path") if isinstance(entry, dict) else None
            if not isinstance(name, str) or ".." in Path(name).parts or Path(name).is_absolute():
                errors.append(f"invalid manifest entry: {entry!r}")
                continue
            path = folder / name
            if not path.is_file():
                errors.append(f"locked file missing: {name}")
                continue
            actual = sha256_file(path)
            if actual != entry.get("sha256"):
                errors.append(f"SHA-256 mismatch: {name}")
            digests[Path(name).name] = entry.get("sha256")
        for path in sorted(folder.glob("protocol_*.json")):
            if path.name not in digests:
                errors.append(f"unlocked protocol definition present: {path.name}")
    except ProtocolLockError as exc:
        errors.append(str(exc))
    except (OSError, ValueError) as exc:
        errors.append(f"lock verification failed: {exc}")

    if errors:
        _record(LockAuditRecord(_now(), str(folder), "rejected", manifest_digest, release, errors))
        raise ProtocolLockError("; ".join(errors))

    _record(LockAuditRecord(_now(), str(folder), "verified", manifest_digest, release, []))
    return VerifiedLock(release=str(release), manifest_sha256=str(manifest_digest), digests=digests)


def write_trust_anchor(
    protocols_dir: Optional[Path] = None,
    trust_anchor: Optional[Path] = None,
    *,
    approved_by: str = "PENDING-OWNER-APPROVAL",
) -> dict:
    """Pin the current MANIFEST.json digest (release-manager operation)."""
    folder = Path(protocols_dir) if protocols_dir is not None else DEFAULT_PROTOCOLS_DIR
    anchor_path = Path(trust_anchor) if trust_anchor is not None else DEFAULT_TRUST_ANCHOR
    manifest_path = folder / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchor = {
        "release": manifest.get("release"),
        "manifest_path": "protocols/MANIFEST.json",
        "manifest_sha256": sha256_file(manifest_path),
        "approved_by": approved_by,
        "opentimestamps_proof": None,
        "opentimestamps_status": "PENDING: proof reference to be supplied by the specification authority (AUD-SPEC-01)",
        "protocol_v1_2_equivalence": "UNRESOLVED: see AUD-SPEC-01 / APGI-PRO-001",
    }
    anchor_path.write_text(json.dumps(anchor, indent=2) + "\n", encoding="utf-8")
    return anchor


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Verify or pin the registered protocol lock")
    parser.add_argument("--pin", action="store_true", help="Rewrite config/protocol_lock.json from MANIFEST.json")
    args = parser.parse_args()
    if args.pin:
        print(json.dumps(write_trust_anchor(), indent=2))
    else:
        lock = verify_protocol_lock()
        print(f"Protocol lock verified: release {lock.release}, {len(lock.digests)} files")
