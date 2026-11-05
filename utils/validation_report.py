"""Validation report contract and signing (S4-VAL-11.1, S4-VAL-09, APGI-VAL-001).

A validation report binds one K-gate classification to everything needed to
reproduce and audit it: protocol lock digest, gate-catalog version, input
dataset provenance (origin/stage/category/version/checksum/licence), the
estimates, software/schema versions, seed and environment.

The field list ``validation_id … created_at`` follows the register. The
authoritative contract is still to be confirmed by the Scientific Reviewer
(S4-VAL-11.1); :data:`REPORT_SCHEMA_VERSION` must be bumped on any change.

Signing uses Ed25519 over the canonical JSON bytes of the report with the
``signature`` block removed. :func:`verify_report` recomputes those bytes,
so any edit of any field — including whitespace-insensitive value changes —
invalidates the signature.
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import jsonschema

from utils.data_provenance import ProvenanceRecord
from utils.kgate import CATALOG_VERSION, GateClassification

REPORT_SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "schemas" / "validation_report.schema.json"


class ReportContractError(ValueError):
    """A validation report violates the report contract."""


class SignatureError(ValueError):
    """A report signature is missing or does not verify."""


def canonical_bytes(report: Dict[str, Any]) -> bytes:
    body = {k: v for k, v in report.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_report(report: Dict[str, Any]) -> None:
    """Raise :class:`ReportContractError` for any contract violation."""
    validator = jsonschema.Draft202012Validator(_schema(), format_checker=jsonschema.FormatChecker())
    errors = sorted(validator.iter_errors(report), key=lambda e: list(e.absolute_path))
    if errors:
        raise ReportContractError("; ".join(f"{list(e.absolute_path)}: {e.message}" for e in errors[:10]))
    # Cross-field rules JSON Schema cannot express cleanly.
    cls = report["classification"]
    if cls["processing_status"] == "NOT_EVALUATED" and cls["decision"] is not None:
        raise ReportContractError("NOT_EVALUATED reports must not carry a decision")
    if cls["processing_status"] == "EVALUATED" and cls["decision"] is None:
        raise ReportContractError("EVALUATED reports must carry a decision")
    if cls["decision"] in ("PASS", "FAIL"):
        bad = [
            d["record_id"]
            for d in report["input_datasets"]
            if d["data_origin"] not in ("human_observed", "imported_empirical")
        ]
        if bad:
            raise ReportContractError(f"non-empirical inputs cannot support a PASS/FAIL decision: {bad}")
    if cls["decision"] == "INDETERMINATE" and cls["next_action"] != "further_sample":
        raise ReportContractError("INDETERMINATE decisions require next_action=further_sample")


def environment_fingerprint() -> Dict[str, Any]:
    """Versions and numerical settings that affect reproducibility (APGI-REP-002)."""
    import os

    import numpy as np

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        info["scipy"] = None
    try:
        cfg = np.show_config(mode="dicts")  # NumPy >= 1.25
        info["blas"] = (cfg or {}).get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:  # pragma: no cover - informational only
        info["blas"] = None
    return info


def _dataset_entry(rec: ProvenanceRecord) -> Dict[str, Any]:
    return {
        "record_id": rec.record_id,
        "data_origin": rec.data_origin.value,
        "processing_stage": rec.processing_stage.value,
        "inferential_category": rec.inferential_category.value,
        "source": rec.source,
        "version": rec.version,
        "sha256": rec.sha256,
        "license": rec.license,
        "parents": list(rec.parents),
    }


def build_report(
    classification: GateClassification,
    *,
    inputs: Iterable[ProvenanceRecord],
    protocol_lock_sha256: str,
    software_version: str,
    source_commit: str,
    seed: Optional[int],
    n: Optional[int] = None,
    validation_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    report = {
        "validation_id": validation_id or str(uuid.uuid4()),
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "gate_id": classification.gate_id,
        "gate_catalog_version": CATALOG_VERSION,
        "protocol_lock_sha256": protocol_lock_sha256,
        "input_datasets": [_dataset_entry(r) for r in inputs],
        "n": n,
        "classification": classification.to_dict(),
        "software": {"name": "apgi-validation", "version": software_version, "source_commit": source_commit},
        "seed": seed,
        "environment": environment_fingerprint(),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    validate_report(report)
    return report


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def generate_signing_key() -> bytes:
    """Return a new raw 32-byte Ed25519 private key (store it as a secret)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    return Ed25519PrivateKey.generate().private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def public_key_for(private_key: bytes) -> bytes:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return Ed25519PrivateKey.from_private_bytes(private_key).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def sign_report(report: Dict[str, Any], private_key: bytes, signer: str) -> Dict[str, Any]:
    """Validate, then attach an Ed25519 signature block to a copy of ``report``."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    validate_report(report)
    payload = canonical_bytes(report)
    key = Ed25519PrivateKey.from_private_bytes(private_key)
    signed = dict(report)
    signed["signature"] = {
        "algorithm": "Ed25519",
        "signer": signer,
        "public_key": base64.b64encode(public_key_for(private_key)).decode(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "value": base64.b64encode(key.sign(payload)).decode(),
    }
    return signed


def verify_report(report: Dict[str, Any], trusted_public_key: bytes) -> None:
    """Raise :class:`SignatureError` unless ``report`` is signed by ``trusted_public_key``."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    sig = report.get("signature")
    if not isinstance(sig, dict) or sig.get("algorithm") != "Ed25519":
        raise SignatureError("report is unsigned or uses an unsupported algorithm")
    if base64.b64decode(sig.get("public_key", "")) != trusted_public_key:
        raise SignatureError("report signed by an untrusted key")
    payload = canonical_bytes(report)
    if hashlib.sha256(payload).hexdigest() != sig.get("payload_sha256"):
        raise SignatureError("report bytes differ from the signed payload")
    try:
        Ed25519PublicKey.from_public_bytes(trusted_public_key).verify(base64.b64decode(sig["value"]), payload)
    except (InvalidSignature, ValueError, KeyError) as exc:
        raise SignatureError("signature does not verify") from exc
    validate_report(report)
