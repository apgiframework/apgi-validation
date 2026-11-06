#!/usr/bin/env python3
"""Release-bound build record for one candidate (APGI-IMP-001, APGI-REL-001).

    python scripts/build_record.py [--image-digest sha256:...] [--out build/build-record.json]

Writes a JSON record binding the candidate to its controlled inputs:
source commit and tree state, requirements.lock digest, protocol-lock digests,
the Dockerfile's pinned base-image digest, the release/ package checksum
digest, the CycloneDX SBOM (from ``utils.dependency_scanner``) and the
interpreter/platform. CI uploads it as an artifact for every main build; a
release certificate (``utils/release_certificate.py``) cites its digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess  # nosec B404 - fixed git invocations
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(*args: str) -> str:
    try:
        return subprocess.run(  # nosec B603 B607
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _base_image_digest() -> Optional[str]:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.is_file():
        return None
    m = re.search(r"^FROM\s+\S+@(sha256:[0-9a-f]{64})", dockerfile.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else None


def _lock_components() -> list:
    comps = []
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if m:
            name, version = m.groups()
            comps.append(
                {"type": "library", "name": name, "version": version, "purl": f"pkg:pypi/{name.lower()}@{version}"}
            )
    return comps


def build_record(image_digest: Optional[str] = None) -> Dict[str, Any]:
    from utils.dependency_scanner import DependencyScanner
    from utils.versioning import APP_VERSION, PROTOCOL_BUNDLE_VERSION

    sbom = DependencyScanner(str(ROOT)).generate_signed_sbom()
    if not sbom.get("components"):
        # The scanner's SBOM can be empty; the lockfile is the authoritative
        # list of what a locked install resolves to, so use its exact pins.
        sbom["components"] = _lock_components()
    sbom_bytes = json.dumps(sbom, sort_keys=True).encode()
    return {
        "record_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "application_version": APP_VERSION,
        "protocol_bundle_version": PROTOCOL_BUNDLE_VERSION,
        "source": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "inputs": {
            "requirements_lock_sha256": _sha256(ROOT / "requirements.lock"),
            "protocol_manifest_sha256": _sha256(ROOT / "protocols" / "MANIFEST.json"),
            "protocol_trust_anchor_sha256": _sha256(ROOT / "config" / "protocol_lock.json"),
            "module_manifest_sha256": _sha256(ROOT / "config" / "protocol_manifest.json"),
            "release_checksums_sha256": _sha256(ROOT / "release" / "checksums.sha256"),
            "dockerfile_sha256": _sha256(ROOT / "Dockerfile"),
            "base_image_digest": _base_image_digest(),
        },
        "outputs": {"image_digest": image_digest},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "sbom_sha256": hashlib.sha256(sbom_bytes).hexdigest(),
        "sbom": sbom,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-digest", help="digest of the built container image, if any")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "build-record.json")
    args = parser.parse_args(argv)
    record = build_record(args.image_digest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"build record written: {args.out} (commit {record['source']['commit'][:12]}, dirty={record['source']['dirty']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
