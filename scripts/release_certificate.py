#!/usr/bin/env python3
"""Create, sign and check release certificates (APGI-REL-001, APGI-REL-005).

    # 1. draft (unsigned) from artifacts + build record
    python scripts/release_certificate.py draft --version 2.0.0 \
        --artifact dist/apgi_validation-2.0.0.tar.gz --build-record build/build-record.json \
        --evidence ci_run=https://github.com/... \
        --approver release_authority="Name" --approver scientific_reviewer="Name" \
        --out build/certificate.json
    # 2. sign (release authority; key file holds the raw 32-byte Ed25519 key)
    python scripts/release_certificate.py sign build/certificate.json --key-file /secure/release.key
    # 3. gate before any archival deposit
    python scripts/release_certificate.py check build/certificate.json --public-key-file release.pub \
        dist/apgi_validation-2.0.0.tar.gz

``check`` exits non-zero unless every artifact is certified, unmodified, and
every approver is named (no PENDING). Deposit tooling must run it first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.protocol_lock import verify_protocol_lock  # noqa: E402
from utils.release_certificate import (  # noqa: E402
    CertificationError,
    assert_deposit_allowed,
    create_certificate,
    sign_certificate,
)


def _pairs(values):
    out = {}
    for v in values or []:
        k, _, val = v.partition("=")
        if not k or not val:
            raise SystemExit(f"expected key=value, got {v!r}")
        out[k] = val
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--version", required=True)
    d.add_argument("--artifact", action="append", required=True, type=Path)
    d.add_argument("--build-record", type=Path)
    d.add_argument("--evidence", action="append", help="name=url")
    d.add_argument("--approver", action="append", help="role=Named Person")
    d.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sign")
    s.add_argument("certificate", type=Path)
    s.add_argument("--key-file", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("certificate", type=Path)
    c.add_argument("--public-key-file", type=Path, required=True)
    c.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "draft":
        record = json.loads(args.build_record.read_text()) if args.build_record else {}
        commit = record.get("source", {}).get("commit", "UNKNOWN")
        lock = verify_protocol_lock()
        evidence = _pairs(args.evidence)
        if args.build_record:
            import hashlib

            evidence["build_record_sha256"] = hashlib.sha256(args.build_record.read_bytes()).hexdigest()
        cert = create_certificate(
            version=args.version,
            source_commit=commit,
            artifacts=args.artifact,
            protocol_lock_sha256=lock.manifest_sha256,
            test_evidence=evidence,
            approvers=_pairs(args.approver) or {"release_authority": "PENDING", "scientific_reviewer": "PENDING"},
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(cert, indent=2) + "\n")
        print(f"draft certificate written: {args.out}")
        return 0
    if args.cmd == "sign":
        cert = json.loads(args.certificate.read_text())
        signed = sign_certificate(cert, args.key_file.read_bytes())
        args.certificate.write_text(json.dumps(signed, indent=2) + "\n")
        print(f"signed: {args.certificate}")
        return 0
    cert = json.loads(args.certificate.read_text())
    try:
        names = assert_deposit_allowed(args.artifacts, cert, args.public_key_file.read_bytes())
    except CertificationError as exc:
        print(f"DEPOSIT REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"deposit allowed for: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
