#!/usr/bin/env python3
"""Build or verify the release/ reproducibility package (S4-VAL-16, S4-VAL-16.1–16.12).

    python scripts/build_release_package.py          # regenerate release/
    python scripts/build_release_package.py --check  # verify release/ (CI gate)

Every artifact is generated from controlled sources in this repository, so the
package can be rebuilt and diffed. ``--check`` verifies:

* all twelve required artifacts exist and are non-empty;
* ``checksums.sha256`` matches every file in the package;
* the protocol manifest equals the verified protocol lock;
* the analysis manifest's file digests equal the current analysis code;
* the gate table equals the in-code K-gate catalog;
* no result table reports a K-gate decision that the package cannot back
  with signed evidence (until such evidence exists, every gate is
  NOT_EVALUATED).

Result artifacts are populated from validated execution only. At the time
of writing no frozen empirical dataset exists (S4-REC-01), so results report
NOT_EVALUATED for every gate — that is the honest content, not a placeholder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess  # nosec B404 - fixed git invocation
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from utils.kgate import CATALOG_VERSION, K_GATES, catalog  # noqa: E402
from utils.protocol_lock import DEFAULT_TRUST_ANCHOR, verify_protocol_lock  # noqa: E402
from utils.validation_report import REPORT_SCHEMA_VERSION  # noqa: E402
from utils.versioning import (  # noqa: E402
    APP_VERSION,
    LOCAL_CONTRACT_VERSION,
    PROTOCOL_BUNDLE_VERSION,
    PROTOCOL_RESULT_SCHEMA_VERSION,
)

RELEASE = ROOT / "release"
REQUIRED = [
    "README.md",
    "CITATION.cff",
    "protocol-manifest.json",
    "analysis-manifest.json",
    "data-dictionary.json",
    "environment.lock",
    "source-commits.json",
    "exclusions.json",
    "gate-table.yaml",
    "results/tables/k-gate-status.csv",
    "results/figures/k-gate-status.svg",
    "checksums.sha256",
]
ANALYSIS_FILES = [
    "utils/kgate.py",
    "utils/psychometrics.py",
    "utils/validation_report.py",
    "utils/data_provenance.py",
    "utils/protocol_lock.py",
    "utils/protocol_loader.py",
    "utils/deviation_register.py",
    "utils/acquisition_checks.py",
    "utils/statistical_tests.py",
    "apgi_core/equations.py",
    "apgi_core/engine.py",
    "apgi_core/temporal_hierarchy.py",
    "config/schemas/validation_report.schema.json",
]
GATE_ESTIMATORS = {
    "K1": "utils.psychometrics.icc_2_1",
    "K2": "utils.psychometrics.icc_2_1",
    "K3": "utils.psychometrics.ordinal_omega",
    "K4": "utils.psychometrics.k4_estimates",
    "K4b": "utils.psychometrics.invariance_level",
    "K5": "utils.psychometrics.pearson_with_ci",
    "K6": "utils.psychometrics.delta_r2_holdout",
    "K6b": "utils.psychometrics.mae_relative_change + top_quintile_precision",
    "K7b": "utils.psychometrics.profile_likelihood_bounded",
    "K8": "utils.psychometrics.credible_interval",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _git_head() -> str:
    try:
        return subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def readme() -> str:
    return f"""# APGI validation — reproducibility package

Application version {APP_VERSION} · protocol bundle {PROTOCOL_BUNDLE_VERSION} ·
K-gate catalog {CATALOG_VERSION} · report schema {REPORT_SCHEMA_VERSION}

**Validation status: no K-gate has been evaluated.** No frozen, consented
empirical dataset exists yet (prerequisites S4-REC-01, S4-REC-02), so every
gate in `results/tables/k-gate-status.csv` is `NOT_EVALUATED`. Synthetic data
shipped in this repository is for method validation only and is never counted
as empirical confirmation.

| File | Content |
|---|---|
| `protocol-manifest.json` | SHA-256 lock of the registered protocol definitions and its trust anchor |
| `analysis-manifest.json` | digests of the analysis code and the estimator used for each gate |
| `data-dictionary.json` | provenance labels and report fields |
| `environment.lock` | exact dependency pins |
| `source-commits.json` | commit this package was generated from; upstream components |
| `exclusions.json` | registered participant/data exclusions |
| `gate-table.yaml` | K1–K8 registered criteria (pending Scientific Reviewer approval) |
| `results/` | gate status table and figure |
| `checksums.sha256` | SHA-256 of every other file |

Rebuild: `python scripts/build_release_package.py`; verify:
`python scripts/build_release_package.py --check`.

Approval of this package for archival deposit requires the release
certificate described in `docs/validation/release-procedure.md` (APGI-REL-005).
"""


def citation() -> str:
    return (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def protocol_manifest() -> str:
    lock = verify_protocol_lock()
    anchor = json.loads(DEFAULT_TRUST_ANCHOR.read_text(encoding="utf-8"))
    return _json(
        {
            "release": lock.release,
            "manifest_sha256": lock.manifest_sha256,
            "trust_anchor": anchor,
            "files": dict(sorted(lock.digests.items())),
        }
    )


def analysis_manifest() -> str:
    return _json(
        {
            "catalog_version": CATALOG_VERSION,
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "files": {p: sha256_file(ROOT / p) for p in ANALYSIS_FILES},
            "gate_estimators": GATE_ESTIMATORS,
            "classifier": "utils.kgate.classify_gate",
        }
    )


def data_dictionary() -> str:
    from utils.data_provenance import DataOrigin, InferentialCategory, ProcessingStage, ProvenanceRecord

    return _json(
        {
            "provenance_record": {
                name: {"required": field.is_required(), "annotation": str(field.annotation)}
                for name, field in ProvenanceRecord.model_fields.items()
            },
            "enums": {
                "data_origin": [o.value for o in DataOrigin],
                "processing_stage": [s.value for s in ProcessingStage],
                "inferential_category": [c.value for c in InferentialCategory],
            },
            "validation_report_schema": "config/schemas/validation_report.schema.json",
            "protocol_result_schema_version": PROTOCOL_RESULT_SCHEMA_VERSION,
            "local_contract_version": LOCAL_CONTRACT_VERSION,
        }
    )


def environment_lock() -> str:
    header = "# Exact pins for this release; generated from requirements.lock (Python 3.11).\n"
    return header + (ROOT / "requirements.lock").read_text(encoding="utf-8")


def source_commits() -> str:
    return _json(
        {
            "apgi-validation": {"generated_from_commit": _git_head()},
            "upstream_components": {
                "apgi-framework": "PENDING: pinned revision not supplied (AUD-SPEC-04)",
                "apgi-api": "PENDING: canonical schema revision not approved (AUD-SPEC-06)",
            },
        }
    )


def exclusions() -> str:
    return _json(
        {
            "status": "no empirical data processed",
            "participant_exclusions": [],
            "record_exclusions": [],
            "note": "Populate only from the registered exclusion rules applied to frozen datasets.",
        }
    )


def gate_table() -> str:
    return yaml.safe_dump(catalog(), sort_keys=False, allow_unicode=True)


def status_rows() -> List[Dict[str, str]]:
    return [
        {
            "gate_id": g.gate_id,
            "title": g.title,
            "processing_status": "NOT_EVALUATED",
            "decision": "",
            "next_action": "collect_evidence",
            "reason": "no frozen empirical dataset (S4-REC-01)",
        }
        for g in K_GATES.values()
    ]


def status_table() -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(status_rows()[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(status_rows())
    return buf.getvalue()


def status_figure() -> str:
    rows = status_rows()
    height = 40 + 26 * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="560" height="{height}" font-family="sans-serif" font-size="13">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="12" y="22" font-weight="bold">K-gate processing status</text>',
    ]
    for i, row in enumerate(rows):
        y = 44 + 26 * i
        parts.append(f'<rect x="12" y="{y}" width="18" height="18" fill="#9aa0a6"/>')
        parts.append(f'<text x="38" y="{y + 14}">{row["gate_id"]}: {row["processing_status"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


GENERATORS = {
    "README.md": readme,
    "CITATION.cff": citation,
    "protocol-manifest.json": protocol_manifest,
    "analysis-manifest.json": analysis_manifest,
    "data-dictionary.json": data_dictionary,
    "environment.lock": environment_lock,
    "source-commits.json": source_commits,
    "exclusions.json": exclusions,
    "gate-table.yaml": gate_table,
    "results/tables/k-gate-status.csv": status_table,
    "results/figures/k-gate-status.svg": status_figure,
}


def checksums(folder: Path) -> str:
    lines = [
        f"{sha256_file(p)}  {p.relative_to(folder).as_posix()}"
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "checksums.sha256"
    ]
    return "\n".join(lines) + "\n"


def build(folder: Path = RELEASE) -> None:
    for name, gen in GENERATORS.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(gen(), encoding="utf-8")
    (folder / "checksums.sha256").write_text(checksums(folder), encoding="utf-8")


def check(folder: Path = RELEASE) -> List[str]:
    errors: List[str] = []
    for name in REQUIRED:
        p = folder / name
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f"missing or empty: release/{name}")
    if errors:
        return errors
    if (folder / "checksums.sha256").read_text(encoding="utf-8") != checksums(folder):
        errors.append("checksums.sha256 does not match package contents")
    # Content that must equal the controlled sources (source-commits is generation metadata).
    for name in (
        "protocol-manifest.json",
        "analysis-manifest.json",
        "gate-table.yaml",
        "data-dictionary.json",
        "environment.lock",
        "CITATION.cff",
        "results/tables/k-gate-status.csv",
    ):
        if (folder / name).read_text(encoding="utf-8") != GENERATORS[name]():
            errors.append(f"release/{name} is stale relative to controlled sources; rebuild the package")
    return errors


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        errors = check()
        for e in errors:
            print(f"FAIL {e}", file=sys.stderr)
        if not errors:
            print(f"release/ package verified ({len(REQUIRED)} artifacts).")
        return 1 if errors else 0
    build()
    print(f"release/ package written ({len(REQUIRED)} artifacts).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
