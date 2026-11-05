# APGI validation — reproducibility package

Application version 1.3.0 · protocol bundle 1.1.0 ·
K-gate catalog 0.1.0 · report schema 1.0.0

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
