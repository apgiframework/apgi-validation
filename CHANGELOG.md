# Changelog

All notable changes to `apgi-validation` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — APGI v1.2 validation-specification remediation

Implements the code-actionable items of the v1.2 software-validation registers.
Full item-by-item status: `docs/validation/traceability.md`. Items needing data,
approvals or other components remain **blocked** and are not reported as done.

### Breaking (scientific)
- `CoreIgnitionSystem.accumulated_signal` now computes the specified
  Sₜ = Πᵉ|zᵉ| + Πⁱ_eff|zⁱ| (APGI-SCI-001); the old quadratic form is
  `quadratic_surprise_signal` (non-canonical). Ignition is strictly `S > θ`.
- `HierarchicalProcessor` uses the canonical L0–L4 hierarchy (APGI-SCI-003);
  level names are now `L0`…`L4`; `num_levels` default 5.
- Power/sample-size helpers raise `UnsupportedMethodError` /
  `MissingScientificDependencyError` instead of returning 0.8 / 50 (APGI-REP-003).
- Hurst estimation raises `ScientificComputationError` instead of returning 0.5 (APGI-INT-003).
- `ProtocolResult` cannot report `success` when incomplete (APGI-API-004) and
  carries schema/protocol/producer versions and dataset provenance (APGI-INT-001, DAT-004).

### Breaking (security/operations)
- Authentication is enforced unless `APGI_ENFORCE_AUTH=0` is set explicitly;
  `--headless` uses the same gate; dev opt-outs and ephemeral master keys are
  refused when `APGI_ENV=production` (APGI-IMP-002). No implicit admin override (API-002).
- Protocol definitions load only after the registered lock verifies
  (`config/protocol_lock.json` trust anchor, per-file SHA-256); any failure aborts
  the whole load (APGI-REP-001, PRO-002).
- `data_repository/empirical_data/` renamed `synthetic_data/`; labels are `synthetic` (S4-VAL-13.5).

### Added
- Record provenance labels, research-store guard (`utils/data_provenance.py`).
- K1–K8 catalog and interval/band/sufficiency classifier (`utils/kgate.py`);
  registered estimators (`utils/psychometrics.py`); signed validation reports
  (`utils/validation_report.py`, `config/schemas/validation_report.schema.json`).
- Consent gate and lineage-wide withdrawal (`utils/consent_gate.py`);
  quarantine/reconciliation (`utils/data_quarantine.py`).
- Release certificate and certificate-before-deposit gate (`utils/release_certificate.py`).
- `release/` reproducibility package (12 artifacts) with builder/verifier.
- `CITATION.cff`, terminology scan, content/history secret scan.
- Validation governance documents under `docs/validation/`.
- CI: `validation-gates.yml` (registered rules, synthetic fixtures, result classification).

### Fixed
- CI-NOLDS (nolds 0.6.3 import failure on Python 3.11): pin nolds 0.6.2 + setuptools < 81.
- CI-BANDIT: severity/confidence policy moved to supported CLI flags.
- CI-ISORT: import order. CI-MEMORY: per-test figure/GC release.
- NumPy 1.26 compatibility of `metabolic_cost` (APGI-REP-005); seeded RNG streams (REP-002).
- Docker builds from `requirements.lock` on a digest-pinned base (APGI-IMP-001).
- Six non-canonical APGI expansions corrected (APGI-SCI-006).

---

## [Unreleased] — APGI protocol specification 1.1.0

Brings this repository into line with protocol specification release 1.1.0,
as already adopted by `apgi-framework`. The package version (`1.3.0`) is
independent of the specification version.

### Protocol specification

- `protocols/` now holds the 1.1.0 release, byte-identical to apgi-framework:
  all eight `protocol_*.json` files, `schemas/protocol.schema.json` (Draft 2020-12,
  `x-version` 1.1.0), `CHANGELOG.md`, `VALIDATION.json` and `MANIFEST.json`. The
  previous files mixed 1.1.0 and 2.1.0 version strings from the pre-release lineage.
- `scripts/validate_protocols.py` and `tests/test_validate_protocols.py` (ported
  from apgi-framework) check schema conformance, cross-field arithmetic, exact
  Protocol 0–7 membership, null κ while uncalibrated, pending-approval rules and
  MANIFEST checksums. Run with `make protocols-check`; CI runs it before the test
  suite.
- `protocols/apgi_protocol_template.json` → `config/apgi_protocol_template.json`.
  The template predates 1.1.0, and the release folder may only contain files
  covered by `MANIFEST.json`.

### Data contract

- `utils/protocol_loader.py` reads the 1.1.0 fields: `study_status` (replaces
  `validation_status`), `submission_blocking_issues` (replaces `caveats`), and
  per-prediction `status`, `indeterminate_rule`, `confirming_thresholds` and
  `falsifying_thresholds`. New: `SPECIFICATION_VERSION`, `Verdict`,
  `classify_prediction()` and `spec_prediction_results()`.
- `utils/protocol_schema.py` adds `PredictionStatus.INDETERMINATE`.
- Registered predictions are reported with the 1.1.0 three-way rule: a failure to
  confirm is not falsification. `VP_09`, `VP_20`, `VP_22`, `FP_04` and `FP_09`
  previously matched legacy keys (`"P1a"`, `"P6a"`, …) that never occur in the
  JSON (`"Pred 1.A"`, …), so every registered prediction was reported FAILED. They
  now link only simulation checks that implement the registered test
  (`Pred 6.A` ← V20.1, `Pred 6.C` ← V20.2, `Pred 3.C` ← P5c); the rest are INDETERMINATE.

### Protocol numbering

- `utils/protocol_registry.py` registers APGI-P00–P07 against the real 1.1.0
  filenames (it pointed at six files that no longer exist) and corrects
  cross-links that used the pre-1.1 numbering: TMS → P05, agent simulations and
  somatic-marker identifiability → P02, fMRI anticipation → P03, DoC → P07.
- `VP_22_FMRIAnticipationExperience` loads APGI-P03 (fMRI anticipation) rather than
  APGI-P05 (causal TMS). Its internal `P5a`–`P5d` keys are unchanged.
- `Validation_GUI.py` extended-protocol labels and
  `Theory/APGI_Somatic_Marker_Identifiability.py` follow the same numbering.

### Naming

- Project name unified as *Allostatic Precision-Gated Ignition* in `pyproject.toml`,
  `main.py`, `README.md` and `docs/Validation.md` (previously "Adaptive Pattern
  Generation and Integration" and "Active Perception and Global Ignition").

### Not changed

- Model defaults in `config/*.yaml` and `utils/threshold_registry.py` are not
  re-derived from 1.1.0 here. Several sit outside the registered ranges
  (e.g. `theta0: 0.03` vs θ₀ band 0.25–0.85; sigmoid `alpha: 8.0`/`10.0` vs
  `gamma_sig` 2–7.5; `tau_S: 1.0` vs 0.2–0.5 s) and need a deliberate review.
