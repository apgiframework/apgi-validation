# Changelog

All notable changes to `apgi-validation` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
