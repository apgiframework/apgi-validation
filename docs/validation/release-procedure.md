# Release, compatibility and change control (APGI-REL-001 … REL-005, APGI-IMP-001/003)

**Status:** procedure defined; **no release candidate has been certified**. Named
approvers are unappointed (AUD-GOV-01), so the certificate gate below cannot pass yet —
by design.

## 1. Version compatibility matrix (APGI-REL-004)

| Domain | Version | Source of truth | Tested relationship |
|---|---|---|---|
| Application | 1.3.0 | `pyproject.toml`, `utils/versioning.py` | equality tested (`test_release_and_terminology.py`) |
| Protocol bundle | 1.1.0 | `protocols/MANIFEST.json`, `config/protocol_lock.json` | app 1.3.0 loads only bundle 1.1.0 (lock + schema tests) |
| Governing protocol | v1.2 | specification (draft) | **UNRESOLVED** — no approved 1.1.0→1.2 mapping (AUD-SPEC-01) |
| Local inter-protocol contract | 1.0 | `utils/protocol_contracts.py` | `tests/test_protocol_contracts.py` |
| ProtocolResult exchange schema | 2.0.0 | `utils/protocol_schema.py` | `tests/test_provenance_and_reports.py` |
| Validation report schema | 1.0.0 | `config/schemas/validation_report.schema.json` | contract + signature tests |
| K-gate catalog | 0.1.0 | `utils/kgate.py` → `release/gate-table.yaml` | `--check` equality |
| Canonical apgi-api schemas | — | not supplied (AUD-SPEC-06) | **none** |

Versioning rule applied: a change that alters a registered decision, a required field
or a canonical formula is **major**; additive optional fields are **minor**; fixes that
leave every fixture output unchanged are **patch**. This release changes the canonical
signal formula and makes result status stricter, so the next application release
must be **2.0.0**; it is left at 1.3.0 here because version bumps are a release-authority
decision.

## 2. Candidate build (APGI-IMP-001)

1. Build from a clean checkout of the candidate commit.
2. Container: `Dockerfile` (digest-pinned base, installs with `-c requirements.lock`).
3. Record: commit SHA, image digest, `requirements.lock` SHA-256, protocol-lock manifest
   SHA-256, SBOM (`python -m utils.dependency_scanner`), `release/checksums.sha256`.
4. Rebuild `release/` (`python scripts/build_release_package.py`) and verify with `--check`.

## 3. Promotion and certificate (APGI-REL-001, APGI-REL-005)

A candidate is promoted only with a signed certificate
(`utils/release_certificate.py`) carrying artifact digests, protocol-lock digest, CI
evidence links and **named** approvers (release authority, scientific reviewer).
`assert_deposit_allowed()` must be called before any OSF/Zenodo deposit; it refuses
uncertified, modified or pending-approval artifacts.

## 4. Deployment, rollback and monitoring (APGI-REL-002, APGI-IMP-003)

* **Deploy:** run the certified image with `APGI_ENV=production`, `APGI_MASTER_KEY`
  set, `APGI_ENFORCE_AUTH` unset.
* **Rollback criteria:** protocol-lock failure at start-up; any K-gate report failing
  signature verification; CI regression on the scientific-conformance or registered-rule
  suites; security finding ≥ MEDIUM.
* **Rollback procedure:** redeploy the previous certified image digest; restore the
  previous `config/protocol_lock.json`; re-verify with `python -m utils.protocol_lock`.
* **Monitoring:** `protocol_lock_audit` log records (outcome `rejected` is an alert);
  `consent_gate` blocked events; quarantine reconciliation counts.
* **Evidence still missing:** a controlled or tabletop rollback record and induced-fault
  evidence for resource/health alerts.

## 5. Change impact and regression selection (APGI-REL-003)

Every material change records, in its PR description: affected claims (CLM), requirements
(APGI-*), risks, and the regression suites selected:

| Change type | Mandatory suites |
|---|---|
| Scientific formula / model | `test_scientific_conformance.py`, `test_theory_equations_core.py`, `test_apgi_engine.py` |
| Protocol definitions / loader | `test_protocol_lock_enforcement.py`, `scripts/validate_protocols.py`, `test_protocol_spec_v1_1.py` |
| Classification / reports | `test_kgate_classification.py`, `test_provenance_and_reports.py`, `test_psychometrics.py` |
| Security / config | `test_data_governance_controls.py`, `test_secret_policy_enforcer_targeted.py`, secret scan |
| Dependencies | full suite on 3.11 + 3.12, `make lock-check`, pip-audit |
