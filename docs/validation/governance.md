# Validation governance (AUD-GOV-01 … AUD-GOV-05)

These items require **people and approvals**, which software cannot supply. This
document defines the records; the entries stay `UNAPPOINTED` / `PENDING` until the
authorities fill them. Nothing here should be read as an approval.

## AUD-GOV-01 — Authorities and independence

| Role | Person | Independence level | Appointed |
|---|---|---|---|
| Scientific authority | UNAPPOINTED | — | — |
| Specification authority | UNAPPOINTED | — | — |
| Release authority | UNAPPOINTED | — | — |
| Scientific Reviewer (K-gates) | UNAPPOINTED | must be independent of the implementer | — |
| Security reviewer | UNAPPOINTED | — | — |

All code delivered so far was produced by the implementation team (including AI-assisted
work). The register notes that agent-built code needs the Scientific Reviewer's
independent check **before it counts as validation**.

## AUD-GOV-02 — Plans

Validation, risk and data plans are to be approved by the authorities above. Inputs
available now: `application-validation-profile.md` (scope, data classes),
`release-procedure.md` (test selection, environments), `formulation-register.md`.

## AUD-GOV-03 — Retained validation record fields

Every retained run record must carry: tester, timestamp, run ID, configuration, build
(commit + image digest), `requirements.lock` SHA-256, protocol-lock SHA-256, seed,
dataset IDs with version + SHA-256, software/schema versions, outputs and their digests,
diff against expected, defects raised, reviewer and review outcome. Signed K-gate reports
(`utils/validation_report.py`) already carry the machine-generated subset; tester and
reviewer fields are added by the reviewer at sign-off. Retention and access-control
schedule: **PENDING** authority approval.

## AUD-GOV-04 — Deviations and expiry

| ID | Requirement | Deviation | Impact | Compensating control | Owner | Expiry | Approved by |
|---|---|---|---|---|---|---|---|
| DEV-001 | APGI-PRO-001 | Bundle 1.1.0 used; v1.2 equivalence unresolved | registered rules may differ from v1.2 | lock verification; no claim of v1.2 support | UNAPPOINTED | PENDING | PENDING |
| DEV-002 | APGI-SCI-002 | continuous dynamics keep quadratic drive | dynamics ≠ canonical SCI-F1 | declared in formulation register; canonical rule uses SCI-F1 | UNAPPOINTED | PENDING | PENDING |
| DEV-003 | APGI-REP-002 | no macOS ARM64 numerical comparison | cross-platform drift undetected | same-environment bitwise test; env fingerprint in reports | UNAPPOINTED | PENDING | PENDING |

An expired or unapproved deviation blocks certification (`release_certificate`
refuses pending approvals). Non-waivable failures: protocol-lock failure, synthetic data
counted as empirical, unsigned/invalid K-gate report.

## AUD-GOV-05 — Traceability

Bidirectional traceability lives in `traceability.md` (requirement/task → implementation
→ tests → status → remaining blocker).
