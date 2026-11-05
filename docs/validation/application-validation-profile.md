# Application Validation Profile — apgi-validation (component A05)

**Status:** DRAFT — proposed by the implementation team, **not approved**. Every
scope decision below needs sign-off by the specification authority (AUD-SPEC-09,
AUD-GOV-01). Until then each "proposed" entry is a proposal, not an approved exclusion.

| Field | Value |
|---|---|
| Application | `apgi-validation` (A05) |
| Application version | 1.3.0 (`utils/versioning.py`) |
| Governing specification | APGI Protocol v1.2 Software Validation Specification (draft) |
| Protocol definitions implemented | Bundle **1.1.0**, Protocols 0–7 (`protocols/`, locked by `config/protocol_lock.json`) |
| Protocol v1.2 equivalence | **UNRESOLVED** (APGI-PRO-001, AUD-SPEC-01). The bundle is *not* presented as v1.2. |
| Release owner | **UNAPPOINTED** (AUD-GOV-01) |
| Independent reviewer | **UNAPPOINTED** — all code in this repository is agent/author-built and has not been independently accepted |

## 1. Declared protocol scope (APGI-PRO-004)

Supported: registered definitions `APGI-P00` … `APGI-P07` at bundle 1.1.0, loaded
only through `utils/protocol_loader.py`, which verifies the protocol lock first.

Not supported as registered protocols: legacy `VP_*` / `FP_*` simulation modules in
`Validation/` and `Falsification/`. They are **simulation and method-validation
tools**; their outputs carry `evidence_class: synthetic` and can never confirm or
falsify a registered prediction (S4-VAL-11.5). Legacy `VP-##`/`FP-##` IDs are rejected
by the loader.

## 2. Functions, platforms, interfaces

| Item | Declared scope |
|---|---|
| Functions | protocol loading/locking; simulation of APGI dynamics; registered K-gate classification; validation report generation and signing; release packaging |
| Platforms | Linux x86-64 (CI, container). macOS ARM64 and Windows: GUI smoke only — **no numerical cross-platform equivalence evidence yet** (APGI-REP-002) |
| Python | 3.11 and 3.12 |
| Interfaces | CLI (`main.py`), Tk GUIs (`*_GUI.py`), library API; **consumer** of `api.apgiframework.com` via `utils/apgi_api_client.py` |
| Server API | None. A05 exposes no network API. |

## 3. Data classes

| Class | Present? | Handling |
|---|---|---|
| Synthetic / simulated | Yes (`data_repository/synthetic_data/`) | `data_origin=synthetic`; never empirical evidence |
| Imported empirical (public datasets) | Loader support only (BIDS) | Requires source, licence, version, SHA-256, acquisition metadata, transform history (`utils/data_provenance.py`) |
| Human-observed research data | **Not held** | Consent gate (`utils/consent_gate.py`) available; enforcement scope pending §4 |
| Credentials / keys | Yes | `utils/secure_key_manager.py`; no ephemeral key in production |

## 4. Applicability decisions (proposed)

| Requirement | Section 5.2 | Proposed decision | Rationale / control |
|---|---|---|---|
| APGI-PRO-003 correlation/idempotency | C | **Applicable to consumer GET/HEAD only.** No write calls are made; no write-retry is introduced. | `apgi_api_client.py` retries idempotent methods only |
| APGI-API-001 contracts | C | **Consumer-only scope.** Canonical contract snapshot/hash pending AUD-SPEC-06 | — |
| APGI-API-002 authorization | – | **Applicable to local protected operations** (GUI/headless runs, key management). Role matrix: RESEARCHER or ADMIN run protocols; GUEST nothing; no implicit admin override | `utils/runtime_profile.py`, `utils/auth_adapter.py` |
| APGI-API-004 errors | – | **Applicable to `ProtocolResult` and client errors.** Incomplete results cannot be `success`; client errors carry status + validated code only | `utils/protocol_schema.py`, `utils/apgi_api_client.py` |
| APGI-INT-004 presentation values | – | **Applicable to GUI/report values derived from K-gate reports** once any exist; Explorer/Appendix/website/quiz are out of A05 | — |
| APGI-CON-001/002 consent & withdrawal | –/C | **Applicable whenever human-observed data is ingested.** A05 currently holds none; the gate and withdrawal orchestrator are implemented so the decision can be enforced | `utils/consent_gate.py` |
| APGI-ACQ-001 acquisition | – | **Not applicable**: A05 performs no acquisition. Acquisition metadata consumed under DAT/INT | — |
| APGI-CLI-001 client requirements | – | **Not applicable**: A05 ships no participant instruments or quiz content | — |

## 5. Permitted development scope

`APGI_ENFORCE_AUTH=0`, `APGI_ALLOW_EPHEMERAL_MASTER_KEY=1`, `APGI_SKIP_PROTOCOL_LOCK=1`
and `APGI_SKIP_MANIFEST_CHECK=1` are development-only and are **refused when
`APGI_ENV=production`**. Authentication is enforced when `APGI_ENFORCE_AUTH` is unset.
