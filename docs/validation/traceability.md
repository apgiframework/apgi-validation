# Traceability register — task and requirement closure status

Bidirectional map for every item in the supplied registers
(`validation-all-tasks`, `validation-all-tasks1`, `validation-still-open-items`,
`version-12-requirements`) → implementation → verification → remaining blocker.

**Status vocabulary.** *Implemented* = code/config delivered and tested here.
*Partial* = the in-repo part is delivered; something outside the repository is still
required. *Blocked* = cannot be completed without data, approvals or other components;
nothing here claims otherwise. **No item is "Done" in the register's sense until the
independent reviewer accepts it (AUD-GOV-01).**

## 1. APGI v1.2 requirements

| Req | Status | Implementation | Verification | Remaining (owner) |
|---|---|---|---|---|
| PRO-001 | Blocked | bundle 1.1.0 locked; not presented as v1.2 | lock tests | approved v1.2 map + fixtures (AUD-SPEC-01) |
| PRO-002 | **Implemented** | strict JSON, schema+version validation, all-or-nothing `load_all_protocols`, load audit | `test_protocol_lock_enforcement.py` (ID-only, wrong version, no partial state) | — |
| PRO-003 | Partial | scope declared (consumer GET/HEAD only) in AVP | — | approval of profile |
| PRO-004 | Partial | `application-validation-profile.md` | — | approval; owner appointment |
| SCI-001 | **Implemented** | canonical Sₜ in both paths; strict `>`; quadratic form renamed non-canonical | golden vectors, audit probe, equality boundary | approved reference implementation comparison |
| SCI-002 | Blocked | formulation declarations (`formulation-register.md`); seeded RNG | — | approved equation IDs + fixtures (AUD-SPEC-03) |
| SCI-003 | **Implemented** | `apgi_core/temporal_hierarchy.py`; processor L0–L4 with range checks; settings 5 levels | boundary/out-of-range tests | confirm L3/L4 numeric bounds |
| SCI-004 | Blocked | draft symbol map, collisions listed | — | Notation Appendix |
| SCI-005 | Partial | `config/parameter_provenance.yaml` (all 74 settings defaults, honestly `assumed`), `utils/parameter_provenance.py` | registry completeness test | sources from scientific authority |
| SCI-006 | **Implemented** | `CITATION.cff`; 6 wrong expansions fixed; `scripts/scan_terminology.py` | CI scan | archive/remote metadata review |
| SCI-007 | Partial | draft CLM register | — | approved claim register; GUI wording |
| REP-001 | **Implemented** (proof pending) | trust anchor `config/protocol_lock.json`, per-file SHA-256, whole-run failure, audit log, dev bypass refused in production (both lock and Python manifest) | tamper/missing/unlisted/coordinated-edit tests | OpenTimestamps proof reference |
| REP-002 | Partial | single seeded stream in `EnhancedSurpriseIgnitionSystem`, `APGISystemDynamics`, `APGILiquidNeuralNetwork`, `APGIFullDynamicModel`; env fingerprint (versions/BLAS/threads) in reports | same-seed bitwise test | macOS ARM64 vs Linux x86-64 tolerance run; remaining global-RNG call sites in measurement equations and VP/FP modules |
| REP-003 | **Implemented** | placeholders removed; Fisher-z pearsonr; named errors; ceil-N guarded; VP fallbacks to 0.80 removed | statsmodels 1e-6 comparison, minimal-N test, probe | recompute protocol-reported power sets (needs approved designs) |
| REP-004 | Partial | `inferential_category` on records; exploratory/synthetic inputs → NOT_EVALUATED | classification tests | deviation records for every registered-plan departure |
| REP-005 | Partial | NumPy-1.26-safe trapezoid; metabolic cost units stated (signal·s) | independent 1e-6 test | calibration tolerances only if physical measurement declared |
| IMP-001 | Partial | Docker base digest-pinned; installs with `-c requirements.lock` | — | release-bound build record, SBOM, image digest |
| IMP-002 | **Implemented** | auth enforced by default; headless gated; dev opt-outs refused in production; ephemeral key refused in production | `test_data_governance_controls.py` | — |
| IMP-003 | Partial | monitoring signals listed in release procedure | — | induced-fault evidence |
| DAT-001 | Blocked | — | — | approved golden dataset |
| DAT-002 | **Implemented** | `utils/data_quarantine.py` (5 named outcomes, notify hook, reconciliation) | reconciliation tests | wiring into each concrete loader |
| DAT-003 | Blocked | — | — | approved per-class retention policy |
| DAT-004 | **Implemented** (code) | provenance records require source/licence/version/SHA-256/history; `ProtocolResult.dataset_provenance` + `assert_exchangeable()`; report binds input hashes | provenance + checksum tests | apply to real datasets |
| API-001 | Blocked | consumer-only scope declared | — | canonical contract (AUD-SPEC-06) |
| API-002 | Partial | admin bypass removed; role lists explicit | permission tests | approved role matrix / object-level scope |
| API-003 | Partial | content+history secret scan in CI; client never echoes response bodies | scan + redaction tests | approved storage encryption for sensitive backups |
| API-004 | **Implemented** | incomplete result cannot be `success`; stable `error_code` | ProtocolResult + client tests | canonical consumer contract |
| INT-001 | Partial | `schema_version`, `protocol_version`, `producer_version` on `ProtocolResult` | exchange tests | canonical schemas + hashes |
| INT-002 | Blocked | — | — | pinned release set of all components |
| INT-003 | **Implemented** | no H=0.5 substitution; no power/N constants | failure-injection tests | loader/API boundary injection beyond these |
| INT-004 | Blocked | scope proposed | — | upstream artifacts |
| REL-001 | Partial | signed release certificate + promotion rule | certificate tests | named approvers, candidate |
| REL-002 | Partial | deploy/rollback/monitoring procedure | — | rollback exercise record |
| REL-003 | Partial | change-type → suite selection table | — | per-change records |
| REL-004 | Partial | compatibility matrix | version equality test | approved v1.2 / canonical rows |
| REL-005 | **Implemented** (gate) | `assert_deposit_allowed` | refusal tests | certificate for a real release |
| CON-001 | Partial | consent gate (absent/withdrawn/expired/scope/purpose) | blocking tests, nothing persisted | scope decision |
| CON-002 | Partial | lineage-wide withdrawal across stores + deletion record | withdrawal test | approved backup policy |
| ACQ-001 | Scope | proposed not applicable | — | approval |
| VAL-001 | Partial | immutable rules (lock), deterministic K classifier, signed reports with lock/input hashes | classifier + report tests | registered fixtures per outcome type |
| CLI-001 | Scope | proposed not applicable | — | approval |

## 2. Supplied task register

| Task | Status | Delivered |
|---|---|---|
| S4-PSY-01/05 (K1/K2 ICC) | Code ready · result **Blocked** | `psychometrics.icc_2_1` (Shrout–Fleiss reference) |
| S4-PSY-09 (K3 ω) | Code ready · Blocked | polychoric + PAF + ordinal ω |
| S4-PSY-13…18 (K4) | Code ready · Blocked | `CFAFit`, `k4_estimates`; SEM estimator family not chosen |
| S4-PSY-22 (K4b) | Code ready · Blocked | `invariance_level` |
| S4-PSY-26 (K5) | Code ready · Blocked | Fisher-z CI; band rule in classifier |
| S7-KC-01/05 (K6/K6b) | Code ready · Blocked | ΔR², MAE change, quintile precision |
| S7-KC-09/13 (K8/K7b) | Code ready · Blocked | credible interval; profile-likelihood bounds |
| S4-VAL-01 | Partial | runtime schema/version/lock refusal |
| S4-VAL-02, 05, 06, 15 | Blocked | need frozen data / S1-REG-04 / S1-REGN-07 |
| S4-VAL-03 | Partial | unsupported-method constants removed |
| S4-VAL-04 | Blocked | needs preregistered grid on frozen data |
| S4-VAL-07 | Implemented (code) | provenance model + research-store guard |
| S4-VAL-08 | Implemented (code) | three-way classifier for every K gate |
| S4-VAL-09 | Implemented (code) | Ed25519 signed reports, tamper detection |
| S4-VAL-10 | Partial | versioned catalog + classifier; per-gate deposits await evaluated results |
| S4-VAL-11, 11.1–11.5 | Implemented (code) | report schema, interval rule, band→further_sample, NOT_EVALUATED, simulation exclusion |
| S4-VAL-12.1–12.4 | Blocked | frozen datasets |
| S4-VAL-13, 13.1–13.4 | Implemented (code) | enums, inheritance, categories, imported requirements |
| S4-VAL-13.5 | Implemented | `empirical_data/` → `synthetic_data/`, labels `synthetic` |
| S4-VAL-13.6 | Partial | guard bars Appendix/Explorer here; server-side store owner must enforce |
| S4-VAL-14 | Blocked | WIF (S0-INF-06.4) |
| S4-VAL-16, 16.1–16.12 | Implemented | `release/` (12 artifacts) + builder/`--check`; results honestly NOT_EVALUATED |
| XC-CI-05, .1–.3 | Implemented in CI config | `validation-gates.yml` (3 named gates) |
| CI-NOLDS | Fixed | nolds 0.6.2 + setuptools < 81 |
| CI-ISORT | Fixed | isort applied |
| CI-BANDIT | Fixed | policy via supported CLI flags |
| CI-SEMGREP | Unblocked | runs once Bandit passes; findings to triage on first run |
| CI-DOWNSTREAM | Unblocked | follows isort fix |
| CI-MEMORY | Diagnosed | see PR notes |
| CI-PROTECTION | Blocked | repository admin must require the gate checks |
| CI-GUI/LOCK/SECRET-PATHS/DEPENDENCIES/RUFF/BLACK | Preserved | unchanged gates still present |

## 3. Protocol-source and specification blockers

AUD-P00…P07-OPEN-*, AUD-SPEC-01…09 and S4-REC-01/02, S1-REG-04, S0-INF-06.4,
S1-REGN-07 require scientific/study/specification authorities or other projects. They
remain **Blocked**; the code preserves deferred/exploratory status (non-confirmatory
inputs and unregistered rules cannot produce a K-gate decision).
