# APGI protocol specification release 1.1

Prepared 2026-10-16 from the local APGI specifications dated 2026-10-16. This package contains all eight protocol JSON files plus their revised schema. Version strings use semantic form `1.1.0`. Original files were not modified.

## Authority and status

The local OSF Preregistration governs empirical designs and statistical decisions. Mathematical Specification governs equations, and Notation Appendix governs symbols. The companion Protocols document supplies additional details where consistent. Source document SHA-256 values are included in each JSON. Embedded editorial instructions were treated as document content, not executed.

The documents describe pre-collection protocols, pending ethics submissions and incomplete analysis locks. This export does not assert that Amendment 1 has already been filed: its stated date is one day after preparation, and the identifier is absent. It is a specification update, not authorization for recruitment or a claim of completed validation.

## Schema migration

The schema retains the familiar protocol fields while supporting explicit nulls for unknown/inapplicable counts and pending approval IDs/dates. It distinguishes seed counts from human cohorts, fixed-N/sequential/Bayesian designs, prediction evidence classes, source provenance and readiness. It no longer forces kappa=100, generic alpha=0.3 or beta_initial_estimate=0.7; kappa is the uncalibrated ATP/bit bridge. The reduced empirical model uses named parameters and requires a frozen parameter/observation manifest.

This is a data-contract migration: consumers that require numeric n_minimum, non-null trial counts, old approval dates or scalar kappa must be updated. The schema validates exact Protocol 0–7 prediction membership. `specification_sections` preserve detailed source context; the curated structured decision fields and conflict notes control where source summaries disagree. Their source line locators refer to the extracted paragraph stream, not page numbers.

## Protocol 0

Replaced heartbeat counting with discrimination; independent replication criteria and central EEG engagement; updated anatomical equivalence bounds to +/-sqrt(0.02); targeted proxy-reclassification tree and 2-of-3 custody.

Input version: `1.0.0`; requested specification version: `1.1.0`.

Open source/lock requirements:

- Pred 0.B central EEG engagement band/direction/normalization/window and authorized pharmacological protocol must be locked; no fixed dose schedule is authorized by this JSON.

- Replication conjunction power simulation pending.

- Pred 0.A/0.C source alternates > and >= at retention boundary; strict > used provisionally from prediction table; exact-boundary results require source clarification.

- Final site-specific eligibility, approvals, roles, observation model and executable analysis manifest must be prospectively locked; source specification is not a completed operational study authorization.

## Protocol 1

Updated personalized cardiac windows; N62 enrollment/N59 power target/N48 conditional floor; six-level 480-trial sessions; continuous IA; separate Pred 1.D coupling, 1.E Weibull decision and 1.F trait effect.

Input version: `2.0.0`; requested specification version: `1.1.0`.

Open source/lock requirements:

- Main-session QUEST top-up is 30 in Data phases but 50 in session-total paragraph (590 vs 610 including mini-blocks); resolve before acquisition lock.

- Usable-trial minima are explicitly labeled proposals in source; require prospective lock.

- Pred 1.E hierarchical priors, comparator parameterization, observation likelihood and acquisition rules require archived lock.

- Final site-specific eligibility, approvals, roles, observation model and executable analysis manifest must be prospectively locked; source specification is not a completed operational study authorization.

## Protocol 2

Seven architectures; independent CP-3a calibration; scaled identifiability diagnostics; deterministic 1000-seed-per-condition sweep; human BIC separated from simulation; new exploratory 2.F/2.G.

Input version: `2.0.0`; requested specification version: `1.1.0`.

Open source/lock requirements:

- CP-3a calibration, scaled information-matrix pre-check and parameter/observation manifest must pass before confirmatory simulation.

- Pilot convergence archive unverified; pilot data cannot enter confirmatory tests.

- Final site-specific eligibility, approvals, roles, observation model and executable analysis manifest must be prospectively locked; source specification is not a completed operational study authorization.

## Protocol 3

N40/N36; 200 trials with value/contrast and foreperiod controls; updated ROIs, fMRI pipeline and parameter-identifiability gate; exploratory timing and trait claims separated.

Input version: `2.0.0`; requested specification version: `1.1.0`.

Open source/lock requirements:

- Framework integration section labels anticipatory PPI as Pred 3.A, whereas detailed prediction registry assigns it Pred 3.B; this export uses 3.B and flags cross-protocol labeling for amendment.

- Composite-parameter fallback and identifiability evaluation grid require prospective lock.

- Final site-specific eligibility, approvals, roles, observation model and executable analysis manifest must be prospectively locked; source specification is not a completed operational study authorization.

## Protocol 4

N64/N55/N48 across two labs; four sessions and three-level contrast grid; signed 15% d-prime contrast and probit GLMM; arousal adjustment is not metabolic mediation; timescale claims deferred/exploratory.

Input version: `1.0.0`; requested specification version: `1.1.0`.

Open source/lock requirements:

- Level-2 mechanistic timescale model/recovery simulation not specified; ordering deferred, session trends descriptive.

- Threshold-recovery analysis remains exploratory; no primary framework falsification vote.

- Final site-specific eligibility, approvals, roles, observation model and executable analysis manifest must be prospectively locked; source specification is not a completed operational study authorization.

## Protocol 5

Right-aINS tFUS/dTMS hierarchy; modality-matched sham and engagement gates; separate resting PCI probes; updated contrast family; 95% HDI ROPE equivalence; Tier3 separate and exploratory.

Input version: `2.0.0`; requested specification version: `1.1.0`.

Open source/lock requirements:

- Approved dosimetry/stimulation parameters, persistence during PCI probing and corrected-alpha simulation required before recruitment.

- Companion Tier-2 engagement threshold differs from OSF full-specification gate; OSF gate used here; reconcile before site activation.

- IA moderation family allocation is not fully resolved by the final three-contrast delta-PCI family; require locked analysis manifest rather than importing older p<0.05 prose.

- Final site-specific eligibility, approvals, roles, observation model and executable analysis manifest must be prospectively locked; source specification is not a completed operational study authorization.

## Protocol 6

N20/N12 CE design; 600-trial absolute contrast grid per OSF; distinct LFP/microwire/aINS subsets; fixed-contrast mixture tests; AC1 validation hold; surrogate-controlled >=3-timescale recovery.

Input version: `2.0.0`; requested specification version: `1.1.0`.

Open source/lock requirements:

- Five samples per 50-ms AC1 window at 100 Hz: finite-sample estimator recovery and power must be archived; otherwise Pred 6.C deferred.

- Absolute contrast grid in OSF conflicts with threshold-relative companion grid; OSF grid governs.

- CE upgrade requires prospective amendment and independent replication in >=12 additional patients.

- Final site-specific eligibility, approvals, roles, observation model and executable analysis manifest must be prospectively locked; source specification is not a completed operational study authorization.

## Protocol 7

Pred 7.G primary at alpha0.048, N80 patients; 300 clean heartbeats and absolute Cz HEP; 7.A estimation-first with four-group AND patient-only McFadden deltaR2; carrier-matched perturbation control; 7.E/7.F exploratory; corrected CRS-R/GOS-E and missingness rules.

Input version: `2.0.0`; requested specification version: `1.1.0`.

Open source/lock requirements:

- Pred 7.A joint-success power and patient-only analyzable floor require archived simulation; estimation-first.

- Carrier-matched sham gas requires physician approval, composition/delivery authorization and registration before any CO2-specific confirmatory claim.

- Pred 7.C group-interaction multiplicity allocation must be locked in addition to the two one-sided 0.025 contrasts.

- Pred 7.G sensitivity-model prose refers to baseline CRS-R without explicitly giving the complete formula; manifest must specify baseline CRS-R plus days post-injury for the beyond-severity claim.

- Pred 7.A binding futility boundary remains inactive until its calibration is archived.

- Final site-specific eligibility, approvals, roles, observation model and executable analysis manifest must be prospectively locked; source specification is not a completed operational study authorization.

## Cross-source resolutions

- Pred 0.C uses OSF +/-sqrt(0.02), not the companion +/-0.10 shorthand.

- Protocol 6 uses the OSF absolute Michelson grid, not the companion c50-relative grid.

- Protocol 5 uses the final OSF three-contrast delta-PCI secondary family; older companion descriptions differ.

- Pred 7.A C-index cannot independently falsify the incremental-prediction claim; the detailed decision rule governs over the abbreviated table.

- Nonsignificance alone does not establish equivalence or falsification. Detailed positive-evidence and indeterminate rules supersede shorthand null statements.

- Generic source paragraphs may still use legacy shorthand; no omitted operational detail, unverified power or unresolved threshold is silently supplied from the old JSON.

## Validation

All eight protocol instances pass the supplied Draft 2020-12 schema with format checking. Forty unique prediction records are present. Trial arithmetic, cohort totals, pending approval fields and original input hashes were checked. See VALIDATION.json. No empirical analysis or power simulation was run.

## 2026-10-16 validation and repository integration audit

The supplied 1.1.0 package passed its original schema and manifest checks. All four cited source-document hashes matched the local FINAL documents, and all eight historical input hashes matched the repository's pre-migration files. These are provenance checks, not scientific validation. The original supplied package was backed up separately before this audit; the files in this release have now been updated in place.

Corrections made against the local specifications:

- Protocol 1: set the unresolved main-session QUEST top-up count to null, preserving the source alternatives 30 and 50. Mark usable-trial minima explicitly as proposals pending prospective lock. This does not resolve the conflicting source schedule.
- Protocol 2.C: restore the missing Kendall tau >0.3 criterion, the +/-5-trial correlation window, median within-seed tau and seed-level bootstrap interval (OSF Table 6 and detailed Protocol 2.C). Ignition events are not independent replicates.
- Protocol 4.A: align the summary with the signed >=15% proportional margin, BF10 and effect-size requirements; preserve the separate secondary rule for nonpositive baseline d-prime (OSF detailed Protocol 4.A).
- Protocol 5.A: replace the universal vertex-sham summary with the binding modality-matched comparisons. Flag 5.C/5.D as conditional on the unresolved IA-moderation multiplicity lock.
- Protocol 6.B: replace nonsignificant occipital dip shorthand with positive model evidence and a direct regional contrast (OSF detailed Protocol 6.B).
- All eight protocols: make the precedence of prediction-specific interval/evidence rules explicit. Nonsignificance and point estimates alone cannot establish equivalence or falsification; no unregistered interval bounds or Bayesian hypotheses have been invented. Source excerpts remain historical context.

Schema and tooling corrections:

- Enforce null kappa while uncalibrated and a positive numeric value when calibrated; restrict the eight reduced quantities to the registered names.
- Pending as well as unsubmitted approvals cannot contain granted approval identifiers/dates or authorize enrollment. Validate readiness booleans, source fields and clinical group counts.
- Replace import-time broad JSON discovery with runtime protocol-only discovery. Add strict JSON parsing, format checks, schema self-validation, cross-field sample/trial/cohort checks, exact eight-protocol membership, report consistency and manifest coverage/checksum validation.
- Add optional local source-hash verification and explicit metadata regeneration. The generated report replaces the ambiguous historical `source_unchanged` flags; it states which checks are reproducible and which scientific/source-lock issues remain unresolved.
- Update repository CI, protocol documentation and mutation regression tests for release 1.1.0. The Python library retains its independent package version.

The previously listed scientific lock requirements remain open. In particular, this audit does not select new statistical hypotheses, supply unverified power simulations, resolve conflicting source thresholds, verify OSF filing, or authorize study activation.
