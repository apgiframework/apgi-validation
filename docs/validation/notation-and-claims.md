# Notation map and claim register (APGI-SCI-004, APGI-SCI-007)

**Status:** DRAFT. The controlled Notation Appendix and the CLM claim register are
owned by the specification/scientific authority and have **not been supplied**
(AUD-SPEC-03, AUD-SPEC-04). The tables below map this repository's identifiers to the
symbols the specification uses so the approved appendix can be applied mechanically.
They are not themselves an approved appendix or register.

## 1. Symbol-to-identifier map (draft)

| Spec symbol | Meaning | Code identifiers | Notes |
|---|---|---|---|
| Sₜ | accumulated signal | `S`, `s_t`, `S_t` | canonical formula SCI-F1 |
| θₜ | dynamic ignition threshold | `theta`, `theta_t`, `theta0` (baseline θ₀) | never a fixed psychophysical threshold |
| Πᵉ | exteroceptive precision | `Pi_e`, `pi_e` | |
| Πⁱ_eff | effective interoceptive precision | `Pi_i_eff`, `pi_i_eff` | |
| zᵉ, zⁱ | z-scored prediction errors | `z_e`, `z_i`, `ze`, `zi`, `eps_e`/`eps_i` **when z-scored** | `eps_*` also denotes raw ε — **collision** |
| εᵉ, εⁱ | raw prediction errors | `eps_e`, `eps_i` | see above |
| Bₜ | broadcast / ignition probability | `B`, `b_t`, `ignition_prob` | |
| Mₜ | somatic marker | `M` | |
| Aₜ | arousal | `A` | |
| α (steepness) | logistic steepness 1/σ_noise | `alpha` in `ignition_probability` | **collision** with α in precision dynamics |
| β_som | somatic modulation gain | `beta`, `beta_som` | **collision**: bare `beta` |
| β_cross | cross-level coupling | `beta_cross` | |
| γ | threshold adaptation gain | `gamma` | **collision** with `gamma_context` |
| τ_S, τ_θ, τ_ℓ | time constants | `tau_S`, `tau_theta`, `HierarchicalLevel.tau` | |

**Open findings (APGI-SCI-004):** bare `alpha`, `beta` and `gamma` appear in several
modules with different meanings (see `utils/criteria_registry.py`,
`utils/apgi_settings.py`). They must receive the disambiguating subscripts defined by
the approved appendix; renaming before that appendix exists would only create a third
convention.

## 2. Claim register (draft, CLM-nnn)

Epistemic labels: **established** (independent literature), **derived** (follows
mathematically from APGI assumptions), **calibrated** (fitted post hoc),
**speculative** (proposed, untested).

| ID | Statement as shown to users | Where | Label | Note |
|---|---|---|---|---|
| CLM-001 | Ignition occurs when precision-weighted surprise Sₜ exceeds a dynamic threshold θₜ | README, docs/APGI-Equations.md, GUIs | **speculative** (core APGI hypothesis) | registered predictions test it; none evaluated |
| CLM-002 | The threshold θₜ is dynamically regulated by allostatic state | docs, `equations.py` | **speculative** | |
| CLM-003 | Five-level temporal hierarchy L0–L4 | docs, `temporal_hierarchy.py` | **derived** from the APGI papers | numeric L3/L4 bounds are interpretations |
| CLM-004 | Validation/falsification scores reported by VP/FP modules | GUIs, `docs/Results.md` | **calibrated/synthetic** | simulation only — **no empirical validation exists** |
| CLM-005 | Clinical phenotype mappings | `docs/CLINICAL-PHENOTYPE-MAPPING.md`, `clinical_phenotype_mapper.py` | **speculative** | must not be read as diagnostic |
| CLM-006 | Hurst/DFA biomarker interpretation (H > 0.5 "healthy") | `full_model.py` | **established** (method) / **speculative** (APGI interpretation) | |
| CLM-007 | Default parameter values | `config/parameter_provenance.yaml` | **assumed** | until sources are supplied |

Reconciliation required (APGI-SCI-007): broad "validation score" language in the GUIs
and `docs/Results.md` must be qualified as *simulation-only*. No K-gate has been
evaluated on empirical data (see `release/results/tables/k-gate-status.csv`).
