# Formulation register (APGI-SCI-001, APGI-SCI-002, APGI-SCI-003)

**Status:** implementation declaration. Equation IDs from the Formal Mathematical
Specification are **not yet confirmed** (spec marks them `[CONFIRM IDs]`, AUD-SPEC-03),
so this register uses local IDs (`SCI-F*`) that must be mapped once approved.

## SCI-F1 — accumulated signal and ignition (APGI-SCI-001)

Specification: **Sₜ = Πᵉ·|zᵉ| + Πⁱ_eff·|zⁱ|**, ignition iff **Sₜ > θₜ** (strict).

| Entry point | Formula | Conforms |
|---|---|---|
| `apgi_core.equations.CoreIgnitionSystem.accumulated_signal` | Πᵉ\|zᵉ\| + Πⁱ_eff\|zⁱ\| | Yes (changed in this release) |
| `apgi_core.engine.APGICoreSignal.accumulated_signal` | Πᵉ\|zᵉ\| + Πⁱ_eff\|zⁱ\| | Yes |
| `CoreIgnitionSystem.ignites`, `APGIIgnitionMechanism.hard_ignition` | `S > θ` | Yes |
| `CoreIgnitionSystem.quadratic_surprise_signal` | ½Πᵉ(εᵉ)² + ½Πⁱ_eff(εⁱ)² | **No — explicitly non-canonical**, kept as `SCI-F1-QUADRATIC-LEGACY` |

### Adjudication record

* **Conflict:** before this release `CoreIgnitionSystem.accumulated_signal` returned the
  quadratic energy form. Audit probe (Πᵉ=2, zᵉ=0.5, Πⁱ=1, zⁱ=1): quadratic 0.75,
  specification 2.0. The two library paths disagreed.
* **Resolution applied:** the governing software specification wins for any
  implementation of ignition. Both library paths now return 2.0 for the probe; golden
  vectors and the equality boundary are tested in `tests/test_scientific_conformance.py`.
* **Open:** the continuous dynamics `DynamicalSystemEquations.signal_dynamics`
  (`dS/dt = −S/τ_S + ½Πᵉ(εᵉ)² + ½Πⁱ_eff(εⁱ)² + σξ`) still uses quadratic drive. The
  specification states formulations are not numerically interchangeable; whether this
  continuous reduced form is approved, and with which drive term, needs the scientific
  authority (AUD-SPEC-03). It is declared below and **not** used for the canonical rule.
* **Downstream effect:** `Validation/VP_09` calls `accumulated_signal`; its simulated
  numbers change. VP outputs are synthetic and carry no registered vote.

## SCI-F2 — dynamic threshold and dynamics (APGI-SCI-002)

θₜ is dynamically regulated in every entry point; none presents it as a fixed
psychophysical threshold.

| Entry point | Formulation | Integration | dt | Reset / refractory | RNG |
|---|---|---|---|---|---|
| `equations.EnhancedSurpriseIgnitionSystem.step` | continuous reduced form (SDE) | Euler–Maruyama | caller `dt` (s) | as coded in `threshold_dynamics`; **pending approval** | one seeded stream `self.rng` |
| `engine.APGISystemDynamics` | continuous ODE/SDE | Euler | caller `dt` | none declared — **pending** | `self.rng` |
| `full_model.APGIFullDynamicModel.step` | discrete update | explicit per step | 1 step | as coded; **pending** | `self.rng` |
| `model.HierarchicalProcessor` | per-level leaky integration | Euler | default 0.01 s | per level | none |

Approved fixture sets per formulation are **not available** (AUD-SPEC-03); conformance
for SCI-002 therefore remains blocked.

## SCI-F3 — temporal hierarchy (APGI-SCI-003)

Canonical levels implemented in `apgi_core/temporal_hierarchy.py` and used by
`HierarchicalProcessor` (codes `L0`–`L4`, out-of-range overrides raise):

| Level | Range | Representative τ |
|---|---|---|
| L0 | 10–100 ms | 50 ms |
| L1 | 100 ms–3 s | 1 s |
| L2 | 1–30 min | 10 min |
| L3 | hours–days → **1 h–7 d (interpretation)** | 1 d |
| L4 | months–years → **30 d–10 y (interpretation)** | 6 months |

The numeric bounds for L3/L4 and the "shared edge belongs to the lower level" rule at
100 ms are implementation interpretations requiring confirmation (AUD-SPEC-07).

`utils/constants.LEVEL_TIMESCALES` (sensory 0.05 s; organ 0.2 s; cognitive 0.5 s;
homeostatic 2.0 s) is a **physiological sub-level scheme inside L0/L1**, not an
alternative hierarchy; it must not be labelled as L0–L4.
