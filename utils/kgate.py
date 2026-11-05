"""K-gate catalog and three-way classifier (S4-VAL-08, S4-VAL-10, S4-VAL-11.x, XC-CI-05.3).

The catalog transcribes the K1–K8 gate criteria from the supplied task
register (S4-PSY-*, S7-KC-*). It is versioned (:data:`CATALOG_VERSION`) and
marked ``approval_status = "PENDING_SCIENTIFIC_REVIEWER"`` until the
registered source is confirmed by the Scientific Reviewer.

Classification has two layers:

``processing_status``
    ``NOT_EVALUATED`` until the evidence-sufficiency rule for the gate is met
    (S4-VAL-11.4). Evidence is insufficient when it is missing, when the
    sample is below the registered minimum, or when any input is not
    empirical/confirmatory — simulation output never counts as empirical
    confirmation (S4-VAL-11.5).

``decision`` (only when ``EVALUATED``)
    Interval rule (S4-VAL-11.2), direction-aware:

    * higher-is-better criterion ``≥ t``: PASS if the interval's lower bound
      ≥ t; FAIL if the upper bound < t (or ≤ the registered falsifier);
      otherwise INDETERMINATE.
    * lower-is-better criterion ``≤ t``: mirror image.

    A registered ambiguous band (e.g. K5: 0.10 < r < 0.15) yields
    INDETERMINATE with ``next_action = "further_sample"`` (S4-VAL-11.3).
    Every INDETERMINATE produced by interval/band uncertainty carries
    ``further_sample``; a binary verdict is never forced.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from utils.data_provenance import ProvenanceRecord, can_support_empirical_confirmation

CATALOG_VERSION = "0.1.0"
CATALOG_APPROVAL_STATUS = "PENDING_SCIENTIFIC_REVIEWER"


class ProcessingStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    EVALUATED = "EVALUATED"


class Decision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


class NextAction(str, Enum):
    NONE = "none"
    FURTHER_SAMPLE = "further_sample"
    COLLECT_EVIDENCE = "collect_evidence"
    RETIRE_ITEMS = "retire_items"


class Direction(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    EXCLUDES_ZERO = "excludes_zero"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class Criterion:
    """One registered quantitative criterion within a gate."""

    criterion_id: str
    metric: str
    direction: Direction
    pass_threshold: Optional[float] = None
    #: Registered falsifier; when omitted the pass threshold doubles as the fail boundary.
    fail_threshold: Optional[float] = None
    #: Registered ambiguous band (lo, hi) mapped to INDETERMINATE + further_sample.
    ambiguous_band: Optional[tuple] = None
    description: str = ""


@dataclass(frozen=True)
class Gate:
    gate_id: str
    title: str
    register_tasks: tuple
    criteria: tuple
    combine: str = "all"  # "all": every criterion must PASS
    min_n: Optional[int] = None
    sample: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["criteria"] = [
            {
                **asdict(c),
                "direction": c.direction.value,
                "ambiguous_band": list(c.ambiguous_band) if c.ambiguous_band else None,
            }
            for c in self.criteria
        ]
        d["register_tasks"] = list(self.register_tasks)
        return d


HB, LB = Direction.HIGHER_IS_BETTER, Direction.LOWER_IS_BETTER

K_GATES: Dict[str, Gate] = {
    g.gate_id: g
    for g in (
        Gate(
            "K1",
            "Test–retest reliability of self-report subscales",
            ("S4-PSY-01",),
            (Criterion("K1.icc", "icc_2_1", HB, 0.70, description="ICC(2,1) ≥ 0.70 per subscale, with interval"),),
            sample="independent retest sample, 2–3 week interval",
        ),
        Gate(
            "K2",
            "Test–retest reliability of behavioural indices",
            ("S4-PSY-05",),
            (Criterion("K2.icc", "icc_2_1", HB, 0.50, description="ICC(2,1) ≥ 0.50"),),
            sample="independent retest sample, 2–3 week interval",
        ),
        Gate(
            "K3",
            "Internal consistency",
            ("S4-PSY-09",),
            (Criterion("K3.omega", "ordinal_omega", HB, 0.80, description="ordinal ω ≥ 0.80 per subscale"),),
        ),
        Gate(
            "K4",
            "Three-factor structure",
            ("S4-PSY-13", "S4-PSY-14", "S4-PSY-15", "S4-PSY-16", "S4-PSY-17", "S4-PSY-18"),
            (
                Criterion(
                    "K4.dbic",
                    "delta_bic_min",
                    HB,
                    10.0,
                    description="ΔBIC ≥ 10 favouring 3 factors vs both alternatives",
                ),
                Criterion(
                    "K4.loading", "min_primary_loading", HB, 0.40, description="all retained primary loadings ≥ 0.40"
                ),
                Criterion("K4.cross", "max_cross_loading", LB, 0.30, description="no retained cross-loading > 0.30"),
                Criterion("K4.cfi", "cfi", HB, 0.90, description="CFI ≥ 0.90"),
                Criterion("K4.rmsea", "rmsea", LB, 0.08, description="RMSEA ≤ 0.08"),
                Criterion("K4.srmr", "srmr", LB, 0.08, description="SRMR ≤ 0.08"),
                Criterion("K4.items", "min_items_per_factor", HB, 4.0, description="each factor retains ≥ 4 items"),
            ),
            sample="independent CFA sample; one estimator family for 1/2/3-factor models",
        ),
        Gate(
            "K4b",
            "Measurement invariance",
            ("S4-PSY-22",),
            (
                Criterion(
                    "K4b.dcfi", "delta_cfi_min", HB, -0.01, description="ΔCFI ≥ −0.01 per step configural→metric→scalar"
                ),
            ),
            notes="Level reached is reported per grouping.",
        ),
        Gate(
            "K5",
            "Self-report vs behavioural threshold convergence",
            ("S4-PSY-26",),
            (
                Criterion(
                    "K5.r",
                    "pearson_r",
                    HB,
                    0.15,
                    fail_threshold=0.10,
                    ambiguous_band=(0.10, 0.15),
                    description="r ≥ 0.15; 0.10–0.15 further sample; ≤ 0.10 falsifier",
                ),
            ),
        ),
        Gate(
            "K6",
            "Criterion prediction on locked holdout",
            ("S7-KC-01",),
            (Criterion("K6.dr2", "delta_r2", HB, 0.03, description="ΔR² ≥ 0.03 on holdout"),),
            sample="locked holdout",
        ),
        Gate(
            "K6b",
            "Practical prediction gain on holdout",
            ("S7-KC-05",),
            (
                Criterion("K6b.mae", "mae_relative_change", LB, -0.15, description="MAE ≤ −15% vs baseline"),
                Criterion(
                    "K6b.precision",
                    "top_quintile_precision",
                    HB,
                    0.60,
                    description="quintile identification precision ≥ 60%",
                ),
            ),
            sample="locked holdout",
        ),
        Gate(
            "K7b",
            "Practical identifiability at deployed trial counts",
            ("S7-KC-13",),
            (
                Criterion(
                    "K7b.bounded",
                    "profile_likelihood_bounded_both_sides",
                    Direction.BOOLEAN,
                    description="profile likelihood bounded on both sides within plausible range",
                ),
            ),
        ),
        Gate(
            "K8",
            "Profile × intervention interaction",
            ("S7-KC-09",),
            (
                Criterion(
                    "K8.interaction",
                    "interaction_effect",
                    Direction.EXCLUDES_ZERO,
                    description="credible interval of the interaction excludes zero",
                ),
            ),
            sample="first 500 experiments",
        ),
    )
}


def catalog() -> Dict[str, Any]:
    """Serializable catalog (source for release/gate-table.yaml)."""
    return {
        "catalog_version": CATALOG_VERSION,
        "approval_status": CATALOG_APPROVAL_STATUS,
        "decision_rule": "interval rule; ambiguous band → INDETERMINATE + further_sample; NOT_EVALUATED before sufficient empirical evidence",
        "gates": [g.to_dict() for g in K_GATES.values()],
    }


@dataclass
class Estimate:
    """Point estimate and interval for one criterion metric."""

    value: Optional[float]
    lower: Optional[float] = None
    upper: Optional[float] = None
    level: float = 0.95

    def bounds(self) -> tuple:
        lo = self.lower if self.lower is not None else self.value
        hi = self.upper if self.upper is not None else self.value
        return lo, hi


@dataclass
class CriterionOutcome:
    criterion_id: str
    decision: Decision
    next_action: NextAction
    rationale: str
    estimate: Optional[Dict[str, Any]] = None


@dataclass
class GateClassification:
    gate_id: str
    processing_status: ProcessingStatus
    decision: Optional[Decision]
    next_action: NextAction
    rationale: str
    criteria: List[CriterionOutcome] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "processing_status": self.processing_status.value,
            "decision": self.decision.value if self.decision else None,
            "next_action": self.next_action.value,
            "rationale": self.rationale,
            "criteria": [
                {**asdict(c), "decision": c.decision.value, "next_action": c.next_action.value} for c in self.criteria
            ],
        }


def _finite(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def classify_criterion(criterion: Criterion, estimate: Any) -> CriterionOutcome:
    """Apply the direction-aware interval rule to one criterion."""
    if criterion.direction is Direction.BOOLEAN:
        if not isinstance(estimate, bool):
            raise TypeError(f"{criterion.criterion_id}: boolean criterion requires a bool estimate")
        if estimate:
            return CriterionOutcome(criterion.criterion_id, Decision.PASS, NextAction.NONE, "criterion satisfied")
        return CriterionOutcome(
            criterion.criterion_id,
            Decision.INDETERMINATE,
            NextAction.FURTHER_SAMPLE,
            "criterion not satisfied; no registered falsifier, so not FAIL",
        )

    if not isinstance(estimate, Estimate):
        raise TypeError(f"{criterion.criterion_id}: expected Estimate, got {type(estimate).__name__}")
    lo, hi = estimate.bounds()
    if not (_finite(lo) and _finite(hi)) or lo > hi:
        raise ValueError(f"{criterion.criterion_id}: invalid interval [{lo}, {hi}]")
    est = asdict(estimate)

    if criterion.direction is Direction.EXCLUDES_ZERO:
        if lo > 0 or hi < 0:
            return CriterionOutcome(
                criterion.criterion_id, Decision.PASS, NextAction.NONE, f"interval [{lo}, {hi}] excludes 0", est
            )
        return CriterionOutcome(
            criterion.criterion_id,
            Decision.INDETERMINATE,
            NextAction.FURTHER_SAMPLE,
            f"interval [{lo}, {hi}] includes 0",
            est,
        )

    t_pass = float(criterion.pass_threshold)  # type: ignore[arg-type]
    t_fail = float(criterion.fail_threshold) if criterion.fail_threshold is not None else t_pass

    if criterion.direction is Direction.HIGHER_IS_BETTER:
        if lo >= t_pass:
            return CriterionOutcome(
                criterion.criterion_id, Decision.PASS, NextAction.NONE, f"lower bound {lo} ≥ {t_pass}", est
            )
        failed = hi <= t_fail if criterion.fail_threshold is not None else hi < t_pass
        if failed:
            return CriterionOutcome(
                criterion.criterion_id, Decision.FAIL, NextAction.NONE, f"upper bound {hi} fails {t_fail}", est
            )
    else:  # LOWER_IS_BETTER
        if hi <= t_pass:
            return CriterionOutcome(
                criterion.criterion_id, Decision.PASS, NextAction.NONE, f"upper bound {hi} ≤ {t_pass}", est
            )
        failed = lo >= t_fail if criterion.fail_threshold is not None else lo > t_pass
        if failed:
            return CriterionOutcome(
                criterion.criterion_id, Decision.FAIL, NextAction.NONE, f"lower bound {lo} fails {t_fail}", est
            )

    band = criterion.ambiguous_band
    if band and estimate.value is not None and band[0] < estimate.value < band[1]:
        reason = f"estimate {estimate.value} in registered ambiguous band {band}"
    else:
        reason = f"interval [{lo}, {hi}] straddles the registered threshold"
    return CriterionOutcome(criterion.criterion_id, Decision.INDETERMINATE, NextAction.FURTHER_SAMPLE, reason, est)


def _combine(outcomes: Sequence[CriterionOutcome]) -> tuple:
    decisions = [o.decision for o in outcomes]
    if any(d is Decision.FAIL for d in decisions):
        return Decision.FAIL, NextAction.NONE
    if all(d is Decision.PASS for d in decisions):
        return Decision.PASS, NextAction.NONE
    return Decision.INDETERMINATE, NextAction.FURTHER_SAMPLE


def evidence_sufficiency(gate: Gate, inputs: Iterable[ProvenanceRecord], n: Optional[int]) -> Optional[str]:
    """Return a reason the evidence is insufficient, or ``None`` if sufficient."""
    records = list(inputs)
    if not records:
        return "no input dataset provenance supplied"
    if not can_support_empirical_confirmation(records):
        origins = sorted({r.data_origin.value for r in records})
        cats = sorted({r.inferential_category.value for r in records})
        return (
            f"inputs are not all empirical+confirmatory (origins={origins}, categories={cats}); "
            "simulation/synthetic output is never counted as empirical confirmation"
        )
    if gate.min_n is not None and (n is None or n < gate.min_n):
        return f"n={n} below registered minimum {gate.min_n}"
    return None


def classify_gate(
    gate_id: str,
    estimates: Optional[Mapping[str, Any]],
    inputs: Iterable[ProvenanceRecord] = (),
    n: Optional[int] = None,
) -> GateClassification:
    """Classify one K gate. Never forces a binary verdict."""
    gate = K_GATES[gate_id]
    insufficiency = evidence_sufficiency(gate, inputs, n)
    if insufficiency is None and not estimates:
        insufficiency = "no estimates supplied"
    if insufficiency is None:
        missing = [c.criterion_id for c in gate.criteria if c.criterion_id not in estimates]  # type: ignore[operator]
        if missing:
            insufficiency = f"missing estimates for {missing}"
    if insufficiency is not None:
        return GateClassification(
            gate_id, ProcessingStatus.NOT_EVALUATED, None, NextAction.COLLECT_EVIDENCE, insufficiency
        )

    outcomes = [classify_criterion(c, estimates[c.criterion_id]) for c in gate.criteria]  # type: ignore[index]
    decision, action = _combine(outcomes)
    if gate_id == "K4" and any(o.criterion_id == "K4.cross" and o.decision is Decision.FAIL for o in outcomes):
        action = NextAction.RETIRE_ITEMS
    rationale = "; ".join(f"{o.criterion_id}: {o.decision.value}" for o in outcomes)
    return GateClassification(gate_id, ProcessingStatus.EVALUATED, decision, action, rationale, outcomes)
