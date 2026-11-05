"""TC-VAL-01 / XC-CI-05.3: K-gate interval, band and sufficiency rules (S4-VAL-11.x)."""

import pytest

from utils.data_provenance import DataOrigin, InferentialCategory, ProcessingStage, ProvenanceRecord
from utils.kgate import (
    K_GATES,
    Decision,
    Estimate,
    NextAction,
    ProcessingStatus,
    catalog,
    classify_criterion,
    classify_gate,
)

SHA = "a" * 64


def empirical(rid="d1", category=InferentialCategory.CONFIRMATORY):
    return ProvenanceRecord(
        record_id=rid,
        data_origin=DataOrigin.HUMAN_OBSERVED,
        processing_stage=ProcessingStage.RAW,
        inferential_category=category,
        source="site-A",
        version="1",
        sha256=SHA,
        license="CC-BY-4.0",
    )


def synthetic(rid="s1"):
    return ProvenanceRecord(
        record_id=rid,
        data_origin=DataOrigin.SYNTHETIC,
        processing_stage=ProcessingStage.RAW,
        inferential_category=InferentialCategory.METHOD_VALIDATION,
    )


K1 = K_GATES["K1"].criteria[0]
K5 = K_GATES["K5"].criteria[0]
RMSEA = [c for c in K_GATES["K4"].criteria if c.criterion_id == "K4.rmsea"][0]


@pytest.mark.parametrize(
    "est,decision",
    [
        (Estimate(0.72, 0.58, 0.82), Decision.INDETERMINATE),  # register example
        (Estimate(0.80, 0.70, 0.88), Decision.PASS),  # lower bound == threshold passes
        (Estimate(0.55, 0.40, 0.69), Decision.FAIL),  # upper bound below threshold
        (Estimate(0.60, 0.50, 0.70), Decision.INDETERMINATE),  # upper bound == threshold
    ],
)
def test_interval_rule_higher_is_better(est, decision):
    out = classify_criterion(K1, est)
    assert out.decision is decision
    if decision is Decision.INDETERMINATE:
        assert out.next_action is NextAction.FURTHER_SAMPLE


@pytest.mark.parametrize(
    "est,decision",
    [
        (Estimate(0.05, 0.03, 0.08), Decision.PASS),
        (Estimate(0.10, 0.081, 0.12), Decision.FAIL),
        (Estimate(0.07, 0.05, 0.09), Decision.INDETERMINATE),
    ],
)
def test_interval_rule_lower_is_better(est, decision):
    assert classify_criterion(RMSEA, est).decision is decision


@pytest.mark.parametrize(
    "est,decision,action",
    [
        (Estimate(0.20, 0.15, 0.25), Decision.PASS, NextAction.NONE),
        (Estimate(0.12, 0.12, 0.12), Decision.INDETERMINATE, NextAction.FURTHER_SAMPLE),  # ambiguous band
        (Estimate(0.08, 0.02, 0.10), Decision.FAIL, NextAction.NONE),  # ≤ 0.10 falsifier
        (Estimate(0.10, 0.10, 0.10), Decision.FAIL, NextAction.NONE),
    ],
)
def test_k5_ambiguous_band(est, decision, action):
    out = classify_criterion(K5, est)
    assert (out.decision, out.next_action) == (decision, action)


def test_k8_excludes_zero():
    crit = K_GATES["K8"].criteria[0]
    assert classify_criterion(crit, Estimate(0.3, 0.1, 0.5)).decision is Decision.PASS
    assert classify_criterion(crit, Estimate(-0.3, -0.5, -0.1)).decision is Decision.PASS
    assert classify_criterion(crit, Estimate(0.1, -0.1, 0.3)).decision is Decision.INDETERMINATE


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        classify_criterion(K1, Estimate(0.7, 0.9, 0.5))


def test_not_evaluated_without_evidence():
    out = classify_gate("K1", {"K1.icc": Estimate(0.9, 0.85, 0.95)}, inputs=[])
    assert out.processing_status is ProcessingStatus.NOT_EVALUATED
    assert out.decision is None
    assert out.next_action is NextAction.COLLECT_EVIDENCE


def test_simulation_never_counts_as_empirical_confirmation():
    out = classify_gate("K1", {"K1.icc": Estimate(0.95, 0.9, 0.99)}, inputs=[synthetic()])
    assert out.processing_status is ProcessingStatus.NOT_EVALUATED
    assert "never counted as empirical confirmation" in out.rationale


def test_mixed_origin_inputs_not_evaluated():
    out = classify_gate("K1", {"K1.icc": Estimate(0.95, 0.9, 0.99)}, inputs=[empirical(), synthetic()])
    assert out.processing_status is ProcessingStatus.NOT_EVALUATED


def test_exploratory_inputs_not_evaluated():
    out = classify_gate(
        "K1", {"K1.icc": Estimate(0.95, 0.9, 0.99)}, inputs=[empirical(category=InferentialCategory.EXPLORATORY)]
    )
    assert out.processing_status is ProcessingStatus.NOT_EVALUATED


def test_missing_sub_criterion_not_evaluated():
    out = classify_gate("K4", {"K4.cfi": Estimate(0.95, 0.93, 0.97)}, inputs=[empirical()])
    assert out.processing_status is ProcessingStatus.NOT_EVALUATED
    assert "missing estimates" in out.rationale


def _k4(cross_hi=0.25):
    return {
        "K4.dbic": Estimate(15, 12, 18),
        "K4.loading": Estimate(0.55, 0.45, 0.65),
        "K4.cross": Estimate(0.2, 0.1, cross_hi) if cross_hi <= 0.3 else Estimate(0.35, 0.31, 0.4),
        "K4.cfi": Estimate(0.95, 0.93, 0.97),
        "K4.rmsea": Estimate(0.05, 0.04, 0.06),
        "K4.srmr": Estimate(0.04, 0.03, 0.05),
        "K4.items": Estimate(5, 5, 5),
    }


def test_k4_conjunction_pass_and_retire_items():
    assert classify_gate("K4", _k4(), inputs=[empirical()]).decision is Decision.PASS
    out = classify_gate("K4", _k4(cross_hi=0.4), inputs=[empirical()])
    assert out.decision is Decision.FAIL
    assert out.next_action is NextAction.RETIRE_ITEMS


def test_k7b_boolean_never_forced_fail():
    assert classify_gate("K7b", {"K7b.bounded": True}, inputs=[empirical()]).decision is Decision.PASS
    out = classify_gate("K7b", {"K7b.bounded": False}, inputs=[empirical()])
    assert out.decision is Decision.INDETERMINATE


def test_classification_is_reproducible():
    args = ("K5", {"K5.r": Estimate(0.12, 0.05, 0.19)})
    a = classify_gate(*args, inputs=[empirical()]).to_dict()
    b = classify_gate(*args, inputs=[empirical()]).to_dict()
    assert a == b


def test_catalog_covers_registered_gates():
    cat = catalog()
    ids = {g["gate_id"] for g in cat["gates"]}
    assert {"K1", "K2", "K3", "K4", "K4b", "K5", "K6", "K6b", "K7b", "K8"} == ids
    assert cat["approval_status"] == "PENDING_SCIENTIFIC_REVIEWER"
