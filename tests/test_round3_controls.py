"""Round-3 register controls: deviations, encryption, retention, acquisition, health, RNG, votes."""

import os
import time

import numpy as np
import pytest

from utils.data_provenance import DataOrigin, InferentialCategory, ProcessingStage, ProvenanceRecord
from utils.deviation_register import DeviationKind, compare_to_plan, flag_result
from utils.kgate import Estimate, ProcessingStatus, classify_gate
from utils.protocol_schema import PredictionResult, ProtocolResult, ResultContractError

SHA = "e" * 64


def _emp():
    return ProvenanceRecord(
        record_id="h",
        data_origin=DataOrigin.HUMAN_OBSERVED,
        processing_stage=ProcessingStage.RAW,
        inferential_category=InferentialCategory.CONFIRMATORY,
        source="s",
        version="1",
        sha256=SHA,
        license="CC0",
    )


# --- APGI-REP-004 -----------------------------------------------------------


def test_absent_step_is_detected_and_propagates():
    devs = compare_to_plan(["load", "filter", "epoch", "fit"], ["load", "epoch", "fit"], {"hp": 0.1}, {"hp": 0.5})
    kinds = {d.kind for d in devs}
    assert DeviationKind.OMITTED_STEP in kinds and DeviationKind.PARAMETER in kinds

    result = ProtocolResult(
        protocol_id="P", named_predictions={"a": PredictionResult(passed=True)}, completion_percentage=100
    )
    flag_result(result, devs)
    dumped = result.model_dump()
    assert dumped["metadata"]["inferential_category"] == "exploratory"
    assert dumped["named_predictions"]["a"]["metadata"]["framework_vote_eligible"] is False

    gate = classify_gate("K1", {"K1.icc": Estimate(0.9, 0.85, 0.95)}, inputs=[_emp()], deviations=devs)
    assert gate.processing_status is ProcessingStatus.NOT_EVALUATED
    assert "deviation" in gate.rationale


def test_no_deviation_no_flags():
    assert compare_to_plan(["a", "b"], ["a", "b"], {"x": 1}, {"x": 1}) == []


def test_report_rejects_decision_with_deviations():
    from utils.validation_report import ReportContractError, build_report, validate_report

    devs = compare_to_plan(["a"], [])
    cls = classify_gate("K1", {"K1.icc": Estimate(0.9, 0.85, 0.95)}, inputs=[_emp()])
    with pytest.raises(ReportContractError, match="deviations"):
        build_report(
            cls,
            inputs=[_emp()],
            protocol_lock_sha256="c" * 64,
            software_version="1.3.0",
            source_commit="abcdef1",
            seed=1,
            deviations=devs,
        )
    ok = build_report(
        cls,
        inputs=[_emp()],
        protocol_lock_sha256="c" * 64,
        software_version="1.3.0",
        source_commit="abcdef1",
        seed=1,
        parameter_names=["APGISettings.theta0"],
    )
    assert ok["parameter_provenance"]["parameters"]["APGISettings.theta0"]["label"] == "assumed"
    validate_report(ok)


# --- APGI-INT-001 -----------------------------------------------------------


def test_export_for_exchange_enforces_contract():
    r = ProtocolResult(
        protocol_id="P", named_predictions={"a": PredictionResult(passed=True)}, completion_percentage=100
    )
    with pytest.raises(ResultContractError):
        r.export_for_exchange()


# --- APGI-API-003 -----------------------------------------------------------


def test_payload_encryption_roundtrip_and_tamper(tmp_path):
    from utils.payload_encryption import PayloadEncryptionError, decrypt_file, encrypt_file

    key = os.urandom(32)
    f = tmp_path / "backup.tar"
    f.write_bytes(b"participant data")
    enc = encrypt_file(f, key=key)
    assert not f.exists() and b"participant data" not in enc.read_bytes()
    assert decrypt_file(enc, key=key).read_bytes() == b"participant data"
    blob = bytearray(enc.read_bytes())
    blob[-1] ^= 1
    enc.write_bytes(bytes(blob))
    with pytest.raises(PayloadEncryptionError):
        decrypt_file(enc, key=key)


def test_renamed_ciphertext_rejected(tmp_path):
    from utils.payload_encryption import PayloadEncryptionError, decrypt_file, encrypt_file

    key = os.urandom(32)
    f = tmp_path / "a.zip"
    f.write_bytes(b"x")
    enc = encrypt_file(f, key=key)
    moved = enc.rename(tmp_path / "b.zip.apgienc")
    with pytest.raises(PayloadEncryptionError):
        decrypt_file(moved, key=key)


# --- APGI-DAT-003 -----------------------------------------------------------


def test_retention_skips_pending_and_lineage_classes(tmp_path):
    from utils.retention_policy import enforce

    for d in ("synth", "human", "imported"):
        (tmp_path / d).mkdir()
        p = tmp_path / d / "f.bin"
        p.write_bytes(b"x")
        old = time.time() - 400 * 86400
        os.utime(p, (old, old))
    policy = {
        "classes": {
            "synthetic": {"paths": ["synth"], "retention_days": 365, "deletion": "delete_file"},
            "human_observed": {"paths": ["human"], "retention_days": "PENDING", "deletion": "lineage_withdrawal"},
            "imported": {"paths": ["imported"], "retention_days": 30, "deletion": "lineage_withdrawal"},
        }
    }
    dry = {a.data_class: a.action for a in enforce(policy, root=tmp_path, dry_run=True)}
    assert dry == {"synthetic": "would_delete", "human_observed": "skipped_pending", "imported": "skipped_lineage"}
    assert (tmp_path / "synth" / "f.bin").exists()
    enforce(policy, root=tmp_path, dry_run=False)
    assert not (tmp_path / "synth" / "f.bin").exists()
    assert (tmp_path / "human" / "f.bin").exists() and (tmp_path / "imported" / "f.bin").exists()


# --- S4-VAL-12.2 ------------------------------------------------------------


def test_acquisition_checks_detect_faults():
    from utils.acquisition_checks import check_recording

    fs = 100.0
    t = np.arange(1000) / fs
    x = np.sin(t)
    assert check_recording(t, x, fs).passed

    t_drop = np.delete(t, np.s_[100:110])
    rep = check_recording(t_drop, np.delete(x, np.s_[100:110]), fs)
    assert "dropouts" in rep.failures()

    x_nan = x.copy()
    x_nan[5] = np.nan
    assert "nonfinite" in check_recording(t, x_nan, fs).failures()

    t_rev = t.copy()
    t_rev[50], t_rev[51] = t_rev[51], t_rev[50]
    assert "monotonic" in check_recording(t_rev, x, fs).failures()

    rep = check_recording(t, x, fs, raw_bytes=b"abc", expected_sha256="0" * 64)
    assert "checksum" in rep.failures()


# --- APGI-IMP-003 -----------------------------------------------------------


def test_health_check_raises_alerts_for_induced_faults(caplog):
    from utils.health_check import HealthResult, run_health_checks

    def full_disk():
        return HealthResult("disk_free", "fail", 0.1)

    def broken():
        raise OSError("audit volume read-only")

    with caplog.at_level("WARNING", logger="apgi.health"):
        report = run_health_checks(
            {"disk_free": full_disk, "audit_log_writable": broken, "ok": lambda: HealthResult("ok", "ok", 1)}
        )
    assert report["status"] == "fail"
    alerts = [r for r in caplog.records if "apgi_alert" in r.getMessage()]
    assert len(alerts) == 2


# --- APGI-REP-002 -----------------------------------------------------------


def test_measurement_noise_uses_supplied_stream():
    from apgi_core.equations import MeasurementEquations

    a = [MeasurementEquations.compute_HEP(5.0, 2.0, 1.0, rng=np.random.default_rng(7)) for _ in range(2)]
    assert a[0] == a[1]
    b = MeasurementEquations.compute_HEP(5.0, 2.0, 1.0, rng=np.random.default_rng(8))
    assert b != a[0]


# --- APGI-VAL-001 -----------------------------------------------------------


def test_only_confirmatory_predictions_carry_framework_votes():
    from utils.protocol_loader import load_all_protocols

    specs = load_all_protocols()
    for spec in specs.values():
        for pred in spec.sub_predictions:
            assert pred.framework_vote_eligible == (pred.status == "confirmatory")
    p6 = specs["APGI-P06"]
    assert all(not p.framework_vote_eligible for p in p6.sub_predictions)


# --- APGI-INT-003 / S4-VAL-12.4 --------------------------------------------


def test_data_quality_is_measured_not_assumed():
    import pandas as pd

    from utils.data_quality_assessment import DataQualityAssessor

    df = pd.DataFrame({"time": [0, 1, 1, 0.5], "v": [1.0, np.inf, 2.0, 3.0]})
    m = DataQualityAssessor().assess_dataset_quality(df)
    assert m["consistency"] < 1.0 and m["validity"] < 1.0
    assert m["consistency"] not in (0.8,) and m["validity"] not in (0.9,)


def test_default_runtime_gates_do_not_pass_without_measurement():
    from utils.performance_gates import GateStatus, PerformanceGateEnforcer, get_default_gates

    enforcer = PerformanceGateEnforcer()
    for g in get_default_gates():
        enforcer.register_gate(g)
    statuses = enforcer.enforce_all()
    assert statuses["protocol_runtime"] is GateStatus.FAILED
    assert enforcer.results["protocol_memory"]["value"] > 0
