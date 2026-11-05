"""TC-DAT-04, TC-VAL-01, TC-API-02: provenance labels, signed reports, ProtocolResult contract."""

import copy

import pytest
from pydantic import ValidationError

from utils.data_provenance import (
    DataOrigin,
    InferentialCategory,
    ProcessingStage,
    ProvenanceError,
    ProvenanceRecord,
    ProvenanceRegistry,
    ResearchStoreGuard,
    ResearchStoreRejection,
    TransformStep,
    file_sha256,
    inherited_origin,
    verify_record_bytes,
)
from utils.kgate import Estimate, classify_gate
from utils.protocol_schema import PredictionResult, ProtocolResult, ResultContractError
from utils.validation_report import (
    ReportContractError,
    SignatureError,
    build_report,
    generate_signing_key,
    public_key_for,
    sign_report,
    validate_report,
    verify_report,
)

SHA = "b" * 64
STEP = TransformStep(step="filter", tool="mne", tool_version="1.7.1")


def raw(rid, origin=DataOrigin.HUMAN_OBSERVED, **kw):
    return ProvenanceRecord(
        record_id=rid,
        data_origin=origin,
        processing_stage=ProcessingStage.RAW,
        inferential_category=kw.pop("category", InferentialCategory.CONFIRMATORY),
        **kw,
    )


def test_only_registered_origins_accepted():
    assert {o.value for o in DataOrigin} == {
        "human_observed",
        "synthetic",
        "replay",
        "imported_empirical",
        "demonstration",
    }
    with pytest.raises(ValidationError):
        ProvenanceRecord(
            record_id="x",
            data_origin="synthetic_realistic",
            processing_stage="raw",
            inferential_category="confirmatory",
        )


def test_labels_are_mandatory():
    with pytest.raises(ValidationError):
        ProvenanceRecord(record_id="x", processing_stage="raw", inferential_category="descriptive")
    with pytest.raises(ValidationError):
        ProvenanceRecord(record_id="x", data_origin="synthetic", processing_stage="raw")


def test_imported_empirical_requires_full_provenance():
    with pytest.raises(ValidationError, match="license"):
        raw(
            "i1",
            DataOrigin.IMPORTED_EMPIRICAL,
            source="openneuro",
            version="1.0",
            sha256=SHA,
            acquisition_metadata={"sfreq": 500},
        )
    ok = raw(
        "i1",
        DataOrigin.IMPORTED_EMPIRICAL,
        source="openneuro:ds000001",
        version="1.0",
        sha256=SHA,
        license="CC0",
        acquisition_metadata={"sfreq": 500},
    )
    assert ok.is_empirical


def test_derived_requires_parents_and_history():
    with pytest.raises(ValidationError):
        ProvenanceRecord(
            record_id="d", data_origin="human_observed", processing_stage="derived", inferential_category="confirmatory"
        )


def test_origin_inheritance():
    assert inherited_origin([DataOrigin.HUMAN_OBSERVED, DataOrigin.IMPORTED_EMPIRICAL]) is DataOrigin.IMPORTED_EMPIRICAL
    assert inherited_origin([DataOrigin.HUMAN_OBSERVED, DataOrigin.SYNTHETIC]) is DataOrigin.SYNTHETIC
    reg = ProvenanceRegistry()
    reg.register(raw("h1"))
    reg.register(raw("s1", DataOrigin.SYNTHETIC, category=InferentialCategory.METHOD_VALIDATION))
    with pytest.raises(ProvenanceError, match="inherited origin synthetic"):
        reg.register(
            ProvenanceRecord(
                record_id="d1",
                data_origin="human_observed",
                processing_stage="derived",
                inferential_category="confirmatory",
                parents=["h1", "s1"],
                transform_history=[STEP],
            )
        )
    reg.register(
        ProvenanceRecord(
            record_id="d1",
            data_origin="synthetic",
            processing_stage="derived",
            inferential_category="exploratory",
            parents=["h1", "s1"],
            transform_history=[STEP],
        )
    )
    assert reg.lineage("d1") == ["s1", "h1"] or set(reg.lineage("d1")) == {"h1", "s1"}


def test_unknown_parent_rejected():
    reg = ProvenanceRegistry()
    with pytest.raises(ProvenanceError, match="unknown parent"):
        reg.register(
            ProvenanceRecord(
                record_id="d",
                data_origin="human_observed",
                processing_stage="derived",
                inferential_category="confirmatory",
                parents=["nope"],
                transform_history=[STEP],
            )
        )


@pytest.mark.parametrize("origin", [DataOrigin.SYNTHETIC, DataOrigin.REPLAY, DataOrigin.DEMONSTRATION])
def test_research_store_rejects_non_empirical(origin):
    guard = ResearchStoreGuard()
    with pytest.raises(ResearchStoreRejection):
        guard.admit(raw("r", origin, category=InferentialCategory.METHOD_VALIDATION))
    assert guard.rejections and not guard.admitted


@pytest.mark.parametrize("system", ["appendix_simulator", "explorer_pre_research"])
def test_research_store_bars_appendix_and_explorer(system):
    with pytest.raises(ResearchStoreRejection, match="barred"):
        ResearchStoreGuard().admit(raw("r", source_system=system))


def test_checksum_verification(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"abc")
    rec = raw("f", sha256=file_sha256(f))
    verify_record_bytes(rec, f)
    f.write_bytes(b"abd")
    with pytest.raises(ProvenanceError, match="checksum mismatch"):
        verify_record_bytes(rec, f)


# ---------------------------------------------------------------------------
# Reports and signatures (S4-VAL-09, S4-VAL-11.1)
# ---------------------------------------------------------------------------


def _report(inputs=None, est=(0.8, 0.72, 0.88)):
    inputs = inputs or [raw("h1", source="site", version="1", sha256=SHA, license="CC-BY")]
    cls = classify_gate("K1", {"K1.icc": Estimate(*est)}, inputs=inputs)
    return build_report(
        cls,
        inputs=inputs,
        protocol_lock_sha256="c" * 64,
        software_version="1.3.0",
        source_commit="abcdef1",
        seed=7,
        n=120,
    )


def test_report_contract_valid_and_signed_roundtrip():
    report = _report()
    key = generate_signing_key()
    signed = sign_report(report, key, signer="test-reviewer")
    verify_report(signed, public_key_for(key))


def test_tampered_report_fails_signature():
    key = generate_signing_key()
    signed = sign_report(_report(), key, signer="r")
    tampered = copy.deepcopy(signed)
    tampered["classification"]["decision"] = "FAIL"
    with pytest.raises(SignatureError):
        verify_report(tampered, public_key_for(key))


def test_untrusted_key_rejected():
    signed = sign_report(_report(), generate_signing_key(), signer="r")
    with pytest.raises(SignatureError, match="untrusted"):
        verify_report(signed, public_key_for(generate_signing_key()))


def test_incomplete_report_rejected():
    report = _report()
    del report["protocol_lock_sha256"]
    with pytest.raises(ReportContractError):
        validate_report(report)


def test_not_evaluated_report_cannot_carry_decision():
    report = _report(inputs=[raw("s", DataOrigin.SYNTHETIC, category=InferentialCategory.METHOD_VALIDATION)])
    assert report["classification"]["processing_status"] == "NOT_EVALUATED"
    report["classification"]["decision"] = "PASS"
    with pytest.raises(ReportContractError):
        validate_report(report)


# ---------------------------------------------------------------------------
# ProtocolResult (APGI-API-004, APGI-INT-001, APGI-DAT-004)
# ---------------------------------------------------------------------------


def test_empty_result_is_not_success():
    r = ProtocolResult(protocol_id="P", named_predictions={}, completion_percentage=0)
    assert r.status == "partial"
    assert any("incomplete" in e for e in r.errors)


def test_explicit_incomplete_success_rejected():
    with pytest.raises(ValidationError):
        ProtocolResult(protocol_id="P", named_predictions={}, completion_percentage=0, status="success")


def test_complete_result_may_succeed():
    r = ProtocolResult(
        protocol_id="P", named_predictions={"a": PredictionResult(passed=True)}, completion_percentage=100
    )
    assert r.status == "success"


def test_versions_present_and_exchange_requires_provenance():
    r = ProtocolResult(
        protocol_id="P", named_predictions={"a": PredictionResult(passed=True)}, completion_percentage=100
    )
    assert r.schema_version and r.producer_version
    with pytest.raises(ResultContractError, match="protocol_version"):
        r.assert_exchangeable()
    r2 = r.model_copy(
        update={
            "protocol_version": "1.1.0",
            "dataset_provenance": [
                {"source": "s", "version": "1", "sha256": SHA, "license": "CC0", "processing_history": ["raw"]}
            ],
        }
    )
    r2.assert_exchangeable()
    r3 = r.model_copy(update={"protocol_version": "1.1.0", "dataset_provenance": [{"source": "s"}]})
    with pytest.raises(ResultContractError, match="missing"):
        r3.assert_exchangeable()
