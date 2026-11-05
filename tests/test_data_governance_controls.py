"""TC-CON-01/02, TC-DAT-02, TC-SEC-01, TC-API-02 controls."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from utils.consent_gate import (
    ConsentError,
    ConsentGate,
    ConsentRecord,
    ConsentRegistry,
    ConsentStatus,
    InMemoryStore,
    Operation,
    WithdrawalOrchestrator,
)
from utils.data_provenance import ProvenanceRecord, ProvenanceRegistry, TransformStep
from utils.data_quarantine import Outcome, QuarantineLedger
from utils.runtime_profile import InsecureConfigurationError, auth_enforced, ephemeral_master_key_allowed

ALL_OPS = {Operation.INGEST, Operation.PROCESS, Operation.EXPORT}


def _consents():
    reg = ConsentRegistry()
    reg.record(ConsentRecord("p1", ConsentStatus.ACTIVE, ALL_OPS, {"apgi-validation"}))
    reg.record(ConsentRecord("p2", ConsentStatus.ACTIVE, {Operation.INGEST}, {"apgi-validation"}))
    reg.record(
        ConsentRecord(
            "p3",
            ConsentStatus.ACTIVE,
            ALL_OPS,
            {"apgi-validation"},
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    return reg


@pytest.mark.parametrize(
    "pid,op,purpose,reason",
    [
        ("absent", Operation.INGEST, "apgi-validation", "no consent record"),
        ("p2", Operation.EXPORT, "apgi-validation", "outside consented scope"),
        ("p1", Operation.PROCESS, "marketing", "outside consented purposes"),
        ("p3", Operation.INGEST, "apgi-validation", "expired"),
    ],
)
def test_consent_blocks_and_nothing_persists(pid, op, purpose, reason):
    gate = ConsentGate(_consents())
    store = InMemoryStore("raw")
    rec = ProvenanceRecord(
        record_id="r1",
        data_origin="human_observed",
        processing_stage="raw",
        inferential_category="confirmatory",
        participant_ids=[pid],
    )
    with pytest.raises(ConsentError, match=reason):
        gate.guarded(op, rec, purpose, lambda: store.put("r1", {}))
    assert not store.contains("r1")
    assert gate.audit[-1]["outcome"] == "blocked"
    assert pid not in str(gate.audit[-1])  # pseudonymised


def test_withdrawn_consent_blocks():
    reg = _consents()
    reg.withdraw("p1")
    with pytest.raises(ConsentError, match="withdrawn"):
        ConsentGate(reg).check(Operation.PROCESS, ["p1"], "apgi-validation")


def test_permitted_operation_runs():
    store = InMemoryStore("raw")
    rec = ProvenanceRecord(
        record_id="r1",
        data_origin="human_observed",
        processing_stage="raw",
        inferential_category="confirmatory",
        participant_ids=["p1"],
    )
    ConsentGate(_consents()).guarded(Operation.INGEST, rec, "apgi-validation", lambda: store.put("r1", {}))
    assert store.contains("r1")


def test_withdrawal_deletes_lineage_across_stores():
    prov = ProvenanceRegistry()
    step = TransformStep(step="epoch", tool="mne", tool_version="1.7.1")
    prov.register(
        ProvenanceRecord(
            record_id="raw1",
            data_origin="human_observed",
            processing_stage="raw",
            inferential_category="confirmatory",
            participant_ids=["p1"],
        )
    )
    prov.register(
        ProvenanceRecord(
            record_id="raw2",
            data_origin="human_observed",
            processing_stage="raw",
            inferential_category="confirmatory",
            participant_ids=["p2"],
        )
    )
    prov.register(
        ProvenanceRecord(
            record_id="feat1",
            data_origin="human_observed",
            processing_stage="derived",
            inferential_category="confirmatory",
            parents=["raw1"],
            transform_history=[step],
        )
    )
    prov.register(
        ProvenanceRecord(
            record_id="agg",
            data_origin="human_observed",
            processing_stage="derived",
            inferential_category="confirmatory",
            parents=["feat1", "raw2"],
            transform_history=[step],
        )
    )
    stores = [InMemoryStore("raw"), InMemoryStore("processed"), InMemoryStore("cache")]
    for s in stores:
        for rid in ("raw1", "raw2", "feat1", "agg"):
            s.put(rid, {})
    consents = _consents()
    record = WithdrawalOrchestrator(prov, stores, consents).withdraw("p1")
    assert set(record.deleted_record_ids) == {"raw1", "feat1", "agg"}
    for s in stores:
        assert not any(s.contains(r) for r in ("raw1", "feat1", "agg"))
        assert s.contains("raw2")
    assert record.backup_disposition == "PENDING_APPROVED_BACKUP_POLICY"
    assert consents.get("p1").status is ConsentStatus.WITHDRAWN


def test_quarantine_reconciles_every_record():
    notify = MagicMock()
    ledger = QuarantineLedger(
        required_fields=("record_id", "session"),
        known_keys={"s1"},
        match_field="session",
        window_end=datetime(2026, 1, 1, tzinfo=timezone.utc),
        notify=notify,
    )
    ledger.ingest(
        [
            {"record_id": "a", "session": "s1", "received_at": "2025-12-31T00:00:00+00:00"},
            {"record_id": "a", "session": "s1", "received_at": "2025-12-31T00:00:00+00:00"},
            {"record_id": "b"},
            "not-an-object",
            {"record_id": "c", "session": "s1", "received_at": "2026-02-01T00:00:00+00:00"},
            {"record_id": "d", "session": "zz", "received_at": "2025-12-30T00:00:00+00:00"},
        ]
    )
    summary = ledger.reconcile()
    assert summary == {"received": 6, "accepted": 1, "malformed": 2, "duplicate": 1, "delayed": 1, "unmatched": 1}
    assert notify.call_count == 5
    assert {q["outcome"] for q in ledger.quarantine} == {o.value for o in Outcome} - {"accepted"}


def test_validator_exception_is_captured_not_suppressed():
    def bad(record):
        raise ValueError("bad units")

    ledger = QuarantineLedger(validator=bad)
    (disp,) = ledger.ingest([{"record_id": "x"}])
    assert disp.outcome is Outcome.MALFORMED and "bad units" in disp.reason


# ---------------------------------------------------------------------------
# APGI-IMP-002 secure defaults
# ---------------------------------------------------------------------------


def test_auth_enforced_by_default(monkeypatch):
    monkeypatch.delenv("APGI_ENFORCE_AUTH", raising=False)
    monkeypatch.delenv("APGI_ENV", raising=False)
    assert auth_enforced() is True


def test_dev_auth_opt_out_refused_in_production(monkeypatch):
    monkeypatch.setenv("APGI_ENFORCE_AUTH", "0")
    monkeypatch.setenv("APGI_ENV", "production")
    with pytest.raises(InsecureConfigurationError):
        auth_enforced()


def test_ephemeral_key_refused_in_production(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")
    monkeypatch.setenv("APGI_ENV", "production")
    with pytest.raises(InsecureConfigurationError):
        ephemeral_master_key_allowed()


def test_secret_policy_exits_on_ephemeral_in_production(monkeypatch):
    from utils.secret_policy_enforcer import enforce_secret_policy

    monkeypatch.delenv("APGI_MASTER_KEY", raising=False)
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")
    monkeypatch.setenv("APGI_ENV", "production")
    with pytest.raises(SystemExit):
        enforce_secret_policy()


def test_headless_requires_token_when_enforced(monkeypatch):
    from utils.runtime_profile import require_entry_point_auth

    monkeypatch.delenv("APGI_ENFORCE_AUTH", raising=False)
    with pytest.raises(SystemExit, match="Authentication required"):
        require_entry_point_auth(None, "Validation_GUI.py --headless")


def test_api_error_does_not_echo_response_body():
    from utils.apgi_api_client import ApgiApiClient, ApgiApiError

    session = MagicMock()
    resp = MagicMock(status_code=500, content=b"x", text="secret-participant-data token=abc")
    resp.json.return_value = {"code": "internal_error", "detail": "secret-participant-data"}
    session.request.return_value = resp
    client = ApgiApiClient(base_url="https://api.example.org", session=session)
    with pytest.raises(ApgiApiError) as info:
        client.get("x")
    assert "secret-participant-data" not in str(info.value)
    assert info.value.error_code == "internal_error"
