"""TC-PRO-02 / TC-REP-01: registered protocol lock and strict loading.

Covers APGI-PRO-002, APGI-REP-001, APGI-VAL-001 and the XC-CI-05.1
"registered rules" CI gate.
"""

import json
import shutil
from pathlib import Path

import pytest

from utils import protocol_loader
from utils.protocol_loader import (
    LOAD_AUDIT_LOG,
    ProtocolLockError,
    ProtocolSchemaError,
    load_all_protocols,
    load_protocol_file,
    validate_protocol_document,
)
from utils.protocol_lock import LOCK_AUDIT_LOG, verify_protocol_lock, write_trust_anchor

ROOT = Path(__file__).resolve().parent.parent
PROTOCOLS = ROOT / "protocols"


@pytest.fixture
def release_copy(tmp_path):
    folder = tmp_path / "protocols"
    shutil.copytree(PROTOCOLS, folder)
    anchor = tmp_path / "protocol_lock.json"
    write_trust_anchor(folder, anchor)
    return folder, anchor


@pytest.fixture(autouse=True)
def _no_dev_bypass(monkeypatch):
    monkeypatch.delenv("APGI_SKIP_PROTOCOL_LOCK", raising=False)
    monkeypatch.delenv("APGI_ENV", raising=False)


def test_repository_lock_verifies():
    lock = verify_protocol_lock()
    assert lock.release == "1.1.0"
    assert len([n for n in lock.digests if n.startswith("protocol_")]) == 8
    assert LOCK_AUDIT_LOG[-1].outcome == "verified"


def test_load_all_returns_complete_set():
    specs = load_all_protocols()
    assert set(specs) == {f"APGI-P0{n}" for n in range(8)}


def test_modified_definition_fails_whole_run(release_copy):
    folder, anchor = release_copy
    target = folder / "protocol_1_cardiac_eeg.json"
    data = json.loads(target.read_text())
    data["title"] = data["title"] + " (edited)"
    target.write_text(json.dumps(data, indent=2))
    with pytest.raises(ProtocolLockError, match="SHA-256 mismatch: protocol_1"):
        verify_protocol_lock(folder, anchor)
    assert LOCK_AUDIT_LOG[-1].outcome == "rejected"


def test_manifest_and_file_edited_together_is_detected(release_copy):
    folder, anchor = release_copy
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    (folder / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ProtocolLockError, match="trust anchor"):
        verify_protocol_lock(folder, anchor)


def test_missing_definition_fails(release_copy):
    folder, anchor = release_copy
    (folder / "protocol_7_doc_biomarker.json").unlink()
    with pytest.raises(ProtocolLockError, match="locked file missing"):
        verify_protocol_lock(folder, anchor)


def test_unlisted_definition_fails(release_copy):
    folder, anchor = release_copy
    shutil.copy(folder / "protocol_0_hep_proxy_validation.json", folder / "protocol_9_extra.json")
    with pytest.raises(ProtocolLockError, match="unlocked protocol definition"):
        verify_protocol_lock(folder, anchor)


def test_missing_trust_anchor_fails(release_copy, tmp_path):
    folder, _ = release_copy
    with pytest.raises(ProtocolLockError, match="trust anchor"):
        verify_protocol_lock(folder, tmp_path / "absent.json")


def test_modified_copy_outside_release_is_refused(tmp_path):
    src = PROTOCOLS / "protocol_2_somatic_agent_sim.json"
    copy = tmp_path / src.name
    data = json.loads(src.read_text())
    data["description"] = "tampered"
    copy.write_text(json.dumps(data))
    with pytest.raises(ProtocolLockError, match="do not match any locked"):
        load_protocol_file(copy)
    assert LOAD_AUDIT_LOG[-1]["outcome"] == "rejected"


def test_id_only_document_rejected():
    with pytest.raises(ProtocolSchemaError, match="schema validation failed"):
        validate_protocol_document({"protocol_id": "APGI-P01"}, "id_only.json")


def test_wrong_version_rejected():
    data = json.loads((PROTOCOLS / "protocol_1_cardiac_eeg.json").read_text())
    data["version"] = "9.9.9"
    with pytest.raises(ProtocolSchemaError):
        validate_protocol_document(data, "wrong_version.json")


def test_load_all_has_no_partial_state(monkeypatch, release_copy):
    """A single bad file aborts the run; nothing is returned or cached."""
    folder, anchor = release_copy
    (folder / "protocol_3_anticipation_fmri.json").write_text("{}")
    monkeypatch.setattr(protocol_loader, "_PROTOCOLS_DIR", folder)
    monkeypatch.setattr(
        protocol_loader, "verify_protocol_lock", lambda d=None, a=None: verify_protocol_lock(folder, anchor)
    )
    result = None
    with pytest.raises(ProtocolLockError):
        result = load_all_protocols()
    assert result is None


def test_dev_bypass_refused_in_production(monkeypatch):
    monkeypatch.setenv("APGI_SKIP_PROTOCOL_LOCK", "1")
    monkeypatch.setenv("APGI_ENV", "production")
    with pytest.raises(ProtocolLockError, match="not permitted"):
        verify_protocol_lock()


def test_dev_bypass_is_audited(monkeypatch):
    monkeypatch.setenv("APGI_SKIP_PROTOCOL_LOCK", "1")
    lock = verify_protocol_lock()
    assert lock.release == "UNVERIFIED-DEV"
    assert LOCK_AUDIT_LOG[-1].outcome == "skipped_dev"
