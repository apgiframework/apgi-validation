"""TC-REL-01/02, TC-SCI-04 (terminology), S4-VAL-16 release package."""

import subprocess
import sys
from pathlib import Path

import pytest

from utils.release_certificate import (
    CertificationError,
    assert_deposit_allowed,
    create_certificate,
    sign_certificate,
)
from utils.validation_report import generate_signing_key, public_key_for

ROOT = Path(__file__).resolve().parent.parent


def _cert(tmp_path, approvers=None):
    art = tmp_path / "apgi_validation-1.3.0.tar.gz"
    art.write_bytes(b"release bytes")
    cert = create_certificate(
        version="1.3.0",
        source_commit="abcdef1",
        artifacts=[art],
        protocol_lock_sha256="c" * 64,
        test_evidence={"ci_run": "https://example.invalid/run/1"},
        approvers=approvers or {"release_authority": "A. Person", "scientific_reviewer": "B. Person"},
    )
    key = generate_signing_key()
    return art, sign_certificate(cert, key), public_key_for(key)


def test_certified_artifact_may_be_deposited(tmp_path):
    art, cert, pub = _cert(tmp_path)
    assert assert_deposit_allowed([art], cert, pub) == [art.name]


def test_modified_artifact_refused(tmp_path):
    art, cert, pub = _cert(tmp_path)
    art.write_bytes(b"changed")
    with pytest.raises(CertificationError, match="differs"):
        assert_deposit_allowed([art], cert, pub)


def test_uncertified_artifact_refused(tmp_path):
    _, cert, pub = _cert(tmp_path)
    other = tmp_path / "other.zip"
    other.write_bytes(b"x")
    with pytest.raises(CertificationError, match="not in the certified"):
        assert_deposit_allowed([other], cert, pub)


def test_pending_approval_refused(tmp_path):
    art, cert, pub = _cert(tmp_path, approvers={"release_authority": "PENDING-OWNER-APPROVAL"})
    with pytest.raises(CertificationError, match="approvals not recorded"):
        assert_deposit_allowed([art], cert, pub)


def test_forged_certificate_refused(tmp_path):
    art, cert, _ = _cert(tmp_path)
    with pytest.raises(CertificationError, match="signature"):
        assert_deposit_allowed([art], cert, public_key_for(generate_signing_key()))


def _run(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, capture_output=True, text=True
    )


def test_release_package_is_complete_and_consistent():
    result = _run("build_release_package.py", "--check")
    assert result.returncode == 0, result.stderr


def test_release_results_never_claim_evaluation_without_evidence():
    table = (ROOT / "release/results/tables/k-gate-status.csv").read_text()
    assert "PASS" not in table and "FAIL" not in table
    assert table.count("NOT_EVALUATED") == 10


def test_terminology_scan_clean():
    assert _run("scan_terminology.py").returncode == 0


def test_secret_content_scan_clean():
    assert _run("scan_secrets.py").returncode == 0


def test_citation_uses_canonical_expansion():
    text = (ROOT / "CITATION.cff").read_text()
    assert "Allostatic Precision-Gated Ignition" in text


def test_app_version_matches_pyproject():
    import tomllib

    from utils.versioning import APP_VERSION

    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"] == APP_VERSION
