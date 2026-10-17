"""Release validation and mutation regressions for APGI protocol specification 1.1."""

import copy
import importlib.util
import json
import pathlib
import shutil

import jsonschema
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("validate_protocols", ROOT / "scripts/validate_protocols.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
PROTOCOLS = ROOT / "protocols"


def protocol(number):
    return validator.read_json(next(PROTOCOLS.glob(f"protocol_{number}_*.json")))


@pytest.fixture
def release(tmp_path):
    target = tmp_path / "release"
    shutil.copytree(PROTOCOLS, target)
    return target


@pytest.fixture
def schema_validator():
    schema = validator.read_json(PROTOCOLS / "schemas/protocol.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_release():
    assert validator.validate_release(PROTOCOLS) == 0


def test_main():
    with pytest.raises(SystemExit) as exc:
        validator.main([])
    assert exc.value.code == 0


def test_all_prediction_membership(schema_validator):
    for n, count in enumerate([3, 6, 7, 4, 3, 5, 5, 7]):
        data = protocol(n)
        schema_validator.validate(data)
        assert [x["id"] for x in data["sub_predictions"]] == [f"Pred {n}.{chr(65 + j)}" for j in range(count)]
        duplicate = copy.deepcopy(data)
        duplicate["sub_predictions"][-1] = duplicate["sub_predictions"][0]
        assert not schema_validator.is_valid(duplicate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "1.0.0"),
        ("schema_version", "2.0.0"),
        ("registry_ep", "EP-7"),
        ("last_updated", "2026-02-30"),
    ],
)
def test_schema_rejects_wrong_version_identity_dates(schema_validator, field, value):
    data = protocol(1)
    data[field] = value
    assert not schema_validator.is_valid(data)


@pytest.mark.parametrize(
    "status,value,valid",
    [
        ("uncalibrated", None, True),
        ("uncalibrated", 100, False),
        ("calibrated", None, False),
        ("calibrated", 2, True),
        ("calibrated", 0, False),
    ],
)
def test_kappa_calibration(schema_validator, status, value, valid):
    data = protocol(1)
    data["apgi_parameters"]["kappa"].update(status=status, value=value)
    assert schema_validator.is_valid(data) is valid


@pytest.mark.parametrize("status", ["pending", "not_yet_submitted"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("enrollment_authorized", True),
        ("approval_date", "2026-10-16"),
        ("protocol_number", "GRANTED-123"),
    ],
)
def test_pending_approval_cannot_look_granted(schema_validator, status, field, value):
    data = protocol(1)
    data["ethics"]["approval_status"] = status
    data["ethics"][field] = value
    assert not schema_validator.is_valid(data)


def test_readiness_and_reduced_quantities(schema_validator):
    data = protocol(1)
    data["readiness"]["registration_verified"] = "false"
    assert not schema_validator.is_valid(data)
    data = protocol(1)
    data["apgi_parameters"]["reduced_quantities"][0] = "legacy_alpha"
    assert not schema_validator.is_valid(data)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("participants", "n_minimum", 1000),
        ("participants", "n_analyzable_target", 1000),
        ("blocks", "total_experimental_trials", 100),
        ("power_analysis", "n_target", 100),
    ],
)
def test_cross_field_constraints(section, field, value):
    data = protocol(1)
    data[section][field] = value
    assert validator.semantic_errors(data)


def test_clinical_group_totals():
    data = protocol(7)
    assert not validator.semantic_errors(data)
    data["participants"]["groups"]["MCS"] += 1
    errors = validator.semantic_errors(data)
    assert any("cohort group sum" in e for e in errors)
    assert any("patient cohort sum" in e for e in errors)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', "{broken"])
def test_invalid_json_is_reported(tmp_path, text, capsys):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}')
    instance = tmp_path / "protocol_test.json"
    instance.write_text(text)
    assert validator.validate_protocols(schema, [instance]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_empty_and_missing_inputs_fail(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}')
    assert validator.validate_protocols(schema, []) == 1
    assert validator.validate_protocols(schema, [tmp_path / "missing.json"]) == 1
    assert validator.validate_protocols(tmp_path / "missing-schema.json", []) == 1
    schema.write_text('{"type": "bogus"}')
    assert validator.validate_protocols(schema, []) == 1


def test_discovery_is_dynamic_and_excludes_metadata(release):
    assert validator.validate_protocols(release / "schemas/protocol.schema.json") == 0
    (release / "protocol_broken.json").write_text("{}")
    assert validator.validate_protocols(release / "schemas/protocol.schema.json") == 1


@pytest.mark.parametrize("action", ["missing", "duplicate", "renamed"])
def test_release_membership(release, action):
    file = next(release.glob("protocol_1_*.json"))
    if action == "missing":
        file.unlink()
    elif action == "duplicate":
        shutil.copyfile(file, release / "protocol_extra.json")
    else:
        file.rename(release / "protocol_9_wrong.json")
    assert validator.validate_release(release, refresh_metadata=True) == 1


def test_manifest_detects_tampering(release):
    (release / "CHANGELOG.md").write_text("Changed without refreshing checksum")
    assert validator.validate_release(release) == 1
    assert validator.validate_release(release, refresh_metadata=True) == 0
    assert validator.validate_release(release) == 0


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "escape", "bad_hash", "version"])
def test_manifest_rejects_invalid_entries(release, mutation):
    path = release / "MANIFEST.json"
    data = validator.read_json(path)
    if mutation == "duplicate":
        data["files"].append(data["files"][0])
    elif mutation == "missing":
        data["files"].pop()
    elif mutation == "escape":
        data["files"][0]["path"] = "../escape"
    elif mutation == "bad_hash":
        data["files"][0]["sha256"] = "0" * 64
    else:
        data["release"] = "1.0.0"
    write(path, data)
    assert validator.manifest_errors(release)


def test_report_is_recomputed_not_trusted(release):
    path = release / "VALIDATION.json"
    data = validator.read_json(path)
    data["total_predictions"] = 41
    write(path, data)
    assert validator.validate_release(release) == 1
    assert validator.validate_release(release, refresh_metadata=True) == 0
    first = (release / "MANIFEST.json").read_bytes()
    assert validator.validate_release(release, refresh_metadata=True) == 0
    assert (release / "MANIFEST.json").read_bytes() == first


def test_failed_validation_does_not_refresh_metadata(release):
    before = (release / "MANIFEST.json").read_bytes()
    file = next(release.glob("protocol_1_*.json"))
    data = validator.read_json(file)
    data["version"] = "1.0.0"
    write(file, data)
    assert validator.validate_release(release, refresh_metadata=True) == 1
    assert (release / "MANIFEST.json").read_bytes() == before


def test_source_provenance_verification(tmp_path):
    path = tmp_path / "source.docx"
    path.write_bytes(b"source fixture")
    docs = {"test": {"provenance": {"source_files": [{"file": path.name, "sha256": validator.sha256(path)}]}}}
    assert not validator.source_errors(docs, tmp_path)
    path.write_bytes(b"changed")
    assert validator.source_errors(docs, tmp_path)
    path.unlink()
    assert validator.source_errors(docs, tmp_path)


def test_source_backed_v11_regressions():
    # OSF Protocol 1: conflicting 30/50 top-up counts are not an acquisition lock.
    assert protocol(1)["blocks"]["session_quest_topup_trials"] is None
    # OSF Table 6 / Protocol 2.C: temporal lead AND tau, with seeds as replicates.
    assert protocol(2)["sub_predictions"][2]["confirming_thresholds"]["kendall_tau_strict_minimum"] == 0.3
    # OSF Protocol 4: preserve the signed proportional margin.
    assert protocol(4)["sub_predictions"][0]["confirming_thresholds"]["proportional_margin"] == 0.15
    # OSF Protocol 5.E: equivalence is an interval decision.
    equiv = protocol(5)["sub_predictions"][4]["confirming_thresholds"]
    assert equiv == {"rope_d": [-0.15, 0.15], "hdi_level": 0.95, "bf01_minimum": 6}
    # OSF Protocol 6.C stays deferred until estimator validation is archived.
    assert protocol(6)["sub_predictions"][2]["evaluation_status"] == "deferred_pending_estimator_validation"
    # OSF Revision 1: the patient-only primary is 7.G at final alpha .048.
    primary = protocol(7)["sub_predictions"][6]
    assert primary["id"] == "Pred 7.G"
    assert primary["confirming_thresholds"]["alpha"] == 0.048
    assert protocol(7)["participants"]["patient_n_target"] == 80


@pytest.mark.parametrize(
    "number,field",
    [(1, "main_signal_trials_per_session"), (5, "no_tms_visual_trials"), (6, "catch_trials")],
)
def test_trial_components(number, field):
    data = protocol(number)
    data["blocks"][field] += 1
    assert validator.semantic_errors(data)


# ===========================================================================
# Additional tests for uncovered error paths
# ===========================================================================


def test_safe_path_empty_name(tmp_path):
    """Test _safe_path with empty string."""
    with pytest.raises(ValueError, match="file path must be a nonempty string"):
        validator._safe_path(tmp_path, "")


def test_safe_path_non_string_name(tmp_path):
    """Test _safe_path with non-string input."""
    with pytest.raises(ValueError, match="file path must be a nonempty string"):
        validator._safe_path(tmp_path, 123)


def test_safe_path_absolute_path(tmp_path):
    """Test _safe_path rejects absolute paths."""
    with pytest.raises(ValueError, match="unsafe file path"):
        validator._safe_path(tmp_path, "/etc/passwd")


def test_safe_path_parent_traversal(tmp_path):
    """Test _safe_path rejects parent directory traversal."""
    with pytest.raises(ValueError, match="unsafe file path"):
        validator._safe_path(tmp_path, "../../../etc/passwd")


def test_safe_path_backslash_escape(tmp_path):
    """Test _safe_path rejects backslashes (path traversal on Windows)."""
    with pytest.raises(ValueError, match="unsafe file path"):
        validator._safe_path(tmp_path, "subdir\\..\\passwd")


def test_safe_path_escapes_folder(tmp_path):
    """Test _safe_path rejects paths that escape folder after resolution."""
    subfolder = tmp_path / "subfolder"
    subfolder.mkdir()
    # Try to reference a path that escapes even after POSIX normalization
    with pytest.raises(ValueError, match="unsafe file path"):
        validator._safe_path(subfolder, "../../etc/passwd")


def test_inspect_protocols_schema_error(tmp_path):
    """Test inspect_protocols with invalid schema."""
    schema_path = tmp_path / "bad_schema.json"
    schema_path.write_text('{"type": "invalid_type_name"}', encoding="utf-8")
    documents, errors = validator.inspect_protocols(schema_path, [])
    assert len(errors) > 0
    assert "Schema" in str(errors) or "type" in str(errors).lower()


def test_inspect_protocols_no_files(tmp_path):
    """Test inspect_protocols with no protocol files."""
    schema_path = PROTOCOLS / "schemas/protocol.schema.json"
    documents, errors = validator.inspect_protocols(schema_path, [])
    assert "no protocol files found" in errors


def test_inspect_protocols_file_read_error(tmp_path):
    """Test inspect_protocols with unreadable file."""
    schema_path = PROTOCOLS / "schemas/protocol.schema.json"
    nonexistent = tmp_path / "nonexistent.json"
    documents, errors = validator.inspect_protocols(schema_path, [nonexistent])
    assert any("nonexistent" in str(e) for e in errors)


def test_manifest_errors_missing_manifest(tmp_path):
    """Test manifest_errors when MANIFEST.json is missing."""
    errors = validator.manifest_errors(tmp_path)
    assert any("MANIFEST.json" in str(e) for e in errors)


def test_manifest_errors_invalid_release(tmp_path):
    """Test manifest_errors with incorrect release version."""
    manifest = {"release": "0.9.0", "files": []}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validator.manifest_errors(tmp_path)
    assert any("incorrect release" in str(e) for e in errors)


def test_manifest_errors_files_not_array(tmp_path):
    """Test manifest_errors when files is not an array."""
    manifest = {"release": validator.RELEASE, "files": "not_an_array"}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validator.manifest_errors(tmp_path)
    assert any("must be an array" in str(e) for e in errors)


def test_manifest_errors_invalid_file_entry(tmp_path):
    """Test manifest_errors with non-dict file entry."""
    validation_json = {
        "release": validator.RELEASE,
        "date": "2026-10-16",
        "schema": "JSON Schema Draft 2020-12",
        "files": [],
        "total_protocols": 0,
        "total_predictions": 0,
        "checks": [],
        "limitations": [],
    }
    (tmp_path / "VALIDATION.json").write_text(json.dumps(validation_json), encoding="utf-8")
    manifest = {"release": validator.RELEASE, "prepared": "2026-10-16", "files": ["not_a_dict"]}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validator.manifest_errors(tmp_path)
    assert any("invalid file entry" in str(e) for e in errors)


def test_manifest_errors_duplicate_entry(tmp_path):
    """Test manifest_errors with duplicate file paths."""
    validation_json = {
        "release": validator.RELEASE,
        "date": "2026-10-16",
        "schema": "JSON Schema Draft 2020-12",
        "files": [],
        "total_protocols": 0,
        "total_predictions": 0,
        "checks": [],
        "limitations": [],
    }
    (tmp_path / "VALIDATION.json").write_text(json.dumps(validation_json), encoding="utf-8")
    manifest = {
        "release": validator.RELEASE,
        "prepared": "2026-10-16",
        "files": [
            {"path": "file.json", "sha256": "abc123"},
            {"path": "file.json", "sha256": "abc123"},
        ],
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validator.manifest_errors(tmp_path)
    assert any("duplicate entry" in str(e) for e in errors)


def test_manifest_errors_missing_validation_json(tmp_path):
    """Test manifest_errors when VALIDATION.json is missing."""
    manifest_data = {"release": validator.RELEASE, "prepared": "2026-10-16", "files": []}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    errors = validator.manifest_errors(tmp_path)
    assert any("VALIDATION.json" in str(e) for e in errors)


def test_source_errors_sha256_mismatch(tmp_path):
    """Test source_errors with SHA256 mismatch."""
    source_file = tmp_path / "source.txt"
    source_file.write_text("content", encoding="utf-8")

    documents = {"protocol.json": {"provenance": {"source_files": [{"file": "source.txt", "sha256": "wronghash"}]}}}
    errors = validator.source_errors(documents, tmp_path)
    assert any("SHA256 mismatch" in str(e) for e in errors)


def test_source_errors_missing_file(tmp_path):
    """Test source_errors with missing source file."""
    documents = {"protocol.json": {"provenance": {"source_files": [{"file": "missing.txt", "sha256": "abc123"}]}}}
    errors = validator.source_errors(documents, tmp_path)
    assert any("missing" in str(e).lower() for e in errors)


def test_validate_release_missing_changelog(release):
    """Test validate_release when CHANGELOG.md is missing."""
    (release / "CHANGELOG.md").unlink()
    result = validator.validate_release(release)
    assert result != 0


def test_validate_release_schema_mismatch(release):
    """Test validate_release with mismatched schema version."""
    schema = validator.read_json(release / "schemas/protocol.schema.json")
    schema["x-version"] = "0.9.0"
    validator._write_json(release / "schemas/protocol.schema.json", schema)
    result = validator.validate_release(release)
    assert result != 0


def test_validate_release_protocol_version_mismatch(release):
    """Test validate_release with protocol version mismatch."""
    proto = validator.read_json(next(release.glob("protocol_0_*.json")))
    proto["version"] = "0.9.0"
    validator._write_json(next(release.glob("protocol_0_*.json")), proto)
    result = validator.validate_release(release)
    assert result != 0


def test_validate_release_filename_protocol_id_disagree(release):
    """Test validate_release when filename and protocol_id disagree."""
    proto = validator.read_json(next(release.glob("protocol_0_*.json")))
    proto_file = next(release.glob("protocol_0_*.json"))
    # Create wrong filename
    new_file = proto_file.parent / "protocol_7_wrong.json"
    validator._write_json(new_file, proto)
    proto_file.unlink()
    result = validator.validate_release(release)
    assert result != 0


def test_validate_release_wrong_protocol_count(release):
    """Test validate_release with missing protocols."""
    proto_file = next(release.glob("protocol_0_*.json"))
    proto_file.unlink()
    result = validator.validate_release(release)
    assert result != 0


def test_validate_release_stale_validation_json(release):
    """Test validate_release when VALIDATION.json is stale."""
    # Run initial validation with refresh
    validator.validate_release(release, refresh_metadata=True)
    # Now modify a protocol
    proto = validator.read_json(next(release.glob("protocol_0_*.json")))
    proto["sub_predictions"] = proto["sub_predictions"][:1]
    validator._write_json(next(release.glob("protocol_0_*.json")), proto)
    # Validation should fail due to stale metadata
    result = validator.validate_release(release)
    assert result != 0
