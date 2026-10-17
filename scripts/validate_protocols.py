"""Validate APGI 1.1 protocols, release metadata, and optional local sources."""

import argparse
import hashlib
import json
import math
import pathlib
import sys

import jsonschema

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "protocols" / "schemas" / "protocol.schema.json"
RELEASE = "1.1.0"
EXPECTED_IDS = {f"APGI-P0{n}" for n in range(8)}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def read_json(path):
    """Reject duplicate keys and non-standard NaN/Infinity constants."""
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_errors(data):
    """Cross-field constraints that standard JSON Schema cannot express."""
    errors = []
    people = data["participants"]
    for key in ("n_minimum", "n_analyzable_target"):
        if people.get(key) is not None and people[key] > people["n_target"]:
            errors.append(f"participants.{key} exceeds n_target")
    if people.get("n_minimum") is not None and people.get("n_analyzable_target") is not None:
        if people["n_minimum"] > people["n_analyzable_target"]:
            errors.append("participants.n_minimum exceeds n_analyzable_target")
    if "groups" in people:
        if sum(people["groups"].values()) != people["n_target"]:
            errors.append("cohort group sum differs from n_target")
        if "patient_n_target" in people:
            patients = sum(v for k, v in people["groups"].items() if k != "healthy_controls")
            if patients != people["patient_n_target"]:
                errors.append("patient cohort sum differs from patient_n_target")
    blocks = data["blocks"]
    values = [blocks.get(k) for k in ("experimental_blocks", "trials_per_block", "total_experimental_trials")]
    if all(v is not None for v in values) and values[0] * values[1] != values[2]:
        errors.append("experimental_blocks * trials_per_block differs from total_experimental_trials")
    components = {
        "APGI-P01": [
            (
                "trials_per_block",
                ["main_signal_trials_per_session", "main_catch_trials_per_session"],
            )
        ],
        "APGI-P05": [
            ("trials_per_block", ["visual_trials_per_session", "heartbeat_trials_per_session"]),
            ("visual_trials_per_session", ["tms_visual_trials", "no_tms_visual_trials"]),
        ],
        "APGI-P06": [
            (
                "total_experimental_trials",
                ["catch_trials", "suprathreshold_trials", "near_threshold_trials"],
            )
        ],
    }
    for total, parts in components.get(data["protocol_id"], []):
        values = [blocks.get(k) for k in [total, *parts]]
        if not all(type(v) is int and v >= 0 for v in values):
            errors.append(f"blocks: {total} and its components must be nonnegative integers")
        elif values[0] != sum(values[1:]):
            errors.append(f"blocks: components do not sum to {total}")
    power = data["power_analysis"]
    if "n_target" in power and power["n_target"] != people["n_target"]:
        errors.append("power_analysis.n_target differs from participants.n_target")
    if data["last_updated"] < "2026-10-16":
        errors.append("last_updated predates the 1.1 release preparation")
    return errors


def inspect_protocols(schema_path, paths):
    documents, errors = {}, []
    try:
        schema = read_json(schema_path)
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        validator = cls(schema, format_checker=jsonschema.FormatChecker())
    except (OSError, ValueError, jsonschema.SchemaError) as exc:
        return {}, [f"{schema_path}: {exc}"]
    paths = list(paths)
    if not paths:
        errors.append("no protocol files found")
    for path in paths:
        try:
            data = read_json(path)
            failures = list(validator.iter_errors(data))
            for error in failures:
                location = ".".join(str(x) for x in error.absolute_path) or "<root>"
                errors.append(f"{path.name}:{location}: {error.message}")
            if failures:
                continue
            documents[path.name] = data
            if isinstance(data, dict) and data.get("schema_version") == RELEASE:
                errors.extend(f"{path.name}: {e}" for e in semantic_errors(data))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    return documents, errors


def _print_result(documents, errors):
    if errors:
        for error in errors:
            print(f"  FAIL {error}", file=sys.stderr)
        return 1
    for name in sorted(documents):
        print(f"  OK   {name}")
    print(f"All {len(documents)} protocols valid.")
    return 0


def validate_protocols(schema_path=SCHEMA_PATH, protocols_glob=None):
    """Validate selected instances; discover files at call time, excluding metadata."""
    schema_path = pathlib.Path(schema_path)
    paths = protocols_glob if protocols_glob is not None else sorted(schema_path.parent.parent.glob("protocol_*.json"))
    return _print_result(*inspect_protocols(schema_path, paths))


def validation_report(documents):
    return {
        "release": RELEASE,
        "date": max(d["last_updated"] for d in documents.values()),
        "schema": "JSON Schema Draft 2020-12",
        "files": [
            {
                "file": name,
                "schema_valid": True,
                "semantic_valid": True,
                "prediction_count": len(d["sub_predictions"]),
            }
            for name, d in sorted(documents.items())
        ],
        "total_protocols": len(documents),
        "total_predictions": sum(len(d["sub_predictions"]) for d in documents.values()),
        "checks": [
            "Strict JSON parsing (duplicate keys and non-finite constants rejected)",
            "Schema self-validation and instance format checking",
            "Exact Protocol 0–7 membership and prediction IDs/counts/uniqueness",
            "Sample minimum/analyzable/enrollment ordering",
            "Main-trial arithmetic and clinical cohort totals",
            "Pending approvals cannot authorize enrollment",
            "Uncalibrated kappa has no numerical estimate",
            "Release manifest coverage and SHA256 verified by validator",
        ],
        "limitations": [
            "Structural and consistency checks are not empirical validation or power simulations.",
            "Source provenance is checked only when --source-dir is supplied; historical input hashes are retained as provenance, not revalidated by this report.",
            "OSF filing stated as 2026-10-16 is not independently verified.",
            "Source conflicts and pending analysis locks remain listed in each protocol and CHANGELOG.md.",
        ],
    }


def _payload_paths(folder):
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.name != "MANIFEST.json"
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )


def _safe_path(folder, name):
    if not isinstance(name, str) or not name:
        raise ValueError("file path must be a nonempty string")
    relative = pathlib.PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"unsafe file path: {name}")
    path = folder / relative
    if not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f"file path escapes folder: {name}")
    return path


def manifest_errors(folder):
    try:
        manifest = read_json(folder / "MANIFEST.json")
        if not isinstance(manifest, dict) or manifest.get("release") != RELEASE:
            return ["MANIFEST.json: incorrect release"]
        entries = manifest.get("files")
        if not isinstance(entries, list):
            return ["MANIFEST.json: files must be an array"]
        expected = {p.relative_to(folder).as_posix() for p in _payload_paths(folder)}
        seen, errors = set(), []
        for entry in entries:
            if not isinstance(entry, dict):
                errors.append("MANIFEST.json: invalid file entry")
                continue
            name = entry.get("path")
            path = _safe_path(folder, name)
            if name in seen:
                errors.append(f"MANIFEST.json: duplicate entry {name}")
            seen.add(name)
            if not path.is_file() or sha256(path) != entry.get("sha256"):
                errors.append(f"MANIFEST.json: missing file or SHA256 mismatch: {name}")
        if seen != expected:
            errors.append(f"MANIFEST.json: coverage mismatch: {sorted(seen ^ expected)}")
        jsonschema.Draft202012Validator(
            {"type": "string", "format": "date"}, format_checker=jsonschema.FormatChecker()
        ).validate(manifest.get("prepared"))
        report = read_json(folder / "VALIDATION.json")
        if not isinstance(report, dict) or manifest.get("prepared") != report.get("date"):
            errors.append("MANIFEST.json: prepared date differs from VALIDATION.json")
        return errors
    except (OSError, ValueError, jsonschema.ValidationError) as exc:
        return [f"MANIFEST.json: {exc}"]


def source_errors(documents, source_dir):
    errors, checked = [], set()
    for data in documents.values():
        for source in data["provenance"]["source_files"]:
            pair = (source["file"], source["sha256"])
            if pair in checked:
                continue
            checked.add(pair)
            try:
                path = _safe_path(source_dir, source["file"])
                if sha256(path) != source["sha256"]:
                    errors.append(f"source SHA256 mismatch: {source['file']}")
            except (OSError, ValueError) as exc:
                errors.append(f"source {source['file']}: {exc}")
    return errors


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_release(folder, *, source_dir=None, refresh_metadata=False):
    """Validate an entire release; metadata regeneration is an explicit operation."""
    folder = pathlib.Path(folder)
    documents, errors = inspect_protocols(
        folder / "schemas/protocol.schema.json", sorted(folder.glob("protocol_*.json"))
    )
    if errors:
        return _print_result(documents, errors)
    ids = [d.get("protocol_id") for d in documents.values()]
    if len(ids) != 8 or set(ids) != EXPECTED_IDS:
        errors.append("release must contain exactly one instance of each Protocol 0–7")
    schema = read_json(folder / "schemas/protocol.schema.json")
    if schema.get("x-version") != RELEASE or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append("release requires the 1.1.0 Draft 2020-12 schema")
    for name, data in documents.items():
        if data.get("version") != RELEASE or data.get("schema_version") != RELEASE:
            errors.append(f"{name}: incorrect release version")
        if not name.startswith(f"protocol_{data['protocol_id'][-1]}_"):
            errors.append(f"{name}: filename and protocol_id disagree")
    if not (folder / "CHANGELOG.md").is_file():
        errors.append("CHANGELOG.md missing")
    if source_dir is not None:
        errors.extend(source_errors(documents, pathlib.Path(source_dir)))
    if errors:
        return _print_result(documents, errors)
    report = validation_report(documents)
    if refresh_metadata:
        _write_json(folder / "VALIDATION.json", report)
        _write_json(
            folder / "MANIFEST.json",
            {
                "release": RELEASE,
                "prepared": report["date"],
                "files": [
                    {"path": p.relative_to(folder).as_posix(), "sha256": sha256(p)} for p in _payload_paths(folder)
                ],
            },
        )
    try:
        if read_json(folder / "VALIDATION.json") != report:
            errors.append("VALIDATION.json is stale or inconsistent; validate then use --refresh-metadata")
    except (OSError, ValueError) as exc:
        errors.append(f"VALIDATION.json: {exc}")
    errors.extend(manifest_errors(folder))
    if source_dir is not None and not errors:
        print("Local source-document SHA256 values verified.")
    return _print_result(documents, errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=pathlib.Path, default=ROOT / "protocols")
    parser.add_argument("--source-dir", type=pathlib.Path, help="Verify provenance against local source documents")
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Regenerate report and checksums after successful protocol validation",
    )
    args = parser.parse_args(argv)
    raise SystemExit(
        validate_release(args.directory, source_dir=args.source_dir, refresh_metadata=args.refresh_metadata)
    )


if __name__ == "__main__":
    main()
