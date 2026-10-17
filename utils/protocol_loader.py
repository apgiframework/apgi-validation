"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1

PROTOCOL MIGRATION: Phase 4 - Hard Deprecation Complete
============================================================================
LEGACY SUPPORT REMOVED: As of Phase 4, only canonical APGI-P## protocol
naming is supported. All legacy VP-##/FP-## protocol IDs will raise ValueError.

The protocols/legacy/ directory has been removed entirely. Legacy protocol
files are no longer searchable and cannot be loaded.

MIGRATION TIMELINE:
  - Phase 1 (2026-06-25): New APGI-P## naming introduced
  - Phase 2 (2026-06-25): Deprecation notices added
  - Phase 3 (2026-09-25): Soft deprecation with warnings
  - Phase 4 (2026-12-25): Hard deprecation - legacy files removed [CURRENT PHASE]

All code must use canonical APGI-P## naming (APGI-P00, APGI-P01, etc.)

Legacy protocols are archived for reference only and are not accessible
via the protocol_loader module. Use canonical naming exclusively.

See PROTOCOL-MIGRATION-USER-GUIDE.md for migration instructions and examples.
============================================================================
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

_PROTOCOLS_DIR = Path(__file__).parent.parent / "protocols"
_TEMPLATE_PATH = Path(__file__).parent.parent / "config" / "apgi_protocol_template.json"

# Protocol specification release the files in protocols/ must carry. Kept in
# step with apgi-framework (protocols/CHANGELOG.md, scripts/validate_protocols.py).
SPECIFICATION_VERSION = "1.1.0"


class Verdict(str, Enum):
    """v1.1 three-way decision for a registered prediction."""

    CONFIRMED = "confirmed"
    FALSIFIED = "falsified"
    INDETERMINATE = "indeterminate"


def classify_prediction(confirmed: bool, falsified: bool = False) -> Verdict:
    """Apply the v1.1 rule that a failure to confirm is not falsification.

    ``falsified`` must come from the prediction's own falsifying rule
    (``falsifying_evidence`` / ``falsifying_thresholds``), never from the
    negation of ``confirmed``.
    """
    if confirmed:
        return Verdict.CONFIRMED
    if falsified:
        return Verdict.FALSIFIED
    return Verdict.INDETERMINATE


@dataclass
class SubPrediction:
    """A single falsifiable sub-prediction from a protocol JSON."""

    id: str
    claim: str
    confirming_evidence: str = ""
    falsifying_evidence: str = ""
    key_measurement_equation: str = ""
    status: str = ""
    indeterminate_rule: str = ""
    confirming_thresholds: Dict[str, Any] = field(default_factory=dict)
    falsifying_thresholds: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubPrediction":
        return cls(
            id=data["id"],
            claim=data.get("claim", ""),
            confirming_evidence=data.get("confirming_evidence", ""),
            falsifying_evidence=data.get("falsifying_evidence", ""),
            key_measurement_equation=data.get("key_measurement_equation", ""),
            status=data.get("status", ""),
            indeterminate_rule=data.get("indeterminate_rule", ""),
            confirming_thresholds=data.get("confirming_thresholds") or {},
            falsifying_thresholds=data.get("falsifying_thresholds") or {},
        )

    @property
    def is_confirmatory(self) -> bool:
        """True for predictions whose outcome carries a registered framework vote."""
        return self.status.startswith("confirmatory")


@dataclass
class ProtocolSpec:
    """Parsed representation of a JSON protocol specification."""

    protocol_id: str
    title: str
    version: str
    paper: str
    description: str
    validation_status: str
    apgi_parameters: Dict[str, Any]
    sub_predictions: List[SubPrediction]
    primary_hypothesis: str
    caveats: List[str]
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolSpec":
        return cls(
            protocol_id=data["protocol_id"],
            title=data.get("title", ""),
            version=data.get("version", "1.0"),
            paper=data.get("paper", ""),
            description=data.get("description", ""),
            # v1.1 replaced validation_status with study_status and folded the
            # caveats into submission_blocking_issues.
            validation_status=data.get("study_status", data.get("validation_status", "unspecified")),
            apgi_parameters=data.get("apgi_parameters", {}),
            sub_predictions=[SubPrediction.from_dict(p) for p in data.get("sub_predictions", [])],
            primary_hypothesis=data.get("primary_hypothesis", ""),
            caveats=data.get("submission_blocking_issues", data.get("caveats", [])),
            _raw=data,
        )

    def get_parameter(self, name: str, default: Any = None) -> Any:
        return self.apgi_parameters.get(name, default)

    @property
    def participants(self) -> Dict[str, Any]:
        """Registered cohort block (n_target, n_minimum, n_analyzable_target, groups)."""
        return self._raw.get("participants", {})

    @property
    def is_current_specification(self) -> bool:
        return self.version == SPECIFICATION_VERSION

    def get_prediction(self, pred_id: str) -> Optional[SubPrediction]:
        for p in self.sub_predictions:
            if p.id == pred_id:
                return p
        return None

    def prediction_ids(self) -> List[str]:
        return [p.id for p in self.sub_predictions]

    def to_dict(self) -> Dict[str, Any]:
        return self._raw


def _iter_protocol_files() -> List[Path]:
    """Return all canonical protocol JSON files (APGI-P## only)."""
    if not _PROTOCOLS_DIR.exists():
        return []
    # Only canonical protocol_*.json files in protocols/ root
    return sorted(_PROTOCOLS_DIR.glob("protocol_*.json"))


def _find_protocol_file(protocol_id: str) -> Optional[Path]:
    """Resolve a protocol ID to its JSON file.

    Phase 4 (Hard Deprecation): ONLY canonical APGI-P## naming is supported.
    Legacy VP-##/FP-## IDs will raise ValueError with clear error message.

    Args:
        protocol_id: The protocol ID (must be in APGI-P## format)

    Returns:
        Path to protocol JSON file, or None if not found

    Raises:
        ValueError: If legacy VP-##/FP-## naming is attempted
    """
    if not _PROTOCOLS_DIR.exists():
        return None

    normalized_id = protocol_id.strip()

    # PHASE 4: REJECT legacy protocol IDs
    if normalized_id.startswith("VP-") or normalized_id.startswith("FP-"):
        raise ValueError(
            f"PROTOCOL MIGRATION - PHASE 4 HARD DEPRECATION:\n"
            f"  Legacy protocol ID '{normalized_id}' is no longer supported.\n"
            f"  Legacy protocol files have been removed permanently from protocols/legacy/.\n"
            f"  You MUST use canonical APGI-P## naming (e.g., APGI-P01, APGI-P02).\n"
            f"  See PROTOCOL-MIGRATION-USER-GUIDE.md for the mapping from legacy to canonical IDs."
        )

    # Canonical APGI-P## naming
    if normalized_id.startswith("APGI-P"):
        num_str = normalized_id.split("P")[-1]
        try:
            num = int(num_str)
        except ValueError:
            num = None
        if num is not None:
            for path in _iter_protocol_files():
                stem = path.stem  # e.g. "protocol_1_eeg_interoceptive_gating"
                parts = stem.split("_")
                if len(parts) >= 2:
                    try:
                        if int(parts[1]) == num:
                            return path
                    except ValueError:
                        continue

    # Generic path: match the embedded protocol_id in JSON
    for path in _iter_protocol_files():
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:  # nosec B110,B112
            continue
        matched = data.get("protocol_id") == normalized_id
        if not matched:
            aliases = data.get("aliases", [])
            matched = isinstance(aliases, list) and normalized_id in aliases
        if matched:
            return path
    return None


def load_protocol(protocol_id: str) -> Optional[ProtocolSpec]:
    """Load a protocol spec by ID (e.g. 'APGI-P01'). Returns None if not found."""
    path = _find_protocol_file(protocol_id)
    if path is None:
        return None
    return load_protocol_file(path)


def load_protocol_file(path: Path) -> ProtocolSpec:
    """Load a ProtocolSpec from a specific JSON file path."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return ProtocolSpec.from_dict(data)


def load_all_protocols() -> Dict[str, ProtocolSpec]:
    """Load all canonical protocol specs (APGI-P## only), keyed by protocol_id.

    Phase 4 (Hard Deprecation): Legacy protocols in protocols/legacy/ are no
    longer accessible. Only canonical APGI-P## protocols can be loaded.

    For legacy protocol references, update your code to use canonical naming
    and review PROTOCOL-MIGRATION-USER-GUIDE.md for the mapping.

    Returns:
        Dict mapping protocol_id (e.g., "APGI-P01") to ProtocolSpec objects
    """
    specs: Dict[str, ProtocolSpec] = {}
    for path in _iter_protocol_files():
        try:
            spec = load_protocol_file(path)
            specs[spec.protocol_id] = spec
        except Exception:  # nosec B110
            pass
    return specs


def get_apgi_parameters(protocol_id: str) -> Dict[str, Any]:
    """Return the apgi_parameters dict for a protocol, or empty dict if not found."""
    spec = load_protocol(protocol_id)
    return spec.apgi_parameters if spec is not None else {}


def spec_prediction_results(
    spec: Optional[ProtocolSpec],
    linked_results: Dict[str, Any],
    pred_map: Dict[str, str],
    sources: List[str],
) -> Dict[str, Any]:
    """Report a protocol's registered predictions alongside simulation results.

    ``pred_map`` links a registered prediction ID (e.g. ``"Pred 6.A"``) to the
    key of the simulation ``PredictionResult`` in ``linked_results`` that
    implements the same test. Under specification 1.1.0 a simulation can only
    confirm (synthetically) or leave a prediction INDETERMINATE: it never runs
    the registered falsifying rule on empirical data, so a missing link or a
    failed simulation check is never reported as falsification.
    """
    from utils.protocol_schema import PredictionResult, PredictionStatus

    results: Dict[str, Any] = {}
    if spec is None:
        return results
    for sub_pred in spec.sub_predictions:
        linked = linked_results.get(pred_map.get(sub_pred.id, ""))
        confirmed = bool(linked is not None and linked.passed)
        verdict = classify_prediction(confirmed)
        results[sub_pred.id] = PredictionResult(
            passed=confirmed,
            value=linked.value if linked is not None else None,
            threshold=linked.threshold if linked is not None else None,
            status=PredictionStatus.PASSED if confirmed else PredictionStatus.INDETERMINATE,
            name=sub_pred.claim,
            evidence=[e for e in (sub_pred.confirming_evidence, sub_pred.falsifying_evidence) if e],
            sources=[spec.protocol_id, *sources],
            metadata={
                "verdict": verdict.value,
                "evidence_class": "synthetic",
                "prediction_status": sub_pred.status,
                "linked_result": pred_map.get(sub_pred.id),
                "specification_version": spec.version,
            },
        )
    return results


def load_protocol_template() -> Dict[str, Any]:
    """Load the legacy (pre-1.1) master protocol template.

    The template predates specification 1.1.0 and lives in config/ so the
    protocols/ release folder holds only the registered files covered by
    protocols/MANIFEST.json. Its benchmarks are not v1.1 decision rules.
    """
    if not _TEMPLATE_PATH.exists():
        return {}
    with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def get_template_protocol(protocol_key: str) -> Optional[Dict[str, Any]]:
    """Extract a specific protocol from the template by key (e.g., 'P01')."""
    template = load_protocol_template()
    protocols = template.get("apgi_validation_metadata", {}).get("protocols", {})
    return protocols.get(protocol_key)


def _convert_benchmarks_to_predictions(benchmarks: Dict[str, Any]) -> List[SubPrediction]:
    """Convert benchmark objects to SubPrediction format."""
    predictions = []
    for bench_id, bench_data in benchmarks.items():
        value = bench_data.get("value")
        value_str = f"{value} {bench_data.get('unit', '')}" if value is not None else "N/A"
        predictions.append(
            SubPrediction(
                id=bench_id,
                claim=f"Expected value: {value_str}",
                confirming_evidence=f"Role: {bench_data.get('role', 'unknown')}",
                key_measurement_equation=bench_data.get("note", ""),
            )
        )
    return predictions


def template_to_protocol_spec(protocol_key: str) -> Optional[ProtocolSpec]:
    """Convert template protocol to ProtocolSpec format."""
    template_proto = get_template_protocol(protocol_key)
    if not template_proto:
        return None

    # Map template fields to ProtocolSpec
    apgi_params = {
        "benchmarks": template_proto.get("benchmarks", {}),
        "falsification_thresholds": template_proto.get("falsification_thresholds", {}),
        "sample_size": template_proto.get("sample_size", {}),
    }
    # Add other template fields to apgi_parameters
    for k, v in template_proto.items():
        if k not in ["id", "status", "benchmarks", "falsification_thresholds", "sample_size"]:
            apgi_params[k] = v

    return ProtocolSpec(
        protocol_id=f"APGI-{protocol_key}",
        title=template_proto.get("id", ""),
        version="2.0",
        paper="",
        description=f"Status: {template_proto.get('status', 'unknown')}",
        validation_status=template_proto.get("status", "unknown"),
        apgi_parameters=apgi_params,
        sub_predictions=_convert_benchmarks_to_predictions(template_proto.get("benchmarks", {})),
        primary_hypothesis="",
        caveats=[],
        _raw=template_proto,
    )
