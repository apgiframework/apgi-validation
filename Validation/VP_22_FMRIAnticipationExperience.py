"""
APGI Protocol 5: fMRI Anticipation vs. Experience (Somatic Marker)
===================================================================

Enhanced implementation of APGI-P03 (specification 1.1.0) for the APGI validation suite.

This module was written against the existing VP-14 and VP-15 fMRI protocol
implementations and follows their simulation-first workflow:

- VP-14 contributes the anticipation-versus-experience framing.
- VP-15 contributes the vmPFC/posterior-insula connectivity emphasis.
- Protocol 5 adds the soma-bias hypothesis, identifiability constraints,
  and a hard falsification rule for vmPFC tracking of primary prediction error.

Predictions implemented
-----------------------
P5a: vmPFC anticipation correlates with posterior-insula/SCR precision channels.
P5b: vmPFC does not correlate with primary interoceptive prediction error during experience.
P5c: vmPFC effect is valence-specific rather than sensory-contrast-driven.
P5d: Removing anticipation collapses vmPFC-posterior-insula coupling.

Critical falsification
----------------------
If vmPFC significantly tracks somatosensory/interoceptive primary prediction error
instead of anticipatory somatic-marker retrieval, APGI's embodied-prioritization
claim is rejected for this protocol.

CANONICAL EP MAPPING
--------------------
VP-22 is the CANONICAL implementation of EP-5 (fMRI Anticipation vs. Experience).
  EP-5: "fMRI study — Anticipation vs. Experience of somatic signals" (P5a–P5d)
VP-14 (VP_14_FMRIAnticipationExperience.py) is the simulation-precursor that has
been superseded by this file. VP-15 covers the vmPFC/precision extension (EP-5 ext.).
T05 in the Validation_GUI sidebar redirects to this file.
V22.x predictions count in the canonical 14-prediction falsification tally.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import stats

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from utils.protocol_schema import PredictionResult, PredictionStatus, ProtocolResult

    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False

try:
    from utils.protocol_loader import load_protocol as _load_protocol_p03
    from utils.protocol_loader import spec_prediction_results

    # Specification 1.1.0 numbers fMRI anticipation vs experience as Protocol 3.
    # The internal P5a–P5d keys below keep the pre-1.1 numbering.
    _PROTOCOL_SPEC_P03 = _load_protocol_p03("APGI-P03")
except Exception:
    _PROTOCOL_SPEC_P03 = None

try:
    from utils.logging_config import apgi_logger as logger  # type: ignore[assignment]
except Exception:
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

try:
    from utils.falsification_thresholds import V15_ANTICIPATORY_CORRELATION_MIN
except ImportError:
    V15_ANTICIPATORY_CORRELATION_MIN = 0.40


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PRIMARY_ALPHA = 0.01
PRECISION_MATCH_TOLERANCE = 0.10
MIN_BETA = 1.20
IDENTIFIABILITY_FIM_MAX = 0.10
NO_ANTICIPATION_COUPLING_MAX = 0.15
ROIS = ("vmPFC", "anterior_insula", "posterior_insula", "somatosensory_cortex", "amygdala")


class Protocol5Config:
    """Configuration for APGI-P03 simulation and validation."""

    def __init__(
        self,
        n_subjects: int = 36,
        alpha: float = PRIMARY_ALPHA,
        min_beta: float = MIN_BETA,
        precision_match_tolerance: float = PRECISION_MATCH_TOLERANCE,
        min_anticipatory_coupling: float = V15_ANTICIPATORY_CORRELATION_MIN,
        no_anticipation_coupling_max: float = NO_ANTICIPATION_COUPLING_MAX,
        identifiability_fim_max: float = IDENTIFIABILITY_FIM_MAX,
        random_seed: int = RANDOM_SEED,
    ):
        self.n_subjects = n_subjects
        self.alpha = alpha
        self.min_beta = min_beta
        self.precision_match_tolerance = precision_match_tolerance
        self.min_anticipatory_coupling = min_anticipatory_coupling
        self.no_anticipation_coupling_max = no_anticipation_coupling_max
        self.identifiability_fim_max = identifiability_fim_max
        self.random_seed = random_seed

    def __str__(self) -> str:
        """Safe string representation."""
        return f"Protocol5Config(n_subjects={self.n_subjects}, alpha={self.alpha})"


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Robust Pearson correlation with deterministic fallback."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        return 0.0, 1.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _safe_ttest_rel(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Paired t-test with stable fallback for degenerate vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        return 0.0, 1.0
    if np.allclose(x, y):
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(x, y, nan_policy="omit")
    if np.isnan(t_stat) or np.isnan(p_val):
        return 0.0, 1.0
    return float(t_stat), float(p_val)


def _safe_linregress(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Linear regression summary with harmless fallback."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"slope": 0.0, "intercept": 0.0, "r": 0.0, "p": 1.0, "stderr": 0.0}
    slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r_val),
        "p": float(p_val),
        "stderr": float(stderr) if stderr is not None else 0.0,
    }


class APGIP5SomaticProtocol:
    """Protocol 5 validator with simulation and empirical-data support."""

    protocol_id = "VP_22_FMRIAnticipationExperience"
    protocol_name = "APGI-P03 fMRI Anticipation vs. Experience (Somatic Marker)"

    def __init__(self, config: Optional[Protocol5Config] = None):
        self.config = config or Protocol5Config()
        self.rois = ROIS

    def simulate_subject_level_data(self) -> Dict[str, np.ndarray]:
        """
        Generate subject-level synthetic data consistent with Protocol 5.

        The simulation keeps beta and interoceptive precision as distinct latent
        contributors and exposes them through separate fMRI and SCR channels.
        """
        rng = np.random.default_rng(self.config.random_seed)
        n = self.config.n_subjects

        beta = rng.normal(loc=1.35, scale=0.12, size=n)
        intero_precision = rng.normal(loc=0.84, scale=0.08, size=n)
        extero_precision = intero_precision + rng.normal(loc=0.0, scale=0.04, size=n)
        fim_offdiag = np.abs(rng.normal(loc=0.05, scale=0.02, size=n))

        valence_signal = rng.uniform(0.35, 0.90, size=n)
        sensory_contrast = rng.uniform(0.10, 0.90, size=n)
        anticipation_present = rng.binomial(1, 0.75, size=n).astype(float)

        anticipatory_marker = 0.55 * beta + 0.35 * intero_precision + 0.25 * valence_signal
        anticipatory_marker += rng.normal(0.0, 0.05, size=n)

        posterior_insula_anticipation = 0.62 * anticipatory_marker + 0.20 * intero_precision
        posterior_insula_anticipation += rng.normal(0.0, 0.06, size=n)

        anterior_insula_anticipation = 0.54 * anticipatory_marker + rng.normal(0.0, 0.06, size=n)
        amygdala_anticipation = 0.48 * anticipatory_marker + 0.15 * valence_signal
        amygdala_anticipation += rng.normal(0.0, 0.07, size=n)

        scr_anticipation = 0.46 * anticipatory_marker + 0.26 * intero_precision
        scr_anticipation += rng.normal(0.0, 0.07, size=n)

        vmPFC_anticipation = 0.70 * anticipatory_marker + 0.25 * beta + 0.22 * valence_signal
        vmPFC_anticipation += 0.03 * sensory_contrast + rng.normal(0.0, 0.05, size=n)

        primary_prediction_error = rng.normal(loc=0.0, scale=0.22, size=n)
        somatosensory_experience = primary_prediction_error + rng.normal(0.0, 0.03, size=n)
        posterior_insula_experience = 0.22 * anticipatory_marker + 0.18 * primary_prediction_error
        posterior_insula_experience += rng.normal(0.0, 0.08, size=n)
        vmPFC_experience = 0.15 * anticipatory_marker + 0.05 * primary_prediction_error
        vmPFC_experience += rng.normal(0.0, 0.06, size=n)

        no_anticipation_vmPFC = 0.22 * beta + 0.09 * valence_signal + rng.normal(0.0, 0.08, size=n)
        no_anticipation_posterior_insula = 0.15 * intero_precision + rng.normal(0.0, 0.08, size=n)

        return {
            "beta": beta,
            "intero_precision": intero_precision,
            "extero_precision": extero_precision,
            "fim_offdiag": fim_offdiag,
            "anticipation_present": anticipation_present,
            "valence_signal": valence_signal,
            "sensory_contrast": sensory_contrast,
            "vmPFC_anticipation": vmPFC_anticipation,
            "anterior_insula_anticipation": anterior_insula_anticipation,
            "posterior_insula_anticipation": posterior_insula_anticipation,
            "amygdala_anticipation": amygdala_anticipation,
            "scr_anticipation": scr_anticipation,
            "vmPFC_experience": vmPFC_experience,
            "posterior_insula_experience": posterior_insula_experience,
            "somatosensory_experience": somatosensory_experience,
            "primary_prediction_error": primary_prediction_error,
            "no_anticipation_vmPFC": no_anticipation_vmPFC,
            "no_anticipation_posterior_insula": no_anticipation_posterior_insula,
        }

    def load_subject_level_data(self, data_path: str) -> Dict[str, np.ndarray]:
        """
        Load empirical subject-level arrays from NPZ or JSON.

        Required arrays:
        - vmPFC_anticipation
        - posterior_insula_anticipation
        - scr_anticipation
        - somatosensory_experience
        - primary_prediction_error
        - valence_signal
        - sensory_contrast
        - no_anticipation_vmPFC
        - no_anticipation_posterior_insula
        - beta
        - intero_precision
        - extero_precision
        - fim_offdiag
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Protocol 5 data path not found: {data_path}")

        if path.suffix == ".npz":
            raw = np.load(path, allow_pickle=True)
            return {key: np.asarray(raw[key], dtype=float) for key in raw.files}

        if path.suffix == ".json":
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            return {key: np.asarray(value, dtype=float) for key, value in payload.items()}

        raise ValueError(f"Unsupported Protocol 5 data format: {path.suffix}")

    def _check_assumptions(self, data: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        beta = np.asarray(data["beta"], dtype=float)
        intero = np.asarray(data["intero_precision"], dtype=float)
        extero = np.asarray(data["extero_precision"], dtype=float)
        fim_offdiag = np.asarray(data["fim_offdiag"], dtype=float)

        mean_beta = float(np.mean(beta))
        matched_gap = float(abs(np.mean(intero) - np.mean(extero)))
        mean_fim_offdiag = float(np.mean(fim_offdiag))

        precision_match_pass = matched_gap <= self.config.precision_match_tolerance
        beta_pass = mean_beta >= self.config.min_beta
        identifiability_pass = mean_fim_offdiag < self.config.identifiability_fim_max

        return {
            "precision_match_pass": precision_match_pass,
            "precision_gap": matched_gap,
            "beta_pass": beta_pass,
            "mean_beta": mean_beta,
            "identifiability_pass": identifiability_pass,
            "fim_offdiag_mean": mean_fim_offdiag,
            "overall_pass": precision_match_pass and beta_pass and identifiability_pass,
        }

    def evaluate_protocol_5(self, data: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate P5a-P5d and the critical falsification condition."""
        alpha = self.config.alpha
        assumptions = self._check_assumptions(data)

        vm_anticipation = np.asarray(data["vmPFC_anticipation"], dtype=float)
        post_anticipation = np.asarray(data["posterior_insula_anticipation"], dtype=float)
        scr_anticipation = np.asarray(data["scr_anticipation"], dtype=float)
        vm_experience = np.asarray(data["vmPFC_experience"], dtype=float)
        somato_experience = np.asarray(data["somatosensory_experience"], dtype=float)
        primary_error = np.asarray(data["primary_prediction_error"], dtype=float)
        valence = np.asarray(data["valence_signal"], dtype=float)
        contrast = np.asarray(data["sensory_contrast"], dtype=float)
        no_ant_vm = np.asarray(data["no_anticipation_vmPFC"], dtype=float)
        no_ant_post = np.asarray(data["no_anticipation_posterior_insula"], dtype=float)

        p5a_conn_r, p5a_conn_p = _safe_pearsonr(vm_anticipation, post_anticipation)
        p5a_scr_r, p5a_scr_p = _safe_pearsonr(vm_anticipation, scr_anticipation)
        p5a_pass = (
            assumptions["overall_pass"]
            and p5a_conn_r >= self.config.min_anticipatory_coupling
            and p5a_conn_p < alpha
            and p5a_scr_r > 0.0
            and p5a_scr_p < alpha
        )

        p5b_vm_pe_r, p5b_vm_pe_p = _safe_pearsonr(vm_experience, primary_error)
        p5b_vm_som_r, p5b_vm_som_p = _safe_pearsonr(vm_experience, somato_experience)
        critical_falsification = (p5b_vm_pe_p < alpha and abs(p5b_vm_pe_r) >= 0.20) or (
            p5b_vm_som_p < alpha and abs(p5b_vm_som_r) >= 0.20
        )
        p5b_pass = not critical_falsification

        valence_fit = _safe_linregress(valence, vm_anticipation)
        contrast_fit = _safe_linregress(contrast, vm_anticipation)
        p5c_pass = (
            valence_fit["p"] < alpha and abs(valence_fit["r"]) > abs(contrast_fit["r"]) and contrast_fit["p"] >= alpha
        )

        ant_coupling = vm_anticipation * post_anticipation
        no_ant_coupling = no_ant_vm * no_ant_post
        p5d_t, p5d_p = _safe_ttest_rel(ant_coupling, no_ant_coupling)
        p5d_pass = (
            np.mean(ant_coupling) > np.mean(no_ant_coupling)
            and p5d_p < alpha
            and np.mean(no_ant_coupling) < self.config.no_anticipation_coupling_max
        )

        return {
            "assumptions": assumptions,
            "P5a": {
                "passed": bool(p5a_pass),
                "vmPFC_posterior_insula_r": p5a_conn_r,
                "vmPFC_posterior_insula_p": p5a_conn_p,
                "vmPFC_scr_r": p5a_scr_r,
                "vmPFC_scr_p": p5a_scr_p,
                "alpha": alpha,
                "claim": "M-hat modulates Pi_i_eff rather than epsilon_i directly",
            },
            "P5b": {
                "passed": bool(p5b_pass),
                "critical_falsification": bool(critical_falsification),
                "vmPFC_primary_error_r": p5b_vm_pe_r,
                "vmPFC_primary_error_p": p5b_vm_pe_p,
                "vmPFC_somatosensory_r": p5b_vm_som_r,
                "vmPFC_somatosensory_p": p5b_vm_som_p,
                "alpha": alpha,
                "claim": "vmPFC should not track primary interoceptive prediction error during experience",
            },
            "P5c": {
                "passed": bool(p5c_pass),
                "valence_r": valence_fit["r"],
                "valence_p": valence_fit["p"],
                "contrast_r": contrast_fit["r"],
                "contrast_p": contrast_fit["p"],
                "alpha": alpha,
                "claim": "Somatic marker effect is valence-specific, not sensory-contrast-driven",
            },
            "P5d": {
                "passed": bool(p5d_pass),
                "anticipation_coupling_mean": float(np.mean(ant_coupling)),
                "no_anticipation_coupling_mean": float(np.mean(no_ant_coupling)),
                "paired_t": p5d_t,
                "paired_p": p5d_p,
                "alpha": alpha,
                "claim": "Removing anticipation collapses vmPFC-posterior-insula coupling",
            },
            "critical_falsification": bool(critical_falsification),
            "alpha": alpha,
        }

    def check_falsification(self, results: Mapping[str, Any]) -> str:
        """Return protocol-level status message."""
        if results.get("critical_falsification"):
            return (
                "FAIL: vmPFC tracks primary interoceptive prediction error during experience; "
                "the embodied-prioritization claim is rejected."
            )
        if not results.get("assumptions", {}).get("overall_pass", False):
            return "FAIL: soma-bias assumptions were not jointly satisfied."
        if not all(results[pred]["passed"] for pred in ("P5a", "P5b", "P5c", "P5d")):
            return "FAIL: one or more Protocol 5 sub-predictions were not supported."
        return "PASS: Protocol 5 supports anticipatory somatic-marker prioritization."

    def run_validation(self, data_path: Optional[str] = None, allow_synthetic: bool = True) -> Dict[str, Any]:
        """Execute Protocol 5 using empirical data when available, else synthetic data."""
        data_source = "synthetic"

        if data_path:
            try:
                dataset = self.load_subject_level_data(data_path)
                data_source = "empirical"
            except Exception as exc:
                logger.error("Protocol 5 empirical load failed: %s", exc)
                if not allow_synthetic:
                    return {
                        "status": "error",
                        "passed": False,
                        "protocol_id": self.protocol_id,
                        "protocol_name": self.protocol_name,
                        "data_source": "empirical",
                        "error": str(exc),
                    }
                dataset = self.simulate_subject_level_data()
        else:
            dataset = self.simulate_subject_level_data()

        results = self.evaluate_protocol_5(dataset)
        status_message = self.check_falsification(results)
        prediction_passes = [results[p]["passed"] for p in ("P5a", "P5b", "P5c", "P5d")]
        all_passed = all(prediction_passes) and not results["critical_falsification"]
        assumptions = results["assumptions"]

        named_predictions = {
            "P5a": {
                "passed": results["P5a"]["passed"],
                "actual": results["P5a"]["vmPFC_posterior_insula_r"],
                "threshold": f"vmPFC-posterior insula r >= {self.config.min_anticipatory_coupling}, p < {self.config.alpha}",
                "secondary_channel": {
                    "vmPFC_scr_r": results["P5a"]["vmPFC_scr_r"],
                    "vmPFC_scr_p": results["P5a"]["vmPFC_scr_p"],
                },
            },
            "P5b": {
                "passed": results["P5b"]["passed"],
                "actual": results["P5b"]["vmPFC_primary_error_r"],
                "threshold": f"non-significant vmPFC-primary error correlation at alpha {self.config.alpha}",
                "critical_falsification": results["P5b"]["critical_falsification"],
            },
            "P5c": {
                "passed": results["P5c"]["passed"],
                "actual": results["P5c"]["valence_r"],
                "threshold": "valence effect significant and stronger than sensory contrast",
                "contrast_r": results["P5c"]["contrast_r"],
            },
            "P5d": {
                "passed": results["P5d"]["passed"],
                "actual": results["P5d"]["anticipation_coupling_mean"]
                - results["P5d"]["no_anticipation_coupling_mean"],
                "threshold": "anticipation coupling > no-anticipation coupling, paired p < 0.01",
            },
        }

        return {
            "status": "COMPLETE" if all_passed else "FAILED",
            "passed": all_passed,
            "protocol_id": self.protocol_id,
            "protocol_name": self.protocol_name,
            "named_predictions": named_predictions,
            "criteria": results,
            "falsification_status": status_message,
            "data_source": data_source,
            "rois": list(self.rois),
            "assumption_tested": "Interoceptive signals receive preferential weighting (soma-bias) in APGI processing.",
            "assumptions": assumptions,
            "metadata": {
                "alpha": self.config.alpha,
                "min_beta": self.config.min_beta,
                "precision_match_tolerance": self.config.precision_match_tolerance,
                "identifiability_fim_max": self.config.identifiability_fim_max,
                "n_subjects": int(len(dataset["beta"])),
                "critical_falsification": results["critical_falsification"],
            },
        }


def run_validation(
    data_path: Optional[str] = None,
    allow_synthetic: bool = True,
    n_subjects: int = 36,
    alpha: float = PRIMARY_ALPHA,
    **_: Any,
) -> Dict[str, Any]:
    """Module entry point compatible with the validation suite."""
    config = Protocol5Config(n_subjects=n_subjects, alpha=alpha)
    protocol = APGIP5SomaticProtocol(config=config)
    return protocol.run_validation(data_path=data_path, allow_synthetic=allow_synthetic)


def run_protocol_main(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return standardized ProtocolResult output when schema support is available."""
    config = config or {}
    result = run_validation(
        data_path=config.get("data_path"),
        allow_synthetic=config.get("allow_synthetic", True),
        n_subjects=config.get("n_subjects", 36),
        alpha=config.get("alpha", PRIMARY_ALPHA),
    )

    if not HAS_SCHEMA:
        return result

    named_predictions: Dict[str, PredictionResult] = {}
    legacy_predictions = result.get("named_predictions", {})

    for pred_id in ("P5a", "P5b", "P5c", "P5d"):
        pred = legacy_predictions.get(pred_id, {})
        passed = bool(pred.get("passed", False))
        named_predictions[pred_id] = PredictionResult(
            passed=passed,
            value=pred.get("actual"),
            threshold=pred.get("threshold"),
            status=PredictionStatus.PASSED if passed else PredictionStatus.FAILED,
            name=pred_id,
            evidence=[],
            sources=["APGI-P03", "VP_22_FMRIAnticipationExperience"],
            metadata={k: v for k, v in pred.items() if k not in {"passed", "actual", "threshold"}},
        )

    # Registered APGI-P03 predictions. P5c (valence, not sensory contrast) tests
    # Pred 3.C. P5d measures vmPFC–posterior-insula coupling, whereas Pred 3.D
    # registers the vmPFC→aINS pathway, so it is not linked.
    named_predictions.update(
        spec_prediction_results(
            _PROTOCOL_SPEC_P03,
            named_predictions,
            {"Pred 3.C": "P5c"},
            ["VP_22_FMRIAnticipationExperience"],
        )
    )

    status = "success" if result.get("passed") else "failed"
    completion = 100

    result_metadata = result.get("metadata") or {}
    return ProtocolResult(
        protocol_id="VP_22_FMRIAnticipationExperience",
        timestamp=datetime.now().isoformat(),
        named_predictions=named_predictions,
        completion_percentage=completion,
        status=status,
        data_sources=[result.get("data_source", "synthetic"), "SCR", "ROI fMRI"],
        methodology="subject_level_somatic_marker_validation",
        errors=[],
        metadata={
            "protocol_spec_id": "APGI-P03",
            "falsification_status": result.get("falsification_status"),
            "rois": result.get("rois", []),
            "assumptions": result.get("assumptions", {}),
            "critical_falsification": result_metadata.get("critical_falsification", False),
        },
    ).to_dict()


def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects to JSON-serializable types."""
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.bool_)):
        return int(obj) if isinstance(obj, np.integer) else bool(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return str(obj) if hasattr(obj, "__dict__") else obj


def _emit_info(message: str) -> None:
    """Best-effort status output that works even when shared logging is unavailable."""
    log_info = getattr(logger, "info", None)
    if callable(log_info):
        try:
            log_info(message)
            return
        except Exception:  # nosec B110
            pass
    print(message)


def _ensure_output_path(output_path: str | Path) -> Path:
    """Create parent directories for an output path and return a Path object."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_results_json(result: Dict[str, Any], output_path: str = "protocol22_results.json") -> Path:
    """Save validation results to JSON file."""
    path = _ensure_output_path(output_path)
    serializable_result = _convert_to_serializable(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, indent=2)
    _emit_info(f"[OK] Results saved to: {path}")
    return path


def save_data_csv(data: Dict[str, np.ndarray], output_path: str = "protocol22_data.csv") -> Path:
    """Save subject-level data to CSV file."""
    path = _ensure_output_path(output_path)
    array_data = {k: np.asarray(v) for k, v in data.items() if isinstance(v, np.ndarray)}
    if not array_data:
        raise ValueError("No subject-level array data available for CSV export.")

    lengths = {k: len(v) for k, v in array_data.items()}
    n_rows = max(lengths.values())
    fieldnames = list(array_data.keys())

    if HAS_PANDAS:
        df = pd.DataFrame(array_data)
        df.to_csv(path, index=False)
    else:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(n_rows):
                row = {}
                for key in fieldnames:
                    values = array_data[key]
                    row[key] = values[idx] if idx < len(values) else ""
                writer.writerow(_convert_to_serializable(row))

    _emit_info(f"[OK] Data saved to: {path}")
    return path


def create_visualization(result: Dict[str, Any], output_path: str = "protocol22_summary.jpg") -> Optional[Path]:
    """Create and save summary visualization of Protocol 5 results."""
    if not HAS_MATPLOTLIB:
        _emit_info("matplotlib not available, skipping visualization")
        return None

    path = _ensure_output_path(output_path)

    named_predictions = result.get("named_predictions", {})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("APGI Protocol 5: fMRI Anticipation vs. Experience (Somatic Marker)", fontsize=14, fontweight="bold")

    # Plot 1: Overall validation status
    ax1 = axes[0, 0]
    passed = result.get("passed", False)
    color = "#2ecc71" if passed else "#e74c3c"
    ax1.bar(["Validation"], [1 if passed else 0], color=color)
    ax1.set_title("Validation Status")
    ax1.set_ylabel("Pass (1) / Fail (0)")
    ax1.set_ylim(0, 1.2)
    status_text = "PASSED" if passed else "FAILED"
    ax1.text(0, 0.5, status_text, ha="center", fontweight="bold", fontsize=12)

    # Plot 2: Named predictions status
    ax2 = axes[0, 1]
    if named_predictions:
        pred_names = list(named_predictions.keys())
        pred_values = [1 if named_predictions[p].get("passed", False) else 0 for p in pred_names]
        colors = ["#2ecc71" if v else "#e74c3c" for v in pred_values]
        ax2.barh(pred_names, pred_values, color=colors)
        ax2.set_title("Named Predictions Status")
        ax2.set_xlabel("Pass (1) / Fail (0)")
        ax2.set_xlim(0, 1.2)

    # Plot 3: Assumption tests
    ax3 = axes[1, 0]
    assumptions = result.get("assumptions", {})
    if assumptions:
        assumption_names = ["Precision Match", "Beta Threshold", "Identifiability"]
        assumption_values = [
            1 if assumptions.get("precision_match_pass", False) else 0,
            1 if assumptions.get("beta_pass", False) else 0,
            1 if assumptions.get("identifiability_pass", False) else 0,
        ]
        colors = ["#2ecc71" if v else "#e74c3c" for v in assumption_values]
        ax3.bar(assumption_names, assumption_values, color=colors)
        ax3.set_title("Assumption Tests")
        ax3.set_ylabel("Pass (1) / Fail (0)")
        ax3.set_ylim(0, 1.2)
        ax3.tick_params(axis="x", rotation=15)

    # Plot 4: Actual values vs thresholds
    ax4 = axes[1, 1]
    if named_predictions:
        pred_names = []
        actual_values = []
        for pred_id, pred_data in named_predictions.items():
            actual = pred_data.get("actual", 0)
            if not isinstance(actual, (int, float, np.integer, np.floating)):
                actual = 0
            pred_names.append(pred_id)
            actual_values.append(float(actual))

        if actual_values:
            ax4.bar(pred_names, actual_values, color="#3498db")
            ax4.set_title("Actual Values")
            ax4.set_ylabel("Value")
            ax4.tick_params(axis="x", rotation=15)
            ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", format="jpg")
    plt.close(fig)
    _emit_info(f"[OK] Visualization saved to: {path}")
    return path


def main(data_path: Optional[str] = None, allow_synthetic: bool = True) -> Dict[str, Any]:
    """CLI-friendly execution wrapper."""
    result = run_validation(data_path=data_path, allow_synthetic=allow_synthetic)
    print(f"Protocol P5 Results: {result['named_predictions']}")
    print(f"Falsification Status: {result['falsification_status']}")

    output_paths = {
        "json": save_results_json(result, "protocol22_results.json"),
        "csv": None,
        "jpg": None,
    }

    protocol = APGIP5SomaticProtocol()
    if data_path:
        try:
            export_data = protocol.load_subject_level_data(data_path)
        except Exception as exc:
            _emit_info(f"CSV export skipped: could not reload empirical data ({exc})")
            export_data = None
    elif allow_synthetic:
        export_data = protocol.simulate_subject_level_data()
    else:
        export_data = None

    if export_data is not None:
        output_paths["csv"] = save_data_csv(export_data, "protocol22_data.csv")

    output_paths["jpg"] = create_visualization(result, "protocol22_summary.jpg")
    result["output_files"] = {k: str(v) for k, v in output_paths.items() if v is not None}

    return result


def validate() -> str:
    """Entry point for main.py validation runner."""
    result = run_validation()
    return str(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="APGI Protocol 5: fMRI Anticipation vs. Experience")
    parser.add_argument(
        "--data-path", type=str, default=None, help="Optional path to empirical subject-level .npz/.json"
    )
    parser.add_argument("--no-synthetic", action="store_true", help="Disable synthetic fallback")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(data_path=args.data_path, allow_synthetic=not args.no_synthetic)
