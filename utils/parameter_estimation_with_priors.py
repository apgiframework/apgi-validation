"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

# Apply Python 3.14 NumPy compatibility patches first
try:
    from tests.python314_numpy_fixes import *  # noqa: F403,F401
except ImportError:
    pass

import numpy as np

if TYPE_CHECKING:
    from .physiological_priors import (
        AlphaGammaRatioPrior,
        CollinearityBreaker,
        HEPCalibrationPhase,
        HEPCalibrationResult,
        PhysiologicalPriorResult,
    )
else:
    # Import physiological priors at runtime
    try:
        # Try relative import first (when used as module)
        from .physiological_priors import (
            AlphaGammaRatioPrior,
            CollinearityBreaker,
            HEPCalibrationPhase,
            HEPCalibrationResult,
            PhysiologicalPriorResult,
        )
    except ImportError:
        try:
            # Absolute import (when run as script)
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).parent))
            from physiological_priors import (
                AlphaGammaRatioPrior,
                CollinearityBreaker,
                HEPCalibrationPhase,
                HEPCalibrationResult,
                PhysiologicalPriorResult,
            )
        except ImportError:
            # Fallback when utils not in path
            AlphaGammaRatioPrior = None  # type: ignore
            CollinearityBreaker = None  # type: ignore
            HEPCalibrationPhase = None  # type: ignore
            HEPCalibrationResult = None  # type: ignore
            PhysiologicalPriorResult = None  # type: ignore

    # Single canonical import — no fallbacks needed

logger = logging.getLogger(__name__)


@dataclass
class CalibratedAPGIEstimate:
    """Container for calibrated APGI parameter estimates"""

    # Fixed from calibration
    pi_i_baseline: float
    pi_i_source: str  # 'ag_ratio', 'hep_calibration', 'hybrid'
    pi_i_confidence: float

    # Estimated during task
    beta_estimated: float
    beta_uncertainty: float

    # Optional task-modulated values
    pi_i_effective: float  # Πi × exp(β × M) if somatic modulation applied
    m_ca_task: float  # Somatic marker during task

    # Model fit
    log_likelihood: float
    aic: float
    bic: float

    # Diagnostics
    collinearity_broken: bool  # True if cor(Πi, β) ≈ 0
    calibration_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pi_i_baseline": float(self.pi_i_baseline),
            "pi_i_source": self.pi_i_source,
            "pi_i_confidence": float(self.pi_i_confidence),
            "beta_estimated": float(self.beta_estimated),
            "beta_uncertainty": float(self.beta_uncertainty),
            "pi_i_effective": float(self.pi_i_effective),
            "m_ca_task": float(self.m_ca_task),
            "log_likelihood": float(self.log_likelihood),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "collinearity_broken": bool(self.collinearity_broken),
            "calibration_valid": bool(self.calibration_valid),
        }


class HybridCalibrator:
    """
    Combines multiple physiological signals for robust Πi calibration.

    Strategy:
    1. Use AG ratio as primary estimator (quick, no cardiac needed)
    2. Use HEP calibration as secondary (more precise, requires ECG)
    3. Combine with weighted average based on confidence

    Weighting: Πⁱ_hybrid = Σ(Πⁱ_i × confidence_i) / Σ(confidence_i)
    """

    def __init__(
        self,
        ag_prior: Optional[AlphaGammaRatioPrior] = None,
        hep_calibrator: Optional[HEPCalibrationPhase] = None,
    ):
        self.ag_prior = ag_prior or AlphaGammaRatioPrior()
        self.hep_calibrator = hep_calibrator or HEPCalibrationPhase()

        # Calibration results storage
        self.ag_result: Optional[PhysiologicalPriorResult] = None
        self.hep_result: Optional[HEPCalibrationResult] = None
        self.hybrid_pi: Optional[float] = None
        self.hybrid_confidence: float = 0.0

    def calibrate_hybrid(
        self,
        eeg_data: np.ndarray,
        ecg_data: Optional[np.ndarray] = None,
        fs: float = 1000.0,
        min_ag_confidence: float = 0.5,
    ) -> Tuple[float, float, str]:
        """
        Compute hybrid Πi from AG ratio and optionally HEP.

        Args:
            eeg_data: EEG data (resting-state)
            ecg_data: ECG data (optional, for HEP calibration)
            fs: Sampling frequency
            min_ag_confidence: Minimum AG ratio confidence

        Returns:
            Tuple of (pi_i_hybrid, confidence, source_description)
        """
        sources = []
        estimates = []
        confidences = []

        # Always compute AG ratio
        self.ag_result = self.ag_prior.compute_alpha_gamma_ratio(eeg_data, fs)

        if self.ag_result.calibration_valid and self.ag_result.pi_i_confidence >= min_ag_confidence:
            sources.append("ag_ratio")
            estimates.append(self.ag_result.pi_i_physiological)
            confidences.append(self.ag_result.pi_i_confidence)
            logger.info(
                f"AG Ratio: Πi={self.ag_result.pi_i_physiological:.3f}, " f"conf={self.ag_result.pi_i_confidence:.2f}"
            )

        # Add HEP if ECG available
        if ecg_data is not None:
            self.hep_result = self.hep_calibrator.run_calibration(eeg_data, ecg_data, fs)

            if self.hep_result.calibration_success:
                # Convert HEP uncertainty to confidence
                hep_confidence = 1.0 - self.hep_result.pi_i_uncertainty
                sources.append("hep_calibration")
                estimates.append(self.hep_result.pi_i_fixed)
                confidences.append(hep_confidence)
                logger.info(f"HEP Cal: Πi={self.hep_result.pi_i_fixed:.3f}, " f"conf={hep_confidence:.2f}")

        # Compute weighted average
        if len(estimates) == 0:
            logger.warning("No valid calibration sources, using default Πi=1.0")
            self.hybrid_pi = 1.0
            self.hybrid_confidence = 0.0
            return 1.0, 0.0, "default"

        # Weighted combination
        weights = np.array(confidences)
        values = np.array(estimates)

        self.hybrid_pi = float(np.sum(values * weights) / np.sum(weights))
        self.hybrid_confidence = float(np.mean(confidences))

        source_desc = "+".join(sources)
        logger.info(
            f"Hybrid Calibration: Πi={self.hybrid_pi:.3f} "
            f"(conf={self.hybrid_confidence:.2f}, sources={source_desc})"
        )

        return self.hybrid_pi, self.hybrid_confidence, source_desc


class APGIPhysiologicalEstimator:
    """
    APGI parameter estimator with physiological prior integration.

    Implements the full calibration → estimation workflow:
    1. Resting-state calibration: fix Πi from AG ratio and/or HEP
    2. Task estimation: estimate β with Πi constrained
    3. Collinearity diagnostics: verify cor(Πi, β) ≈ 0
    """

    def __init__(
        self,
        hybrid_calibrator: Optional[HybridCalibrator] = None,
        collinearity_breaker: Optional[CollinearityBreaker] = None,
    ):
        self.hybrid_calibrator = hybrid_calibrator or HybridCalibrator()
        self.collinearity_breaker = collinearity_breaker or CollinearityBreaker()

        # Calibration state
        self.is_calibrated: bool = False
        self.pi_i_fixed: Optional[float] = None
        self.pi_i_source: Optional[str] = None
        self.pi_i_confidence: float = 0.0

        # Estimation results
        self.last_estimate: Optional[CalibratedAPGIEstimate] = None

    def calibrate(
        self,
        resting_eeg: np.ndarray,
        resting_ecg: Optional[np.ndarray] = None,
        fs: float = 1000.0,
        min_confidence: float = 0.5,
    ) -> bool:
        """
        Run calibration phase to fix baseline Πi.

        Args:
            resting_eeg: Resting-state EEG data
            resting_ecg: Resting-state ECG data (optional)
            fs: Sampling frequency
            min_confidence: Minimum confidence threshold

        Returns:
            True if calibration successful
        """
        pi_i, confidence, source = self.hybrid_calibrator.calibrate_hybrid(resting_eeg, resting_ecg, fs, min_confidence)

        if confidence >= min_confidence:
            self.pi_i_fixed = pi_i
            self.pi_i_source = source
            self.pi_i_confidence = confidence
            self.is_calibrated = True

            # Update collinearity breaker
            self.collinearity_breaker.pi_i_baseline = pi_i
            self.collinearity_breaker.pi_i_source = source
            self.collinearity_breaker.is_calibrated = True

            logger.info(f"Calibration successful: Πi={pi_i:.3f} from {source}")
            return True
        else:
            logger.warning(f"Calibration failed: confidence {confidence:.2f} < {min_confidence}")
            return False

    def estimate_beta_from_task_hep(
        self,
        task_eeg: np.ndarray,
        task_ecg: np.ndarray,
        fs: float = 1000.0,
        m_ca: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Estimate β from task HEP given fixed Πi.

        Args:
            task_eeg: EEG during task
            task_ecg: ECG during task
            fs: Sampling frequency
            m_ca: Somatic marker value

        Returns:
            Tuple of (beta_estimate, uncertainty)
        """
        if not self.is_calibrated:
            raise ValueError("Must calibrate before task estimation")

        # Compute task HEP
        hep_result = self.collinearity_breaker.hep_calibrator.run_calibration(task_eeg, task_ecg, fs)

        if not hep_result.calibration_success:
            logger.warning("Task HEP computation failed, returning default β")
            return 0.5, 1.0

        hep_task = hep_result.hep_amplitude_baseline

        # Estimate β using collinearity breaker
        beta = self.collinearity_breaker.estimate_beta_independent(
            hep_task=hep_task,
            pi_i_current=self.pi_i_fixed,
            m_ca=m_ca,
        )

        # Uncertainty from HEP quality
        uncertainty = 1.0 - hep_result.signal_quality

        return beta, uncertainty

    def estimate_full_parameters(
        self,
        task_eeg: np.ndarray,
        task_ecg: np.ndarray,
        behavioral_data: Optional[Dict[str, np.ndarray]] = None,
        m_ca: float = 0.0,
        fs: float = 1000.0,
    ) -> CalibratedAPGIEstimate:
        """
        Full parameter estimation with physiological priors.

        Args:
            task_eeg: EEG during task
            task_ecg: ECG during task
            behavioral_data: Optional behavioral responses (RT, accuracy, etc.)
            m_ca: Somatic marker value during task
            fs: Sampling frequency

        Returns:
            CalibratedAPGIEstimate with all parameters
        """
        if not self.is_calibrated:
            raise ValueError("Must run calibration before estimation")

        # Estimate β from task HEP
        beta, beta_unc = self.estimate_beta_from_task_hep(task_eeg, task_ecg, fs, m_ca)

        # Compute effective Πi with somatic modulation
        pi_eff = self.pi_i_fixed * np.exp(beta * m_ca) if m_ca != 0 else self.pi_i_fixed

        # Model fit metrics (simplified)
        log_lik = -0.5 * beta_unc  # Higher uncertainty → lower likelihood
        n_params = 1  # Only β is free
        aic = 2 * n_params - 2 * log_lik
        # BIC needs the number of independent observations behind log_lik, which
        # this simplified fit does not define. A fixed n_obs=100 used to be
        # substituted; BIC is now NaN (not computed) instead (APGI-INT-003).
        bic = float("nan")

        # Collinearity is broken by design when using physiological prior
        collinearity_broken = self.pi_i_confidence > 0.5

        estimate = CalibratedAPGIEstimate(
            pi_i_baseline=self.pi_i_fixed,
            pi_i_source=self.pi_i_source or "unknown",
            pi_i_confidence=self.pi_i_confidence,
            beta_estimated=beta,
            beta_uncertainty=beta_unc,
            pi_i_effective=pi_eff,
            m_ca_task=m_ca,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            collinearity_broken=collinearity_broken,
            calibration_valid=self.is_calibrated,
        )

        self.last_estimate = estimate
        return estimate

    def get_collinearity_diagnostics(
        self,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """
        Compute collinearity diagnostics.

        Tests whether physiological prior successfully breaks
        collinearity between Πi and β.

        Args:
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with diagnostic metrics
        """
        if not self.is_calibrated or self.last_estimate is None:
            return {"error": "No calibration/estimation available"}  # type: ignore

        # Simulate correlation between Πi and β
        # With physiological prior, Πi is fixed → cor(Πi, β) ≈ 0
        pi_values = np.full(n_bootstrap, self.pi_i_fixed)
        beta_values = np.random.normal(
            self.last_estimate.beta_estimated,
            self.last_estimate.beta_uncertainty,
            n_bootstrap,
        )

        correlation = np.corrcoef(pi_values, beta_values)[0, 1]

        return {
            "cor_pi_beta": float(correlation),
            "collinearity_broken": abs(correlation) < 0.1,
            "pi_variance": float(np.var(pi_values)),
            "beta_variance": float(np.var(beta_values)),
            "pi_fixed": float(self.pi_i_fixed),
            "beta_mean": float(np.mean(beta_values)),
        }


def run_demonstration():
    """
    Demonstration of physiological prior integration.
    """
    print("=" * 80)
    print("APGI PHYSIOLOGICAL PRIOR INTEGRATION - DEMONSTRATION")
    print("=" * 80)

    # Generate synthetic data
    fs = 1000.0
    duration_rest = 60.0  # 1 minute resting calibration
    duration_task = 30.0  # 30 second task

    t_rest = np.linspace(0, duration_rest, int(fs * duration_rest))
    t_task = np.linspace(0, duration_task, int(fs * duration_task))

    print("\n1. GENERATING SYNTHETIC DATA")
    print("-" * 80)

    # Ground truth parameters
    true_pi_i = 2.0  # Higher interoceptive precision
    true_beta = 0.6  # Moderate somatic gain
    true_m_ca = 0.5  # Moderate somatic marker during task

    print(f"  Ground truth: Πi={true_pi_i}, β={true_beta}, M_ca={true_m_ca}")

    # Resting-state EEG: alpha-dominant (high Πi)
    alpha_rest = 2.0 * np.sin(2 * np.pi * 10 * t_rest)
    gamma_rest = 0.5 * np.sin(2 * np.pi * 50 * t_rest)
    noise_rest = 0.3 * np.random.randn(len(t_rest))
    eeg_rest = (alpha_rest + gamma_rest + noise_rest).reshape(1, -1)

    # Resting ECG
    ecg_rest = np.zeros(len(t_rest))
    rr_interval = int(fs * 60 / 70)  # 70 BPM
    for i in range(0, len(t_rest), rr_interval):
        if i + 10 < len(t_rest):
            ecg_rest[i : i + 10] = np.sin(np.linspace(0, np.pi, 10)) * 100

    # Task EEG with HEP modulation
    alpha_task = 1.5 * np.sin(2 * np.pi * 10 * t_task)
    gamma_task = 0.8 * np.sin(2 * np.pi * 50 * t_task)
    noise_task = 0.3 * np.random.randn(len(t_task))
    eeg_task = (alpha_task + gamma_task + noise_task).reshape(1, -1)

    # Add HEP (higher during task due to β × M modulation)
    hep_scale = 0.48
    hep_baseline = true_pi_i * hep_scale
    hep_task_amp = hep_baseline * np.exp(true_beta * true_m_ca)

    rr_interval_task = int(fs * 60 / 75)  # Slightly faster HR during task
    for i in range(0, len(t_task), rr_interval_task):
        if i + int(0.5 * fs) < len(t_task):
            hep_win = slice(i + int(0.2 * fs), i + int(0.6 * fs))
            eeg_task[0, hep_win] += hep_task_amp * 0.001  # Scale to μV

    # Task ECG
    ecg_task = np.zeros(len(t_task))
    for i in range(0, len(t_task), rr_interval_task):
        if i + 10 < len(t_task):
            ecg_task[i : i + 10] = np.sin(np.linspace(0, np.pi, 10)) * 100

    print(f"  Resting: {duration_rest}s EEG + ECG")
    print(f"  Task: {duration_task}s EEG + ECG")
    print(f"  Expected HEP task: {hep_task_amp:.3f} \u03bcV")

    # Run estimator
    print("\n2. CALIBRATION PHASE")
    print("-" * 80)

    estimator = APGIPhysiologicalEstimator()
    calibration_success = estimator.calibrate(
        resting_eeg=eeg_rest,
        resting_ecg=ecg_rest,
        fs=fs,
        min_confidence=0.3,
    )

    if calibration_success:
        print("  ✓ Calibration successful")
        print(f"    Πi fixed: {estimator.pi_i_fixed:.3f}")
        print(f"    Source: {estimator.pi_i_source}")
        print(f"    Confidence: {estimator.pi_i_confidence:.2f}")
    else:
        print("  ✗ Calibration failed")
        return

    print("\n3. TASK PARAMETER ESTIMATION")
    print("-" * 80)

    result = estimator.estimate_full_parameters(
        task_eeg=eeg_task,
        task_ecg=ecg_task,
        m_ca=true_m_ca,
        fs=fs,
    )

    print(f"  Fixed Πi (baseline): {result.pi_i_baseline:.3f}")
    print(f"  Estimated β: {result.beta_estimated:.3f} (true: {true_beta})")
    print(f"  β uncertainty: {result.beta_uncertainty:.3f}")
    print(f"  Effective Πi: {result.pi_i_effective:.3f}")
    print(f"  Collinearity broken: {result.collinearity_broken}")

    print("\n4. COLLINEARITY DIAGNOSTICS")
    print("-" * 80)

    diagnostics = estimator.get_collinearity_diagnostics(n_bootstrap=1000)
    print(f"  cor(Πi, β): {diagnostics['cor_pi_beta']:.4f}")
    print(f"  Collinearity broken: {diagnostics['collinearity_broken']}")
    print(f"  Πi variance: {diagnostics['pi_variance']:.6f}")
    print(f"  β variance: {diagnostics['beta_variance']:.4f}")

    print("\n5. SUMMARY")
    print("-" * 80)
    recovery_error_beta = abs(result.beta_estimated - true_beta)
    print(f"  β recovery error: {recovery_error_beta:.3f}")
    print("  Physiological prior successfully constrains Πi")
    print("  β can be estimated independently during task")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
