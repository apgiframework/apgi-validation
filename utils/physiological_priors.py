"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# NumPy 1.26 (locked) has only trapz; NumPy 2 renamed it trapezoid (APGI-REP-005).
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# Create a minimal signal module fallback for scipy compatibility issues
class SignalFallback:
    @staticmethod
    def welch(x, fs, nperseg=None, window="hann"):
        """Fallback Welch method implementation."""
        if nperseg is None:
            nperseg = min(1024, len(x) // 2)  # Use larger nperseg for better frequency resolution

        # Simple FFT-based PSD
        nfft = nperseg
        if len(x) < nfft:
            x = np.pad(x, (0, nfft - len(x)), "constant")

        # Apply window
        if window == "hann":
            w = np.hanning(nfft)
        else:
            w = np.ones(nfft)

        x_windowed = x[:nfft] * w
        fft = np.fft.fft(x_windowed)
        psd = np.abs(fft) ** 2 / (np.sum(w**2) * fs)

        freqs = np.fft.fftfreq(nfft, 1 / fs)
        # Return only positive frequencies
        pos_freq_idx = freqs >= 0
        return freqs[pos_freq_idx], psd[pos_freq_idx]


# Try to import scipy, but use fallback if it fails
try:
    from scipy import signal
except (ImportError, TypeError, AttributeError) as e:
    signal = SignalFallback()
    print(f"Warning: Using fallback signal implementation due to scipy compatibility issues: {e}")

# Import centralized spectral band constants
try:
    from .constants import EEG_ALPHA_BAND_HZ, EEG_GAMMA_BAND_HZ
except ImportError:
    try:
        from utils.constants import EEG_ALPHA_BAND_HZ, EEG_GAMMA_BAND_HZ
    except ImportError:
        # Fallback values if constants unavailable
        EEG_ALPHA_BAND_HZ = (8.0, 12.0)
        EEG_GAMMA_BAND_HZ = (30.0, 80.0)

logger = logging.getLogger(__name__)


@dataclass
class PhysiologicalPriorResult:
    """Container for physiological prior estimation results"""

    # Alpha/Gamma ratio metrics
    alpha_power: float
    gamma_power: float
    ag_ratio: float  # Alpha/Gamma power ratio
    ag_ratio_z: float  # Z-scored ratio (normalized)

    # Derived precision estimate
    pi_i_physiological: float  # Πi estimated from physiology
    pi_i_confidence: float  # Confidence in estimate (0-1)

    # Calibration status
    calibration_valid: bool
    warnings: list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_power": float(self.alpha_power),
            "gamma_power": float(self.gamma_power),
            "ag_ratio": float(self.ag_ratio),
            "ag_ratio_z": float(self.ag_ratio_z),
            "pi_i_physiological": float(self.pi_i_physiological),
            "pi_i_confidence": float(self.pi_i_confidence),
            "calibration_valid": bool(self.calibration_valid),
            "warnings": self.warnings,
        }


@dataclass
class HEPCalibrationResult:
    """Container for HEP calibration results"""

    # Resting-state HEP metrics
    hep_amplitude_baseline: float  # Mean HEP during rest (μV)
    hep_amplitude_std: float  # Standard deviation
    hep_trials_used: int  # Number of R-peaks/HEP epochs

    # Derived precision estimate
    pi_i_fixed: float  # Baseline Πi fixed from HEP
    pi_i_uncertainty: float  # Uncertainty in estimate

    # Calibration parameters
    calibration_duration_sec: float
    heart_rate_bpm: float
    signal_quality: float  # 0-1 quality metric

    # Status
    calibration_success: bool
    warnings: list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hep_amplitude_baseline": float(self.hep_amplitude_baseline),
            "hep_amplitude_std": float(self.hep_amplitude_std),
            "hep_trials_used": int(self.hep_trials_used),
            "pi_i_fixed": float(self.pi_i_fixed),
            "pi_i_uncertainty": float(self.pi_i_uncertainty),
            "calibration_duration_sec": float(self.calibration_duration_sec),
            "heart_rate_bpm": float(self.heart_rate_bpm),
            "signal_quality": float(self.signal_quality),
            "calibration_success": bool(self.calibration_success),
            "warnings": self.warnings,
        }


class AlphaGammaRatioPrior:
    """
    Compute Alpha/Gamma power ratio as physiological prior for Πi.

    Theory: The alpha/gamma ratio reflects cortical excitation/inhibition
    balance, serving as a proxy for interoceptive precision. Higher alpha
    relative to gamma indicates heightened interoceptive sensitivity.

    The physiological prior is computed as:
        Πⁱ_physiological = f(AG_ratio) = baseline × (1 + κ × AG_ratio_z)

    where κ is the coupling constant derived from empirical data.
    """

    # Empirical coupling constant (κ)
    # Based on: Jones et al. (2010), Allen et al. (2011)
    # AG ratio → Πi scaling factor
    AG_TO_PI_COUPLING: float = 0.35  # κ: unitless coupling

    # Physiological bounds for Πi
    PI_MIN: float = 0.1
    PI_MAX: float = 10.0

    # Reference population statistics (for z-scoring)
    # From: large-scale EEG datasets (n > 1000)
    REF_AG_RATIO_MEAN: float = 1.2  # Typical AG ratio at rest
    REF_AG_RATIO_STD: float = 0.4  # Population variability

    def __init__(
        self,
        coupling_constant: Optional[float] = None,
        reference_mean: Optional[float] = None,
        reference_std: Optional[float] = None,
    ):
        """
        Initialize Alpha/Gamma ratio prior estimator.

        Args:
            coupling_constant: AG ratio → Πi coupling (κ). Default 0.35.
            reference_mean: Population mean AG ratio for z-scoring. Default 1.2.
            reference_std: Population std AG ratio for z-scoring. Default 0.4.
        """
        self.coupling = coupling_constant or self.AG_TO_PI_COUPLING
        self.ref_mean = reference_mean or self.REF_AG_RATIO_MEAN
        self.ref_std = reference_std or self.REF_AG_RATIO_STD

    def compute_band_power(
        self,
        eeg_data: np.ndarray,
        fs: float,
        freq_range: Tuple[float, float],
    ) -> float:
        """
        Compute band power using Welch's method.

        Args:
            eeg_data: EEG data (channels × samples or 1D)
            fs: Sampling frequency in Hz
            freq_range: Frequency band (low, high) in Hz

        Returns:
            Band power (integrated PSD)
        """
        # Ensure 2D
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)

        total_power = 0.0
        n_channels = eeg_data.shape[0]

        for ch in range(n_channels):
            # Welch PSD
            f, psd = signal.welch(
                eeg_data[ch],
                fs=fs,
                nperseg=min(256, len(eeg_data[ch]) // 4),
                window="hann",
            )

            # Extract frequency band
            freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
            if np.any(freq_mask):
                band_freqs = f[freq_mask]
                band_psd = psd[freq_mask]

                # Integrate PSD over frequency.
                # If the mask captures a single FFT bin (common with short nperseg),
                # trapezoid integration returns 0. Approximate area using Δf.
                if band_freqs.size >= 2:
                    band_power = _trapezoid(band_psd, band_freqs)
                else:
                    if f.size >= 2:
                        df = float(f[1] - f[0])
                    else:
                        df = float(freq_range[1] - freq_range[0])
                    band_power = float(band_psd[0]) * df
                total_power += band_power

        # Average across channels
        return total_power / n_channels if n_channels > 0 else 0.0

    def compute_alpha_gamma_ratio(
        self,
        eeg_data: np.ndarray,
        fs: float = 1000.0,
        alpha_band: Tuple[float, float] = EEG_ALPHA_BAND_HZ,
        gamma_band: Tuple[float, float] = EEG_GAMMA_BAND_HZ,
    ) -> PhysiologicalPriorResult:
        """
        Compute Alpha/Gamma power ratio and derive Πi physiological prior.

        Args:
            eeg_data: EEG data array (channels × samples or 1D)
            fs: Sampling frequency in Hz
            alpha_band: Alpha frequency range (default 8-12 Hz)
            gamma_band: Gamma frequency range (default 30-80 Hz)

        Returns:
            PhysiologicalPriorResult with AG ratio and derived Πi
        """
        warnings = []

        # Validate input (note: eeg_data is 2D, so n_channels is shape[0])
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)

        n_channels, n_samples = eeg_data.shape

        if n_samples < fs * 2:  # Need at least 2 seconds
            warnings.append(f"Short data: {n_samples / fs:.1f}s (recommend >2s)")

        # Compute alpha power
        alpha_power = self.compute_band_power(eeg_data, fs, alpha_band)

        # Compute gamma power
        gamma_power = self.compute_band_power(eeg_data, fs, gamma_band)

        # Check for valid power values
        if alpha_power <= 0 or gamma_power <= 0:
            warnings.append("Invalid power values detected")
            return PhysiologicalPriorResult(
                alpha_power=alpha_power,
                gamma_power=gamma_power,
                ag_ratio=0.0,
                ag_ratio_z=0.0,
                pi_i_physiological=1.0,  # Default neutral
                pi_i_confidence=0.0,
                calibration_valid=False,
                warnings=warnings,
            )

        # Compute AG ratio
        ag_ratio = alpha_power / gamma_power

        # Z-score relative to population
        ag_ratio_z = (ag_ratio - self.ref_mean) / self.ref_std

        # Derive Πi from AG ratio
        # Πⁱ = baseline × (1 + κ × z-score)
        # Baseline Πi = 1.0 (neutral precision)
        pi_baseline = 1.0
        pi_i_phys = pi_baseline * (1.0 + self.coupling * ag_ratio_z)

        # Clip to physiological bounds
        pi_i_phys = np.clip(pi_i_phys, self.PI_MIN, self.PI_MAX)

        # Compute confidence based on data quality
        # Factors: data length, SNR, ratio magnitude
        snr_alpha = self._estimate_snr(eeg_data, fs, alpha_band)
        snr_gamma = self._estimate_snr(eeg_data, fs, gamma_band)
        avg_snr = (snr_alpha + snr_gamma) / 2

        # Confidence: higher with more data and better SNR
        confidence = min(1.0, avg_snr / 10.0)  # Normalize SNR to 0-1
        if n_samples < fs * 5:  # Less than 5 seconds
            confidence *= 0.7

        # Valid if we have reasonable power values
        calibration_valid = alpha_power > 0 and gamma_power > 0

        logger.info(
            f"AG Ratio: {ag_ratio:.3f} (z={ag_ratio_z:.2f}) -> " f"Πi={pi_i_phys:.3f} (confidence={confidence:.2f})"
        )

        return PhysiologicalPriorResult(
            alpha_power=alpha_power,
            gamma_power=gamma_power,
            ag_ratio=ag_ratio,
            ag_ratio_z=ag_ratio_z,
            pi_i_physiological=pi_i_phys,
            pi_i_confidence=confidence,
            calibration_valid=calibration_valid,
            warnings=warnings,
        )

    def _estimate_snr(
        self,
        eeg_data: np.ndarray,
        fs: float,
        freq_range: Tuple[float, float],
    ) -> float:
        """Estimate SNR in a frequency band"""
        if eeg_data.ndim == 2:
            # Use first channel
            data = eeg_data[0]
        else:
            data = eeg_data

        # Compute PSD
        f, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data) // 4))

        # Signal band
        signal_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        # Noise band (neighboring frequencies)
        noise_mask = ((f >= freq_range[0] - 5) & (f < freq_range[0])) | ((f > freq_range[1]) & (f <= freq_range[1] + 5))

        signal_power = np.mean(psd[signal_mask]) if np.any(signal_mask) else 1e-10
        noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 1e-10

        return 10 * np.log10(signal_power / noise_power + 1e-10)


class HEPCalibrationPhase:
    """
    Automated calibration phase using Resting-State Heartbeat Evoked Potential.

    Records HEP during resting-state (typically 2-5 minutes) to estimate
    baseline interoceptive precision (Πi), which is then fixed during the
    primary task. This allows β to be treated as the sole free parameter.

    Theory: Resting-state HEP amplitude reflects tonic interoceptive precision
    independent of task-related modulation. By fixing Πi from this baseline,
    we disambiguate precision (Πi) from gain modulation (β × M_ca).

    Equation:
        HEP_rest ∝ Πⁱ_baseline × M_baseline

    With neutral somatic markers at rest (M ≈ 0), HEP directly indexes Πi:
        Πⁱ_baseline ≈ HEP_rest / HEP_scale_factor
    """

    # HEP scale factor: μV per unit precision
    # From: Park et al. (2014), Pollatos et al. (2007)
    HEP_SCALE_FACTOR: float = 0.48  # μV per unit Πi

    # Minimum calibration duration
    MIN_CALIBRATION_SEC: float = 60.0  # 1 minute minimum
    RECOMMENDED_CALIBRATION_SEC: float = 180.0  # 3 minutes recommended

    # HEP window: 200-600ms post R-peak (standard in literature)
    HEP_WINDOW_START_MS: float = 200.0
    HEP_WINDOW_END_MS: float = 600.0

    # Minimum R-peaks for reliable HEP
    MIN_R_PEAKS: int = 30  # Minimum 30 heartbeats

    def __init__(
        self,
        hep_scale_factor: Optional[float] = None,
        min_calibration_sec: Optional[float] = None,
    ):
        """
        Initialize HEP calibration phase.

        Args:
            hep_scale_factor: μV per unit precision. Default 0.48.
            min_calibration_sec: Minimum calibration duration. Default 60s.
        """
        self.hep_scale = hep_scale_factor or self.HEP_SCALE_FACTOR
        self.min_duration = min_calibration_sec or self.MIN_CALIBRATION_SEC

    def detect_r_peaks(
        self,
        ecg_data: np.ndarray,
        fs: float = 1000.0,
    ) -> np.ndarray:
        """
        Detect R-peaks in ECG data.

        Args:
            ecg_data: ECG signal (1D array)
            fs: Sampling frequency in Hz

        Returns:
            Array of R-peak sample indices
        """
        # Bandpass filter for QRS complex (5-15 Hz)
        b, a = signal.butter(4, [5, 15], btype="bandpass", fs=fs)
        filtered_ecg = signal.filtfilt(b, a, ecg_data)

        # Find peaks with minimum distance (300ms between R-peaks)
        min_distance = int(0.3 * fs)
        prominence = np.std(filtered_ecg) * 0.5

        peaks, _ = signal.find_peaks(
            filtered_ecg,
            distance=min_distance,
            prominence=prominence,
        )

        return peaks

    def epoch_eeg_around_r_peaks(
        self,
        eeg_data: np.ndarray,
        r_peaks: np.ndarray,
        fs: float,
        tmin: float = -0.2,
        tmax: float = 0.8,
    ) -> np.ndarray:
        """
        Create EEG epochs around R-peaks for HEP computation.

        Args:
            eeg_data: EEG data (channels × samples)
            r_peaks: R-peak indices
            fs: Sampling frequency
            tmin: Start time relative to R-peak (seconds)
            tmax: End time relative to R-peak (seconds)

        Returns:
            Epochs array (n_epochs × n_channels × n_samples)
        """
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)

        n_samples_epoch = int((tmax - tmin) * fs)

        epochs = []
        for r_peak in r_peaks:
            start = int(r_peak + tmin * fs)
            end = start + n_samples_epoch

            # Check bounds
            if start >= 0 and end <= eeg_data.shape[1]:
                epoch = eeg_data[:, start:end]
                if epoch.shape[1] == n_samples_epoch:
                    epochs.append(epoch)

        return np.array(epochs) if epochs else np.array([])

    def compute_hep_amplitude(
        self,
        epochs: np.ndarray,
        fs: float,
        window_start_ms: float = 200.0,
        window_end_ms: float = 600.0,
    ) -> Tuple[float, float]:
        """
        Compute HEP amplitude from epochs.

        Args:
            epochs: EEG epochs (n_epochs × n_channels × n_samples)
            fs: Sampling frequency
            window_start_ms: HEP window start post R-peak (ms)
            window_end_ms: HEP window end post R-peak (ms)

        Returns:
            Tuple of (mean_amplitude, std_amplitude) in μV
        """
        if epochs.size == 0:
            return 0.0, 0.0

        # Convert ms to samples
        tmin = -0.2  # Epoch starts 200ms before R-peak
        start_sample = int((window_start_ms / 1000 - tmin) * fs)
        end_sample = int((window_end_ms / 1000 - tmin) * fs)

        # Extract HEP window
        hep_window = epochs[:, :, start_sample:end_sample]

        # Average across time within window
        hep_mean_time = np.mean(hep_window, axis=2)  # n_epochs × n_channels

        # Average across epochs and channels
        mean_amplitude = float(np.mean(hep_mean_time))
        std_amplitude = float(np.std(hep_mean_time))

        return mean_amplitude, std_amplitude

    def run_calibration(
        self,
        eeg_data: np.ndarray,
        ecg_data: np.ndarray,
        fs: float = 1000.0,
        duration_sec: Optional[float] = None,
    ) -> HEPCalibrationResult:
        """
        Run resting-state HEP calibration phase.

        Args:
            eeg_data: EEG data (channels × samples)
            ecg_data: ECG data (1D array)
            fs: Sampling frequency in Hz
            duration_sec: Actual calibration duration (auto-detected if None)

        Returns:
            HEPCalibrationResult with baseline Πi estimate
        """
        warnings = []

        # Determine duration
        if duration_sec is None:
            duration_sec = len(ecg_data) / fs

        # Check minimum duration
        if duration_sec < self.min_duration:
            warnings.append(f"Short calibration: {duration_sec:.1f}s < {self.min_duration}s recommended")

        # Detect R-peaks
        r_peaks = self.detect_r_peaks(ecg_data, fs)
        n_peaks = len(r_peaks)

        if n_peaks < self.MIN_R_PEAKS:
            warnings.append(f"Insufficient R-peaks: {n_peaks} < {self.MIN_R_PEAKS}")
            return HEPCalibrationResult(
                hep_amplitude_baseline=0.0,
                hep_amplitude_std=0.0,
                hep_trials_used=0,
                pi_i_fixed=1.0,  # Default neutral
                pi_i_uncertainty=1.0,
                calibration_duration_sec=duration_sec,
                heart_rate_bpm=0.0,
                signal_quality=0.0,
                calibration_success=False,
                warnings=warnings,
            )

        # Compute heart rate
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = np.mean(rr_intervals)
        heart_rate_bpm = float(60.0 / mean_rr if mean_rr > 0 else 0.0)

        # Epoch EEG around R-peaks
        epochs = self.epoch_eeg_around_r_peaks(eeg_data, r_peaks, fs)

        if epochs.size == 0:
            warnings.append("Could not create epochs around R-peaks")
            return HEPCalibrationResult(
                hep_amplitude_baseline=0.0,
                hep_amplitude_std=0.0,
                hep_trials_used=0,
                pi_i_fixed=1.0,
                pi_i_uncertainty=1.0,
                calibration_duration_sec=duration_sec,
                heart_rate_bpm=heart_rate_bpm,
                signal_quality=0.0,
                calibration_success=False,
                warnings=warnings,
            )

        # Compute HEP amplitude
        hep_mean, hep_std = self.compute_hep_amplitude(
            epochs,
            fs,
            window_start_ms=self.HEP_WINDOW_START_MS,
            window_end_ms=self.HEP_WINDOW_END_MS,
        )

        # Estimate baseline Πi from HEP
        # HEP = Πi × scale_factor (assuming neutral M_ca at rest)
        # Therefore: Πi = HEP / scale_factor
        pi_i_fixed = hep_mean / self.hep_scale

        # Clip to physiological bounds
        pi_i_fixed = np.clip(pi_i_fixed, 0.1, 10.0)

        # Uncertainty decreases with more trials
        # Heuristic: uncertainty ∝ 1/√n
        pi_i_uncertainty = 1.0 / np.sqrt(n_peaks)

        # Signal quality: based on HEP SNR and number of trials
        hep_snr = hep_mean / (hep_std + 1e-10)
        signal_quality = min(1.0, (hep_snr / 3.0) * (n_peaks / 100))

        logger.info(
            f"HEP Calibration: {n_peaks} R-peaks, HEP={hep_mean:.2f}μV, "
            f"Πi_fixed={pi_i_fixed:.3f} (HR={heart_rate_bpm:.1f}bpm)"
        )

        return HEPCalibrationResult(
            hep_amplitude_baseline=hep_mean,
            hep_amplitude_std=hep_std,
            hep_trials_used=n_peaks,
            pi_i_fixed=pi_i_fixed,
            pi_i_uncertainty=pi_i_uncertainty,
            calibration_duration_sec=duration_sec,
            heart_rate_bpm=heart_rate_bpm,
            signal_quality=signal_quality,
            calibration_success=True,
            warnings=warnings,
        )


class CollinearityBreaker:
    """
    Break collinearity between Πi and β using physiological priors.

    When Πi is constrained by biological measurements (alpha/gamma ratio
    or HEP calibration), β can be estimated independently during active
    task performance. This resolves the statistical identifiability problem.

    Methods:
        1. Physiological prior: Πi ~ AG_ratio (breaks cor(Πi, AG_ratio))
        2. Calibration lock: Πi = HEP_rest / scale (fixes Πi, frees β)
    """

    def __init__(
        self,
        ag_prior: Optional[AlphaGammaRatioPrior] = None,
        hep_calibrator: Optional[HEPCalibrationPhase] = None,
    ):
        """
        Initialize collinearity breaker.

        Args:
            ag_prior: Alpha/Gamma ratio prior estimator
            hep_calibrator: HEP calibration phase handler
        """
        self.ag_prior = ag_prior or AlphaGammaRatioPrior()
        self.hep_calibrator = hep_calibrator or HEPCalibrationPhase()

        # Storage for calibrated values
        self.pi_i_baseline: Optional[float] = None
        self.pi_i_source: Optional[str] = None  # 'ag_ratio' or 'hep_calibration'
        self.is_calibrated: bool = False

    def calibrate_from_ag_ratio(
        self,
        eeg_data: np.ndarray,
        fs: float = 1000.0,
        confidence_threshold: float = 0.5,
    ) -> bool:
        """
        Calibrate Πi from alpha/gamma ratio.

        Args:
            eeg_data: Resting-state EEG data
            fs: Sampling frequency
            confidence_threshold: Minimum confidence for valid calibration

        Returns:
            True if calibration successful
        """
        result = self.ag_prior.compute_alpha_gamma_ratio(eeg_data, fs)

        if result.calibration_valid and result.pi_i_confidence >= confidence_threshold:
            self.pi_i_baseline = result.pi_i_physiological
            self.pi_i_source = "ag_ratio"
            self.is_calibrated = True

            logger.info(
                f"AG Ratio Calibration: Πi={self.pi_i_baseline:.3f} " f"(confidence={result.pi_i_confidence:.2f})"
            )
            return True
        else:
            logger.warning(
                f"AG Ratio Calibration failed: valid={result.calibration_valid}, "
                f"confidence={result.pi_i_confidence:.2f}"
            )
            return False

    def calibrate_from_hep(
        self,
        eeg_data: np.ndarray,
        ecg_data: np.ndarray,
        fs: float = 1000.0,
    ) -> bool:
        """
        Calibrate Πi from resting-state HEP.

        Args:
            eeg_data: EEG data during calibration
            ecg_data: ECG data during calibration
            fs: Sampling frequency

        Returns:
            True if calibration successful
        """
        result = self.hep_calibrator.run_calibration(eeg_data, ecg_data, fs)

        if result.calibration_success:
            self.pi_i_baseline = result.pi_i_fixed
            self.pi_i_source = "hep_calibration"
            self.is_calibrated = True

            logger.info(f"HEP Calibration: Πi={self.pi_i_baseline:.3f} " f"(uncertainty={result.pi_i_uncertainty:.3f})")
            return True
        else:
            logger.warning("HEP Calibration failed")
            return False

    def get_pi_i_for_task(
        self,
        modulation_factor: float = 1.0,
    ) -> float:
        """
        Get calibrated Πi for active task.

        During task execution, Πi can be modulated by somatic markers:
            Πⁱ_task = Πⁱ_baseline × modulation_factor

        But the baseline remains fixed from calibration.

        Args:
            modulation_factor: Multiplicative modulation (default 1.0 = no change)

        Returns:
            Calibrated Πi value
        """
        if not self.is_calibrated or self.pi_i_baseline is None:
            logger.warning("No calibration available, using default Πi=1.0")
            return 1.0

        return self.pi_i_baseline * modulation_factor

    def estimate_beta_independent(
        self,
        hep_task: float,
        pi_i_current: Optional[float] = None,
        m_ca: float = 0.0,
    ) -> float:
        """
        Estimate β independently given fixed Πi.

        With Πi fixed from calibration, we can solve for β:
            HEP_task = Πⁱ × exp(β × M_ca) × scale
            β = ln(HEP_task / (Πⁱ × scale)) / M_ca

        Args:
            hep_task: HEP amplitude during task
            pi_i_current: Current Πi (uses baseline if None)
            m_ca: Somatic marker value

        Returns:
            Estimated β value
        """
        if pi_i_current is None:
            pi_i_current = self.get_pi_i_for_task()

        if m_ca == 0:
            # Cannot estimate β without somatic modulation
            # Return default value
            return 0.5

        # Invert the HEP equation
        # HEP = Πi × exp(β × M) × scale
        # β = ln(HEP / (Πi × scale)) / M
        expected_baseline = pi_i_current * self.hep_calibrator.hep_scale

        if expected_baseline <= 0 or hep_task <= 0:
            return 0.5

        beta_estimated = np.log(hep_task / expected_baseline) / m_ca  # type: ignore[operator]

        # Clip to physiological bounds
        beta_estimated = np.clip(beta_estimated, 0.3, 0.8)

        return beta_estimated


def quick_test():
    """Quick test of physiological priors functionality"""
    print("=" * 70)
    print("PHYSIOLOGICAL PRIORS - QUICK TEST")
    print("=" * 70)

    # Test 1: Alpha/Gamma Ratio
    print("\n1. Testing Alpha/Gamma Ratio Prior")
    print("-" * 70)

    # Generate synthetic EEG with known alpha/gamma ratio
    fs = 1000.0
    duration = 10.0
    t = np.linspace(0, duration, int(fs * duration))

    # Create alpha-dominant signal (high Πi)
    alpha_component = 2.0 * np.sin(2 * np.pi * 10 * t)
    gamma_component = 0.5 * np.sin(2 * np.pi * 50 * t)
    noise = 0.3 * np.random.randn(len(t))
    eeg_alpha_dominant = (alpha_component + gamma_component + noise).reshape(1, -1)

    ag_prior = AlphaGammaRatioPrior()
    result = ag_prior.compute_alpha_gamma_ratio(eeg_alpha_dominant, fs)

    print(f"  Alpha Power: {result.alpha_power:.4f}")
    print(f"  Gamma Power: {result.gamma_power:.4f}")
    print(f"  AG Ratio: {result.ag_ratio:.3f}")
    print(f"  → Πi estimate: {result.pi_i_physiological:.3f}")
    print(f"  Confidence: {result.pi_i_confidence:.2f}")
    print(f"  Valid: {result.calibration_valid}")

    # Test 2: HEP Calibration
    print("\n2. Testing HEP Calibration Phase")
    print("-" * 70)

    # Generate synthetic ECG with R-peaks
    ecg = np.zeros(len(t))
    heart_rate = 60  # BPM
    rr_interval = int(fs * 60 / heart_rate)
    for i in range(0, len(t), rr_interval):
        if i + 10 < len(t):
            ecg[i : i + 10] = np.sin(np.linspace(0, np.pi, 10)) * 100  # QRS complex

    # Add synthetic HEP in EEG (heartbeat-related potential)
    eeg_with_hep = eeg_alpha_dominant.copy()
    for i in range(0, len(t), rr_interval):
        if i + int(0.5 * fs) < len(t):
            hep_window = slice(i + int(0.2 * fs), i + int(0.6 * fs))
            eeg_with_hep[0, hep_window] += 0.5  # 0.5 μV HEP

    hep_cal = HEPCalibrationPhase()
    hep_result = hep_cal.run_calibration(eeg_with_hep, ecg, fs)

    print(f"  HEP Amplitude: {hep_result.hep_amplitude_baseline:.3f} μV")
    print(f"  R-peaks used: {hep_result.hep_trials_used}")
    print(f"  Heart Rate: {hep_result.heart_rate_bpm:.1f} BPM")
    print(f"  → Πi fixed: {hep_result.pi_i_fixed:.3f}")
    print(f"  Signal Quality: {hep_result.signal_quality:.2f}")
    print(f"  Success: {hep_result.calibration_success}")

    # Test 3: Collinearity Breaker
    print("\n3. Testing Collinearity Breaker")
    print("-" * 70)

    breaker = CollinearityBreaker()

    # Calibrate from AG ratio
    breaker.calibrate_from_ag_ratio(eeg_alpha_dominant, fs)
    pi_val = breaker.pi_i_baseline if breaker.pi_i_baseline is not None else 1.0
    print(f"  Calibrated from AG ratio: Πi={pi_val:.3f}")

    # Get Πi for task
    pi_task = breaker.get_pi_i_for_task(modulation_factor=1.0)
    print(f"  Πi for task: {pi_task:.3f}")

    # Estimate β given fixed Πi
    hep_task = 0.6  # μV during task (higher than baseline)
    m_ca = 0.5  # Moderate somatic marker
    beta_est = breaker.estimate_beta_independent(hep_task, pi_task, m_ca)
    print(f"  Estimated β from HEP={hep_task}μV, M={m_ca}: {beta_est:.3f}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    quick_test()
