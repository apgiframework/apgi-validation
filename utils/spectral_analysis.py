"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Apply Python 3.14 NumPy compatibility patches first
try:
    from tests.python314_numpy_fixes import *  # noqa: F403,F401
except ImportError:
    pass

import numpy as np
from scipy import signal

# Aperiodic-slope fitting is provided by fooof (<=1.x) or its successor
# specparam (>=2.x), which renamed the model class FOOOF -> SpectralModel.
# Importing only `specparam.FOOOF` silently disabled this analysis on any
# machine with specparam 2.x installed: the guard reported the library as
# missing while a perfectly working one was present.
FOOOF = None
SPECPARAM_AVAILABLE = False

for _module, _attr in (
    ("fooof", "FOOOF"),  # legacy API, matches the attribute names used below
    ("specparam", "FOOOF"),  # specparam 1.x compatibility shim
    ("specparam", "SpectralModel"),  # specparam 2.x
):
    try:
        FOOOF = getattr(__import__(_module, fromlist=[_attr]), _attr)
        SPECPARAM_AVAILABLE = True
        break
    except (ImportError, AttributeError):
        continue

logger = logging.getLogger(__name__)


def compute_spectral_slope_specparam(
    freqs: np.ndarray,
    power_spectrum: np.ndarray,
    freq_range: Tuple[float, float] = (1.0, 40.0),
    peak_width_limits: Optional[Tuple[float, float]] = None,
    max_n_peaks: Optional[int] = None,
    min_peak_height: Optional[float] = None,
    peak_threshold: Optional[float] = None,
    aperiodic_mode: str = "fixed",
) -> Dict[str, Union[float, bool, str]]:
    """
    Compute spectral slope (aperiodic exponent) using FOOOF/specparam.

    This function replaces manual log-log regression with proper aperiodic
    component extraction as required by APGI falsification criteria.

    Args:
        freqs: Array of frequencies in Hz
        power_spectrum: Array of power spectral density values
        freq_range: Frequency range for fitting (default: 1-40 Hz)
        peak_width_limits: Tuple of (min, max) peak width limits
        max_n_peaks: Maximum number of peaks to fit
        min_peak_height: Minimum peak height
        peak_threshold: Peak detection threshold
        aperiodic_mode: Mode for aperiodic fitting ('fixed' or 'knee')

    Returns:
        Dictionary containing:
            - exponent: Aperiodic exponent (β_spec)
            - offset: Aperiodic offset
            - r_squared: Goodness of fit
            - error: Fitting error
            - knee: Knee frequency (if aperiodic_mode='knee')
            - n_peaks: Number of peaks detected
            - fit_success: Whether fitting succeeded
            - error_message: Error message if fitting failed
    """
    if not SPECPARAM_AVAILABLE:
        return {
            "exponent": np.nan,
            "offset": np.nan,
            "r_squared": np.nan,
            "error": np.nan,
            "knee": np.nan,
            "n_peaks": 0,
            "fit_success": False,
            "error_message": "specparam/fooof not available",
        }

    # Validate inputs
    if len(freqs) != len(power_spectrum):
        return {
            "exponent": np.nan,
            "offset": np.nan,
            "r_squared": np.nan,
            "error": np.nan,
            "knee": np.nan,
            "n_peaks": 0,
            "fit_success": False,
            "error_message": "freqs and power_spectrum must have same length",
        }

    if len(freqs) < 10:
        return {
            "exponent": np.nan,
            "offset": np.nan,
            "r_squared": np.nan,
            "error": np.nan,
            "knee": np.nan,
            "n_peaks": 0,
            "fit_success": False,
            "error_message": "Insufficient data points for fitting",
        }

    # Filter to frequency range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_filtered = freqs[freq_mask]
    power_filtered = power_spectrum[freq_mask]

    if len(freqs_filtered) < 10:
        return {
            "exponent": np.nan,
            "offset": np.nan,
            "r_squared": np.nan,
            "error": np.nan,
            "knee": np.nan,
            "n_peaks": 0,
            "fit_success": False,
            "error_message": f"Insufficient data points in frequency range {freq_range}",
        }

    # Check for valid values
    if not np.all(np.isfinite(freqs_filtered)) or not np.all(np.isfinite(power_filtered)):
        return {
            "exponent": np.nan,
            "offset": np.nan,
            "r_squared": np.nan,
            "error": np.nan,
            "knee": np.nan,
            "n_peaks": 0,
            "fit_success": False,
            "error_message": "Non-finite values in input data",
        }

    # Ensure positive power values
    power_filtered = np.maximum(power_filtered, 1e-10)

    try:
        # Initialize FOOOF with specified parameters
        fm = FOOOF if SPECPARAM_AVAILABLE else None
        if fm is None:
            return {
                "exponent": np.nan,
                "offset": np.nan,
                "r_squared": np.nan,
                "error": np.nan,
                "knee": np.nan,
                "n_peaks": 0,
                "fit_success": False,
                "error_message": "specparam package not available",
            }

        fm = fm(
            peak_width_limits=peak_width_limits or [1.0, 8.0],
            max_n_peaks=max_n_peaks or 6,
            min_peak_height=min_peak_height or 0.0,
            peak_threshold=peak_threshold or 2.0,
            aperiodic_mode=aperiodic_mode,
        )

        # Fit the spectrum
        fm.fit(freqs_filtered, power_filtered, freq_range)

        # Extract results
        results = {
            "exponent": (fm.aperiodic_params_[1] if len(fm.aperiodic_params_) >= 2 else np.nan),
            "offset": (fm.aperiodic_params_[0] if len(fm.aperiodic_params_) >= 1 else np.nan),
            "r_squared": fm.r_squared_,
            "error": fm.error_,
            "knee": (
                fm.aperiodic_params_[1] if aperiodic_mode == "knee" and len(fm.aperiodic_params_) >= 2 else np.nan
            ),
            "n_peaks": len(fm.peak_params_) if hasattr(fm, "peak_params_") else 0,
            "fit_success": True,
            "error_message": None,
        }

        # Validate results
        if not np.isfinite(results["exponent"]):
            results["fit_success"] = False
            results["error_message"] = "Non-finite exponent value"

        if results["r_squared"] < 0.5:  # Poor fit
            logger.warning(f"Poor FOOOF fit: R² = {results['r_squared']:.3f}")

        return results

    except Exception as e:
        logger.error(f"FOOOF fitting failed: {str(e)}")
        return {
            "exponent": np.nan,
            "offset": np.nan,
            "r_squared": np.nan,
            "error": np.nan,
            "knee": np.nan,
            "n_peaks": 0,
            "fit_success": False,
            "error_message": str(e),
        }


def validate_specparam_fit(results: Dict[str, Union[float, bool, str]]) -> bool:
    """
    Validate specparam fit results against APGI criteria.

    Args:
        results: Results dictionary from compute_spectral_slope_specparam

    Returns:
        True if fit meets APGI quality criteria, False otherwise
    """
    if not results.get("fit_success", False):
        return False

    # Check R² threshold (APGI requires R² >= 0.90)
    r_squared = results.get("r_squared", 0.0)
    try:
        r_squared_float = float(r_squared)
    except (ValueError, TypeError):
        return False
    if r_squared_float < 0.90:
        return False

    # Check exponent is in reasonable range
    exponent = results.get("exponent", np.nan)
    try:
        exponent_float = float(exponent)
    except (ValueError, TypeError):
        return False
    if not np.isfinite(exponent_float) or exponent_float < 0.1 or exponent_float > 5.0:
        return False

    return True


def compute_power_spectrum(
    input_signal: np.ndarray,
    fs: float,
    method: str = "welch",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum density for spectral analysis.

    Args:
        input_signal: Input signal
        fs: Sampling frequency in Hz
        method: Method for PSD computation ('welch' or 'periodogram')
        **kwargs: Additional arguments passed to scipy.signal method

    Returns:
        Tuple of (freqs, power_spectrum)
    """
    if method == "welch":
        # Default parameters for Welch's method
        defaults = {
            "window": "hann",
            "nperseg": min(256, len(input_signal) // 4),
            "noverlap": None,
            "nfft": None,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
        }
        defaults.update(kwargs)

        freqs, power_spectrum = signal.welch(input_signal, fs=fs, **defaults)

    elif method == "periodogram":
        # Default parameters for periodogram
        defaults = {
            "window": "boxcar",
            "nfft": None,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
        }
        defaults.update(kwargs)

        freqs, power_spectrum = signal.periodogram(input_signal, fs=fs, **defaults)

    else:
        raise ValueError(f"Unknown method: {method}")

    return freqs, power_spectrum


def create_fooof_frequencies(fs: float, n_fft: int = None, freq_range: Tuple[float, float] = (1.0, 40.0)) -> np.ndarray:
    """
    Create evenly spaced frequencies suitable for FOOOF analysis.

    FOOOF requires evenly spaced frequencies in linear space.

    Args:
        fs: Sampling frequency in Hz
        n_fft: Number of FFT points (auto-calculated if None)
        freq_range: Frequency range for analysis

    Returns:
        Array of evenly spaced frequencies
    """
    if n_fft is None:
        # Auto-calculate n_fft to get good frequency resolution
        # Aim for at least 1 Hz resolution in the target range
        target_resolution = 0.5  # Hz
        n_fft = int(fs / target_resolution)
        # Make n_fft a power of 2 for efficiency
        n_fft = 2 ** int(np.ceil(np.log2(n_fft)))

    # Generate frequencies
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)

    # Filter to desired range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    return freqs[freq_mask]


def generate_synthetic_spectra(
    freqs: np.ndarray,
    exponent: float,
    offset: float,
    peak_freqs: Optional[List[float]] = None,
    peak_amplitudes: Optional[List[float]] = None,
    peak_widths: Optional[List[float]] = None,
    noise_level: float = 0.1,
) -> np.ndarray:
    """
    Generate synthetic power spectra with known aperiodic exponent.

    Useful for testing and validation of spectral analysis methods.

    Args:
        freqs: Array of frequencies in Hz
        exponent: Aperiodic exponent (β_spec)
        offset: Aperiodic offset
        peak_freqs: List of peak frequencies in Hz
        peak_amplitudes: List of peak amplitudes
        peak_widths: List of peak widths (standard deviation)
        noise_level: Level of additive noise

    Returns:
        Synthetic power spectrum
    """
    # Aperiodic component: 10^(offset) * f^(-exponent)
    aperiodic = 10**offset * freqs ** (-exponent)

    # Add periodic peaks if specified
    periodic = np.zeros_like(freqs)
    if peak_freqs is not None:
        peak_amplitudes = peak_amplitudes or [1.0] * len(peak_freqs)
        peak_widths = peak_widths or [1.0] * len(peak_freqs)

        for freq, amp, width in zip(peak_freqs, peak_amplitudes, peak_widths):
            # Gaussian peak
            peak = amp * np.exp(-0.5 * ((freqs - freq) / width) ** 2)
            periodic += peak

    # Combine components
    spectrum = aperiodic + periodic

    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.max(spectrum), len(spectrum))
        spectrum = np.maximum(spectrum + noise, 1e-10)

    return spectrum


def batch_compute_spectral_slopes(
    signals: List[np.ndarray],
    fs: float,
    freq_range: Tuple[float, float] = (1.0, 40.0),
    **fooof_kwargs,
) -> List[Dict[str, Union[float, bool, str]]]:
    """
    Compute spectral slopes for multiple signals.

    Args:
        signals: List of input signals
        fs: Sampling frequency in Hz
        freq_range: Frequency range for fitting
        **fooof_kwargs: Additional arguments for FOOOF fitting

    Returns:
        List of results dictionaries
    """
    results = []

    for i, sig in enumerate(signals):
        try:
            # Compute power spectrum
            freqs, power_spectrum = compute_power_spectrum(sig, fs)

            # Compute spectral slope
            result = compute_spectral_slope_specparam(freqs, power_spectrum, freq_range, **fooof_kwargs)

            # Add signal index
            result["signal_index"] = i
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process signal {i}: {str(e)}")
            results.append(
                {
                    "exponent": np.nan,
                    "offset": np.nan,
                    "r_squared": np.nan,
                    "error": np.nan,
                    "knee": np.nan,
                    "n_peaks": 0,
                    "fit_success": False,
                    "error_message": str(e),
                    "signal_index": i,
                }
            )

    return results


def compare_fooof_vs_loglog(
    freqs: np.ndarray,
    power_spectrum: np.ndarray,
    freq_range: Tuple[float, float] = (1.0, 40.0),
    **fooof_kwargs,
) -> Dict[str, Any]:
    """
    Compare FOOOF results with manual log-log regression.

    Useful for validation and demonstrating the difference between methods.

    Args:
        freqs: Array of frequencies in Hz
        power_spectrum: Array of power spectral density values
        freq_range: Frequency range for analysis
        **fooof_kwargs: Additional arguments for FOOOF fitting

    Returns:
        Dictionary with comparison results
    """
    # FOOOF method
    fooof_results = compute_spectral_slope_specparam(freqs, power_spectrum, freq_range, **fooof_kwargs)

    # Manual log-log regression
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_filtered = freqs[freq_mask]
    power_filtered = power_spectrum[freq_mask]

    try:
        # Ensure positive values for log
        freqs_log = np.log10(freqs_filtered)
        power_log = np.log10(np.maximum(power_filtered, 1e-10))

        # Linear regression
        coeffs = np.polyfit(freqs_log, power_log, 1)
        loglog_exponent = -coeffs[0]  # Negative because power ~ f^(-exponent)
        loglog_offset = coeffs[1]

        # Calculate R²
        predicted = np.polyval(coeffs, freqs_log)
        ss_res = np.sum((power_log - predicted) ** 2)
        ss_tot = np.sum((power_log - np.mean(power_log)) ** 2)
        loglog_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        loglog_results = {
            "exponent": loglog_exponent,
            "offset": loglog_offset,
            "r_squared": loglog_r_squared,
            "fit_success": True,
            "error_message": None,
        }

    except Exception as e:
        loglog_results = {
            "exponent": np.nan,
            "offset": np.nan,
            "r_squared": np.nan,
            "fit_success": False,
            "error_message": str(e),
        }

    return {
        "fooof": fooof_results,
        "loglog": loglog_results,
        "difference": {
            "exponent_diff": (
                fooof_results["exponent"] - loglog_results["exponent"]
                if np.isfinite(fooof_results["exponent"]) and np.isfinite(loglog_results["exponent"])
                else np.nan
            ),
            "r_squared_diff": (
                fooof_results["r_squared"] - loglog_results["r_squared"]
                if np.isfinite(fooof_results["r_squared"]) and np.isfinite(loglog_results["r_squared"])
                else np.nan
            ),
        },
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Spectral Analysis Utilities - Example Usage")
    print("=" * 50)

    # Generate test frequencies
    freqs = np.linspace(1, 40, 100)

    # Generate synthetic spectrum with known exponent
    true_exponent = 1.0
    true_offset = 0.0
    spectrum = generate_synthetic_spectra(
        freqs,
        true_exponent,
        true_offset,
        peak_freqs=[10, 20],
        peak_amplitudes=[2.0, 1.0],
        peak_widths=[2.0, 3.0],
    )

    # Compare methods
    comparison = compare_fooof_vs_loglog(freqs, spectrum)

    print(f"True exponent: {true_exponent}")
    fooof_exponent = float(comparison["fooof"]["exponent"])
    loglog_exponent = float(comparison["loglog"]["exponent"])
    fooof_r2 = float(comparison["fooof"]["r_squared"])
    loglog_r2 = float(comparison["loglog"]["r_squared"])
    exponent_diff = float(comparison["difference"]["exponent_diff"])

    print(f"FOOOF exponent: {float(fooof_exponent):.3f}")
    print(f"Log-log exponent: {float(loglog_exponent):.3f}")
    print(f"FOOOF R²: {float(fooof_r2):.3f}")
    print(f"Log-log R²: {float(loglog_r2):.3f}")
    print(f"Exponent difference: {float(exponent_diff):.3f}")
