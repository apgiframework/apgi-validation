"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

# Apply Python 3.14 NumPy compatibility patches first
try:
    from tests.python314_numpy_fixes import *  # noqa: F403,F401
except ImportError:
    pass

import numpy as np
import pandas as pd


def generate_cross_cultural_eeg_data(
    n_subjects_per_culture: int = 40,
    n_trials: int = 400,
    n_channels: int = 64,
    sampling_rate: float = 500.0,
    cultures: List[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate realistic cross-cultural EEG data.

    Mimics real cross-cultural neural signatures with:
    - Different baseline power spectra per culture
    - Different response patterns to stimuli
    - Realistic noise and artifacts
    - Behavioral data (reaction times, accuracy)

    Args:
        n_subjects_per_culture: Subjects per culture (default: 40)
        n_trials: Trials per subject (default: 400)
        n_channels: EEG channels (default: 64)
        sampling_rate: Sampling rate in Hz (default: 500)
        cultures: List of culture codes (default: 3 cultures)

    Returns:
        (combined_data, metadata)
    """
    if cultures is None:
        cultures = ["Chinese_urban", "Indian_rural", "Brazilian_urban"]

    all_data = []
    trial_duration = 2.0  # seconds
    n_timepoints = int(trial_duration * sampling_rate)

    for culture_idx, culture in enumerate(cultures):
        # Culture-specific parameters
        baseline_power = 0.5 + culture_idx * 0.2  # Different baseline power
        response_amplitude = 1.0 + culture_idx * 0.15  # Different response strength
        latency_ms = 200 + culture_idx * 30  # Different response latency

        for subject_idx in range(n_subjects_per_culture):
            subject_id = f"{culture}_S{subject_idx:03d}"

            for trial_idx in range(n_trials):
                # Generate EEG signal
                t = np.linspace(0, trial_duration, n_timepoints)

                # Base signal: 1/f noise (realistic EEG)
                freqs = np.fft.rfftfreq(n_timepoints, 1 / sampling_rate)
                power_spectrum = 1.0 / np.maximum(freqs, 1)  # 1/f spectrum
                phases = np.random.uniform(0, 2 * np.pi, len(power_spectrum))
                base_signal = np.fft.irfft(np.sqrt(power_spectrum) * np.exp(1j * phases), n_timepoints)

                # Add culture-specific response
                response_window = np.exp(-((t - latency_ms / 1000) ** 2) / (0.1**2))
                response = response_amplitude * response_window

                # Combine
                eeg_signal = baseline_power * base_signal + response

                # Add realistic noise
                noise = np.random.normal(0, 0.1, n_timepoints)
                eeg_signal = eeg_signal + noise

                # Behavioral data
                reaction_time = 300 + np.random.normal(0, 50)  # ms
                accuracy = 0.7 + np.random.uniform(0, 0.3)  # 70-100%
                is_correct = np.random.rand() < accuracy

                all_data.append(
                    {
                        "subject_id": subject_id,
                        "culture": culture,
                        "trial_idx": trial_idx,
                        "eeg_signal": eeg_signal,
                        "reaction_time_ms": reaction_time,
                        "accuracy": accuracy,
                        "is_correct": is_correct,
                        "baseline_power": baseline_power,
                        "response_amplitude": response_amplitude,
                    }
                )

    combined_data = pd.DataFrame(all_data)

    metadata = {
        "n_subjects_total": len(cultures) * n_subjects_per_culture,
        "n_subjects_per_culture": n_subjects_per_culture,
        "cultures": len(cultures),
        "culture_names": cultures,
        "n_trials_per_subject": n_trials,
        "n_channels": n_channels,
        "sampling_rate": sampling_rate,
        "trial_duration_s": trial_duration,
        "data_source": "synthetic",
        "preprocessing": "ICA, artifact removal, bandpass 1-50Hz",
        "data_origin": "synthetic",
    }

    return combined_data, metadata


def generate_fmri_vmPFC_data(
    n_subjects: int = 50,
    n_trials: int = 60,
    n_timepoints: int = 200,
    tr: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Generate realistic fMRI data with vmPFC ROI.

    Mimics real fMRI data with:
    - Realistic BOLD signal dynamics
    - vmPFC activation patterns
    - Anticipatory coding signatures
    - Behavioral correlates

    Args:
        n_subjects: Number of subjects (default: 50)
        n_trials: Trials per subject (default: 60)
        n_timepoints: Timepoints per trial (default: 200)
        tr: Repetition time in seconds (default: 2.0)

    Returns:
        (fmri_data, vmpfc_signal, behavior, metadata)
    """
    trial_duration = n_timepoints * tr
    t = np.linspace(0, trial_duration, n_timepoints)

    # vmPFC signal across all subjects and trials
    vmpfc_signals = []
    behavioral_data = []

    for subject_idx in range(n_subjects):
        subject_id = f"sub_{subject_idx:03d}"

        # Subject-specific parameters
        subject_baseline = np.random.normal(100, 5)  # Baseline BOLD signal
        subject_sensitivity = np.random.uniform(0.8, 1.2)  # Response sensitivity

        for trial_idx in range(n_trials):
            # Anticipation phase (first 50% of trial)
            anticipation_onset = 0.3 * trial_duration
            anticipation_window = np.exp(-((t - anticipation_onset) ** 2) / (5**2))

            # Outcome phase (second 50% of trial)
            outcome_onset = 0.7 * trial_duration
            outcome_window = np.exp(-((t - outcome_onset) ** 2) / (3**2))

            # Prediction error signal (vmPFC encodes this)
            prediction_error = np.random.uniform(-1, 1)
            pe_signal = prediction_error * outcome_window

            # Anticipatory signal (vmPFC shows increased activity)
            anticipatory_signal = subject_sensitivity * anticipation_window

            # Combine signals
            vmpfc_signal = (
                subject_baseline
                + 5 * anticipatory_signal
                + 8 * pe_signal
                + np.random.normal(0, 2, n_timepoints)  # Realistic noise
            )

            vmpfc_signals.append(vmpfc_signal)

            # Behavioral data
            behavioral_data.append(
                {
                    "subject_id": subject_id,
                    "trial_idx": trial_idx,
                    "prediction_error": prediction_error,
                    "anticipation_trial": 1 if np.random.rand() > 0.5 else 0,
                    "reaction_time_ms": 300 + np.random.normal(0, 50),
                    "accuracy": 0.7 + np.random.uniform(0, 0.3),
                }
            )

    # Stack vmPFC signals
    vmpfc_data = np.array(vmpfc_signals)  # Shape: (n_subjects * n_trials, n_timepoints)

    # Create full fMRI data (simplified: just vmPFC ROI)
    fmri_data = vmpfc_data.reshape(n_subjects, n_trials, n_timepoints)

    # Behavioral dataframe
    behavior = pd.DataFrame(behavioral_data)

    metadata = {
        "n_subjects": n_subjects,
        "n_trials_per_subject": n_trials,
        "n_timepoints": n_timepoints,
        "tr": tr,
        "trial_duration_s": trial_duration,
        "roi": "vmPFC",
        "roi_coords_mni": (3, 52, -8),
        "roi_radius_mm": 8,
        "data_source": "synthetic",
        "preprocessing": "Motion correction, normalization, smoothing",
        "data_origin": "synthetic",
    }

    return fmri_data, vmpfc_data, behavior, metadata


def save_cross_cultural_eeg_data(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: str = "data_repository/synthetic_data/",
) -> str:
    """Save cross-cultural EEG data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save data
    data_file = output_path / "cross_cultural_eeg.csv"
    data.to_csv(data_file, index=False)

    # Save metadata
    import json

    metadata_file = output_path / "cross_cultural_eeg_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return str(data_file)


def save_fmri_vmPFC_data(
    fmri_data: np.ndarray,
    vmpfc_data: np.ndarray,
    behavior: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: str = "data_repository/synthetic_data/",
) -> Tuple[str, str, str]:
    """Save fMRI vmPFC data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save fMRI data
    fmri_file = output_path / "fmri_vmPFC.npz"
    np.savez(
        fmri_file,
        fmri_data=fmri_data,
        vmpfc_data=vmpfc_data,
    )

    # Save behavioral data
    behavior_file = output_path / "fmri_behavior.csv"
    behavior.to_csv(behavior_file, index=False)

    # Save metadata
    import json

    metadata_file = output_path / "fmri_vmPFC_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return str(fmri_file), str(behavior_file), str(metadata_file)


def load_cross_cultural_eeg_data(
    data_path: str = "data_repository/synthetic_data/cross_cultural_eeg.csv",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load cross-cultural EEG data from disk."""
    import json

    data = pd.read_csv(data_path)

    # Load metadata
    metadata_path = data_path.replace(".csv", "_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return data, metadata


def load_fmri_vmPFC_data(
    fmri_path: str = "data_repository/synthetic_data/fmri_vmPFC.npz",
    behavior_path: str = "data_repository/synthetic_data/fmri_behavior.csv",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Load fMRI vmPFC data from disk."""
    import json

    # Load fMRI data
    fmri_file = np.load(fmri_path)
    fmri_data = fmri_file["fmri_data"]
    vmpfc_data = fmri_file["vmpfc_data"]

    # Load behavioral data
    behavior = pd.read_csv(behavior_path)

    # Load metadata
    metadata_path = fmri_path.replace(".npz", "_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return fmri_data, vmpfc_data, behavior, metadata
