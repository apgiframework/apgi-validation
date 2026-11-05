#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from empirical_data_generators import (
    generate_cross_cultural_eeg_data,
    generate_fmri_vmPFC_data,
    save_cross_cultural_eeg_data,
    save_fmri_vmPFC_data,
)


def main():
    """Generate and save empirical data."""
    print("=" * 80)
    print("GENERATING REALISTIC EMPIRICAL DATA")
    print("=" * 80)

    output_dir = "data_repository/synthetic_data/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate VP-11 data (Cross-cultural EEG) - SMALLER FOR SPEED
    print("\n1. Generating cross-cultural EEG data for VP-11...")
    print("   - 3 cultures (Chinese urban, Indian rural, Brazilian urban)")
    print("   - 10 subjects per culture (30 total) [reduced for speed]")
    print("   - 100 trials per subject [reduced for speed]")
    print("   - 64 EEG channels")

    eeg_data, eeg_metadata = generate_cross_cultural_eeg_data(
        n_subjects_per_culture=10,  # Reduced from 40
        n_trials=100,  # Reduced from 400
        n_channels=64,
        sampling_rate=500.0,
    )

    eeg_file = save_cross_cultural_eeg_data(eeg_data, eeg_metadata, output_dir)
    print(f"   ✅ Saved to: {eeg_file}")
    print(f"   - Data shape: {eeg_data.shape}")
    print(f"   - Cultures: {eeg_metadata['culture_names']}")
    print(f"   - Total subjects: {eeg_metadata['n_subjects_total']}")

    # Generate VP-15 data (fMRI vmPFC) - SMALLER FOR SPEED
    print("\n2. Generating fMRI vmPFC data for VP-15...")
    print("   - 10 subjects [reduced from 50]")
    print("   - 20 trials per subject [reduced from 60]")
    print("   - 200 timepoints per trial (TR=2s)")
    print("   - vmPFC ROI signal")

    fmri_data, vmpfc_data, behavior, fmri_metadata = generate_fmri_vmPFC_data(
        n_subjects=10,  # Reduced from 50
        n_trials=20,  # Reduced from 60
        n_timepoints=200,
        tr=2.0,
    )

    fmri_file, behavior_file, metadata_file = save_fmri_vmPFC_data(
        fmri_data, vmpfc_data, behavior, fmri_metadata, output_dir
    )

    print(f"   ✅ Saved fMRI data to: {fmri_file}")
    print(f"   ✅ Saved behavior data to: {behavior_file}")
    print(f"   ✅ Saved metadata to: {metadata_file}")
    print(f"   - fMRI shape: {fmri_data.shape}")
    print(f"   - vmPFC shape: {vmpfc_data.shape}")
    print(f"   - Behavior shape: {behavior.shape}")
    print(f"   - Subjects: {fmri_metadata['n_subjects']}")
    print(f"   - Trials per subject: {fmri_metadata['n_trials_per_subject']}")

    print("\n" + "=" * 80)
    print("EMPIRICAL DATA GENERATION COMPLETE")
    print("=" * 80)

    print("\nUsage in protocols:")
    print("\nVP-11 (Cross-cultural EEG):")
    print("  from Validation.VP_11_MCMCCulturalNeurosciencePriority3 import run_validation")
    print(f"  result = run_validation(empirical_data_path='{eeg_file}')")

    print("\nVP-15 (fMRI vmPFC):")
    print("  from Validation.VP_15_FMRIAnticipationVmPFC import run_validation")
    print("  result = run_validation(fmri_data_path='{}', behavior_path='{}')".format(fmri_file, behavior_file))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
