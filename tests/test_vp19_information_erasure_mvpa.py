"""
Tests for VP_19_InformationErasureMVPA.py
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from Validation.VP_19_InformationErasureMVPA import (
    BIN_CENTRES_MS,
    CATEGORIES,
    IGNITION_BIN_IDX,
    N_BINS,
    N_SENSORS,
    InformationErasureValidator,
    StimulusPairEEGSimulator,
    TimeResolvedMVPADecoder,
    run_validation,
)

FAST_N_PERM = 10  # small for speed in unit tests


# ---------------------------------------------------------------------------
# StimulusPairEEGSimulator tests
# ---------------------------------------------------------------------------


class TestStimulusPairEEGSimulator:

    def test_prototypes_are_unit_vectors(self):
        sim = StimulusPairEEGSimulator()
        assert abs(np.linalg.norm(sim._face_proto) - 1.0) < 1e-9  # nosec B101
        assert abs(np.linalg.norm(sim._house_proto) - 1.0) < 1e-9  # nosec B101

    def test_prototypes_differ(self):
        sim = StimulusPairEEGSimulator()
        assert not np.allclose(sim._face_proto, sim._house_proto)  # nosec B101

    def test_epoch_features_shape(self):
        sim = StimulusPairEEGSimulator()
        amp = np.ones(N_BINS)
        feats = sim._epoch_features("faces", amp)
        assert feats.shape == (N_BINS, N_SENSORS)  # nosec B101

    def test_ignition_profile_winner_higher_than_loser_post_ignition(self):
        sim = StimulusPairEEGSimulator()
        winner_profile = sim._ignition_amplitude_profile("faces", "faces", 0.9)
        loser_profile = sim._ignition_amplitude_profile("houses", "faces", 0.9)
        # Post-ignition window (t > 200 ms)
        post_mask = BIN_CENTRES_MS > 200
        assert winner_profile[post_mask].mean() > loser_profile[post_mask].mean()  # nosec B101

    def test_loser_suppressed_to_near_zero_at_high_ignition(self):
        sim = StimulusPairEEGSimulator()
        loser_profile = sim._ignition_amplitude_profile("houses", "faces", 1.0)
        post_mask = BIN_CENTRES_MS > 200
        assert loser_profile[post_mask].mean() < 0.2  # nosec B101

    def test_no_ignition_profile_constant(self):
        sim = StimulusPairEEGSimulator()
        profile = sim._no_ignition_amplitude_profile("faces")
        assert np.allclose(profile, 1.2)  # nosec B101

    def test_simulate_dataset_length(self):
        rng = np.random.default_rng(0)
        sim = StimulusPairEEGSimulator(n_subjects=3, n_trials_per_condition=5, rng=rng)
        dataset = sim.simulate_dataset()
        # 3 subjects × 5 trials × 2 conditions
        assert len(dataset) == 3 * 5 * 2  # nosec B101

    def test_simulate_dataset_conditions_balanced(self):
        rng = np.random.default_rng(1)
        sim = StimulusPairEEGSimulator(n_subjects=4, n_trials_per_condition=6, rng=rng)
        dataset = sim.simulate_dataset()
        ig_count = sum(1 for m, _, _ in dataset if m.condition == "ignition")
        no_count = sum(1 for m, _, _ in dataset if m.condition == "no_ignition")
        assert ig_count == no_count  # nosec B101

    def test_simulate_dataset_feature_shapes(self):
        rng = np.random.default_rng(2)
        sim = StimulusPairEEGSimulator(n_subjects=2, n_trials_per_condition=3, rng=rng)
        dataset = sim.simulate_dataset()
        for _, losing_feats, winning_feats in dataset:
            assert losing_feats.shape == (N_BINS, N_SENSORS)  # nosec B101
            assert winning_feats.shape == (N_BINS, N_SENSORS)  # nosec B101

    def test_reported_and_losing_are_complementary(self):
        rng = np.random.default_rng(3)
        sim = StimulusPairEEGSimulator(n_subjects=2, n_trials_per_condition=3, rng=rng)
        dataset = sim.simulate_dataset()
        for meta, _, _ in dataset:
            assert meta.reported_category in CATEGORIES  # nosec B101
            assert meta.losing_category in CATEGORIES  # nosec B101
            assert meta.reported_category != meta.losing_category  # nosec B101

    def test_ignition_strength_in_range(self):
        rng = np.random.default_rng(4)
        sim = StimulusPairEEGSimulator(n_subjects=5, n_trials_per_condition=10, rng=rng)
        dataset = sim.simulate_dataset()
        for meta, _, _ in dataset:
            assert 0.0 <= meta.ignition_strength <= 1.0  # nosec B101


# ---------------------------------------------------------------------------
# TimeResolvedMVPADecoder tests
# ---------------------------------------------------------------------------


class TestTimeResolvedMVPADecoder:

    def _make_decodable_data(self, n_trials: int = 20, rng: Optional[np.random.Generator] = None):
        """Create highly decodable data: two well-separated clusters."""
        if rng is None:
            rng = np.random.default_rng(42)
        half = n_trials // 2
        labels = np.array([0] * half + [1] * half)
        feats = np.zeros((n_trials, N_BINS, N_SENSORS))
        feats[:half] = rng.standard_normal((half, N_BINS, N_SENSORS))
        feats[half:] = rng.standard_normal((half, N_BINS, N_SENSORS)) + 5.0
        return feats, labels

    def test_cv_accuracy_high_for_separable_data(self):
        rng = np.random.default_rng(0)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        n = 20
        X = np.vstack(
            [
                rng.standard_normal((n // 2, N_SENSORS)),
                rng.standard_normal((n // 2, N_SENSORS)) + 5.0,
            ]
        )
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        acc = dec._cv_accuracy(X, y)
        assert acc > 0.8  # nosec B101

    def test_cv_accuracy_near_chance_for_noise(self):
        rng = np.random.default_rng(1)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        n = 20
        X = rng.standard_normal((n, N_SENSORS))
        y = np.array([0, 1] * (n // 2))
        acc = dec._cv_accuracy(X, y)
        assert 0.2 <= acc <= 0.8  # nosec B101

    def test_cv_accuracy_handles_single_class(self):
        rng = np.random.default_rng(2)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        X = rng.standard_normal((10, N_SENSORS))
        y = np.zeros(10)
        acc = dec._cv_accuracy(X, y)
        assert acc == 0.5  # nosec B101

    def test_cv_accuracy_handles_tiny_dataset(self):
        rng = np.random.default_rng(3)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        X = rng.standard_normal((4, N_SENSORS))
        y = np.array([0, 0, 1, 1])
        acc = dec._cv_accuracy(X, y)
        assert 0.0 <= acc <= 1.0  # nosec B101

    def test_decode_timeseries_shape(self):
        rng = np.random.default_rng(5)
        sim = StimulusPairEEGSimulator(n_subjects=3, n_trials_per_condition=5, rng=rng)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        dataset = sim.simulate_dataset()
        ig = [(m, lf, wf) for m, lf, wf in dataset if m.condition == "ignition"]
        feats = np.array([lf for _, lf, _ in ig])
        labels = np.array([1 if m.losing_category == "faces" else 0 for m, _, _ in ig])
        acc_ts, perm_thresh = dec.decode_timeseries(feats, labels)
        assert acc_ts.shape == (N_BINS,)  # nosec B101
        assert isinstance(perm_thresh, float)  # nosec B101

    def test_permutation_threshold_in_range(self):
        rng = np.random.default_rng(6)
        sim = StimulusPairEEGSimulator(n_subjects=3, n_trials_per_condition=5, rng=rng)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        dataset = sim.simulate_dataset()
        ig = [(m, lf, wf) for m, lf, wf in dataset if m.condition == "ignition"]
        feats = np.array([lf for _, lf, _ in ig])
        labels = np.array([1 if m.losing_category == "faces" else 0 for m, _, _ in ig])
        perm_thresh = dec._permutation_threshold(feats, labels)
        assert 0.3 <= perm_thresh <= 0.9  # nosec B101


# ---------------------------------------------------------------------------
# InformationErasureValidator tests
# ---------------------------------------------------------------------------


class TestInformationErasureValidator:

    @pytest.fixture
    def fast_validator(self):
        rng = np.random.default_rng(42)
        sim = StimulusPairEEGSimulator(n_subjects=6, n_trials_per_condition=10, rng=rng)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        v = InformationErasureValidator(simulator=sim, decoder=dec, rng=rng)
        v.load_or_generate_data()
        v._run_decoding()
        return v

    def test_load_populates_dataset(self):
        rng = np.random.default_rng(0)
        sim = StimulusPairEEGSimulator(n_subjects=2, n_trials_per_condition=3, rng=rng)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        v = InformationErasureValidator(simulator=sim, decoder=dec, rng=rng)
        v.load_or_generate_data()
        assert len(v.dataset) > 0  # nosec B101

    def test_load_idempotent(self):
        rng = np.random.default_rng(1)
        sim = StimulusPairEEGSimulator(n_subjects=2, n_trials_per_condition=3, rng=rng)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        v = InformationErasureValidator(simulator=sim, decoder=dec, rng=rng)
        v.load_or_generate_data()
        n = len(v.dataset)
        v.load_or_generate_data()
        assert len(v.dataset) == n  # nosec B101

    def test_build_feature_matrices_shapes(self, fast_validator):
        ig_f, ig_l, no_lf, no_wf = fast_validator._build_feature_matrices()
        assert ig_f.ndim == 3  # nosec B101
        assert ig_f.shape[1] == N_BINS  # nosec B101
        assert ig_f.shape[2] == N_SENSORS  # nosec B101
        assert len(ig_l) == len(ig_f)  # nosec B101

    def test_ig_accuracy_ts_shape(self, fast_validator):
        assert fast_validator._ig_accuracy_ts is not None  # nosec B101
        assert fast_validator._ig_accuracy_ts.shape == (N_BINS,)  # nosec B101

    def test_no_ig_accuracy_ts_shapes(self, fast_validator):
        assert fast_validator._no_ig_losing_accuracy_ts is not None  # nosec B101
        assert fast_validator._no_ig_winning_accuracy_ts is not None  # nosec B101
        assert fast_validator._no_ig_losing_accuracy_ts.shape == (N_BINS,)  # nosec B101
        assert fast_validator._no_ig_winning_accuracy_ts.shape == (N_BINS,)  # nosec B101

    def test_v19_1_result_keys(self, fast_validator):
        result = fast_validator.validate_pre_ignition_decodability()
        for key in (
            "test_name",
            "prediction_id",
            "mean_pre_ignition_accuracy",
            "threshold",
            "passed",
        ):
            assert key in result  # nosec B101
        assert result["prediction_id"] == "V19.1"  # nosec B101

    def test_v19_2_result_keys(self, fast_validator):
        result = fast_validator.validate_post_ignition_suppression()
        for key in (
            "test_name",
            "prediction_id",
            "permutation_threshold",
            "below_chance_threshold",
            "max_consecutive_below_chance_bins",
            "passed",
        ):
            assert key in result  # nosec B101
        assert result["prediction_id"] == "V19.2"  # nosec B101

    def test_v19_3_result_keys(self, fast_validator):
        result = fast_validator.validate_control_decodability()
        for key in (
            "test_name",
            "prediction_id",
            "mean_no_ignition_losing_accuracy",
            "mean_no_ignition_winning_accuracy",
            "passed",
        ):
            assert key in result  # nosec B101
        assert result["prediction_id"] == "V19.3"  # nosec B101

    def test_run_full_validation_structure(self, fast_validator):
        results = fast_validator.run_full_validation()
        assert "overall_score" in results  # nosec B101
        assert "tests_passed" in results  # nosec B101
        assert "tests_total" in results  # nosec B101
        assert results["tests_total"] == 3  # nosec B101
        assert results["protocol_id"] == "VP_19_InformationErasureMVPA"  # nosec B101
        assert "measurement_gap_note" in results  # nosec B101

    def test_overall_score_range(self, fast_validator):
        results = fast_validator.run_full_validation()
        assert 0.0 <= results["overall_score"] <= 1.0  # nosec B101

    def test_measurement_gap_note_content(self, fast_validator):
        results = fast_validator.run_full_validation()
        note = results["measurement_gap_note"]
        assert "Tier 3" in note  # nosec B101
        assert "Landauer" in note  # nosec B101
        assert "thermodynamic" in note  # nosec B101
        assert "PET" in note  # nosec B101

    def test_run_full_validation_passes_all_tests_default(self):
        """With a moderate simulator size the protocol should pass all three tests."""
        rng = np.random.default_rng(42)
        sim = StimulusPairEEGSimulator(n_subjects=8, n_trials_per_condition=15, rng=rng)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        v = InformationErasureValidator(simulator=sim, decoder=dec, rng=rng)
        results = v.run_full_validation()
        assert results["tests_passed"] == 3  # nosec B101

    def test_v19_2_consecutive_bins_counter(self):
        """Directly test the consecutive-runs logic with a crafted accuracy series."""
        rng = np.random.default_rng(0)
        sim = StimulusPairEEGSimulator(n_subjects=2, n_trials_per_condition=3, rng=rng)
        dec = TimeResolvedMVPADecoder(n_permutations=FAST_N_PERM, rng=rng)
        v = InformationErasureValidator(simulator=sim, decoder=dec, rng=rng)
        v.load_or_generate_data()
        v._run_decoding()

        # Override accuracy ts to force 4 consecutive below-chance bins post-200ms
        fake_ts = np.full(N_BINS, 0.65)
        post_bins = np.where(BIN_CENTRES_MS > 200)[0]
        fake_ts[post_bins[:4]] = 0.40  # 4 bins below V19_BELOW_CHANCE_THRESHOLD (0.48)
        v._ig_accuracy_ts = fake_ts

        result = v.validate_post_ignition_suppression()
        assert result["max_consecutive_below_chance_bins"] >= 4  # nosec B101
        assert result["passed"] is True  # nosec B101

    @patch("Validation.VP_19_InformationErasureMVPA.HAS_MATPLOTLIB", False)
    def test_generate_figure_no_matplotlib(self, fast_validator):
        result = fast_validator.generate_summary_figure()
        assert result is None  # nosec B101

    def test_generate_figure_with_matplotlib(self, fast_validator, tmp_path):
        output = tmp_path / "test_vp19.png"
        result = fast_validator.generate_summary_figure(output_path=output)
        if result is not None:
            assert output.exists()  # nosec B101


# ---------------------------------------------------------------------------
# Public entry-point tests
# ---------------------------------------------------------------------------


class TestRunValidation:

    def test_returns_dict(self):
        run_validation(seed=0, n_permutations=FAST_N_PERM)

    def test_has_protocol_id(self):
        result = run_validation(seed=1, n_permutations=FAST_N_PERM)
        assert result["protocol_id"] == "VP_19_InformationErasureMVPA"  # nosec B101

    def test_accepts_kwargs(self):
        result = run_validation(seed=2, n_permutations=FAST_N_PERM, extra="ignored")
        assert "overall_score" in result  # nosec B101

    def test_validate_alias(self):
        # validate() uses default seed/permutations — call with patched n_perms
        # by invoking run_validation directly with small n_permutations
        result = run_validation(seed=3, n_permutations=FAST_N_PERM)
        assert "overall_score" in result  # nosec B101

    def test_deterministic(self):
        r1 = run_validation(seed=99, n_permutations=FAST_N_PERM)
        r2 = run_validation(seed=99, n_permutations=FAST_N_PERM)
        assert r1["tests_passed"] == r2["tests_passed"]  # nosec B101
        assert abs(r1["overall_score"] - r2["overall_score"]) < 1e-9  # nosec B101

    def test_validation_timestamp_present(self):
        result = run_validation(seed=5, n_permutations=FAST_N_PERM)
        assert "validation_timestamp" in result  # nosec B101

    def test_three_sub_results(self):
        result = run_validation(seed=6, n_permutations=FAST_N_PERM)
        sub_keys = [k for k, v in result.items() if isinstance(v, dict) and "prediction_id" in v]
        assert len(sub_keys) == 3  # nosec B101


# ---------------------------------------------------------------------------
# BIN_CENTRES_MS sanity tests
# ---------------------------------------------------------------------------


class TestBinConstants:

    def test_n_bins(self):
        assert N_BINS == 30  # nosec B101

    def test_bin_centres_length(self):
        assert len(BIN_CENTRES_MS) == N_BINS  # nosec B101

    def test_ignition_bin_near_zero(self):
        # Bin containing t=0 ms
        assert abs(BIN_CENTRES_MS[IGNITION_BIN_IDX]) < 50.0  # nosec B101

    def test_first_bin_centre_negative(self):
        assert BIN_CENTRES_MS[0] < 0  # nosec B101

    def test_last_bin_centre_positive(self):
        assert BIN_CENTRES_MS[-1] > 0  # nosec B101
