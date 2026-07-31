"""
Tests for VP_18_EEGMicrostateGFPP3b.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from Validation.VP_18_EEGMicrostateGFPP3b import (
    GFP_WINDOW_MS,
    GFPMicrostateValidator,
    GFPResult,
    OddballEEGSimulator,
    run_validation,
    validate,
)

# ---------------------------------------------------------------------------
# OddballEEGSimulator tests
# ---------------------------------------------------------------------------


class TestOddballEEGSimulator:

    def test_default_init(self):
        sim = OddballEEGSimulator(n_subjects=2, n_trials_per_condition=4)
        if sim.fs != 1000.0:
            raise AssertionError(f"Expected fs=1000.0, got {sim.fs}")
        if sim.n_electrodes != 64:
            raise AssertionError(f"Expected n_electrodes=64, got {sim.n_electrodes}")
        if len(sim.timepoints_ms) != sim.n_timepoints:
            raise AssertionError(f"Expected timepoints length {sim.n_timepoints}, got {len(sim.timepoints_ms)}")

    def test_gfp_window_indices_nonempty(self):
        sim = OddballEEGSimulator()
        if sim.window_idx.sum() <= 0:
            raise ValueError(f"Expected positive window_idx sum, got {sim.window_idx.sum()}")

    def test_gfp_window_covers_correct_range(self):
        sim = OddballEEGSimulator()
        t_in_window = sim.timepoints_ms[sim.window_idx]
        if t_in_window.min() < GFP_WINDOW_MS[0]:
            raise AssertionError(f"Window min {t_in_window.min()} < {GFP_WINDOW_MS[0]}")
        if t_in_window.max() > GFP_WINDOW_MS[1]:
            raise AssertionError(f"Window max {t_in_window.max()} > {GFP_WINDOW_MS[1]}")

    def test_simulate_ignition_epoch(self):
        sim = OddballEEGSimulator(n_subjects=1, n_trials_per_condition=1)
        result = sim._simulate_epoch(0, 0, "ignition", 0.9)
        if not isinstance(result, GFPResult):
            raise TypeError(f"Expected GFPResult, got {type(result)}")
        if result.epoch_meta.condition != "ignition":
            raise AssertionError(f"Expected condition ignition, got {result.epoch_meta.condition}")
        if result.gfp_auc <= 0:
            raise ValueError(f"Expected positive GFP AUC, got {result.gfp_auc}")
        if len(result.gfp_timeseries) != sim.n_timepoints:
            raise AssertionError(f"Expected timeseries length {sim.n_timepoints}, got {len(result.gfp_timeseries)}")

    def test_simulate_no_ignition_epoch(self):
        sim = OddballEEGSimulator(n_subjects=1, n_trials_per_condition=1)
        result = sim._simulate_epoch(0, 0, "no_ignition", 0.1)
        if not isinstance(result, GFPResult):
            raise TypeError(f"Expected GFPResult, got {type(result)}")
        if result.epoch_meta.condition != "no_ignition":
            raise AssertionError(f"Expected condition no_ignition, got {result.epoch_meta.condition}")
        if result.gfp_auc <= 0:
            raise ValueError(f"Expected positive GFP AUC, got {result.gfp_auc}")

    def test_ignition_gfp_auc_greater_than_no_ignition_on_average(self):
        """Ignition trials should produce systematically larger GFP-AUC."""
        rng = np.random.default_rng(0)
        sim = OddballEEGSimulator(n_subjects=10, n_trials_per_condition=20, rng=rng)
        dataset = sim.simulate_dataset()
        ig = [r.gfp_auc for r in dataset if r.epoch_meta.condition == "ignition"]
        no = [r.gfp_auc for r in dataset if r.epoch_meta.condition == "no_ignition"]
        if not np.mean(ig) > np.mean(no):
            raise ValueError(f"Expected ignition AUC mean {np.mean(ig)} > no-ignition AUC mean {np.mean(no)}")

    def test_simulate_dataset_length(self):
        sim = OddballEEGSimulator(n_subjects=3, n_trials_per_condition=5)
        dataset = sim.simulate_dataset()
        # 3 subjects × 5 trials × 2 conditions
        if len(dataset) != 3 * 5 * 2:
            raise AssertionError(f"Expected dataset length 30, got {len(dataset)}")

    def test_spatial_cov_shape(self):
        sim = OddballEEGSimulator()
        cov = sim._build_spatial_cov()
        if cov.shape != (64, 64):
            raise AssertionError(f"Expected cov shape (64, 64), got {cov.shape}")
        # Diagonal should be 1 (distance 0)
        if not np.allclose(np.diag(cov), 1.0):
            raise AssertionError(f"Expected diagonal close to 1.0, got {np.diag(cov)}")

    def test_p3b_template_zero_before_stimulus(self):
        sim = OddballEEGSimulator()
        template = sim._p3b_template(amplitude=1.0)
        pre_stim = template[sim.timepoints_ms < 0]
        if not np.all(pre_stim == 0.0):
            raise AssertionError(f"Expected pre-stimulus values to be 0.0, got {pre_stim}")

    def test_p3b_template_peak_near_380ms(self):
        sim = OddballEEGSimulator()
        template = sim._p3b_template(amplitude=1.0)
        peak_t = sim.timepoints_ms[np.argmax(template)]
        if not (360.0 <= peak_t <= 400.0):
            raise AssertionError(f"Expected peak time between 360.0-400.0ms, got {peak_t}")

    def test_gfp_timeseries_nonnegative(self):
        sim = OddballEEGSimulator(n_subjects=1, n_trials_per_condition=1)
        result = sim._simulate_epoch(0, 0, "ignition", 0.8)
        if not np.all(result.gfp_timeseries >= 0):
            raise ValueError("GFP timeseries contains negative values")


# ---------------------------------------------------------------------------
# GFPMicrostateValidator tests
# ---------------------------------------------------------------------------


class TestGFPMicrostateValidator:

    @pytest.fixture
    def small_validator(self):
        rng = np.random.default_rng(42)
        sim = OddballEEGSimulator(n_subjects=10, n_trials_per_condition=30, rng=rng)
        v = GFPMicrostateValidator(simulator=sim)
        v.load_or_generate_data()
        return v

    def test_load_or_generate_data_populates_dataset(self, small_validator):
        if len(small_validator.dataset) <= 0:
            raise ValueError("Dataset should contain data")

    def test_load_or_generate_data_idempotent(self, small_validator):
        original_len = len(small_validator.dataset)
        small_validator.load_or_generate_data()
        if len(small_validator.dataset) != original_len:
            raise AssertionError(f"Expected dataset length {original_len}, got {len(small_validator.dataset)}")

    def test_split_by_condition_balanced(self, small_validator):
        ig, no, st, auc = small_validator._split_by_condition()
        if len(ig) != len(no):
            raise AssertionError(f"Expected equal ignition/no-ignition counts, got ig={len(ig)}, no={len(no)}")
        if len(st) != len(ig) + len(no):
            raise AssertionError(f"Expected st count {len(ig) + len(no)}, got {len(st)}")
        if len(auc) != len(st):
            raise AssertionError(f"Expected auc count {len(st)}, got {len(auc)}")

    def test_validate_gfp_auc_ignition_effect_keys(self, small_validator):
        result = small_validator.validate_gfp_auc_ignition_effect()
        for key in (
            "test_name",
            "prediction_id",
            "t_statistic",
            "p_value",
            "cohens_d",
            "passed",
        ):
            assert key in result  # nosec B101
        if result["prediction_id"] != "V18.1":
            raise AssertionError(f"Expected prediction_id V18.1, got {result['prediction_id']}")

    def test_validate_proportional_advantage_keys(self, small_validator):
        result = small_validator.validate_proportional_advantage()
        for key in (
            "test_name",
            "prediction_id",
            "proportional_advantage",
            "threshold",
            "passed",
        ):
            assert key in result  # nosec B101
        if result["prediction_id"] != "V18.2":
            raise AssertionError(f"Expected prediction_id V18.2, got {result['prediction_id']}")

    def test_validate_st_correlation_keys(self, small_validator):
        result = small_validator.validate_st_correlation()
        for key in (
            "test_name",
            "prediction_id",
            "pearson_r",
            "p_value",
            "threshold_r",
            "passed",
        ):
            assert key in result  # nosec B101
        if result["prediction_id"] != "V18.3":
            raise AssertionError(f"Expected prediction_id V18.3, got {result['prediction_id']}")

    def test_run_full_validation_structure(self, small_validator):
        results = small_validator.run_full_validation()
        required_keys = [
            "overall_score",
            "tests_passed",
            "tests_total",
            "protocol_id",
            "measurement_gap_note",
        ]
        for key in required_keys:
            if key not in results:
                raise KeyError(f"Missing required key: {key}")
        if results["tests_total"] != 3:
            raise AssertionError(f"Expected tests_total=3, got {results['tests_total']}")
        if results["protocol_id"] != "VP_18_EEGMicrostateGFPP3b":
            raise AssertionError(f"Expected protocol_id VP_18_EEGMicrostateGFPP3b, got {results['protocol_id']}")

    def test_run_full_validation_score_range(self, small_validator):
        results = small_validator.run_full_validation()
        if not (0.0 <= results["overall_score"] <= 1.0):
            raise AssertionError(f"Expected overall_score in [0.0, 1.0], got {results['overall_score']}")

    def test_run_full_validation_passes_all_tests(self):
        """With default parameters the simulation should pass all three tests."""
        rng = np.random.default_rng(42)
        sim = OddballEEGSimulator(n_subjects=24, n_trials_per_condition=60, rng=rng)
        v = GFPMicrostateValidator(simulator=sim)
        results = v.run_full_validation()
        if results["tests_passed"] != 3:
            raise AssertionError(f"Expected tests_passed=3, got {results['tests_passed']}")

    def test_measurement_gap_note_content(self, small_validator):
        results = small_validator.run_full_validation()
        note = results["measurement_gap_note"]
        required_terms = ["Tier 3", "Landauer", "thermodynamic"]
        missing_terms = [term for term in required_terms if term not in note]
        if missing_terms:
            raise AssertionError(f"Note missing required terms: {missing_terms}")

    @patch("Validation.VP_18_EEGMicrostateGFPP3b.HAS_MATPLOTLIB", False)
    def test_generate_summary_figure_no_matplotlib(self, small_validator):
        result = small_validator.generate_summary_figure()
        if result is not None:
            raise ValueError("Expected None result when matplotlib unavailable")

    def test_generate_summary_figure_with_matplotlib(self, small_validator, tmp_path):
        output = tmp_path / "test_figure.png"
        result = small_validator.generate_summary_figure(output_path=output)
        if result is not None:
            if not output.exists():
                raise FileNotFoundError(f"Expected output file to exist: {output}")

    def test_proportional_advantage_zero_denominator(self):
        """Edge case: no-ignition AUC is zero."""
        rng = np.random.default_rng(0)
        sim = OddballEEGSimulator(n_subjects=2, n_trials_per_condition=5, rng=rng)
        v = GFPMicrostateValidator(simulator=sim)
        v.load_or_generate_data()
        # Monkey-patch all no-ignition AUCs to 0
        for r in v.dataset:
            if r.epoch_meta.condition == "no_ignition":
                object.__setattr__(r, "gfp_auc", 0.0)
        result = v.validate_proportional_advantage()
        if result["proportional_advantage"] != 0.0:
            raise AssertionError(f"Expected proportional_advantage=0.0, got {result['proportional_advantage']}")

    def test_cohens_d_calculation_positive(self, small_validator):
        result = small_validator.validate_gfp_auc_ignition_effect()
        # Ignition AUC mean should exceed no-ignition AUC mean → d > 0
        if result["cohens_d"] <= 0:
            raise ValueError(f"Expected positive cohens_d, got {result['cohens_d']}")

    def test_st_correlation_sign_positive(self, small_validator):
        result = small_validator.validate_st_correlation()
        # Higher Sₜ should yield higher GFP-AUC → positive correlation
        if result["pearson_r"] <= 0:
            raise ValueError(f"Expected positive pearson_r, got {result['pearson_r']}")


# ---------------------------------------------------------------------------
# Public entry-point tests
# ---------------------------------------------------------------------------


class TestRunValidation:

    def test_run_validation_returns_dict(self):
        run_validation(seed=99)

    def test_run_validation_has_protocol_id(self):
        result = run_validation(seed=0)
        if result["protocol_id"] != "VP_18_EEGMicrostateGFPP3b":
            raise AssertionError(f"Expected protocol_id VP_18_EEGMicrostateGFPP3b, got {result['protocol_id']}")

    def test_run_validation_accepts_kwargs(self):
        result = run_validation(seed=7, extra_param="ignored")
        assert "overall_score" in result  # nosec B101

    def test_validate_alias(self):
        result = validate()
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict, got {type(result)}")
        if "overall_score" not in result:
            raise KeyError("Missing overall_score in result")

    def test_run_validation_deterministic(self):
        r1 = run_validation(seed=42)
        r2 = run_validation(seed=42)
        if r1["overall_score"] != r2["overall_score"]:
            raise AssertionError(f"Expected equal overall_score, got {r1['overall_score']} vs {r2['overall_score']}")
        if r1["tests_passed"] != r2["tests_passed"]:
            raise AssertionError(f"Expected equal tests_passed, got {r1['tests_passed']} vs {r2['tests_passed']}")
