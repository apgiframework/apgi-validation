"""Regression tests: comprehensive validation must reflect real check results.

Historical defect: run_comprehensive_validation() called each validate_*()
function but discarded every return value, initialising its results dict to {}.
`statuses` was therefore always empty, so `errors == 0 and failed == 0` held
vacuously and the function reported overall_status="PASSED" no matter what the
individual checks found — a validation framework that could not fail.
"""

from unittest.mock import patch

import pytest

from utils import validation_runner
from utils.validation_runner import run_comprehensive_validation, validate_parameter_consistency

CHECK_NAMES = [
    "fp02_data_variance",
    "fp03_dependencies",
    "fp04_te_computation",
    "fp05_empirical_data",
    "parameter_consistency",
]


class TestResultsAreAggregated:
    def test_every_check_appears_in_results(self):
        results = run_comprehensive_validation()
        for name in CHECK_NAMES:
            assert name in results, f"{name} result was discarded"

    def test_summary_counts_all_checks(self):
        summary = run_comprehensive_validation()["summary"]
        assert summary["total_checks"] == len(CHECK_NAMES)
        assert summary["passed"] + summary["failed"] + summary["errors"] == len(CHECK_NAMES)

    def test_protocol_results_mirrors_individual_results(self):
        results = run_comprehensive_validation()
        for name in CHECK_NAMES:
            assert results["protocol_results"][name] == results[name]

    def test_each_result_carries_a_status(self):
        results = run_comprehensive_validation()
        for name in CHECK_NAMES:
            assert results[name]["status"] in {"PASSED", "FAILED", "ERROR"}


class TestOverallStatusReflectsFailures:
    """The aggregate must be able to report failure — the core of the defect."""

    def test_a_failing_check_makes_overall_failed(self):
        with patch.object(
            validation_runner,
            "validate_fp02_data_variance",
            return_value={"status": "FAILED", "reason": "injected"},
        ):
            results = run_comprehensive_validation()

        assert results["overall_status"] == "FAILED"
        assert results["summary"]["failed"] >= 1

    def test_an_erroring_check_makes_overall_error(self):
        with patch.object(
            validation_runner,
            "validate_fp03_dependencies",
            return_value={"status": "ERROR", "error": "injected"},
        ):
            results = run_comprehensive_validation()

        assert results["overall_status"] == "ERROR"
        assert results["summary"]["errors"] >= 1

    def test_error_dominates_failure(self):
        with (
            patch.object(
                validation_runner,
                "validate_fp02_data_variance",
                return_value={"status": "FAILED", "reason": "injected"},
            ),
            patch.object(
                validation_runner,
                "validate_fp03_dependencies",
                return_value={"status": "ERROR", "error": "injected"},
            ),
        ):
            results = run_comprehensive_validation()

        assert results["overall_status"] == "ERROR"

    def test_all_passing_gives_overall_passed(self):
        passing = {"status": "PASSED"}
        with (
            patch.object(validation_runner, "validate_fp02_data_variance", return_value=passing),
            patch.object(validation_runner, "validate_fp03_dependencies", return_value=passing),
            patch.object(validation_runner, "validate_fp04_te_computation", return_value=passing),
            patch.object(validation_runner, "validate_fp05_empirical_data", return_value=passing),
            patch.object(validation_runner, "validate_parameter_consistency", return_value=passing),
        ):
            results = run_comprehensive_validation()

        assert results["overall_status"] == "PASSED"
        assert results["summary"]["passed"] == len(CHECK_NAMES)


class TestParameterConsistencyThresholds:
    """F1.1 ships two regimes; the consistency check must accept either."""

    def test_consistency_check_passes(self):
        result = validate_parameter_consistency()
        assert result["status"] == "PASSED", f"Unexpected failures: {result.get('failed_checks')}"

    def test_no_failed_checks_reported(self):
        assert validate_parameter_consistency()["failed_checks"] == []

    def test_active_threshold_is_a_declared_regime(self):
        from utils.falsification_thresholds import (
            F1_1_MIN_ADVANTAGE_PCT,
            F1_1_MIN_ADVANTAGE_PCT_PAPER_SPEC,
            F1_1_MIN_ADVANTAGE_PCT_SIMULATION,
        )

        assert F1_1_MIN_ADVANTAGE_PCT in (
            F1_1_MIN_ADVANTAGE_PCT_PAPER_SPEC,
            F1_1_MIN_ADVANTAGE_PCT_SIMULATION,
        )

    def test_simulation_threshold_not_stricter_than_paper_spec(self):
        from utils.falsification_thresholds import (
            F1_1_MIN_ADVANTAGE_PCT_PAPER_SPEC,
            F1_1_MIN_ADVANTAGE_PCT_SIMULATION,
        )

        assert F1_1_MIN_ADVANTAGE_PCT_SIMULATION <= F1_1_MIN_ADVANTAGE_PCT_PAPER_SPEC

    def test_consistency_checks_are_json_serialisable(self):
        import json

        json.dumps(validate_parameter_consistency())
