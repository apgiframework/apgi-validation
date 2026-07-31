"""
Comprehensive tests for utils/validation_runner.py functions - 100% coverage target.

This file tests:
- Individual validation functions for each protocol
- Error handling and fallback mechanisms
- Data validation and quality checks
- Comprehensive suite execution
- Integration with falsification protocols
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.validation_runner import (
        run_comprehensive_validation,
        validate_fp02_data_variance,
        validate_fp03_dependencies,
        validate_fp04_te_computation,
        validate_fp05_empirical_data,
        validate_parameter_consistency,
    )

    VALIDATION_RUNNER_AVAILABLE = True
except ImportError as e:
    VALIDATION_RUNNER_AVAILABLE = False
    print(f"Warning: validation_runner not available for testing: {e}")


class TestValidationRunnerFunctions:
    """Comprehensive tests for validation runner functions."""

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_validate_fp02_data_variance(self):
        """Test FP-02 data variance validation."""
        result = validate_fp02_data_variance()

        assert "status" in result, "Result should contain status field"
        assert result["status"] in ["PASSED", "FAILED", "ERROR"], "Status should be one of the expected values"

        if result["status"] == "PASSED":
            assert "variance_metrics" in result, "PASSED result should contain variance_metrics"
            assert "sample_size_adequacy" in result, "PASSED result should contain sample_size_adequacy"
        elif result["status"] == "FAILED":
            assert "errors" in result, "FAILED result should contain errors"
        elif result["status"] == "ERROR":
            assert "error_message" in result, "ERROR result should contain error_message"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_validate_fp03_dependencies(self):
        """Test FP-03 dependency validation."""
        result = validate_fp03_dependencies()

        assert "status" in result, "Result should contain status field"
        assert result["status"] in ["PASSED", "FAILED", "ERROR"], "Status should be one of the expected values"

        if result["status"] == "PASSED":
            assert (
                "shared_falsification_available" in result
            ), "PASSED result should contain shared_falsification_available"
            assert "aggregator_available" in result, "PASSED result should contain aggregator_available"
        elif result["status"] == "FAILED":
            assert "errors" in result, "FAILED result should contain errors"
        elif result["status"] == "ERROR":
            assert "error_message" in result, "ERROR result should contain error_message"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_validate_fp04_te_computation(self):
        """Test FP-04 TE computation validation."""
        result = validate_fp04_te_computation()

        assert "status" in result, "Result should contain status field"
        assert result["status"] in ["PASSED", "FAILED", "ERROR"], "Status should be one of the expected values"

        if result["status"] == "PASSED":
            assert "te_mean" in result, "PASSED result should contain te_mean"
            assert "te_values_valid" in result, "PASSED result should contain te_values_valid"
        elif result["status"] == "FAILED":
            assert "errors" in result, "FAILED result should contain errors"
        elif result["status"] == "ERROR":
            assert "error_message" in result, "ERROR result should contain error_message"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_validate_fp05_empirical_data(self):
        """Test FP-05 empirical data validation."""
        result = validate_fp05_empirical_data()

        assert "status" in result, "Result should contain status field"
        assert result["status"] in ["PASSED", "FAILED", "ERROR"], "Status should be one of the expected values"

        if result["status"] == "PASSED":
            assert "compliance_rate" in result, "PASSED result should contain compliance_rate"
            assert "theta_gamma_valid_ratio" in result, "PASSED result should contain theta_gamma_valid_ratio"
        elif result["status"] == "FAILED":
            assert "errors" in result, "FAILED result should contain errors"
        elif result["status"] == "ERROR":
            assert "error_message" in result, "ERROR result should contain error_message"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_validate_parameter_consistency(self):
        """Test parameter consistency validation."""
        result = validate_parameter_consistency()

        assert "status" in result, "Result should contain status field"
        assert result["status"] in ["PASSED", "FAILED", "ERROR"], "Status should be one of the expected values"

        if result["status"] == "PASSED":
            assert "consistency_checks" in result, "PASSED result should contain consistency_checks"
            assert "all_consistent" in result, "PASSED result should contain all_consistent"
        elif result["status"] == "FAILED":
            assert "failed_checks" in result, "FAILED result should name the checks that failed"
            assert result["failed_checks"], "FAILED result should list at least one failed check"
        elif result["status"] == "ERROR":
            assert "error" in result, "ERROR result should contain the error message"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_run_comprehensive_validation(self):
        """Test comprehensive validation suite."""
        result = run_comprehensive_validation()

        assert "summary" in result, "Result should contain summary"
        assert "fp02_data_variance" in result, "Result should contain fp02_data_variance"
        assert "fp03_dependencies" in result, "Result should contain fp03_dependencies"
        assert "fp04_te_computation" in result, "Result should contain fp04_te_computation"
        assert "fp05_empirical_data" in result, "Result should contain fp05_empirical_data"
        assert "parameter_consistency" in result, "Result should contain parameter_consistency"

        # Check summary statistics
        summary = result["summary"]
        assert "total_checks" in summary, "Summary should contain total_checks"
        assert "passed" in summary, "Summary should contain passed"
        assert "failed" in summary, "Summary should contain failed"
        assert "errors" in summary, "Summary should contain errors"
        assert summary["total_checks"] == 5, "Should have 5 total checks"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_error_handling_temp_directory_issue(self):
        """Test error handling for temporary directory issues."""
        # Mock temp directory error
        with patch("tempfile.mkdtemp", side_effect=OSError("No usable temporary directory")):
            result = validate_fp02_data_variance()

            assert result["status"] == "ERROR", "Should return ERROR status when temp directory fails"
            assert "temp directory" in result["error_message"].lower(), "Error message should mention temp directory"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_import_error_handling(self):
        """Test handling of missing dependencies."""
        # Mock import error for falsification modules
        with patch.dict(
            "sys.modules",
            {"Falsification.FP_02_AgentComparisonConvergenceBenchmark": None},
        ):
            result = validate_fp02_data_variance()

            assert result["status"] in ["ERROR", "FAILED"], "Should return ERROR or FAILED status when module missing"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_result_serialization(self):
        """Test that results can be serialized to JSON."""
        result = run_comprehensive_validation()

        # Should be JSON serializable
        try:
            json_str = json.dumps(result)
            assert isinstance(json_str, str), "JSON serialization should produce string"

            # Should be able to load back
            loaded_result = json.loads(json_str)
            assert loaded_result == result, "Deserialized result should match original"
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result not JSON serializable: {e}")

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_validation_performance(self):
        """Test validation function performance."""
        import time

        # Time comprehensive suite execution
        start_time = time.time()
        run_comprehensive_validation()
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 300, "Should complete within 5 minutes (300 seconds)"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_individual_protocol_isolation(self):
        """Test that individual protocols can be run in isolation."""
        # Run each protocol individually
        protocols = [
            ("FP-02", validate_fp02_data_variance),
            ("FP-03", validate_fp03_dependencies),
            ("FP-04", validate_fp04_te_computation),
            ("FP-05", validate_fp05_empirical_data),
            ("Parameter", validate_parameter_consistency),
        ]

        results = {}
        for protocol_name, validator_func in protocols:
            try:
                result = validator_func()
                results[protocol_name] = result

                # Verify basic structure
                assert "status" in result, f"Protocol {protocol_name} result should contain status"

            except Exception as e:
                pytest.fail(f"Protocol {protocol_name} failed: {e}")

        # Should have results for all protocols
        assert len(results) == 5, "Should have results for all 5 protocols"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_validation_consistency(self):
        """Test that validation results are consistent across runs."""
        # Run comprehensive suite twice
        result1 = run_comprehensive_validation()
        result2 = run_comprehensive_validation()

        # Overall status should be consistent
        assert result1["overall_status"] == result2["overall_status"], "Overall status should be consistent across runs"

        # Protocol counts should be consistent
        assert (
            result1["summary"]["total_protocols"] == result2["summary"]["total_protocols"]
        ), "Protocol counts should be consistent across runs"

        # Individual protocol statuses should be consistent (unless they have randomness)
        for protocol in result1["protocol_results"]:
            if protocol in result2["protocol_results"]:
                status1 = result1["protocol_results"][protocol].get("status")
                status2 = result2["protocol_results"][protocol].get("status")

                # Allow for ERROR status differences due to environmental factors
                if status1 != "ERROR" and status2 != "ERROR":
                    assert status1 == status2, f"Protocol {protocol} status should be consistent across runs"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_validation_output_structure(self):
        """Test that validation outputs have expected structure."""
        result = run_comprehensive_validation()

        # Check top-level structure
        required_keys = ["overall_status", "protocol_results", "summary"]
        for key in required_keys:
            assert key in result, f"Result should contain {key}"

        # Check protocol results structure
        for protocol, protocol_result in result["protocol_results"].items():
            assert "status" in protocol_result, f"Protocol {protocol} result should contain status"

            # If passed, should have metrics
            if protocol_result["status"] == "PASSED":
                # Should have some kind of metrics (specific keys vary by protocol)
                assert len(protocol_result) > 1, f"Protocol {protocol} should have more than just status"

        # Check summary structure
        summary = result["summary"]
        required_summary_keys = [
            "total_checks",
            "passed",
            "failed",
            "errors",
        ]
        for key in required_summary_keys:
            assert key in summary, f"Summary should contain {key}"
            assert isinstance(summary[key], int), f"Summary {key} should be an integer"

        # Summary counts should add up
        total = summary["passed"] + summary["failed"] + summary["errors"]
        assert total == summary["total_checks"], "Summary counts should add up to total_checks"
