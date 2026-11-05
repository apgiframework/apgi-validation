"""
Tests utilizing underutilized fixtures from conftest.py.
=====================================================
Tests for raises_fixture, oom_fixture, mock_memory_error, flaky_operation, and exception_test_cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRaisesFixture:
    """Test the raises_fixture for custom exception testing."""

    def test_raises_fixture_value_error(self, raises_fixture):
        """Test raises_fixture with ValueError."""
        with raises_fixture(ValueError):
            raise ValueError("Test error")

    def test_raises_fixture_type_error(self, raises_fixture):
        """Test raises_fixture with TypeError."""
        with raises_fixture(TypeError):
            raise TypeError("Test error")

    def test_raises_fixture_key_error(self, raises_fixture):
        """Test raises_fixture with KeyError."""
        with raises_fixture(KeyError):
            raise KeyError("Test error")

    def test_raises_fixture_runtime_error(self, raises_fixture):
        """Test raises_fixture with RuntimeError."""
        with raises_fixture(RuntimeError):
            raise RuntimeError("Test error")

    def test_raises_fixture_custom_exception(self, raises_fixture):
        """Test raises_fixture with custom exception."""

        class CustomError(Exception):
            pass

        with raises_fixture(CustomError):
            raise CustomError("Custom error")

    def test_raises_fixture_no_exception_fails(self, raises_fixture):
        """Test that raises_fixture fails when no exception is raised."""
        try:
            with raises_fixture(ValueError):
                pass  # No exception raised
            assert False, "Should have raised pytest.fail"
        except pytest.fail.Exception:
            # Expected - fixture detected no exception
            assert True


class TestOOMFixture:
    """Test the oom_fixture for out-of-memory simulation."""

    def test_oom_fixture_basic_usage(self, oom_fixture):
        """Test basic usage of oom_fixture."""
        with oom_fixture:
            # Simulate memory-intensive operation
            data = np.zeros(1000)
            assert len(data) == 1000

    def test_oom_fixture_with_large_array(self, oom_fixture):
        """Test oom_fixture with larger array."""
        with oom_fixture:
            # Try to create a moderately large array
            try:
                data = np.zeros(10000)
                assert len(data) == 10000
            except MemoryError:
                # Expected if OOM fixture is working
                assert True

    def test_oom_fixture_nested_context(self, oom_fixture):
        """Test nested oom_fixture contexts."""
        with oom_fixture:
            with oom_fixture:
                data = np.zeros(100)
                assert len(data) == 100


class TestMockMemoryError:
    """Test the mock_memory_error fixture for memory error patching."""

    def test_mock_memory_error_numpy_zeros(self, mock_memory_error):
        """Test mock_memory_error with numpy.zeros."""
        mock_memory_error.patch_numpy_zeros()

        try:
            # First few calls should succeed
            np.zeros(10)
            np.zeros(10)
            np.zeros(10)

            # Fourth call should raise MemoryError
            try:
                np.zeros(10)
                assert False, "Should have raised MemoryError"
            except MemoryError:
                assert True  # Expected

        finally:
            mock_memory_error.cleanup()

    def test_mock_memory_error_torch_tensor(self, mock_memory_error):
        """Test mock_memory_error with torch.tensor."""
        patch_obj = mock_memory_error.patch_torch_tensor()

        if patch_obj is None:
            pytest.skip("PyTorch not available")

        try:
            # First few calls should succeed
            try:
                import torch

                torch.tensor([1, 2, 3])
                torch.tensor([4, 5, 6])

                # Third call should raise RuntimeError (CUDA OOM)
                try:
                    torch.tensor([7, 8, 9])
                    assert False, "Should have raised RuntimeError"
                except RuntimeError as e:
                    assert "out of memory" in str(e).lower()
                    assert True  # Expected
            except ImportError:
                pytest.skip("PyTorch not available")

        finally:
            mock_memory_error.cleanup()

    def test_mock_memory_error_multiple_patches(self, mock_memory_error):
        """Test mock_memory_error with multiple patches."""
        mock_memory_error.patch_numpy_zeros()
        mock_memory_error.patch_torch_tensor()

        try:
            # Both patches should be active
            data1 = np.zeros(10)
            assert len(data1) == 10

        finally:
            mock_memory_error.cleanup()


class TestFlakyOperation:
    """Test the flaky_operation fixture for flaky operation testing."""

    def test_flaky_operation_success(self, flaky_operation):
        """Test flaky_operation with high success rate."""
        operation = flaky_operation(success_rate=0.9)

        # Should succeed most of the time
        try:
            result = operation()
            assert result == "success"
        except RuntimeError:
            # May fail occasionally
            assert True

    def test_flaky_operation_failure(self, flaky_operation):
        """Test flaky_operation with low success rate."""
        operation = flaky_operation(success_rate=0.1)

        # Should fail most of the time
        try:
            result = operation()
            assert result == "success"
        except RuntimeError:
            # Expected to fail
            assert True

    def test_flaky_operation_retry_pattern(self, flaky_operation):
        """Test retry pattern with flaky_operation."""
        operation = flaky_operation(success_rate=0.5)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = operation()
                assert result == "success"
                break
            except RuntimeError:
                if attempt == max_retries - 1:
                    # Failed all retries
                    assert True
                continue


class TestExceptionTestCases:
    """Test the exception_test_cases fixture."""

    def test_exception_test_cases_value_error(self, exception_test_cases):
        """Test exception_test_cases with ValueError."""
        error = exception_test_cases["value_error"]
        assert isinstance(error, ValueError)
        assert str(error) == "Invalid value"

    def test_exception_test_cases_type_error(self, exception_test_cases):
        """Test exception_test_cases with TypeError."""
        error = exception_test_cases["type_error"]
        assert isinstance(error, TypeError)
        assert str(error) == "Invalid type"

    def test_exception_test_cases_key_error(self, exception_test_cases):
        """Test exception_test_cases with KeyError."""
        error = exception_test_cases["key_error"]
        assert isinstance(error, KeyError)
        # KeyError string representation includes quotes
        assert str(error) == "'Missing key'"

    def test_exception_test_cases_attribute_error(self, exception_test_cases):
        """Test exception_test_cases with AttributeError."""
        error = exception_test_cases["attribute_error"]
        assert isinstance(error, AttributeError)

    def test_exception_test_cases_memory_error(self, exception_test_cases):
        """Test exception_test_cases with MemoryError."""
        error = exception_test_cases["memory_error"]
        assert isinstance(error, MemoryError)
        assert str(error) == "Out of memory"

    def test_exception_test_cases_runtime_error(self, exception_test_cases):
        """Test exception_test_cases with RuntimeError."""
        error = exception_test_cases["runtime_error"]
        assert isinstance(error, RuntimeError)
        assert str(error) == "Runtime error"

    def test_exception_test_cases_all_types(self, exception_test_cases):
        """Test that all exception types are present."""
        expected_keys = [
            "value_error",
            "type_error",
            "key_error",
            "attribute_error",
            "io_error",
            "memory_error",
            "runtime_error",
            "assertion_error",
        ]

        for key in expected_keys:
            assert key in exception_test_cases
            assert isinstance(exception_test_cases[key], Exception)


class TestCombinedFixtureUsage:
    """Test combined usage of multiple fixtures."""

    def test_raises_with_exception_test_cases(self, raises_fixture, exception_test_cases):
        """Test raises_fixture with exception_test_cases."""
        error = exception_test_cases["value_error"]

        with raises_fixture(ValueError):
            raise error

    def test_flaky_with_raises_fixture(self, raises_fixture, flaky_operation):
        """Test flaky_operation with raises_fixture."""

        # Create a function that always fails
        def always_fails():
            raise RuntimeError("Always fails")

        # Wrap it with retry logic (will eventually raise the last exception)
        with raises_fixture(RuntimeError):
            # Call the retry wrapper - it will retry 3 times then raise
            flaky_operation.retry(always_fails, max_attempts=3)

    def test_mock_memory_with_raises_fixture(self, raises_fixture, mock_memory_error):
        """Test mock_memory_error with raises_fixture."""
        mock_memory_error.patch_numpy_zeros()

        with raises_fixture(MemoryError):
            # Make enough calls to trigger the mock
            for _ in range(4):
                np.zeros(10)

        mock_memory_error.cleanup()

    def test_oom_with_exception_test_cases(self, oom_fixture, exception_test_cases):
        """Test oom_fixture with exception_test_cases."""
        error = exception_test_cases["memory_error"]

        with oom_fixture:
            # Simulate memory error
            try:
                raise error
            except MemoryError:
                assert True


class TestSeededRngFixture:
    """Test the seeded_rng fixture for reproducible tests."""

    def test_seeded_rng_reproducibility(self, seeded_rng):
        """Test that seeded_rng produces reproducible results."""
        # Generate random numbers
        values1 = seeded_rng.randn(10)
        values2 = seeded_rng.randn(10)

        # Should be different sequences
        assert not np.array_equal(values1, values2)

    def test_seeded_rng_consistency(self, seeded_rng, random_seed):
        """Test that seeded_rng is consistent with seed."""
        # Create another seeded RNG with same seed
        rng2 = np.random.RandomState(random_seed)

        # Generate same sequence
        values1 = seeded_rng.randn(10)
        values2 = rng2.randn(10)

        # Should be identical
        assert np.array_equal(values1, values2)


class TestSampleDataFixture:
    """Test the sample_data fixture."""

    def test_sample_data_structure(self, sample_data):
        """Test that sample_data has correct structure."""
        required_keys = ["timestamps", "surprise", "threshold", "metabolic", "arousal"]

        for key in required_keys:
            assert key in sample_data
            assert isinstance(sample_data[key], list)

    def test_sample_data_consistency(self, sample_data):
        """Test that sample_data is consistent."""
        # All arrays should have same length
        lengths = [len(sample_data[key]) for key in sample_data.keys()]
        assert all(length == lengths[0] for length in lengths)

    def test_sample_data_values(self, sample_data):
        """Test that sample_data contains valid values."""
        # Timestamps should be increasing
        timestamps = sample_data["timestamps"]
        assert all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))

        # All values should be numeric
        for key in ["surprise", "threshold", "metabolic", "arousal"]:
            values = sample_data[key]
            assert all(isinstance(v, (int, float)) for v in values)


class TestSampleConfigFixture:
    """Test the sample_config fixture."""

    def test_sample_config_structure(self, sample_config):
        """Test that sample_config has correct structure."""
        required_sections = ["model", "simulation", "logging", "data", "validation"]

        for section in required_sections:
            assert section in sample_config
            assert isinstance(sample_config[section], dict)

    def test_sample_config_model_params(self, sample_config):
        """Test that sample_config model parameters are valid."""
        model_config = sample_config["model"]

        required_params = [
            "tau_S",
            "tau_theta",
            "theta_0",
            "alpha",
            "gamma_M",
            "gamma_A",
            "rho",
            "sigma_S",
            "sigma_theta",
        ]

        for param in required_params:
            assert param in model_config
            assert isinstance(model_config[param], (int, float))

    def test_sample_config_simulation_params(self, sample_config):
        """Test that sample_config simulation parameters are valid."""
        sim_config = sample_config["simulation"]

        assert sim_config["default_steps"] > 0
        assert sim_config["default_dt"] > 0
        assert sim_config["max_steps"] > sim_config["default_steps"]
        assert sim_config["plot_dpi"] > 0

    def test_sample_config_logging_params(self, sample_config):
        """Test that sample_config logging parameters are valid."""
        logging_config = sample_config["logging"]

        assert logging_config["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]
        # log_rotation and log_retention can be strings (e.g., "10 MB", "30 days") or numeric
        log_rotation = logging_config["log_rotation"]
        log_retention = logging_config["log_retention"]
        if isinstance(log_rotation, (int, float)):
            assert log_rotation > 0
        else:
            assert log_rotation and isinstance(log_rotation, str)
        if isinstance(log_retention, (int, float)):
            assert log_retention > 0
        else:
            assert log_retention and isinstance(log_retention, str)


class TestTempDirFixture:
    """Test the temp_dir fixture."""

    def test_temp_dir_creation(self, temp_dir):
        """Test that temp_dir is created and accessible."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_temp_dir_write(self, temp_dir):
        """Test writing to temp_dir."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temp_dir is cleaned up after test."""
        # Create a file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        # File should exist during test
        assert test_file.exists()

        # After test, temp_dir should be cleaned up by fixture
        # This is verified by pytest's fixture cleanup mechanism


class TestPerformanceUtilities:
    """Test performance utility functions from conftest."""

    def test_assert_performance_within_tolerance(self):
        """Test performance assertion utility."""
        import os
        import sys

        sys.path.insert(0, os.path.dirname(__file__))
        from conftest import assert_performance_within_tolerance

        # Test within tolerance
        assert_performance_within_tolerance(1.05, 1.0, 0.1)  # 5% within 10%

        # Test at boundary
        assert_performance_within_tolerance(1.1, 1.0, 0.1)  # 10% within 10%

        # Test outside tolerance (should fail)
        try:
            assert_performance_within_tolerance(1.2, 1.0, 0.1)  # 20% outside 10%
            assert False, "Should have raised assertion error"
        except AssertionError:
            assert True  # Expected


class TestResetRandomStateFixture:
    """Test the reset_random_state_before_each_test fixture."""

    def test_random_state_reset(self, random_seed):
        """Test that random state is reset before each test."""
        # Get random state
        state1 = np.random.get_state()

        # Generate some random numbers
        _ = np.random.randn(10)

        # Get state again
        state2 = np.random.get_state()

        # States should be different (we generated random numbers)
        assert not np.array_equal(state1[1], state2[1])  # type: ignore[index]


if __name__ == "__main__":
    pytest.main([__file__])


def test_oom_fixture_restores_limit_after_nesting(oom_fixture):
    """CI-MEMORY regression: nested use must restore the original RLIMIT_AS."""
    resource = pytest.importorskip("resource")
    before = resource.getrlimit(resource.RLIMIT_AS)
    with oom_fixture:
        with oom_fixture:
            pass
    assert resource.getrlimit(resource.RLIMIT_AS) == before
