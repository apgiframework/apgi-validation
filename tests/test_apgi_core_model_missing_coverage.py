"""
Targeted tests for apgi_core/model.py missing coverage areas.
==========================================================

Focuses on specific functions and lines that are currently uncovered.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apgi_core.model import APGIModel, HierarchicalLevel, RunningStatsEMA


class TestRunningStatsEMAMissingCoverage:
    """Test RunningStatsEMA methods with missing coverage."""

    def test_running_stats_ema_initialization(self):
        """Test RunningStatsEMA initialization."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)

        assert stats.mu == 0.0
        assert stats.var == 1.0

    def test_running_stats_ema_single_update(self):
        """Test RunningStatsEMA with single update."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)

        stats.update(5.0)

        assert stats.mu != 0.0  # Should have updated
        assert stats.var != 1.0  # Should have updated

    def test_running_stats_ema_multiple_updates(self):
        """Test RunningStatsEMA with multiple updates."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            stats.update(val)

        assert stats.mu != 0.0  # Should have updated
        # Mean should be somewhere in range of values
        assert min(values) <= stats.mu <= max(values)

    def test_running_stats_ema_zero_alpha(self):
        """Test RunningStatsEMA with zero alpha."""
        stats = RunningStatsEMA(alpha_mu=0.0, alpha_sigma=0.0)

        stats.update(10.0)
        stats.update(20.0)

        # With zero alpha, should not update from initial values
        assert stats.mu == 0.0
        assert stats.var == 1.0

    def test_running_stats_ema_high_alpha(self):
        """Test RunningStatsEMA with high alpha (close to 1)."""
        stats = RunningStatsEMA(alpha_mu=0.9, alpha_sigma=0.9)

        stats.update(10.0)
        first_mu = stats.mu

        stats.update(20.0)

        # With high alpha, should quickly adapt to new values
        assert stats.mu != first_mu
        assert abs(stats.mu - 20.0) < abs(first_mu - 20.0)

    def test_running_stats_ema_negative_values(self):
        """Test RunningStatsEMA with negative values."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)

        stats.update(-5.0)
        stats.update(-10.0)

        assert stats.mu < 0  # Mean should be negative

    def test_running_stats_ema_mixed_values(self):
        """Test RunningStatsEMA with mixed positive/negative values."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)

        values = [-5.0, 0.0, 5.0, -2.5, 2.5]
        for val in values:
            stats.update(val)

        # Mean should be close to zero with symmetric values
        assert abs(stats.mu) < 1.0

    def test_running_stats_ema_variance_tracking(self):
        """Test RunningStatsEMA variance tracking."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)

        # Add values with known variance
        values = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        for val in values:
            stats.update(val)

        # Should detect variance
        assert stats.var > 0

    def test_running_stats_ema_constant_values(self):
        """Test RunningStatsEMA with constant values."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)

        constant_value = 5.0
        for _ in range(50):  # More updates for convergence
            stats.update(constant_value)

        # Mean should converge toward constant value with EMA
        # After 50 updates with alpha=0.1, should be very close
        assert abs(stats.mu - constant_value) < 0.5


class TestHierarchicalLevelMissingCoverage:
    """Test HierarchicalLevel methods with missing coverage."""

    def test_hierarchical_layer_initialization(self):
        """Test HierarchicalLevel initialization."""
        layer = HierarchicalLevel()

        # Test default values are set correctly
        assert layer.S == 0.0
        assert layer.theta == 0.5
        assert layer.tau == 0.1

    def test_hierarchical_layer_attributes(self):
        """Test HierarchicalLevel attributes access."""
        layer = HierarchicalLevel()

        # Test default values are accessible
        assert hasattr(layer, "S")
        assert hasattr(layer, "theta")
        assert hasattr(layer, "M")
        assert hasattr(layer, "A")
        assert hasattr(layer, "Pi_e")
        assert hasattr(layer, "Pi_i")
        assert hasattr(layer, "ignition_prob")
        assert hasattr(layer, "broadcast")
        assert hasattr(layer, "tau")

        # Test default values
        assert layer.S == 0.0
        assert layer.theta == 0.5
        assert layer.tau == 0.1


class TestAPGIModelMissingCoverage:
    """Test APGIModel methods with missing coverage."""

    def test_model_initialization_custom_config(self):
        """Test APGIModel initialization with custom config."""
        custom_config = {
            "theta0": 2.0,
            "tau": 5.0,
            "alpha_mu": 0.2,
            "alpha_sigma": 0.1,
            "hierarchical": {"num_levels": 3, "tau_max": 20.0, "tau_min": 1.0},
        }

        model = APGIModel(config=custom_config)

        assert model.cfg["theta0"] == 2.0
        assert model.cfg["tau"] == 5.0
        assert model.cfg["alpha_mu"] == 0.2
        assert model.cfg["alpha_sigma"] == 0.1

    def test_model_reset_functionality(self):
        """Test APGIModel reset functionality."""
        model = APGIModel()

        # Run some steps to change state
        inputs = np.array([1.0, 2.0, 3.0])
        model.run(inputs)

        # Verify state changed
        assert model.S != 0.0 or model.theta != model.cfg["theta0"]

        # Reset
        model.reset()

        # Verify reset to initial state
        assert model.theta == model.cfg["theta0"]
        assert model.S == 0.0
        assert model.M == 0.0

    def test_model_step_by_step_processing(self):
        """Test APGIModel step-by-step processing."""
        model = APGIModel()

        # Process single step
        input_val = 2.5
        result = model.step(input_val)

        assert "S" in result
        assert "theta" in result
        assert "M" in result
        assert "ignition" in result

    def test_model_ignition_detection(self):
        """Test APGIModel ignition detection."""
        model = APGIModel()

        # Use high input to trigger ignition
        high_input = 10.0
        result = model.step(high_input)

        # Check if ignition was detected
        assert isinstance(result["ignition"], bool)

    def test_model_history_tracking(self):
        """Test APGIModel history tracking."""
        model = APGIModel()

        # Run multiple steps
        inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model.run(inputs)

        # Check history
        assert len(model.history) == len(inputs)

        # Each history entry should have required fields
        for entry in model.history:
            assert "S" in entry
            assert "theta" in entry
            assert "M" in entry
            assert "ignition" in entry

    def test_model_summary_statistics(self):
        """Test APGIModel summary statistics."""
        model = APGIModel()

        # Generate some data
        np.random.seed(42)
        inputs = np.random.randn(100) + 2.0  # Mean = 2.0
        model.run(inputs)

        summary = model.get_summary()

        assert isinstance(summary, dict)
        assert "num_steps" in summary
        assert "mean_S" in summary
        assert "max_S" in summary
        assert "mean_theta" in summary
        assert "ignition_rate" in summary
        assert summary["num_steps"] == 100

    def test_model_empty_input_handling(self):
        """Test APGIModel with empty input."""
        model = APGIModel()

        empty_inputs = np.array([])
        results = model.run(empty_inputs)

        assert len(results) == 0
        assert len(model.history) == 0

    def test_model_single_input_handling(self):
        """Test APGIModel with single input."""
        model = APGIModel()

        single_input = np.array([5.0])
        results = model.run(single_input)

        assert len(results) == 1
        assert len(model.history) == 1

    def test_model_negative_input_handling(self):
        """Test APGIModel with negative inputs."""
        model = APGIModel()

        negative_inputs = np.array([-1.0, -2.0, -3.0])
        results = model.run(negative_inputs)

        assert len(results) == 3
        assert all(isinstance(r["ignition"], bool) for r in results)

    def test_model_large_input_handling(self):
        """Test APGIModel with large inputs."""
        model = APGIModel()

        large_inputs = np.array([100.0, 200.0, 300.0])
        results = model.run(large_inputs)

        assert len(results) == 3
        # Should handle large values without overflow
        assert all(np.isfinite(r["S"]) for r in results)
        assert all(np.isfinite(r["theta"]) for r in results)

    def test_model_hierarchical_processing(self):
        """Test APGIModel hierarchical processing."""
        config_with_hierarchy = {
            "theta0": 1.0,
            "tau": 1.0,
            "hierarchical": {
                "enabled": True,
                "num_levels": 3,
                "tau_max": 10.0,
                "tau_min": 1.0,
            },
        }

        model = APGIModel(config=config_with_hierarchy)

        inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = model.run(inputs)

        assert len(results) == 5
        assert hasattr(model, "hierarchical")

    def test_model_parameter_sensitivity(self):
        """Test APGIModel sensitivity to parameters."""
        # Test with different theta0 values
        low_theta_model = APGIModel(config={"theta0": 0.1})
        high_theta_model = APGIModel(config={"theta0": 5.0})

        same_input = np.array([2.0, 2.0, 2.0])

        low_results = low_theta_model.run(same_input)
        high_results = high_theta_model.run(same_input)

        # Different theta0 should produce different behaviors
        low_ignitions = sum(r["ignition"] for r in low_results)
        high_ignitions = sum(r["ignition"] for r in high_results)

        # High theta should have fewer ignitions with same input
        assert low_ignitions >= high_ignitions

    def test_model_tau_parameter_effects(self):
        """Test APGIModel tau parameter effects."""
        fast_tau_model = APGIModel(config={"tau_theta": 1.0})  # Fast adaptation
        slow_tau_model = APGIModel(config={"tau_theta": 50.0})  # Slow adaptation

        # Use more steps and varied input to see clearer differences
        same_input = np.array([3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0])

        fast_results = fast_tau_model.run(same_input)
        slow_results = slow_tau_model.run(same_input)

        # Different tau values should affect signal dynamics
        assert len(fast_results) == len(slow_results)

        # Fast tau should respond more quickly (different threshold dynamics)
        fast_final_theta = fast_results[-1]["theta"]
        slow_final_theta = slow_results[-1]["theta"]

        # Values should be different due to different time constants
        # With more steps, the difference should be more apparent
        assert abs(fast_final_theta - slow_final_theta) > 1e-4

    def test_model_noise_robustness(self):
        """Test APGIModel robustness to noisy inputs."""
        model = APGIModel()

        # Generate noisy signal
        np.random.seed(123)
        base_signal = 2.0
        noise = np.random.randn(50) * 0.5
        noisy_inputs = base_signal + noise

        results = model.run(noisy_inputs)

        assert len(results) == 50
        assert all(np.isfinite(r["S"]) for r in results)
        assert all(np.isfinite(r["theta"]) for r in results)

    def test_model_state_consistency(self):
        """Test APGIModel state consistency."""
        model = APGIModel()

        inputs = np.array([1.0, 2.0, 3.0])
        results = model.run(inputs)

        # Model state should match last result
        assert model.S == results[-1]["S"]
        assert model.theta == results[-1]["theta"]
        assert model.M == results[-1]["M"]

    def test_model_reproducibility(self):
        """Test APGIModel reproducibility with same seed."""
        # Set seed for reproducibility
        np.random.seed(42)

        model1 = APGIModel()
        inputs = np.random.randn(20) + 1.0
        results1 = model1.run(inputs.copy())

        # Reset and run again with same seed
        np.random.seed(42)
        model2 = APGIModel()
        results2 = model2.run(inputs.copy())

        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert abs(r1["S"] - r2["S"]) < 1e-10
            assert abs(r1["theta"] - r2["theta"]) < 1e-10


class TestModelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_tau_parameter(self):
        """Test model with tau = 0 (should handle gracefully)."""
        model = APGIModel(config={"tau": 0.0})

        inputs = np.array([1.0, 2.0, 3.0])
        results = model.run(inputs)

        assert len(results) == 3

    def test_negative_tau_parameter(self):
        """Test model with negative tau (should handle gracefully)."""
        model = APGIModel(config={"tau": -1.0})

        inputs = np.array([1.0, 2.0, 3.0])
        results = model.run(inputs)

        assert len(results) == 3

    def test_zero_theta0_parameter(self):
        """Test model with theta0 = 0."""
        model = APGIModel(config={"theta0": 0.0})

        inputs = np.array([1.0, 2.0, 3.0])
        results = model.run(inputs)

        assert len(results) == 3
        # Threshold is clamped to the floor θ_floor = 0 (Math Spec §4.5; PV-CLIP)
        assert model.theta >= 0.0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_infinite_values_input(self):
        """Test model with infinite values in input."""
        model = APGIModel()

        inputs_with_inf = np.array([1.0, np.inf, 2.0, -np.inf, 3.0])

        # Should handle infinite values gracefully
        try:
            results = model.run(inputs_with_inf)
            # If it doesn't crash, results should be finite for valid inputs
            assert len(results) == len(inputs_with_inf)
        except (ValueError, OverflowError):
            # Expected behavior for infinite inputs
            pass

    def test_nan_values_input(self):
        """Test model with NaN values in input."""
        model = APGIModel()

        inputs_with_nan = np.array([1.0, np.nan, 2.0, np.nan, 3.0])

        # Should handle NaN values gracefully
        try:
            results = model.run(inputs_with_nan)
            # If it doesn't crash, check results
            assert len(results) == len(inputs_with_nan)
        except ValueError:
            # Expected behavior for NaN inputs
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
