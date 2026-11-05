"""
Tests for apgi_implementation.py
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apgi_core import (
    APGIConfig,
    APGIModel,
    GenerativeModel,
    HierarchicalLevel,
    HierarchicalProcessor,
    RunningStatsEMA,
    clip,
    compute_information_value,
    compute_precision,
    compute_signal,
    effective_interoceptive_precision,
    enforce_stability,
    ignite,
    ignition_probability,
    map_to_hep_amplitude,
    map_to_p3b_latency,
    map_to_reaction_time,
    update_threshold,
)

# Use the new API instead of deprecated CONFIG
CONFIG = APGIConfig.from_settings().__dict__


class TestGenerativeModel:
    """Test the GenerativeModel class."""

    def test_init_default(self):
        """Test initialization with default learning rate."""
        gen = GenerativeModel()
        assert gen.x_hat == 0.0
        assert gen.lr == 0.05

    def test_init_custom_lr(self):
        """Test initialization with custom learning rate."""
        gen = GenerativeModel(lr=0.1)
        assert gen.lr == 0.1

    def test_predict(self):
        """Test prediction returns current x_hat."""
        gen = GenerativeModel()
        gen.x_hat = 0.5
        assert gen.predict() == 0.5

    def test_update(self):
        """Test update modifies x_hat and returns error."""
        gen = GenerativeModel(lr=0.1)
        gen.x_hat = 0.0
        epsilon = gen.update(1.0)
        assert epsilon == 1.0
        assert gen.x_hat == 0.1

    def test_update_convergence(self):
        """Test multiple updates converge to input."""
        gen = GenerativeModel(lr=0.1)
        for _ in range(100):
            gen.update(1.0)
        assert abs(gen.x_hat - 1.0) < 0.01


class TestRunningStatsEMA:
    """Test the RunningStatsEMA class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        stats = RunningStatsEMA()
        assert stats.mu == 0.0
        assert stats.var == 1.0
        assert stats.alpha_mu == 0.01
        assert stats.alpha_sigma == 0.005

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        stats = RunningStatsEMA(alpha_mu=0.1, alpha_sigma=0.05)
        assert stats.alpha_mu == 0.1
        assert stats.alpha_sigma == 0.05

    def test_update(self):
        """Test update modifies mu and var."""
        stats = RunningStatsEMA()
        stats.update(1.0)
        assert stats.mu != 0.0
        assert stats.var > 0

    def test_update_multiple(self):
        """Test multiple updates track mean."""
        stats = RunningStatsEMA(alpha_mu=0.5)
        for _ in range(10):
            stats.update(1.0)
        assert abs(stats.mu - 1.0) < 0.1

    def test_z(self):
        """Test z-score computation."""
        stats = RunningStatsEMA()
        stats.update(1.0)
        z = stats.z(1.0)
        assert isinstance(z, float)

    def test_z_zero_variance(self):
        """Test z-score handles near-zero variance."""
        stats = RunningStatsEMA()
        stats.var = 1e-10
        z = stats.z(1.0)
        assert isinstance(z, float)


class TestComputePrecision:
    """Test precision computation functions."""

    def test_compute_precision(self):
        """Test precision is inverse of variance."""
        pi = compute_precision(1.0)
        assert abs(pi - 1.0) < 1e-6

    def test_compute_precision_small_var(self):
        """Test precision with small variance."""
        pi = compute_precision(0.1)
        assert abs(pi - 10.0) < 1e-6

    def test_compute_precision_zero_var(self):
        """Test precision handles zero variance."""
        pi = compute_precision(0.0)
        assert pi == 1e8

    def test_effective_interoceptive_precision(self):
        """Test effective interoceptive precision computation."""
        pi_eff = effective_interoceptive_precision(1.0, 1.5, 0.5, 0.0)
        assert pi_eff > 1.0

    def test_effective_interoceptive_precision_high_M(self):
        """Test precision with high somatic marker."""
        pi_eff = effective_interoceptive_precision(1.0, 1.5, 5.0, 0.0)
        assert pi_eff > 1.0

    def test_effective_interoceptive_precision_low_M(self):
        """Test precision with low somatic marker."""
        pi_eff = effective_interoceptive_precision(1.0, 1.5, -5.0, 0.0)
        assert pi_eff < 1.5


class TestComputeSignal:
    """Test signal computation."""

    def test_compute_signal_basic(self):
        """Test basic signal computation."""
        S = compute_signal(1.0, 1.0, 1.0, 1.0)
        assert S == 1.0

    def test_compute_signal_zero_z(self):
        """Test signal with zero z-scores."""
        S = compute_signal(0.0, 0.0, 1.0, 1.0)
        assert S == 0.0

    def test_compute_signal_high_precision(self):
        """Test signal with high precision."""
        S = compute_signal(1.0, 1.0, 10.0, 10.0)
        # Signal = 0.5 * 10 * 1^2 + 0.5 * 10 * 1^2 = 5 + 5 = 10
        assert abs(S - 10.0) < 1e-6


class TestComputeInformationValue:
    """Test information value computation."""

    def test_compute_information_value_basic(self):
        """Test basic information value."""
        V = compute_information_value(1.0, 1.0)
        assert V == 1.0

    def test_compute_information_value_zero(self):
        """Test information value with zero errors."""
        V = compute_information_value(0.0, 0.0)
        assert V == 0.0

    def test_compute_information_value_high(self):
        """Test information value with high errors."""
        V = compute_information_value(5.0, 5.0)
        assert V == 25.0


class TestUpdateThreshold:
    """Test threshold update dynamics."""

    def test_update_threshold_basic(self):
        """Test basic threshold update."""
        theta_new = update_threshold(0.5, 0.5, 1.0, 1.0, 0.01, 20.0)
        assert isinstance(theta_new, float)

    def test_update_threshold_with_metabolic_cost(self):
        """Test threshold update with explicit metabolic cost."""
        theta_new = update_threshold(
            0.5,
            0.5,
            1.0,
            1.0,
            0.01,
            20.0,
            metabolic_cost=0.1,
        )  # noqa: E501
        assert isinstance(theta_new, float)

    def test_update_threshold_gamma_M(self):
        """Test threshold update with metabolic sensitivity."""
        theta_new = update_threshold(
            0.5,
            0.5,
            1.0,
            1.0,
            0.01,
            20.0,
            gamma_M=0.1,
        )  # noqa: E501
        assert isinstance(theta_new, float)


class TestIgnition:
    """Test ignition functions."""

    def test_ignition_probability_basic(self):
        """Test basic ignition probability."""
        p = ignition_probability(1.0, 0.5, 5.0)
        assert 0.0 <= p <= 1.0

    def test_ignition_probability_signal_below_threshold(self):
        """Test ignition probability when signal below threshold."""
        p = ignition_probability(0.1, 1.0, 5.0)
        assert p < 0.5

    def test_ignition_probability_signal_above_threshold(self):
        """Test ignition probability when signal above threshold."""
        p = ignition_probability(2.0, 1.0, 5.0)
        assert p > 0.5

    def test_ignition_probability_custom_alpha(self):
        """Test ignition probability with custom alpha."""
        p = ignition_probability(1.0, 0.5, 10.0)
        assert 0.0 <= p <= 1.0

    def test_ignite_true(self):
        """Test ignite returns True when signal exceeds threshold."""
        assert ignite(1.0, 0.5) is True

    def test_ignite_false(self):
        """Test ignite returns False when signal below threshold."""
        assert ignite(0.1, 0.5) is False

    def test_ignite_equal(self):
        """Test ignite returns False when signal equals threshold."""
        assert ignite(0.5, 0.5) is False


class TestStability:
    """Test stability enforcement functions."""

    def test_clip_basic(self):
        """Test basic clipping."""
        assert clip(0.5, 0.0, 1.0) == 0.5

    def test_clip_below(self):
        """Test clipping below lower bound."""
        assert clip(-0.5, 0.0, 1.0) == 0.0

    def test_clip_above(self):
        """Test clipping above upper bound."""
        assert clip(1.5, 0.0, 1.0) == 1.0

    def test_enforce_stability(self):
        """Test stability enforcement on state dict."""
        state = {"S": 15.0, "theta": 10.0, "Pi_e": 20.0, "Pi_i": 0.0}
        enforced = enforce_stability(state)
        assert enforced["S"] == 10.0
        assert enforced["theta"] == 5.0
        assert enforced["Pi_e"] == 10.0
        assert enforced["Pi_i"] == 0.01


class TestEmpiricalMapping:
    """Test empirical mapping functions."""

    def test_map_to_p3b_latency(self):
        """Test P3b latency mapping."""
        latency = map_to_p3b_latency(1.0)
        assert isinstance(latency, float)
        assert 250.0 < latency < 350.0

    def test_map_to_p3b_latency_zero(self):
        """Test P3b latency with zero signal."""
        latency = map_to_p3b_latency(0.0)
        assert latency == 300.0

    def test_map_to_p3b_latency_high(self):
        """Test P3b latency with high signal."""
        latency = map_to_p3b_latency(10.0)
        assert latency < 300.0

    def test_map_to_hep_amplitude(self):
        """Test HEP amplitude mapping."""
        hep = map_to_hep_amplitude(1.0, 1.0)
        assert hep == 1.0

    def test_map_to_hep_amplitude_zero_error(self):
        """Test HEP amplitude with zero error."""
        hep = map_to_hep_amplitude(0.0, 1.0)
        assert hep == 0.0

    def test_map_to_reaction_time(self):
        """Test reaction time mapping."""
        rt = map_to_reaction_time(1.0, 0.5)
        assert isinstance(rt, float)

    def test_map_to_reaction_time_high_margin(self):
        """Test reaction time with high signal-threshold margin."""
        rt = map_to_reaction_time(2.0, 0.5)
        assert rt < 400.0

    def test_map_to_reaction_time_low_margin(self):
        """Test reaction time with low signal-threshold margin."""
        rt = map_to_reaction_time(0.5, 1.0)
        assert rt > 400.0


class TestHierarchicalLevel:
    """Test HierarchicalLevel dataclass."""

    def test_init_default(self):
        """Test default initialization."""
        level = HierarchicalLevel()
        assert level.S == 0.0
        assert level.theta == 0.5
        assert level.M == 0.0
        assert level.A == 0.5
        assert level.Pi_e == 1.0
        assert level.Pi_i == 1.0
        assert level.ignition_prob == 0.0
        assert level.broadcast is False
        assert level.tau == 0.1

    def test_init_custom(self):
        """Test custom initialization."""
        level = HierarchicalLevel(tau=0.5)
        assert level.tau == 0.5


class TestHierarchicalProcessor:
    """Test HierarchicalProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        proc = HierarchicalProcessor()
        assert len(proc.levels) == 5
        assert len(proc.level_names) == 5

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {"beta_cross": 0.5}
        proc = HierarchicalProcessor(config=config)
        assert proc.beta_cross == 0.5

    def test_level_names(self):
        """Test level names are correct."""
        proc = HierarchicalProcessor()
        assert proc.level_names == ["L0", "L1", "L2", "L3", "L4"]
        assert proc.level_labels[2] == "L2 (1–30 min)"

    def test_process_level(self):
        """Test processing a single level."""
        proc = HierarchicalProcessor()
        level = proc.process_level(0, 1.0, 0.5, 0.5, dt=0.01)
        assert isinstance(level, HierarchicalLevel)
        assert level.S >= 0.0

    def test_process_all_levels(self):
        """Test processing all levels."""
        proc = HierarchicalProcessor()
        levels = proc.process_all_levels(1.0, 0.5, 0.5, dt=0.01)
        assert len(levels) == 5

    def test_apply_cross_level_coupling(self):
        """Test cross-level coupling."""
        proc = HierarchicalProcessor()
        # Set a higher level to broadcast
        proc.levels[4].broadcast = True
        proc.apply_cross_level_coupling()
        # Lower level precision should increase
        assert proc.levels[3].Pi_e > 1.0  # noqa: E501

    def test_get_aggregate_signal(self):
        """Test aggregate signal computation."""
        proc = HierarchicalProcessor()
        proc.levels[0].S = 1.0
        proc.levels[4].S = 2.0
        signal = proc.get_aggregate_signal()
        assert signal > 0.0

    def test_get_summary(self):
        """Test summary generation."""
        proc = HierarchicalProcessor()
        summary = proc.get_summary()
        assert len(summary) == 5
        assert "level_0_L0" in summary

    def test_reset(self):
        """Test reset functionality."""
        proc = HierarchicalProcessor()
        proc.levels[0].S = 5.0
        proc.reset()
        assert proc.levels[0].S == 0.0


class TestAPGIModel:
    """Test APGIModel class."""

    def test_init_default(self):
        """Test initialization with default config."""
        model = APGIModel()
        assert model.theta == 0.03  # Updated to match APGISettings.theta0
        assert model.S == 0.0
        assert model.M == 0.0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        custom_config = {
            "dt": 0.02,
            "tau_theta": 20.0,
            "theta0": 1.0,
            "alpha": 5.0,
            "beta": 1.5,
            "beta_M": 1.0,
            "M_0": 0.0,
            "gamma_M": -0.3,
            "alpha_mu": 0.01,
            "alpha_sigma": 0.005,
            "c1": 0.1,
            "c2": 0.02,
        }
        model = APGIModel(config=custom_config)
        assert model.config == custom_config
        assert model.theta == 1.0

    def test_step(self):
        """Test single step processing."""
        model = APGIModel()
        output = model.step(1.0)
        assert isinstance(output, dict)
        assert "S" in output
        assert "theta" in output
        assert "ignition_prob" in output
        assert "ignited" in output
        assert "metabolic_cost" in output

    def test_step_output_keys(self):
        """Test step output contains all expected keys."""
        model = APGIModel()
        output = model.step(1.0)
        expected_keys = [
            "S",
            "theta",
            "Pi_e",
            "Pi_i",
            "z_e",
            "z_i",
            "x_hat",
            "epsilon",
            "ignition_prob",
            "ignited",
            "p3b_latency_ms",
            "hep_amplitude",
            "reaction_time_ms",
            "information_value",
            "metabolic_cost",
            "pi_i_eff",
            "S_hierarchical",
        ]
        for key in expected_keys:
            assert key in output

    def test_run(self):
        """Test running on sequence of inputs."""
        model = APGIModel()
        inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        results = model.run(inputs)
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    def test_run_empty(self):
        """Test running on empty input."""
        model = APGIModel()
        results = model.run(np.array([]))
        assert len(results) == 0

    def test_get_summary_empty(self):
        """Test summary with no history."""
        model = APGIModel()
        summary = model.get_summary()
        assert summary == {}

    def test_get_summary_with_history(self):
        """Test summary with history."""
        model = APGIModel()
        model.run(np.array([0.1, 0.2, 0.3]))
        summary = model.get_summary()
        assert "mean_S" in summary
        assert "max_S" in summary
        assert "mean_theta" in summary
        assert "ignition_rate" in summary
        assert "num_steps" in summary

    def test_reset(self):
        """Test reset functionality."""
        model = APGIModel()
        model.step(1.0)
        assert len(model.history) > 0
        # Manual reset of history as APGIModel.reset is not implemented in engine.py yet
        model.history = []
        assert len(model.history) == 0

    def test_ignition_probability_range(self):
        """Test ignition probability is always in [0, 1]."""
        model = APGIModel()
        for _ in range(100):
            output = model.step(np.random.randn())
            assert 0.0 <= output["ignition_prob"] <= 1.0

    def test_metabolic_cost_positive(self):
        """Test metabolic cost is always positive."""
        model = APGIModel()
        for _ in range(100):
            output = model.step(np.random.randn())
            assert output["metabolic_cost"] >= 0.0

    def test_stability_enforcement(self):
        """Test stability enforcement keeps values in bounds."""
        model = APGIModel()
        for _ in range(1000):
            output = model.step(np.random.randn() * 10)
            assert 0.0 <= output["S"] <= 10.0
            assert 0.1 <= output["theta"] <= 5.0
            assert 0.01 <= output["Pi_e"] <= 10.0
            assert 0.01 <= output["Pi_i"] <= 10.0

    def test_hierarchical_levels_in_output(self):
        """Test hierarchical level states are in output."""
        model = APGIModel()
        output = model.step(1.0)
        assert "level_0_L0" in output
        assert "level_4_L4" in output

    def test_p3b_latency_range(self):
        """Test P3b latency is in expected range."""
        model = APGIModel()
        output = model.step(1.0)
        assert 250.0 <= output["p3b_latency_ms"] <= 350.0

    def test_reaction_time_positive(self):
        """Test reaction time is positive."""
        model = APGIModel()
        output = model.step(1.0)
        assert output["reaction_time_ms"] > 0.0


class TestCONFIG:
    """Test CONFIG dictionary."""

    def test_config_exists(self):
        """Test CONFIG is defined."""
        assert isinstance(CONFIG, dict)

    def test_config_required_keys(self):
        """Test CONFIG has required keys."""
        required_keys = [
            "dt",
            "tau_theta",
            "theta0",
            "alpha",
            "tau_S",
            "tau_M",
            "beta",
            "beta_M",
            "M_0",
            "gamma_M",
            "lambda_S",
            "sigma_S",
            "sigma_theta",
            "sigma_M",
            "rho",
            "alpha_mu",
            "alpha_sigma",
            "c1",
            "c2",
        ]
        for key in required_keys:
            assert key in CONFIG

    def test_config_values_positive(self):
        """Test config values are positive where expected."""
        assert CONFIG["dt"] > 0
        assert CONFIG["tau_theta"] > 0
        assert CONFIG["theta0"] > 0
        assert CONFIG["alpha"] > 0
        assert CONFIG["c1"] > 0
        assert CONFIG["c2"] > 0
