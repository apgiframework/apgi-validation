"""
Tests for Theory/APGI_Information_Theoretic_Bandwidth.py module
========================================================================

Comprehensive test suite for APGI Information-Theoretic Bandwidth analysis.
Tests all classes and functions to achieve >90% coverage.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBinaryEntropy:
    """Test binary entropy function."""

    def test_binary_entropy_basic(self):
        """Test basic binary entropy calculation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _binary_entropy

        # Test with known probabilities
        p = np.array([0.5, 0.5])
        result = _binary_entropy(p)
        expected = -0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)  # = 1.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_binary_entropy_edge_cases(self):
        """Test binary entropy edge cases."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _binary_entropy

        # Test with p=0 and p=1 (should handle clipping)
        p_zero = np.array([0.0, 1.0])
        result_zero = _binary_entropy(p_zero)
        assert np.all(np.isfinite(result_zero))  # Should not be NaN due to clipping

        # _binary_entropy is elementwise binary entropy,
        #   H(p) = -p·log2 p - (1-p)·log2(1-p),
        # not the Shannon entropy of a multi-outcome distribution. The previous
        # expectation (2.0 bits for four 0.25 entries) confused the two.
        p = np.array([0.25, 0.25, 0.25, 0.25])
        expected = -0.25 * np.log2(0.25) - 0.75 * np.log2(0.75)  # ≈ 0.8113
        np.testing.assert_allclose(_binary_entropy(p), np.full(4, expected), rtol=1e-10)

        # Maximum at p = 0.5, and symmetric about it.
        np.testing.assert_allclose(_binary_entropy(np.array([0.5])), np.array([1.0]), rtol=1e-10)
        np.testing.assert_allclose(_binary_entropy(np.array([0.3])), _binary_entropy(np.array([0.7])), rtol=1e-10)


class TestSigmoid:
    """Test sigmoid function."""

    def test_sigmoid_basic(self):
        """Test basic sigmoid functionality."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _sigmoid

        x = np.array([0.0, 1.0, -1.0])
        result = _sigmoid(x, alpha=1.0)

        # Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731, Sigmoid(-1) ≈ 0.269
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_sigmoid_clipping(self):
        """Test sigmoid clipping for large values."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _sigmoid

        x = np.array([1000.0, -1000.0])
        result = _sigmoid(x, alpha=1.0)

        # Should be clipped to avoid overflow
        assert np.all(result >= 0.0) & np.all(result <= 1.0)


class TestIgnitionProbPerLevel:
    """Test ignition probability per level function."""

    def test_ignition_prob_basic(self):
        """Test basic ignition probability calculation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _ignition_prob_per_level

        stimulus_levels = np.array([1.0, 2.0, 3.0])
        Pi_e = 0.8
        Pi_i = 0.2
        theta = 0.3
        alpha = 5.0

        result = _ignition_prob_per_level(stimulus_levels, Pi_e, Pi_i, theta, alpha)

        # Should return probabilities for each level
        assert result.shape == (3,)
        assert np.all(result >= 0.0) & np.all(result <= 1.0)

    def test_ignition_prob_edge_cases(self):
        """Test ignition probability edge cases."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _ignition_prob_per_level

        stimulus_levels = np.array([1.0, 2.0, 3.0])
        theta = 0.3
        alpha = 5.0

        # _ignition_prob_per_level returns P(ignition | stimulus = level) for
        # each level — NOT a probability distribution over levels. The previous
        # expectations ([1,0,0] / [0,0,1], "all probability at lowest level")
        # treated it as a distribution and could never hold.
        high_precision = _ignition_prob_per_level(stimulus_levels, Pi_e=1.0, Pi_i=0.0, theta=theta, alpha=alpha)
        assert np.all((high_precision >= 0.0) & (high_precision <= 1.0))
        # Stronger exteroceptive evidence must not lower ignition probability.
        assert np.all(
            np.diff(high_precision) >= -1e-12
        ), f"Ignition probability should be non-decreasing in stimulus strength, got {high_precision}"

        # With no exteroceptive weight the stimulus cannot drive ignition, so
        # the response is flat across levels *in expectation*. The function
        # integrates over Monte Carlo interoceptive noise, so allow sampling
        # spread rather than demanding exact equality.
        no_drive = _ignition_prob_per_level(stimulus_levels, Pi_e=0.0, Pi_i=1.0, theta=theta, alpha=alpha)
        assert np.all((no_drive >= 0.0) & (no_drive <= 1.0))
        assert (
            np.ptp(no_drive) < 0.15
        ), f"Without exteroceptive drive, ignition should not track stimulus level: {no_drive}"


class TestMutualInfoBits:
    """Test mutual information bits function."""

    def test_mutual_info_identical_labels_equals_entropy(self):
        """I(X;X) = H(X), the maximum, not zero.

        This assertion previously demanded 0.0 for identical labels, which is
        mathematically backwards — identical variables share *all* their
        information. The error was invisible because a sklearn/scipy skew made
        the function raise TypeError before the comparison was reached.
        """
        from Theory.APGI_Information_Theoretic_Bandwidth import _mutual_info_bits

        labels = np.array([1, 2, 3, 1, 2])
        # p = [2/5, 2/5, 1/5] -> H = -(0.4·log2 0.4)·2 - 0.2·log2 0.2
        probabilities = np.array([2 / 5, 2 / 5, 1 / 5])
        expected_entropy = float(-np.sum(probabilities * np.log2(probabilities)))

        assert _mutual_info_bits(labels, labels) == pytest.approx(expected_entropy, abs=1e-9)

    def test_mutual_info_is_symmetric(self):
        """I(A;B) == I(B;A)."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _mutual_info_bits

        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([0, 1, 1, 2, 2, 0])
        assert _mutual_info_bits(a, b) == pytest.approx(_mutual_info_bits(b, a), abs=1e-9)

    def test_mutual_info_independent_labels_near_zero(self):
        """Independent variables share no information."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _mutual_info_bits

        rng = np.random.default_rng(42)
        a = rng.integers(0, 2, size=4000)
        b = rng.integers(0, 2, size=4000)
        assert _mutual_info_bits(a, b) < 0.01

    def test_mutual_info_never_negative(self):
        """MI is non-negative by definition."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _mutual_info_bits

        rng = np.random.default_rng(7)
        for _ in range(10):
            a = rng.integers(0, 4, size=200)
            b = rng.integers(0, 4, size=200)
            assert _mutual_info_bits(a, b) >= 0.0

    def test_mutual_info_different(self):
        """Deterministically related labels carry positive information."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _mutual_info_bits

        labels_a = np.array([1, 1, 1, 2, 2])
        labels_b = np.array([2, 2, 2, 1, 1])
        result = _mutual_info_bits(labels_a, labels_b)
        assert result > 0.0  # Should be positive for different distributions


class TestBandwidthDerivation:
    """Test BandwidthDerivation class."""

    def test_bandwidth_derivation_init(self):
        """Test BandwidthDerivation initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        model = BandwidthDerivation()
        assert hasattr(model, "compute")
        assert hasattr(model, "parameter_sweep")

    def test_bandwidth_derivation_compute(self):
        """Test bandwidth computation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        model = BandwidthDerivation()

        # Test with default parameters
        result = model.compute()

        # Field names taken from the BandwidthResult dataclass. The previous
        # assertions named "bandwidth_bps" and "prediction_in_range", neither of
        # which has ever existed on this type.
        assert hasattr(result, "bandwidth_bits_per_second")
        assert hasattr(result, "bandwidth_in_gamma_prediction")
        assert isinstance(result.bandwidth_bits_per_second, float)
        assert isinstance(result.bandwidth_in_gamma_prediction, bool)

        # Information-theoretic invariants.
        assert result.I_bits_per_trial >= 0.0
        assert 0.0 <= result.P_ignition <= 1.0
        assert result.H_B >= result.H_B_given_S, "Conditioning cannot increase entropy"
        assert result.bandwidth_bits_per_second >= 0.0

    def test_bandwidth_derivation_parameter_sweep(self):
        """Test parameter sweep functionality."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        model = BandwidthDerivation()

        # parameter_sweep returns a dict of sampled arrays and summary
        # statistics, not an object with bandwidth_matrix/optimal_params.
        result = model.parameter_sweep()
        assert isinstance(result, dict)
        for key in ("bandwidths_gamma", "bandwidth_mean", "bandwidth_median", "n_samples"):
            assert key in result, f"parameter_sweep result missing {key}"
        assert result["n_samples"] > 0
        assert result["bandwidth_p5"] <= result["bandwidth_median"] <= result["bandwidth_p95"]


class TestPrecisionWeightingAnalyzer:
    """Test PrecisionWeightingAnalyzer class."""

    def test_precision_weighting_init(self):
        """Test PrecisionWeightingAnalyzer initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation, PrecisionWeightingAnalyzer

        bandwidth_module = BandwidthDerivation()
        analyzer = PrecisionWeightingAnalyzer(bandwidth_module)
        assert hasattr(analyzer, "compare")
        assert analyzer.bw == bandwidth_module

    def test_precision_weighting_compare(self):
        """Test precision weighting comparison."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation, PrecisionWeightingAnalyzer

        bandwidth_module = BandwidthDerivation()
        analyzer = PrecisionWeightingAnalyzer(bandwidth_module)

        # Field names from the PrecisionGainResult dataclass. The previous
        # assertions named "delta_I_pct" and "mi_gain_bits", neither of which
        # exists on this type.
        result = analyzer.compare(theta_baseline=0.1, alpha=0.5)
        for field in ("I_precision_on", "I_precision_off", "delta_I", "delta_I_pct_capacity"):
            assert hasattr(result, field), f"PrecisionGainResult missing {field}"

        assert isinstance(result.delta_I, float)
        # delta_I is the information gained by enabling precision weighting.
        assert result.delta_I == pytest.approx(result.I_precision_on - result.I_precision_off, abs=1e-9)
        assert result.bandwidth_on_bps >= 0.0
        assert result.bandwidth_off_bps >= 0.0


class TestClinicalProfile:
    """Test ClinicalProfile class."""

    def test_clinical_profile_init(self):
        """Test ClinicalProfile initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import ClinicalProfile

        profile = ClinicalProfile(
            label="Test Disorder",
            Pi_e=1.0,
            Pi_i=0.1,
            theta_baseline=0.1,
            alpha=0.5,
            sigma_theta=0.1,
            description="Test profile",
        )

        assert profile.label == "Test Disorder"
        assert profile.Pi_e == 1.0
        assert profile.alpha == 0.5

    def test_clinical_profile_validation(self):
        """Test clinical profile validation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import ClinicalProfile

        # Test valid profile
        valid_profile = ClinicalProfile(
            label="Valid",
            Pi_e=1.0,
            Pi_i=0.1,
            theta_baseline=0.1,
            alpha=0.5,
            sigma_theta=0.1,
            description="Valid profile",
        )
        assert valid_profile.validate_parameters()

        # Test invalid profile. The previous "invalid" fixture here had no
        # negative parameter at all despite its comment, so it was
        # indistinguishable from the valid one.
        invalid_profile = ClinicalProfile(
            label="Test Disorder",
            Pi_e=-1.0,  # precision is an inverse variance: must be positive
            Pi_i=0.8,
            theta_baseline=0.3,
            alpha=0.5,
            sigma_theta=0.05,
            description="Test profile",
        )
        assert not invalid_profile.validate_parameters()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"Pi_e": -1.0},
            {"Pi_e": 0.0},
            {"Pi_i": -0.2},
            {"alpha": 0.0},
            {"sigma_theta": -0.1},
            {"Pi_e": float("nan")},
            {"theta_baseline": float("inf")},
        ],
    )
    def test_invalid_profiles_rejected(self, overrides):
        from Theory.APGI_Information_Theoretic_Bandwidth import ClinicalProfile

        params = dict(
            label="X",
            Pi_e=1.0,
            Pi_i=0.5,
            theta_baseline=0.3,
            alpha=2.0,
            sigma_theta=0.0,
            description="d",
        )
        params.update(overrides)
        assert not ClinicalProfile(**params).validate_parameters()

    def test_canonical_profiles_are_valid(self):
        """Every shipped clinical profile must be physically meaningful."""
        from Theory.APGI_Information_Theoretic_Bandwidth import CLINICAL_PROFILES

        invalid = [name for name, p in CLINICAL_PROFILES.items() if not p.validate_parameters()]
        assert invalid == [], f"Canonical profiles failed validation: {invalid}"


class TestClinicalBandwidthMapper:
    """Test ClinicalBandwidthMapper class."""

    def test_clinical_mapper_init(self):
        """Test ClinicalBandwidthMapper initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation, ClinicalBandwidthMapper

        bandwidth_module = BandwidthDerivation()
        mapper = ClinicalBandwidthMapper(bandwidth_module)
        assert hasattr(mapper, "compute_all")
        assert mapper.bw == bandwidth_module

    def test_clinical_mapper_compute(self):
        """Test clinical bandwidth computation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import (
            BandwidthDerivation,
            ClinicalBandwidthMapper,
            ClinicalProfile,
        )

        bandwidth_module = BandwidthDerivation()
        mapper = ClinicalBandwidthMapper(bandwidth_module)

        # Create test profiles. compute_all contrasts every disorder against an
        # "HC" (healthy control) baseline, so that key is required — omitting it
        # previously produced a bare KeyError from deep inside the heatmap step.
        profiles = {
            "HC": ClinicalProfile(
                label="Healthy Control",
                Pi_e=2.5,
                Pi_i=0.8,
                theta_baseline=0.3,
                alpha=5.0,
                sigma_theta=0.0,
                description="Test HC baseline",
            ),
            "MDD": ClinicalProfile(
                label="MDD",
                Pi_e=1.0,
                Pi_i=0.1,
                theta_baseline=0.1,
                alpha=0.5,
                sigma_theta=0.1,
                description="Test MDD profile",
            ),
            "ADHD": ClinicalProfile(
                label="ADHD",
                Pi_e=0.8,
                Pi_i=0.05,
                theta_baseline=0.05,
                alpha=0.3,
                sigma_theta=0.05,
                description="Test ADHD profile",
            ),
        }

        result = mapper.compute_all(profiles=profiles)

        # Field names from the ClinicalBandwidthResult dataclass. The previous
        # assertion looked for a "clinical" attribute that has never existed.
        assert hasattr(result, "bandwidth_bps")
        assert hasattr(result, "heatmap_bandwidth")
        assert set(result.bandwidth_bps) == {"HC", "MDD", "ADHD"}
        assert result.bandwidth_HC == pytest.approx(result.bandwidth_bps["HC"])
        # Both disorder profiles carry lower precision than HC, so both should
        # show a bandwidth deficit relative to the baseline.
        assert result.bandwidth_bps["MDD"] < result.bandwidth_bps["HC"]
        assert result.bandwidth_bps["ADHD"] < result.bandwidth_bps["HC"]

    def test_clinical_mapper_requires_hc_baseline(self):
        """Omitting the reference profile must fail with a clear message."""
        from Theory.APGI_Information_Theoretic_Bandwidth import (
            BandwidthDerivation,
            ClinicalBandwidthMapper,
            ClinicalProfile,
        )

        mapper = ClinicalBandwidthMapper(BandwidthDerivation())
        profiles = {
            "MDD": ClinicalProfile(
                label="MDD",
                Pi_e=1.0,
                Pi_i=0.1,
                theta_baseline=0.1,
                alpha=0.5,
                sigma_theta=0.1,
                description="Test MDD profile",
            )
        }

        with pytest.raises(KeyError, match="HC"):
            mapper.compute_all(profiles=profiles)


class TestBandwidthConfig:
    """Test BandwidthConfig class."""

    def test_bandwidth_config_init(self):
        """Test BandwidthConfig initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthConfig

        config = BandwidthConfig()
        assert hasattr(config, "N_trials")
        assert hasattr(config, "N_stimulus_bins")

    def test_bandwidth_config_custom(self):
        """Test BandwidthConfig with custom parameters."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthConfig

        config = BandwidthConfig(N_trials=5000, N_stimulus_bins=50)

        assert config.N_trials == 5000
        assert config.N_stimulus_bins == 50


class TestBandwidthAnalyzer:
    """Test BandwidthAnalyzer class."""

    def test_bandwidth_analyzer_init(self):
        """Test BandwidthAnalyzer initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthAnalyzer, BandwidthConfig

        config = BandwidthConfig()
        analyzer = BandwidthAnalyzer(config)
        assert hasattr(analyzer, "run")
        assert analyzer.config == config

    def test_bandwidth_analyzer_run(self):
        """Test BandwidthAnalyzer run method."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthAnalyzer, BandwidthConfig

        config = BandwidthConfig(N_trials=100)  # Small number for testing
        analyzer = BandwidthAnalyzer(config)

        # Run the real pipeline. At N_trials=100 it completes in well under a
        # second, so mocking bought nothing — and the previous mocks patched
        # attributes named bw/pw/cb, which do not exist on this class
        # (they are bw_module/precision_module/clinical_module), so the test
        # raised AttributeError before reaching any assertion.
        result = analyzer.run()

        assert hasattr(result, "nominal")
        assert hasattr(result, "sweep")
        assert hasattr(result, "clinical")
        assert hasattr(result, "precision_gain")

        assert result.nominal.bandwidth_bits_per_second >= 0.0
        assert result.sweep["n_samples"] > 0
        assert "HC" in result.clinical.bandwidth_bps


class TestRunBandwidthAnalysis:
    """Test run_bandwidth_analysis function."""

    def test_run_bandwidth_analysis_default(self):
        """Test run_bandwidth_analysis with default config."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthConfig, run_bandwidth_analysis

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            # Create a simple config file
            config_data = {"n_trials": 10, "output_dir": str(temp_dir)}

            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Run analysis with BandwidthConfig object
            result = run_bandwidth_analysis(config=BandwidthConfig(N_trials=10))

            # Verify result structure
            assert hasattr(result, "nominal")
            assert hasattr(result, "sweep")
            assert hasattr(result, "clinical")

    def test_run_bandwidth_analysis_custom(self):
        """Test run_bandwidth_analysis with custom config."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthConfig, run_bandwidth_analysis

        custom_config = BandwidthConfig(N_trials=50, N_stimulus_bins=20)

        result = run_bandwidth_analysis(config=custom_config)

        # Verify result structure
        assert hasattr(result, "nominal")
        assert hasattr(result, "sweep")
        assert hasattr(result, "clinical")


class TestErrorHandling:
    """Test error handling in bandwidth module."""

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        model = BandwidthDerivation()

        # Test with invalid parameter ranges
        with pytest.raises((ValueError, TypeError)):
            model.compute(Pi_e=-1.0)  # Invalid negative precision

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"Pi_e": -1.0},
            {"Pi_e": 0.0},
            {"Pi_i": -0.5},
            {"Pi_i": 0.0},
            {"alpha": -2.0},
            {"alpha": 0.0},
            {"stimulus_rate_Hz": -5.0},
            {"sigma_theta": -0.1},
            {"Pi_e": float("nan")},
            {"Pi_e": float("inf")},
            {"theta_baseline": float("nan")},
        ],
    )
    def test_non_physical_parameters_rejected(self, kwargs):
        """Precisions and gains are positive by definition.

        Accepting a negative precision produced a plausible-looking bandwidth
        from physically meaningless input — silently wrong output, the worst
        failure mode for a validation framework.
        """
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        with pytest.raises(ValueError):
            BandwidthDerivation().compute(**kwargs)

    def test_valid_parameters_accepted(self):
        """The guard must not reject legitimate parameter sets."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        result = BandwidthDerivation().compute(Pi_e=1.0, Pi_i=0.5, alpha=2.0, sigma_theta=0.0)
        assert result is not None

    def test_mutual_info_falls_back_when_sklearn_raises(self):
        """Dependency skew must degrade to the exact NumPy path, not crash.

        Some scipy/sklearn combinations raise
        "matrix.sum() got an unexpected keyword argument 'initial'" inside
        mutual_info_score.
        """
        import numpy as np

        from Theory import APGI_Information_Theoretic_Bandwidth as mod

        labels = np.array([0, 1, 2, 3] * 25)

        def _boom(*args, **kwargs):
            raise TypeError("matrix.sum() got an unexpected keyword argument 'initial'")

        with patch.object(mod, "_skl_mi", _boom):
            mi = mod._mutual_info_bits(labels, labels)

        # Four equiprobable, perfectly correlated labels carry log2(4) = 2 bits.
        assert mi == pytest.approx(2.0, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
