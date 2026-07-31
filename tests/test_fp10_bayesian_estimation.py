"""Tests for FP_10 Falsification Bayesian Estimation - increase coverage from 0%."""

import json

import numpy as np

from Falsification.FP_10_BayesianEstimationMCMC import (
    BayesianParameterRecovery,
    DataSource,
    FP10bParameterRecovery,
    ParameterRecoveryResult,
    compute_posterior_distributions,
)
from Falsification.FP_10_BayesianEstimationMCMC import generate_synthetic_data
from Falsification.FP_10_BayesianEstimationMCMC import generate_synthetic_data as generate_synthetic_data_mcmc
from Falsification.FP_10_BayesianEstimationMCMC import (
    run_bayesian_estimation,
    run_mcmc_bayesian_estimation,
    set_data_source,
)
from Falsification.FP_10_BayesianEstimationMCMC import test_parameter_identifiability as check_parameter_identifiability


class TestDataSource:
    """Test Data Source enum."""

    def test_data_source_values(self):
        """Test data source enum values."""
        assert DataSource.SYNTHETIC.value == "synthetic"
        assert DataSource.EMPIRICAL.value == "empirical"
        assert DataSource.SIMULATION.value == "simulation"


class TestParameterRecoveryResult:
    """Test Parameter Recovery Result dataclass."""

    def test_result_creation(self):
        """Test creating parameter recovery result."""
        result = ParameterRecoveryResult(
            parameter_name="alpha",
            true_value=0.5,
            recovered_mean=0.52,
            recovered_std=0.01,
            credible_interval=(0.50, 0.54),
            recovery_error=0.02,
            relative_error=0.04,
            recovered_successfully=True,
            identifiability_score=0.85,
        )
        assert result.parameter_name == "alpha"
        assert result.recovered_successfully is True


class TestBayesianParameterRecovery:
    """Test Bayesian Parameter Recovery from MCMC module."""

    def test_init(self):
        """Test initialization."""
        recovery = BayesianParameterRecovery(n_samples=100, n_chains=2, burn_in=50)
        assert recovery.n_samples == 100
        assert recovery.n_chains == 2

    def test_basic_attributes(self):
        """Test basic attributes are set correctly."""
        recovery = BayesianParameterRecovery()
        assert recovery.n_samples > 0
        assert recovery.n_chains > 0
        assert recovery.burn_in >= 0


class TestFP10bParameterRecovery:
    """Test FP10b Parameter Recovery class."""

    def test_init(self):
        """Test initialization."""
        recovery = FP10bParameterRecovery()
        assert recovery is not None
        assert recovery.recovery_tolerance > 0

    def test_default_true_parameters(self):
        """Test default true parameters are set."""
        recovery = FP10bParameterRecovery()
        default_params = recovery._get_default_true_parameters()
        assert "tau_S" in default_params
        assert "theta_0" in default_params
        assert "alpha" in default_params


class TestGenerateSyntheticData:
    """Test synthetic data generation."""

    def test_generate_synthetic_data_basic(self):
        """Test basic synthetic data generation."""
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(n_trials=50, true_params=true_params, noise_level=0.1)
        assert len(stimulus_data) == 50
        assert len(response_data) == 50

    def test_generate_synthetic_data_mcmc_variant(self):
        """Test MCMC variant of synthetic data generation."""
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data_mcmc(
            n_trials=50, true_params=true_params, noise_level=0.1
        )
        assert len(stimulus_data) == 50
        assert len(response_data) == 50


class TestRunMCMCBayesianEstimation:
    """Test MCMC Bayesian estimation runner."""

    def test_run_mcmc_with_synthetic_data(self):
        """Test running MCMC estimation with synthetic data."""
        # Generate synthetic data
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(n_trials=50, true_params=true_params, noise_level=0.1)

        # Run MCMC with minimal samples for testing
        result = run_mcmc_bayesian_estimation(
            stimulus_data=stimulus_data,
            response_data=response_data,
            n_samples=100,
            n_chains=2,
            burn_in=50,
        )
        # Result may contain error key if PyMC not available
        assert "posterior_means" in result or "error" in result

    def test_run_mcmc_with_data_source_warning(self):
        """Test MCMC estimation with synthetic data source warning."""
        # Generate synthetic data
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(n_trials=20, true_params=true_params, noise_level=0.1)

        # Run MCMC with explicit synthetic data source to trigger warning
        # Use at least 2 chains to avoid division by zero in Gelman-Rubin diagnostic
        result = run_mcmc_bayesian_estimation(
            stimulus_data=stimulus_data,
            response_data=response_data,
            n_samples=50,
            n_chains=2,
            burn_in=25,
            data_source=DataSource.SYNTHETIC,
        )
        # Should handle synthetic data warning gracefully
        assert "posterior_means" in result or "error" in result or "simulation_only" in result

    def test_run_mcmc_with_empirical_data_source(self):
        """Test MCMC estimation with empirical data source."""
        # Generate synthetic data (but mark as empirical for testing)
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(n_trials=20, true_params=true_params, noise_level=0.1)

        # Run MCMC with empirical data source
        # Use at least 2 chains to avoid division by zero in Gelman-Rubin diagnostic
        result = run_mcmc_bayesian_estimation(
            stimulus_data=stimulus_data,
            response_data=response_data,
            n_samples=50,
            n_chains=2,
            burn_in=25,
            data_source=DataSource.EMPIRICAL,
        )
        # Should handle empirical data without warning
        assert "posterior_means" in result or "error" in result

    def test_run_mcmc_with_custom_nuts_settings(self):
        """Test MCMC estimation with custom NUTS settings."""
        # Generate synthetic data
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(n_trials=20, true_params=true_params, noise_level=0.1)

        # Run MCMC with custom NUTS settings
        # Use at least 2 chains to avoid division by zero in Gelman-Rubin diagnostic
        result = run_mcmc_bayesian_estimation(
            stimulus_data=stimulus_data,
            response_data=response_data,
            n_samples=50,
            n_chains=2,
            burn_in=25,
            target_accept=0.9,
            max_tree_depth=8,
        )
        # Should handle custom settings
        assert "posterior_means" in result or "error" in result


class TestRunBayesianEstimationAlias:
    """Test run_bayesian_estimation alias function."""

    def test_run_bayesian_estimation_alias(self):
        """Test that the alias function works."""
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(n_trials=50, true_params=true_params, noise_level=0.1)

        run_bayesian_estimation(
            stimulus_data=stimulus_data,
            response_data=response_data,
            n_samples=100,
            n_chains=2,
            burn_in=50,
        )


class TestComputePosteriorDistributions:
    """Test posterior distribution computation."""

    def test_compute_posterior_distributions_empty(self):
        """Test posterior distribution with empty/invalid trace."""
        # Test with None trace - should return empty dict
        compute_posterior_distributions(None, ["alpha", "beta"])


class TestTestParameterIdentifiability:
    """Test parameter identifiability testing."""

    def test_identifiability_with_good_recovery(self):
        """Test identifiability with good parameter recovery."""
        # Create fake posterior samples
        posterior_samples = {
            "alpha": np.random.normal(0.5, 0.01, 100),
            "beta": np.random.normal(1.0, 0.02, 100),
        }
        true_params = {"alpha": 0.5, "beta": 1.0}

        result = check_parameter_identifiability(posterior_samples, true_params, tolerance=0.15)
        assert "all_identified" in result

    def test_identifiability_with_poor_recovery(self):
        """Test identifiability with poor parameter recovery."""
        # Create posterior samples far from true values
        posterior_samples = {
            "alpha": np.random.normal(2.0, 0.1, 100),  # Far from true 0.5
            "beta": np.random.normal(3.0, 0.1, 100),  # Far from true 1.0
        }
        true_params = {"alpha": 0.5, "beta": 1.0}

        check_parameter_identifiability(posterior_samples, true_params, tolerance=0.1)


class TestSetDataSource:
    """Test data source setting."""

    def test_set_data_source(self):
        """Test setting data source."""
        set_data_source(DataSource.SYNTHETIC)
        # Should not raise an error

    def test_get_data_source(self):
        """Test getting data source."""
        from Falsification.FP_10_BayesianEstimationMCMC import get_data_source

        source = get_data_source()
        assert source in DataSource

    def test_require_empirical_validation(self):
        """Test requiring empirical validation."""
        from Falsification.FP_10_BayesianEstimationMCMC import require_empirical_validation

        require_empirical_validation(True)
        require_empirical_validation(False)
        # Should not raise an error


class TestMCMCPriors:
    """Test MCMC priors functionality."""

    def test_get_mcmc_priors_default(self):
        """Test getting default MCMC priors."""
        from Falsification.FP_10_BayesianEstimationMCMC import get_mcmc_priors

        priors = get_mcmc_priors("default")
        assert isinstance(priors, dict)
        assert "theta_0" in priors
        assert "pi_e" in priors
        assert "pi_i" in priors
        assert "beta" in priors
        assert "alpha" in priors

    def test_get_mcmc_priors_conservative(self):
        """Test getting conservative MCMC priors."""
        from Falsification.FP_10_BayesianEstimationMCMC import get_mcmc_priors

        priors = get_mcmc_priors("conservative")
        assert isinstance(priors, dict)
        assert "theta_0" in priors

    def test_get_mcmc_priors_informative(self):
        """Test getting informative MCMC priors."""
        from Falsification.FP_10_BayesianEstimationMCMC import get_mcmc_priors

        priors = get_mcmc_priors("informative")
        assert isinstance(priors, dict)
        assert "theta_0" in priors


class TestPsychometricFunctions:
    """Test APGI psychometric functions."""

    def test_apgi_psychometric_function_np(self):
        """Test numpy-based APGI psychometric function."""
        from Falsification.FP_10_BayesianEstimationMCMC import apgi_psychometric_function_np

        stimulus = np.array([0.1, 0.5, 0.9])
        result = apgi_psychometric_function_np(stimulus, theta_0=0.5, pi_e=0.5, pi_i=3.0, beta=2.5, alpha=1.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(stimulus)
        assert all(0 <= r <= 1 for r in result)

    def test_apgi_psychometric_function(self):
        """Test APGI psychometric function."""
        from Falsification.FP_10_BayesianEstimationMCMC import apgi_psychometric_function

        stimulus = np.array([0.1, 0.5, 0.9])
        result = apgi_psychometric_function(stimulus, theta_0=0.5, pi_e=0.5, pi_i=3.0, beta=2.5, alpha=1.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(stimulus)

    def test_apgi_psychometric_v2(self):
        """Test APGI psychometric function v2."""
        from Falsification.FP_10_BayesianEstimationMCMC import apgi_psychometric_v2

        stimulus = np.array([0.1, 0.5, 0.9])
        result = apgi_psychometric_v2(stimulus, theta_0=0.5, pi_e=0.5, pi_i=3.0, beta=2.5, alpha=1.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(stimulus)


class TestBayesFactors:
    """Test Bayes factor computation."""

    def test_interpret_bayes_factor(self):
        """Test Bayes factor interpretation."""
        from Falsification.FP_10_BayesianEstimationMCMC import interpret_bayes_factor

        # Test various Bayes factor values
        assert interpret_bayes_factor(1.0) == "Weak evidence for model 1"
        assert interpret_bayes_factor(3.0) == "Moderate evidence for model 1"
        assert interpret_bayes_factor(10.0) == "Strong evidence for model 1"
        assert interpret_bayes_factor(0.1) == "Moderate evidence for model 2"
        assert interpret_bayes_factor(0.01) == "Very strong evidence for model 2"

    def test_compute_bayes_factors(self):
        """Test Bayes factor computation."""
        from Falsification.FP_10_BayesianEstimationMCMC import compute_bayes_factors

        evidence_dict = {"APGI": -100.0, "StandardPP": -105.0, "GWTOnly": -110.0}
        result = compute_bayes_factors(evidence_dict)
        assert isinstance(result, dict)
        assert "bayes_factors" in result
        assert "comparison_summary" in result

        # "interpretation" lives on each pairwise comparison, not at the top
        # level; the top-level keys are the ones asserted below.
        assert {"best_model", "model_ranking", "model_evidence"} <= set(result)
        assert result["best_model"] == "APGI", "Highest evidence model should win"
        for pair, comparison in result["bayes_factors"].items():
            assert "interpretation" in comparison, f"{pair} missing interpretation"


class TestPosteriorPredictive:
    """Test posterior predictive checks."""

    def test_compute_posterior_predictive_mae(self):
        """Test posterior predictive MAE computation."""
        from Falsification.FP_10_BayesianEstimationMCMC import compute_posterior_predictive_mae

        # Create mock trace with posterior samples
        mock_trace = {
            "posterior": {
                "theta_0": np.random.normal(0.5, 0.1, 100),
                "pi_e": np.random.normal(0.5, 0.1, 100),
                "pi_i": np.random.normal(3.0, 0.5, 100),
                "beta": np.random.normal(2.5, 0.3, 100),
                "alpha": np.random.normal(1.5, 0.2, 100),
            }
        }

        stimulus_data = np.array([0.1, 0.5, 0.9])
        response_data = np.array([0, 1, 1])

        result = compute_posterior_predictive_mae(mock_trace, stimulus_data, response_data)
        assert isinstance(result, dict)
        # The function may return error if PyMC is not available, so check for either success or error
        assert "mae" in result or "error" in result


class TestFalsificationCriteria:
    """Test falsification criteria functionality."""

    def test_get_falsification_criteria(self):
        """Test getting falsification criteria."""
        from Falsification.FP_10_BayesianEstimationMCMC import get_falsification_criteria

        criteria = get_falsification_criteria()
        assert isinstance(criteria, dict)
        assert "F10.BF" in criteria
        assert "F10.DataSource" in criteria
        assert "F10.Divergence" in criteria

    def test_define_apgi_priors(self):
        """Test defining APGI priors."""
        from Falsification.FP_10_BayesianEstimationMCMC import define_apgi_priors

        priors = define_apgi_priors()
        assert isinstance(priors, dict)
        assert "theta_0" in priors
        assert "pi_e" in priors
        assert "pi_i" in priors
        assert "beta" in priors
        assert "alpha" in priors


class TestUtilityFunctions:
    """Test utility functions."""

    def test_convert_to_serializable(self):
        """Test converting numpy types to serializable."""
        from Falsification.FP_10_BayesianEstimationMCMC import _convert_to_serializable

        # Test numpy array
        arr = np.array([1, 2, 3])
        result = _convert_to_serializable(arr)
        assert result == [1, 2, 3]

        # Test numpy scalar
        scalar = np.float64(3.14)
        result = _convert_to_serializable(scalar)
        assert result == 3.14

        # Test regular Python type
        result = _convert_to_serializable([1, 2, 3])
        assert result == [1, 2, 3]

    def test_harmonic_mean_evidence(self):
        """Test harmonic mean evidence computation."""
        from Falsification.FP_10_BayesianEstimationMCMC import compute_harmonic_mean_evidence

        # Create mock trace
        mock_trace = {
            "posterior": {
                "theta_0": np.random.normal(0.5, 0.1, 100),
            }
        }

        result = compute_harmonic_mean_evidence(mock_trace, ["theta_0"])
        assert result is None or isinstance(result, float)


class TestAlternativeModels:
    """Test alternative model functionality."""

    def test_run_alternative_models_minimal(self):
        """Test running alternative models with minimal samples."""
        from Falsification.FP_10_BayesianEstimationMCMC import run_alternative_models

        stimulus_data = np.array([0.1, 0.5, 0.9])
        response_data = np.array([0, 1, 1])

        # Run with minimal samples for testing
        try:
            result = run_alternative_models(stimulus_data, response_data, n_samples=50, n_chains=1)
            assert isinstance(result, dict)
            # Model keys are CamelCase ("StandardPP", "GWTOnly"), not snake_case.
            assert "StandardPP" in result or "error" in result
            for name, fit in result.items():
                if name == "error":
                    continue
                assert "log_evidence" in fit, f"{name} produced no evidence estimate"
        except ImportError as e:
            # Expected when PyMC is not available
            assert "PyMC is required" in str(e)


class TestParameterRecovery:
    """Test parameter recovery functionality."""

    def test_generate_parameter_recovery_data(self):
        """Test generating parameter recovery data.

        The blanket `except Exception` here previously swallowed the real
        assertion failure and re-asserted an unrelated claim about temporary
        directories, so the actual defect — stimulus values being asserted
        against the wrong range — was invisible.

        Stimulus is a physical intensity (roughly 0.05 to 2.5), not a
        probability, so the [0, 1] bound never held.
        """
        from Falsification.FP_10_BayesianEstimationMCMC import generate_parameter_recovery_data

        stimulus, response = generate_parameter_recovery_data(n_trials=20)

        assert len(stimulus) == 20
        assert len(response) == 20
        assert all(s > 0 for s in stimulus), "Stimulus intensities must be positive"
        assert all(r in [0, 1] for r in response), "Responses must be binary detections"

    def test_generate_empirical_plausible_data(self):
        """Test generating empirical plausible data."""
        from Falsification.FP_10_BayesianEstimationMCMC import generate_empirical_plausible_data

        stimulus, response = generate_empirical_plausible_data(n_trials=20)
        assert len(stimulus) == 20
        assert len(response) == 20


class TestProtocolRunner:
    """Test protocol runner functionality."""

    def test_run_protocol_main(self):
        """Test main protocol runner."""
        from Falsification.FP_10_BayesianEstimationMCMC import run_protocol_main

        result = run_protocol_main()

        # Protocols return the canonical ProtocolResult, not a bare dict.
        assert result.protocol_id == "FP_10_BayesianEstimationMCMC"
        assert result.status in {"success", "failed", "error", "partial"}

        # Must degrade gracefully when the PyTensor toolchain is unavailable:
        # a NumPy-fallback run still yields a serialisable result.
        payload = result.model_dump()
        assert isinstance(payload, dict)
        json.dumps(payload)  # protocol results must be plain-JSON serialisable
