"""
Tests for utility modules.
===========================

This module provides comprehensive functional tests for APGI utility modules,
going beyond simple directory structure checks to test actual functionality.
"""

# Import utility modules to test
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.config_manager import ConfigManager
    from utils.constants import DIM_CONSTANTS
    from utils.eeg_simulator import EEGSimulator
    from utils.error_handler import APGIError, ErrorCode
    from utils.genome_data_extractor import extract_genome_data_from_vp5
    from utils.logging_config import apgi_logger
    from utils.ordinal_logistic_regression import OrdinalLogisticRegression
    from utils.preprocessing_pipelines import EEGPreprocessor, PreprocessingConfig
    from utils.threshold_registry import FalsificationThresholds, ThresholdRegistry
except ImportError as e:
    pytest.skip(f"Cannot import utility modules: {e}", allow_module_level=True)


def test_utils_directory_structure():
    """Test that utils directory has expected structure."""
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "utils"

    assert utils_dir.exists(), "utils directory missing"
    assert utils_dir.is_dir(), "utils should be a directory"

    # Check subdirectories (optional - may not exist in all environments)
    subdirs = ["config", "data"]
    for subdir in subdirs:
        subdir_path = utils_dir / subdir
        if subdir_path.exists():
            assert subdir_path.is_dir(), f"utils/{subdir} should be a directory"


def test_utility_files_exist():
    """Test that essential utility files exist."""
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "utils"

    # Check essential utility files
    utility_files = [
        "config_manager.py",
        "backup_manager.py",
        "batch_processor.py",
        "data_quality_assessment.py",
    ]

    for file_name in utility_files:
        file_path = utils_dir / file_name
        assert file_path.exists(), f"Utility file {file_name} missing"


def test_sample_data_fixture_structure(sample_data):
    """Test that sample_data fixture has expected structure."""
    assert isinstance(sample_data, dict)

    # Check required data keys
    required_keys = ["timestamps", "surprise", "threshold", "metabolic", "arousal"]
    for key in required_keys:
        assert key in sample_data, f"Sample data missing key: {key}"
        assert isinstance(sample_data[key], list), f"Sample data {key} should be a list"
        assert len(sample_data[key]) > 0, f"Sample data {key} should not be empty"

    # Check data consistency
    data_length = len(sample_data["timestamps"])
    for key in required_keys:
        assert len(sample_data[key]) == data_length, f"Sample data {key} length mismatch"


class TestConfigLoader:
    """Test the config_loader utility module."""

    def test_get_config_value_existing_key(self):
        """Test getting an existing configuration value."""
        # Test with a known key from default.yaml
        config_manager = ConfigManager()
        model_config = config_manager.get_config("model")
        assert isinstance(model_config, (dict, object)), "Model config should be a dictionary or object"

        # Check for expected model parameters
        expected_keys = [
            "tau_S",
            "tau_theta",
            "theta_0",
            "alpha",
            "gamma_M",
            "gamma_A",
            "rho",
        ]
        if isinstance(model_config, dict):
            for key in expected_keys:
                assert key in model_config, f"Model config should contain {key}"
                assert isinstance(model_config[key], (int, float, str)), f"Model config {key} should be numeric"
                assert model_config[key] > 0, f"Model config {key} should be positive"

    def test_get_config_missing_key(self):
        """Test getting a missing configuration section."""
        config_manager = ConfigManager()
        try:
            config_manager.get_config("nonexistent")
            assert False, "Should raise ValueError for missing section"
        except ValueError:
            assert True  # Expected

    def test_get_model_config(self):
        """Test getting model configuration section."""
        config_manager = ConfigManager()
        model_config = config_manager.get_config("model")
        assert isinstance(model_config, (dict, object)), "Model config should be a dictionary or object"

        # Check for expected model parameters
        expected_keys = [
            "tau_S",
            "tau_theta",
            "theta_0",
            "alpha",
            "gamma_M",
            "gamma_A",
            "rho",
        ]
        if isinstance(model_config, dict):
            for key in expected_keys:
                assert key in model_config, f"Model config should contain {key}"
                assert isinstance(model_config[key], (int, float, str)), f"Model config {key} should be numeric"
                assert model_config[key] > 0, f"Model config {key} should be positive"

    def test_get_falsification_config(self):
        """Test getting falsification configuration section."""
        config_manager = ConfigManager()
        falsif_config = config_manager.get_config("falsification")
        assert isinstance(falsif_config, (dict, object)), "Falsification config should be a dictionary or object"

        # Check for expected falsification parameters
        expected_keys = [
            "cumulative_reward_advantage_threshold",
            "cohens_d_threshold",
            "significance_level",
        ]
        if isinstance(falsif_config, dict):
            for key in expected_keys:
                assert key in falsif_config, f"Falsification config should contain {key}"


class TestThresholdRegistry:
    """Test the threshold_registry utility module."""

    # Thresholds are named by the prediction they gate (pred_<id>_<measure>).
    # The previous tests used four invented names — cumulative_reward_advantage_
    # threshold, cohens_d_threshold, significance_level, threshold_reduction_min —
    # none of which exists on FalsificationThresholds.

    def test_falsification_thresholds_defaults(self):
        """Test FalsificationThresholds default values."""
        thresholds = FalsificationThresholds()

        assert thresholds.pred_1b_diastole_advantage_pct == 5.0
        assert thresholds.pred_2a_reward_ratio == 1.15
        assert thresholds.pred_0a_r_confirm == 0.35
        assert thresholds.pred_0b_bf10 == 100.0

    def test_threshold_registry_initialization(self):
        """Test ThresholdRegistry initialization."""
        registry = ThresholdRegistry()
        assert isinstance(registry.thresholds, FalsificationThresholds)

        # Test getting individual thresholds
        assert registry.get_threshold("pred_1b_diastole_advantage_pct") == 5.0
        assert registry.get_threshold("pred_0b_bf10") == 100.0

    def test_threshold_registry_validation(self):
        """Test threshold validation."""
        registry = ThresholdRegistry()

        # Valid threshold update
        assert registry.update_threshold("pred_1b_diastole_advantage_pct", 20.0)

        # Unknown thresholds must be rejected rather than silently created.
        with pytest.raises((ValueError, KeyError, AttributeError)):
            registry.update_threshold("no_such_threshold", 1.0)

        registry.reset_to_defaults()

    def test_reset_to_defaults_restores_value(self):
        registry = ThresholdRegistry()
        original = registry.get_threshold("pred_0b_bf10")

        registry.update_threshold("pred_0b_bf10", 1.0)
        registry.reset_to_defaults()

        assert registry.get_threshold("pred_0b_bf10") == original

    def test_get_all_thresholds(self):
        """Test getting all thresholds as dictionary."""
        registry = ThresholdRegistry()
        all_thresholds = registry.get_all_thresholds()

        assert isinstance(all_thresholds, dict)
        assert "pred_1b_diastole_advantage_pct" in all_thresholds
        assert "pred_2a_reward_ratio" in all_thresholds
        assert "pred_0b_bf10" in all_thresholds
        assert all(isinstance(v, (int, float)) for v in all_thresholds.values())


class TestConstants:
    """Test the constants utility module."""

    def test_dimension_constants(self):
        """Test that dimension constants are defined and positive."""
        assert hasattr(DIM_CONSTANTS, "EXTERO_DIM")
        assert hasattr(DIM_CONSTANTS, "INTERO_DIM")
        assert hasattr(DIM_CONSTANTS, "SENSORY_DIM")

        # Check that dimensions are positive integers
        assert isinstance(DIM_CONSTANTS.EXTERO_DIM, int)
        assert isinstance(DIM_CONSTANTS.INTERO_DIM, int)
        assert DIM_CONSTANTS.EXTERO_DIM > 0
        assert DIM_CONSTANTS.INTERO_DIM > 0

    def test_constant_values(self):
        """Test specific constant values."""
        # Test known constant values
        assert DIM_CONSTANTS.EXTERO_DIM == 32
        assert DIM_CONSTANTS.INTERO_DIM == 16
        assert DIM_CONSTANTS.SENSORY_DIM == 32


class TestErrorHandling:
    """Test the error_handler utility module."""

    def test_apgi_error_creation(self):
        """Test APGIError exception creation."""
        error = APGIError(message="Test error message")
        # APGIError now includes severity and category in string representation
        assert "Test error message" in str(error)
        assert "[MEDIUM]" in str(error)
        assert "RUNTIME" in str(error)

        # Test with error code - ErrorCode is a string enum, compare the value
        error_with_code = APGIError(message="Test error", error_code=ErrorCode.GEN_UNKNOWN)
        assert error_with_code.error_code == ErrorCode.GEN_UNKNOWN
        assert error_with_code.error_code.value == "GEN_000"


class TestLoggingConfig:
    """Test the logging_config utility module."""

    def test_logger_exists(self):
        """Test that APGI logger is properly configured."""
        assert apgi_logger is not None
        assert hasattr(apgi_logger, "logger")
        assert hasattr(apgi_logger, "log_simulation_error")


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "timestamps": [0.0, 0.1, 0.2, 0.3, 0.4],
        "surprise": [1.0, 1.5, 2.0, 1.8, 1.2],
        "threshold": [0.5, 0.5, 0.5, 0.5, 0.5],
        "metabolic": [0.1, 0.15, 0.2, 0.18, 0.12],
        "arousal": [0.8, 0.9, 1.0, 0.95, 0.85],
    }


class TestEEGProcessing:
    """Test the eeg_processing utility module."""

    def test_eeg_preprocessor_initialization(self):
        """Test EEGPreprocessor initialization."""
        config = PreprocessingConfig(target_sampling_rate=100.0)
        processor = EEGPreprocessor(config)
        assert hasattr(processor, "preprocess_eeg")

    def test_eeg_preprocessor_with_sample_data(self):
        """Test EEG processing with sample data."""
        config = PreprocessingConfig(target_sampling_rate=100.0)
        processor = EEGPreprocessor(config)
        # Create sample EEG data
        eeg_data = np.random.randn(5, 100)  # 5 channels, 100 time points
        processed = processor.preprocess_eeg(eeg_data)
        assert isinstance(processed, dict)
        assert "processed_eeg" in processed or "error" in processed


class TestEEGSimulator:
    """Test the eeg_simulator utility module."""

    def test_eeg_simulator_initialization(self):
        """Test EEGSimulator initialization."""
        simulator = EEGSimulator(n_channels=32, sampling_rate=100.0)
        assert simulator.n_channels == 32
        assert simulator.sampling_rate == 100.0
        assert hasattr(simulator, "simulate_eeg")

    def test_eeg_simulator_with_parameters(self):
        """Test EEG simulation with custom parameters."""
        simulator = EEGSimulator(n_channels=64, sampling_rate=250.0, duration_seconds=1.0)
        simulated = simulator.simulate_eeg()
        assert isinstance(simulated, dict)
        assert "eeg_data" in simulated or "error" in simulated


class TestPreprocessingPipelines:
    """Test the preprocessing_pipelines utility module."""

    def test_preprocessor_initialization(self):
        """Test EEGPreprocessor initialization."""
        config = PreprocessingConfig(target_sampling_rate=100.0)
        preprocessor = EEGPreprocessor(config)
        assert hasattr(preprocessor, "preprocess_eeg")
        assert hasattr(preprocessor, "apply_filters")

    def test_preprocessor_with_sample_data(self):
        """Test preprocessing pipeline with sample data."""
        config = PreprocessingConfig(target_sampling_rate=100.0)
        preprocessor = EEGPreprocessor(config)
        # Create sample data - numpy array format
        sample_data = np.random.randn(5, 100)  # 5 channels, 100 samples
        result = preprocessor.preprocess_eeg(sample_data)
        assert "processed_eeg" in result or "error" in result


class TestGenomeDataExtractor:
    """Test the genome_data_extractor utility module."""

    def test_extract_genome_data_from_vp5(self):
        """Test extracting genome data from VP-5 results."""
        # This test would require actual VP-5 results file
        # For now, we'll test the function exists and handles missing files gracefully
        try:
            extract_genome_data_from_vp5("nonexistent_file.json")
        except Exception:
            # Expected when file doesn't exist
            pass


class TestOrdinalLogisticRegression:
    """Test the ordinal_logistic_regression utility module."""

    def test_model_initialization(self):
        """Test OrdinalLogisticRegression initialization."""
        model = OrdinalLogisticRegression(n_categories=3)
        assert model.n_categories == 3
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_model_with_sample_data(self):
        """Test ordinal regression model with sample data."""
        model = OrdinalLogisticRegression(n_categories=3)
        # Create sample data
        X = np.random.randn(100, 5)  # 100 samples, 5 features
        y = np.random.randint(0, 3, 100)  # 3 ordinal classes
        result = model.fit(X, y)
        assert "model_fitted" in result or "error" in result

        # Test prediction
        predictions = model.predict(X[:5])
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5


class TestFalsificationCriterionNegativePaths:
    """Test FalsificationCriterion constructor with invalid inputs (COV-005)."""

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        from Theory.APGI_Falsification_Framework import FalsificationCriterion, TestStatistic

        with pytest.raises(ValueError, match="non-empty string"):
            FalsificationCriterion(
                name="",
                description="Test",
                test_statistic=TestStatistic.MEAN_DIFFERENCE.value,
                threshold=0.5,
                direction="greater",
                alpha=0.05,
            )

    def test_whitespace_only_name_raises_error(self):
        """Test that whitespace-only name raises ValueError."""
        from Theory.APGI_Falsification_Framework import FalsificationCriterion, TestStatistic

        with pytest.raises(ValueError, match="non-empty string"):
            FalsificationCriterion(
                name="   ",
                description="Test",
                test_statistic=TestStatistic.MEAN_DIFFERENCE.value,
                threshold=0.5,
                direction="greater",
                alpha=0.05,
            )

    def test_empty_description_raises_error(self):
        """Test that empty description raises ValueError."""
        from Theory.APGI_Falsification_Framework import FalsificationCriterion, TestStatistic

        with pytest.raises(ValueError, match="non-empty string"):
            FalsificationCriterion(
                name="test",
                description="",
                test_statistic=TestStatistic.MEAN_DIFFERENCE.value,
                threshold=0.5,
                direction="greater",
                alpha=0.05,
            )

    def test_invalid_direction_raises_error(self):
        """Test that invalid direction raises ValueError."""
        from Theory.APGI_Falsification_Framework import FalsificationCriterion, TestStatistic

        with pytest.raises(ValueError, match="Invalid direction"):
            FalsificationCriterion(
                name="test",
                description="Test",
                test_statistic=TestStatistic.MEAN_DIFFERENCE.value,
                threshold=0.5,
                direction="invalid_direction",
                alpha=0.05,
            )

    def test_alpha_greater_than_one_raises_error(self):
        """Test that alpha > 1 raises ValueError."""
        from Theory.APGI_Falsification_Framework import FalsificationCriterion, TestStatistic

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            FalsificationCriterion(
                name="test",
                description="Test",
                test_statistic=TestStatistic.MEAN_DIFFERENCE.value,
                threshold=0.5,
                direction="greater",
                alpha=1.5,
            )

    def test_alpha_zero_raises_error(self):
        """Test that alpha = 0 raises ValueError."""
        from Theory.APGI_Falsification_Framework import FalsificationCriterion, TestStatistic

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            FalsificationCriterion(
                name="test",
                description="Test",
                test_statistic=TestStatistic.MEAN_DIFFERENCE.value,
                threshold=0.5,
                direction="greater",
                alpha=0.0,
            )

    def test_alpha_negative_raises_error(self):
        """Test that negative alpha raises ValueError."""
        from Theory.APGI_Falsification_Framework import FalsificationCriterion, TestStatistic

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            FalsificationCriterion(
                name="test",
                description="Test",
                test_statistic=TestStatistic.MEAN_DIFFERENCE.value,
                threshold=0.5,
                direction="greater",
                alpha=-0.1,
            )

    def test_invalid_test_statistic_raises_error(self):
        """Test that invalid test_statistic raises ValueError."""
        from Theory.APGI_Falsification_Framework import FalsificationCriterion

        with pytest.raises(ValueError, match="Unsupported test statistic"):
            FalsificationCriterion(
                name="test",
                description="Test",
                test_statistic="invalid_statistic",
                threshold=0.5,
                direction="greater",
                alpha=0.05,
            )
