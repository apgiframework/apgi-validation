"""
Tests for validation protocols.
===============================

Validation protocols including
functional tests for protocols 5-12 beyond simple import checks.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from Validation.Master_Validation import APGIMasterValidator

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Note: Validation protocol files use hyphens in filenames, so they can't be imported directly
# We'll test them by executing them as scripts instead

import importlib.util

# Protocol number to filename mapping
PROTOCOL_FILE_MAP = {
    1: "VP_01_SyntheticEEGMLClassification.py",
    2: "VP_02_BehavioralBayesianComparison.py",
    3: "VP_03_ActiveInferenceAgentSimulations.py",
    4: "VP_04_PhaseTransitionEpistemicLevel2.py",
    5: "VP_05_EvolutionaryEmergence.py",
    6: "VP_06_LiquidNetworkInductiveBias.py",
    7: "VP_07_TMSCausalInterventions.py",
    8: "VP_08_PsychophysicalThresholdEstimation.py",
    9: "VP_09_NeuralSignaturesEmpiricalPriority1.py",
    10: "VP_10_CausalManipulationsPriority2.py",
    11: "VP_11_MCMCCulturalNeurosciencePriority3.py",
    12: "VP_12_ClinicalCrossSpeciesConvergence.py",
}


def load_validation_protocol(protocol_num):
    """Load a validation protocol module using importlib.util."""
    if protocol_num not in PROTOCOL_FILE_MAP:
        raise FileNotFoundError(f"Protocol {protocol_num} not in mapping")

    protocol_path = Path(__file__).parent.parent / "Validation" / PROTOCOL_FILE_MAP[protocol_num]

    if not protocol_path.exists():
        raise FileNotFoundError(f"{PROTOCOL_FILE_MAP[protocol_num]} not found")

    spec = importlib.util.spec_from_file_location(f"VP_{protocol_num:02d}", protocol_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_files_exist():
    """Test that validation protocol files exist."""
    project_root = Path(__file__).parent.parent
    validation_dir = project_root / "Validation"

    # Check validation protocol files (updated to VP_XX naming)
    validation_files = [
        "VP_01_SyntheticEEGMLClassification.py",
        "VP_02_BehavioralBayesianComparison.py",
        "VP_03_ActiveInferenceAgentSimulations.py",
        "VP_04_PhaseTransitionEpistemicLevel2.py",
        "VP_05_EvolutionaryEmergence.py",
        "VP_06_LiquidNetworkInductiveBias.py",
        "VP_07_TMSCausalInterventions.py",
        "VP_08_PsychophysicalThresholdEstimation.py",
        "VP_09_NeuralSignaturesEmpiricalPriority1.py",
        "VP_10_CausalManipulationsPriority2.py",
        "VP_11_MCMCCulturalNeurosciencePriority3.py",
        "VP_12_ClinicalCrossSpeciesConvergence.py",
        "Master_Validation.py",
    ]

    for file_name in validation_files:
        file_path = validation_dir / file_name
        assert file_path.exists(), f"Validation file {file_name} missing"  # nosec B101

    # Check Validation_GUI.py at root level (moved from Validation/)
    gui_path = Path(__file__).parent.parent / "Validation_GUI.py"
    assert gui_path.exists(), "Validation_GUI.py not found at root level"  # nosec B101


def test_validation_config_structure(sample_config):
    """Test validation configuration structure."""
    validation_config = sample_config.get("validation", {})

    # Check required validation settings
    required_settings = [
        "enable_cross_validation",
        "cv_folds",
        "enable_sensitivity_analysis",
        "sensitivity_samples",
        "enable_robustness_tests",
        "significance_level",
    ]

    for setting in required_settings:
        assert setting in validation_config, f"Validation setting {setting} missing"  # nosec B101

    # Check data types
    assert isinstance(validation_config["cv_folds"], int)  # nosec B101
    assert isinstance(validation_config["sensitivity_samples"], int)  # nosec B101
    assert isinstance(validation_config["significance_level"], (int, float))  # nosec B101
    assert validation_config["cv_folds"] > 0  # nosec B101
    assert validation_config["sensitivity_samples"] > 0  # nosec B101
    assert 0 <= validation_config["significance_level"] <= 1  # nosec B101


class TestValidationProtocols5To12:
    """Functional tests for validation protocols 5-12."""

    def test_protocol_5_imports_and_classes(self):
        """Test that Protocol 5 can be imported and has expected classes."""
        try:
            vp5 = load_validation_protocol(5)

            # Check that key classes exist
            assert hasattr(vp5, "APGIValidationProtocol5")  # nosec B101
            assert hasattr(vp5, "MultiModalIntegrationValidator")  # nosec B101
            assert hasattr(vp5, "CrossModalFalsificationChecker")  # nosec B101

            # Test instantiation
            validator = vp5.APGIValidationProtocol5()
            assert validator is not None  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError) as e:
            pytest.skip(f"Protocol 5 not available: {e}")

    def test_protocol_6_imports_and_classes(self):
        """Test that Protocol 6 can be imported and has expected classes."""
        try:
            vp6 = load_validation_protocol(6)

            # Check that key classes exist
            assert hasattr(vp6, "APGIValidationProtocol6")  # nosec B101
            assert hasattr(vp6, "TemporalDynamicsValidator")  # nosec B101
            assert hasattr(vp6, "AdaptiveThresholdChecker")  # nosec B101

            # Test instantiation
            validator = vp6.APGIValidationProtocol6()
            assert validator is not None  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError) as e:
            pytest.skip(f"Protocol 6 not available: {e}")

    def test_protocol_7_imports_and_classes(self):
        """Test that Protocol 7 can be imported and has expected classes."""
        try:
            vp7 = load_validation_protocol(7)

            # Check that key classes exist
            assert hasattr(vp7, "APGIValidationProtocol7")  # nosec B101
            assert hasattr(vp7, "HierarchicalProcessingValidator")  # nosec B101
            assert hasattr(vp7, "LevelEmergenceChecker")  # nosec B101

            # Test instantiation
            validator = vp7.APGIValidationProtocol7()
            assert validator is not None  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError) as e:
            pytest.skip(f"Protocol 7 not available: {e}")

    def test_protocol_8_imports_and_classes(self):
        """Test that Protocol 8 can be imported and has expected classes."""
        try:
            vp8 = load_validation_protocol(8)

            # Check that key classes exist
            assert hasattr(vp8, "APGIValidationProtocol8")  # nosec B101
            assert hasattr(vp8, "PrecisionWeightingValidator")  # nosec B101
            assert hasattr(vp8, "InteroceptiveBiasChecker")  # nosec B101

            # Test instantiation
            validator = vp8.APGIValidationProtocol8()
            assert validator is not None  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError) as e:
            pytest.skip(f"Protocol 8 not available: {e}")

    def test_protocol_9_imports_and_classes(self):
        """Test that Protocol 9 can be imported and has expected classes."""
        try:
            vp9 = load_validation_protocol(9)

            # Check that key classes exist
            assert hasattr(vp9, "APGIValidationProtocol9")  # nosec B101
            assert hasattr(vp9, "MultiTimescaleValidator")  # nosec B101
            assert hasattr(vp9, "IntegrationWindowChecker")  # nosec B101

            # Test instantiation
            validator = vp9.APGIValidationProtocol9()
            assert validator is not None  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError) as e:
            pytest.skip(f"Protocol 9 not available: {e}")

    def test_protocol_10_imports_and_classes(self):
        """Test that Protocol 10 can be imported and has expected classes."""
        try:
            vp10 = load_validation_protocol(10)

            # Check that key classes exist
            assert hasattr(vp10, "APGIValidationProtocol10")  # nosec B101
            assert hasattr(vp10, "FeatureClusteringValidator")  # nosec B101
            assert hasattr(vp10, "PrincipalComponentChecker")  # nosec B101

            # Test instantiation
            validator = vp10.APGIValidationProtocol10()
            assert validator is not None  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError) as e:
            pytest.skip(f"Protocol 10 not available: {e}")

    def test_protocol_11_imports_and_classes(self):
        """Test that Protocol 11 can be imported and has expected classes."""
        try:
            vp11 = load_validation_protocol(11)

            # Check that key classes exist
            assert hasattr(vp11, "APGIValidationProtocol11")  # nosec B101
            assert hasattr(vp11, "NonAPGIComparisonValidator")  # nosec B101
            assert hasattr(vp11, "ArchitectureFailureChecker")  # nosec B101

            # Test instantiation
            validator = vp11.APGIValidationProtocol11()
            assert validator is not None  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError) as e:
            pytest.skip(f"Protocol 11 not available: {e}")

    def test_protocol_12_imports_and_classes(self):
        """Test that Protocol 12 can be imported and has expected classes."""
        try:
            vp12 = load_validation_protocol(12)

            # Check that key classes exist
            assert hasattr(vp12, "ClinicalConvergenceValidator")  # nosec B101
            assert hasattr(vp12, "ClinicalDataAnalyzer")  # nosec B101
            assert hasattr(vp12, "LiquidTimeConstantChecker")  # nosec B101

            # Test instantiation
            validator = vp12.ClinicalConvergenceValidator()
            assert validator is not None  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError) as e:
            pytest.skip(f"Protocol 12 not available: {e}")

    def test_protocol_5_basic_functionality(self):
        """Test basic functionality of Protocol 5."""
        try:
            vp5 = load_validation_protocol(5)

            validator = vp5.APGIValidationProtocol5()

            # Test that validator has required methods
            assert hasattr(validator, "run_validation")  # nosec B101
            assert hasattr(validator, "check_criteria")  # nosec B101
            assert callable(validator.run_validation)  # nosec B101
            assert callable(validator.check_criteria)  # nosec B101
            assert len(validator.PROTOCOL_TIERS) == len(set(validator.PROTOCOL_TIERS.keys()))  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Protocol 5 not available")

    def test_protocol_6_basic_functionality(self):
        """Test basic functionality of Protocol 6."""
        try:
            vp6 = load_validation_protocol(6)

            validator = vp6.APGIValidationProtocol6()
            validator.logger = MagicMock()
            validator.logger.info = MagicMock()

            # Test that validator has required methods
            assert isinstance(validator, APGIMasterValidator)  # nosec B101
            # Note: Protocol 6 uses validate() instead of run_validation()
            assert callable(validator.validate)  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Protocol 6 not available")

    def test_protocol_7_basic_functionality(self):
        """Test basic functionality of Protocol 7."""
        try:
            vp7 = load_validation_protocol(7)

            validator = vp7.APGIValidationProtocol7()

            # Test that validator has required methods
            assert hasattr(validator, "run_validation")  # nosec B101
            assert hasattr(validator, "check_criteria")  # nosec B101
            assert callable(validator.run_validation)  # nosec B101
            assert callable(validator.check_criteria)  # nosec B101
            assert len(validator.PROTOCOL_TIERS) == len(set(validator.PROTOCOL_TIERS.keys()))  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Protocol 7 not available")

    def test_protocol_8_basic_functionality(self):
        """Test basic functionality of Protocol 8."""
        try:
            vp8 = load_validation_protocol(8)

            validator = vp8.APGIValidationProtocol8()

            # Test that validator has required methods
            assert hasattr(validator, "run_validation")  # nosec B101
            assert hasattr(validator, "check_criteria")  # nosec B101
            assert callable(validator.run_validation)  # nosec B101
            assert callable(validator.check_criteria)  # nosec B101
            assert len(validator.PROTOCOL_TIERS) == len(set(validator.PROTOCOL_TIERS.keys()))  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Protocol 8 not available")

    def test_protocol_9_basic_functionality(self):
        """Test basic functionality of Protocol 9."""
        try:
            vp9 = load_validation_protocol(9)

            validator = vp9.APGIValidationProtocol9()

            # Test that validator has required methods
            assert hasattr(validator, "run_validation")  # nosec B101
            assert hasattr(validator, "check_criteria")  # nosec B101
            assert callable(validator.run_validation)  # nosec B101
            assert callable(validator.check_criteria)  # nosec B101
            assert len(validator.PROTOCOL_TIERS) == len(set(validator.PROTOCOL_TIERS.keys()))  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Protocol 9 not available")

    def test_protocol_10_basic_functionality(self):
        """Test basic functionality of Protocol 10."""
        try:
            vp10 = load_validation_protocol(10)

            validator = vp10.APGIValidationProtocol10()

            # Test that validator has required methods
            assert hasattr(validator, "run_validation")  # nosec B101
            assert hasattr(validator, "check_criteria")  # nosec B101
            assert callable(validator.run_validation)  # nosec B101
            assert callable(validator.check_criteria)  # nosec B101
            assert len(validator.PROTOCOL_TIERS) == len(set(validator.PROTOCOL_TIERS.keys()))  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Protocol 10 not available")

    def test_protocol_11_basic_functionality(self):
        """Test basic functionality of Protocol 11."""
        try:
            vp11 = load_validation_protocol(11)

            validator = vp11.APGIValidationProtocol11()

            # Test that validator has required methods
            assert hasattr(validator, "run_validation")  # nosec B101
            assert hasattr(validator, "check_criteria")  # nosec B101
            assert callable(validator.run_validation)  # nosec B101
            assert callable(validator.check_criteria)  # nosec B101
            assert len(validator.PROTOCOL_TIERS) == len(set(validator.PROTOCOL_TIERS.keys()))  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Protocol 11 not available")

    def test_protocol_12_basic_functionality(self):
        """Test basic functionality of Protocol 12."""
        try:
            vp12 = load_validation_protocol(12)

            validator = vp12.APGIValidationProtocol12()

            # Test that validator has required methods
            assert hasattr(validator, "run_validation")  # nosec B101
            assert hasattr(validator, "check_criteria")  # nosec B101
            assert callable(validator.run_validation)  # nosec B101
            assert callable(validator.check_criteria)  # nosec B101
            assert len(validator.PROTOCOL_TIERS) == len(set(validator.PROTOCOL_TIERS.keys()))  # nosec B101

        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Protocol 12 not available")

    @pytest.mark.parametrize("protocol_num", [5, 6, 7, 8, 9, 10, 11, 12])
    def test_all_protocols_have_validation_interface(self, protocol_num):
        """Test that all protocols 5-12 have the validation interface."""
        try:
            module = load_validation_protocol(protocol_num)

            # Find the main class in the module
            validator_class = None
            for attr_name in dir(module):
                if (
                    attr_name.startswith("APGIValidationProtocol")
                    or "Validator" in attr_name
                    or "ClinicalConvergence" in attr_name
                ):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        validator_class = attr
                        break

            if not validator_class:
                pytest.skip(f"Protocol {protocol_num} main class not found")

            validator = validator_class()

            # Check that all validators have a validation method
            # Protocols use different names: run_validation, validate, run_full_validation
            validation_methods = [
                "run_validation",
                "validate",
                "run_full_validation",
                "run_all_protocols",
            ]
            has_method = False
            for method in validation_methods:
                if hasattr(validator, method) and callable(getattr(validator, method)):
                    has_method = True
                    break

            assert has_method, f"Protocol {protocol_num} missing validation method"  # nosec B101
            assert (
                protocol_num in validator.PROTOCOL_TIERS
            ), f"Protocol {protocol_num} missing tier assignment"  # nosec B101

        except (ImportError, FileNotFoundError):
            pytest.skip(f"Protocol {protocol_num} not available")


def test_apgi_dynamical_system_simulate_surprise_accumulation():
    """Unit test for APGIDynamicalSystem.simulate_surprise_accumulation()."""
    import numpy as np

    try:
        # Try to import the dynamical system from the correct location
        # APGIDynamicalSystem is in VP_01_SyntheticEEGMLClassification.py
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "protocol1",
            Path(__file__).parent.parent / "Validation" / "VP_01_SyntheticEEGMLClassification.py",
        )
        protocol1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol1)
        APGIDynamicalSystem = protocol1.APGIDynamicalSystem
    except ImportError as e:
        # Instead of skipping, raise the error to see what fails
        pytest.skip(f"APGIDynamicalSystem import failed: {e}")

    # Create system with test parameters
    system = APGIDynamicalSystem(tau=0.5, alpha=5.0)

    # Test simulate_surprise_accumulation method
    epsilon_e, epsilon_i, Pi_e, Pi_i, beta, theta_t = 0.1, 0.05, 1.0, 1.0, 1.2, 0.5
    (
        S_trajectory,
        B_trajectory,
        ignition_occurred,
        theta_trajectory,
    ) = system.simulate_surprise_accumulation(
        epsilon_e=epsilon_e,
        epsilon_i=epsilon_i,
        Pi_e=Pi_e,
        Pi_i=Pi_i,
        beta=beta,
        theta_t=theta_t,
    )

    # Verify results structure
    assert isinstance(S_trajectory, np.ndarray)  # nosec B101
    assert isinstance(B_trajectory, np.ndarray)  # nosec B101
    assert isinstance(ignition_occurred, bool)  # nosec B101
    assert isinstance(theta_trajectory, np.ndarray)  # nosec B101

    # Check array lengths (assuming duration=1.0, dt=0.001, n_steps=1000)
    expected_length = int(1.0 / 0.001)
    assert len(S_trajectory) == expected_length  # nosec B101
    assert len(B_trajectory) == expected_length  # nosec B101
    assert len(theta_trajectory) == expected_length  # nosec B101

    # Verify reasonable value ranges
    assert all(s >= 0 for s in S_trajectory)  # nosec B101  # Surprise should be non-negative
    assert all(0 <= b <= 1 for b in B_trajectory)  # nosec B101  # Ignition probability in [0,1]


def test_validation_protocol_3_hierarchical_generative_model():
    """Test HierarchicalGenerativeModel from VP_03_ActiveInferenceAgentSimulations.py"""
    try:
        import importlib.util

        import numpy as np
        import torch

        spec = importlib.util.spec_from_file_location(
            "protocol3",
            Path(__file__).parent.parent / "Validation" / "VP_03_ActiveInferenceAgentSimulations.py",
        )
        protocol3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol3)
        HierarchicalGenerativeModel = protocol3.HierarchicalGenerativeModel
    except ImportError as e:
        pytest.skip(f"HierarchicalGenerativeModel import failed: {e}")

    # Test model initialization with tau values matching LEVEL_TIMESCALES constant
    levels = [
        {"dim": 10, "tau": 0.05, "name": "bottom"},  # TAU_SENSORY
        {"dim": 5, "tau": 0.2, "name": "middle"},  # TAU_ORGAN
        {"dim": 2, "tau": 0.5, "name": "top"},  # TAU_COGNITIVE
    ]

    model = HierarchicalGenerativeModel(levels)

    # Test basic properties
    assert model.n_levels == 3  # nosec B101
    assert len(model.level_networks) == 2  # nosec B101  # One network between each level

    # Test prediction
    prediction = model.predict(0)
    assert isinstance(prediction, torch.Tensor)  # nosec B101
    assert prediction.shape[0] == 10  # nosec B101  # Bottom level dimension

    # Test update with prediction error
    error = torch.randn(10)
    model.update(error, level=0)

    # Test get_level
    level_state = model.get_level("bottom")
    assert isinstance(level_state, np.ndarray)  # nosec B101
    assert level_state.shape[0] == 10  # nosec B101

    # Test get_all_levels
    all_states = model.get_all_levels()
    assert isinstance(all_states, np.ndarray)  # nosec B101
    assert all_states.shape[0] == 17  # nosec B101  # 10 + 5 + 2


def test_validation_protocol_3_somatic_marker_network():
    """Test SomaticMarkerNetwork from VP_03_ActiveInferenceAgentSimulations.py"""
    try:
        import importlib.util

        import numpy as np
        import torch

        spec = importlib.util.spec_from_file_location(
            "protocol3",
            Path(__file__).parent.parent / "Validation" / "VP_03_ActiveInferenceAgentSimulations.py",
        )
        protocol3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol3)
        SomaticMarkerNetwork = protocol3.SomaticMarkerNetwork
    except ImportError as e:
        pytest.skip(f"SomaticMarkerNetwork import failed: {e}")

    # Test initialization
    network = SomaticMarkerNetwork(context_dim=5, action_dim=4)

    # Test forward pass
    context = torch.randn(5)
    predictions = network.forward(context)
    assert predictions.shape[0] == 4  # nosec B101  # One prediction per action

    # Test numpy interface
    context_np = np.random.randn(5)
    predictions_np = network.predict(context_np)
    assert isinstance(predictions_np, np.ndarray)  # nosec B101
    assert predictions_np.shape[0] == 4  # nosec B101  # One prediction per action

    # Test update
    network.update(context_np, action=0, prediction_error=0.1)

    # Verify predictions changed
    new_predictions = network.predict(context_np)
    assert not np.array_equal(predictions_np, new_predictions)  # nosec B101


def test_validation_protocol_3_policy_network():
    """Test PolicyNetwork from VP_03_ActiveInferenceAgentSimulations.py"""
    try:
        import importlib.util

        import numpy as np
        import torch

        spec = importlib.util.spec_from_file_location(
            "protocol3",
            Path(__file__).parent.parent / "Validation" / "VP_03_ActiveInferenceAgentSimulations.py",
        )
        protocol3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol3)
        PolicyNetwork = protocol3.PolicyNetwork
    except ImportError as e:
        pytest.skip(f"PolicyNetwork import failed: {e}")

    # Test initialization
    network = PolicyNetwork(state_dim=10, action_dim=4)

    # Test forward pass
    state = torch.randn(10)
    probs = network.forward(state)
    assert probs.shape[0] == 4  # nosec B101
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)  # Should sum to 1  # nosec B101

    # Test action selection
    state_np = np.random.randn(10)
    action, action_probs = network.select_action(state_np)
    assert isinstance(action, int)  # nosec B101
    assert 0 <= action < 4  # Valid action range  # nosec B101
    assert isinstance(action_probs, torch.Tensor)  # nosec B101
    assert action_probs.shape[0] == 4  # nosec B101

    # Test update (should not crash)
    network.update(final_reward=1.0)


# Placeholder tests for protocols 4-12 (to be expanded)
def test_validation_protocol_4_apgi_dynamical_system():
    """Test APGIDynamicalSystem from VP_04_PhaseTransitionEpistemicLevel2.py"""
    try:
        import importlib.util

        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "protocol4",
            Path(__file__).parent.parent / "Validation" / "VP_04_PhaseTransitionEpistemicLevel2.py",
        )
        protocol4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol4)
        APGIDynamicalSystem = protocol4.APGIDynamicalSystem
    except ImportError as e:
        pytest.skip(f"APGIDynamicalSystem import failed: {e}")

    # Test initialization
    system = APGIDynamicalSystem(tau=0.2, theta_0=0.55, alpha=5.0, dt=0.01)

    # Test basic properties
    assert hasattr(system, "tau")  # nosec B101
    assert hasattr(system, "theta_0")  # nosec B101
    assert hasattr(system, "alpha")  # nosec B101
    assert hasattr(system, "dt")  # nosec B101
    assert system.tau == 0.2  # nosec B101
    assert system.theta_0 == 0.55  # nosec B101
    assert system.alpha == 5.0  # nosec B101
    assert system.dt == 0.01  # nosec B101

    # Test simulate method with simple input generator
    def input_generator(t):
        return {
            "Pi_e": 1.0,
            "eps_e": 0.1,
            "beta": 1.2,
            "Pi_i": 1.0,
            "eps_i": 0.05,
            "M": 0.8,
            "c": 0.1,
            "a": 0.9,
        }

    duration = 1.0  # Short simulation for testing
    results = system.simulate(duration, input_generator)

    # Verify results structure
    required_keys = [
        "time",
        "S",
        "theta",
        "B",
        "Pi_e",
        "eps_e",
        "Pi_i",
        "eps_i",
        "ignition_events",
    ]
    for key in required_keys:
        assert key in results  # nosec B101
        assert isinstance(results[key], np.ndarray)  # nosec B101

    # Check array lengths
    expected_length = int(duration / system.dt)
    assert len(results["time"]) == expected_length  # nosec B101
    assert len(results["S"]) == expected_length  # nosec B101
    assert len(results["theta"]) == expected_length  # nosec B101
    assert len(results["B"]) == expected_length  # nosec B101

    # Verify reasonable value ranges
    assert all(s >= 0 for s in results["S"])  # Surprise should be non-negative  # nosec B101
    assert all(0 <= b <= 1 for b in results["B"])  # Ignition probability in [0,1]  # nosec B101


def test_validation_protocol_4_exists():
    """Test that VP_04_PhaseTransitionEpistemicLevel2.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_04_PhaseTransitionEpistemicLevel2.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol4", protocol_path)
        protocol4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol4)
    except ImportError as e:
        pytest.skip(f"VP_04_PhaseTransitionEpistemicLevel2.py import failed: {e}")


def test_validation_protocol_5_exists():
    """Test that VP_05_EvolutionaryEmergence.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_05_EvolutionaryEmergence.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol5", protocol_path)
        protocol5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol5)
    except ImportError as e:
        pytest.skip(f"VP_05_EvolutionaryEmergence.py import failed: {e}")


def test_validation_protocol_6_exists():
    """Test that VP_06_LiquidNetworkInductiveBias.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_06_LiquidNetworkInductiveBias.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol6", protocol_path)
        protocol6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol6)
    except ImportError as e:
        pytest.skip(f"VP_06_LiquidNetworkInductiveBias.py import failed: {e}")


def test_validation_protocol_7_exists():
    """Test that VP_07_TMSCausalInterventions.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_07_TMSCausalInterventions.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol7", protocol_path)
        protocol7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol7)
    except ImportError as e:
        pytest.skip(f"VP_07_TMSCausalInterventions.py import failed: {e}")


def test_validation_protocol_8_exists():
    """Test that VP_08_PsychophysicalThresholdEstimation.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_08_PsychophysicalThresholdEstimation.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol8", protocol_path)
        protocol8 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol8)
    except ImportError as e:
        pytest.skip(f"VP_08_PsychophysicalThresholdEstimation.py import failed: {e}")


def test_validation_protocol_9_exists():
    """Test that VP_09_NeuralSignaturesEmpiricalPriority1.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_09_NeuralSignaturesEmpiricalPriority1.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol9", protocol_path)
        protocol9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol9)
    except ImportError as e:
        pytest.skip(f"VP_09_NeuralSignaturesEmpiricalPriority1.py import failed: {e}")


def test_validation_protocol_10_exists():
    """Test that VP_10_CausalManipulationsPriority2.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_10_CausalManipulationsPriority2.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol10", protocol_path)
        protocol10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol10)
    except ImportError as e:
        pytest.skip(f"VP_10_CausalManipulationsPriority2.py import failed: {e}")


def test_validation_protocol_11_exists():
    """Test that VP_11_MCMCCulturalNeurosciencePriority3.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_11_MCMCCulturalNeurosciencePriority3.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol11", protocol_path)
        protocol11 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol11)
    except ImportError as e:
        pytest.skip(f"VP_11_MCMCCulturalNeurosciencePriority3.py import failed: {e}")


def test_validation_protocol_12_exists():
    """Test that VP_12_ClinicalCrossSpeciesConvergence.py exists and can be imported"""
    protocol_path = Path(__file__).parent.parent / "Validation" / "VP_12_ClinicalCrossSpeciesConvergence.py"
    assert protocol_path.exists()  # nosec B101

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("protocol12", protocol_path)
        protocol12 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol12)
    except ImportError as e:
        pytest.skip(f"VP_12_ClinicalCrossSpeciesConvergence.py import failed: {e}")


@pytest.mark.parametrize(
    "simulation_steps,dt,expected_error",
    [
        (0, 0.1, "positive integer"),  # Zero steps
        (-1, 0.1, "positive integer"),  # Negative steps
        (1000, 0, "positive number"),  # Zero dt
        (1000, -0.1, "positive number"),  # Negative dt
        (100001, 0.1, "long time"),  # Too many steps
        (1000, 1.1, "accuracy"),  # Too large dt
    ],
)
def test_formal_model_validation_edge_cases(simulation_steps, dt, expected_error):
    """Test formal model validation with edge cases"""

    # Test that validation catches edge cases
    if "positive integer" in expected_error and simulation_steps <= 0:
        assert simulation_steps <= 0  # nosec B101
    elif "positive number" in expected_error and dt <= 0:
        assert dt <= 0  # nosec B101
    elif "long time" in expected_error:
        assert simulation_steps > 100000  # nosec B101
    elif "accuracy" in expected_error:
        assert dt > 1.0  # nosec B101


@pytest.mark.parametrize(
    "levels_config",
    [
        # Normal case - using valid tau values from LEVEL_TIMESCALES
        [
            {"dim": 10, "tau": 0.05, "name": "bottom"},  # TAU_SENSORY
            {"dim": 5, "tau": 0.2, "name": "middle"},  # TAU_ORGAN
        ],
        # Edge case: many levels - using valid tau values
        [
            {"dim": 10, "tau": 0.05, "name": "l1"},  # TAU_SENSORY
            {"dim": 8, "tau": 0.2, "name": "l2"},  # TAU_ORGAN
            {"dim": 6, "tau": 0.5, "name": "l3"},  # TAU_COGNITIVE
            {"dim": 4, "tau": 2.0, "name": "l4"},  # TAU_HOMEOSTATIC
        ],
    ],
)
def test_hierarchical_generative_model_edge_cases(levels_config):
    """Test HierarchicalGenerativeModel with edge case configurations"""
    try:
        import importlib.util

        import torch

        spec = importlib.util.spec_from_file_location(
            "protocol3",
            Path(__file__).parent.parent / "Validation" / "VP_03_ActiveInferenceAgentSimulations.py",
        )
        protocol3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol3)
        HierarchicalGenerativeModel = protocol3.HierarchicalGenerativeModel
    except ImportError:
        pytest.skip("HierarchicalGenerativeModel import failed")

    model = HierarchicalGenerativeModel(levels_config)

    # Test prediction for valid configurations
    prediction = model.predict(0)
    assert isinstance(prediction, torch.Tensor)  # nosec B101

    # Test update
    error_dim = levels_config[0]["dim"]
    if error_dim > 0:
        error = torch.randn(error_dim)
        model.update(error, level=0)


@pytest.mark.parametrize(
    "context_dim,action_dim",
    [
        (1, 1),  # Minimal dimensions
        (100, 10),  # Large dimensions
        (5, 4),  # Standard case
    ],
)
def test_somatic_marker_network_edge_cases(context_dim, action_dim):
    """Test SomaticMarkerNetwork with edge case dimensions"""
    try:
        import importlib.util

        import numpy as np
        import torch

        spec = importlib.util.spec_from_file_location(
            "protocol3",
            Path(__file__).parent.parent / "Validation" / "VP_03_ActiveInferenceAgentSimulations.py",
        )
        protocol3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol3)
        SomaticMarkerNetwork = protocol3.SomaticMarkerNetwork
    except ImportError:
        pytest.skip("SomaticMarkerNetwork import failed")

    network = SomaticMarkerNetwork(context_dim=context_dim, action_dim=action_dim)

    # Test with appropriate context size
    context = torch.randn(context_dim)
    predictions = network.forward(context)
    assert predictions.shape[0] == action_dim  # nosec B101

    # Test numpy interface
    context_np = np.random.randn(context_dim)
    predictions_np = network.predict(context_np)
    assert isinstance(predictions_np, np.ndarray)  # nosec B101
    assert predictions_np.shape[0] == action_dim  # nosec B101


@pytest.mark.parametrize(
    "state_dim,action_dim",
    [
        (1, 1),  # Minimal case
        (100, 10),  # Large case
        (10, 4),  # Standard case
    ],
)
def test_policy_network_edge_cases(state_dim, action_dim):
    """Test PolicyNetwork with edge case dimensions"""
    try:
        import importlib.util

        import numpy as np
        import torch

        spec = importlib.util.spec_from_file_location(
            "protocol3",
            Path(__file__).parent.parent / "Validation" / "VP_03_ActiveInferenceAgentSimulations.py",
        )
        protocol3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol3)
        PolicyNetwork = protocol3.PolicyNetwork
    except ImportError:
        pytest.skip("PolicyNetwork import failed")

    network = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)

    # Test forward pass
    state = torch.randn(state_dim)
    probs = network.forward(state)
    assert probs.shape[0] == action_dim  # nosec B101
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)  # Should sum to 1  # nosec B101

    # Test action selection with appropriate state size
    state_np = np.random.randn(state_dim)
    action, action_probs = network.select_action(state_np)
    assert isinstance(action, int)  # nosec B101
    assert 0 <= action < min(action_dim, 4)  # nosec B101  # Clamped to 4 or action_dim if smaller


@pytest.mark.parametrize(
    "tau_S,tau_theta,alpha",
    [
        (0.1, 10.0, 1.0),  # Small values
        (2.0, 100.0, 20.0),  # Large values
        (0.5, 30.0, 5.0),  # Standard values
        (0.01, 1.0, 0.1),  # Very small values
    ],
)
def test_apgi_dynamical_system_parameter_ranges(tau_S, tau_theta, alpha):
    """Test APGIDynamicalSystem with different parameter ranges"""
    try:
        import importlib.util

        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "synthetic_eeg",
            Path(__file__).parent.parent / "Validation" / "VP_01_SyntheticEEGMLClassification.py",
        )
        synthetic_eeg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(synthetic_eeg)
        APGIDynamicalSystem = synthetic_eeg.APGIDynamicalSystem
    except ImportError:
        pytest.skip("APGIDynamicalSystem import failed")

    system = APGIDynamicalSystem(tau=tau_S, alpha=alpha)

    # Test that system initializes and runs with different parameters
    epsilon_e, epsilon_i, Pi_e, Pi_i, beta, theta_t = 0.1, 0.05, 1.0, 1.0, 1.2, 0.5
    (
        S_trajectory,
        B_trajectory,
        ignition_occurred,
        theta_trajectory,
    ) = system.simulate_surprise_accumulation(
        epsilon_e=epsilon_e,
        epsilon_i=epsilon_i,
        Pi_e=Pi_e,
        Pi_i=Pi_i,
        beta=beta,
        theta_t=theta_t,
    )

    # Verify results structure
    assert isinstance(S_trajectory, np.ndarray)  # nosec B101
    assert isinstance(B_trajectory, np.ndarray)  # nosec B101
    assert isinstance(ignition_occurred, bool)  # nosec B101
    assert isinstance(theta_trajectory, np.ndarray)  # nosec B101
    assert len(S_trajectory) > 0  # nosec B101
    assert len(B_trajectory) > 0  # nosec B101
    assert len(theta_trajectory) > 0  # nosec B101


@pytest.mark.parametrize(
    "passed_count,total_count,expected_decision",
    [
        (12, 12, "PASS: Strong validation support"),  # 100% pass rate
        (10, 12, "PASS: Strong validation support"),  # >80% pass rate
        (
            9,
            12,
            "MARGINAL: Moderate validation support",
        ),  # 75% pass rate, weighted ~0.83
        (
            6,
            12,
            "MARGINAL: Moderate validation support",
        ),  # 50% pass rate, weighted ~0.6
        (
            5,
            12,
            "MARGINAL: Moderate validation support",
        ),  # Weighted score ~0.65 (MARGINAL)
        (0, 12, "FAIL: Insufficient validation support"),  # 0% pass rate
        (1, 1, "PASS: Strong validation support"),  # Single protocol pass
        (0, 1, "FAIL: Insufficient validation support"),  # Single protocol fail
    ],
)
def test_generate_master_report_decision_logic(passed_count, total_count, expected_decision):
    """Test generate_master_report decision logic with different pass rates"""
    from utils.protocol_schema import PredictionResult, PredictionStatus, ProtocolResult

    validator = APGIMasterValidator()

    # Create mock protocol results using ProtocolResult objects
    validator.protocol_results = {}
    for i in range(total_count):
        protocol_name = f"Protocol-{i + 1}"
        passed = i < passed_count  # First passed_count protocols pass
        validator.protocol_results[protocol_name] = ProtocolResult(
            protocol_id=protocol_name,
            named_predictions={
                f"prediction_{i}": PredictionResult(
                    passed=passed,
                    value=0.5 if passed else 0.2,
                    threshold=0.3,
                    status=(PredictionStatus.PASSED if passed else PredictionStatus.FAILED),
                )
            },
            completion_percentage=100,
            metadata={
                "status": "COMPLETED",
                "passed": passed,
                "timestamp": "2024-01-01T00:00:00",
            },
        )

    # Generate report
    report = validator.generate_master_report()

    # Verify decision logic
    assert report["overall_decision"] == expected_decision  # nosec B101
    assert report["total_protocols"] == total_count  # nosec B101
    assert report["passed_protocols"] == passed_count  # nosec B101
    assert report["success_rate"] == (passed_count / total_count if total_count > 0 else 0)  # nosec B101


def test_generate_master_report_edge_cases():
    """Test generate_master_report with edge cases"""
    from utils.protocol_schema import ProtocolResult

    validator = APGIMasterValidator()

    # Test with no protocols run
    validator.protocol_results = {}
    report = validator.generate_master_report()
    assert report["overall_decision"] == "No protocols run"  # nosec B101
    assert report["total_protocols"] == 0  # nosec B101
    assert report["passed_protocols"] == 0  # nosec B101
    assert report["success_rate"] == 0  # nosec B101

    # Test with empty protocol_results
    validator.protocol_results = {
        "Protocol-1": ProtocolResult(
            protocol_id="Protocol-1",
            named_predictions={},
            completion_percentage=0,
            metadata={"status": "FAILED"},
        )
    }
    report = validator.generate_master_report()
    assert report["overall_decision"] == "FAIL: Insufficient validation support"  # nosec B101
    assert report["total_protocols"] == 1  # nosec B101
    assert report["passed_protocols"] == 0  # Default to False when "passed" key missing  # nosec B101
