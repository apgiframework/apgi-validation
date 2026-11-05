"""Tests for FP_03 Framework Level MultiProtocol - increase coverage from 8%."""

from unittest.mock import MagicMock, patch

from Falsification.FP_03_FrameworkLevelMultiProtocol import (
    FrameworkValidator,
    MultiProtocolRunner,
    ProtocolConfig,
    ProtocolResult,
    run_multi_protocol_framework,
    validate_framework_consistency,
)


class TestProtocolConfig:
    """Test Protocol Configuration."""

    def test_config_creation(self):
        """Test creating protocol config."""
        config = ProtocolConfig(name="test_protocol", enabled=True, params={"param1": 1.0, "param2": 2.0})
        assert config.name == "test_protocol"
        assert config.enabled is True
        assert config.params["param1"] == 1.0

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ProtocolConfig(name="test", enabled=True, params={})
        result = config.to_dict()
        assert result["name"] == "test"


class TestProtocolResult:
    """Test Protocol Result."""

    def test_result_creation(self):
        """Test creating protocol result."""
        result = ProtocolResult(protocol_name="test", success=True, data={"value": 42}, errors=[])
        assert result.protocol_name == "test"
        assert result.success is True
        assert result.data["value"] == 42

    def test_result_is_valid(self):
        """Test result validity check."""
        valid_result = ProtocolResult("test", True, {}, [])
        assert valid_result.is_valid() is True

        invalid_result = ProtocolResult("test", False, {}, ["error"])
        assert invalid_result.is_valid() is False


class TestMultiProtocolRunner:
    """Test Multi-Protocol Runner."""

    def test_init(self):
        """Test initialization."""
        runner = MultiProtocolRunner()
        assert runner is not None

    def test_add_protocol(self):
        """Test adding a protocol."""
        runner = MultiProtocolRunner()
        config = ProtocolConfig("proto1", True, {})
        runner.add_protocol(config)
        assert len(runner.protocols) == 1

    def test_run_single_protocol(self):
        """Test running a single protocol."""
        runner = MultiProtocolRunner()
        config = ProtocolConfig("test_proto", True, {"test": True})
        runner.add_protocol(config)

        with patch.object(runner, "_execute_protocol") as mock_exec:
            mock_exec.return_value = ProtocolResult(
                protocol_name="test_proto",
                success=True,
                data={"result": "ok"},
                errors=[],
            )
            results = runner.run_all()
            assert len(results) == 1
            assert results[0].success is True

    def test_run_disabled_protocol(self):
        """Test that disabled protocols are skipped."""
        runner = MultiProtocolRunner()
        config = ProtocolConfig("disabled_proto", False, {})
        runner.add_protocol(config)

        results = runner.run_all()
        # Disabled protocols should not produce results
        assert len(results) == 0


class TestFrameworkValidator:
    """Test Framework Validator."""

    def test_init(self):
        """Test initialization."""
        validator = FrameworkValidator()
        assert validator is not None

    def test_validate_consistency(self):
        """Test validating consistency across results."""
        validator = FrameworkValidator()

        results = [
            ProtocolResult("proto1", True, {"metric": 1.0}, []),
            ProtocolResult("proto2", True, {"metric": 1.1}, []),
        ]

        validator.validate_consistency(results)

    def test_validate_with_conflicting_results(self):
        """Test validation with conflicting results."""
        validator = FrameworkValidator()

        results = [
            ProtocolResult("proto1", True, {"value": 100}, []),
            ProtocolResult("proto2", True, {"value": 200}, []),
        ]

        result = validator.validate_consistency(results, tolerance=0.1)
        # Results with large differences should fail consistency check
        assert result.get("consistent") is False


class TestRunMultiProtocolFramework:
    """Test main framework runner."""

    def test_run_with_empty_config(self):
        """Test running with empty config."""
        run_multi_protocol_framework([])

    def test_run_with_protocols(self):
        """Test running with protocols."""
        configs = [
            ProtocolConfig("proto1", True, {}),
            ProtocolConfig("proto2", True, {}),
        ]

        with patch("Falsification.FP_03_FrameworkLevelMultiProtocol.MultiProtocolRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.run_all.return_value = [
                ProtocolResult(
                    protocol_name="proto1",
                    success=True,
                    data={},
                    errors=[],
                ),
                ProtocolResult(
                    protocol_name="proto2",
                    success=True,
                    data={},
                    errors=[],
                ),
            ]
            MockRunner.return_value = mock_runner

            result = run_multi_protocol_framework(configs)
            # APGI-API-004: the mocked protocols return no data, so the
            # framework result carries no predictions and must not be
            # reported as scientific success.
            assert result.get("status") == "partial"
            assert "incomplete result: not reported as success" in result.get("errors", [])


class TestValidateFrameworkConsistency:
    """Test framework consistency validation."""

    def test_valid_consistency(self):
        """Test validating consistent framework."""
        results = {
            "proto1": {"status": "success", "metric": 0.5},
            "proto2": {"status": "success", "metric": 0.51},
        }
        result = validate_framework_consistency(results, tolerance=0.1)
        assert result["consistent"] is True

    def test_invalid_consistency(self):
        """Test validating inconsistent framework."""
        results = {
            "proto1": {"status": "success", "metric": 0.1},
            "proto2": {"status": "success", "metric": 0.9},
        }
        result = validate_framework_consistency(results, tolerance=0.1)
        assert result["consistent"] is False
