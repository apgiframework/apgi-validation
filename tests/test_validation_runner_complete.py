"""
Comprehensive tests for utils/validation_runner.py - 100% coverage target.

This file tests:
- Validation runner initialization and configuration
- Protocol execution and management
- Result aggregation and reporting
- Error handling and recovery
- Performance monitoring
- Concurrent execution
- Resource management
"""

import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.validation_runner import ExecutionStatus, ProtocolConfig, ValidationResult, ValidationRunner

    VALIDATION_RUNNER_AVAILABLE = True
except ImportError as e:
    VALIDATION_RUNNER_AVAILABLE = False
    print(f"Warning: validation_runner not available for testing: {e}")


class TestValidationRunnerComplete:
    """Comprehensive tests for ValidationRunner functionality."""

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_runner_initialization(self):
        """Test validation runner initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir, max_concurrent_protocols=2, timeout_seconds=300)

            assert runner.output_dir == Path(temp_dir), "Runner output_dir should match temp_dir"
            assert runner.max_concurrent_protocols == 2, "Runner max_concurrent_protocols should be 2"
            assert runner.timeout_seconds == 300, "Runner timeout_seconds should be 300"
            assert runner.active_protocols == {}, "Runner active_protocols should be empty dict"
            assert runner.completed_protocols == [], "Runner completed_protocols should be empty list"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_protocol_configuration(self):
        """Test protocol configuration handling."""
        config = ProtocolConfig(
            protocol_id="VP_01",
            protocol_name="Synthetic EEG Classification",
            parameters={"n_subjects": 100, "n_sessions": 2},
            dependencies=[],
            priority=1,
            timeout_seconds=600,
        )

        assert config.protocol_id == "VP_01", "Config protocol_id should be VP_01"
        assert config.protocol_name == "Synthetic EEG Classification", "Config protocol_name should match"
        assert config.parameters["n_subjects"] == 100, "Config n_subjects should be 100"
        assert config.dependencies == [], "Config dependencies should be empty list"
        assert config.priority == 1, "Config priority should be 1"
        assert config.timeout_seconds == 600, "Config timeout_seconds should be 600"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_add_protocol_to_runner(self):
        """Test adding protocols to the runner."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            config = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Test Protocol",
                parameters={"test_param": "test_value"},
                dependencies=[],
                priority=1,
                timeout_seconds=300,
            )

            runner.add_protocol(config)

            assert "VP_01" in runner.protocol_configs, "VP_01 should be in runner protocol_configs"
            assert (
                runner.protocol_configs["VP_01"].protocol_id == "VP_01"
            ), "Stored config should have correct protocol_id"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_protocol_execution_success(self):
        """Test successful protocol execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Mock protocol execution
            mock_protocol = Mock()
            mock_protocol.run.return_value = {
                "status": "completed",
                "results": {"accuracy": 0.85, "precision": 0.90},
                "execution_time": 120.5,
                "metadata": {"n_subjects": 100},
            }

            config = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Test Protocol",
                parameters={"test_param": "test_value"},
                dependencies=[],
                priority=1,
                timeout_seconds=300,
            )

            # Execute protocol
            result = runner.execute_protocol(config, mock_protocol)

            assert result.status == ExecutionStatus.COMPLETED, "Protocol should have completed status"
            assert result.protocol_id == "VP_01", "Result protocol_id should be VP_01"
            assert "accuracy" in result.results, "Result should contain accuracy metric"
            assert result.execution_time == 120.5, "Result execution_time should be 120.5"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_protocol_execution_timeout(self):
        """Test protocol execution timeout handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir, timeout_seconds=1)

            # Mock protocol that takes too long
            mock_protocol = Mock()
            mock_protocol.run.side_effect = lambda: time.sleep(2)

            config = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Test Protocol",
                parameters={"test_param": "test_value"},
                dependencies=[],
                priority=1,
                timeout_seconds=int(0.5),  # Very short timeout
            )

            # Execute protocol (should timeout)
            result = runner.execute_protocol(config, mock_protocol)

            assert result.status == ExecutionStatus.TIMEOUT, "Protocol should have timeout status"
            assert "timeout" in result.error_message.lower(), "Error message should mention timeout"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_protocol_execution_error(self):
        """Test protocol execution error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Mock protocol that raises an exception
            mock_protocol = Mock()
            mock_protocol.run.side_effect = ValueError("Test error")

            config = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Test Protocol",
                parameters={"test_param": "test_value"},
                dependencies=[],
                priority=1,
                timeout_seconds=300,
            )

            # Execute protocol (should fail)
            result = runner.execute_protocol(config, mock_protocol)

            assert result.status == ExecutionStatus.FAILED, "Protocol should have failed status"
            assert "Test error" in result.error_message, "Error message should contain 'Test error'"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_dependency_resolution(self):
        """Test protocol dependency resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Create protocols with dependencies
            config1 = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Base Protocol",
                parameters={},
                dependencies=[],
                priority=1,
                timeout_seconds=300,
            )

            config2 = ProtocolConfig(
                protocol_id="VP_02",
                protocol_name="Dependent Protocol",
                parameters={},
                dependencies=["VP_01"],
                priority=2,
                timeout_seconds=300,
            )

            config3 = ProtocolConfig(
                protocol_id="VP_03",
                protocol_name="Another Dependent",
                parameters={},
                dependencies=["VP_01", "VP_02"],
                priority=3,
                timeout_seconds=300,
            )

            runner.add_protocol(config1)
            runner.add_protocol(config2)
            runner.add_protocol(config3)

            # Test dependency resolution
            execution_order = runner.resolve_dependencies()

            assert execution_order[0] == "VP_01", "First protocol should be VP_01 (no dependencies)"
            assert execution_order[1] == "VP_02", "Second protocol should be VP_02 (depends on VP_01)"
            assert execution_order[2] == "VP_03", "Third protocol should be VP_03 (depends on VP_01 and VP_02)"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Create circular dependency
            config1 = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Protocol 1",
                parameters={},
                dependencies=["VP_02"],
                priority=1,
                timeout_seconds=300,
            )

            config2 = ProtocolConfig(
                protocol_id="VP_02",
                protocol_name="Protocol 2",
                parameters={},
                dependencies=["VP_01"],
                priority=2,
                timeout_seconds=300,
            )

            runner.add_protocol(config1)
            runner.add_protocol(config2)

            # Should detect circular dependency
            with pytest.raises(ValueError, match="circular dependency"):
                runner.resolve_dependencies()

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_concurrent_execution(self):
        """Test concurrent protocol execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir, max_concurrent_protocols=2)

            # Mock protocols
            protocols = []
            configs = []

            for i in range(3):
                mock_protocol = Mock()
                mock_protocol.run.return_value = {
                    "status": "completed",
                    "results": {"protocol_id": i},
                    "execution_time": 0.1,
                }
                protocols.append(mock_protocol)

                config = ProtocolConfig(
                    protocol_id=f"VP_{i:02d}",
                    protocol_name=f"Protocol {i}",
                    parameters={},
                    dependencies=[],
                    priority=i,
                    timeout_seconds=300,
                )
                configs.append(config)
                runner.add_protocol(config)

            # Execute all protocols
            results = runner.run_all_protocols()

            assert len(results) == 3, "Should have 3 protocol results"
            assert all(
                result.status == ExecutionStatus.COMPLETED for result in results
            ), "All protocols should have completed status"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_result_aggregation(self):
        """Test result aggregation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Add mock results
            result1 = ValidationResult(
                protocol_id="VP_01",
                protocol_name="Protocol 1",
                status=ExecutionStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                execution_time=120.5,
                results={"accuracy": 0.85, "precision": 0.90},
                error_message=None,
                output_files=["results_01.json"],
            )

            result2 = ValidationResult(
                protocol_id="VP_02",
                protocol_name="Protocol 2",
                status=ExecutionStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                execution_time=95.2,
                results={"accuracy": 0.78, "recall": 0.82},
                error_message=None,
                output_files=["results_02.json"],
            )

            runner.completed_protocols = [result1, result2]

            # Aggregate results
            aggregated = runner.aggregate_results()

            assert "summary" in aggregated, "Aggregated results should contain summary"
            assert aggregated["summary"]["total_protocols"] == 2, "Summary should show 2 total protocols"
            assert aggregated["summary"]["completed_protocols"] == 2, "Summary should show 2 completed protocols"
            assert "protocols" in aggregated, "Aggregated results should contain protocols"
            assert len(aggregated["protocols"]) == 2, "Should have 2 protocols in results"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Create test result
            result = ValidationResult(
                protocol_id="VP_01",
                protocol_name="Test Protocol",
                status=ExecutionStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                execution_time=120.5,
                results={"accuracy": 0.85},
                error_message=None,
                output_files=["test_results.json"],
            )

            runner.completed_protocols = [result]

            # Save results
            results_file = Path(temp_dir) / "validation_results.json"
            runner.save_results(str(results_file))

            assert results_file.exists(), "Results file should be created after saving"

            # Load results
            loaded_runner = ValidationRunner(output_dir=temp_dir)
            loaded_runner.load_results(str(results_file))

            assert len(loaded_runner.completed_protocols) == 1, "Should have 1 completed protocol after loading"
            assert loaded_runner.completed_protocols[0].protocol_id == "VP_01", "Loaded protocol should be VP_01"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_progress_tracking(self):
        """Test progress tracking during execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            progress_updates = []

            def progress_callback(protocol_id, progress, status):
                progress_updates.append((protocol_id, progress, status))

            runner.set_progress_callback(progress_callback)

            # Mock protocol
            mock_protocol = Mock()
            mock_protocol.run.return_value = {
                "status": "completed",
                "results": {"test": "data"},
                "execution_time": 0.1,
            }

            config = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Test Protocol",
                parameters={},
                dependencies=[],
                priority=1,
                timeout_seconds=300,
            )

            # Execute with progress tracking
            runner.execute_protocol(config, mock_protocol)

            # Should have received progress updates
            assert len(progress_updates) > 0, "Should have received progress updates"
            assert any(update[0] == "VP_01" for update in progress_updates), "Should have progress updates for VP_01"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_resource_cleanup(self):
        """Test resource cleanup after execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Mock protocol with resources
            mock_protocol = Mock()
            mock_resource = Mock()
            mock_protocol.resources = [mock_resource]
            mock_protocol.run.return_value = {
                "status": "completed",
                "results": {"test": "data"},
                "execution_time": 0.1,
            }

            config = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Test Protocol",
                parameters={},
                dependencies=[],
                priority=1,
                timeout_seconds=300,
            )

            # Execute protocol
            runner.execute_protocol(config, mock_protocol)

            # Cleanup resources
            runner.cleanup_resources()

            # Should have called cleanup on resources
            mock_resource.cleanup.assert_called_once()

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_priority_based_execution(self):
        """Test priority-based protocol execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Create protocols with different priorities
            configs = []
            for i in range(3):
                config = ProtocolConfig(
                    protocol_id=f"VP_{i:02d}",
                    protocol_name=f"Protocol {i}",
                    parameters={},
                    dependencies=[],
                    priority=3 - i,  # Reverse priority: 2, 1, 0
                    timeout_seconds=300,
                )
                configs.append(config)
                runner.add_protocol(config)

            # Get execution order
            execution_order = runner.get_priority_order()

            # Should be ordered by priority (highest first)
            assert execution_order[0] == "VP_00", "First protocol should be VP_00 (priority 2)"
            assert execution_order[1] == "VP_01", "Second protocol should be VP_01 (priority 1)"
            assert execution_order[2] == "VP_02", "Third protocol should be VP_02 (priority 0)"

    @pytest.mark.skipif(not VALIDATION_RUNNER_AVAILABLE, reason="validation_runner not available")
    def test_error_recovery_mechanism(self):
        """Test error recovery during execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ValidationRunner(output_dir=temp_dir)

            # Mock protocol that fails initially
            mock_protocol = Mock()
            call_count = 0

            def failing_run():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ValueError("Temporary failure")
                return {
                    "status": "completed",
                    "results": {"test": "data"},
                    "execution_time": 0.1,
                }

            mock_protocol.run = failing_run

            config = ProtocolConfig(
                protocol_id="VP_01",
                protocol_name="Test Protocol",
                parameters={},
                dependencies=[],
                priority=1,
                timeout_seconds=300,
                max_retries=2,
            )

            # Execute with retry
            result = runner.execute_protocol_with_retry(config, mock_protocol)

            # Should succeed on retry
            assert result.status == ExecutionStatus.COMPLETED, "Protocol should complete after retry"
            assert call_count == 2, "Should have made 2 attempts (1 fail + 1 success)"
