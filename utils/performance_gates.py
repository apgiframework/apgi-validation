"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List


class GateStatus(Enum):
    """Status of a performance gate check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class PerformanceGate:
    """A performance gate with thresholds and enforcement logic."""

    name: str
    description: str
    threshold: float
    unit: str
    check_function: Callable[[], float]
    comparison: str = "<="  # "<=", ">=", "<", ">"
    critical: bool = True
    enabled: bool = True


class PerformanceGateEnforcer:
    """Enforces performance gates for protocol runtime and throughput."""

    def __init__(self) -> None:
        self.gates: List[PerformanceGate] = []
        self.results: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)

    def register_gate(self, gate: PerformanceGate) -> None:
        """Register a performance gate."""
        self.gates.append(gate)

    def enforce_all(self) -> Dict[str, GateStatus]:
        """Enforce all registered performance gates."""
        self.results = {}
        overall_status = GateStatus.PASSED

        for gate in self.gates:
            if not gate.enabled:
                self.results[gate.name] = {
                    "status": GateStatus.SKIPPED,
                    "message": "Gate disabled",
                }
                continue

            try:
                value = gate.check_function()
                passed = self._check_threshold(gate, value)

                status = GateStatus.PASSED if passed else GateStatus.FAILED
                if not passed and not gate.critical:
                    status = GateStatus.WARNING

                self.results[gate.name] = {
                    "status": status,
                    "value": value,
                    "threshold": gate.threshold,
                    "unit": gate.unit,
                    "message": f"{gate.name}: {value} {gate.unit} {gate.comparison} {gate.threshold}",
                }

                if status == GateStatus.FAILED and gate.critical:
                    overall_status = GateStatus.FAILED
                elif status == GateStatus.WARNING and overall_status == GateStatus.PASSED:
                    overall_status = GateStatus.WARNING

                self.logger.info(
                    f"Gate '{gate.name}': {status.value} " f"({value} {gate.unit} {gate.comparison} {gate.threshold})"
                )

            except Exception as e:
                self.logger.error(f"Gate '{gate.name}' check failed: {e}")
                self.results[gate.name] = {
                    "status": (GateStatus.FAILED if gate.critical else GateStatus.WARNING),
                    "message": f"Check failed: {e}",
                }
                if gate.critical:
                    overall_status = GateStatus.FAILED

        return {gate.name: self.results[gate.name]["status"] for gate in self.gates}

    def _check_threshold(self, gate: PerformanceGate, value: float) -> bool:
        """Check if value meets the gate threshold."""
        if gate.comparison == "<=":
            return value <= gate.threshold
        elif gate.comparison == ">=":
            return value >= gate.threshold
        elif gate.comparison == "<":
            return value < gate.threshold
        elif gate.comparison == ">":
            return value > gate.threshold
        return False

    def get_report(self) -> str:
        """Generate a performance gate report."""
        lines = ["Performance Gate Report", "=" * 50, ""]

        for gate in self.gates:
            if gate.name in self.results:
                result = self.results[gate.name]
                status_icon = {
                    GateStatus.PASSED: "✅",
                    GateStatus.FAILED: "❌",
                    GateStatus.SKIPPED: "⏭️",
                    GateStatus.WARNING: "⚠️",
                }.get(result["status"], "❓")

                lines.append(f"{status_icon} {gate.name}")
                lines.append(f"   {gate.description}")
                if "value" in result:
                    lines.append(
                        f"   Result: {result['value']} {result['unit']} " f"{gate.comparison} {result['threshold']}"
                    )
                lines.append(f"   Status: {result['status'].value}")
                lines.append("")

        return "\n".join(lines)


# Predefined performance gates for common scenarios
def create_protocol_runtime_gate(
    protocol_name: str, max_runtime_seconds: float, critical: bool = True
) -> PerformanceGate:
    """Create a runtime gate for a specific protocol."""

    def check_runtime() -> float:
        # This would be replaced with actual runtime measurement
        # For now, return a placeholder
        return 0.0

    return PerformanceGate(
        name=f"{protocol_name}_runtime",
        description=f"Maximum runtime for {protocol_name}",
        threshold=max_runtime_seconds,
        unit="seconds",
        check_function=check_runtime,
        comparison="<=",
        critical=critical,
    )


def create_memory_usage_gate(protocol_name: str, max_memory_mb: float, critical: bool = True) -> PerformanceGate:
    """Create a memory usage gate for a specific protocol."""

    def check_memory() -> float:
        # This would be replaced with actual memory measurement
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB

    return PerformanceGate(
        name=f"{protocol_name}_memory",
        description=f"Maximum memory usage for {protocol_name}",
        threshold=max_memory_mb,
        unit="MB",
        check_function=check_memory,
        comparison="<=",
        critical=critical,
    )


def create_throughput_gate(operation_name: str, min_throughput: float, critical: bool = False) -> PerformanceGate:
    """Create a throughput gate for an operation."""

    def check_throughput() -> float:
        # This would be replaced with actual throughput measurement
        return 0.0

    return PerformanceGate(
        name=f"{operation_name}_throughput",
        description=f"Minimum throughput for {operation_name}",
        threshold=min_throughput,
        unit="operations/sec",
        check_function=check_throughput,
        comparison=">=",
        critical=critical,
    )


# Default SLA thresholds
DEFAULT_SLA_THRESHOLDS = {
    "protocol_runtime": 3600.0,  # 1 hour max for protocol execution
    "protocol_memory": 16384.0,  # 16GB max memory usage
    "test_suite_runtime": 300.0,  # 5 minutes max for test suite
    "validation_throughput": 1.0,  # Minimum 1 validation per second
}


def _peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (measured, Linux/macOS)."""
    import resource
    import sys as _sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux and bytes on macOS.
    return peak / (1024 * 1024) if _sys.platform == "darwin" else peak / 1024


def _not_measured(metric: str) -> Callable[[], float]:
    """A check that refuses to pass without a real measurement.

    The default gates used to return 0.0, so every runtime gate passed without
    measuring anything (APGI-IMP-003). Callers must register a gate with a real
    measurement function; until then the gate reports a failed check.
    """

    def check() -> float:
        raise RuntimeError(f"{metric} was not measured; register a gate with a real measurement")

    return check


def get_default_gates() -> List[PerformanceGate]:
    """Get default performance gates for APGI framework."""
    gates = []

    # Protocol execution gates
    gates.append(
        PerformanceGate(
            name="protocol_runtime",
            description="Maximum runtime for protocol execution",
            threshold=DEFAULT_SLA_THRESHOLDS["protocol_runtime"],
            unit="seconds",
            check_function=_not_measured("protocol_runtime"),
            comparison="<=",
            critical=True,
        )
    )

    gates.append(
        PerformanceGate(
            name="protocol_memory",
            description="Maximum memory usage for protocol execution",
            threshold=DEFAULT_SLA_THRESHOLDS["protocol_memory"],
            unit="MB",
            check_function=_peak_rss_mb,
            comparison="<=",
            critical=True,
        )
    )

    # Test suite gates
    gates.append(
        PerformanceGate(
            name="test_suite_runtime",
            description="Maximum runtime for test suite",
            threshold=DEFAULT_SLA_THRESHOLDS["test_suite_runtime"],
            unit="seconds",
            check_function=_not_measured("test_suite_runtime"),
            comparison="<=",
            critical=False,
        )
    )

    return gates


if __name__ == "__main__":
    # Example usage
    enforcer = PerformanceGateEnforcer()
    for gate in get_default_gates():
        enforcer.register_gate(gate)

    results = enforcer.enforce_all()
    print(enforcer.get_report())
