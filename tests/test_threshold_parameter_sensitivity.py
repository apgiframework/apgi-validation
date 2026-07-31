"""Regression tests: every exposed threshold parameter must affect the output.

These guard against the "dead parameter" class of bug, where a simulation loop
relaxes a state variable toward its own initial value, making the associated
time constant a no-op while still reporting plausible-looking results.

Historical defect: Validation_GUI._run_parameter_simulation_worker integrated
    dtheta_dt = (theta_0 - theta) / tau_theta
with theta initialised to theta_0, so dtheta_dt was identically zero and the
GUI's tau_theta slider could not influence any reported number.
"""

import pytest

from apgi_core import compute_information_value, update_threshold

# Canonical defaults mirrored from Validation_GUI.reset_parameters()
DEFAULT_TAU_S = 0.5
DEFAULT_THETA_0 = 0.5
DEFAULT_ALPHA = 5.0
GAMMA_M = -0.3

STEPS = 1000
DT = 0.01


def simulate(tau_theta, tau_S=DEFAULT_TAU_S, theta_0=DEFAULT_THETA_0, gamma_M=GAMMA_M):
    """Replicate the GUI parameter-explorer loop and return its trajectory."""
    S = 0.0
    theta = theta_0
    theta_history = []

    for _ in range(STEPS):
        S += ((1.0 - S) / tau_S) * DT
        V_info = compute_information_value(S, S)
        theta = update_threshold(
            theta=theta,
            theta0=theta_0,
            S=S,
            V_info=V_info,
            dt=DT,
            tau_theta=tau_theta,
            gamma_M=gamma_M,
        )
        theta_history.append(theta)

    return theta_history


class TestThresholdAdaptationIsLive:
    """tau_theta must have an observable effect on the threshold trajectory."""

    def test_threshold_departs_from_baseline(self):
        """Threshold must actually move; a constant theta means dead dynamics."""
        history = simulate(tau_theta=30.0)
        assert (
            max(abs(t - DEFAULT_THETA_0) for t in history) > 1e-6
        ), "Threshold never departed from theta_0 — adaptation dynamics are a no-op"

    @pytest.mark.parametrize("tau_a,tau_b", [(5.0, 30.0), (30.0, 60.0), (5.0, 60.0)])
    def test_distinct_tau_theta_gives_distinct_trajectory(self, tau_a, tau_b):
        """Different adaptation timescales must produce different thresholds."""
        final_a = simulate(tau_theta=tau_a)[-1]
        final_b = simulate(tau_theta=tau_b)[-1]
        assert final_a != pytest.approx(final_b, abs=1e-6), (
            f"tau_theta={tau_a} and tau_theta={tau_b} produced identical thresholds "
            f"({final_a}) — the parameter is not wired into the dynamics"
        )

    def test_faster_adaptation_stays_closer_to_baseline(self):
        """Smaller tau_theta = stronger homeostatic restoration toward theta_0."""
        fast = simulate(tau_theta=5.0)[-1]
        slow = simulate(tau_theta=60.0)[-1]
        assert abs(fast - DEFAULT_THETA_0) < abs(
            slow - DEFAULT_THETA_0
        ), "Faster adaptation should hold the threshold nearer baseline"

    def test_monotonic_in_tau_theta(self):
        """Displacement from baseline should grow monotonically with tau_theta."""
        displacements = [abs(simulate(tau_theta=t)[-1] - DEFAULT_THETA_0) for t in (5.0, 15.0, 30.0, 60.0)]
        assert displacements == sorted(displacements), f"Displacement not monotonic in tau_theta: {displacements}"


class TestOtherParametersAreLive:
    """The remaining exposed sliders must also influence the simulation."""

    def test_tau_s_affects_threshold(self):
        """Signal timescale feeds the threshold via metabolic coupling."""
        fast_signal = simulate(tau_theta=30.0, tau_S=0.2)[-1]
        slow_signal = simulate(tau_theta=30.0, tau_S=2.0)[-1]
        assert fast_signal != pytest.approx(slow_signal, abs=1e-6)

    def test_theta_0_affects_threshold(self):
        """Baseline threshold must shift the trajectory."""
        low = simulate(tau_theta=30.0, theta_0=0.3)[-1]
        high = simulate(tau_theta=30.0, theta_0=0.7)[-1]
        assert low != pytest.approx(high, abs=1e-6)

    def test_gamma_m_affects_threshold(self):
        """Metabolic sensitivity must shift the trajectory."""
        neutral = simulate(tau_theta=30.0, gamma_M=0.0)[-1]
        coupled = simulate(tau_theta=30.0, gamma_M=-0.3)[-1]
        assert neutral != pytest.approx(coupled, abs=1e-6)


@pytest.mark.parametrize("gui_module", ["Validation_GUI.py", "Protocols_GUI.py"])
class TestGUIDelegatesToCanonicalModel:
    """No GUI may re-implement threshold dynamics locally."""

    def _source(self, gui_module):
        from pathlib import Path

        return (Path(__file__).resolve().parents[1] / gui_module).read_text(encoding="utf-8")

    def test_gui_imports_canonical_update_threshold(self, gui_module):
        """Guards against a GUI drifting from apgi_core."""
        text = self._source(gui_module)
        assert "update_threshold(" in text, f"{gui_module} no longer delegates to apgi_core.update_threshold"

    def test_gui_has_no_local_threshold_ode(self, gui_module):
        """The dead-parameter defect was a locally re-implemented ODE."""
        text = self._source(gui_module)
        assert "dtheta_dt = (params[" not in text, f"{gui_module} re-introduced a local threshold ODE"
