import numpy as np
import pytest

from apgi_core.equations import (
    CoreIgnitionSystem,
    DerivedQuantities,
    DynamicalSystemEquations,
    FoundationalEquations,
    RunningStatistics,
)


def test_prediction_error():
    assert FoundationalEquations.prediction_error(5.0, 3.0) == 2.0
    assert FoundationalEquations.prediction_error(-1.0, 1.0) == -2.0


def test_z_score():
    assert FoundationalEquations.z_score(2.0, 0.0, 1.0) == 2.0
    # Zero or near-zero standard deviation should return 0.0
    assert FoundationalEquations.z_score(2.0, 1.0, 0.0) == 0.0
    assert FoundationalEquations.z_score(2.0, 1.0, 1e-11) == 0.0


def test_precision():
    assert FoundationalEquations.precision(2.0) == 0.5
    assert FoundationalEquations.precision(0.5) == 2.0
    # Variance <= 0 should cap at 1e6
    assert FoundationalEquations.precision(0.0) == 1e6
    assert FoundationalEquations.precision(-1.0) == 1e6


def test_accumulated_signal():
    Pi_e = 2.0
    eps_e = 0.5
    Pi_i_eff = 1.0
    eps_i = 1.0
    # Canonical (APGI-SCI-001): 2.0 * |0.5| + 1.0 * |1.0| = 2.0
    assert CoreIgnitionSystem.accumulated_signal(Pi_e, eps_e, Pi_i_eff, eps_i) == 2.0
    assert CoreIgnitionSystem.accumulated_signal(Pi_e, -eps_e, Pi_i_eff, -eps_i) == 2.0
    # Legacy quadratic form is retained under an explicit non-canonical name
    assert CoreIgnitionSystem.quadratic_surprise_signal(Pi_e, eps_e, Pi_i_eff, eps_i) == 0.75


def test_ignition_strict_inequality():
    assert CoreIgnitionSystem.ignites(1.0000001, 1.0)
    assert not CoreIgnitionSystem.ignites(1.0, 1.0)
    assert not CoreIgnitionSystem.ignites(0.9999999, 1.0)


def test_effective_interoceptive_precision():
    # Pi_i_baseline * (1.0 + beta * sigmoid_M)
    res = CoreIgnitionSystem.effective_interoceptive_precision(1.0, 0.0, 0.5, 0.0)
    # sigmoid(0) = 0.5. 1.0 * (1.0 + 0.5 * 0.5) = 1.25
    assert np.isclose(res, 1.25)


def test_ignition_probability():
    # z = alpha * (S - theta)
    # S = 5.0, theta = 3.0, alpha = 1.0 -> z = 2.0
    res = CoreIgnitionSystem.ignition_probability(5.0, 3.0, 1.0)
    assert np.isclose(res, 1.0 / (1.0 + np.exp(-2.0)))

    # z = -2.0
    res2 = CoreIgnitionSystem.ignition_probability(1.0, 3.0, 1.0)
    assert np.isclose(res2, np.exp(-2.0) / (1.0 + np.exp(-2.0)))


def test_signal_dynamics():
    rng = np.random.default_rng(42)
    S_new = DynamicalSystemEquations.signal_dynamics(
        S=1.0,
        Pi_e=1.0,
        eps_e=1.0,
        Pi_i_eff=1.0,
        eps_i=1.0,
        tau_S=0.5,
        sigma_S=0.1,
        dt=0.01,
        rng=rng,
    )
    assert S_new >= 0.0


def test_threshold_dynamics():
    rng = np.random.default_rng(42)
    theta_new = DynamicalSystemEquations.threshold_dynamics(
        theta=1.0,
        theta_0_sleep=0.5,
        theta_0_alert=1.5,
        A=0.8,
        gamma_M=0.1,
        M=1.0,
        lambda_S=0.05,
        S=2.0,
        tau_theta=30.0,
        sigma_theta=0.1,
        dt=0.01,
        rng=rng,
    )
    assert theta_new >= 0.01


def test_somatic_marker_dynamics():
    rng = np.random.default_rng(42)
    M_new = DynamicalSystemEquations.somatic_marker_dynamics(
        M=0.0,
        eps_i=1.0,
        beta_M=1.0,
        gamma_context=0.2,
        C=1.0,
        tau_M=1.0,
        sigma_M=0.1,
        dt=0.01,
        rng=rng,
    )
    assert -2.0 <= M_new <= 2.0


def test_arousal_dynamics():
    rng = np.random.default_rng(42)
    A_new = DynamicalSystemEquations.arousal_dynamics(A=0.5, A_target=0.8, tau_A=10.0, sigma_A=0.05, dt=0.01, rng=rng)
    assert 0.0 <= A_new <= 1.0


def test_compute_arousal_target():
    A_target = DynamicalSystemEquations.compute_arousal_target(t=10.0, max_eps=2.0, eps_i_history=[0.5, 1.0, 1.5])
    assert 0.0 <= A_target <= 1.0

    A_target_empty = DynamicalSystemEquations.compute_arousal_target(t=10.0, max_eps=2.0, eps_i_history=[])
    assert 0.0 <= A_target_empty <= 1.0


def test_precision_dynamics():
    rng = np.random.default_rng(42)
    Pi_new = DynamicalSystemEquations.precision_dynamics(
        Pi=1.0, Pi_target=2.0, alpha_Pi=0.1, sigma_Pi=0.05, dt=0.01, rng=rng
    )
    assert Pi_new >= 0.01


def test_running_statistics():
    stats = RunningStatistics()
    assert stats.get_z_score(1.0) == 0.0

    mu, std = stats.update(2.0, dt=1.0)
    assert stats._n_updates == 1
    assert std >= 0.1  # variance is max(0.01, ...) -> std >= 0.1

    z = stats.get_z_score(3.0)
    assert z > 0


def test_latency_to_ignition():
    t_star = DerivedQuantities.latency_to_ignition(S_0=1.0, theta=2.0, I=5.0, tau_S=0.5)
    assert t_star >= 0.0

    # No ignition possible
    t_star_inf = DerivedQuantities.latency_to_ignition(S_0=1.0, theta=2.0, I=3.0, tau_S=0.5)
    assert t_star_inf == float("inf")

    # Already at steady state
    t_star_zero = DerivedQuantities.latency_to_ignition(S_0=2.5, theta=2.0, I=5.0, tau_S=0.5)
    assert t_star_zero == 0.0


def test_metabolic_cost():
    history = np.array([1.0, 2.0, 3.0])
    cost = DerivedQuantities.metabolic_cost(history, dt=0.1)
    assert cost > 0.0

    cost_truncated = DerivedQuantities.metabolic_cost(history, dt=0.1, T_ignition=0.25)
    assert cost_truncated > 0.0


def test_hierarchical_level_dynamics():
    with pytest.raises(ValueError, match="out of range"):
        DerivedQuantities.hierarchical_level_dynamics(
            level=6,
            S=1.0,
            theta=2.0,
            Pi_e=1.0,
            Pi_i=1.0,
            eps_e=1.0,
            eps_i=1.0,
            tau=1.0,
            beta_cross=0.1,
            B_higher=0.5,
        )

    S_new, theta_new, Pi_modulated = DerivedQuantities.hierarchical_level_dynamics(
        level=1,
        S=1.0,
        theta=2.0,
        Pi_e=1.0,
        Pi_i=1.0,
        eps_e=1.0,
        eps_i=1.0,
        tau=1.0,
        beta_cross=0.1,
        B_higher=0.5,
    )
    assert S_new >= 0.0
    assert theta_new >= 0.01
    assert Pi_modulated == 1.0 * (1.0 + 0.1 * 0.5)
