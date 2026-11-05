"""TC-SCI-01/02/03, TC-REP-02/03/05 and INT-003 failure-propagation tests."""

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower

from apgi_core import temporal_hierarchy as th
from apgi_core.engine import APGICoreSignal
from apgi_core.equations import CoreIgnitionSystem, DerivedQuantities, EnhancedSurpriseIgnitionSystem
from apgi_core.errors import ScientificComputationError, UnsupportedMethodError
from utils.statistical_tests import compute_power_analysis, compute_required_n

# ---------------------------------------------------------------------------
# APGI-SCI-001: canonical signal and strict ignition boundary
# ---------------------------------------------------------------------------

GOLDEN = [
    # (Pi_e, z_e, Pi_i_eff, z_i, expected S)
    (2.0, 0.5, 1.0, 1.0, 2.0),
    (1.0, -1.5, 0.5, 2.0, 2.5),
    (0.0, 3.0, 0.0, -3.0, 0.0),
    (1.25, 0.8, 2.0, -0.25, 1.5),
]


@pytest.mark.parametrize("pi_e,z_e,pi_i,z_i,expected", GOLDEN)
def test_signal_golden_vectors_agree_across_implementations(pi_e, z_e, pi_i, z_i, expected):
    a = CoreIgnitionSystem.accumulated_signal(pi_e, z_e, pi_i, z_i)
    b = APGICoreSignal.accumulated_signal(pi_e, z_e, pi_i, z_i)
    assert a == b
    assert math.isclose(a, expected, rel_tol=0, abs_tol=1e-12)


def test_probe_from_audit_now_matches_specification():
    # Audit probe: quadratic form gave 0.75, specification requires 2.0.
    assert CoreIgnitionSystem.accumulated_signal(2.0, 0.5, 1.0, 1.0) == 2.0


@pytest.mark.parametrize("delta,ignites", [(np.nextafter(1.0, 2.0) - 1.0, True), (0.0, False), (-1e-12, False)])
def test_equality_boundary(delta, ignites):
    assert CoreIgnitionSystem.ignites(1.0 + delta, 1.0) is ignites


# ---------------------------------------------------------------------------
# APGI-SCI-003: canonical temporal hierarchy
# ---------------------------------------------------------------------------

BOUNDARIES = [
    (0.010, "L0"),
    (0.100, "L0"),  # shared edge belongs to the lower level
    (0.1000001, "L1"),
    (3.0, "L1"),
    (60.0, "L2"),
    (1800.0, "L2"),
    (3600.0, "L3"),
    (7 * th.DAY, "L3"),
    (30 * th.DAY, "L4"),
    (10 * th.YEAR, "L4"),
]


@pytest.mark.parametrize("seconds,code", BOUNDARIES)
def test_boundaries_classify(seconds, code):
    assert th.classify_timescale(seconds).code == code


@pytest.mark.parametrize(
    "seconds", [0.0099, 3.0001, 59.9, 1800.1, 3599.0, 8 * th.DAY, 10 * th.YEAR + 1, -1.0, float("nan")]
)
def test_out_of_range_rejected(seconds):
    with pytest.raises(th.TimescaleOutOfRangeError):
        th.classify_timescale(seconds)


def test_representative_taus_lie_in_their_levels():
    for lvl in th.TEMPORAL_LEVELS:
        th.validate_level_timescale(lvl.index, lvl.representative_tau_seconds)


def test_processor_rejects_out_of_range_override():
    from apgi_core.model import HierarchicalProcessor

    with pytest.raises(th.TimescaleOutOfRangeError):
        HierarchicalProcessor(config={"level_taus": [0.1, 0.2, 0.4, 1.0, 5.0]})


# ---------------------------------------------------------------------------
# APGI-REP-003: power / sample size arithmetic, no placeholders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d,n,alpha", [(0.5, 64, 0.05), (0.8, 26, 0.05), (0.6, 100, 0.01)])
def test_ttest_ind_power_matches_statsmodels(d, n, alpha):
    expected = TTestIndPower().power(effect_size=d, nobs1=n, alpha=alpha, ratio=1.0, alternative="two-sided")
    assert math.isclose(compute_power_analysis(d, n, alpha, "ttest_ind"), expected, rel_tol=1e-6)


def test_ttest_rel_power_matches_statsmodels():
    expected = TTestPower().power(effect_size=0.7, nobs=50, alpha=0.01, alternative="two-sided")
    assert math.isclose(compute_power_analysis(0.7, 50, 0.01, "ttest_rel"), expected, rel_tol=1e-6)


def test_pearsonr_probe_no_longer_returns_placeholder():
    power = compute_power_analysis(0.01, 5, 0.05, "pearsonr")
    assert power < 0.06  # audit probe returned 0.8
    z = stats.norm.ppf(0.975)
    delta = math.atanh(0.01) * math.sqrt(2)
    assert math.isclose(power, stats.norm.cdf(delta - z) + stats.norm.cdf(-delta - z), rel_tol=1e-6)


@pytest.mark.parametrize("test_type,effect", [("ttest_ind", 0.5), ("ttest_rel", 0.4), ("pearsonr", 0.3)])
def test_required_n_is_minimal_ceiling(test_type, effect):
    n = compute_required_n(effect, 0.8, 0.05, test_type)
    assert compute_power_analysis(effect, n, 0.05, test_type) >= 0.8
    assert compute_power_analysis(effect, n - 1, 0.05, test_type) < 0.8


def test_known_sample_sizes():
    assert compute_required_n(0.5, 0.8, 0.05, "ttest_ind") == 64
    assert compute_required_n(0.3, 0.8, 0.05, "pearsonr") == 85


@pytest.mark.parametrize("fn", [compute_power_analysis, compute_required_n])
def test_unsupported_test_type_raises(fn):
    with pytest.raises(UnsupportedMethodError):
        fn(0.5, 30, 0.05, "anova") if fn is compute_power_analysis else fn(0.5, 0.8, 0.05, "anova")


# ---------------------------------------------------------------------------
# APGI-INT-003: no substituted Hurst exponent
# ---------------------------------------------------------------------------


def test_hurst_failure_propagates(monkeypatch):
    from apgi_core import full_model

    model = full_model.APGIFullDynamicModel()

    class Boom:
        @staticmethod
        def dfa(x):
            raise RuntimeError("dfa broke")

        @staticmethod
        def hurst_rs(x):
            raise RuntimeError("rs broke")

    monkeypatch.setattr(full_model, "nolds", Boom, raising=False)
    monkeypatch.setattr(full_model, "HAS_NOLDS", True)
    with pytest.raises(ScientificComputationError):
        model.compute_hurst_exponent_dfa(np.random.default_rng(0).standard_normal(500))


def test_hurst_missing_dependency_propagates(monkeypatch):
    from apgi_core import full_model

    monkeypatch.setattr(full_model, "HAS_NOLDS", False)
    with pytest.raises(ScientificComputationError, match="No default"):
        full_model.APGIFullDynamicModel().compute_hurst_exponent_dfa(np.zeros(200) + np.arange(200))


# ---------------------------------------------------------------------------
# APGI-REP-002: seeded, repeatable orchestrated dynamics
# ---------------------------------------------------------------------------


def _trajectory(seed):
    system = EnhancedSurpriseIgnitionSystem(seed=seed)
    out = []
    for i in range(50):
        state = system.step({"observed_e": np.sin(i / 5), "observed_i": 0.1}, dt=0.01)
        out.append((system.S, system.theta, system.M))
    return np.array(out)


def test_same_seed_bitwise_identical():
    a, b = _trajectory(123), _trajectory(123)
    assert a.tobytes() == b.tobytes()


def test_different_seed_differs():
    assert not np.array_equal(_trajectory(1), _trajectory(2))


# ---------------------------------------------------------------------------
# APGI-REP-005: metabolic cost runs on the pinned NumPy and has stated units
# ---------------------------------------------------------------------------


def test_metabolic_cost_trapezoid_independent():
    s = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    # Independent hand calculation: dt * (0/2 + 1 + 2 + 1 + 0/2) = 0.1 * 4 = 0.4 [signal·s]
    assert math.isclose(DerivedQuantities.metabolic_cost(s, 0.1), 0.4, rel_tol=1e-6)
