"""Reference-value checks for the K-gate analysis estimators (S4-PSY-*, S7-KC-*)."""

import math

import numpy as np
import pytest

from utils.psychometrics import (
    CFAFit,
    credible_interval,
    delta_r2_holdout,
    icc_2_1,
    invariance_level,
    k4_estimates,
    mae_relative_change,
    omega_from_loadings,
    one_factor_loadings,
    pearson_with_ci,
    polychoric_correlation,
    profile_likelihood_bounded,
    top_quintile_precision,
)

# Shrout & Fleiss (1979), Table 2: 6 targets × 4 judges; ICC(2,1) = 0.29.
SHROUT_FLEISS = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float
)


def test_icc_matches_shrout_fleiss():
    est = icc_2_1(SHROUT_FLEISS)
    assert round(est.value, 2) == 0.29
    assert est.lower < est.value < est.upper


def test_icc_perfect_agreement_near_one():
    x = np.repeat(np.arange(10.0)[:, None], 3, axis=1) + np.array([0, 1e-9, -1e-9])
    assert icc_2_1(x).value > 0.999


def test_icc_rejects_missing():
    x = SHROUT_FLEISS.copy()
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        icc_2_1(x)


def test_omega_closed_form():
    # λ = 0.7 × 4: ω = 2.8² / (2.8² + 4·0.51) = 7.84 / 9.88
    assert math.isclose(omega_from_loadings([0.7] * 4), 7.84 / 9.88, rel_tol=1e-12)


def test_one_factor_loadings_recover_known_structure():
    lam = np.array([0.8, 0.7, 0.6, 0.5])
    r = np.outer(lam, lam)
    np.fill_diagonal(r, 1.0)
    assert np.allclose(one_factor_loadings(r), lam, atol=1e-4)


def test_polychoric_recovers_latent_correlation():
    rng = np.random.default_rng(1)
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=3000)
    x = np.digitize(z[:, 0], [-0.5, 0.3, 1.0])
    y = np.digitize(z[:, 1], [-0.8, 0.0, 0.9])
    assert abs(polychoric_correlation(x, y) - 0.6) < 0.05


def test_pearson_ci_contains_estimate():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200)
    y = 0.3 * x + rng.standard_normal(200)
    est = pearson_with_ci(x, y)
    assert est.lower < est.value < est.upper


def test_cfa_indices_and_k4():
    def fit(k, ll, params, chi2, df):
        return CFAFit(k, chi2, df, 900.0, 66, ll, params, 400, 0.04)

    fits = {1: fit(1, -5200, 24, 400, 54), 2: fit(2, -5150, 25, 300, 53), 3: fit(3, -5000, 27, 70, 51)}
    lam = np.zeros((12, 3))
    primary = np.repeat([0, 1, 2], 4)
    lam[np.arange(12), primary] = 0.6
    fits[3].loadings = lam
    est = k4_estimates(fits, primary)
    assert est["K4.dbic"].value > 10
    assert est["K4.loading"].value == pytest.approx(0.6)
    assert est["K4.cross"].value == 0.0
    assert est["K4.items"].value == 4
    assert fits[3].cfi == pytest.approx(1 - 19 / 834)
    assert fits[3].rmsea == pytest.approx(math.sqrt(19 / (51 * 399)))
    lo, hi = fits[3].rmsea_ci()
    assert lo <= fits[3].rmsea <= hi


def test_invariance_level():
    out = invariance_level({"configural": 0.95, "metric": 0.945, "scalar": 0.93})
    assert out["level_reached"] == "metric"
    assert out["delta_cfi"]["scalar"] == pytest.approx(-0.015)


def test_holdout_metrics():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(300)
    base = 0.3 * y + rng.standard_normal(300)
    full = 0.9 * y + 0.3 * rng.standard_normal(300)
    dr2 = delta_r2_holdout(y, base, full, n_boot=300)
    assert dr2.value > 0.03 and dr2.lower <= dr2.value <= dr2.upper
    mae = mae_relative_change(y, base, full, n_boot=300)
    assert mae.value < -0.15
    prec = top_quintile_precision(y, full)
    assert 0 <= prec.lower <= prec.value <= prec.upper <= 1


def test_profile_likelihood():
    quad = lambda t: -0.5 * ((t - 1.0) / 0.2) ** 2  # noqa: E731
    assert profile_likelihood_bounded(quad, 1.0, 0.0, 2.0)["bounded"]
    flat_right = lambda t: -0.5 * ((t - 1.0) / 0.2) ** 2 if t < 1 else 0.0  # noqa: E731
    assert not profile_likelihood_bounded(flat_right, 1.0, 0.0, 2.0)["bounded"]


def test_credible_interval():
    est = credible_interval(np.random.default_rng(4).normal(0.5, 0.1, 20000))
    assert est.lower > 0


def test_one_factor_loadings_preserve_mixed_signs():
    lam = np.array([-0.8, -0.7, -0.6, 0.5])
    r = np.outer(lam, lam)
    np.fill_diagonal(r, 1.0)
    out = one_factor_loadings(r)
    # Overall sign is arbitrary; the relative signs must be preserved.
    assert np.allclose(out, lam, atol=1e-4) or np.allclose(out, -lam, atol=1e-4)
