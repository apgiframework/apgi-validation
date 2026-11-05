"""Registered K-gate analysis estimators (S4-PSY-*, S7-KC-*).

These functions produce the :class:`utils.kgate.Estimate` objects that the
K-gate classifier consumes. They are verified against published reference
values and closed-form cases in ``tests/test_psychometrics.py``.

Status: the *code* is complete; every registered result remains BLOCKED until
the frozen, consented participant / retest / CFA / holdout datasets exist
(S4-REC-01, S4-REC-02). Nothing here may be run on synthetic data and reported
as a K-gate outcome — the classifier refuses non-empirical inputs.

K4/K4b require fitted CFA models. No SEM engine is in the locked dependency
set, so the K4 helpers take fit statistics (χ², df, log-likelihood, loadings)
from the registered estimator and apply the registered criteria; choosing and
locking the estimator family is an open owner decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize, stats

from utils.kgate import Estimate

# ---------------------------------------------------------------------------
# K1 / K2: ICC(2,1) — two-way random effects, absolute agreement, single rater
# ---------------------------------------------------------------------------


def icc_2_1(data: np.ndarray, confidence: float = 0.95) -> Estimate:
    """ICC(2,1) (Shrout & Fleiss 1979; McGraw & Wong 1996, case 2A) with F-based CI.

    Args:
        data: ``n_subjects × k_occasions`` array without missing values.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("ICC(2,1) needs an n×k matrix with n ≥ 2 and k ≥ 2")
    if not np.all(np.isfinite(x)):
        raise ValueError("ICC(2,1) input contains missing/non-finite values")
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * np.sum((x.mean(axis=1) - grand) ** 2)
    ss_cols = n * np.sum((x.mean(axis=0) - grand) ** 2)
    ss_total = np.sum((x - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

    # McGraw & Wong (1996) confidence interval for ICC(A,1).
    a = k * icc / (n * (1 - icc))
    b = 1 + k * icc * (n - 1) / (n * (1 - icc))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    alpha = 1 - confidence
    f_hi = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f_lo = stats.f.ppf(1 - alpha / 2, v, n - 1)
    lower = n * (msr - f_hi * mse) / (f_hi * (k * msc + (k * n - k - n) * mse) + n * msr)
    upper = n * (f_lo * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f_lo * msr)
    return Estimate(float(icc), float(lower), float(upper), confidence)


# ---------------------------------------------------------------------------
# K3: ordinal ω (polychoric correlations → one-factor loadings → ω)
# ---------------------------------------------------------------------------


def _thresholds(x: np.ndarray, categories: Sequence[int]) -> np.ndarray:
    counts = np.array([np.sum(x == c) for c in categories], dtype=float)
    cum = np.cumsum(counts)[:-1] / len(x)
    cum = np.clip(cum, 1e-6, 1 - 1e-6)
    return np.concatenate([[-np.inf], stats.norm.ppf(cum), [np.inf]])


def polychoric_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Two-step ML polychoric correlation (Olsson 1979) for two ordinal items."""
    x, y = np.asarray(x), np.asarray(y)
    cx, cy = np.unique(x), np.unique(y)
    if len(cx) < 2 or len(cy) < 2:
        raise ValueError("each item needs at least two observed categories")
    tx, ty = _thresholds(x, cx), _thresholds(y, cy)
    table = np.array([[np.sum((x == a) & (y == b)) for b in cy] for a in cx], dtype=float)
    big = 8.0
    tx_c, ty_c = np.clip(tx, -big, big), np.clip(ty, -big, big)

    def cell_probs(rho: float) -> np.ndarray:
        mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
        cdf = np.array([[mvn.cdf([a, b]) for b in ty_c] for a in tx_c])
        p = cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]
        return np.clip(p, 1e-12, None)

    res = optimize.minimize_scalar(
        lambda r: -np.sum(table * np.log(cell_probs(r))), bounds=(-0.999, 0.999), method="bounded"
    )
    return float(res.x)


def polychoric_matrix(items: np.ndarray) -> np.ndarray:
    items = np.asarray(items)
    p = items.shape[1]
    r = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            r[i, j] = r[j, i] = polychoric_correlation(items[:, i], items[:, j])
    return r


def one_factor_loadings(corr: np.ndarray, iterations: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Principal-axis factoring, one factor, iterated communalities."""
    r = np.array(corr, dtype=float)
    h2 = 1 - 1 / np.diag(np.linalg.pinv(r))
    for _ in range(iterations):
        reduced = r.copy()
        np.fill_diagonal(reduced, h2)
        vals, vecs = np.linalg.eigh(reduced)
        lam = vecs[:, -1] * math.sqrt(max(vals[-1], 0.0))
        new_h2 = np.clip(lam**2, 0, 0.999)
        if np.max(np.abs(new_h2 - h2)) < tol:
            break
        h2 = new_h2
    return np.abs(lam) if np.sum(lam) < 0 else lam


def omega_from_loadings(loadings: Sequence[float]) -> float:
    """McDonald's ω for a congeneric one-factor model with standardized loadings."""
    lam = np.asarray(loadings, dtype=float)
    if np.any(np.abs(lam) >= 1):
        raise ValueError("standardized loadings must be in (-1, 1)")
    s = lam.sum() ** 2
    return float(s / (s + np.sum(1 - lam**2)))


def ordinal_omega(
    items: np.ndarray, n_boot: int = 500, confidence: float = 0.95, rng: Optional[np.random.Generator] = None
) -> Estimate:
    """Ordinal ω (Zumbo et al. 2007) with percentile bootstrap interval."""
    items = np.asarray(items)
    rng = rng if rng is not None else np.random.default_rng(0)
    point = omega_from_loadings(one_factor_loadings(polychoric_matrix(items)))
    boots = []
    for _ in range(n_boot):
        sample = items[rng.integers(0, len(items), len(items))]
        try:
            boots.append(omega_from_loadings(one_factor_loadings(polychoric_matrix(sample))))
        except ValueError:
            continue
    if len(boots) < max(10, n_boot // 2):
        raise ValueError("too few valid bootstrap replicates for an ω interval")
    a = (1 - confidence) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return Estimate(point, float(lo), float(hi), confidence)


# ---------------------------------------------------------------------------
# K4 / K4b: CFA criteria from registered model fits
# ---------------------------------------------------------------------------


@dataclass
class CFAFit:
    n_factors: int
    chi2: float
    df: int
    chi2_baseline: float
    df_baseline: int
    loglik: float
    n_params: int
    n_obs: int
    srmr: float
    #: items × factors standardized loading matrix (retained items only)
    loadings: Optional[np.ndarray] = None

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * math.log(self.n_obs)

    @property
    def cfi(self) -> float:
        d_model = max(self.chi2 - self.df, 0.0)
        d_base = max(self.chi2_baseline - self.df_baseline, d_model)
        return 1.0 if d_base == 0 else 1 - d_model / d_base

    @property
    def rmsea(self) -> float:
        return math.sqrt(max(self.chi2 - self.df, 0.0) / (self.df * (self.n_obs - 1)))

    def rmsea_ci(self, confidence: float = 0.90) -> tuple:
        """Noncentral-χ² interval for RMSEA (Steiger 1990)."""

        def solve(target: float) -> float:
            def f(nc: float) -> float:
                if nc > 0:
                    return float(stats.ncx2.cdf(self.chi2, self.df, nc) - target)
                return float(stats.chi2.cdf(self.chi2, self.df) - target)

            if f(1e-9) < 0:
                return 0.0
            return optimize.brentq(f, 1e-9, max(10 * self.chi2, 100.0))

        a = (1 - confidence) / 2
        lam_lo, lam_hi = solve(1 - a), solve(a)
        den = self.df * (self.n_obs - 1)
        return math.sqrt(lam_lo / den), math.sqrt(lam_hi / den)


def k4_estimates(fits: Mapping[int, CFAFit], primary_factor: Sequence[int]) -> Dict[str, Estimate]:
    """Build K4 criterion estimates from 1-, 2- and 3-factor fits of one estimator family."""
    if set(fits) != {1, 2, 3}:
        raise ValueError("K4 requires 1-, 2- and 3-factor fits")
    three = fits[3]
    if three.loadings is None:
        raise ValueError("3-factor fit must supply its standardized loading matrix")
    lam = np.abs(np.asarray(three.loadings, dtype=float))
    idx = np.asarray(primary_factor)
    primary = lam[np.arange(len(idx)), idx]
    cross = lam.copy()
    cross[np.arange(len(idx)), idx] = 0.0
    dbic = min(fits[1].bic, fits[2].bic) - three.bic
    items_per_factor = min(int(np.sum(idx == f)) for f in range(lam.shape[1]))
    r_lo, r_hi = three.rmsea_ci()
    return {
        "K4.dbic": Estimate(dbic),
        "K4.loading": Estimate(float(primary.min())),
        "K4.cross": Estimate(float(cross.max())),
        "K4.cfi": Estimate(three.cfi),
        "K4.rmsea": Estimate(three.rmsea, r_lo, r_hi, 0.90),
        "K4.srmr": Estimate(three.srmr),
        "K4.items": Estimate(float(items_per_factor)),
    }


INVARIANCE_STEPS = ("configural", "metric", "scalar")


def invariance_level(cfi_by_step: Mapping[str, float], max_drop: float = -0.01) -> Dict[str, object]:
    """K4b: highest invariance level reached with ΔCFI ≥ −0.01 at every step."""
    reached = "none"
    deltas: Dict[str, float] = {}
    if "configural" in cfi_by_step:
        reached = "configural"
    for prev, step in zip(INVARIANCE_STEPS, INVARIANCE_STEPS[1:]):
        if step not in cfi_by_step or prev not in cfi_by_step:
            break
        deltas[step] = cfi_by_step[step] - cfi_by_step[prev]
        if deltas[step] < max_drop:
            break
        reached = step
    worst = min(deltas.values()) if deltas else float("nan")
    return {"level_reached": reached, "delta_cfi": deltas, "estimate": Estimate(worst)}


# ---------------------------------------------------------------------------
# K5: correlation with Fisher-z interval
# ---------------------------------------------------------------------------


def pearson_with_ci(x: Sequence[float], y: Sequence[float], confidence: float = 0.95) -> Estimate:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) < 4:
        raise ValueError("need ≥ 4 paired observations")
    r = float(np.corrcoef(x, y)[0, 1])
    z = math.atanh(r)
    se = 1 / math.sqrt(len(x) - 3)
    q = stats.norm.ppf(1 - (1 - confidence) / 2)
    return Estimate(r, math.tanh(z - q * se), math.tanh(z + q * se), confidence)


# ---------------------------------------------------------------------------
# K6 / K6b: holdout prediction
# ---------------------------------------------------------------------------


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def _bootstrap(stat: Callable[[np.ndarray], float], n: int, n_boot: int, rng: np.random.Generator, confidence: float):
    vals = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    a = (1 - confidence) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return float(lo), float(hi)


def delta_r2_holdout(
    y: Sequence[float],
    pred_baseline: Sequence[float],
    pred_full: Sequence[float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Estimate:
    """K6: ΔR² of full vs baseline predictions on the locked holdout, bootstrap CI."""
    y, b, f = (np.asarray(v, float) for v in (y, pred_baseline, pred_full))
    rng = rng if rng is not None else np.random.default_rng(0)
    point = _r2(y, f) - _r2(y, b)
    lo, hi = _bootstrap(lambda i: _r2(y[i], f[i]) - _r2(y[i], b[i]), len(y), n_boot, rng, confidence)
    return Estimate(point, lo, hi, confidence)


def mae_relative_change(
    y: Sequence[float],
    pred_baseline: Sequence[float],
    pred_full: Sequence[float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Estimate:
    """K6b: (MAE_full − MAE_baseline)/MAE_baseline; −0.15 means a 15 % reduction."""
    y, b, f = (np.asarray(v, float) for v in (y, pred_baseline, pred_full))
    rng = rng if rng is not None else np.random.default_rng(0)

    def stat(i: np.ndarray) -> float:
        mb = np.mean(np.abs(y[i] - b[i]))
        return float(np.mean(np.abs(y[i] - f[i])) / mb - 1)

    lo, hi = _bootstrap(stat, len(y), n_boot, rng, confidence)
    return Estimate(stat(np.arange(len(y))), lo, hi, confidence)


def top_quintile_precision(y: Sequence[float], pred: Sequence[float], confidence: float = 0.95) -> Estimate:
    """K6b: precision of predicted top quintile for the true top quintile, Wilson interval."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    k = max(1, int(round(0.2 * len(y))))
    true_top = set(np.argsort(-y)[:k])
    pred_top = np.argsort(-pred)[:k]
    hits = sum(1 for i in pred_top if i in true_top)
    p = hits / k
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    den = 1 + z**2 / k
    centre = (p + z**2 / (2 * k)) / den
    half = z * math.sqrt(p * (1 - p) / k + z**2 / (4 * k**2)) / den
    return Estimate(p, centre - half, centre + half, confidence)


# ---------------------------------------------------------------------------
# K7b: profile likelihood boundedness
# ---------------------------------------------------------------------------


def profile_likelihood_bounded(
    profile_loglik: Callable[[float], float],
    mle: float,
    lower: float,
    upper: float,
    confidence: float = 0.95,
    grid: int = 200,
) -> Dict[str, object]:
    """K7b: is the likelihood-ratio interval closed on both sides within [lower, upper]?"""
    cut = stats.chi2.ppf(confidence, 1) / 2
    peak = profile_loglik(mle)
    left = np.linspace(lower, mle, grid)
    right = np.linspace(mle, upper, grid)
    drop_left = [peak - profile_loglik(v) for v in left]
    drop_right = [peak - profile_loglik(v) for v in right]
    bounded_left = max(drop_left) >= cut
    bounded_right = max(drop_right) >= cut
    return {"bounded": bool(bounded_left and bounded_right), "left": bounded_left, "right": bounded_right}


def credible_interval(samples: Sequence[float], mass: float = 0.95) -> Estimate:
    """K8: equal-tailed credible interval from posterior draws of the interaction."""
    s = np.asarray(samples, float)
    a = (1 - mass) / 2
    lo, hi = np.quantile(s, [a, 1 - a])
    return Estimate(float(np.median(s)), float(lo), float(hi), mass)


__all__: List[str] = [
    "icc_2_1",
    "polychoric_correlation",
    "polychoric_matrix",
    "one_factor_loadings",
    "omega_from_loadings",
    "ordinal_omega",
    "CFAFit",
    "k4_estimates",
    "invariance_level",
    "pearson_with_ci",
    "delta_r2_holdout",
    "mae_relative_change",
    "top_quintile_precision",
    "profile_likelihood_bounded",
    "credible_interval",
]
