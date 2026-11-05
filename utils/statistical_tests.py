"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

from typing import Any, Optional, Tuple, Union

# Apply Python 3.14 NumPy compatibility patches first
try:
    from tests.python314_numpy_fixes import *  # noqa: F403,F401
except ImportError:
    pass

import numpy as np
from scipy import stats

from apgi_core.errors import MissingScientificDependencyError, UnsupportedMethodError

# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_sample_array(data: Union[np.ndarray, list], min_n: int = 2, name: str = "data") -> np.ndarray:
    """
    Validate that data is a proper sample array with sufficient samples.

    Args:
        data: Input data (array or list)
        min_n: Minimum required sample size
        name: Name of the variable for error messages

    Returns:
        Validated numpy array

    Raises:
        ValueError: If data is invalid or insufficient
    """
    if not isinstance(data, (np.ndarray, list)):
        raise ValueError(f"{name} must be an array or list, got {type(data)}")

    arr = np.asarray(data)

    if len(arr) < min_n:
        raise ValueError(
            f"{name} requires N≥{min_n}, got {len(arr)}. "
            f"Degenerate statistical tests with single-value arrays produce "
            f"mathematically meaningless results."
        )

    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")

    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains Inf values")

    return arr


def validate_paired_arrays(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    min_n: int = 2,
    name_x: str = "x",
    name_y: str = "y",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate that two arrays are paired and have sufficient samples.

    Args:
        x: First array
        y: Second array
        min_n: Minimum required sample size
        name_x: Name of first variable
        name_y: Name of second variable

    Returns:
        Tuple of validated numpy arrays

    Raises:
        ValueError: If arrays are invalid or mismatched
    """
    x_arr = validate_sample_array(x, min_n, name_x)
    y_arr = validate_sample_array(y, min_n, name_y)

    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"{name_x} and {name_y} must have same length for paired tests: " f"got {len(x_arr)} vs {len(y_arr)}"
        )

    return x_arr, y_arr


# =============================================================================
# SAFE STATISTICAL TESTS
# =============================================================================


def safe_ttest_1samp(
    data: Union[np.ndarray, list],
    popmean: float = 0.0,
    alpha: float = 0.05,
    min_n: int = 30,
) -> Tuple[float, float, bool]:
    """
    Safe one-sample t-test with proper input validation.

    Args:
        data: Sample data (must have N≥30 for meaningful results)
        popmean: Population mean under null hypothesis
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (t_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    data_arr = validate_sample_array(data, min_n, "data")

    t_stat, p_value = stats.ttest_1samp(data_arr, popmean)

    # Handle NaN p-values (can occur with zero variance)
    if np.isnan(p_value):
        p_value = 1.0  # No significant difference when variance is zero

    significant = bool(p_value < alpha)  # Convert to Python boolean

    return t_stat, p_value, significant


def safe_ttest_ind(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 30,
) -> Tuple[float, float, bool]:
    """
    Safe independent two-sample t-test with proper input validation.

    Args:
        x: First sample data
        y: Second sample data
        alpha: Significance level
        min_n: Minimum required sample size per group

    Returns:
        Tuple of (t_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr = validate_sample_array(x, min_n, "x")
    y_arr = validate_sample_array(y, min_n, "y")

    t_stat, p_value = stats.ttest_ind(x_arr, y_arr)
    significant = p_value < alpha

    return t_stat, p_value, significant


def safe_ttest_rel(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 30,
) -> Tuple[float, float, bool]:
    """
    Safe paired two-sample t-test with proper input validation.

    Args:
        x: First sample data
        y: Second sample data (must be same length as x)
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (t_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr, y_arr = validate_paired_arrays(x, y, min_n, "x", "y")

    t_stat, p_value = stats.ttest_rel(x_arr, y_arr)
    significant = p_value < alpha

    return t_stat, p_value, significant


def safe_pearsonr(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 30,
) -> Tuple[float, float, bool]:
    """
    Safe Pearson correlation with proper input validation.

    Args:
        x: First variable data
        y: Second variable data (must be same length as x)
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (correlation_coefficient, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr, y_arr = validate_paired_arrays(x, y, min_n, "x", "y")

    # Check for zero variance
    if np.var(x_arr) == 0 or np.var(y_arr) == 0:
        raise ValueError("Cannot compute correlation with zero variance in one or both arrays")

    # Convert to simple numpy arrays if needed to handle structured dtypes
    if hasattr(x_arr, "dtype") and x_arr.dtype.names:
        # Handle structured arrays by converting to simple float arrays
        x_simple = np.asarray(x_arr, dtype=float).flatten()
        y_simple = np.asarray(y_arr, dtype=float).flatten()
        corr, p_value = stats.pearsonr(x_simple, y_simple)
    else:
        # Use arrays as-is for simple dtypes
        corr, p_value = stats.pearsonr(x_arr, y_arr)
    significant = bool(p_value < alpha)  # Convert to Python boolean

    return corr, p_value, significant


def safe_spearmanr(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 10,
) -> Tuple[float, float, bool]:
    """
    Safe Spearman rank correlation with proper input validation.

    Args:
        x: First variable data
        y: Second variable data (must be same length as x)
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (correlation_coefficient, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr, y_arr = validate_paired_arrays(x, y, min_n, "x", "y")

    corr, p_value = stats.spearmanr(x_arr, y_arr)
    significant = p_value < alpha

    return corr, p_value, significant


def safe_binomtest(k: int, n: int, p: float = 0.5, alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Safe binomial test (modern scipy.stats.binomtest, not deprecated binom_test).

    Args:
        k: Number of successes
        n: Number of trials
        p: Expected probability under null hypothesis
        alpha: Significance level

    Returns:
        Tuple of (p_value, significant)

    Raises:
        ValueError: If parameters are invalid
    """
    if n < 1:
        raise ValueError(f"Number of trials n must be ≥1, got {n}")

    if k < 0 or k > n:
        raise ValueError(f"Number of successes k must be in [0, n], got {k} for n={n}")

    if p <= 0 or p >= 1:
        raise ValueError(f"Probability p must be in (0, 1), got {p}")

    # Use modern binomtest (not deprecated binom_test)
    result = stats.binomtest(k, n, p, alternative="two-sided")
    p_value = result.pvalue
    significant = p_value < alpha

    return p_value, significant


def safe_mannwhitneyu(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    alpha: float = 0.05,
    min_n: int = 20,
) -> Tuple[float, float, bool]:
    """
    Safe Mann-Whitney U test with proper input validation.

    Args:
        x: First sample data
        y: Second sample data
        alpha: Significance level
        min_n: Minimum required sample size per group

    Returns:
        Tuple of (u_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr = validate_sample_array(x, min_n, "x")
    y_arr = validate_sample_array(y, min_n, "y")

    u_stat, p_value = stats.mannwhitneyu(x_arr, y_arr, alternative="two-sided")
    significant = p_value < alpha

    return u_stat, p_value, significant


def safe_wilcoxon(
    x: Union[np.ndarray, list],
    y: Optional[Union[np.ndarray, list]] = None,
    alpha: float = 0.05,
    min_n: int = 20,
) -> Tuple[float, float, bool]:
    """
    Safe Wilcoxon signed-rank test with proper input validation.

    Args:
        x: First sample data (or differences if y is None)
        y: Second sample data (optional)
        alpha: Significance level
        min_n: Minimum required sample size

    Returns:
        Tuple of (statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    if y is None:
        x_arr = validate_sample_array(x, min_n, "x")
        stat, p_value = stats.wilcoxon(x_arr)
    else:
        x_arr, y_arr = validate_paired_arrays(x, y, min_n, "x", "y")
        stat, p_value = stats.wilcoxon(x_arr, y_arr)

    significant = p_value < alpha

    return stat, p_value, significant


def safe_anova_oneway(
    *samples: Union[np.ndarray, list], alpha: float = 0.05, min_n: int = 20
) -> Tuple[float, float, bool]:
    """
    Safe one-way ANOVA with proper input validation.

    Args:
        *samples: Variable number of sample arrays
        alpha: Significance level
        min_n: Minimum required sample size per group

    Returns:
        Tuple of (f_statistic, p_value, significant)

    Raises:
        ValueError: If data is invalid or insufficient
    """
    if len(samples) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    validated_samples = []
    for i, sample in enumerate(samples):
        validated_samples.append(validate_sample_array(sample, min_n, f"sample_{i}"))

    f_stat, p_value = stats.f_oneway(*validated_samples)
    significant = p_value < alpha

    return f_stat, p_value, significant


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================


def compute_cohens_d(x: Union[np.ndarray, list], y: Union[np.ndarray, list], min_n: int = 30) -> float:
    """
    Compute Cohen's d effect size for two samples.

    Args:
        x: First sample data
        y: Second sample data
        min_n: Minimum required sample size per group

    Returns:
        Cohen's d effect size

    Raises:
        ValueError: If data is invalid or insufficient
    """
    x_arr = validate_sample_array(x, min_n, "x")
    y_arr = validate_sample_array(y, min_n, "y")

    # Pooled standard deviation
    n1, n2 = len(x_arr), len(y_arr)
    var1, var2 = np.var(x_arr, ddof=1), np.var(y_arr, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    cohens_d = (np.mean(x_arr) - np.mean(y_arr)) / pooled_std
    return cohens_d


def compute_cliffs_delta(x: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
    """
    Compute Cliff's delta effect size for non-parametric data.

    Cliff's delta is a robust effect size measure that quantifies the degree
    of dominance between two independent samples. It represents the probability
    that a randomly selected observation from group x exceeds a randomly
    selected observation from group y, minus the reverse probability.

    Mathematical formula:
        δ = (n_greater - n_less) / (n_x * n_y)

    Where:
    - n_greater: count of pairs where x_i > y_j
    - n_less: count of pairs where x_i < y_j
    - n_x, n_y: sample sizes of groups x and y

    Interpretation:
    - δ = 0: No effect (distributions identical)
    - δ > 0: Observations in x tend to be larger than in y
    - δ < 0: Observations in x tend to be smaller than in y
    - |δ| = 1: Complete separation (all x > all y or vice versa)

    This measure is particularly useful for:
    - Non-normal distributions
    - Ordinal data
    - Small sample sizes
    - Robust effect size estimation

    Args:
        x: First sample data
        y: Second sample data

    Returns:
        Cliff's delta (range: -1 to 1)

    Raises:
        ValueError: If data is invalid
    """
    x_arr = validate_sample_array(x, min_n=2, name="x")
    y_arr = validate_sample_array(y, min_n=2, name="y")

    # Count number of x > y, x < y, x = y
    n_greater = 0
    n_less = 0
    n_equal = 0

    for xi in x_arr:
        for yi in y_arr:
            if xi > yi:
                n_greater += 1
            elif xi < yi:
                n_less += 1
            else:
                n_equal += 1

    n_total = len(x_arr) * len(y_arr)

    if n_total == 0:
        return 0.0

    # Cliff's delta formula
    cliffs_delta = (n_greater - n_less) / n_total
    return cliffs_delta


def compute_eta_squared(f_statistic: float, df_between: int, df_within: int) -> float:
    """
    Compute eta-squared from ANOVA F statistic.

    Args:
        f_statistic: F statistic from ANOVA
        df_between: Degrees of freedom between groups
        df_within: Degrees of freedom within groups

    Returns:
        Eta-squared effect size
    """
    eta_squared = (f_statistic * df_between) / (f_statistic * df_between + df_within)
    return eta_squared


# =============================================================================
# POWER ANALYSIS
# =============================================================================


SUPPORTED_POWER_TESTS = ("ttest_ind", "ttest_rel", "pearsonr")


def _require_statsmodels_power() -> Any:
    try:
        from statsmodels.stats import power as sm_power
    except ImportError as exc:  # pragma: no cover - statsmodels is a locked dependency
        raise MissingScientificDependencyError(
            "statsmodels is required for validated power/sample-size calculations; "
            "no approximate constant is substituted (APGI-REP-003)"
        ) from exc
    return sm_power


def _check_power_inputs(effect_size: float, alpha: float, test_type: str) -> None:
    if test_type not in SUPPORTED_POWER_TESTS:
        raise UnsupportedMethodError(
            f"power analysis for test_type={test_type!r} is not implemented; "
            f"supported: {', '.join(SUPPORTED_POWER_TESTS)}. Supply a design-specific simulation instead."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not np.isfinite(effect_size) or effect_size == 0:
        raise ValueError(f"effect_size must be finite and non-zero, got {effect_size}")
    if test_type == "pearsonr" and not (-1.0 < effect_size < 1.0):
        raise ValueError(f"correlation effect size must be in (-1, 1), got {effect_size}")


def _fisher_z_power(r: float, n: int, alpha: float) -> float:
    """Two-sided power for H0: rho=0 via the Fisher z approximation (Cohen 1988, ch. 3)."""
    if n <= 3:
        raise ValueError("pearsonr power requires n > 3")
    z_crit = stats.norm.ppf(1 - alpha / 2)
    delta = abs(np.arctanh(r)) * np.sqrt(n - 3)
    return float(stats.norm.cdf(delta - z_crit) + stats.norm.cdf(-delta - z_crit))


def compute_power_analysis(
    effect_size: float,
    n_per_group: int,
    alpha: float = 0.05,
    test_type: str = "ttest_ind",
) -> float:
    """
    Compute two-sided statistical power for a given effect size and sample size.

    Supported designs (APGI-REP-003): ``ttest_ind`` (Cohen's d, equal groups,
    statsmodels ``TTestIndPower``), ``ttest_rel`` (Cohen's d_z, statsmodels
    ``TTestPower``) and ``pearsonr`` (r, Fisher-z approximation).

    Raises:
        UnsupportedMethodError: for any other ``test_type``. No placeholder
            power is returned.
        MissingScientificDependencyError: if statsmodels is unavailable.
    """
    _check_power_inputs(effect_size, alpha, test_type)
    if test_type == "pearsonr":
        return _fisher_z_power(effect_size, int(n_per_group), alpha)
    sm_power = _require_statsmodels_power()
    if test_type == "ttest_ind":
        power = sm_power.TTestIndPower().power(
            effect_size=effect_size, nobs1=n_per_group, alpha=alpha, ratio=1.0, alternative="two-sided"
        )
    else:
        power = sm_power.TTestPower().power(
            effect_size=effect_size, nobs=n_per_group, alpha=alpha, alternative="two-sided"
        )
    return float(power)


def compute_required_n(
    effect_size: float,
    desired_power: float = 0.80,
    alpha: float = 0.05,
    test_type: str = "ttest_ind",
) -> int:
    """
    Smallest integer sample size (per group) reaching ``desired_power``.

    The returned N is the ceiling of the continuous solution and is verified
    so that ``compute_power_analysis(N) >= desired_power`` while N-1 falls short.

    Raises:
        UnsupportedMethodError, MissingScientificDependencyError: see
            :func:`compute_power_analysis`.
    """
    _check_power_inputs(effect_size, alpha, test_type)
    if not (0.0 < desired_power < 1.0):
        raise ValueError(f"desired_power must be in (0, 1), got {desired_power}")
    if test_type == "pearsonr":
        z_a = stats.norm.ppf(1 - alpha / 2)
        z_b = stats.norm.ppf(desired_power)
        n_cont = ((z_a + z_b) / abs(np.arctanh(effect_size))) ** 2 + 3
    else:
        sm_power = _require_statsmodels_power()
        if test_type == "ttest_ind":
            n_cont = sm_power.TTestIndPower().solve_power(
                effect_size=effect_size, power=desired_power, alpha=alpha, ratio=1.0, alternative="two-sided"
            )
        else:
            n_cont = sm_power.TTestPower().solve_power(
                effect_size=effect_size, power=desired_power, alpha=alpha, alternative="two-sided"
            )
    n = int(np.ceil(float(np.asarray(n_cont).item()) - 1e-9))
    # Guard the ceiling against approximation drift in the continuous solver.
    while compute_power_analysis(effect_size, n, alpha, test_type) < desired_power:
        n += 1
    return n


# =============================================================================
# BOOTSTRAP METHODS
# =============================================================================


def bootstrap_ci(
    data: Union[np.ndarray, list],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean",
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        statistic: Statistic to compute ("mean", "median", "std")

    Returns:
        Tuple of (statistic_value, lower_bound, upper_bound)
    """
    data_arr = validate_sample_array(data, min_n=2, name="data")

    bootstrap_stats_list = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_arr, size=len(data_arr), replace=True)

        if statistic == "mean":
            bootstrap_stats_list.append(np.mean(sample))
        elif statistic == "median":
            bootstrap_stats_list.append(np.median(sample))
        elif statistic == "std":
            bootstrap_stats_list.append(np.std(sample, ddof=1))
        else:
            bootstrap_stats_list.append(np.mean(sample))

    bootstrap_stats = np.array(bootstrap_stats_list)

    if statistic == "mean":
        stat_value = np.mean(data_arr)
    elif statistic == "median":
        stat_value = np.median(data_arr)
    elif statistic == "std":
        stat_value = np.std(data_arr, ddof=1)
    else:
        stat_value = np.mean(data_arr)

    lower = np.percentile(bootstrap_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 + ci) / 2 * 100)

    return stat_value, lower, upper


def bootstrap_one_sample_test(
    data: Union[np.ndarray, list],
    null_value: float = 0.0,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Perform one-sample test using bootstrap resampling.

    Bootstrap testing provides a non-parametric alternative to traditional
    parametric tests by resampling the observed data with replacement.
    This method is particularly valuable when:
    - Data violates normality assumptions
    - Sample sizes are small
    - Distribution shape is unknown or complex

    Method:
    1. Generate B bootstrap samples by resampling with replacement
    2. Compute statistic (mean) for each bootstrap sample
    3. Create empirical null distribution around observed statistic
    4. Calculate two-sided p-value using reflection method

    The reflection method for two-sided p-values:
    - If observed ≥ null: p = P(bootstrap ≥ 2×null - observed)
    - If observed < null: p = P(bootstrap ≤ 2×null - observed)

    This creates a symmetric confidence interval around the null hypothesis value.

    Args:
        data: Sample data to test against null hypothesis
        null_value: Null hypothesis value (default: 0.0)
        n_bootstrap: Number of bootstrap resamples (default: 1000)
        alpha: Significance level for hypothesis test (default: 0.05)

    Returns:
        Tuple of (test_statistic, p_value):
        - test_statistic: Standardized difference from null (t-like statistic)
        - p_value: Two-sided bootstrap p-value

    Note:
        - Bootstrap provides valid inference under minimal assumptions
        - P-value accuracy improves with more bootstrap samples
        - For very small samples (n < 10), consider exact tests
    """
    data_arr = validate_sample_array(data, min_n=2, name="data")

    if len(data_arr) < 2:
        return 0.0, 1.0

    observed_mean = np.mean(data_arr)
    bootstrap_means_list = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
        bootstrap_means_list.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means_list)

    # Two-sided p-value
    if observed_mean >= null_value:
        p_value = np.mean(bootstrap_means >= 2 * null_value - observed_mean)
    else:
        p_value = np.mean(bootstrap_means <= 2 * null_value - observed_mean)

    # Test statistic
    test_stat = (
        (observed_mean - null_value) / (np.std(data_arr) / np.sqrt(len(data_arr))) if np.std(data_arr) > 0 else 0.0
    )

    return test_stat, p_value


# =============================================================================
# PERMUTATION TESTS
# =============================================================================


def permutation_test(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    n_permutations: int = 10000,
    statistic: str = "mean_diff",
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Perform permutation test for two samples.

    Args:
        x: First sample data
        y: Second sample data
        n_permutations: Number of permutations
        statistic: Statistic to test ("mean_diff", "median_diff", "corr")
        alpha: Significance level

    Returns:
        Tuple of (observed_statistic, p_value, significant)
    """
    x_arr = validate_sample_array(x, min_n=2, name="x")
    y_arr = validate_sample_array(y, min_n=2, name="y")

    combined = np.concatenate([x_arr, y_arr])
    n_x = len(x_arr)

    # Compute observed statistic
    if statistic == "mean_diff":
        observed = np.mean(x_arr) - np.mean(y_arr)
    elif statistic == "median_diff":
        observed = np.median(x_arr) - np.median(y_arr)
    elif statistic == "corr":
        observed = np.corrcoef(x_arr, y_arr)[0, 1]
    else:
        observed = np.mean(x_arr) - np.mean(y_arr)

    # Permutation distribution
    perm_stats_list = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_x = combined[:n_x]
        perm_y = combined[n_x:]

        if statistic == "mean_diff":
            perm_stat = np.mean(perm_x) - np.mean(perm_y)
        elif statistic == "median_diff":
            perm_stat = np.median(perm_x) - np.median(perm_y)
        elif statistic == "corr":
            # Handle correlation with minimal samples
            if len(perm_x) >= 2 and len(perm_y) >= 2:
                try:
                    perm_stat = np.corrcoef(perm_x, perm_y)[0, 1]
                    if np.isnan(perm_stat):
                        perm_stat = 0.0  # Default to no correlation
                except (ValueError, RuntimeWarning):
                    perm_stat = 0.0
            else:
                perm_stat = 0.0
        else:
            perm_stat = np.mean(perm_x) - np.mean(perm_y)

        perm_stats_list.append(perm_stat)

    perm_stats = np.array(perm_stats_list)

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    significant = bool(p_value < alpha)  # Convert to Python boolean

    return observed, p_value, significant


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# =============================================================================


def apply_multiple_comparison_correction(
    p_values: Union[np.ndarray, list],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> dict:
    """
    Apply multiple comparison correction to control FWER or FDR.

    Args:
        p_values: Array of p-values from multiple hypothesis tests
        method: Correction method - "bonferroni", "holm", "fdr_bh" (Benjamini-Hochberg),
                or "fdr_by" (Benjamini-Yekutieli)
        alpha: Family-wise significance level (default 0.05)

    Returns:
        Dictionary containing:
        - corrected_p_values: Array of corrected p-values
        - significant: Boolean array indicating which tests are significant
        - reject: Boolean array (same as significant, for compatibility with statsmodels)
        - method: The correction method used
        - alpha: The significance level used
        - n_tests: Number of tests performed

    Raises:
        ValueError: If method is not supported or p_values is invalid

    Example:
        >>> p_values = [0.01, 0.04, 0.02, 0.06, 0.001, 0.03]
        >>> result = apply_multiple_comparison_correction(p_values, method="bonferroni")
        >>> print(result['corrected_p_values'])
        >>> print(result['significant'])
    """
    p_arr = np.asarray(p_values)

    if len(p_arr) == 0:
        raise ValueError("p_values cannot be empty")

    if np.any((p_arr < 0) | (p_arr > 1)):
        raise ValueError("All p-values must be between 0 and 1")

    n_tests = len(p_arr)
    original_shape = p_arr.shape
    p_flat = p_arr.flatten()

    # Sort p-values for methods that require ordered testing
    sorted_indices = np.argsort(p_flat)
    sorted_p = p_flat[sorted_indices]

    if method == "bonferroni":
        # Bonferroni correction: multiply each p-value by number of tests
        corrected = np.minimum(sorted_p * n_tests, 1.0)

    elif method == "holm":
        # Holm-Bonferroni step-down procedure
        # More powerful than simple Bonferroni while controlling FWER
        corrected = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected[i] = min(p * (n_tests - i), 1.0)
        # Ensure monotonicity (corrected p-values should be non-decreasing)
        corrected = np.maximum.accumulate(corrected)

    elif method in ["fdr_bh", "benjamini-hochberg"]:
        # Benjamini-Hochberg FDR control (independent/positive dependent tests)
        corrected = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected[i] = p * n_tests / (i + 1)
        # Ensure monotonicity
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected = np.minimum(corrected, 1.0)

    elif method in ["fdr_by", "benjamini-yekutieli"]:
        # Benjamini-Yekutieli FDR control (arbitrary dependence)
        # More conservative than BH, works under any dependence structure
        corrected = np.zeros_like(sorted_p)
        harmonic_sum = np.sum(1.0 / np.arange(1, n_tests + 1))
        for i, p in enumerate(sorted_p):
            corrected[i] = p * n_tests * harmonic_sum / (i + 1)
        # Ensure monotonicity
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected = np.minimum(corrected, 1.0)

    else:
        raise ValueError(
            f"Unknown correction method: {method}. " f"Supported methods: 'bonferroni', 'holm', 'fdr_bh', 'fdr_by'"
        )

    # Restore original order
    corrected_restored = np.zeros_like(corrected)
    corrected_restored[sorted_indices] = corrected
    corrected_final = corrected_restored.reshape(original_shape)

    # Determine significance
    significant = corrected_final < alpha

    return {
        "corrected_p_values": corrected_final,
        "significant": significant,
        "reject": significant,  # For compatibility with statsmodels
        "method": method,
        "alpha": alpha,
        "n_tests": n_tests,
    }


class StatisticalAnalyzer:
    """Comprehensive statistical analysis toolkit for APGI validation."""

    def __init__(self):
        """Initialize the statistical analyzer."""
        pass

    def analyze_classification_results(self, y_true, y_pred, class_names=None) -> dict:
        """Analyze classification results with comprehensive metrics."""
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Statistical tests
        try:
            from scipy import stats

            # McNemar's test for overall accuracy
            n_total = len(y_true)
            n_correct = int(accuracy * n_total)
            k = 2  # binary classification
            mcnemar_stat = ((n_correct * k) - n_total) / (n_total * (k - 1))
            mcnemar_p = stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")

            # Chi-square test for classification (only if sample size is adequate)
            try:
                chi2_stat = stats.chi2_contingency(cm, correction=True)
                chi2_p = chi2_stat.pvalue
            except ValueError:
                # Fall back to Fisher's exact test for small samples
                chi2_stat = None
                chi2_p = None

        except ImportError:
            mcnemar_stat = mcnemar_p = chi2_stat = None

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "auc_roc": auc,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "confusion_matrix": cm,
            "mcnemar_test": {"statistic": mcnemar_stat, "p_value": mcnemar_p} if mcnemar_stat is not None else None,
            "chi2_test": {"statistic": chi2_stat, "p_value": chi2_p} if chi2_stat is not None else None,
            "class_names": class_names,
            "n_samples": len(y_true),
            "n_features": y_pred.shape[1] if len(y_pred.shape) > 1 else 1,
        }
