"""
Edge Case Tests for Statistical Validation Functions
============================================

Tests for edge cases, boundary conditions, and error scenarios
in statistical validation functions to ensure robustness.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.statistical_tests import (
    bootstrap_one_sample_test,
    compute_cliffs_delta,
    permutation_test,
    safe_pearsonr,
    safe_ttest_1samp,
    validate_paired_arrays,
    validate_sample_array,
)


class TestStatisticalEdgeCases:
    """Test edge cases in statistical functions."""

    def test_pearsonr_perfect_correlation(self):
        """Test Pearson correlation with perfect correlation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # Perfect positive correlation

        r, p, significant = safe_pearsonr(x, y, min_n=3)

        assert abs(r - 1.0) < 1e-10
        assert p < 0.05
        assert significant is True

    def test_pearsonr_perfect_negative_correlation(self):
        """Test Pearson correlation with perfect negative correlation."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]  # Perfect negative correlation

        r, p, significant = safe_pearsonr(x, y, min_n=3)

        assert abs(r + 1.0) < 1e-10
        assert p < 0.05
        assert significant is True

    def test_pearsonr_zero_variance_error(self):
        """Test Pearson correlation raises error with zero variance."""
        x = [1, 1, 1, 1, 1]  # Zero variance
        y = [2, 3, 4, 5, 6]

        with pytest.raises(ValueError, match="zero variance"):
            safe_pearsonr(x, y, min_n=3)

    def test_pearsonr_insufficient_samples(self):
        """Test Pearson correlation raises error with insufficient samples."""
        x = [1, 2]  # Only 2 samples, below default min_n=30
        y = [3, 4]

        with pytest.raises(ValueError, match="requires N≥30"):
            safe_pearsonr(x, y)

    def test_pearsonr_with_nan_values(self):
        """Test Pearson correlation raises error with NaN values."""
        x = [1, 2, np.nan, 4, 5]
        y = [2, 3, 4, 5, 6]

        with pytest.raises(ValueError, match="NaN values"):
            safe_pearsonr(x, y, min_n=3)

    def test_pearsonr_with_inf_values(self):
        """Test Pearson correlation raises error with infinite values."""
        x = [1, 2, np.inf, 4, 5]
        y = [2, 3, 4, 5, 6]

        with pytest.raises(ValueError, match="Inf values"):
            safe_pearsonr(x, y, min_n=3)

    def test_ttest_1samp_single_value(self):
        """Test one-sample t-test with identical values."""
        x = [5, 5, 5, 5, 5]  # All same value

        t, p, significant = safe_ttest_1samp(x, popmean=5, min_n=3)

        # Zero-variance data makes the t-statistic 0/0 — undefined, not zero.
        # NaN is the mathematically correct answer; the previous assertion
        # (abs(t) < 1e-10) demanded a finite value the statistic cannot have.
        assert np.isnan(t), f"t should be undefined for zero-variance data, got {t}"
        assert np.isnan(p) or p == 1.0  # p-value should be NaN or 1 (no difference)
        assert significant is False

    def test_ttest_1samp_insufficient_samples(self):
        """Test one-sample t-test with insufficient samples."""
        x = [1, 2]  # Below min_n=30

        with pytest.raises(ValueError, match="requires N≥30"):
            safe_ttest_1samp(x)

    def test_cliffs_delta_identical_distributions(self):
        """Test Cliff's delta with identical distributions."""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]  # Identical to x

        delta = compute_cliffs_delta(x, y)

        assert delta == 0.0

    def test_cliffs_delta_complete_separation(self):
        """Test Cliff's delta with complete separation."""
        x = [10, 11, 12, 13, 14]
        y = [1, 2, 3, 4, 5]  # All x > all y

        delta = compute_cliffs_delta(x, y)

        assert delta == 1.0

    def test_cliffs_delta_reverse_complete_separation(self):
        """Test Cliff's delta with complete reverse separation."""
        x = [1, 2, 3, 4, 5]
        y = [10, 11, 12, 13, 14]  # All x < all y

        delta = compute_cliffs_delta(x, y)

        assert delta == -1.0

    def test_cliffs_delta_minimal_samples(self):
        """Test Cliff's delta with minimal valid sample size."""
        x = [1, 2]  # Minimum 2 samples
        y = [3, 4]

        delta = compute_cliffs_delta(x, y)

        assert -1.0 <= delta <= 1.0

    def test_bootstrap_small_sample(self):
        """Test bootstrap with very small sample."""
        x = [1, 2]  # Only 2 samples

        stat, p = bootstrap_one_sample_test(x, null_value=0, n_bootstrap=100)

        assert isinstance(stat, (int, float))
        assert 0.0 <= p <= 1.0

    def test_bootstrap_with_null_variance(self):
        """Test bootstrap with zero variance data."""
        x = [5, 5, 5, 5, 5]  # Zero variance

        stat, p = bootstrap_one_sample_test(x, null_value=5, n_bootstrap=100)

        assert stat == 0.0  # No difference from null
        assert p == 1.0  # No evidence against null

    def test_permutation_test_minimal_samples(self):
        """Test permutation test with minimal samples."""
        x = [1, 2]  # Minimum 2 samples
        y = [3, 4]

        observed, p, significant = permutation_test(x, y, n_permutations=100)

        assert isinstance(observed, (int, float))
        assert 0.0 <= p <= 1.0
        assert isinstance(significant, bool)

    def test_validate_sample_array_edge_cases(self):
        """Test sample array validation with edge cases."""
        # Test with list input
        arr = validate_sample_array([1, 2, 3], min_n=2, name="test")
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3

        # Test with numpy array input
        np_arr = np.array([1, 2, 3])
        arr2 = validate_sample_array(np_arr, min_n=2, name="test2")
        assert np.array_equal(arr, arr2)

    def test_validate_sample_array_invalid_inputs(self):
        """Test sample array validation rejects invalid inputs."""
        # Test insufficient samples
        with pytest.raises(ValueError, match="requires N≥2"):
            validate_sample_array([1], min_n=2, name="test")

        # Test with NaN values
        with pytest.raises(ValueError, match="NaN values"):
            validate_sample_array([1, 2, np.nan], min_n=2, name="test")

        # Test with infinite values
        with pytest.raises(ValueError, match="Inf values"):
            validate_sample_array([1, 2, np.inf], min_n=2, name="test")

        # A one-element list fails the N>=2 check before any dtype inspection,
        # so that is the error actually raised.
        with pytest.raises(ValueError, match="N≥2"):
            validate_sample_array(["not_an_array"], min_n=2, name="test")

        # Non-numeric contents in an otherwise long-enough array.
        with pytest.raises((ValueError, TypeError)):
            validate_sample_array(["a", "b", "c"], min_n=2, name="test")

    def test_validate_paired_arrays_mismatched_lengths(self):
        """Test paired array validation with mismatched lengths."""
        x = [1, 2, 3]
        y = [4, 5]  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            validate_paired_arrays(x, y, min_n=2, name_x="x", name_y="y")

    def test_pearsonr_mismatched_lengths(self):
        """Test Pearson correlation with mismatched array lengths."""
        x = [1, 2, 3, 4, 5]
        y = [2, 3, 4]  # Shorter

        with pytest.raises(ValueError, match="must have same length"):
            safe_pearsonr(x, y, min_n=3)

    def test_statistical_functions_with_extreme_values(self):
        """Test statistical functions handle extreme numeric values."""
        # Very large values
        x_large = [1e10, 2e10, 3e10, 4e10, 5e10]
        y_large = [2e10, 4e10, 6e10, 8e10, 1e11]

        r, p, significant = safe_pearsonr(x_large, y_large, min_n=3)
        assert abs(r - 1.0) < 1e-10  # Should still detect perfect correlation
        assert not np.isnan(r)
        assert not np.isnan(p)

        # Very small values
        x_small = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        y_small = [2e-10, 4e-10, 6e-10, 8e-10, 1e-9]

        r2, p2, significant2 = safe_pearsonr(x_small, y_small, min_n=3)
        assert abs(r2 - 1.0) < 1e-10
        assert not np.isnan(r2)
        assert not np.isnan(p2)

    def test_statistical_functions_numeric_stability(self):
        """Test numerical stability with near-equal values."""
        # Values that are very close but not identical
        x = [1.0000001, 1.0000002, 1.0000003, 1.0000004, 1.0000005]
        y = [1.0000002, 1.0000003, 1.0000004, 1.0000005, 1.0000006]

        r, p, significant = safe_pearsonr(x, y, min_n=3)

        # Should detect very high correlation but not perfect
        assert r > 0.999
        assert not np.isnan(r)
        assert not np.isnan(p)
        assert 0.0 <= p <= 1.0


class TestBoundaryConditions:
    """Test boundary conditions in validation functions."""

    def test_exact_boundary_sample_sizes(self):
        """Test functions work exactly at minimum sample size boundaries."""
        # Test with exactly minimum samples
        x = [1, 2]  # Exactly min_n=2
        y = [3, 4]

        # These should work without error
        delta = compute_cliffs_delta(x, y)
        assert -1.0 <= delta <= 1.0

        observed, p, sig = permutation_test(x, y, n_permutations=10)
        assert isinstance(observed, (int, float))
        assert 0.0 <= p <= 1.0
        assert isinstance(sig, bool)

    def test_alpha_boundary_values(self):
        """Test statistical functions with alpha boundary values."""
        x = list(range(1, 35))  # 34 samples
        y = list(range(2, 36))  # 34 samples, slightly shifted

        # Test with alpha = 0 (should never be significant)
        r1, p1, sig1 = safe_pearsonr(x, y, alpha=0.0, min_n=30)
        assert sig1 is False  # Never significant with alpha=0

        # Test with alpha = 1 (should always be significant unless p-value is exactly 1)
        r2, p2, sig2 = safe_pearsonr(x, y, alpha=1.0, min_n=30)
        # With perfect correlation, p-value might be 0, but with high correlation it should be significant
        # We'll just check that the function doesn't crash and returns reasonable values
        assert isinstance(sig2, bool)

    def test_empty_array_handling(self):
        """Test proper error handling with empty arrays."""
        with pytest.raises(ValueError, match="requires N≥2"):
            validate_sample_array([], min_n=2, name="empty")

        with pytest.raises(ValueError, match="requires N≥2"):
            compute_cliffs_delta([], [1, 2, 3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
