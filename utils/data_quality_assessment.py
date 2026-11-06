#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

from typing import Dict

# Apply Python 3.14 NumPy compatibility patches first
try:
    from tests.python314_numpy_fixes import *  # noqa: F403,F401
except ImportError:
    pass

import numpy as np
import pandas as pd

# Try to import logging config
try:
    from utils.logging_config import apgi_logger as logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]


class DataQualityAssessor:
    """Assesses data quality metrics for APGI validation"""

    def __init__(self):
        self.quality_metrics = {}

    def assess_dataset_quality(self, data: pd.DataFrame) -> Dict:
        """
        Assess overall quality of a dataset

        Args:
            data: DataFrame with experimental data

        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "completeness": self._check_completeness(data),
            "consistency": self._check_consistency(data),
            "validity": self._check_validity(data),
            "overall_score": 0.0,
        }

        # Calculate overall score
        metrics["overall_score"] = np.mean(  # type: ignore[assignment]
            [metrics["completeness"], metrics["consistency"], metrics["validity"]]
        )

        self.quality_metrics = metrics
        return metrics

    def _check_completeness(self, data: pd.DataFrame) -> float:
        """Check data completeness (percentage of non-null values)"""
        if data.empty:
            return 0.0

        total_cells = data.size
        non_null_cells = data.notna().sum().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0

    #: Optional per-column valid ranges, e.g. {"rt_ms": (100, 3000)}. Columns
    #: without a declared range are checked for finiteness only.
    valid_ranges: Dict[str, tuple] = {}

    def _check_consistency(self, data: pd.DataFrame) -> float:
        """Fraction of rows passing structural consistency checks (S4-VAL-12.2).

        A row is inconsistent when it duplicates another row exactly, or when a
        monotonic time column (``timestamp``/``time``/``onset``) goes backwards.
        Measured, never assumed: an empty frame scores 0.0.
        """
        if data.empty:
            return 0.0
        bad = data.duplicated(keep="first").to_numpy().copy()
        for col in ("timestamp", "time", "onset"):
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                backwards = data[col].diff().to_numpy() < 0
                bad |= np.nan_to_num(backwards, nan=False).astype(bool)
        return float(1.0 - bad.mean())

    def _check_validity(self, data: pd.DataFrame) -> float:
        """Fraction of numeric cells that are finite and inside any declared range."""
        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            return 0.0 if data.empty else 1.0
        values = numeric.to_numpy(dtype=float)
        ok = np.isfinite(values)
        for j, col in enumerate(numeric.columns):
            if col in self.valid_ranges:
                lo, hi = self.valid_ranges[col]
                with np.errstate(invalid="ignore"):
                    ok[:, j] &= (values[:, j] >= lo) & (values[:, j] <= hi)
        return float(ok.mean())

    def generate_quality_report(self) -> str:
        """Generate text report of quality assessment"""
        if not self.quality_metrics:
            return "No quality assessment performed"

        report = "Data Quality Assessment Report\n"
        report += "=" * 40 + "\n"

        for metric, value in self.quality_metrics.items():
            if isinstance(value, float):
                report += f"{metric}: {value:.2%}\n"
            else:
                report += f"{metric}: {value}\n"

        return report


# Module-level functions
def assess_data_quality(data: pd.DataFrame) -> Dict:
    """Convenience function for data quality assessment"""
    assessor = DataQualityAssessor()
    return assessor.assess_dataset_quality(data)


def generate_quality_report(data: pd.DataFrame) -> str:
    """Convenience function for quality report generation"""
    assessor = DataQualityAssessor()
    assessor.assess_dataset_quality(data)
    return assessor.generate_quality_report()
