#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    np = None
    pd = None
    print(f"Warning: NumPy/Pandas not available in data_validation.py: {e}")

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


@dataclass
class ValidationConfig:
    """Configuration for data validation thresholds."""

    # Missing data thresholds
    missing_data_threshold: float = 10.0  # percentage

    # Outlier thresholds
    outlier_threshold: float = 5.0  # percentage
    outlier_zscore_threshold: float = 1.5
    quartile_q1: float = 0.25  # First quartile for IQR
    quartile_q3: float = 0.75  # Third quartile for IQR
    extreme_value_percentage_threshold: float = 0.01  # 1% threshold for extreme values

    # Signal quality thresholds
    signal_quality_threshold: float = 70.0  # percentage
    extreme_value_zscore: float = 5.0
    repeated_values_threshold: float = 0.1  # percentage of unique values
    jump_detection_threshold: float = 0.001  # percentage of jumps
    jump_std_multiplier: float = 5.0

    # Temporal consistency thresholds
    temporal_irregular_threshold: float = 5.0  # percentage
    temporal_quality_threshold: float = 80.0

    # Data range thresholds
    eeg_min_range: float = -500.0  # microvolts
    eeg_max_range: float = 500.0
    pupil_min_range: float = 1.0  # mm
    pupil_max_range: float = 10.0
    eda_min_range: float = 0.0  # microsiemens
    eda_max_range: float = 10.0
    hr_min_range: float = 30.0  # BPM
    hr_max_range: float = 200.0

    # Overall quality thresholds
    overall_poor_threshold: float = 70.0
    overall_acceptable_threshold: float = 85.0

    # Default filter parameters
    default_low_freq: float = 0.5  # Hz
    default_high_freq: float = 40.0  # Hz
    default_cutoff_freq: float = 40.0  # Hz
    default_sampling_rate: float = 1000.0  # Hz


class DataValidator:
    """Comprehensive data validation for APGI framework."""

    def __init__(self, strict_mode=False, config: ValidationConfig = None):
        self.strict_mode = strict_mode
        self.config = config or ValidationConfig()
        self.validation_results: Dict[str, Any] = {}

    def validate_file_format(self, file_path: Union[str, Path]) -> Dict:
        """Validate file format and structure."""
        file_path = Path(file_path)
        results: Dict[str, Any] = {
            "file_path": str(file_path),
            "file_exists": file_path.exists(),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "file_extension": file_path.suffix.lower(),
            "is_readable": False,
            "format_valid": False,
            "errors": [],
            "warnings": [],
        }

        if not file_path.exists():
            results["errors"].append(f"File does not exist: {file_path}")
            return results

        try:
            # Check file size to prevent DoS from large files
            file_size = file_path.stat().st_size
            max_file_size = getattr(self.config, "max_file_size_mb", 100) * 1024 * 1024  # Default 100MB
            if file_size > max_file_size:
                results["errors"].append(
                    f"File too large: {file_size / (1024 * 1024):.1f}MB exceeds limit of {max_file_size / (1024 * 1024):.0f}MB"
                )  # type: ignore[assignment]
                return results

            if file_path.suffix.lower() == ".csv":
                # Add nrows limit to prevent OOM on large files
                max_rows = getattr(self.config, "max_csv_rows", 1000000)
                df = pd.read_csv(file_path, nrows=max_rows, encoding="utf-8")
                results["is_readable"] = True
                results["format_valid"] = self._validate_csv_structure(df, results)
            elif file_path.suffix.lower() == ".json":
                # Check file size before loading to prevent memory exhaustion
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                max_size_mb = getattr(self.config, "max_json_size_mb", 50)
                if file_size_mb > max_size_mb:
                    results["errors"].append(
                        f"JSON file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
                    )  # type: ignore[assignment]
                    return results
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)
                    except RecursionError:
                        results["errors"].append(
                            "JSON file rejected: nesting depth exceeds safe limit (possible DoS)"
                        )  # type: ignore[assignment]
                        return results
                results["is_readable"] = True
                results["format_valid"] = self._validate_json_structure(data, results)
            elif file_path.suffix.lower() == ".h5" or file_path.suffix.lower() == ".hdf5":
                if not HDF5_AVAILABLE:
                    results["errors"].append("HDF5 support not available - install h5py")  # type: ignore[assignment]
                    return results
                df = self._read_hdf5_file(file_path)
                results["is_readable"] = True
                results["format_valid"] = self._validate_hdf5_structure(df, results)
            else:
                results["errors"].append(f"Unsupported file format: {file_path.suffix}")  # type: ignore[assignment]
        except (
            FileNotFoundError,
            PermissionError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as e:
            results["errors"].append(f"Error reading file: {type(e).__name__}: {e}")  # type: ignore[assignment]

        return results

    def _validate_csv_structure(self, df: pd.DataFrame, results: Dict) -> bool:
        """Validate CSV DataFrame structure."""
        required_columns = ["timestamp", "EEG_Cz", "pupil_diameter", "eda"]
        # Note: optional_columns defined but not used in current implementation
        # optional_columns = [
        #     "eeg_pz",
        #     "heart_rate",
        #     "event_marker",
        #     "subject_id",
        #     "session_id",
        # ]

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results["errors"].append(f"Missing required columns: {missing_columns}")
            return False

        # Check data types
        for col in required_columns:
            if col == "timestamp":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        results["errors"].append(f"Column {col} cannot be converted to datetime")
                        return False
            else:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        pd.to_numeric(df[col], errors="coerce")
                    except (ValueError, TypeError):
                        results["errors"].append(f"Column {col} cannot be converted to numeric")
                        return False

        # Check for missing values
        missing_counts = df[required_columns].isnull().sum()
        if missing_counts.any():
            results["warnings"].append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")

        # Check data ranges using configurable thresholds
        self._validate_data_ranges(df, results)

        return len(results["errors"]) == 0

    def _validate_json_structure(self, data: Dict, results: Dict) -> bool:
        """Validate JSON data structure."""
        if not isinstance(data, dict):
            results["errors"].append("JSON root must be an object")
            return False

        # Check for required sections
        if "metadata" not in data:
            results["errors"].append("Missing 'metadata' section")
            return False

        if "data" not in data:
            results["errors"].append("Missing 'data' section")
            return False

        # Validate metadata
        metadata = data["metadata"]
        required_metadata = ["subject_id", "session_id", "sampling_rate", "duration"]
        for field in required_metadata:
            if field not in metadata:
                results["errors"].append(f"Missing metadata field: {field}")

        # Check first record structure
        data_records = data["data"]
        first_record = data_records[0]
        required_fields = ["timestamp", "EEG_Cz", "pupil_diameter", "eda"]
        for field in required_fields:
            if field not in first_record:
                results["errors"].append(f"Missing data field: {field}")

        return len(results["errors"]) == 0

    def _validate_data_ranges(self, df: pd.DataFrame, results: Dict):
        """Validate physiological data ranges."""
        # EEG ranges (microvolts)
        eeg_cols = [col for col in df.columns if col.lower().startswith("eeg")]
        for col in eeg_cols:
            if col in df.columns:
                eeg_data = df[col].dropna()
                if len(eeg_data) > 0:
                    eeg_min, eeg_max = eeg_data.min(), eeg_data.max()
                    if eeg_min < self.config.eeg_min_range or eeg_max > self.config.eeg_max_range:
                        results["warnings"].append(f"{col}: Unusual EEG range ({eeg_min:.2f} to {eeg_max:.2f} μV)")

        # Pupil diameter (mm)
        if "pupil_diameter" in df.columns:
            pupil_data = df["pupil_diameter"].dropna()
            if len(pupil_data) > 0:
                pupil_min, pupil_max = pupil_data.min(), pupil_data.max()
                if pupil_min < self.config.pupil_min_range or pupil_max > self.config.pupil_max_range:
                    results["warnings"].append(f"Pupil diameter: Unusual range ({pupil_min:.2f} to {pupil_max:.2f} mm)")

        # EDA (microsiemens)
        if "eda" in df.columns:
            eda_data = df["eda"].dropna()
            if len(eda_data) > 0:
                eda_min, eda_max = eda_data.min(), eda_data.max()
                if eda_min < self.config.eda_min_range or eda_max > self.config.eda_max_range:
                    results["warnings"].append(f"EDA: Unusual range ({eda_min:.3f} to {eda_max:.3f} μS)")

        # Heart rate (BPM)
        if "heart_rate" in df.columns:
            hr_data = df["heart_rate"].dropna()
            if len(hr_data) > 0:
                hr_min, hr_max = hr_data.min(), hr_data.max()
                if hr_min < self.config.hr_min_range or hr_max > self.config.hr_max_range:
                    results["warnings"].append(f"Heart rate: Unusual range ({hr_min:.1f} to {hr_max:.1f} BPM)")

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics."""
        quality_metrics: Dict[str, Any] = {
            "total_samples": len(df),
            "missing_data": {},
            "outliers": {},
            "signal_quality": {},
            "temporal_consistency": {},
            "overall_score": 0.0,
        }

        # Missing data analysis
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                missing_count = df[col].isnull().sum()
                missing_percent = (missing_count / len(df)) * 100
                quality_metrics["missing_data"][col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_percent),
                }

        # Outlier detection using configurable threshold
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(self.config.quartile_q1)
                    Q3 = data.quantile(self.config.quartile_q3)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.outlier_zscore_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_zscore_threshold * IQR

                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    quality_metrics["outliers"][col] = {
                        "count": len(outliers),
                        "percentage": (len(outliers) / len(data)) * 100,
                    }

        # Signal quality metrics
        if "EEG_Cz" in df.columns:
            quality_metrics["signal_quality"]["EEG_Cz"] = self._assess_signal_quality(df["EEG_Cz"])

        if "pupil_diameter" in df.columns:
            quality_metrics["signal_quality"]["pupil_diameter"] = self._assess_signal_quality(df["pupil_diameter"])

        if "eda" in df.columns:
            quality_metrics["signal_quality"]["eda"] = self._assess_signal_quality(df["eda"])

        # Temporal consistency
        if "timestamp" in df.columns:
            quality_metrics["temporal_consistency"] = self._assess_temporal_consistency(df)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_metrics)
        quality_metrics["quality_score"] = quality_score
        quality_metrics["overall_score"] = quality_score  # Backward compatibility

        return quality_metrics

    def _assess_signal_quality(self, signal: pd.Series) -> Dict[str, Any]:
        """Assess quality of a physiological signal."""
        signal_clean = signal.dropna()
        if len(signal_clean) == 0:
            return {"score": 0.0, "issues": ["No valid data"]}

        issues = []
        score = 100.0

        # Check for flat segments using configurable threshold
        if len(signal_clean.unique()) < len(signal_clean) * self.config.repeated_values_threshold:
            issues.append("Many repeated values")
            score -= 20

        # Check for extreme values
        signal_mean = signal_clean.mean()
        signal_std = signal_clean.std()

        if signal_std == 0:
            issues.append("No signal variation")
            score -= 30
        else:
            z_scores = np.abs((signal_clean - signal_mean) / signal_std)
            extreme_count = (z_scores > self.config.extreme_value_zscore).sum()
            if extreme_count > len(signal_clean) * self.config.extreme_value_percentage_threshold:
                issues.append(f"Many extreme values ({extreme_count})")
                score -= 15

        # Check for sudden jumps using configurable thresholds
        if len(signal_clean) > 1:
            diff = np.diff(signal_clean)
            jump_threshold = signal_std * self.config.jump_std_multiplier
            jumps = np.abs(diff) > jump_threshold
            if jumps.sum() > len(diff) * self.config.jump_detection_threshold:
                issues.append("Sudden jumps detected")
                score -= 10

        return {
            "score": max(0, score),
            "issues": issues,
            "mean": float(signal_mean),
            "std": float(signal_std),
            "samples": len(signal_clean),
        }

    def _assess_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess temporal consistency of the data."""
        if "timestamp" not in df.columns:
            return {"score": 0.0, "issues": ["No timestamp column"]}

        try:
            timestamps = pd.to_datetime(df["timestamp"])
            time_diffs = timestamps.diff().dropna()

            issues = []
            score = 100.0

            # Check for irregular sampling
            if len(time_diffs) > 0:
                expected_interval = time_diffs.mode().iloc[0]  # Most common interval
                irregular_diffs = time_diffs[time_diffs != expected_interval]

                if len(irregular_diffs) > len(time_diffs) * (self.config.temporal_irregular_threshold / 100):
                    issues.append(f"Irregular sampling: {len(irregular_diffs)} irregular intervals")
                    score -= 20

                # Check for gaps
                large_gaps = time_diffs[time_diffs > expected_interval * 2]
                if len(large_gaps) > 0:
                    issues.append(f"Data gaps: {len(large_gaps)} large gaps detected")
                    score -= 15

            return {
                "score": max(0, score),
                "issues": issues,
                "expected_interval": (str(expected_interval) if len(time_diffs) > 0 else "unknown"),
                "total_duration": (str(timestamps.iloc[-1] - timestamps.iloc[0]) if len(timestamps) > 1 else "unknown"),
            }

        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            ValueError,
            TypeError,
            OverflowError,
            MemoryError,
        ) as e:
            return {
                "score": 0.0,
                "issues": [f"Error processing timestamps: {type(e).__name__}: {e}"],
            }

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        score = 100.0

        # Penalize missing data using configurable threshold
        total_missing = sum(info["percentage"] for info in metrics["missing_data"].values())
        if total_missing > self.config.missing_data_threshold:
            score -= min(30, total_missing / 2)

        # Penalize outliers using configurable threshold
        total_outliers = sum(info["percentage"] for info in metrics["outliers"].values())
        if total_outliers > self.config.outlier_threshold:
            score -= min(20, total_outliers)

        # Penalize poor signal quality using configurable threshold
        signal_scores = [info["score"] for info in metrics["signal_quality"].values()]
        if signal_scores:
            avg_signal_score = sum(signal_scores) / len(signal_scores)
            if avg_signal_score < self.config.signal_quality_threshold:
                score -= (100 - avg_signal_score) * 0.3

        # Penalize temporal inconsistency using configurable threshold
        if "temporal_consistency" in metrics:
            temporal_score = metrics["temporal_consistency"]["score"]
            if temporal_score < self.config.temporal_quality_threshold:
                score -= (100 - temporal_score) * 0.2

        return max(0, score)

    def generate_validation_report(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        file_path = Path(file_path)

        report: Dict[str, Any] = {
            "file_info": self.validate_file_format(file_path),
            "data_quality": {},
            "recommendations": [],
            "validation_timestamp": datetime.now().isoformat(),
            "validator_version": "1.0.0",
        }

        # Load data for quality assessment
        if report["file_info"]["is_readable"]:  # type: ignore[index]
            try:
                if file_path.suffix.lower() == ".csv":
                    # Limit rows for large files to prevent memory exhaustion
                    max_rows_for_validation = getattr(self.config, "max_rows_for_validation", 100000)
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > 10:  # For files > 10MB, limit rows
                        df = pd.read_csv(file_path, nrows=max_rows_for_validation)
                        report["data_quality"][
                            "warning"
                        ] = f"Large file detected ({file_size_mb:.1f}MB). Validation performed on first {max_rows_for_validation} rows only."
                    else:
                        df = pd.read_csv(file_path)
                elif file_path.suffix.lower() == ".json":
                    # For JSON, check file size and warn if large
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > 50:  # JSON files are typically smaller
                        report["data_quality"][
                            "warning"
                        ] = f"Large JSON file detected ({file_size_mb:.1f}MB). Memory usage may be high."
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    df = pd.DataFrame(data["data"])
                elif file_path.suffix.lower() in [".h5", ".hdf5"]:
                    df = self._read_hdf5_file(file_path)

                report["data_quality"] = self.validate_data_quality(df)
                report["recommendations"] = self._generate_recommendations(report)

            except (
                FileNotFoundError,
                PermissionError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                json.JSONDecodeError,
                UnicodeDecodeError,
                ValueError,
                TypeError,
                MemoryError,
            ) as e:
                report["data_quality"]["error"] = f"{type(e).__name__}: {e}"

        return report

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate data improvement recommendations."""
        recommendations = []

        # File format recommendations
        if report["file_info"]["errors"]:
            recommendations.append("Fix file format errors before processing")

        # Missing data recommendations using configurable threshold
        missing_data = report["data_quality"].get("missing_data", {})
        for col, info in missing_data.items():
            if info["percentage"] > self.config.missing_data_threshold:
                recommendations.append(f"Consider imputation or removal of missing data in {col}")

        # Outlier recommendations using configurable threshold
        outliers = report["data_quality"].get("outliers", {})
        for col, info in outliers.items():
            if info["percentage"] > self.config.outlier_threshold:
                recommendations.append(f"Review outliers in {col} ({info['percentage']:.1f}%)")

        # Signal quality recommendations using configurable threshold
        signal_quality = report["data_quality"].get("signal_quality", {})
        for col, info in signal_quality.items():
            if info["score"] < self.config.signal_quality_threshold:
                recommendations.append(f"Improve signal quality for {col}: {', '.join(info['issues'])}")

        # Temporal consistency recommendations using configurable threshold
        temporal = report["data_quality"].get("temporal_consistency", {})
        if temporal.get("score", 100) < self.config.temporal_quality_threshold:
            recommendations.append(f"Address temporal consistency issues: {', '.join(temporal.get('issues', []))}")

        # Overall quality recommendations using configurable thresholds
        overall_score = report["data_quality"].get("overall_score", 100)
        if overall_score < self.config.overall_poor_threshold:
            recommendations.append("Overall data quality is poor - consider data cleaning and preprocessing")
        elif overall_score < self.config.overall_acceptable_threshold:
            recommendations.append("Data quality is acceptable but could be improved with preprocessing")

        return recommendations

    def _read_hdf5_file(self, file_path: Path) -> pd.DataFrame:
        """Read HDF5 file and convert to DataFrame."""
        try:
            with h5py.File(file_path, "r") as f:
                # Limit HDF5 traversal to prevent DoS
                max_keys = 1000  # Limit number of keys
                max_depth = 5  # Limit nesting depth

                def _count_keys(group, depth=0):
                    """Count keys with depth limit."""
                    if depth > max_depth:
                        raise ValueError("HDF5 file exceeds maximum depth limit")
                    count = 0
                    for key in group.keys():
                        count += 1
                        if count > max_keys:
                            raise ValueError("HDF5 file exceeds maximum key count")
                        if isinstance(group[key], h5py.Group):
                            count += _count_keys(group[key], depth + 1)
                    return count

                # Check key count
                _count_keys(f)

                # Check if data is stored as a dataset or group
                if "data" in f:
                    if isinstance(f["data"], h5py.Dataset):
                        # Direct dataset
                        data = f["data"][:]
                        if "columns" in f.attrs:
                            columns = f.attrs["columns"]
                            if isinstance(columns, (list, np.ndarray)):
                                columns = list(columns)
                        else:
                            # Generate column names
                            columns = [f"col_{i}" for i in range(data.shape[1])]
                        return pd.DataFrame(data, columns=columns)
                    else:
                        # Group structure - convert each dataset
                        data_dict = {}
                        for key in f["data"].keys():
                            if isinstance(f["data"][key], h5py.Dataset):
                                data_dict[key] = f["data"][key][:]
                        return pd.DataFrame(data_dict)
                else:
                    # Root level datasets
                    data_dict = {}
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            data_dict[key] = f[key][:]
                    return pd.DataFrame(data_dict)
        except (OSError, ValueError, KeyError) as e:
            raise ValueError(f"Error reading HDF5 file: {e}")

    def _validate_hdf5_structure(self, df: pd.DataFrame, results: Dict) -> bool:
        """Validate HDF5 DataFrame structure."""
        if df.empty:
            results["errors"].append("HDF5 file contains no data")
            return False

        # Check for required columns (same as CSV validation)
        required_columns = ["timestamp", "EEG_Cz", "pupil_diameter", "eda"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results["errors"].append(f"Missing required columns: {missing_columns}")
            return False

        # Validate data types
        for col in required_columns:
            if col in df.columns:
                if col == "timestamp":
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        try:
                            pd.to_datetime(df[col])
                        except (ValueError, TypeError):
                            results["errors"].append(f"Column {col} cannot be converted to datetime")
                            return False
                else:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            pd.to_numeric(df[col], errors="coerce")
                        except (ValueError, TypeError):
                            results["errors"].append(f"Column {col} cannot be converted to numeric")
                            return False

        # Check for missing values
        missing_counts = df[required_columns].isnull().sum()
        if missing_counts.any():
            results["warnings"].append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")

        # Check data ranges
        self._validate_data_ranges(df, results)

        return len(results["errors"]) == 0


class DataPreprocessor(DataValidator):
    """Data preprocessing utilities for APGI framework."""

    def __init__(self, config=None):
        super().__init__()
        self.preprocessing_steps = []
        self.config = config

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from various formats."""
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".json":
            # Check file size before loading to prevent memory exhaustion
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            max_size_mb = getattr(self.config, "max_json_size_mb", 50)
            if file_size_mb > max_size_mb:
                raise ValueError(f"JSON file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)")
            with open(file_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data["data"])
        elif file_path.suffix.lower() in [".h5", ".hdf5"]:
            if not HDF5_AVAILABLE:
                raise ValueError("HDF5 support not available - install h5py")
            df = self._read_hdf5_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Convert timestamp if present
        if "timestamp" in df.columns:
            df = df.copy()  # Ensure we're working with a copy to avoid SettingWithCopyWarning
            # Convert to datetime, handling both string and datetime inputs
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def clean_missing_data(self, df: pd.DataFrame, strategy: str = "interpolate") -> pd.DataFrame:
        """Clean missing data using various strategies."""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        if strategy == "interpolate":
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate()
        elif strategy == "forward_fill":
            df_clean[numeric_cols] = df_clean[numeric_cols].ffill()
        elif strategy == "backward_fill":
            df_clean[numeric_cols] = df_clean[numeric_cols].bfill()
        elif strategy == "mean":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == "median":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == "drop":
            df_clean = df_clean.dropna(subset=numeric_cols)

        self.preprocessing_steps.append(f"Missing data cleaned using {strategy} strategy")
        return df_clean

    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        outliers_removed = 0

        # Build a single mask to filter the dataframe only once
        combined_mask = pd.Series(False, index=df_clean.index)

        for col in numeric_cols:
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                combined_mask |= outlier_mask

            elif method == "zscore":
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outlier_mask = z_scores > threshold
                combined_mask |= outlier_mask

        outliers_removed = combined_mask.sum()
        if outliers_removed > 0:
            df_clean = df_clean[~combined_mask]

        self.preprocessing_steps.append(f"Removed {outliers_removed} outliers using {method} method")
        return df_clean

    def filter_signals(
        self,
        df: pd.DataFrame,
        filter_type: str = "bandpass",
        columns: List[str] = None,
        **filter_params,
    ) -> pd.DataFrame:
        """Apply signal filtering to specified columns."""
        from scipy import signal

        df_filtered = df.copy()

        if columns is None:
            columns = [col for col in df.columns if col.startswith("eeg") or col.startswith("pupil") or col == "eda"]

        for col in columns:
            if col in df_filtered.columns:
                data = df_filtered[col].dropna()

                if filter_type == "bandpass":
                    low_freq = filter_params.get(
                        "low_freq",
                        (getattr(self.config, "default_low_freq", 1.0) if self.config else 1.0),
                    )
                    high_freq = filter_params.get(
                        "high_freq",
                        (getattr(self.config, "default_high_freq", 50.0) if self.config else 50.0),
                    )
                    fs = filter_params.get(
                        "sampling_rate",
                        (getattr(self.config, "default_sampling_rate", 1000.0) if self.config else 1000.0),
                    )

                    nyquist = fs / 2
                    low = low_freq / nyquist
                    high = high_freq / nyquist

                    b, a = signal.butter(4, [low, high], btype="band")
                    filtered_data = signal.filtfilt(b, a, data)

                    df_filtered.loc[data.index, col] = filtered_data

                elif filter_type == "lowpass":
                    cutoff_freq = filter_params.get(
                        "cutoff_freq",
                        (getattr(self.config, "default_cutoff_freq", 50.0) if self.config else 50.0),
                    )
                    fs = filter_params.get(
                        "sampling_rate",
                        (getattr(self.config, "default_sampling_rate", 1000.0) if self.config else 1000.0),
                    )

                    nyquist = fs / 2
                    cutoff = cutoff_freq / nyquist

                    b, a = signal.butter(4, cutoff, btype="low")
                    filtered_data = signal.filtfilt(b, a, data)

                    df_filtered.loc[data.index, col] = filtered_data

        self.preprocessing_steps.append(f"Applied {filter_type} filter to {columns}")
        return df_filtered

    def normalize_data(self, df: pd.DataFrame, method: str = "zscore", columns: List[str] = None) -> pd.DataFrame:
        """Normalize data using various methods."""
        df_normalized = df.copy()

        if columns is None:
            columns = list(df_normalized.select_dtypes(include=[np.number]).columns)  # type: ignore[assignment]

        for col in columns:
            if col in df_normalized.columns:
                data = df_normalized[col]

                if method == "zscore":
                    mean = data.mean()
                    std = data.std()
                    if std == 0:
                        df_normalized.loc[:, col] = 0
                    else:
                        df_normalized.loc[:, col] = (data - mean) / std
                elif method == "minmax":
                    min_val = data.min()
                    max_val = data.max()
                    if max_val == min_val:
                        df_normalized[col] = 0.5
                    else:
                        df_normalized[col] = (data - min_val) / (max_val - min_val)
                elif method == "robust":
                    median = data.median()
                    mad = np.median(np.abs(data - median))
                    if mad == 0:
                        df_normalized[col] = 0
                    else:
                        df_normalized[col] = (data - median) / mad

        self.preprocessing_steps.append(f"Normalized {columns} using {method} method")
        return df_normalized

    def resample_data(self, df: pd.DataFrame, target_rate: float, time_column: str = "timestamp") -> pd.DataFrame:
        """Resample data to target sampling rate."""
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found")

        df_resampled = df.copy()
        df_resampled = df_resampled.set_index(time_column)

        # Calculate target frequency
        target_freq = f"{int(1000000 / target_rate)}us"

        # Resample numeric columns
        numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
        df_resampled = df_resampled[numeric_cols].resample(target_freq).mean()

        # Forward fill non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            df_non_numeric = df[non_numeric_cols].copy()
            df_non_numeric[time_column] = pd.to_datetime(df[time_column])
            df_non_numeric = df_non_numeric.set_index(time_column)
            df_non_numeric = df_non_numeric.resample(target_freq).ffill()
            df_resampled = pd.concat([df_resampled, df_non_numeric], axis=1)

        self.preprocessing_steps.append(f"Resampled data to {target_rate} Hz")
        return df_resampled.reset_index()

    def save_processed_data(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = "csv",
        include_metadata: bool = True,
    ):
        """Save processed data with metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            data_dict: dict[str, object] = {
                "data": df.to_dict("records"),
                "preprocessing_steps": self.preprocessing_steps,
                "processing_timestamp": datetime.now().isoformat(),
            }

            if include_metadata:
                data_dict["metadata"] = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported output format: {format}")


def main():
    """Demonstrate data validation and preprocessing."""
    print("APGI Framework - Data Validation & Preprocessing")
    print("=" * 50)

    # Initialize validator
    validator = DataValidator(strict_mode=True)

    # Validate sample data files
    data_dir = Path("data_repository")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))

        for csv_file in csv_files[:2]:  # Validate first 2 files
            print(f"\nValidating: {csv_file.name}")
            report = validator.generate_validation_report(csv_file)

            print(f"  File readable: {report['file_info']['is_readable']}")
            print(f"  Format valid: {report['file_info']['format_valid']}")
            print(f"  Overall quality: {report['data_quality'].get('overall_score', 'N/A'):.1f}")

            if report["recommendations"]:
                print("  Recommendations:")
                for rec in report["recommendations"][:3]:
                    print(f"    - {rec}")

    # Demonstrate preprocessing
    print("\nDemonstrating preprocessing pipeline...")

    try:
        preprocessor = DataPreprocessor()

        # Load demo data
        demo_file = data_dir / "demo_demo.csv"
        if demo_file.exists():
            df = preprocessor.load_data(demo_file)
            print(f"  Loaded data: {df.shape}")

            # Clean missing data
            df_clean = preprocessor.clean_missing_data(df, strategy="interpolate")
            print(f"  Cleaned missing data: {df_clean.shape}")

            # Remove outliers
            df_no_outliers = preprocessor.remove_outliers(df_clean, method="iqr")
            print(f"  Removed outliers: {df_no_outliers.shape}")

            # Normalize data
            df_normalized = preprocessor.normalize_data(df_no_outliers, method="zscore")
            print(f"  Normalized data: {df_normalized.shape}")

            # Save processed data
            output_file = data_dir / "demo_demo_processed.json"
            preprocessor.save_processed_data(df_normalized, output_file, format="json")
            print(f"  Saved processed data: {output_file}")

    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        TypeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        MemoryError,
    ) as e:
        print(f"  Error in preprocessing: {type(e).__name__}: {e}")

    print("\nData validation and preprocessing utilities ready!")


# =============================================================================
# P4 DATA DEPENDENCY VALIDATION (VP-14, VP-15, FP-09)
# =============================================================================


def validate_fmri_dataset(
    data_path: Union[str, Path],
    required_conditions: Optional[List[str]] = None,
    min_subjects: int = 30,
    min_tr: float = 1.5,
    max_tr: float = 3.0,
    min_trials: int = 60,
    check_nifti: bool = True,
) -> Dict[str, Any]:
    """
    Validate fMRI dataset for VP-14/VP-15 protocols.

    Args:
        data_path: Path to BIDS dataset or NPZ file
        required_conditions: List of required trial conditions (e.g., ["threat", "safe"])
        min_subjects: Minimum number of subjects required
        min_tr: Minimum acceptable TR (repetition time)
        max_tr: Maximum acceptable TR
        min_trials: Minimum trials per subject
        check_nifti: Whether to validate NIfTI files (requires nibabel)

    Returns:
        Validation report dictionary
    """
    report = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "metadata": {},
        "subjects": [],
        "n_subjects": 0,
    }

    data_path = Path(data_path)

    if not data_path.exists():
        report["errors"].append(f"Data path does not exist: {data_path}")  # type: ignore[attr-defined]
        return report

    # Check if it's a single NPZ file
    if data_path.suffix == ".npz":
        return _validate_fmri_npz(data_path, report, required_conditions, min_trials)

    # Check if it's a BIDS dataset
    if data_path.is_dir():
        return _validate_bids_fmri(
            data_path,
            report,
            required_conditions,
            min_subjects,
            min_tr,
            max_tr,
            min_trials,
            check_nifti,
        )

    report["errors"].append(  # type: ignore[attr-defined]
        f"Unsupported data format: {data_path.suffix}. Use .npz or BIDS directory."
    )
    return report


def _validate_fmri_npz(
    npz_path: Path,
    report: Dict[str, Any],
    required_conditions: Optional[List[str]],
    min_trials: int,
) -> Dict[str, Any]:
    """Validate pre-processed NPZ fMRI data."""
    try:
        data = np.load(npz_path, allow_pickle=False)

        # Check required fields
        required_fields = ["vmPFC_bold", "conditions", "dt", "trial_duration"]
        for field in required_fields:
            if field not in data:
                report["errors"].append(f"Missing required field: {field}")

        # Check data dimensions
        if "vmPFC_bold" in data:
            vmPFC = data["vmPFC_bold"]
            report["metadata"]["n_timepoints"] = len(vmPFC)

            # Estimate number of trials
            trial_duration = float(data.get("trial_duration", 12.0))
            dt = float(data.get("dt", 1.0))
            n_trials_est = len(vmPFC) / (trial_duration / dt)
            report["metadata"]["n_trials_estimated"] = int(n_trials_est)

            if n_trials_est < min_trials:
                report["warnings"].append(f"Low trial count: ~{int(n_trials_est)} (minimum {min_trials})")

        # Check conditions
        if "conditions" in data and required_conditions:
            conditions = data["conditions"].tolist()
            found_conditions = set()
            for cond in conditions:
                if isinstance(cond, dict):
                    cond_type = cond.get("trial_type", cond.get("is_threat", "unknown"))
                    found_conditions.add(str(cond_type).lower())
                elif isinstance(cond, (tuple, list)) and len(cond) > 0:
                    found_conditions.add(str(cond[0]).lower())

            for req in required_conditions:
                if req.lower() not in found_conditions:
                    report["warnings"].append(f"Condition '{req}' not found in data")

        if not report["errors"]:
            report["valid"] = True
            report["subjects"] = [npz_path.stem]
            report["n_subjects"] = 1

    except Exception as e:
        report["errors"].append(f"Error loading NPZ: {e}")

    return report


def _validate_bids_fmri(
    bids_path: Path,
    report: Dict[str, Any],
    required_conditions: Optional[List[str]],
    min_subjects: int,
    min_tr: float,
    max_tr: float,
    min_trials: int,
    check_nifti: bool,
) -> Dict[str, Any]:
    """Validate BIDS-formatted fMRI dataset."""
    # Check for participants.tsv
    participants_file = bids_path / "participants.tsv"
    if not participants_file.exists():
        report["warnings"].append("No participants.tsv found (not strict BIDS)")

    # Find subject directories
    sub_dirs = list(bids_path.glob("sub-*"))
    if len(sub_dirs) < min_subjects:
        report["warnings"].append(f"Only {len(sub_dirs)} subjects found (minimum {min_subjects} recommended)")

    report["metadata"]["n_subjects_found"] = len(sub_dirs)

    # Validate each subject
    valid_subjects = []
    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        func_dir = sub_dir / "func"

        if not func_dir.exists():
            report["warnings"].append(f"{sub_id}: No func/ directory")
            continue

        # Find bold files
        bold_files = list(func_dir.glob("*_bold.nii*"))
        if not bold_files:
            report["warnings"].append(f"{sub_id}: No BOLD files found")
            continue

        # Check for events.tsv if conditions required
        if required_conditions:
            events_files = list(func_dir.glob("*_events.tsv"))
            if not events_files:
                report["warnings"].append(f"{sub_id}: No events.tsv (required for conditions)")

        valid_subjects.append(sub_id)

    report["subjects"] = valid_subjects
    report["n_subjects"] = len(valid_subjects)

    # Check TR from a sample file if nibabel available
    if check_nifti and valid_subjects:
        try:
            import nibabel as nib

            sample_file = (
                bids_path / valid_subjects[0] / "func" / f"{valid_subjects[0]}_task-anticipation_run-01_bold.nii.gz"
            )
            if sample_file.exists():
                img = nib.load(str(sample_file))
                zooms = img.header.get_zooms()  # type: ignore[attr-defined]
                if len(zooms) >= 4:
                    tr = float(zooms[-1])
                    report["metadata"]["tr"] = tr
                    if tr < min_tr or tr > max_tr:
                        report["warnings"].append(f"TR = {tr}s outside range [{min_tr}, {max_tr}]")
        except ImportError:
            report["warnings"].append("nibabel not available - skipping NIfTI validation")
        except Exception as e:
            report["warnings"].append(f"Error reading sample NIfTI: {e}")

    if not report["errors"]:
        report["valid"] = True

    return report


def validate_doc_eeg_dataset(
    data_path: Union[str, Path],
    min_patients: int = 50,
    required_modalities: Optional[List[str]] = None,
    check_bids: bool = True,
) -> Dict[str, Any]:
    """Validate DoC EEG dataset for FP-09."""
    if required_modalities is None:
        required_modalities = ["rest", "pci", "hep"]

    report = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "metadata": {},
        "patients": [],
        "n_patients": 0,
        "diagnoses": {},
    }

    data_path = Path(data_path)

    if not data_path.exists():
        report["errors"].append(f"Data path does not exist: {data_path}")  # type: ignore[attr-defined]
        return report

    # Check if it's a single NPZ file
    if data_path.suffix == ".npz":
        return _validate_doc_npz(data_path, report, required_modalities, min_patients)

    # Check if it's a BIDS dataset
    if data_path.is_dir():
        return _validate_bids_doc(data_path, report, required_modalities, min_patients, check_bids)

    report["errors"].append(  # type: ignore[attr-defined]
        f"Unsupported data format: {data_path.suffix}. Use .npz or BIDS directory."
    )
    return report


def _validate_doc_npz(
    npz_path: Path,
    report: Dict[str, Any],
    required_modalities: List[str],
    min_patients: int,
) -> Dict[str, Any]:
    """Validate pre-processed DoC NPZ data."""
    try:
        data = np.load(npz_path, allow_pickle=False)

        # Check required fields
        required_fields = ["pci_scores", "hep_amplitudes", "consciousness_labels"]
        for field in required_fields:
            if field not in data:
                report["errors"].append(f"Missing required field: {field}")

        # Check patient count
        if "pci_scores" in data:
            n_patients = len(data["pci_scores"])
            report["metadata"]["n_patients"] = n_patients
            report["n_patients"] = n_patients

            if n_patients < min_patients:
                report["warnings"].append(f"Low patient count: {n_patients} (minimum {min_patients} recommended)")

            # Check diagnosis distribution
            if "consciousness_labels" in data:
                labels = data["consciousness_labels"]
                unique, counts = np.unique(labels, return_counts=True)
                report["diagnoses"] = {
                    "VS/UWS": int(counts[0]) if 0 in unique else 0,
                    "MCS": int(counts[1]) if 1 in unique else 0,
                    "EMCS/Conscious": int(counts[2]) if 2 in unique else 0,
                }

                # Check balance
                min_count = min(report["diagnoses"].values())
                if min_count < 10:
                    report["warnings"].append(f"Imbalanced diagnoses: minimum category has only {min_count} patients")

        if not report["errors"]:
            report["valid"] = True
            report["patients"] = [f"patient_{i}" for i in range(report["n_patients"])]

    except Exception as e:
        report["errors"].append(f"Error loading NPZ: {e}")

    return report


def _validate_bids_doc(
    bids_path: Path,
    report: Dict[str, Any],
    required_modalities: List[str],
    min_patients: int,
    check_bids: bool,
) -> Dict[str, Any]:
    """Validate BIDS-formatted DoC EEG dataset."""
    # Check for participants.tsv with clinical data
    participants_file = bids_path / "participants.tsv"
    if participants_file.exists():
        try:
            import pandas as pd

            participants = pd.read_csv(participants_file, sep="\t")
            report["metadata"]["n_participants_tsv"] = len(participants)

            # Check for diagnosis column
            if "diagnosis" in participants.columns:
                diag_counts = participants["diagnosis"].value_counts().to_dict()
                report["diagnoses"] = diag_counts
        except ImportError:
            report["warnings"].append("pandas not available - skipping participants.tsv validation")
        except Exception as e:
            report["warnings"].append(f"Error reading participants.tsv: {e}")
    else:
        report["warnings"].append("No participants.tsv found (not strict BIDS)")

    # Find subject directories
    sub_dirs = list(bids_path.glob("sub-*"))
    if len(sub_dirs) < min_patients:
        report["warnings"].append(f"Only {len(sub_dirs)} patients found (minimum {min_patients} recommended)")

    report["metadata"]["n_patients_found"] = len(sub_dirs)

    # Validate each patient
    valid_patients = []
    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        eeg_dir = sub_dir / "eeg"
        clinical_dir = sub_dir / "clinical"

        if not eeg_dir.exists():
            report["warnings"].append(f"{sub_id}: No eeg/ directory")
            continue

        # Check for required modalities
        for modality in required_modalities:
            modality_files = list(eeg_dir.glob(f"*_{modality}_*.set")) or list(eeg_dir.glob(f"*_{modality}_*.fif"))
            if not modality_files:
                report["warnings"].append(f"{sub_id}: No {modality} recording found")

        # Check for clinical data
        if clinical_dir.exists():
            crs_r_file = clinical_dir / f"{sub_id}_crs-r.tsv"
            if not crs_r_file.exists():
                report["warnings"].append(f"{sub_id}: No CRS-R scores (clinical assessment)")
        else:
            report["warnings"].append(f"{sub_id}: No clinical/ directory")

        valid_patients.append(sub_id)

    report["patients"] = valid_patients
    report["n_patients"] = len(valid_patients)

    if not report["errors"]:
        report["valid"] = True

    return report


def load_real_data_stub(protocol_id: str, data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Stub function for loading real P4 data - returns clear status when unavailable.

    Args:
        protocol_id: One of "VP-14", "VP-15", "FP-09"
        data_path: Optional path to real data

    Returns:
        Dictionary with status and data (if available)
    """
    if data_path and Path(data_path).exists():
        # Real data provided - validate and load
        if protocol_id in ["VP-14", "VP-15"]:
            validation = validate_fmri_dataset(data_path)
            if validation["valid"]:
                return {
                    "status": "empirical_loaded",
                    "protocol_id": protocol_id,
                    "validation": validation,
                    "data_path": data_path,
                }
        elif protocol_id == "FP-09":
            validation = validate_doc_eeg_dataset(data_path)
            if validation["valid"]:
                return {
                    "status": "empirical_loaded",
                    "protocol_id": protocol_id,
                    "validation": validation,
                    "data_path": data_path,
                }

    # No real data - return stub status
    status_messages = {
        "VP-14": "fMRI Anticipation - requires real clinical fMRI data",
        "VP-15": "fMRI vmPFC - requires real clinical fMRI data",
        "FP-09": "DoC Neural Signatures - requires real DoC cohort data",
    }

    return {
        "status": "data_required",
        "protocol_id": protocol_id,
        "message": status_messages.get(protocol_id, "Unknown protocol"),
        "action_required": "Collect external clinical data per docs/DATA_REQUIREMENTS.md",
        "synthetic_fallback": True,
    }


if __name__ == "__main__":
    main()
