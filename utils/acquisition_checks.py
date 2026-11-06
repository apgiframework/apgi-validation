"""Acquisition-quality checks for incoming recordings (S4-VAL-12.2).

Replaces assumed quality scores with measured checks on a recording's
timestamps and samples:

* ``monotonic``   — timestamps strictly increase (no reordering/duplicates);
* ``rate``        — median sampling rate within ``rate_tolerance`` of nominal;
* ``drift``       — cumulative clock drift vs. the nominal rate;
* ``dropouts``    — gaps longer than ``gap_factor`` × nominal period;
* ``nonfinite``   — NaN/Inf samples;
* ``flatline``    — runs of identical samples longer than ``flatline_s``;
* ``checksum``    — SHA-256 of the raw bytes against the recorded digest.

The default bounds below are engineering defaults, NOT owner-approved
instrument tolerances (AUD-SPEC-07); pass approved values via
:class:`AcquisitionBounds` once they exist.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class AcquisitionBounds:
    rate_tolerance: float = 0.01  # fractional deviation of median rate from nominal
    max_drift_s: float = 0.010  # cumulative clock drift over the recording
    gap_factor: float = 1.5  # gap > gap_factor × period counts as a dropout
    max_dropout_fraction: float = 0.001
    flatline_s: float = 0.5
    approved: bool = False  # True only for owner-approved instrument bounds


@dataclass
class AcquisitionReport:
    passed: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bounds: Dict[str, Any] = field(default_factory=dict)

    def failures(self) -> List[str]:
        return [name for name, c in self.checks.items() if not c["passed"]]


def check_recording(
    timestamps_s: np.ndarray,
    samples: np.ndarray,
    nominal_rate_hz: float,
    raw_bytes: Optional[bytes] = None,
    expected_sha256: Optional[str] = None,
    bounds: AcquisitionBounds = AcquisitionBounds(),
) -> AcquisitionReport:
    t = np.asarray(timestamps_s, dtype=float)
    x = np.asarray(samples, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("need at least two timestamps")
    if x.shape[0] != len(t):
        raise ValueError("samples and timestamps differ in length")
    period = 1.0 / nominal_rate_hz
    dt = np.diff(t)
    checks: Dict[str, Dict[str, Any]] = {}

    n_bad_order = int(np.sum(dt <= 0))
    checks["monotonic"] = {"passed": n_bad_order == 0, "non_increasing_steps": n_bad_order}

    pos = dt[dt > 0]
    median_rate = float(1.0 / np.median(pos)) if len(pos) else float("nan")
    rate_err = abs(median_rate - nominal_rate_hz) / nominal_rate_hz if np.isfinite(median_rate) else float("inf")
    checks["rate"] = {
        "passed": rate_err <= bounds.rate_tolerance,
        "median_rate_hz": median_rate,
        "relative_error": rate_err,
    }

    gaps = dt > bounds.gap_factor * period
    missing = float(np.sum(np.round(dt[gaps] / period) - 1)) if gaps.any() else 0.0
    expected_n = len(t) + missing
    dropout_fraction = missing / expected_n if expected_n else 0.0
    checks["dropouts"] = {
        "passed": dropout_fraction <= bounds.max_dropout_fraction,
        "gaps": int(gaps.sum()),
        "estimated_missing_samples": int(missing),
        "fraction": dropout_fraction,
    }

    # Drift: elapsed time vs. what the sample count implies at the nominal rate,
    # after accounting for detected dropouts.
    implied = (len(t) - 1 + missing) * period
    drift = float((t[-1] - t[0]) - implied)
    checks["drift"] = {"passed": abs(drift) <= bounds.max_drift_s, "drift_s": drift}

    n_nonfinite = int(np.sum(~np.isfinite(x)))
    checks["nonfinite"] = {"passed": n_nonfinite == 0, "count": n_nonfinite}

    # Constant runs are measured per channel; one dead channel fails the check.
    channels = x.reshape(len(t), -1)
    longest_by_channel = []
    for ch in range(channels.shape[1]):
        col = channels[:, ch]
        longest, run = 1, 1
        for a, b in zip(col[:-1], col[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        longest_by_channel.append(longest * period)
    worst = max(longest_by_channel)
    checks["flatline"] = {
        "passed": worst <= bounds.flatline_s,
        "longest_constant_run_s": worst,
        "worst_channel": int(np.argmax(longest_by_channel)),
    }

    if expected_sha256 is not None:
        if raw_bytes is None:
            checks["checksum"] = {"passed": False, "reason": "expected digest given but no raw bytes"}
        else:
            actual = hashlib.sha256(raw_bytes).hexdigest()
            checks["checksum"] = {"passed": actual == expected_sha256, "sha256": actual}

    return AcquisitionReport(passed=all(c["passed"] for c in checks.values()), checks=checks, bounds=asdict(bounds))
