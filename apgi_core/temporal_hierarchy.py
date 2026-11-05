"""Canonical APGI five-level temporal hierarchy (APGI-SCI-003).

The governing specification fixes the hierarchy as:

====  ======================  =========================
Level Canonical label         Timescale range
====  ======================  =========================
L0    L0 (10–100 ms)          10 ms  – 100 ms
L1    L1 (100 ms–3 s)         100 ms – 3 s
L2    L2 (1–30 min)           1 min  – 30 min
L3    L3 (hours–days)         1 h    – 7 days
L4    L4 (months–years)       ~1 month – 10 years
====  ======================  =========================

"hours to days" and "months to years" are not given numerically in the
specification; the numerical bounds used for L3 (1 h – 7 d) and L4
(30 d – 3650 d) are an implementation interpretation recorded in
``docs/validation/formulation-register.md`` and require owner confirmation
(AUD-SPEC-07). Ranges are inclusive on both ends; where adjacent ranges
touch (100 ms) the value belongs to the *lower* level. Values in the gaps
(3 s – 1 min, 30 min – 1 h, 7 d – 30 d) or outside 10 ms – 10 y belong to
no level and are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

MS = 1e-3
SECOND = 1.0
MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0
YEAR = 365.0 * DAY


@dataclass(frozen=True)
class TemporalLevel:
    index: int
    code: str
    label: str
    min_seconds: float
    max_seconds: float
    representative_tau_seconds: float
    description: str

    def contains(self, seconds: float) -> bool:
        return self.min_seconds <= seconds <= self.max_seconds


TEMPORAL_LEVELS: Tuple[TemporalLevel, ...] = (
    TemporalLevel(0, "L0", "L0 (10–100 ms)", 10 * MS, 100 * MS, 50 * MS, "fast sensory/ignition dynamics"),
    TemporalLevel(1, "L1", "L1 (100 ms–3 s)", 100 * MS, 3 * SECOND, 1 * SECOND, "trial-level integration"),
    TemporalLevel(2, "L2", "L2 (1–30 min)", 1 * MINUTE, 30 * MINUTE, 10 * MINUTE, "session/state adaptation"),
    TemporalLevel(3, "L3", "L3 (hours–days)", 1 * HOUR, 7 * DAY, 1 * DAY, "circadian/allostatic regulation"),
    TemporalLevel(4, "L4", "L4 (months–years)", 30 * DAY, 10 * YEAR, 0.5 * YEAR, "developmental/trait change"),
)

LEVEL_CODES: List[str] = [lvl.code for lvl in TEMPORAL_LEVELS]
LEVEL_LABELS: List[str] = [lvl.label for lvl in TEMPORAL_LEVELS]
REPRESENTATIVE_TAUS: List[float] = [lvl.representative_tau_seconds for lvl in TEMPORAL_LEVELS]


class TimescaleOutOfRangeError(ValueError):
    """A timescale does not fall inside any canonical level (or the declared one)."""


def classify_timescale(seconds: float) -> TemporalLevel:
    """Return the canonical level containing ``seconds`` or raise."""
    if not isinstance(seconds, (int, float)) or seconds != seconds or seconds <= 0:
        raise TimescaleOutOfRangeError(f"timescale must be a positive finite number of seconds, got {seconds!r}")
    for level in TEMPORAL_LEVELS:
        if level.contains(seconds):
            return level
    raise TimescaleOutOfRangeError(f"timescale {seconds} s is outside every canonical APGI level (L0–L4)")


def validate_level_timescale(level: int, seconds: float) -> None:
    """Raise unless ``seconds`` lies within canonical level ``level`` (0–4)."""
    if level not in range(len(TEMPORAL_LEVELS)):
        raise TimescaleOutOfRangeError(f"unknown level index {level}; expected 0–4")
    spec = TEMPORAL_LEVELS[level]
    if not spec.contains(seconds):
        raise TimescaleOutOfRangeError(
            f"{spec.code} requires {spec.min_seconds:g}–{spec.max_seconds:g} s, got {seconds:g} s"
        )


def level_for_code(code: str) -> Optional[TemporalLevel]:
    for level in TEMPORAL_LEVELS:
        if level.code == code:
            return level
    return None


def as_table() -> List[Dict[str, object]]:
    """Label inventory used by documentation/GUI generators."""
    return [
        {
            "code": lvl.code,
            "label": lvl.label,
            "min_seconds": lvl.min_seconds,
            "max_seconds": lvl.max_seconds,
            "representative_tau_seconds": lvl.representative_tau_seconds,
        }
        for lvl in TEMPORAL_LEVELS
    ]
