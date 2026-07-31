"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1

Canonical JSON sanitisation for protocol results.

Protocol results are written to ``*_results.json`` and read back by dashboards,
aggregators and external tooling using the standard library, so they must
contain only plain JSON types.

NumPy scalars are the usual leak. Under NumPy 2.x ``np.True_`` reports
``__class__.__name__ == "bool"`` while *not* being a Python ``bool``, so
``json.dumps`` fails with the baffling message "Object of type bool is not JSON
serializable". :func:`to_json_safe` converts these at the boundary.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["to_json_safe", "APGIJSONEncoder", "dump_json", "dumps_json"]


def to_json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into plain JSON-serialisable types.

    Handles NumPy scalars and arrays (including ``np.bool_``), sets, tuples,
    ``Path``, ``datetime``/``date``, ``Decimal``, ``Enum`` and dataclass-like
    objects. Unknown types fall back to ``str`` so serialisation never raises.

    Args:
        obj: Arbitrary Python object.

    Returns:
        A structure composed only of dict, list, str, int, float, bool, None.
    """
    # Order matters: np.bool_ must be checked before np.integer, because NumPy
    # booleans are not integers but some scalar checks overlap.
    if obj is None or isinstance(obj, (str, int, float, bool)) and not isinstance(obj, np.generic):
        # Guard against NaN/Infinity, which are not valid JSON.
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None
        return obj

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(obj, np.complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.ndarray):
        return [to_json_safe(item) for item in obj.tolist()]
    if isinstance(obj, np.generic):
        return to_json_safe(obj.item())

    if isinstance(obj, dict):
        # JSON object keys must be strings.
        return {str(key): to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_json_safe(item) for item in obj]

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return to_json_safe(obj.value)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {str(k): to_json_safe(v) for k, v in vars(obj).items() if not k.startswith("_")}

    return str(obj)


class APGIJSONEncoder(json.JSONEncoder):
    """JSON encoder that falls back to :func:`to_json_safe`."""

    def default(self, o: Any) -> Any:
        return to_json_safe(o)


def dumps_json(data: Any, **kwargs: Any) -> str:
    """Serialise ``data`` to a JSON string, sanitising first."""
    kwargs.setdefault("indent", 2)
    return json.dumps(to_json_safe(data), **kwargs)


def dump_json(data: Any, path: Path, **kwargs: Any) -> None:
    """Write ``data`` to ``path`` as sanitised JSON."""
    kwargs.setdefault("indent", 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_json_safe(data), handle, **kwargs)
