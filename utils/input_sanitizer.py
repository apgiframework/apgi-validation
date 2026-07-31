"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, cast

T = TypeVar("T", int, float)


class InputSanitizer:
    """
    Centralized utility for sanitizing and validating inputs from CLI and GUI.
    Prevents injection attacks and ensures numerical stability.
    """

    @staticmethod
    def sanitize_string(value: str, pattern: str = r"^[a-zA-Z0-9_\-\.]+$") -> str:
        """
        Sanitizes a string against a whitelist pattern.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value)}")

        if not re.match(pattern, value):
            raise ValueError(f"String contains illegal characters: {value}")

        return value

    @staticmethod
    def sanitize_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
        """
        Validates a path to prevent path traversal attacks.
        """
        path = Path(path_str).resolve()

        if base_dir:
            base_dir = base_dir.resolve()
            if not str(path).startswith(str(base_dir)):
                raise ValueError(f"Path traversal detected: {path_str} is outside {base_dir}")

        return path

    @staticmethod
    def sanitize_numeric(
        value: Any,
        expected_type: type[T],
        min_val: Optional[T] = None,
        max_val: Optional[T] = None,
    ) -> T:
        """
        Ensures a value is a valid number within the specified range.
        """
        try:
            num_val = expected_type(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric input: {value} (expected {expected_type.__name__})")

        if min_val is not None and num_val < min_val:
            raise ValueError(f"Value {num_val} is below minimum {min_val}")

        if max_val is not None and num_val > max_val:
            raise ValueError(f"Value {num_val} is above maximum {max_val}")

        return cast(T, num_val)

    @classmethod
    def sanitize_params(cls, params: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sanitizes a dictionary of parameters against a schema.

        Schema format:
        {
            "param_name": {"type": float, "min": 0.0, "max": 1.0, "required": True}
        }
        """
        sanitized = {}
        for name, spec in schema.items():
            if name not in params:
                if spec.get("required", False):
                    raise ValueError(f"Missing required parameter: {name}")
                sanitized[name] = spec.get("default")
                continue

            val = params[name]
            p_type = spec.get("type", float)

            if p_type in (int, float):
                sanitized[name] = cls.sanitize_numeric(
                    val,
                    p_type,
                    min_val=spec.get("min"),
                    max_val=spec.get("max"),
                )
            elif p_type is str:
                sanitized[name] = cls.sanitize_string(val, pattern=spec.get("pattern", r"^[a-zA-Z0-9_\-\.]+$"))
            else:
                sanitized[name] = val  # Passthrough for complex types (should be handled by Pydantic usually)

        return sanitized
