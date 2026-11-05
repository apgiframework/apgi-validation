"""Import nolds on every supported Python (CI-NOLDS).

nolds 0.6.3 loads its bundled datasets with
``importlib.resources.files("nolds.datasets")``. Python 3.12+ accepts a module
there; Python 3.11 accepts only packages and raises ``TypeError`` at import.
Older nolds releases instead need ``pkg_resources``, which setuptools >= 81
removed, and pinning setuptools < 81 leaves known vulnerabilities unfixable.

This shim retries ``files()`` with the module's parent package for the
duration of the nolds import only, then restores the original function. It
changes no numerical code.
"""

from __future__ import annotations

import importlib.resources as _resources
from types import ModuleType


def import_nolds() -> ModuleType:
    original = _resources.files

    def _files(anchor=None):  # type: ignore[no-untyped-def]
        try:
            return original(anchor)
        except TypeError:
            if isinstance(anchor, str) and "." in anchor:
                return original(anchor.rsplit(".", 1)[0])
            raise

    _resources.files = _files  # type: ignore[assignment]
    try:
        import nolds
    finally:
        _resources.files = original  # type: ignore[assignment]
    return nolds
