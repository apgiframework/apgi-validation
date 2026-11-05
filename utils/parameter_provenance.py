"""Scientific provenance labels for default parameters (APGI-SCI-005).

Runtime configuration origin (env/file/default) is not scientific provenance.
This module reads ``config/parameter_provenance.yaml`` and exposes the label
and source for each default so outputs can carry them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import yaml

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "parameter_provenance.yaml"
LABELS = ("analytically_derived", "post_hoc_calibrated", "assumed")


@lru_cache(maxsize=1)
def registry() -> Dict[str, Dict[str, object]]:
    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    return data["parameters"]


def provenance_for(names: Iterable[str]) -> Dict[str, Dict[str, object]]:
    """Label/source for each named parameter; unknown names are reported as unregistered."""
    reg = registry()
    return {
        n: (
            {"label": reg[n]["label"], "source": reg[n]["source"]}
            if n in reg
            else {"label": "UNREGISTERED", "source": None}
        )
        for n in names
    }


def settings_provenance() -> Dict[str, Dict[str, object]]:
    """Provenance of every APGISettings default, for attaching to result metadata."""
    from utils.apgi_settings import APGISettings

    return provenance_for(f"APGISettings.{n}" for n in APGISettings.model_fields)
