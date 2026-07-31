"""Cross-platform GUI smoke tests for APGI GUIs.

Run with a real Tk display:

    APGI_REAL_TKINTER=1 xvfb-run -a python -m pytest tests/test_gui_smoke_cross_platform.py

Without ``APGI_REAL_TKINTER=1`` the suite-wide conftest replaces tkinter with a
MagicMock and the widget tests skip — instantiating a mock proves nothing. That
skip used to fire unconditionally in CI, so the only job running pytest was
green while exercising no GUI at all. ``test_real_tkinter_is_active`` now makes
that misconfiguration visible, and ``test_gui_class_is_importable`` gives real
coverage on every run regardless of display availability.
"""

import os
import sys

import pytest

REAL_TKINTER = os.environ.get("APGI_REAL_TKINTER", "").strip().lower() in {"1", "true", "yes"}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every GUI entry point in the repository. Protocols_GUI and OSF_GUI were
# previously absent, leaving the two largest GUIs with no coverage at all.
GUI_TARGETS = [
    pytest.param(("gui.script_runner_gui", "ScriptRunnerGUI"), id="Theory_GUI"),
    pytest.param(("Validation_GUI", "APGIValidationGUI"), id="Validation_GUI"),
    pytest.param(("Protocols_GUI", "APGIValidationGUI"), id="Protocols_GUI"),
    pytest.param(("Falsification_GUI", "ProtocolRunnerGUI"), id="Falsification_GUI"),
    pytest.param(("Tests_GUI", "TestsGUI"), id="Tests_GUI"),
    pytest.param(("Utils_GUI", "UtilsRunnerGUI"), id="Utils_GUI"),
    pytest.param(("OSF_GUI", "OSFProtocolGUI"), id="OSF_GUI"),
]


def _import_gui(module_name, class_name):
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    module = __import__(module_name, fromlist=[class_name])
    gui_class = getattr(module, class_name, None)
    assert gui_class is not None, f"{module_name} does not expose {class_name}"
    return gui_class


def test_real_tkinter_is_active():
    """Guard: a mocked-tkinter CI run must skip visibly, not pass silently."""
    if not REAL_TKINTER:
        pytest.skip("Set APGI_REAL_TKINTER=1 (with a display, e.g. xvfb-run) to run GUI smoke tests")

    import tkinter as tk
    from unittest.mock import MagicMock

    assert not isinstance(tk, MagicMock), "tkinter is mocked despite APGI_REAL_TKINTER=1"
    assert not isinstance(tk.Tk, MagicMock), "tkinter.Tk is mocked despite APGI_REAL_TKINTER=1"


@pytest.fixture()
def tk_root():
    """Provide a hidden real Tk root, destroyed after the test."""
    if not REAL_TKINTER:
        pytest.skip("Set APGI_REAL_TKINTER=1 (with a display, e.g. xvfb-run) to run GUI smoke tests")

    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No usable display for Tk: {exc}")

    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


@pytest.mark.parametrize("target", GUI_TARGETS)
def test_gui_class_is_importable(target):
    """Import-level check that runs even with mocked tkinter.

    Catches syntax errors, broken imports and renamed classes on every CI run,
    including platforms with no display.
    """
    _import_gui(*target)


@pytest.mark.parametrize("target", GUI_TARGETS)
def test_gui_smoke_cross_platform(tk_root, target):
    """Instantiate each GUI to validate basic window creation across platforms."""
    gui_class = _import_gui(*target)

    instance = gui_class(tk_root)
    assert instance is not None

    tk_root.update_idletasks()
