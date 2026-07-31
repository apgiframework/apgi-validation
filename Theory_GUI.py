#!/usr/bin/env python3
"""
APGI Theory GUI — entry point.

All implementation lives in gui/: gui/theme.py, gui/script_runner_gui.py,
gui/headless_runner.py. This module re-exports everything so existing import
sites (tests, Validation_GUI, verify_* utilities) continue to work unchanged.
"""

import os
import sys

os.environ["MPLBACKEND"] = "Agg"
os.environ["MATPLOTLIB_BACKEND"] = "Agg"
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import matplotlib

matplotlib.use("Agg", force=True)

_original_use = matplotlib.use


def _locked_use(backend, *args, **kwargs):
    """Prevent backend switching to GUI backends."""
    if backend.lower() not in ("agg", "svg", "pdf", "ps", "pgf", "cairo", "inline"):
        import warnings

        warnings.warn(f"Blocking backend switch to {backend}")
        return
    return _original_use(backend, *args, **kwargs)


matplotlib.use = _locked_use

import matplotlib.pyplot as plt

plt.switch_backend("Agg")

import logging
import tempfile
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="lifelines")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

try:
    import torch

    TORCH_AVAILABLE = True
    _ = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TMPDIR"] = tempfile.gettempdir()
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

for _cache_dir in [
    os.path.join(tempfile.gettempdir(), "matplotlib_cache"),
    os.path.expanduser("~/.cache/matplotlib"),
]:
    try:
        os.makedirs(_cache_dir, exist_ok=True)
    except OSError as exc:
        logger.debug("Failed to create cache directory %s: %s", _cache_dir, exc)

from gui.headless_runner import HeadlessRunner  # noqa: E402
from gui.script_runner_gui import ScriptRunnerGUI  # noqa: E402

# ── Re-exports for backward compatibility ─────────────────────────────────────
from gui.theme import APGICard  # noqa: E402
from gui.theme import COLORS, FONTS, APGIButtons, _resolve_font, apply_apgi_theme, create_empty_state, show_status

__all__ = [
    "COLORS",
    "FONTS",
    "APGIButtons",
    "APGICard",
    "apply_apgi_theme",
    "create_empty_state",
    "show_status",
    "_resolve_font",
    "ScriptRunnerGUI",
    "HeadlessRunner",
    "TORCH_AVAILABLE",
    "logger",
]


def main():
    import argparse
    import platform

    parser = argparse.ArgumentParser(description="APGI Theory Framework GUI / Headless Runner")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run all Theory scripts without launching the GUI",
    )
    parser.add_argument(
        "--script",
        metavar="NAME",
        default=None,
        help="(headless) Run only matching scripts",
    )
    parser.add_argument("--token", help="JWT authentication token for secured operations")
    args = parser.parse_args()

    if not args.headless:
        try:
            from utils.security_gateway import Role, SecurityGateway

            gateway = SecurityGateway()
            if args.token:
                gateway.require_roles(args.token, [Role.RESEARCHER, Role.ADMIN])
            elif os.environ.get("APGI_ENFORCE_AUTH", "0") == "1":
                raise SystemExit(
                    "Authentication required: pass --token to start the GUI. "
                    "For local development only, set APGI_ENFORCE_AUTH=0 to enable an auto-generated dev token."
                )
            else:
                try:
                    from utils.auth_adapter import get_auth_manager

                    auth_manager = get_auth_manager()
                    dev_token = auth_manager.generate_token("dev_user", Role.RESEARCHER, 24)
                    print("Development mode: Generated token (valid 24 hours)")
                    args.token = dev_token
                except Exception as e:
                    print(f"Note: Running without authentication ({e})")
        except ImportError:
            pass
        except PermissionError as e:
            print(f"Authentication failed: {e}")
            sys.exit(1)

    if args.headless:
        runner = HeadlessRunner()
        if args.script:
            protocols = runner._discover_protocols()
            filtered = {k: v for k, v in protocols.items() if args.script.lower() in k.lower()}
            if not filtered:
                print(f"No scripts matching '{args.script}' found.")
                sys.exit(1)
            runner.log_message(f"Running {len(filtered)} script(s) matching '{args.script}'\n")
            passed, failed = [], []
            for display_name, protocol_info in filtered.items():
                runner.log_message(f"Running: {display_name}")
                ok, msg = runner._execute_protocol(display_name, protocol_info)
                if ok:
                    runner.log_message("  ✓ PASS")
                    passed.append(display_name)
                else:
                    runner.log_message(f"  ✗ FAIL: {msg}")
                    failed.append((display_name, msg))
            sys.exit(0 if not failed else 1)
        else:
            exit_code = runner.run_all()
            sys.exit(exit_code)

    # GUI mode
    if platform.system() == "Darwin":
        pass  # macOS requires tkinter on main thread

    import tkinter as tk

    root = tk.Tk()
    ScriptRunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
