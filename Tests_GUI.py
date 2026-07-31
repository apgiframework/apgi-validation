#!/usr/bin/env python3
"""
Tests GUI for APGI validation framework
======================================

A simple GUI for running and displaying test results.
"""

import logging
import os
import queue
import subprocess  # nosec B404
import sys
import threading

# Add current directory to path
sys.path.insert(0, os.getcwd())
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from utils.constants import VisualConstants

    _VC = VisualConstants()
    _HC_GREY = _VC.HC_GREY
    _ALLOSTATIC_PURPLE = _VC.ALLOSTATIC_PURPLE
    _IGNITION_GREEN = _VC.IGNITION_GREEN
    _THETA_RED = _VC.THETA_RED
    _ST_BLUE = _VC.ST_BLUE
except Exception:
    # Fallback hex values matching VisualConstants defaults
    _HC_GREY = "#999999"
    _ALLOSTATIC_PURPLE = "#762A83"
    _IGNITION_GREEN = "#41AB5D"
    _THETA_RED = "#D6604D"
    _ST_BLUE = "#2166AC"


def _pick_font(candidates: list, size: int = 10) -> str:
    """Return the first available font family from candidates."""
    available = set(tkfont.families())
    for name in candidates:
        if name in available:
            return name
    return "TkDefaultFont"


def apply_apgi_theme(root):
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use("clam")

    body_font = _pick_font(["Noto Sans", "Segoe UI", "Helvetica", "Arial"])
    mono_font = _pick_font(["Noto Sans Mono", "Consolas", "Courier New", "Courier"])

    # Configure Global Elements
    style.configure("TFrame", background=_HC_GREY)
    style.configure(
        "TLabel",
        background=_HC_GREY,
        foreground=_ALLOSTATIC_PURPLE,
        font=(body_font, 10),
    )
    style.configure("Header.TLabel", font=(body_font, 12, "bold"))

    # Custom Card Style
    style.configure("Card.TFrame", background="#ffffff", borderwidth=1, relief="solid")

    # Button Styling
    style.configure("TButton", padding=6, background="#e9ecef")
    style.map(
        "TButton",
        background=[("active", "#dee2e6"), ("disabled", "#f1f3f5")],
        foreground=[("disabled", "#adb5bd")],
    )

    # Primary Button Styling
    style.configure(
        "Primary.TButton",
        background=_IGNITION_GREEN,
        foreground="white",
        font=(body_font, 10, "bold"),
        padding=8,
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#0f3d1a")],
        foreground=[("active", "white")],
    )

    # Danger Button Styling
    style.configure(
        "Danger.TButton",
        background=_THETA_RED,
        foreground="white",
        font=(body_font, 10, "bold"),
        padding=8,
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#5a161d")],
        foreground=[("active", "white")],
    )

    # Secondary Button Styling
    style.configure(
        "Secondary.TButton",
        background=_ST_BLUE,
        foreground="white",
        font=(body_font, 10),
        padding=6,
    )
    style.map(
        "Secondary.TButton",
        background=[("active", "#1f5a82")],
        foreground=[("active", "white")],
    )

    # Checkbutton Styling
    style.configure("Card.TCheckbutton", background="#ffffff")

    # Status styling
    style.configure("Success.TLabel", foreground=_IGNITION_GREEN, font=(body_font, 10, "bold"))
    style.configure("Error.TLabel", foreground=_THETA_RED, font=(body_font, 10, "bold"))

    # Configure root window
    root.configure(background=_HC_GREY)

    return style, mono_font


class APGIButtons:
    """Standard button configurations for APGI applications."""

    @staticmethod
    def primary(parent, text, command):
        """Primary action button (green)."""
        return ttk.Button(parent, text=text, command=command, style="Primary.TButton", cursor="hand2")

    @staticmethod
    def danger(parent, text, command):
        """Danger/Stop button (red)."""
        return ttk.Button(parent, text=text, command=command, style="Danger.TButton", cursor="hand2")

    @staticmethod
    def secondary(parent, text, command):
        """Secondary action button (blue)."""
        return ttk.Button(
            parent,
            text=text,
            command=command,
            style="Secondary.TButton",
            cursor="hand2",
        )


class APGICard(ttk.Frame):
    """Standardized information card for all APGI apps."""

    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.container = ttk.Frame(self, padding=15, style="Card.TFrame")
        self.container.pack(fill="both", expand=True)

        if title:
            self.lbl_title = ttk.Label(self.container, text=title.upper(), style="Header.TLabel")
            self.lbl_title.pack(anchor="w", pady=(0, 10))


class TestsGUI:
    """Simple GUI for running tests.
    ======================================
    Provides a tkinter-based interface to run tests with real-time
    output display and results visualization.
    """

    def __init__(self, root: tk.Tk = None):
        """Initialize the Tests GUI.

        Args:
            root: Tkinter root window (creates new one if None)
        """
        self.root = root or tk.Tk()
        self.root.title("APGI Tests GUI")
        self.root.geometry("900x700")
        _, self._mono_font = apply_apgi_theme(self.root)
        self._output_queue: queue.Queue = queue.Queue()
        self._test_thread: threading.Thread = None

        self._create_menu_bar()
        self._bind_shortcuts()
        self.setup_ui()

    def _create_menu_bar(self) -> None:
        """Create a consistent menu bar across APGI GUIs."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Run Tests", command=self.run_tests_gui, accelerator="Ctrl+R")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit_application, accelerator="Ctrl+Q")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Clear Output", command=self.clear_output, accelerator="Ctrl+L")

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts for common actions."""
        self.root.bind("<Control-q>", lambda e: self.quit_application())
        self.root.bind("<Command-q>", lambda e: self.quit_application())
        self.root.bind("<Control-r>", lambda e: self.run_tests_gui())
        self.root.bind("<Control-l>", lambda e: self.clear_output())

    def setup_ui(self):
        """Setup the user interface."""
        # Top metric bar / header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(fill="x")
        ttk.Label(header_frame, text="APGI TEST ENVIRONMENT", style="Header.TLabel").pack(side="left")

        # Main content
        main_content = ttk.Frame(self.root, padding="10")
        main_content.pack(fill="both", expand=True)

        # Grid layout
        main_content.columnconfigure(0, weight=0, minsize=250)
        main_content.columnconfigure(1, weight=1)
        main_content.rowconfigure(0, weight=1)

        # Left Sidebar (Controls)
        sidebar = ttk.Frame(main_content)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        config_card = APGICard(sidebar, "Test Configuration")
        config_card.pack(fill="x", pady=(0, 15))

        ttk.Label(config_card.container, text="Test Type:").pack(anchor="w", pady=(0, 5))
        self.test_type_var = tk.StringVar(value="all")
        test_type_combo = ttk.Combobox(
            config_card.container,
            textvariable=self.test_type_var,
            values=["unit", "integration", "performance", "all"],
            state="readonly",
        )
        test_type_combo.pack(fill="x", pady=(0, 15))

        self.coverage_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config_card.container,
            text="Run with Coverage",
            variable=self.coverage_var,
            style="Card.TCheckbutton",
        ).pack(anchor="w", pady=(0, 20))

        # Buttons Frame for alignment
        btn_frame = ttk.Frame(sidebar)
        btn_frame.pack(fill="x")

        APGIButtons.primary(btn_frame, "RUN TESTS", self.run_tests_gui).pack(fill="x", pady=(0, 10))
        APGIButtons.secondary(btn_frame, "CLEAR OUTPUT", self.clear_output).pack(fill="x")

        # Workspace (Output)
        workspace = APGICard(main_content, "Test Output")
        workspace.grid(row=0, column=1, sticky="nsew")

        self.status_label = ttk.Label(workspace.container, text="Ready", style="TLabel")
        self.status_label.pack(anchor="w", pady=(0, 5))

        self.output_text = scrolledtext.ScrolledText(
            workspace.container,
            bg=_ALLOSTATIC_PURPLE,
            fg=_HC_GREY,
            font=(self._mono_font, 10),
            borderwidth=0,
            highlightthickness=0,
        )
        self.output_text.pack(fill="both", expand=True)

    def show_status(self, status_type, message):
        """Update status label."""
        if status_type == "success":
            self.status_label.config(text=f"✔ {message}", style="Success.TLabel")
        elif status_type == "error":
            self.status_label.config(text=f"✖ {message}", style="Error.TLabel")
        else:
            self.status_label.config(text=f"ℹ {message}", style="TLabel")

    def run_tests_gui(self):
        """Run tests from GUI — spawns a background thread to avoid freezing the event loop."""
        if self._test_thread and self._test_thread.is_alive():
            self.show_status("info", "Tests already running...")
            return

        self.show_status("info", "Running tests...")
        self.output_text.delete(1.0, tk.END)

        config = {
            "test_type": self.test_type_var.get(),
            "coverage": self.coverage_var.get(),
        }

        self._test_thread = threading.Thread(
            target=self._run_tests_background,
            args=(config,),
            daemon=True,
        )
        self._test_thread.start()
        self.root.after(100, self._poll_output_queue)

    def _run_tests_background(self, config: Dict[str, Any]) -> None:
        """Background worker: runs pytest and feeds output into the queue."""
        results = self.run_tests(config)
        self._output_queue.put(("DONE", results))

    def _poll_output_queue(self) -> None:
        """Called via after() to drain the output queue on the main thread."""
        try:
            while True:
                item = self._output_queue.get_nowait()
                tag, data = item
                if tag == "DONE":
                    self.display_results(data)
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_output_queue)

    def clear_output(self):
        """Clear the output display."""
        self.output_text.delete(1.0, tk.END)

    def quit_application(self):
        """Quit the application gracefully."""
        try:
            if self._test_thread and self._test_thread.is_alive():
                self.show_status("info", "Stopping tests...")
        except Exception as exc:
            logger.debug("Failed to signal test thread shutdown: %s", exc)
        self.root.quit()
        self.root.destroy()

    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "APGI Tests GUI\nTest Runner Interface\nVersion 1.3.0",
        )
        self.show_status("info", "Ready")

    def run_tests(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests based on configuration.

        Args:
            config: Test configuration dictionary

        Returns:
            Dictionary containing test results
        """
        test_type = config.get("test_type", "all")
        coverage = config.get("coverage", False)

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", "tests/"]

        # If it's a specific type of test, add marker logic or path logic
        # For APGI validation, usually we just run the tests
        if test_type != "all":
            # If tests have markers
            cmd.extend(["-m", test_type])

        if coverage:
            cmd.append("--cov=.")
            cmd.append("--cov-report=term-missing")

        # Add verbose output
        cmd.append("-v")

        # Run tests
        try:
            cwd = Path(__file__).parent
            self.output_text.insert(tk.END, f"> {' '.join(cmd)}\n\n")
            self.output_text.update()

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, shell=False)  # nosec B603

            return {
                "test_results": {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
                "config": config,
            }

        except Exception as e:
            return {
                "test_results": {
                    "success": False,
                    "error": str(e),
                    "stdout": "",
                    "stderr": "",
                    "returncode": -1,
                },
                "config": config,
            }

    def display_results(self, results: Dict[str, Any]):
        """Display test results in the GUI.

        Args:
            results: Test results dictionary
        """
        self.output_text.delete(1.0, tk.END)

        test_results = results.get("test_results", {})

        error = test_results.get("error")
        if error:
            self.show_status("error", "Tests Failed - Exception occurred")
            self.output_text.insert(tk.END, f"Exception:\n{error}\n")
            return

        if test_results.get("success", False):
            self.show_status("success", "Tests Passed Successfully")
        else:
            self.show_status("error", "Tests Failed")

        # Display stdout
        stdout = test_results.get("stdout", "")
        if stdout:
            self.output_text.insert(tk.END, stdout)

        # Display stderr
        stderr = test_results.get("stderr", "")
        if stderr:
            self.output_text.insert(tk.END, "\nErrors:\n" + stderr)

        # Scroll to bottom
        self.output_text.see(tk.END)

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Launch the tests GUI with optional auth enforcement."""
    import argparse

    parser = argparse.ArgumentParser(description="APGI Tests GUI")
    parser.add_argument(
        "--token",
        help="JWT authentication token for secured operations",
    )
    args = parser.parse_args()

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
            except Exception as exc:
                print(f"Note: Running without authentication ({exc})")
    except ImportError:
        pass
    except PermissionError as exc:
        print(f"Authentication failed: {exc}")
        sys.exit(1)

    gui = TestsGUI()
    gui.run()


if __name__ == "__main__":
    main()
