#!/usr/bin/env python3
"""
Simple GUI to run all utils folder scripts
==========================================

A tkinter-based GUI that allows running all scripts in the utils folder
with output display and error handling.
"""

import logging
import os
import platform
import queue
import re
import select
import shlex
import subprocess  # nosec B404
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, simpledialog, ttk
from typing import Any, Dict, List, Optional

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("UtilsRunner")


def strip_ansi_codes(text: str) -> str:
    """Strip ANSI escape codes from text for clean display in GUI.

    Args:
        text: Text that may contain ANSI escape codes.

    Returns:
        Clean text with ANSI codes removed.
    """
    # ANSI escape sequence pattern: ESC[ followed by numbers/letters and ending with m
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def is_progress_bar_line(text: str) -> bool:
    """Check if text is a tqdm progress bar line.

    Args:
        text: Text to check.

    Returns:
        True if the line appears to be a tqdm progress bar.
    """
    # Match patterns like: "Processing jobs:   0%|          | 0/1 [00:00<?, ?it/s]"
    # or "Loading data:  17%|█▋        | 1/6 [00:00<00:00,  6.36it/s]"
    # Pattern: any text, then percentage, progress bar blocks, count, and timing
    progress_pattern = re.compile(r"^.*\d+%\|[\s█▏▎▍▌▋▊▉]+\|\s*\d+/\d+\s*\[.*\].*$")  # noqa: BLK100
    return bool(progress_pattern.match(text))


def should_filter_line(text: str) -> bool:
    """Determine if a line should be filtered from GUI output.

    Args:
        text: Text line to evaluate.

    Returns:
        True if the line should be suppressed.
    """
    if not text or text.isspace():
        return True

    # Filter tqdm progress bar lines
    if is_progress_bar_line(text):
        return True

    # Filter carriage return lines (progress bar updates)
    if "\r" in text and ("%" in text or "it/s" in text):
        return True

    return False


class UtilsRunnerGUI:
    """Simple GUI for running utils scripts.

    Provides a tkinter-based interface to run utility scripts from the utils folder
    with real-time output display, error handling, and process management.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("APGI Utils Scripts Runner")

        # Responsive window sizing with DPI awareness
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(int(screen_width * 0.7), 1200)
        window_height = min(int(screen_height * 0.7), 800)
        window_width = max(window_width, 800)
        window_height = max(window_height, 600)
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(640, 480)

        # Set window icon
        self._set_window_icon()

        # Add menu bar for consistency
        self._create_menu_bar()

        # Get utils directory
        self.utils_dir = Path(__file__).parent / "utils"

        # Validate utils directory exists
        if not self.utils_dir.exists():
            print(f"Warning: Utils directory not found at {self.utils_dir}")
            self.utils_dir = None

        # Output tag constants
        self.TAG_INFO = "info"
        self.TAG_ERROR = "error"
        self.TAG_SUCCESS = "success"
        self.TAG_WARNING = "warning"

        # Output queue for thread-safe updates
        self.output_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        # Load configuration
        self.config = self.load_config()

        self.scripts = self.get_script_list()

        # Store running processes
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self._run_all_cancelled = False

        # Track daemon threads for cleanup
        self.daemon_threads: List[threading.Thread] = []

        # Maximum lines to keep in output to prevent performance issues
        self.max_output_lines = self._get_configured_buffer_size()

        self.setup_ui()

        # Add keyboard shortcut for quitting (Ctrl+Q or Cmd+Q)
        self.root.bind("<Control-q>", self.quit_application)
        self.root.bind("<Command-q>", self.quit_application)
        self.root.bind("<Control-r>", lambda e: self.run_selected_script())
        self.root.bind("<Control-Shift-R>", lambda e: self.run_all_scripts())
        self.root.bind("<Control-s>", lambda e: self.stop_selected_script())
        self.root.bind("<Control-l>", lambda e: self.clear_output())

        # Handle window close button
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)

    def _get_configured_buffer_size(self) -> int:
        """Get configured output buffer size from environment or use default."""
        import os

        env_size = os.environ.get("APGI_GUI_BUFFER_SIZE")
        if env_size:
            try:
                size = int(env_size)
                if 100 <= size <= 10000:
                    return size
            except ValueError:
                pass
        return 2000  # Default for Utils-GUI

    def _looks_like_json(self, text: str) -> bool:
        """Check if text looks like JSON output.

        Args:
            text: Text to check.

        Returns:
            True if text appears to be JSON.
        """
        text = text.strip()
        # Check for common JSON patterns
        if text.startswith("{") and text.endswith("}"):
            return True
        if text.startswith("[") and text.endswith("]"):
            return True
        # Check for JSON-like structure with quotes and colons
        if '"' in text and ":" in text and ("{" in text or "[" in text):
            return True
        return False

    def _check_for_common_errors(self, text: str, script_name: str) -> None:
        """Check output for common error patterns and provide helpful messages.

        Args:
            text: Output text to check.
            script_name: Name of the script being run.
        """
        text_lower = text.lower()

        # Check for missing numpy
        if "no module named 'numpy'" in text_lower or "numpy not available" in text_lower:
            self.log_output(
                f"⚠️  {script_name} requires NumPy. Install with: pip install numpy",
                self.TAG_WARNING,
            )

        # Check for import errors
        if "importerror" in text_lower or "module not found" in text_lower:
            # Try to extract module name
            import re

            module_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", text_lower)
            if module_match:
                module_name = module_match.group(1)
                self.log_output(
                    f"⚠️  {script_name} requires {module_name}. Install with: pip install {module_name}",
                    self.TAG_WARNING,
                )

        # Check for permission errors
        if "permission denied" in text_lower or "permissionerror" in text_lower:
            self.log_output(
                f"⚠️  {script_name} has permission issues. Check file permissions.",
                self.TAG_WARNING,
            )

        # Check for file not found errors
        if "file not found" in text_lower or "filenotfounderror" in text_lower:
            self.log_output(
                f"⚠️  {script_name} can't find a required file.",
                self.TAG_WARNING,
            )

    def _set_window_icon(self) -> None:
        """Set application window icon if available."""
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.png"
            if icon_path.exists():
                from PIL import Image, ImageTk

                img = Image.open(icon_path)
                photo = ImageTk.PhotoImage(img)
                self.root.iconphoto(True, photo)  # type: ignore[arg-type]
                self._icon_image = photo  # Keep reference
            else:
                # Icon not found, use default tk icon (no warning needed)
                pass
        except (IOError, OSError, AttributeError, ImportError) as e:
            # Icon loading error - use default tk icon with logging
            if hasattr(self, "log_output") and self.root and hasattr(self.root, "destroy"):
                self.log_output(f"Warning: Icon loading failed: {e}", self.TAG_WARNING)  # noqa: BLK100
            pass  # Use default tk icon

    def _create_menu_bar(self) -> None:
        """Create menu bar for consistency with other APGI GUIs."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Run Selected", command=self.run_selected_script, accelerator="Ctrl+R")
        file_menu.add_command(label="Run All", command=self.run_all_scripts, accelerator="Ctrl+Shift+R")
        file_menu.add_command(label="Stop", command=self.stop_selected_script, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit_application, accelerator="Ctrl+Q")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Clear Output", command=self.clear_output, accelerator="Ctrl+L")

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _show_about(self) -> None:
        """Show about dialog."""
        from tkinter import messagebox

        messagebox.showinfo(
            "About",
            "APGI Utils Runner\nUtility Script Interface\nVersion 1.3.0",
        )

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default.

        Returns:
            Configuration dictionary with default values if file not found.
        """
        config_path = Path(__file__).parent / "utils" / "utils_script_config.json"  # noqa: BLK100

        # Detect environment for appropriate timeout settings
        import os

        env = os.environ.get("APGI_ENV", "development").lower()

        # Environment-specific timeout settings
        env_timeouts = {
            "development": {
                "default": 300,
                "utilities": 300,
            },  # 5 minutes for dev
            "testing": {
                "default": 600,
                "utilities": 600,
            },  # 10 minutes for testing
            "production": {
                "default": 3600,
                "utilities": 3600,
            },  # 1 hour for prod
        }

        env_timeout = env_timeouts.get(env, env_timeouts["development"])

        default_config = {
            "script_categories": {
                "utilities": {
                    "description": "General utility scripts",
                    "scripts": [],
                    "timeout": env_timeout["utilities"],
                }
            },
            "default_settings": {
                "timeout": env_timeout["default"],
                "max_retries": 2,
                "retry_delay": 1,
                "enable_execution_time": True,
                "log_level": "INFO",
                "environment": env,
            },
        }

        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    import json

                    return json.load(f)
            else:
                # Create default config file
                if config_path.parent.exists():
                    with open(config_path, "w", encoding="utf-8") as f:
                        import json

                        json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            self.log_output(f"Error loading config: {e}", self.TAG_WARNING)
            return default_config

    def get_script_timeout(self, script_name: str) -> int:
        """Get timeout for a specific script from configuration.

        Args:
            script_name: Name of the script.

        Returns:
            Timeout in seconds.
        """
        for category in self.config.get("script_categories", {}).values():
            if script_name in category.get("scripts", []):
                return category.get("timeout", self.config["default_settings"]["timeout"])
        return self.config["default_settings"]["timeout"]

    def prompt_for_arguments(self, script_name: str) -> List[str]:
        """Prompt user for script arguments.

        Args:
            script_name: Name of the script.

        Returns:
            List of arguments to pass to the script.
        """
        dialog = simpledialog.askstring(
            "Script Arguments",
            f"Enter arguments for {script_name} (optional):",
            parent=self.root,
        )

        if dialog:
            # Split arguments while respecting quotes
            try:
                return shlex.split(dialog)
            except ValueError:
                # Fallback to simple split if shlex fails
                return dialog.split()
        return []

    def get_script_list(self) -> List[Path]:
        """Get all Python scripts in utils directory.

        Returns:
            List of Path objects for executable Python scripts.
        """
        scripts = []
        if self.utils_dir and self.utils_dir.exists() and self.utils_dir.is_dir():  # noqa: BLK100  # noqa: BLK100
            for file_path in self.utils_dir.glob("*.py"):
                if file_path.name != "__init__.py" and self._is_executable_script(file_path):  # noqa: BLK100
                    scripts.append(file_path)
        return sorted(scripts)

    def _is_executable_script(self, script_path: Path) -> bool:
        """Check if a script is executable.

        Args:
            script_path: Path to the script file.

        Returns:
            True if script appears to be executable, False otherwise.
        """
        try:
            # Check if file has shebang or contains if __name__ == "__main__":
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Check for shebang
                first_line = content.split("\n")[0].strip()
                if first_line.startswith("#!"):
                    return True
                # Check for if __name__ == "__main__": pattern
                # Also check for variations with different spacing
                # Use regex to be more robust
                import re

                # Pattern for if __name__ == "__main__": with various spacing
                name_main_pattern = re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', re.IGNORECASE)
                if name_main_pattern.search(content):
                    return True
                # Also check for main() function that might be called
                # Look for def main() or def main( with any arguments
                main_def_pattern = re.compile(r"def\s+main\s*\(", re.IGNORECASE)
                if main_def_pattern.search(content):
                    # Check if main() is called anywhere in the file
                    # Look for main() call with any arguments
                    main_call_pattern = re.compile(r"^\s*main\s*\(", re.MULTILINE)
                    if main_call_pattern.search(content):
                        return True
                    # Also check for if __name__ == "__main__": main() pattern
                    # which we already checked above, but just in case
                # For utils directory, be more permissive: if it's a .py file
                # (not __init__.py), assume it's executable
                # This matches the old behavior that found 102 scripts
                return True
        except (UnicodeDecodeError, IOError):
            return False

    def setup_ui(self):
        """Setup the user interface.

        Creates the main layout with script list, control buttons,
        status display, and output area.
        """
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="APGI Utils Scripts Runner",
            font=("Arial", 16, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Scripts list frame
        list_frame = ttk.LabelFrame(main_frame, text="Available Scripts", padding="5")  # noqa: BLK100
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))  # noqa: BLK100

        # Scripts listbox with scrollbar
        list_scrollbar = ttk.Scrollbar(list_frame)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scripts_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, height=15, width=40)
        self.scripts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.config(command=self.scripts_listbox.yview)

        # Populate scripts list
        for script in self.scripts:
            self.scripts_listbox.insert(tk.END, script.name)

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky="new", padx=(0, 10))  # noqa: BLK100

        # Buttons
        self.run_button = ttk.Button(
            control_frame,
            text="Run Selected",
            command=self.run_selected_script,
        )
        self.run_button.pack(pady=5, fill=tk.X)

        self.run_args_button = ttk.Button(
            control_frame,
            text="Run with Arguments",
            command=self.run_selected_script_with_args,
        )
        self.run_args_button.pack(pady=5, fill=tk.X)

        self.run_all_button = ttk.Button(control_frame, text="Run All Scripts", command=self.run_all_scripts)
        self.run_all_button.pack(pady=5, fill=tk.X)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Selected",
            command=self.stop_selected_script,
            state=tk.DISABLED,
        )
        self.stop_button.pack(pady=5, fill=tk.X)

        self.clear_button = ttk.Button(control_frame, text="Clear Output", command=self.clear_output)
        self.clear_button.pack(pady=5, fill=tk.X)

        self.quit_button = ttk.Button(control_frame, text="Quit", command=self.quit_application)
        self.quit_button.pack(pady=5, fill=tk.X)

        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="5")  # noqa: BLK100
        status_frame.pack(pady=20, fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack()

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate", length=200)  # noqa: BLK100
        self.progress.pack(pady=10, fill=tk.X)

        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=(10, 0),
        )

        # Output text area
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD, font=("Courier", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for different output types
        self.output_text.tag_config(self.TAG_INFO, foreground="black")
        self.output_text.tag_config(self.TAG_ERROR, foreground="red")
        self.output_text.tag_config(self.TAG_SUCCESS, foreground="green")
        self.output_text.tag_config(self.TAG_WARNING, foreground="orange")

        # Start processing output queue
        self.process_output_queue()

    def log_output(self, message: str, tag: str = None):
        """Add message to output queue for thread-safe logging.

        Args:
            message: The message to log.
            tag: The color tag to apply (info, error, success, warning).
        """
        if tag is None:
            tag = self.TAG_INFO
        self.output_queue.put((message, tag))

    def _log_output(self, message: str, tag: str):
        """Thread-safe output logging."""
        self.output_text.insert(tk.END, message + "\n", tag)
        self.output_text.see(tk.END)

        # Limit output lines to prevent performance issues
        try:
            line_count = int(self.output_text.index("end-1c").split(".")[0])
            if line_count > self.max_output_lines + 50:  # Larger buffer to reduce frequent trimming
                # Batch delete multiple lines at once for better performance
                lines_to_delete = line_count - self.max_output_lines + 50
                self.output_text.delete(1.0, f"{lines_to_delete + 1}.0")
        except (ValueError, tk.TclError):
            # Handle edge cases where text widget is empty or has unexpected content
            pass

    def process_output_queue(self):
        """Process queued output messages in the main thread."""
        try:
            while True:
                message, tag = self.output_queue.get_nowait()
                self._log_output(message, tag)
        except queue.Empty:
            pass
        self.root.after(100, self.process_output_queue)

    def update_status(self, message: str):
        """Update status label.

        Args:
            message: The status message to display.
        """
        self.root.after_idle(lambda: self._safe_update_status(message))

    def _safe_update_status(self, message: str):
        """Thread-safe status update."""
        self.status_label.config(text=message)

    def get_selected_script(self) -> Optional[Path]:
        """Get the currently selected script.

        Returns:
            Path to selected script or None if no selection.
        """
        selection = self.scripts_listbox.curselection()
        if selection:
            index = selection[0]
            return self.scripts[index]
        return None

    def run_selected_script(self):
        """Run the selected script in a separate thread."""
        script = self.get_selected_script()
        if not script:
            self.log_output("No script selected", self.TAG_ERROR)
            return

        self.run_script(script)

    def run_selected_script_with_args(self):
        """Run the selected script with custom arguments."""
        script = self.get_selected_script()
        if not script:
            self.log_output("No script selected", self.TAG_ERROR)
            return

        args = self.prompt_for_arguments(script.name)
        self.run_script(script, args=args)

    def run_all_scripts(self):
        """Run all scripts sequentially without blocking GUI.

        Uses fully asynchronous execution with proper
        thread scheduling to keep UI responsive.
        """
        if not self.scripts:
            self.log_output("No scripts found", self.TAG_ERROR)
            return

        self.progress.start()
        self._set_buttons_state(tk.DISABLED)

        # Track completion state
        completed_count = [0]
        cancelled = [False]

        def run_script_async(index):
            """Run a single script asynchronously."""
            if cancelled[0] or index >= len(self.scripts):
                self._finish_run_all()
                return

            script = self.scripts[index]
            self.log_output(
                f"Running script {index + 1}/{len(self.scripts)}: {script.name}",
                self.TAG_INFO,
            )
            self.root.after(0, lambda: self.scripts_listbox.selection_clear(0, tk.END))  # noqa: BLK100
            self.root.after(0, lambda idx=index: self.scripts_listbox.selection_set(idx))  # type: ignore[misc, arg-type]
            self.root.after(0, lambda idx=index: self.scripts_listbox.see(idx))  # type: ignore[misc, arg-type]

            # Run without waiting - truly async
            success = self.run_script(script, wait=False)
            if not success:
                self.log_output(f"Failed to start {script.name}", self.TAG_ERROR)  # noqa: BLK100

            # Monitor completion and schedule next
            def monitor_and_continue():
                if self._run_all_cancelled:
                    self._finish_run_all()
                    return

                process = self.running_processes.get(script.name)
                if process and process.poll() is None:
                    # Still running, check again
                    self.root.after(100, monitor_and_continue)
                    return

                # Completed - schedule next script
                completed_count[0] += 1
                if completed_count[0] < len(self.scripts) and not cancelled[0]:
                    # Use threading for next script
                    threading.Thread(
                        target=run_script_async,
                        args=(completed_count[0],),
                        daemon=True,
                    ).start()
                else:
                    self._finish_run_all()

            self.root.after(100, monitor_and_continue)

        def start_run_all():
            """Start the async chain."""
            self._run_all_cancelled = False
            threading.Thread(target=run_script_async, args=(0,), daemon=True).start()  # noqa: BLK100

        self.root.after(0, start_run_all)

    def _set_buttons_state(self, state):
        """Set the state of control buttons."""
        self.run_button.config(state=state)
        self.run_args_button.config(state=state)
        self.run_all_button.config(state=state)
        self.stop_button.config(state=tk.NORMAL if state == tk.DISABLED else tk.DISABLED)

    def _finish_run_all(self):
        """Clean up after run_all completes."""
        self.log_output("All scripts execution completed", self.TAG_SUCCESS)
        self.update_status("Ready")
        self.progress.stop()
        self._set_buttons_state(tk.NORMAL)

    def run_script(
        self,
        script: Path,
        wait: bool = False,
        retry_count: int = 0,
        timeout: int = None,
        args: List[str] = None,
    ) -> bool:
        """Run a single script.

        Args:
            script: Path to the script to run.
            wait: If True, block until script completes and return success status.
                 If False, run asynchronously and return True immediately.
            retry_count: Current retry attempt (for internal use).
            timeout: Maximum execution time in seconds (uses config if None).
            args: Additional arguments to pass to the script.

        Returns:
            True if script started successfully, False if error occurred.
            When wait=True, returns True if script completed with return code 0.
        """
        script_name = script.name

        # Prepare command
        cmd = [sys.executable, str(script)]

        # Add auto flag for scripts that support it to prevent hanging
        auto_scripts = {"quick_deploy.py", "setup.py"}
        if script_name in auto_scripts:
            cmd.append("--auto")

        # Validate and add user-provided arguments with whitelist
        if args:
            # Whitelist of allowed arguments to prevent command injection
            allowed_args = {
                "--auto",
                "--help",
                "--verbose",
                "--quiet",
                "--dry-run",
            }
            validated_args = []
            for arg in args:
                if arg in allowed_args:
                    validated_args.append(arg)
                else:
                    self.log_output(
                        f"Warning: Argument '{arg}' not in whitelist, ignoring",
                        self.TAG_WARNING,
                    )
            cmd.extend(validated_args)

        # Use configuration timeout if not specified
        if timeout is None:
            timeout = self.get_script_timeout(script.name)

        # Start process
        try:
            # Validate CWD exists before using it
            if not self.utils_dir:
                self.log_output("Error: Utils directory not configured", self.TAG_ERROR)
                return False

            # Run from utils directory instead of root to avoid sitecustomize.py
            # being executed. This fixes issues with sitecustomize.py interfering
            # with script execution.
            cwd_path = self.utils_dir
            if not cwd_path.exists():
                self.log_output(f"Error: CWD does not exist: {cwd_path}", self.TAG_ERROR)
                self.progress.stop()
                self.update_status("Error")
                self._update_button_states()
                return False

            # Set environment variables for faster execution in validation mode
            env = os.environ.copy()
            env["APGI_TEST_MODE"] = "1"  # Enable test mode for faster sampling
            # Force UTF-8 encoding to handle emoji characters in output
            env["PYTHONIOENCODING"] = "utf-8"

            process = subprocess.Popen(  # nosec B603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                cwd=cwd_path,
                env=env,
            )

            self.running_processes[script_name] = process

            # Read output in real-time
            output_thread = threading.Thread(
                target=self._read_output,
                args=(process, script_name, timeout),
                daemon=True,
            )
            self.daemon_threads.append(output_thread)
            output_thread.start()

            # Update button states
            self._update_button_states()

            if wait:
                # Wait for process and output thread
                process.wait(timeout=timeout + 5)  # Buffer for output reading
                output_thread.join(timeout=5.0)
                return process.returncode == 0
            else:
                return True

        except Exception as e:
            self.log_output(f"Error running {script.name}: {str(e)}", self.TAG_ERROR)
            self.root.after_idle(self.progress.stop)
            self.update_status("Error")
            # Reset button states on error (thread-safe)
            self.root.after_idle(self._update_button_states)

            # Retry logic for exceptions
            if retry_count < 2:
                self.log_output(
                    f"Retrying {script.name} due to exception (attempt {retry_count + 1}/2)",
                    self.TAG_WARNING,
                )
                # Use after_idle to avoid blocking GUI
                self.root.after(
                    1000,
                    lambda: self.run_script(script, wait, retry_count + 1, timeout),
                )
                return True  # Return True to indicate retry scheduled
            return False

    def _read_output(self, process, script_name, script_timeout=None):
        """Read output from subprocess in real-time with improved Windows handling."""
        if script_timeout is None:
            script_timeout = self.get_script_timeout(script_name)
        start_time = time.time()
        return_code = None  # Track return code throughout

        # Use a queue for thread-safe communication on Windows and macOS
        output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        reader_thread = None

        # On Windows and macOS, use a separate thread for reading to avoid blocking
        # (select.select doesn't work properly with pipes on macOS)
        if platform.system() in ("Windows", "Darwin"):

            def _async_reader():
                try:
                    while True:
                        try:
                            # Use non-blocking read with timeout
                            line = process.stdout.readline()
                            if line:
                                output_queue.put((line, "output"))
                            else:
                                # Check if process has ended
                                if process.poll() is not None:
                                    break
                                time.sleep(0.05)  # Short sleep before retry
                        except (IOError, OSError) as e:
                            # Don't log every read error, just put in queue
                            output_queue.put((f"READ_ERROR: {e}", "error"))
                            break
                except Exception as e:
                    output_queue.put((f"THREAD_ERROR: {e}", "error"))

            reader_thread = threading.Thread(target=_async_reader, daemon=True)
            reader_thread.start()

        try:
            while True:
                ready = False
                output = None

                if platform.system() in ("Windows", "Darwin") and reader_thread:
                    # On Windows and macOS, read from queue with timeout
                    try:
                        output = output_queue.get(timeout=0.1)
                        ready = True
                    except queue.Empty:
                        ready = False
                        # Check if process has ended and thread is done
                        if process.poll() is not None and not reader_thread.is_alive():
                            break
                else:
                    # On other Unix-like systems, use select
                    try:
                        ready_list, _, _ = select.select([process.stdout], [], [], 1.0)
                        ready = bool(ready_list)
                    except (ValueError, select.error) as e:
                        # Handle select errors gracefully
                        logging.warning(f"Select error: {e}")
                        ready = False

                    if ready:
                        try:
                            output = process.stdout.readline()
                        except (IOError, OSError) as e:
                            logging.warning(f"Error reading process stdout: {e}")
                            ready = False

                if ready and output:
                    # On Windows and macOS, output is a tuple (text, type) from the queue
                    if isinstance(output, tuple) and len(output) == 2:
                        output_text, _output_type = output
                    else:
                        output_text = output

                    if isinstance(output_text, str) and output_text.startswith(("READ_ERROR:", "THREAD_ERROR:")):
                        # Handle reader thread errors
                        self.log_output(
                            f"Subprocess reader error: {output_text}",
                            self.TAG_ERROR,
                        )
                    elif not output_text and process.poll() is not None:
                        break
                    elif output_text:
                        # Ensure output_text is a string before calling rstrip()
                        if isinstance(output_text, str):
                            # Strip ANSI codes for clean display in GUI
                            clean_output = strip_ansi_codes(output_text.rstrip())
                            # Filter out progress bars and empty lines
                            if clean_output and not should_filter_line(clean_output):
                                # Check if this looks like JSON output and format it better
                                if self._looks_like_json(clean_output):
                                    self.log_output(
                                        f"JSON output from {script_name}:",
                                        self.TAG_INFO,
                                    )
                                    try:
                                        # Try to pretty-print JSON
                                        import json

                                        json_data = json.loads(clean_output)
                                        pretty_json = json.dumps(json_data, indent=2)
                                        self.log_output(pretty_json, self.TAG_INFO)
                                    except json.JSONDecodeError:
                                        # If not valid JSON, log as-is
                                        self.log_output(clean_output, self.TAG_INFO)
                                else:
                                    # Check for common error patterns and provide helpful messages
                                    self._check_for_common_errors(clean_output, script_name)
                                    self.log_output(clean_output, self.TAG_INFO)
                        else:
                            # Log unexpected type for debugging
                            self.log_output(
                                f"Unexpected output type: {type(output_text)}",
                                self.TAG_WARNING,
                            )

                # Check timeout
                if time.time() - start_time > script_timeout:
                    self.log_output(
                        f"Script timeout after {script_timeout} seconds",
                        self.TAG_ERROR,
                    )
                    self._terminate_process(process, script_name)
                    break

                # Check if process has ended
                if process.poll() is not None:
                    # Give reader thread a moment to finish
                    if reader_thread:
                        reader_thread.join(timeout=0.5)
                    break

                # Small sleep to prevent busy waiting
                time.sleep(0.05)

        except Exception as e:
            logging.error(f"Unexpected error in _read_output: {e}")
            self.log_output(f"Unexpected error reading script output: {e}", self.TAG_ERROR)
        finally:
            # Clean up thread
            if reader_thread and reader_thread.is_alive():
                reader_thread.join(timeout=1.0)

            # Wait for process to fully terminate and get return code
            try:
                # First check if process already ended
                return_code = process.poll()
                if return_code is None:
                    # Process still running, wait for it
                    return_code = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # If process doesn't terminate, try to get poll result
                process.terminate()
                try:
                    return_code = process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = process.wait(timeout=2.0)
            except Exception as e:
                # If wait fails, try poll as fallback
                logging.warning(f"Error waiting for process {script_name}: {e}")
                return_code = process.poll()
                if return_code is None:
                    return_code = -1  # Indicate unknown status

            elapsed_time = time.time() - start_time

            # Log result with proper return code handling
            if return_code == 0:
                self.log_output(
                    f"✅ {script_name} completed successfully in {elapsed_time:.2f}s",
                    self.TAG_SUCCESS,
                )
            elif return_code is None:
                self.log_output(
                    f"⚠️ {script_name} completed but return code unknown",
                    self.TAG_WARNING,
                )
            else:
                self.log_output(
                    f"❌ {script_name} failed with return code {return_code}",
                    self.TAG_ERROR,
                )

            # Remove from running processes
            if script_name in self.running_processes:
                del self.running_processes[script_name]

            # Re-enable run button if no processes running
            if not self.running_processes:
                self.root.after(0, self._update_button_states)

    def _terminate_process(self, process, script_name):
        """Safely terminate a process with proper cleanup."""
        try:
            process.terminate()
            # Wait for graceful termination
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.log_output(
                    "Process did not terminate gracefully, forcing kill",
                    self.TAG_WARNING,
                )
                process.kill()
                # Final wait for kill to take effect
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.log_output(
                        "Process could not be killed, may be zombie",
                        self.TAG_ERROR,
                    )
        except (ProcessLookupError, OSError) as e:
            # Process already terminated or doesn't exist
            logging.debug(f"Process termination error (likely already terminated): {e}")
        if not self.running_processes:
            self.root.after(0, self._update_button_states)

    def _force_kill_process(self, process: subprocess.Popen, script_name: str):
        """Force kill a stubborn process.

        Args:
            process: The subprocess to kill.
            script_name: Name of the script for logging.
        """
        try:
            if process.poll() is None:  # Process is still running
                process.terminate()  # Try graceful termination first
                try:
                    process.wait(timeout=5)  # Wait up to 5 seconds
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if termination fails
                    self.log_output(f"Force killed {script_name}", self.TAG_WARNING)
        except Exception as e:
            self.log_output(f"Error killing process {script_name}: {e}", self.TAG_ERROR)

    def _update_button_states(self):
        """Update button states based on running processes."""
        if self.running_processes:
            self.run_button.config(state=tk.DISABLED)
            self.run_args_button.config(state=tk.DISABLED)
            self.run_all_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.run_button.config(state=tk.NORMAL)
            self.run_args_button.config(state=tk.NORMAL)
            self.run_all_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _update_stop_button_state(self):
        """Update stop button state based on running processes."""
        if self.running_processes:
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.stop_button.config(state=tk.DISABLED)

    def stop_selected_script(self):
        """Stop the selected script if it's running."""
        script = self.get_selected_script()
        if not script:
            self.log_output("No script selected", self.TAG_ERROR)
            return

        if script.name in self.running_processes:
            try:
                process = self.running_processes[script.name]
                self._force_kill_process(process, script.name)
                self.log_output(f"Stopped: {script.name}", self.TAG_WARNING)
                if script.name in self.running_processes:
                    del self.running_processes[script.name]

                # If we are in "run all" mode, cancel the sequence
                self._run_all_cancelled = True

                # Disable stop button if no processes are running
                self._update_stop_button_state()
                self.progress.stop()
                self.update_status("Ready")
            except Exception as e:
                self.log_output(f"Error stopping {script.name}: {str(e)}", self.TAG_ERROR)
        else:
            self.log_output(f"Script {script.name} is not running", self.TAG_WARNING)

    def clear_output(self):
        """Clear the output text area."""
        self.output_text.delete(1.0, tk.END)
        self.log_output("Output cleared", self.TAG_INFO)

    def quit_application(self, event=None):
        """Quit the application safely.

        Stops all running processes and closes the application.
        """
        self._run_all_cancelled = True
        # Stop all running processes
        for script_name, process in list(self.running_processes.items()):
            try:
                if process.poll() is None:  # Process is still running
                    self._force_kill_process(process, script_name)
                    self.log_output(f"Terminated process: {script_name}", self.TAG_WARNING)
            except Exception as e:
                self.log_output(
                    f"Error terminating process {script_name}: {e}",
                    self.TAG_ERROR,
                )

        # Clear running processes
        self.running_processes.clear()

        # Clean up tracked daemon threads
        for thread in self.daemon_threads:
            if thread.is_alive():
                try:
                    thread.join(timeout=1.0)  # Wait up to 1 second for thread to finish
                except (RuntimeError, ValueError) as e:
                    self.log_output(f"Warning: Thread join error: {e}", self.TAG_WARNING)
                    pass  # Thread join error - continue cleanup
        self.daemon_threads.clear()

        # Log quit message
        self.log_output("Quitting application...", self.TAG_INFO)

        # Wait a moment for messages to be processed
        self.root.update_idletasks()

        # Quit the application
        self.root.quit()
        self.root.destroy()


def main():
    """Launch the utils runner GUI."""
    import argparse

    parser = argparse.ArgumentParser(description="APGI Utils GUI Runner")
    parser.add_argument(
        "--token",
        help="JWT authentication token for secured operations",
    )
    args = parser.parse_args()

    # Security check for GUI mode - GUI scripts run in dev mode by default
    try:
        from utils.security_gateway import Role, SecurityGateway

        gateway = SecurityGateway()

        if args.token:
            # Validate token and check GUI access permissions
            gateway.require_roles(args.token, [Role.RESEARCHER, Role.ADMIN])
        elif os.environ.get("APGI_ENFORCE_AUTH", "0") == "1":
            raise SystemExit(
                "Authentication required: pass --token to start the GUI. "
                "For local development only, set APGI_ENFORCE_AUTH=0 to enable an auto-generated dev token."
            )
        else:
            # Development mode: auto-generate a token so the GUI works without CLI args.
            # Reached only when APGI_ENFORCE_AUTH=0 is set explicitly (local dev).
            try:
                from utils.auth_adapter import get_auth_manager

                auth_manager = get_auth_manager()
                dev_token = auth_manager.generate_token("dev_user", Role.RESEARCHER, 24)  # 24 hour expiry
                print("Development mode: Generated token (valid 24 hours)")
                args.token = dev_token
            except Exception as e:
                # If token generation fails, continue without authentication
                # This allows GUI to work even if auth system has issues
                print(f"Note: Running without authentication ({e})")
    except ImportError:
        # Security gateway not available - allow GUI to run without authentication
        pass
    except PermissionError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    try:
        # Create and run the GUI
        root = tk.Tk()
        _ = UtilsRunnerGUI(root)

        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

        root.mainloop()

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("This script requires tkinter, which should come with Python.")
        sys.exit(1)
    except tk.TclError as e:
        print(f"❌ Tkinter Error: {e}")
        print("There was an error initializing the GUI.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
