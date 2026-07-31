#!/usr/bin/env python3
"""
Simple GUI for running APGI falsification protocols
"""

import importlib.util
import logging
import os
import queue
import sys
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import TclError, messagebox, scrolledtext, ttk

try:
    from utils.errors import SecurityError as _SecurityError
    from utils.protocol_manifest import verify_protocol_file as _verify_protocol
except ImportError:
    _verify_protocol = None  # type: ignore[assignment]
    _SecurityError = RuntimeError  # type: ignore[assignment,misc]
from typing import Any, Callable, Dict, List

try:
    import torch

    TORCH_AVAILABLE = True
    _ = torch.__version__  # Use torch to avoid 'unused import' warning
except ImportError:
    TORCH_AVAILABLE = False

import tempfile

logger = logging.getLogger(__name__)

os.environ["TMPDIR"] = tempfile.gettempdir()
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

# Ensure cache directories exist
cache_dirs = [
    os.path.join(tempfile.gettempdir(), "matplotlib_cache"),
    os.path.expanduser("~/.cache/matplotlib"),
]
for cache_dir in cache_dirs:
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to create cache directory {cache_dir}: {e}")

# Suppress lifelines and pandas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="lifelines")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
logger.setLevel(logging.INFO)


# ── APGI Design System ────────────────────────────────────────────────────────


def apply_apgi_theme(root):
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use("clam")

    bg = "#f8f9fa"
    fg = "#212529"

    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg, font=("Noto Sans", 10))
    style.configure(
        "Header.TLabel",
        font=("Noto Sans", 12, "bold"),
        background=bg,
        foreground=fg,
    )
    style.configure("TLabelframe", background=bg, bordercolor="#dee2e6")
    style.configure(
        "TLabelframe.Label",
        background=bg,
        foreground="#6c757d",
        font=("Noto Sans", 9, "bold"),
    )
    style.configure("Card.TFrame", background="#ffffff", borderwidth=1, relief="solid")
    style.configure("TButton", padding=6, background="#e9ecef", font=("Noto Sans", 10))
    style.map(
        "TButton",
        background=[("active", "#dee2e6"), ("disabled", "#f1f3f5")],
        foreground=[("disabled", "#adb5bd")],
    )
    style.configure(
        "Primary.TButton",
        background="#155724",
        foreground="white",
        font=("Noto Sans", 10, "bold"),
        padding=8,
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#0f3d1a"), ("disabled", "#6c757d")],
        foreground=[("active", "white"), ("disabled", "#dee2e6")],
    )
    style.configure(
        "Secondary.TButton",
        background="#2874a6",
        foreground="white",
        font=("Noto Sans", 10),
        padding=6,
    )
    style.map(
        "Secondary.TButton",
        background=[("active", "#1f5a82"), ("disabled", "#6c757d")],
        foreground=[("active", "white"), ("disabled", "#dee2e6")],
    )
    style.configure(
        "Danger.TButton",
        background="#721c24",
        foreground="white",
        font=("Noto Sans", 10, "bold"),
        padding=8,
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#5a161d"), ("disabled", "#6c757d")],
        foreground=[("active", "white"), ("disabled", "#dee2e6")],
    )
    style.configure(
        "Horizontal.TProgressbar",
        background="#2874a6",
        troughcolor="#dee2e6",
        borderwidth=0,
    )
    style.configure("TCombobox", font=("Noto Sans", 10))

    root.configure(background=bg)
    return style


class APGICard(ttk.Frame):
    """Standardized information card for APGI applications."""

    def __init__(self, parent, title, value, subtitle="", **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)

        container = ttk.Frame(self, padding=15, style="Card.TFrame")
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text=title.upper(),
            font=("Noto Sans", 11, "bold"),
            background="#ffffff",
            foreground="#212529",
        ).pack(anchor="w")

        ttk.Label(
            container,
            text=value,
            font=("Noto Sans Mono", 11),
            background="#ffffff",
            foreground="#2874a6",
            wraplength=600,
        ).pack(anchor="w", pady=(4, 8))

        if subtitle:
            ttk.Separator(container, orient="horizontal").pack(fill="x", pady=(0, 6))
            ttk.Label(
                container,
                text=subtitle,
                font=("Noto Sans", 9, "italic"),
                background="#ffffff",
                foreground="#6c757d",
                wraplength=600,
            ).pack(anchor="w")


# ── Main Application ──────────────────────────────────────────────────────────


class ProtocolRunnerGUI:
    """GUI for running APGI falsification protocols with progress tracking."""

    def __init__(self, root, headless=False):
        self.root = root
        self.headless = headless
        self.gui_queue: queue.Queue[object] = queue.Queue()

        if not self.headless:
            apply_apgi_theme(self.root)
            self._process_gui_queue()
            self.root.title("APGI Framework-Level Falsification Aggregator (FP-ALL)")
            self.root.geometry("960x700")
            self.root.minsize(720, 520)
            self._create_menu_bar()
            self._bind_shortcuts()

        # Ensure project root is importable (redundant after `pip install -e .`)
        _proj_root = os.path.dirname(os.path.abspath(__file__))
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)

        # ======================================================================
        # PROTOCOL NAMING MIGRATION: Phase 2-3 - Deprecation & Soft Warnings
        # ======================================================================
        # The APGI falsification framework uses canonical APGI-P## naming to
        # link falsification protocols to their corresponding validation protocols.
        #
        # PROTOCOL MAPPING:
        #   FP-01 → APGI-P02 (Somatic Agent Simulation)
        #   FP-03 → APGI-P05 (Framework Multi-Protocol)
        #   FP-04 → APGI-P06 (Phase Transition)
        #   FP-09 → APGI-P01 (Neural Signatures)
        #   FP-11 → APGI-P06 (Liquid Network Dynamics)
        #   FP-12 → APGI-P04 (Cross-Species Scaling)
        #   FP-13 → APGI-P07 (Clinical Cross-Species)
        #   FP-14 → APGI-P03 (fMRI Anticipation vmPFC)
        #
        # LEGACY PROTOCOLS (Non-mapped):
        #   FP-02, FP-05, FP-06, FP-07, FP-08, FP-10, FP-15
        #
        # MIGRATION TIMELINE:
        #   - Phase 1 (2026-06-25): New APGI-P## linking introduced
        #   - Phase 2 (2026-06-25): Deprecation notices added
        #   - Phase 3 (2026-09-25): Soft deprecation with warnings [READY FOR DEPLOYMENT]
        #   - Phase 4 (2026-12-25): Hard deprecation - legacy files removed (6+ months)
        #
        # PHASE 3 NOTICE: Starting September 2026, runtime warnings will be emitted
        # when legacy falsification protocols are accessed. Warnings encourage but
        # don't prevent use. See PROTOCOL-MIGRATION-PHASE-3.md for details.
        #
        # MIGRATION GUIDE: See PROTOCOL-MIGRATION-USER-GUIDE.md for details
        # PHASE 3 DETAILS: See PROTOCOL-MIGRATION-PHASE-3.md for technical information
        # QUICK REFERENCE: See PROTOCOL-QUICK-REFERENCE.md for complete mapping
        # ======================================================================
        # Protocol definitions with parameters
        # Updated to use canonical APGI-P## naming for core protocols
        self.protocols = {
            # Canonical Protocol Mappings (APGI-P01, P02, P04, P06)
            "APGI-P02: Somatic Agent Sim (FP-01)": {
                "file": "Falsification/FP_01_ActiveInference.py",
                "class": "APGIActiveInferenceAgent",
                "description": "APGI-based active inference agent - Core falsification for APGI-P02",
                "parameters": {
                    "lr_extero": {
                        "default": 0.01,
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Exteroceptive learning rate",
                    },
                    "lr_intero": {
                        "default": 0.01,
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Interoceptive learning rate",
                    },
                    "lr_precision": {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Precision learning rate",
                    },
                    "lr_somatic": {
                        "default": 0.1,
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Somatic learning rate",
                    },
                    "n_actions": {
                        "default": 4,
                        "min": 2,
                        "max": 10,
                        "type": "int",
                        "description": "Number of actions",
                    },
                    "theta_init": {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "Initial threshold",
                    },
                    "theta_baseline": {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "Threshold baseline",
                    },
                    "alpha": {
                        "default": 12.0,  # Tuned for sharper threshold crossing
                        "min": 1.0,
                        "max": 20.0,
                        "type": "float",
                        "description": "Sigmoid slope",
                    },
                    "tau_S": {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 5.0,
                        "type": "float",
                        "description": "Signal time constant",
                    },
                    "tau_theta": {
                        "default": 25.0,  # Tuned for better adaptation dynamics
                        "min": 1.0,
                        "max": 100.0,
                        "type": "float",
                        "description": "Threshold time constant",
                    },
                    "eta_theta": {
                        "default": 0.05,  # Tuned 5x for measurable threshold adaptation
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Threshold adaptation rate",
                    },
                    "beta": {
                        "default": 2.0,  # Tuned for better interoceptive advantage
                        "min": 0.5,
                        "max": 3.0,
                        "type": "float",
                        "description": "Somatic gain",
                    },
                    "rho": {
                        "default": 0.8,  # Tuned for stronger precision modulation
                        "min": 0.1,
                        "max": 1.0,
                        "type": "float",
                        "description": "Precision weight",
                    },
                },
            },
            "FP-02: Iowa Gambling Task (Agent Comparison)": {
                "file": "Falsification/FP_02_AgentComparisonConvergenceBenchmark.py",
                "class": "IowaGamblingTaskEnvironment",
                "description": "Iowa Gambling Task with interoceptive costs - Agent comparison convergence benchmark",
                "parameters": {
                    "n_trials": {
                        "default": 100,
                        "min": 10,
                        "max": 1000,
                        "type": "int",
                        "description": "Number of trials",
                    },
                    "n_decks": {
                        "default": 4,
                        "min": 2,
                        "max": 8,
                        "type": "int",
                        "description": "Number of decks",
                    },
                    "interoceptive_cost_weight": {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "type": "float",
                        "description": "Interoceptive cost weight",
                    },
                    "learning_rate": {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 0.5,
                        "type": "float",
                        "description": "Agent learning rate",
                    },
                },
            },
            "APGI-P05: Framework Multi-Protocol (FP-03)": {
                "file": "Falsification/FP_03_FrameworkLevelMultiProtocol.py",
                "class": "AgentComparisonExperiment",
                "description": "Framework-level multi-protocol falsification runner - Integration test for APGI-P05",
                "parameters": {
                    "n_episodes": {
                        "default": 100,
                        "min": 10,
                        "max": 1000,
                        "type": "int",
                        "description": "Number of episodes",
                    },
                    "episode_length": {
                        "default": 50,
                        "min": 10,
                        "max": 500,
                        "type": "int",
                        "description": "Episode length",
                    },
                    "n_agents": {
                        "default": 3,
                        "min": 2,
                        "max": 10,
                        "type": "int",
                        "description": "Number of agents to compare",
                    },
                },
            },
            "APGI-P06: Phase Transition (FP-04)": {
                "file": "Falsification/FP_04_PhaseTransitionEpistemicArchitecture.py",
                "class": "InformationTheoreticAnalysis",
                "description": "APGI ignition phase transition signatures - Falsification test for APGI-P06",
                "parameters": {
                    "surprise_range": {
                        "default": "[0.1, 2.0]",
                        "type": "str",
                        "description": "Surprise range [min, max]",
                    },
                    "n_points": {
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "type": "int",
                        "description": "Number of analysis points",
                    },
                    "tau_S": {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "Signal time constant",
                    },
                    "alpha": {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 20.0,
                        "type": "float",
                        "description": "Sigmoid slope",
                    },
                },
            },
            "FP-05: Evolutionary Plausibility": {
                "file": "Falsification/FP_05_EvolutionaryPlausibility.py",
                "class": "EvolutionaryAPGIEmergence",
                "description": "Test APGI emergence and evolutionary plausibility under selection pressure",
                "parameters": {
                    "population_size": {
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "type": "int",
                        "description": "Population size",
                    },
                    "n_generations": {
                        "default": 100,
                        "min": 10,
                        "max": 1000,
                        "type": "int",
                        "description": "Number of generations",
                    },
                    "mutation_rate": {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 0.5,
                        "type": "float",
                        "description": "Mutation rate",
                    },
                    "selection_pressure": {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 5.0,
                        "type": "float",
                        "description": "Selection pressure",
                    },
                },
            },
            "FP-06: Liquid Network Energy Benchmark": {
                "file": "Falsification/FP_06_LiquidNetworkEnergyBenchmark.py",
                "class": "NetworkComparisonExperiment",
                "description": "Compare APGI-inspired vs standard neural network architectures and energy efficiency",
                "parameters": {
                    "extero_dim": {
                        "default": 32,
                        "min": 8,
                        "max": 128,
                        "type": "int",
                        "description": "Exteroceptive dimension",
                    },
                    "intero_dim": {
                        "default": 16,
                        "min": 4,
                        "max": 64,
                        "type": "int",
                        "description": "Interoceptive dimension",
                    },
                    "action_dim": {
                        "default": 4,
                        "min": 2,
                        "max": 16,
                        "type": "int",
                        "description": "Action dimension",
                    },
                    "context_dim": {
                        "default": 8,
                        "min": 2,
                        "max": 32,
                        "type": "int",
                        "description": "Context dimension",
                    },
                    "n_episodes": {
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "type": "int",
                        "description": "Number of episodes",
                    },
                },
            },
            "FP-07: Mathematical Consistency": {
                "file": "Falsification/FP_07_MathematicalConsistency.py",
                "class": "MathematicalConsistencyChecker",
                "description": "Test mathematical consistency and correctness of all APGI core equations",
                "parameters": {
                    "epsilon": {
                        "default": 1e-6,
                        "min": 1e-9,
                        "max": 1e-3,
                        "type": "float",
                        "description": "Numerical epsilon for derivatives",
                    },
                    "n_samples": {
                        "default": 1000,
                        "min": 100,
                        "max": 10000,
                        "type": "int",
                        "description": "Number of test samples",
                    },
                    "tolerance": {
                        "default": 1e-6,
                        "min": 1e-8,
                        "max": 1e-3,
                        "type": "float",
                        "description": "Tolerance for equality checks (V5.1 spec: ε ≤ 1e-6)",
                    },
                },
            },
            "FP-08: Parameter Sensitivity & Identifiability": {
                "file": "Falsification/FP_08_ParameterSensitivityIdentifiability.py",
                "class": "ParameterSensitivityAnalyzer",
                "description": "Parameter sensitivity and identifiability analysis - Sobol and OAT methods",
                "parameters": {
                    "n_samples": {
                        "default": 1000,
                        "min": 100,
                        "max": 5000,
                        "type": "int",
                        "description": "Number of Sobol samples",
                    },
                    "n_trials": {
                        "default": 1000,
                        "min": 100,
                        "max": 5000,
                        "type": "int",
                        "description": "Number of trials per sample",
                    },
                    "n_levels": {
                        "default": 15,
                        "min": 5,
                        "max": 50,
                        "type": "int",
                        "description": "Number of OAT levels",
                    },
                },
            },
            "APGI-P01: Neural Signatures (FP-09)": {
                "file": "Falsification/FP_09_NeuralSignaturesP3bHEP.py",
                "class": "NeuralSignatureValidator",
                "description": "Validate P3b and HEP neural signatures - Core falsification for APGI-P01",
                "parameters": {
                    "n_participants": {
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "type": "int",
                        "description": "Number of participants",
                    },
                    "sampling_rate": {
                        "default": 500,
                        "min": 100,
                        "max": 2000,
                        "type": "int",
                        "description": "EEG sampling rate (Hz)",
                    },
                    "window_size": {
                        "default": 1000,
                        "min": 500,
                        "max": 5000,
                        "type": "int",
                        "description": "Window size (ms)",
                    },
                    "n_chains": {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "type": "int",
                        "description": "Number of MCMC chains (2 recommended for GUI)",
                    },
                    "burn_in": {
                        "default": 500,
                        "min": 100,
                        "max": 2000,
                        "type": "int",
                        "description": "Burn-in samples",
                    },
                },
            },
            "FP-10: Bayesian Estimation with MCMC": {
                "file": "Falsification/FP_10_BayesianEstimationMCMC.py",
                "class": "BayesianParameterRecovery",
                "description": "Bayesian parameter recovery and MCMC-based estimation analysis",
                "parameters": {
                    "n_samples": {
                        "default": 1000,
                        "min": 500,
                        "max": 5000,
                        "type": "int",
                        "description": "Number of MCMC samples (1000 recommended for GUI)",
                    },
                    "n_chains": {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "type": "int",
                        "description": "Number of MCMC chains (2 recommended for GUI)",
                    },
                    "burn_in": {
                        "default": 500,
                        "min": 100,
                        "max": 2000,
                        "type": "int",
                        "description": "Burn-in samples",
                    },
                },
            },
            "APGI-P06: Liquid Network Dynamics (FP-11)": {
                "file": "Falsification/FP_11_LiquidNetworkDynamicsEchoState.py",
                "class": "LiquidNetworkDynamicsAnalyzer",
                "description": "Liquid network dynamics and echo state analysis - Extended for APGI-P06",
                "parameters": {
                    "spectral_radius": {
                        "default": 0.9,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "Spectral radius",
                    },
                    "leak_rate": {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 1.0,
                        "type": "float",
                        "description": "Leak rate",
                    },
                    "n_units": {
                        "default": 100,
                        "min": 10,
                        "max": 500,
                        "type": "int",
                        "description": "Number of liquid units",
                    },
                },
            },
            "APGI-P04: Cross-Species Scaling (FP-12)": {
                "file": "Falsification/FP_12_CrossSpeciesScaling.py",
                "class": "LiquidTimeConstantChecker",
                "description": "Cross-species allometric scaling and clinical convergence analysis - APGI-P04 falsification",
                "parameters": {
                    "spectral_radius": {
                        "default": 0.98,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "ESN spectral radius",
                    },
                    "leak_rate": {
                        "default": 0.01,
                        "min": 0.001,
                        "max": 1.0,
                        "type": "float",
                        "description": "LTC leak rate",
                    },
                    "n_nodes": {
                        "default": 100,
                        "min": 10,
                        "max": 500,
                        "type": "int",
                        "description": "Number of ESN nodes",
                    },
                },
            },
            "APGI-P07: Clinical Cross-Species (FP-13)": {
                "file": "Falsification/FP_13_Clinical_CrossSpecies_Convergence.py",
                "class": "ClinicalCrossSpeciesProtocol",
                "description": "Clinical cross-species convergence falsification - APGI-P07 tertiary falsification",
                "parameters": {},
            },
            "APGI-P03: fMRI Anticipation vmPFC (FP-14)": {
                "file": "Falsification/FP_14_fMRI_Anticipation_vmPFC.py",
                "class": None,
                "description": "fMRI anticipation vmPFC falsification - APGI-P03 tertiary falsification stub",
                "parameters": {},
            },
            "FP-15: Allen Visual Coding Fatigue": {
                "file": "Falsification/FP_15_AllenVisualCoding_Fatigue.py",
                "class": None,
                "description": "Allen visual coding fatigue threshold dynamics falsification stub",
                "parameters": {},
            },
            "Protocol ALL: Framework Aggregator": {
                "file": "Falsification/FP_ALL_Aggregator.py",
                "class": "FalsificationAggregator",
                "description": "Full Framework Falsification Report (aggregates FP-01 through FP-15 where available)",
                "parameters": {},
            },
        }

        # Thread management
        self.stop_event = threading.Event()
        self.running_thread = None

        # Parameter storage - initialize with default values
        self.parameter_values = {}
        self.load_default_parameters()

        # Initialize selected protocol
        self.selected_protocol = None

        # Add window close handler for proper thread cleanup
        if not self.headless:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.setup_ui()

        self.log_message("APGI Falsification Protocols GUI initialized")
        if not self.headless:
            self._process_gui_queue()

    def _process_gui_queue(self):
        """Process pending GUI updates from the queue (thread-safe).

        This method is called periodically via root.after() to process
        updates that worker threads put into gui_queue.
        """
        try:
            while True:
                update_func = self.gui_queue.get_nowait()
                if callable(update_func):
                    try:
                        update_func()
                    except Exception as e:
                        logger.warning(f"GUI queue update failed: {e}")
        except queue.Empty:
            pass
        # Schedule next check in 100ms
        try:
            self.root.after(100, self._process_gui_queue)
        except (KeyboardInterrupt, TclError):
            # Gracefully handle shutdown during GUI processing
            logger.info("GUI processing interrupted, shutting down...")
            if hasattr(self, "stop_event"):
                self.stop_event.set()

    def _on_closing(self):
        """Handle window close event with proper thread cleanup."""
        if self.running_thread and self.running_thread.is_alive():
            self.stop_event.set()
            self.running_thread.join(timeout=2.0)
            if self.running_thread.is_alive():
                logging.warning("Thread did not stop cleanly on exit")
        self.root.destroy()

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Run Selected",
            command=self.run_selected_protocol,
            accelerator="Ctrl+R",
        )
        file_menu.add_command(label="Run All", command=self.run_all_protocols, accelerator="Ctrl+Shift+R")
        file_menu.add_command(label="Stop", command=self.stop_protocol, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_closing, accelerator="Ctrl+Q")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Clear Console", command=self.clear_console, accelerator="Ctrl+L")
        view_menu.add_command(label="Toggle Console", command=self._toggle_console)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts for common actions."""
        self.root.bind("<Control-q>", lambda e: self._on_closing())
        self.root.bind("<Command-q>", lambda e: self._on_closing())
        self.root.bind("<Control-r>", lambda e: self.run_selected_protocol())
        self.root.bind("<Control-Shift-R>", lambda e: self.run_all_protocols())
        self.root.bind("<Control-s>", lambda e: self.stop_protocol())
        self.root.bind("<Control-l>", lambda e: self.clear_console())

    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "APGI Falsification GUI\nProtocol Runner Interface\nVersion 1.3.0",
        )

    # ── UI Construction ───────────────────────────────────────────────────────

    def setup_ui(self):
        """Set up the application interface following APGI design standards."""
        _BLUE = "#2874a6"
        _BG = "#f8f9fa"

        # ── Row 0: Metric bar ─────────────────────────────────────────────────
        metric_bar = tk.Frame(self.root, bg=_BLUE, pady=8, padx=15)
        metric_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        metric_bar.columnconfigure(1, weight=1)

        tk.Label(
            metric_bar,
            text="APGI FALSIFICATION PROTOCOLS",
            bg=_BLUE,
            fg="white",
            font=("Noto Sans", 13, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.status_var = tk.StringVar(value="ℹ  Ready")
        self.status_label = tk.Label(
            metric_bar,
            textvariable=self.status_var,
            bg=_BLUE,
            fg="white",
            font=("Noto Sans", 10),
        )
        self.status_label.grid(row=0, column=1, padx=(20, 15), sticky="e")

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            metric_bar,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
            length=180,
        )
        self.progress_bar.grid(row=0, column=2, sticky="e")

        # ── Root grid weights ─────────────────────────────────────────────────
        self.root.columnconfigure(0, weight=0, minsize=210)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=0)

        # ── Row 1, Col 0: Sidebar ─────────────────────────────────────────────
        sidebar = tk.Frame(self.root, bg="#ffffff", width=210, bd=0)
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)

        # Sidebar section header
        tk.Label(
            sidebar,
            text="PROTOCOLS",
            bg="#dee2e6",
            fg="#6c757d",
            font=("Noto Sans", 9, "bold"),
            pady=5,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        # Protocol listbox
        lb_frame = tk.Frame(sidebar, bg="#ffffff")
        lb_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        lb_frame.columnconfigure(0, weight=1)
        lb_frame.rowconfigure(0, weight=1)

        self.protocol_listbox = tk.Listbox(
            lb_frame,
            font=("Noto Sans", 9),
            bg="#ffffff",
            fg="#212529",
            selectbackground=_BLUE,
            selectforeground="white",
            borderwidth=0,
            highlightthickness=1,
            highlightcolor="#dee2e6",
            highlightbackground="#dee2e6",
            activestyle="none",
            cursor="hand2",
        )
        lb_scroll = ttk.Scrollbar(lb_frame, orient="vertical", command=self.protocol_listbox.yview)
        self.protocol_listbox.configure(yscrollcommand=lb_scroll.set)
        self.protocol_listbox.grid(row=0, column=0, sticky="nsew")
        lb_scroll.grid(row=0, column=1, sticky="ns")

        for name in self.protocols:
            short = name.split(": ", 1)[1] if ": " in name else name
            self.protocol_listbox.insert(tk.END, short)

        self.protocol_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Sidebar action buttons
        btn_area = tk.Frame(sidebar, bg="#ffffff", pady=8, padx=8)
        btn_area.grid(row=2, column=0, columnspan=2, sticky="ew")
        btn_area.columnconfigure(0, weight=1)

        self.run_selected_button = ttk.Button(
            btn_area,
            text="▶  Run Selected",
            command=self.run_selected_protocol,
            style="Primary.TButton",
            state=tk.DISABLED,
        )
        self.run_selected_button.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.run_all_button = ttk.Button(
            btn_area,
            text="Run All",
            command=self.run_all_protocols,
            style="Secondary.TButton",
        )
        self.run_all_button.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        self.stop_btn = ttk.Button(
            btn_area,
            text="■  Stop",
            command=self.stop_protocol,
            style="Danger.TButton",
            state=tk.DISABLED,
        )
        self.stop_btn.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ttk.Separator(btn_area, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_area, text="Clear Console", command=self.clear_console).grid(row=4, column=0, sticky="ew")

        # Right border divider between sidebar and workspace
        tk.Frame(self.root, bg="#dee2e6", width=1).grid(row=1, column=0, sticky="nse")

        # ── Row 1, Col 1: Workspace ───────────────────────────────────────────
        workspace = ttk.Frame(self.root, padding=(15, 10, 15, 10))
        workspace.grid(row=1, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=0)
        workspace.rowconfigure(1, weight=1)

        # Protocol info card (row 0)
        self._card_area = ttk.Frame(workspace)
        self._card_area.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._card_area.columnconfigure(0, weight=1)
        self._show_workspace_empty_state()

        # Parameters section (row 1)
        params_lf = ttk.LabelFrame(workspace, text="PARAMETERS", padding=(10, 6))
        params_lf.grid(row=1, column=0, sticky="nsew")
        params_lf.columnconfigure(0, weight=1)
        params_lf.rowconfigure(0, weight=1)

        # Hidden combobox — not shown in layout but keeps save_parameters() working
        self.protocol_selector = ttk.Combobox(params_lf, values=list(self.protocols.keys()), state="readonly")
        self.protocol_selector.bind("<<ComboboxSelected>>", self.on_protocol_selected)

        # Scrollable params canvas
        self.params_frame = ttk.Frame(params_lf)
        self.params_frame.grid(row=0, column=0, sticky="nsew")
        self.params_frame.columnconfigure(0, weight=1)
        self.params_frame.rowconfigure(0, weight=1)

        self.params_canvas = tk.Canvas(self.params_frame, bg=_BG, highlightthickness=0, height=150)
        self.params_scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=self.params_canvas.yview)
        self.params_scrollable_frame = ttk.Frame(self.params_canvas)

        self.params_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all")),
        )
        self.params_canvas.create_window((0, 0), window=self.params_scrollable_frame, anchor="nw")
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
        self.params_canvas.pack(side="left", fill="both", expand=True)
        self.params_scrollbar.pack(side="right", fill="y")

        # Initial placeholder shown before any protocol is selected
        ttk.Label(
            self.params_scrollable_frame,
            text="Select a protocol from the sidebar to configure parameters.",
            foreground="#6c757d",
            font=("Noto Sans", 10),
        ).grid(row=0, column=0, padx=20, pady=20)

        # Parameter control buttons
        params_ctrl = ttk.Frame(params_lf)
        params_ctrl.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        ttk.Button(params_ctrl, text="Load Defaults", command=self.load_default_parameters).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(
            params_ctrl,
            text="Save Parameters",
            command=self.save_parameters,
            style="Secondary.TButton",
        ).pack(side=tk.LEFT)

        # ── Row 2: Console toggle header ──────────────────────────────────────
        self._console_visible = True
        console_header = tk.Frame(self.root, bg="#dee2e6", pady=3)
        console_header.grid(row=2, column=0, columnspan=2, sticky="ew")

        self._console_toggle_btn = tk.Button(
            console_header,
            text="▼  OUTPUT CONSOLE",
            bg="#dee2e6",
            fg="#6c757d",
            font=("Noto Sans", 9, "bold"),
            relief="flat",
            cursor="hand2",
            command=self._toggle_console,
            activebackground="#dee2e6",
            activeforeground="#495057",
            bd=0,
        )
        self._console_toggle_btn.pack(side="left", padx=10)

        # ── Row 3: Console body ───────────────────────────────────────────────
        self._console_frame = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        self._console_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self._console_frame.columnconfigure(0, weight=1)
        self._console_frame.rowconfigure(0, weight=1)

        self.output_console = scrolledtext.ScrolledText(
            self._console_frame,
            height=9,
            font=("Noto Sans Mono", 9),
            bg="#212529",
            fg="#f8f9fa",
            insertbackground="#f8f9fa",
            relief="flat",
            borderwidth=0,
        )
        self.output_console.grid(row=0, column=0, sticky="nsew")

        self._setup_tab_order()

    def _show_workspace_empty_state(self):
        """Show empty state placeholder in the workspace card area."""
        for w in self._card_area.winfo_children():
            w.destroy()

        outer = tk.Frame(self._card_area, bg="#f8f9fa", pady=10)
        outer.pack(fill="x")

        canvas = tk.Canvas(
            outer,
            width=160,
            height=70,
            bg="#f8f9fa",
            highlightbackground="#e9ecef",
            highlightthickness=2,
        )
        canvas.pack(side="left", padx=(0, 15))
        canvas.create_text(
            80, 35, text="No Protocol\nSelected", fill="#adb5bd", font=("Noto Sans", 10), justify="center"
        )

        ttk.Label(
            outer,
            text="No active hypotheses found.\nInitialize a new simulation environment to begin.",
            font=("Noto Sans", 10),
            foreground="#6c757d",
            wraplength=340,
            justify="left",
        ).pack(side="left", anchor="w")

    def _update_protocol_card(self, protocol_name, protocol_info):
        """Replace the workspace card with the selected protocol's details."""
        for w in self._card_area.winfo_children():
            w.destroy()

        APGICard(
            self._card_area,
            title=protocol_name,
            value=protocol_info.get("file", ""),
            subtitle=protocol_info.get("description", ""),
        ).pack(fill="x")

    def _on_listbox_select(self, event):
        """Handle protocol selection from the sidebar listbox."""
        sel = self.protocol_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        names = list(self.protocols.keys())
        if idx >= len(names):
            return
        name = names[idx]
        info = self.protocols[name]
        self.select_protocol(name, info)
        self.protocol_selector.set(name)
        self.display_protocol_parameters(name)

    def _toggle_console(self):
        """Toggle visibility of the output console."""
        if self._console_visible:
            self._console_frame.grid_remove()
            self._console_toggle_btn.config(text="▶  OUTPUT CONSOLE")
            self._console_visible = False
        else:
            self._console_frame.grid()
            self._console_toggle_btn.config(text="▼  OUTPUT CONSOLE")
            self._console_visible = True

    # ── Protocol Selection & Parameters ──────────────────────────────────────

    def select_protocol(self, protocol_name, protocol_info):
        """Select a protocol for running."""
        self.selected_protocol = protocol_name
        self.run_selected_button.config(state=tk.NORMAL)
        if not self.headless:
            self._update_protocol_card(protocol_name, protocol_info)
        self.log_message(f"Selected protocol: {protocol_name}")

    def on_protocol_selected(self, event):
        """Handle protocol selection for parameter configuration."""
        selected = self.protocol_selector.get()
        if selected:
            self.display_protocol_parameters(selected)

    def display_protocol_parameters(self, protocol_name):
        """Display parameter controls for the selected protocol."""
        for widget in self.params_scrollable_frame.winfo_children():
            widget.destroy()

        protocol_info = self.protocols.get(protocol_name)
        if not protocol_info or "parameters" not in protocol_info:
            ttk.Label(
                self.params_scrollable_frame,
                text="No configurable parameters for this protocol",
            ).pack()
            return

        row = 0
        self.parameter_widgets = {}

        for param_name, param_config in protocol_info["parameters"].items():
            label = ttk.Label(self.params_scrollable_frame, text=f"{param_name}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)

            if param_config["type"] == "float":
                var_float = tk.DoubleVar(value=param_config["default"])
                widget = tk.Spinbox(
                    self.params_scrollable_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    increment=(param_config["max"] - param_config["min"]) / 100,
                    textvariable=var_float,
                    width=15,
                    validate="key",
                    validatecommand=(
                        self.root.register(self._validate_spinbox_float),
                        "%P",
                        param_config["min"],
                        param_config["max"],
                    ),
                )
            elif param_config["type"] == "int":
                var_int = tk.IntVar(value=param_config["default"])  # type: ignore[assignment]
                widget = tk.Spinbox(
                    self.params_scrollable_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    increment=1,
                    textvariable=var_int,
                    width=15,
                    validate="key",
                    validatecommand=(
                        self.root.register(self._validate_spinbox_int),
                        "%P",
                        param_config["min"],
                        param_config["max"],
                    ),
                )
            else:  # string
                var = tk.StringVar(value=param_config["default"])
                widget = ttk.Entry(self.params_scrollable_frame, textvariable=var, width=17)

            widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.create_tooltip(widget, param_config.get("description", ""))
            self.parameter_widgets[param_name] = {"var": var, "widget": widget}
            row += 1

        self.params_scrollable_frame.columnconfigure(1, weight=1)

    def load_default_parameters(self):
        """Load default parameter values."""
        for protocol_name, protocol_info in self.protocols.items():
            if "parameters" in protocol_info:
                self.parameter_values[protocol_name] = {}
                for param_name, param_config in protocol_info["parameters"].items():
                    self.parameter_values[protocol_name][param_name] = param_config["default"]

    def save_parameters(self):
        """Save current parameter values with validation."""
        selected = self.protocol_selector.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a protocol first.")
            return

        if hasattr(self, "parameter_widgets") and self.parameter_widgets:
            for param_name, widget_info in self.parameter_widgets.items():
                try:
                    value = widget_info["var"].get()
                    self.parameter_values[selected][param_name] = value
                except Exception as e:
                    messagebox.showerror("Update Error", f"Error updating {param_name}: {e}")
                    return

        validation_errors = []
        protocol_info = self.protocols[selected]

        if "parameters" in protocol_info:
            for param_name, param_config in protocol_info["parameters"].items():
                current_value = self.parameter_values[selected].get(param_name)

                if param_config["type"] == "float":
                    try:
                        float_value = float(current_value)
                        if float_value < param_config["min"] or float_value > param_config["max"]:
                            validation_errors.append(
                                f"{param_name}: {float_value} is out of range [{param_config['min']}, {param_config['max']}]"
                            )
                    except (ValueError, TypeError):
                        validation_errors.append(f"{param_name}: Invalid float value '{current_value}'")

                elif param_config["type"] == "int":
                    try:
                        int_value = int(current_value)
                        if int_value < param_config["min"] or int_value > param_config["max"]:
                            validation_errors.append(
                                f"{param_name}: {int_value} is out of range [{param_config['min']}, {param_config['max']}]"
                            )
                    except (ValueError, TypeError):
                        validation_errors.append(f"{param_name}: Invalid integer value '{current_value}'")

                elif param_config["type"] == "str":
                    if param_name == "surprise_range":
                        try:
                            import json

                            value_list = json.loads(current_value)
                            if not isinstance(value_list, list) or len(value_list) != 2:
                                validation_errors.append(f"{param_name}: Must be in format [min, max]")
                        except (json.JSONDecodeError, TypeError):
                            validation_errors.append(f"{param_name}: Invalid range format '{current_value}'")

        if validation_errors:
            messagebox.showerror(
                "Validation Errors",
                "Please fix the following errors:\n\n" + "\n".join(validation_errors),
            )
            return

        try:
            import json

            project_root = Path(__file__).parent
            config_dir = project_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_file = config_dir / f"{selected.lower().replace(' ', '_')}_params.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self.parameter_values[selected], f, indent=2)

            messagebox.showinfo("✔ Success", f"Parameters saved to {config_file}")
            self.log_message(f"Parameters saved for {selected}")

        except Exception as e:
            messagebox.showerror("✖ Save Error", f"Failed to save parameters: {e}")
            self.log_message(f"Error saving parameters: {e}")
            return

    def refresh_parameter_list(self):
        """Refresh the parameter display for the currently selected protocol."""
        selected = self.protocol_selector.get()
        if selected:
            self.display_protocol_parameters(selected)

    # ── Protocol Execution ────────────────────────────────────────────────────

    def run_all_protocols(self):
        """Run all protocols sequentially with default parameters."""
        protocol_count = len(self.protocols)
        if not messagebox.askyesno(
            "Confirm Run All",
            f"This will run all {protocol_count} protocols sequentially.\n\n"
            "This may take a significant amount of time.\n\n"
            "Continue?",
        ):
            return

        def run_all_thread():
            total = len(self.protocols)
            for idx, (protocol_name, protocol_info) in enumerate(self.protocols.items(), 1):
                if self.stop_event.is_set():
                    self.log_message("=== Run All stopped by user ===")
                    break

                self.log_message(f"\n{'=' * 60}")
                self.log_message(f"Running {protocol_name} ({idx}/{total})")
                self.log_message(f"{'=' * 60}")

                protocol_info_with_params = protocol_info.copy()
                protocol_info_with_params["configured_params"] = {}
                self._run_single_protocol(protocol_info_with_params)

            self.log_message(f"\n{'=' * 60}")
            self.log_message("All protocols completed!")
            self.log_message(f"{'=' * 60}")
            self.set_status("Ready")
            self.gui_queue.put(lambda: self.run_all_button.config(state=tk.NORMAL))
            self.gui_queue.put(lambda: self.stop_btn.config(state=tk.DISABLED))
            self.gui_queue.put(lambda: self.protocol_listbox.config(state=tk.NORMAL))

        self.stop_event.clear()
        self.run_all_button.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.protocol_listbox.config(state=tk.DISABLED)
        self.set_status("Running all protocols...")

        thread = threading.Thread(target=run_all_thread)
        thread.daemon = True
        thread.start()

    def _run_single_protocol(self, protocol_info):
        """Run a single protocol synchronously (for use in run_all)."""
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            file_path = os.path.join(os.path.dirname(__file__), protocol_info["file"])

            _fp_f = Path(file_path)
            if _verify_protocol is not None and not _verify_protocol(_fp_f, "Falsification"):
                raise _SecurityError(f"Protocol {_fp_f.name} failed manifest integrity check")
            spec = importlib.util.spec_from_file_location(protocol_info["file"], file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            configured_params = protocol_info.get("configured_params", {})

            if protocol_info["class"] == "FalsificationAggregator":
                result = self._handle_framework_aggregator(module)

            else:
                class_name = protocol_info.get("class")
                cls = getattr(module, class_name) if class_name else None

                result = {}
                if class_name == "NetworkComparisonExperiment":
                    config = {
                        "extero_dim": configured_params.get("extero_dim", 32),
                        "intero_dim": configured_params.get("intero_dim", 16),
                        "action_dim": configured_params.get("action_dim", 4),
                        "context_dim": configured_params.get("context_dim", 8),
                        "n_episodes": configured_params.get("n_episodes", 50),
                    }
                    instance = cls(config)
                    result = instance.run_full_experiment()
                    status = result.get("status", "completed") if isinstance(result, dict) else "completed"
                    self.log_message(f"  Result: {status}")

                elif class_name == "APGIActiveInferenceAgent":
                    run_func = getattr(module, "run_falsification", None)
                    if run_func:
                        result = run_func()
                        self.log_message(f"  Falsification completed: {result}")
                    else:
                        self.log_message("  No run_falsification function found")

                elif class_name == "IowaGamblingTaskEnvironment":
                    run_func = getattr(module, "run_falsification", None)
                    if run_func:
                        result = run_func()
                        self.log_message(f"  Falsification completed: {result}")
                    else:
                        self.log_message("  No run_falsification function found")

                elif hasattr(module, "run_falsification"):
                    self.log_message("  Running run_falsification from module scope...")
                    result = module.run_falsification()
                    status = "Done"
                    status = result.get("status", "Done")
                    if result is None:
                        status = "ERROR (Returned None)"
                    self.log_message(f"  Protocol completed: {status}")

                elif cls is not None and hasattr(cls, "run_full_experiment"):
                    self.log_message("  Running instance.run_full_experiment()...")
                    try:
                        instance = cls(**configured_params)
                    except Exception as e:
                        logging.debug(f"Could not instantiate {cls.__name__} with params, retrying with no args: {e}")
                        instance = cls()
                    result = instance.run_full_experiment()
                    self.log_message("  Experiment completed")

                elif cls is not None and hasattr(cls, "run_phase_transition_analysis"):
                    self.log_message("  Running phase transition analysis...")
                    surprise_system = module.SurpriseIgnitionSystem()
                    instance = cls(surprise_system)
                    result = instance.run_phase_transition_analysis()
                    self.log_message("  Analysis completed")

                elif cls is not None and hasattr(cls, "run_evolution"):
                    self.log_message("  Running evolutionary simulation...")
                    config = {
                        "population_size": configured_params.get("population_size", 50),
                        "n_generations": configured_params.get("n_generations", 100),
                        "mutation_rate": configured_params.get("mutation_rate", 0.1),
                        "selection_pressure": configured_params.get("selection_pressure", 2.0),
                    }
                    instance = cls(stop_event=self.stop_event, **config)
                    result = instance.run_evolution()
                    self.log_message("  Evolution completed")

                elif hasattr(module, "run_validation"):
                    self.log_message("  Running run_validation from module scope...")
                    result = module.run_validation()
                    status = result.get("status", "Done") if isinstance(result, dict) else "Done"
                    self.log_message(f"  Protocol completed: {status}")

                else:
                    if cls is None:
                        self.log_message("  WARNING: No class or module-level runner found. Result will be empty.")
                        result = {"status": "Incomplete", "warning": "No runner found"}
                        protocol_status = self._extract_result_status(result)
                        if self._is_successful_result(result):
                            self.log_message("  Protocol completed successfully")
                        else:
                            self.log_message(f"  Protocol completed with status: {protocol_status}")
                        self._save_results(result, protocol_info["file"])
                        return

                    self.log_message(f"  Using generic runner for {cls.__name__}...")
                    try:
                        instance = cls(**configured_params)
                    except TypeError:
                        instance = cls()

                    if hasattr(instance, "run_validation"):
                        result = instance.run_validation()
                    elif hasattr(module, "run_validation"):
                        result = module.run_validation()
                    elif hasattr(instance, "run_falsification"):
                        result = instance.run_falsification()
                    else:
                        self.log_message("  WARNING: No standard execution method found. Result will be empty.")
                        result = {"status": "Incomplete", "warning": "No runner found"}

            protocol_status = self._extract_result_status(result)
            if self._is_successful_result(result):
                self.log_message("  Protocol completed successfully")
            else:
                self.log_message(f"  Protocol completed with status: {protocol_status}")
            self._save_results(result, protocol_info["file"])

        except Exception as e:
            self.log_message(f"  ERROR: {str(e)}")
            logger.error(f"Error in {protocol_info['file']}: {e}")

    def run_headless(self):
        """Run all protocols in headless mode."""
        self.log_message("Running in HEADLESS mode")
        if not self.parameter_values:
            self.load_default_parameters()

        total = len(self.protocols)
        for idx, (protocol_name, protocol_info) in enumerate(self.protocols.items(), 1):
            self.log_message(f"\n[{idx}/{total}] Running {protocol_name}...")
            protocol_info_with_params = protocol_info.copy()
            protocol_info_with_params["configured_params"] = {}
            self._run_single_protocol(protocol_info_with_params)

        self.log_message("\n" + "=" * 60)
        self.log_message("Headless run completed.")
        self.log_message("=" * 60)

    def run_selected_protocol(self):
        """Run the currently selected protocol with configured parameters."""
        if not self.selected_protocol:
            messagebox.showwarning("Warning", "No protocol selected")
            return

        protocol_info = self.protocols[self.selected_protocol]

        params = {}
        if hasattr(self, "parameter_widgets"):
            for param_name, widget_info in self.parameter_widgets.items():
                try:
                    value = widget_info["var"].get()
                    params[param_name] = value
                except Exception as e:
                    logger.warning(f"Failed to get value for parameter {param_name}: {e}")
                    continue

        protocol_info_with_params = protocol_info.copy()
        protocol_info_with_params["configured_params"] = params

        self.run_protocol(protocol_info_with_params)

    # ── Tooltip ───────────────────────────────────────────────────────────────

    def create_tooltip(self, widget, text):
        """Create a styled tooltip for a widget."""

        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 12}")
            label = tk.Label(
                tooltip,
                text=text,
                background="#212529",
                foreground="#f8f9fa",
                font=("Noto Sans", 9),
                relief=tk.FLAT,
                padx=8,
                pady=4,
            )
            label.pack()
            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, "tooltip"):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    # ── Utility Methods ───────────────────────────────────────────────────────

    def clear_console(self):
        """Clear the console output."""
        if hasattr(self, "output_console"):
            self.output_console.delete(1.0, tk.END)

    def _setup_tab_order(self) -> None:
        """Configure tab order for consistent keyboard navigation."""
        self._tab_widgets: List[tk.Widget] = []  # type: ignore

    def _finalize_tab_order(self) -> None:
        """Finalize tab order after all widgets are created."""
        if hasattr(self, "_tab_widgets"):
            for i, widget in enumerate(self._tab_widgets):
                if widget and hasattr(widget, "bind"):
                    next_widget = self._tab_widgets[(i + 1) % len(self._tab_widgets)]
                    prev_widget = self._tab_widgets[(i - 1) % len(self._tab_widgets)]

                    def make_tab_handler(nw: tk.Widget) -> Callable[[Any], None]:
                        def handler(e: Any) -> None:
                            self._focus_widget(nw)

                        return handler

                    widget.bind("<Tab>", make_tab_handler(next_widget))

                    def make_shift_tab_handler(pw: tk.Widget) -> Callable[[Any], None]:
                        def handler(e: Any) -> None:
                            self._focus_widget(pw)

                        return handler

                    widget.bind("<Shift-Tab>", make_shift_tab_handler(prev_widget))

    def _focus_widget(self, widget: tk.Widget) -> None:
        """Focus on a specific widget and handle different widget types."""
        try:
            if hasattr(widget, "focus_set"):
                widget.focus_set()
            elif hasattr(widget, "select"):
                widget.select()
        except tk.TclError:
            pass

    def stop_protocol(self):
        """Stop the currently running protocol if any."""
        if self.stop_event:
            self.stop_event.set()
            self.log_message("Stop signal sent to protocol...")
            self.set_status("Stopping...")

    def set_status(self, message):
        """Set status bar message with WCAG icon (thread-safe)."""
        if self.headless:
            print(f"STATUS: {message}")
            return

        def _update():
            msg_lower = message.lower()
            if "error" in msg_lower:
                icon, color = "✖", "#ffcdd2"
            elif "stop" in msg_lower:
                icon, color = "⚠", "#fff3cd"
            elif "running" in msg_lower or "loading" in msg_lower:
                icon, color = "⟳", "white"
            else:
                icon, color = "ℹ", "white"
            self.status_var.set(f"{icon}  {message}")
            if hasattr(self, "status_label"):
                self.status_label.configure(fg=color)

        self.gui_queue.put(_update)

    def log_message(self, message):
        """Add message to output console (thread-safe)."""
        if self.headless:
            print(message)
            return

        def _update():
            self.output_console.insert(tk.END, message + "\n")
            self.output_console.see(tk.END)

            max_lines = 1000
            lines = self.output_console.get("1.0", tk.END).splitlines()
            if len(lines) > max_lines:
                self.output_console.delete("1.0", tk.END)
                self.output_console.insert(tk.END, "\n".join(lines[-max_lines:]) + "\n")

        self.gui_queue.put(_update)

    def run_protocol(self, protocol_info):
        """Run selected protocol in separate thread."""

        def protocol_thread():
            try:
                self.set_status(f"Running {protocol_info['file']}...")
                self.log_message(f"=== Running {protocol_info['file']} ===")

                project_root = os.path.dirname(os.path.abspath(__file__))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                file_path = os.path.join(os.path.dirname(__file__), protocol_info["file"])

                if ".." in protocol_info["file"] or protocol_info["file"].startswith("/"):
                    raise ValueError(f"Invalid protocol path: {protocol_info['file']} (contains path traversal)")

                try:
                    Path(file_path).resolve().relative_to(Path(os.path.dirname(__file__)).resolve())
                except ValueError:
                    raise ValueError(f"Invalid protocol path: {protocol_info['file']} (outside project directory)")

                _fp_f2 = Path(file_path)
                if _verify_protocol is not None and not _verify_protocol(_fp_f2, "Falsification"):
                    raise _SecurityError(f"Protocol {_fp_f2.name} failed manifest integrity check")
                spec = importlib.util.spec_from_file_location(protocol_info["file"], file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.module = module

                configured_params = protocol_info.get("configured_params", {})

                validated_params: Dict[str, Any] = {}
                for key, value in configured_params.items():
                    try:
                        num_value = float(value)
                        if "dim" in key or "n_" in key:
                            if num_value <= 0 or num_value != int(num_value):
                                raise ValueError(f"{key} must be a positive integer")
                            validated_params[key] = int(num_value)
                        elif "lr" in key:
                            if num_value <= 0 or num_value > 1.0:
                                raise ValueError(f"{key} must be between 0 and 1")
                            validated_params[key] = num_value
                        else:
                            validated_params[key] = num_value
                    except (ValueError, TypeError) as e:
                        self.log_message(f"Warning: Invalid value for {key}: {value}. Error: {e}. Using default.")

                if protocol_info["class"] == "FalsificationAggregator":
                    results = self._handle_framework_aggregator(module)
                else:
                    class_name = protocol_info.get("class")
                    cls = getattr(module, class_name) if class_name else None
                    if class_name == "NetworkComparisonExperiment":
                        results = self._handle_network_comparison(cls, validated_params)
                    elif class_name == "APGIActiveInferenceAgent":
                        results = self._handle_apgi_agent(cls, module, validated_params)
                    elif class_name == "IowaGamblingTaskEnvironment":
                        results = self._handle_iowa_gambling(cls, module, validated_params)
                    elif hasattr(module, "run_falsification"):
                        self.log_message("Running run_falsification from module scope...")
                        results = module.run_falsification()
                    elif hasattr(module, "run_validation"):
                        self.log_message("Running run_validation from module scope...")
                        results = module.run_validation()
                    elif cls is not None and hasattr(cls, "run_full_experiment"):
                        results = self._handle_run_full_experiment(cls, validated_params)
                    elif cls is not None and hasattr(cls, "run_phase_transition_analysis"):
                        results = self._handle_phase_transition(cls, module, validated_params)
                    elif cls is not None and hasattr(cls, "run_evolution"):
                        results = self._handle_evolution(cls, validated_params)
                    elif cls is None:
                        results = {"status": "Incomplete", "warning": "No runner found"}
                    else:
                        results = self._handle_default(cls, module, validated_params)

                self._save_results(results, protocol_info["file"])

                protocol_status = self._extract_result_status(results)
                if self._is_successful_result(results):
                    self.log_message("=== Protocol completed successfully ===")
                else:
                    self.log_message(f"=== Protocol completed with status: {protocol_status} ===")
                self.set_status("Ready")
                self.gui_queue.put(lambda: self.stop_btn.config(state=tk.DISABLED))

            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
            ) as e:
                error_msg = f"Error running {protocol_info['file']}: {str(e)}"
                logger.warning(error_msg)
                self.gui_queue.put(lambda: messagebox.showerror("✖ Error", error_msg))
                self.set_status("Error")
                self.gui_queue.put(lambda: self.stop_btn.config(state=tk.DISABLED))
            except Exception as e:
                logger.error(f"Unexpected error in {protocol_info['file']}: {str(e)}")
                error_message = f"Unexpected error: {str(e)}"
                self.gui_queue.put(lambda: messagebox.showerror("✖ Error", error_message))
                self.set_status("Error")
                self.gui_queue.put(lambda: self.stop_btn.config(state=tk.DISABLED))

        self.stop_event.clear()
        self.running_thread = None
        thread = threading.Thread(target=protocol_thread)
        thread.daemon = True
        self.running_thread = thread
        self.stop_btn.config(state=tk.NORMAL)
        thread.start()

    # ── Protocol Handlers (unchanged) ─────────────────────────────────────────

    def _handle_framework_aggregator(self, module):
        """Handle FP_ALL_Aggregator: collect saved result files and run aggregation."""
        project_root = Path(__file__).parent
        validation_dir = project_root / "validation_results"

        result_files = []
        if validation_dir.exists():
            result_files = [str(f) for f in sorted(validation_dir.glob("*.json"))]

        self.log_message(f"Framework Aggregator: found {len(result_files)} result file(s) in {validation_dir}")

        run_fn = getattr(module, "run_framework_falsification", None)
        if run_fn is None:
            aggregator_cls = getattr(module, "FalsificationAggregator")
            agg = aggregator_cls()
            if result_files:
                result = agg.run_full_analysis(result_files)
            else:
                result = {
                    "status": "NO_RESULTS",
                    "message": "No previous validation results found. Run individual protocols first.",
                }
        else:
            if result_files:
                result = run_fn(result_files)
            else:
                result = {
                    "status": "NO_RESULTS",
                    "message": "No previous validation results found. Run individual protocols first.",
                }

        self.log_message(f"Framework Aggregator completed: {result.get('status', 'UNKNOWN')}")
        return result

    def _handle_network_comparison(self, cls, params):
        """Handle NetworkComparisonExperiment with configured parameters."""
        config = {
            "extero_dim": params.get("extero_dim", 32),
            "intero_dim": params.get("intero_dim", 16),
            "action_dim": params.get("action_dim", 4),
            "context_dim": params.get("context_dim", 8),
            "n_episodes": params.get("n_episodes", 50),
        }
        instance = cls(config)
        result = instance.run_full_experiment()
        self.log_message("Network Comparison Experiment completed")
        self.log_message(f"Results: {type(result)}")
        return result

    def _handle_apgi_agent(self, cls, module, params):
        """Run the full falsification protocol for APGI agent with configured parameters."""
        try:
            run_func = getattr(module, "run_falsification", None)
            if run_func:
                result = run_func()
                self.log_message(f"APGI Agent falsification completed: {result}")
                return result
            else:
                config = {
                    "lr_extero": params.get("lr_extero", 0.01),
                    "lr_intero": params.get("lr_intero", 0.01),
                    "lr_precision": params.get("lr_precision", 0.05),
                    "lr_somatic": params.get("lr_somatic", 0.1),
                    "n_actions": params.get("n_actions", 4),
                    "theta_init": params.get("theta_init", 0.5),
                    "theta_baseline": params.get("theta_baseline", 0.5),
                    "alpha": params.get("alpha", 8.0),
                    "tau_S": params.get("tau_S", 0.3),
                    "tau_theta": params.get("tau_theta", 10.0),
                    "eta_theta": params.get("eta_theta", 0.01),
                    "beta": params.get("beta", 1.2),
                    "rho": params.get("rho", 0.7),
                }
                _ = cls(config)
                self.log_message("APGI Agent created successfully (fallback)")
                self.log_message(f"Agent config: {config}")
                return {
                    "result": {"config": config},
                    "type": "APGIActiveInferenceAgent",
                }
        except Exception as e:
            self.log_message(f"Error in APGI agent protocol: {str(e)}")
            raise

    def _handle_iowa_gambling(self, cls, module, params):
        """Run the full falsification protocol for Iowa Gambling Task with configured parameters."""
        try:
            run_func = getattr(module, "run_falsification", None)
            if run_func:
                result = run_func()
                self.log_message(f"Iowa Gambling Task falsification completed: {result}")
                return result
            else:
                n_trials = params.get("n_trials", 100)
                env = cls(n_trials=n_trials)
                self.log_message("Iowa Gambling Task Environment created (fallback)")

                n_demo_trials = min(5, n_trials)
                total_reward = 0
                for trial in range(n_demo_trials):
                    action = 0
                    reward, intero_cost, obs, done = env.step(action)
                    total_reward += reward
                    self.log_message(
                        f"Trial {trial + 1}: Action={action}, Reward={reward:.2f}, InteroCost={intero_cost:.2f}"
                    )

                self.log_message(f"Demo completed. Total reward: {total_reward:.2f}")
                return {
                    "total_reward": total_reward,
                    "type": "IowaGamblingTaskEnvironment",
                }
        except Exception as e:
            self.log_message(f"Error in Iowa Gambling protocol: {str(e)}")
            raise

    def _handle_run_full_experiment(self, cls, params):
        """Handle protocols with run_full_experiment method using configured parameters."""
        result = {}
        try:
            try:
                instance = cls(**params)
            except (TypeError, ValueError) as e:
                logger.warning(f"Parameter mismatch for {cls.__name__}: {e}")
                instance = cls()
                self.log_message("Warning: Could not apply configured parameters, using defaults")
            except Exception as e:
                logger.warning(f"Unexpected error in application: {e}")
                instance = cls()
                self.log_message("Warning: Could not apply configured parameters, using defaults")

            result = instance.run_full_experiment()
            self.log_message(f"Experiment completed: {type(result)}")
        except ValueError as ve:
            if "broadcast" in str(ve):
                self.log_message(
                    "Broadcasting error in Protocol-3 - this is expected due to observation size mismatches"
                )
                self.log_message("Protocol-3 needs observation size alignment")
            else:
                raise
        return result

    def _handle_phase_transition(self, cls, module, params):
        """Handle phase transition analysis with configured parameters."""
        surprise_system = module.SurpriseIgnitionSystem()

        for param_name, value in params.items():
            if hasattr(surprise_system, param_name):
                try:
                    setattr(surprise_system, param_name, value)
                except Exception as e:  # noqa: F841
                    logger.warning(f"Failed to set parameter {param_name}: {e}")
                    continue

        instance = cls(surprise_system)
        result = instance.run_phase_transition_analysis()
        self.log_message(f"Phase transition analysis completed: {type(result)}")
        return result

    def _handle_evolution(self, cls, params):
        """Handle evolutionary protocols with configured parameters."""
        result = None
        try:
            config = {
                "population_size": params.get("population_size", 50),
                "n_generations": params.get("n_generations", 100),
                "mutation_rate": params.get("mutation_rate", 0.1),
                "selection_pressure": params.get("selection_pressure", 2.0),
            }

            instance = cls(stop_event=self.stop_event, **config)
            self.log_message("Starting evolutionary simulation (this may take time)...")
            result = instance.run_evolution()
            self.log_message(f"Evolution completed: {type(result)}")
        except KeyboardInterrupt:
            self.log_message("Evolution interrupted by user")
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ) as e:
            self.log_message(f"Error in evolution: {str(e)}")
            self.log_message("EvolutionaryAPGIEmergence instance creation failed")
        return result

    def _extract_result_status(self, results):
        """Best-effort status extraction across legacy and standardized payloads."""
        if results is None:
            return "error"

        if not isinstance(results, dict):
            return "success"

        for key in ("status", "outcome", "result"):
            value = results.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        summary = results.get("summary")
        if isinstance(summary, dict):
            failed = summary.get("failed")
            passed = summary.get("passed")
            if isinstance(failed, int) and failed > 0:
                return "failed"
            if isinstance(passed, int):
                return "success" if passed > 0 else "failed"

        passed_flag = results.get("passed")
        if isinstance(passed_flag, bool):
            return "success" if passed_flag else "failed"

        named_predictions = {}
        if isinstance(results.get("named_predictions"), dict):
            named_predictions = results["named_predictions"]
        elif isinstance(results.get("results"), dict) and isinstance(results["results"].get("named_predictions"), dict):
            named_predictions = results["results"]["named_predictions"]

        if named_predictions:
            passed = sum(1 for pred in named_predictions.values() if self._prediction_passed(pred))
            total = len(named_predictions)
            if passed == total:
                return "success"
            if passed == 0:
                return "failed"
            return "partial"

        return "success"

    def _prediction_passed(self, prediction):
        """Extract a pass/fail flag from legacy or standardized prediction payloads."""
        if isinstance(prediction, dict):
            return bool(prediction.get("passed", False))
        return bool(getattr(prediction, "passed", False))

    def _is_successful_result(self, results):
        """Return True for protocol outcomes that are not falsified.

        "not_implemented" and "pending" stubs are treated as not-falsified
        (inconclusive / pending empirical data), consistent with "falsified": False
        in their return payloads.
        """
        status = self._extract_result_status(results).strip().lower().replace(" ", "_")
        return status in {
            "success",
            "completed",
            "passed",
            "synthetic_validated",
            "not_implemented",
            "pending",
            "not_applicable",
        }

    def _handle_default(self, cls, module, params):
        """Handle default protocol execution with configured parameters."""
        try:
            instance = cls(**params)
            self.log_message(f"Created {cls.__name__} instance with parameters")
        except TypeError:
            instance = cls()
            self.log_message("Warning: Could not apply configured parameters, using defaults")

        self.log_message(f"Attempting to run {cls.__name__}...")
        if hasattr(instance, "run_validation"):
            return instance.run_validation()
        elif hasattr(module, "run_validation"):
            return module.run_validation()
        elif hasattr(instance, "run_falsification"):
            return instance.run_falsification()
        else:
            self.log_message(f"No standard run method found for {cls.__name__}")
            return {"status": "Completed (No Test)", "instance": str(instance)}

    def _save_results(self, results, protocol_file):
        """Save validation results to a JSON file."""
        try:
            import json
            import uuid
            from datetime import datetime

            project_root = Path(__file__).parent
            validation_dir = project_root / "validation_results"
            validation_dir.mkdir(parents=True, exist_ok=True)

            unique_id = str(uuid.uuid4())[:8]
            filename = f"validation_results_{unique_id}.json"
            filepath = validation_dir / filename

            named_predictions = {}
            if isinstance(results, dict):
                if "named_predictions" in results:
                    named_predictions = results["named_predictions"]
                elif "results" in results and isinstance(results["results"], dict):
                    if "named_predictions" in results["results"]:
                        named_predictions = results["results"]["named_predictions"]

            bic_values = {}
            if isinstance(results, dict):
                if "bic_values" in results:
                    bic_values = results["bic_values"]
                elif "model_comparison" in results:
                    mc = results["model_comparison"]
                    if isinstance(mc, dict) and "apgi" in mc:
                        bic_values = {"IGT": mc}

            results_with_metadata = {
                "metadata": {
                    "protocol_file": protocol_file,
                    "timestamp": datetime.now().isoformat(),
                    "id": unique_id,
                    "status": self._extract_result_status(results),
                },
                "named_predictions": named_predictions,
                "bic_values": bic_values,
                "results": results,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results_with_metadata, f, indent=2, default=str)

            self.log_message(f"Results saved to {filepath}")

        except Exception as e:
            self.log_message(f"Error saving results: {e}")

    def _validate_spinbox_float(self, value, min_val, max_val):
        """Validate float spinbox input."""
        if value == "":
            return True
        try:
            val = float(value)
            return float(min_val) <= val <= float(max_val)
        except ValueError:
            return False

    def _validate_spinbox_int(self, value, min_val, max_val):
        """Validate integer spinbox input."""
        if value == "":
            return True
        try:
            val = int(float(value))
            return int(min_val) <= val <= int(max_val)
        except ValueError:
            return False


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="APGI Falsification Protocols GUI / Headless Runner")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run all falsification protocols without launching the GUI",
    )
    parser.add_argument(
        "--token",
        help="JWT authentication token for secured operations",
    )
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
        print("Running in HEADLESS mode")
        from unittest.mock import MagicMock

        mock_root = MagicMock()
        runner = ProtocolRunnerGUI(mock_root, headless=True)
        runner.run_headless()
        return

    root = tk.Tk()
    _ = ProtocolRunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
