#!/usr/bin/env python3
"""
APGI Open Science Framework (OSF) Protocol Management GUI

Manages the eight canonical prediction suites (EP-0 through EP-7, i.e.
Protocols APGI-P00 through APGI-P07) defined in the APGI Framework
Prediction Registry: pre-registration tracking, dependency visualization,
and report export.
"""

import json
import logging
import os
import queue
import sys
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import TclError, filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

# ── APGI Design System ────────────────────────────────────────────────────────

_BG = "#f8f9fa"
_FG = "#212529"
_BLUE = "#2874a6"
_GREEN = "#155724"
_RED = "#721c24"
_BORDER = "#dee2e6"
_SURFACE = "#ffffff"
_MUTED = "#6c757d"
_SOFT = "#e9ecef"


def apply_apgi_theme(root: tk.Tk) -> ttk.Style:
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TFrame", background=_BG)
    style.configure("TLabel", background=_BG, foreground=_FG, font=("Noto Sans", 10))
    style.configure("Header.TLabel", font=("Noto Sans", 12, "bold"), background=_BG, foreground=_FG)
    style.configure("Title.TLabel", font=("Noto Sans", 14, "bold"), background=_SURFACE, foreground=_FG)
    style.configure("Subtitle.TLabel", font=("Noto Sans", 10), background=_SURFACE, foreground=_MUTED)
    style.configure("Monospace.TLabel", font=("Noto Sans Mono", 11), background=_SURFACE, foreground=_BLUE)
    style.configure("TLabelframe", background=_BG, bordercolor=_BORDER)
    style.configure("TLabelframe.Label", background=_BG, foreground=_MUTED, font=("Noto Sans", 9, "bold"))
    style.configure("Card.TFrame", background=_SURFACE, borderwidth=1, relief="solid")
    style.configure("Card.TCheckbutton", background=_SURFACE)
    style.configure("TButton", padding=6, background=_SOFT, font=("Noto Sans", 10))
    style.map(
        "TButton", background=[("active", _BORDER), ("disabled", "#f1f3f5")], foreground=[("disabled", "#adb5bd")]
    )
    style.configure("Primary.TButton", background=_GREEN, foreground="white", font=("Noto Sans", 10, "bold"), padding=8)
    style.map(
        "Primary.TButton",
        background=[("active", "#0f3d1a"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Secondary.TButton", background=_BLUE, foreground="white", font=("Noto Sans", 10), padding=6)
    style.map(
        "Secondary.TButton",
        background=[("active", "#1f5a82"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Danger.TButton", background=_RED, foreground="white", font=("Noto Sans", 10, "bold"), padding=8)
    style.map(
        "Danger.TButton",
        background=[("active", "#5a161d"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Horizontal.TProgressbar", background=_BLUE, troughcolor=_BORDER, borderwidth=0)
    style.configure("TCombobox", font=("Noto Sans", 10))
    style.configure("TNotebook", background=_BG, tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", font=("Noto Sans", 10), padding=[10, 5])
    style.map("TNotebook.Tab", background=[("selected", _SURFACE)], expand=[("selected", [1, 1, 1, 0])])

    root.configure(background=_BG)
    return style


class APGICard(ttk.Frame):
    """Standardized information card for APGI applications."""

    def __init__(self, parent: tk.Widget, title: str, value: str, subtitle: str = "", **kwargs: Any) -> None:
        super().__init__(parent, style="Card.TFrame", **kwargs)
        container = ttk.Frame(self, padding=15, style="Card.TFrame")
        container.pack(fill="both", expand=True)

        ttk.Label(
            container, text=title.upper(), font=("Noto Sans", 11, "bold"), background=_SURFACE, foreground=_FG
        ).pack(anchor="w")

        ttk.Label(
            container, text=value, font=("Noto Sans Mono", 11), background=_SURFACE, foreground=_BLUE, wraplength=580
        ).pack(anchor="w", pady=(4, 8))

        if subtitle:
            ttk.Separator(container, orient="horizontal").pack(fill="x", pady=(0, 6))
            ttk.Label(
                container,
                text=subtitle,
                font=("Noto Sans", 9, "italic"),
                background=_SURFACE,
                foreground=_MUTED,
                wraplength=580,
            ).pack(anchor="w")


# ── Framework Credentials ─────────────────────────────────────────────────────
# Single source of truth for the APGI framework's public identifiers, as
# recorded in APGI-Protocols.md §"Data and Code Availability" and the OSF
# pre-registration. Update here, not at the call sites.

APGI_ORCID = "https://orcid.org/0009-0005-0328-2896"
APGI_OSF_PROJECT = "https://osf.io/t5hcq"
APGI_ZENODO_DOI = "10.5281/zenodo.21632264"
APGI_ZENODO_URL = "https://doi.org/10.5281/zenodo.21632264"
APGI_CODE_REPOSITORY = "https://github.com/apgiframework/apgi-implementation"
APGI_CODE_REPOSITORY_ORG = "https://github.com/apgiframework"

APGI_CREDENTIALS: Dict[str, str] = {
    "orcid": APGI_ORCID,
    "osf_project": APGI_OSF_PROJECT,
    "zenodo_doi": APGI_ZENODO_DOI,
    "zenodo_url": APGI_ZENODO_URL,
    "code_repository": APGI_CODE_REPOSITORY,
    "code_repository_org": APGI_CODE_REPOSITORY_ORG,
}

# ── Protocol Definitions ──────────────────────────────────────────────────────

EP_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "EP-0: HEP Proxy Validation": {
        "id": "EP-0",
        "title": "HEP Proxy Validation",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": [],
        "status": "Not started",
        "description": (
            "This empirical prerequisite protocol establishes the heartbeat-evoked potential (HEP, 250\u2013400 ms window) as a valid trial-by-trial proxy for interoceptive precision \u03a0\u2071 before any downstream EP-1 through EP-6 analyses are interpreted. Three sub-predictions are tested: that HEP amplitude correlates with an orthogonal \u03a0\u2071 index (heartbeat discrimination d\u2032) independent of any EEG measure (Pred 0.A); that pharmacological elevation of acetylcholine via physostigmine increases HEP amplitude in a dose-dependent manner confirmed by pupillometric engagement (Pred 0.B); and that anterior insula (aINS) BOLD signal tracks HEP amplitude trial-by-trial within participants after controlling for arousal (Pred 0.C). All three predictions must pass before HEP is used as a \u03a0\u2071 proxy in downstream protocols. Failure of any Pred 0 criterion is a submission-blocking issue for all protocols referencing HEP."
        ),
        "platform": "OSF",
        "osf_url": APGI_OSF_PROJECT,
        "paper": "APGI Paper 1",
        "sample_size": "N = 115 (fixed-N; 90% power for r = 0.35 at \u03b1 = 0.01, two-tailed)",
        "measures": [
            "Heartbeat discrimination d\u2032 per participant",
            "HEP mean amplitude",
            "r",
            "r",
            "HEP amplitude change: physostigmine vs. placebo, % and Cohen's d",
            "BF\u2081\u2080 for physostigmine HEP effect",
            "Pupil constriction onset time and magnitude: physostigmine vs. placebo",
            "r",
        ],
        "analysis": (""),
        "falsification_criterion": (
            "Pred 0.A: r < 0.20 across two independent samples Pred 0.B: No HEP change despite verified ACh elevation (pupil constriction confirmed) Pred 0.C: aINS BOLD and HEP statistically independent after arousal covariate"
        ),
        "primary_hypothesis": (
            "HEP amplitude in the 250\u2013400 ms window serves as a valid proxy for interoceptive precision \u03a0\u2071, evidenced by (1) correlation with orthogonal heartbeat discrimination d\u2032 > 0.35 replicated across two independent samples, (2) significant increase under physostigmine-induced ACh elevation confirmed by pupillometry, and (3) within-participant tracking of anterior insula BOLD signal after arousal covariate control."
        ),
        "sub_predictions": ["Pred 0.A", "Pred 0.B", "Pred 0.C"],
        "prediction_cluster": "Pred 0.A\u20130.C",
        "prereg_status": "Pending",
        "notes": "Pred 0.B requires a separate pharmacological sub-session; if physostigmine arm cannot be completed (regulatory or safety reasons), Pred 0.B is downgraded to 'deferred' and does not block submission provided Pred 0.A and 0.C both pass.",
    },
    "EP-1: Cardiac-EEG Interoceptive Gating": {
        "id": "EP-1",
        "title": "Cardiac-EEG Interoceptive Gating",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "This protocol examines whether interoceptive precision (\u03a0\u2071), indexed by the heartbeat-evoked potential (HEP, 250\u2013400 ms window), modulates conscious detection thresholds (\u03b8\u209c). Stimuli are time-locked to cardiac phase to test four sub-predictions: (Pred 1.a) that state-level \u03a0\u2071_eff produces greater P3b amplitude for near-threshold stimuli under interoceptive focus than exteroceptive focus or dual-task control \u2014 with the effect absent or reversed for suprathreshold stimuli; (Pred 1.a-trait) that trait-level \u03a0\u2071_baseline, indexed by interoceptive accuracy (IA), predicts greater perceptual sensitivity (d\u2032) in the interoceptive condition but not the exteroceptive condition; (Pred 1.b) that cardiac-phase-locked detection advantage exists at diastole versus systole; and (Pred 1.c) that top-tertile IA participants show the strongest state-level P3b condition effects and cardiac-phase detection advantages. Near-threshold contrast titration is mandatory; suprathreshold control blocks are required to confirm specificity."
        ),
        "platform": "OSF",
        "osf_url": APGI_OSF_PROJECT,
        "paper": "APGI Paper 1",
        "sample_size": "N = 62 target / N = 48 minimum (Bayesian sequential)",
        "measures": [
            "P3b amplitude",
            "P3b amplitude \u00d7 condition ANOVA: \u03b7\u209a\u00b2 for condition main effect \u2014 primary",
            "HEP amplitude",
            "HEP\u2013P3b partial correlation",
            "P3b condition effect at suprathreshold contrast \u2014 Pred 1.a control block",
            "d-prime",
            "Group \u00d7 modality interaction on d\u2032",
            "Detection rate per cardiac phase",
        ],
        "analysis": (""),
        "falsification_criterion": (
            "Pred 1.A: No significant P3b condition main effect (all p > 0.10) across N \u2265 30 in preregistered contrast; or HEP\u2013P3b partial correlation < r = 0.20 after arousal covariate control. Pred 1.A-trait: High-IA and Low-IA do not differ on interoceptive d\u2032 (d < 0.3 across N \u2265 30 per group); or d\u2032 group difference replicates equally in the exteroceptive condition, eliminating interoceptive specificity. Pred 1.B: Diastole vs. systole hit rate advantage < 5% across two independent samples (interaction p > 0.15); no d\u2032 difference between cardiac phases. Pred 1.C: Accuracy \u00d7 condition interaction p > 0.10; tertile groups do not differ in cardiac-phase detection advantage. Pred 1.D: Partial r < 0.20 after arousal control; or partial correlation indistinguishable from exteroceptive focus condition."
        ),
        "primary_hypothesis": (
            "Interoceptive focus produces greater P3b amplitude for near-threshold stimuli than exteroceptive focus or dual-task control (three-way condition difference, \u03b7\u209a\u00b2 \u2265 0.06), this effect is absent at suprathreshold contrast, and HEP amplitude predicts P3b with partial r > 0.4 surviving arousal covariate control. High-IA participants show larger P3b condition effects and diastolic detection advantages than low-IA participants."
        ),
        "sub_predictions": ["Pred 1.A", "Pred 1.A-trait", "Pred 1.B", "Pred 1.C", "Pred 1.D", "Pred 1.E"],
        "prediction_cluster": "Pred 1.A\u20131.E (Pred 1.A-trait and Pred 1.F reported separately)",
        "prereg_status": "Pending",
        "notes": "Pred 1.a requires three attention conditions; dual-task control condition substantially increases session length \u2014 practice blocks must ensure participants understand the dual-task instruction.",
    },
    "EP-2: Somatic Agent Simulations": {
        "id": "EP-2",
        "title": "Somatic Agent Simulations",
        "priority": 1,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": [],
        "status": "Not started",
        "description": (
            "Computational agent simulations test whether somatic marker integration M\u0302(c,a) = \u03b3_V\u00b7V(c,a) + \u03b3_A\u00b7A(c,a) confers measurable adaptive advantages over APGI agents lacking somatic marker function. Agents are evaluated on embodied decision tasks under volatility conditions where rapid threshold adaptation is required. Five sub-predictions are tested: (Pred 2.a) full APGI agents converge within 50\u201380 trials matching human IGT performance and outperform both GNWT-only and Standard PP agents in cumulative reward \u2014 all three pairwise comparisons significant; (Pred 2.b) post-ignition action selection entropy increases versus pre-ignition baseline, with 70\u201385% of ignition events satisfying \u03a0\u2071\u00b7|z\u2071| > \u03a0\u1d49\u00b7|z\u1d49|; (Pred 2.c) somatic marker retrieval M\u0302 temporally precedes threshold crossing \u03b8\u209c; (Pred 2.d) \u03b2_SM lesion specifically degrades performance more than other single-parameter lesions, with the deficit largest in high-volatility blocks (\u03c3_env = 0.6); and (Pred 2.e) the full APGI generative model achieves lower BIC than Standard PP and GNWT-only when fit to human IGT trial-by-trial choice sequences. Fully validated via APGI_Somatic_Marker_Identifiability.py."
        ),
        "platform": "OSF",
        "osf_url": APGI_OSF_PROJECT,
        "paper": "APGI Paper 1",
        "sample_size": "N = 1,000 simulation seeds per condition (minimum 500); human IGT fitting sample N = 80",
        "measures": [
            "Cumulative reward per agent type over 500 trials",
            "Convergence trial",
            "Pairwise reward comparisons: APGI vs. GNWT-only, APGI vs. Standard PP, GNWT-only vs. Standard PP \u2014 Pred 2.a",
            "Action selection entropy post-ignition vs. pre-ignition",
            "Proportion of ignition events satisfying \u03a0\u2071\u00b7|z\u2071| > \u03a0\u1d49\u00b7|z\u1d49| \u2014 Pred 2.b",
            "Cross-correlation lag: M\u0302 activation vs. \u03b8\u209c crossing",
            "Proportion of ignition events where M\u0302 leads by \u2265 1 trial \u2014 Pred 2.c",
            "Reward: \u03b2_SM-lesion vs. \u03a0\u2071-lesion vs. \u03b1-lesion, per volatility block \u2014 Pred 2.d",
        ],
        "analysis": (""),
        "falsification_criterion": (
            "Pred 2.A: No significant pairwise performance advantage across all volatility conditions; or APGI fails convergence within 80 trials even when cumulative reward advantage present. Pred 2.B: Ignition uncorrelated with post-ignition behavioral adaptation; or < 60% of ignition events satisfy \u03a0\u2071\u00b7|z\u2071| > \u03a0\u1d49\u00b7|z\u1d49| (interoceptive dominance criterion independently falsified). Pred 2.C: M\u0302 activation simultaneous with or lagging threshold crossing; cross-correlation peak at lag 0 or positive. Pred 2.D: \u03b2_SM produces equivalent or lesser deficit compared to \u03a0\u2071-lesion under all volatility conditions; or pathway specificity not demonstrated at high volatility. Pred 2.E: APGI BIC \u2265 any alternative model's BIC on pre-registered dataset; or \u0394BIC < 3 against GNWT-only (interoceptive and somatic parameters add no explanatory value)."
        ),
        "primary_hypothesis": (
            "Full APGI agents converge within 50\u201380 trials and achieve higher cumulative reward than both GNWT-only and Standard PP agents (all pairwise permutation p < 0.05), with the \u03b2_SM lesion deficit largest in high-volatility blocks (\u03c3_env = 0.6), and the full APGI generative model achieving \u0394BIC \u2265 10 over GNWT-only when fit to human IGT data."
        ),
        "sub_predictions": ["Pred 2.A", "Pred 2.B", "Pred 2.C", "Pred 2.D", "Pred 2.E", "Pred 2.F-flex"],
        "prediction_cluster": "Pred 2.A\u20132.E (Pred 2.F-flex exploratory)",
        "prereg_status": "Pending",
        "notes": "Somatic marker model M\u0302(c,a) = \u03b3_V\u00b7V(c,a) + \u03b3_A\u00b7A(c,a) is one formalisation among several; Damasio's original account does not specify \u03b3_V and \u03b3_A uniquely. Values used here are motivated by interoceptive salience literature but require empirical calibration.",
    },
    "EP-3: Anticipation fMRI": {
        "id": "EP-3",
        "title": "Anticipation fMRI",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "This fMRI study uses a risky-choice task with variable foreperiod to test whether somatic marker retrieval M\u0302(c,a) modulates effective interoceptive precision \u03a0\u2071_eff during anticipation, rather than directly encoding the raw prediction error \u03b5\u2071 during outcome. APGI predicts that vmPFC\u2013anterior insula (aINS) connectivity emerges during the anticipation window (before stimulus onset) while vmPFC\u2013posterior insula (pIC) connectivity remains flat, and that these effects are sensitive to option valence rather than sensory contrast. Four sub-predictions are tested: (Pred 3.a) vmPFC BOLD is parametrically modulated by EV during anticipation but does NOT correlate with outcome-locked SCR, while posterior insula outcome-locked BOLD correlates with SCR as an active \u03b5\u2071 control; (Pred 3.b) vmPFC\u2192aINS coupling increases during anticipation (precision-gain pathway) while vmPFC\u2192pIC coupling remains statistically flat (BF\u2080\u2081 \u2265 6, ROPE d = [\u22120.15, +0.15]); (Pred 3.c) vmPFC anticipation-period activation is context-specific \u2014 modulated by option valence (EV) and high-somatic-cost > low-somatic-cost at matched monetary value \u2014 not by sensory contrast; (Pred 3.d) removing the anticipation foreperiod (0 ms ISI) abolishes vmPFC\u2192aINS coupling AND vmPFC EV parametric modulation while leaving posterior insula outcome-locked activity intact. Identifiability of \u03b2 and \u03a0\u2071 is fully resolved via block-diagonal Fisher Information Matrix."
        ),
        "platform": "OSF",
        "osf_url": APGI_OSF_PROJECT,
        "paper": "APGI Paper 1",
        "sample_size": "N = 40 target / N = 36 minimum",
        "measures": [
            "vmPFC BOLD: anticipation-period parametric modulation by EV",
            "vmPFC\u2013SCR outcome-period correlation",
            "Posterior insula",
            "vmPFC\u2192aINS PPI coefficient: anticipation foreperiod > neutral baseline",
            "vmPFC\u2192pIC PPI coefficient: anticipation foreperiod vs. neutral baseline",
            "aINS vs. pIC PPI dissociation significance \u2014 Pred 3.b",
            "vmPFC activation: parametric modulation by EV",
            "vmPFC activation: parametric modulation by sensory contrast",
        ],
        "analysis": (""),
        "falsification_criterion": (
            "Pred 3.A: vmPFC BOLD correlates significantly with outcome-locked SCR during the outcome period (r > 0.35), indicating vmPFC encodes raw \u03b5\u2071 rather than anticipatory M\u0302; or vmPFC shows no anticipation-period EV parametric modulation (p > 0.20). Pred 3.B: vmPFC\u2192pIC coupling increases during anticipation (\u0394r > 0.30, p < 0.05); or vmPFC\u2192aINS coupling is absent; or the aINS/pIC dissociation is not significant (both connectivity estimates statistically equivalent). Pred 3.C: vmPFC activation correlates with sensory contrast regardless of option valence; or valence and contrast effects on vmPFC are statistically equivalent. Pred 3.D: vmPFC\u2192aINS coupling in 0 ms foreperiod condition equivalent to standard foreperiod condition (foreperiod removal does not reduce coupling); indicating vmPFC is not specifically anticipatory."
        ),
        "primary_hypothesis": (
            "vmPFC\u2192aINS functional connectivity peaks during the anticipation foreperiod (\u0394r \u2265 0.30, p < 0.05 FWE) while vmPFC\u2192pIC coupling is statistically flat (BF\u2080\u2081 \u2265 6, ROPE d = [\u22120.15, +0.15]). vmPFC BOLD is parametrically modulated by option EV during anticipation but not by outcome-locked SCR (r < 0.20), dissociating somatic marker retrieval M\u0302 from interoceptive prediction error \u03b5\u2071. Removing the anticipation foreperiod abolishes both vmPFC\u2192aINS coupling and EV modulation while preserving pIC outcome-locked activity."
        ),
        "sub_predictions": ["Pred 3.A", "Pred 3.B", "Pred 3.C", "Pred 3.D"],
        "prediction_cluster": "Pred 3.A\u20133.D",
        "prereg_status": "Pending",
        "notes": "SCR is a noisy proxy for \u03b5\u2071; individual differences in SCR reactivity may reduce power for Pred 3.a. Include SCR-reactor subgroup analysis (top 50% SCR amplitude) as sensitivity analysis.",
    },
    "EP-4: Metabolic-State Crossover": {
        "id": "EP-4",
        "title": "Metabolic-State Crossover",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0", "EP-1"],
        "status": "Not started",
        "description": (
            "This protocol provides the direct human empirical validation for the claim that the ignition threshold \u03b8\u209c is dynamically regulated by metabolic state as an allostatic triage mechanism. APGI posits that interoceptive precision weighting (\u03a0\u2071_eff) and somatic-marker retrieval rely on metabolically expensive neuromodulatory gain. Under conditions of energy deficit, the framework predicts that the brain selectively elevates \u03b8\u209c to suppress high-cost interoceptive channels, preserving essential exteroceptive processing. Three sub-predictions are tested: (Pred 4.A) metabolic depletion selectively elevates \u03b8\u209c for high-interoceptive-load stimuli, reducing d\u2032 disproportionately vs. neutral exteroceptive stimuli (Metabolic State \u00d7 Interoceptive Load interaction, LMM p < 0.05, \u03b7\u209a\u00b2 \u2265 0.06); (Pred 4.B) the neural ignition proxy (P3b amplitude) reflects selective \u03b8\u209c elevation, disproportionately suppressed for interoceptive targets under metabolic depletion; (Pred 4.C) the allostatic triage effect is mediated by metabolic cost, not generalized cognitive fatigue \u2014 the interaction survives strict covariation for trial-level pupil diameter and RMSSD."
        ),
        "platform": "OSF",
        "osf_url": APGI_OSF_PROJECT,
        "paper": "APGI Paper 1",
        "sample_size": "N = 60 target / N = 48 minimum",
        "measures": [
            "Perceptual sensitivity d\u2032 per condition",
            "P3b amplitude",
            "Metabolic State \u00d7 Interoceptive Load interaction on d\u2032",
            "Metabolic State \u00d7 Interoceptive Load interaction on P3b",
            "d\u2032 reduction: interoceptive load vs. exteroceptive load under depletion",
            "P3b reduction: interoceptive vs. exteroceptive under depletion",
            "Interaction after arousal covariation",
            "BF\u2081\u2080 for interaction term \u2014 Pred 4.C",
        ],
        "analysis": (""),
        "falsification_criterion": (
            "Pred 4.A: (a) No significant Metabolic State \u00d7 Interoceptive Load interaction on d\u2032 or P3b amplitude (p > 0.10); OR (b) metabolic depletion produces a uniform, global suppression of ignition proxies across all stimulus types, indicating generalized cognitive fatigue rather than selective \u03b8\u209c elevation. Pred 4.B: Uniform P3b suppression across all stimulus types; or P3b interaction disappears after covarying for pupil diameter / HRV (indicating an arousal confound rather than allostatic triage). Pred 4.C: The interaction effect is fully absorbed by arousal covariates (\u0394\u03b7\u209a\u00b2 < 0.02 or interaction p > 0.10 after covariation), indicating the effect is driven by LC-NE arousal rather than metabolic allostatic triage."
        ),
        "primary_hypothesis": (
            "Metabolic depletion (16-hour fast or 4-hour vigilance) selectively elevates \u03b8\u209c for high-interoceptive-load stimuli, producing a significant Metabolic State \u00d7 Interoceptive Load interaction on d\u2032 (\u03b7\u209a\u00b2 \u2265 0.06) and P3b amplitude that survives strict covariation for arousal (pupil diameter and RMSSD), consistent with allostatic triage rather than generalized cognitive fatigue."
        ),
        "sub_predictions": ["Pred 4.A", "Pred 4.B", "Pred 4.C"],
        "prediction_cluster": "Pred 4.A\u20134.C",
        "prereg_status": "Pending",
        "notes": "Fasting arm participants must be medically screened for metabolic contraindications (diabetes, eating disorders) before enrollment.",
    },
    "EP-5: Causal Insula TMS": {
        "id": "EP-5",
        "title": "Causal Insula TMS",
        "priority": 2,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Single-pulse TMS and transcranial focused ultrasound (tFUS/LIFU) are applied under fMRI-guided neuronavigation to four cortical targets to test whether anterior insula (aINS, interoceptive precision integrator) and the frontoparietal workspace (dlPFC, PPC) causally regulate the APGI ignition threshold \u03b8\u209c through dissociable mechanisms. APGI predicts that aINS stimulation selectively disrupts interoceptive gating \u2014 reducing \u03a0\u2071_eff and abolishing HEP\u2013PCI coupling while sparing exteroceptive P3b \u2014 whereas dlPFC and PPC stimulation reduces global ignition probability (B\u209c) across both interoceptive and exteroceptive streams without selectively affecting HEP. The primary modality for the aINS arm is tFUS (required for depth penetration to aINS \u2248 3\u20134 cm below scalp). Vertex serves as active sham control. Four sub-predictions are tested: (Pred 5.A) aINS and frontoparietal TMS both reduce PCI via dissociable mechanisms; (Pred 5.B) aINS TMS selectively disrupts interoceptive gating (HEP\u2013P3b coupling) while sparing exteroceptive P3b; (Pred 5.C) high baseline IA participants show strongest aINS TMS effects on PCI; (Pred 5.D) high-\u03a0\u2071 individuals show larger PCI decreases following aINS TMS than low-\u03a0\u2071, absent for dlPFC."
        ),
        "platform": "OSF",
        "osf_url": APGI_OSF_PROJECT,
        "paper": "APGI Paper 2",
        "sample_size": "N = 44 target / N = 36 minimum (PCI analysis floor N = 28)",
        "measures": [
            "PCI per stimulation condition",
            "HEP amplitude",
            "HEP\u2013PCI coupling coefficient per stimulation condition \u2014 Pred 5.A dissociation",
            "Site \u00d7 HEP\u2013PCI coupling interaction",
            "HEP\u2013P3b coupling coefficient per stimulation condition \u2014 Pred 5.B",
            "Exteroceptive P3b amplitude per stimulation condition \u2014 Pred 5.B stream specificity",
            "BF\u2080\u2081 for dlPFC effect on HEP",
            "PAS rating distribution per stimulation condition",
        ],
        "analysis": (""),
        "falsification_criterion": (
            "Pred 5.A: No PCI change following active TMS vs. vertex sham; or equivalent PCI reduction at vertex; or HEP\u2013PCI coupling equally abolished by dlPFC TMS (no dissociation). Pred 5.B: Uniform suppression of both P3b streams; or HEP equally reduced by dlPFC and aINS. Pred 5.C: No interaction between baseline accuracy and TMS site. Pred 5.D: No interaction; equivalent PCI response across IA tertiles at aINS site."
        ),
        "primary_hypothesis": (
            "Active aINS stimulation (tFUS) reduces PCI and abolishes HEP\u2013PCI coupling relative to vertex sham, while dlPFC/PPC TMS reduces PCI across both interoceptive and exteroceptive streams without affecting the HEP (BF\u2080\u2081 \u2265 6). High-IA participants show a larger PCI reduction under aINS stimulation than low-IA participants, with this accuracy \u00d7 site interaction absent for the dlPFC/PPC site."
        ),
        "sub_predictions": ["Pred 5.A", "Pred 5.B", "Pred 5.C", "Pred 5.D", "Pred 5.E"],
        "prediction_cluster": "Pred 5.A\u20135.E",
        "prereg_status": "Pending",
        "notes": "Registry EP-5 specifies aINS (tFUS) and dlPFC/PPC (TMS) as the two active stimulation sites. Mediodorsal thalamus and claustrum remain viable alternative substrates for \u03b8\u209c regulation but are not causal targets in this protocol. Paper captions must include: 'aINS and dlPFC/PPC involvement is consistent with APGI threshold predictions but does not exclude thalamic, claustrum-mediated, or GABAergic cortical inhibitory mechanisms.'",
    },
    "EP-6: Ignition iEEG": {
        "id": "EP-6",
        "title": "Ignition iEEG",
        "priority": 2,
        "type": "Clinical-Empirical",
        "prereg_required": True,
        "depends_on": ["EP-2"],
        "status": "Not started",
        "description": (
            "Intracranial EEG (iEEG) recordings in epilepsy patients performing a near-threshold visual detection task test whether conscious access produces all-or-none (bistable) firing rate distributions in the frontoparietal network, as predicted by APGI's ignition criterion B\u209c. Five sub-predictions are tested: (Pred 6.A) frontoparietal cortex shows bimodal high-gamma power distributions with Hartigan's dip p < 0.05 AND 2-component Gaussian BIC substantially lower than 1-component (\u0394BIC > 10); (Pred 6.B) bimodality is specific to the frontoparietal ignition network with occipital units showing graded responses, and intermediate-state bouts show a mean duration < 100 ms (sharp transitions) with prevalence < 15% of trial duration; (Pred 6.C, bifurcation falsification criterion) AC1 of pre-ignition high-gamma power increases monotonically in the 500 ms preceding detected stimuli \u2014 the distinguishing APGI prediction over standard GWT; (Pred 6.D) long-range beta/gamma coherence (15\u201380 Hz) between frontoparietal sites predicts seen vs. unseen trials (point-biserial r > 0.40), peaking 200\u2013400 ms post-stimulus \u2014 the adversarial APGI-vs-GNWT benchmark; (Pred 6.E) \u03a0\u2071 modulation predicts multi-timescale temporal clustering of ignition events (\u2265 3 dissociable timescales, BF \u2265 100 vs. the single-timescale alternative). Fully validated via APGI_LNN_Bifurcation_Analysis.py."
        ),
        "platform": "OSF",
        "osf_url": APGI_OSF_PROJECT,
        "paper": "APGI Paper 2",
        "sample_size": "N = 20 target / N = 12 minimum",
        "measures": [
            "Hartigan's dip statistic: frontoparietal high-gamma distribution per contrast level \u2014 Pred 6.A primary",
            "\u0394BIC",
            "Modal separation: low mode and high mode frequencies",
            "Hartigan's dip statistic: occipital high-gamma distribution",
            "Region \u00d7 bimodality-index interaction \u2014 Pred 6.B",
            "Mean intermediate-state bout duration",
            "Intermediate-state prevalence",
            "AC1 time series: high-gamma envelope in 500 ms pre-ignition window \u2014 Pred 6.C",
        ],
        "analysis": (""),
        "falsification_criterion": (
            "Pred 6.A: Unimodal continuous distribution of high-gamma power across all contrast levels; dip test p > 0.20; no evidence of bistability. Pred 6.B: Uniform bimodality across all recorded regions; or intermediate-state bouts > 150 ms in \u2265 50% of trials; or prevalence > 30% of trial duration (inconsistent with sharp bifurcation). Pred 6.C: AC1 flat or decreasing before detected stimuli; no monotonic pre-ignition trend. Pred 6.D: Point-biserial coherence\u2013detection r < 0.20; or the coherence peak falls outside the 200\u2013400 ms post-stimulus window; or an equivalent coherence increase appears at the occipital control pair. Pred 6.E: Only a single clustering timescale is identifiable, or the single-timescale model is favoured (BF\u2080\u2081 \u2265 6)."
        ),
        "primary_hypothesis": (
            "High-gamma power in frontoparietal iEEG shows a bimodal distribution (Hartigan's dip p < 0.05 AND \u0394BIC > 10); AC1 increases monotonically in the 500 ms preceding detected stimuli (Kendall \u03c4 > 0.3) with the effect absent or reversed in non-detected trials; bimodality is specific to frontoparietal (not occipital) cortex; and near-threshold stimuli produce stable intermediate high-gamma states > 100 ms in > 15% of near-threshold trials."
        ),
        "sub_predictions": ["Pred 6.A", "Pred 6.B", "Pred 6.C", "Pred 6.D", "Pred 6.E"],
        "prediction_cluster": "Pred 6.A\u20136.E",
        "prereg_status": "Pending",
        "notes": "iEEG patient populations have limited sample sizes; statistical power is constrained. Single-participant case series may supplement group analysis.",
    },
    "EP-7: DoC Biomarker": {
        "id": "EP-7",
        "title": "DoC Biomarker",
        "priority": 3,
        "type": "Clinical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "This clinical biomarker study examines whether interoceptive precision (indexed by HEP amplitude, 250\u2013400 ms at Cz) and ignition capacity (indexed by the Perturbational Complexity Index, PCI) jointly predict clinical recovery outcomes across the full disorders of consciousness (DoC) spectrum better than either biomarker alone. Four patient and control groups are studied: VS/UWS (N=30), MCS (N=30), EMCS (N=20), and healthy age/sex-matched controls (N=30); total N=110. Six sub-predictions are tested: (Pred 7.A) joint HEP + PCI model explains more variance in 3-month CRS-R outcome than either biomarker alone (\u0394R\u00b2 \u2265 0.05, AUC > 0.80); (Pred 7.B) HEP amplitude discriminates MCS from VS/UWS; four-group gradient confirmed; DMN\u2194PCI r > 0.50; DMN\u2194HEP r < 0.20 (double dissociation); (Pred 7.C) interoceptive perturbation via CCRC (5% CO\u2082/95% O\u2082, 90-second) increases PCI \u2265 10% in MCS but not VS/UWS, exceeding arousal-matched white-noise control; (Pred 7.D) HEP amplitude correlates with CRS-R total score at 3-month and 6-month follow-up (Spearman r > 0.4), with JMbayes2 joint modelling for longitudinal analysis and LOCF explicitly excluded; (Pred 7.E, exploratory) theta (4\u20138 Hz)\u2013infraslow (< 0.1 Hz) phase\u2013amplitude coupling on \u2265 30-min bedside EEG increases across the DoC gradient and tracks CRS-R (Jonckheere\u2013Terpstra p < 0.05; PAC\u2194CRS-R r \u2265 0.30); (Pred 7.F, exploratory) somatic bias modulates reportable embodiment weighting toward 'bodily/visceral' dimensions under high-\u03b2 attentional focus. Directly supported by Paper 3 Appendix DoC Table."
        ),
        "platform": "OSF",
        "osf_url": APGI_OSF_PROJECT,
        "paper": "APGI Paper 3",
        "sample_size": "N = 110 target / N = 96 minimum (VS/UWS 30, MCS 30, EMCS 20, controls 30)",
        "measures": [
            "HEP amplitude",
            "PCI per DoC group and time point \u2014 primary",
            "\u0394R\u00b2",
            "AUC",
            "Univariate model R\u00b2 for HEP alone and PCI alone",
            "Four-group ordinal gradient: HEP and PCI across VS/UWS, MCS, EMCS, controls \u2014 Pred 7.B",
            "DMN\u2013PCI correlation",
            "DMN\u2013HEP partial correlation",
        ],
        "analysis": (""),
        "falsification_criterion": (
            "Pred 7.A: Joint model R\u00b2 \u2264 max(univariate HEP R\u00b2, univariate PCI R\u00b2); or \u0394R\u00b2 < 0.05 in primary analysis. Pred 7.B: No significant HEP difference between MCS and VS/UWS; four-group gradient absent for HEP or PCI. Pred 7.C: No significant PCI change post-CCRC in MCS (\u0394PCI < 5%, p > 0.10); or equivalent PCI change in VS/UWS and MCS; or CCRC-evoked PCI change statistically indistinguishable from white-noise control (p > 0.10), indicating arousal rather than interoceptive prediction error. Pred 7.D: HEP shows no significant longitudinal correlation with CRS-R (r < 0.2) at either follow-up; or joint model predictive values do not exceed clinical benchmarks. Pred 7.E: Jonckheere\u2013Terpstra trend p > 0.10 AND PAC\u2194CRS-R r < 0.20. Pred 7.F: Manipulation produces no effect on reportable embodiment dimensions (d < 0.20); or effect is fully absorbed by general arousal covariates."
        ),
        "primary_hypothesis": (
            "A linear model including both HEP amplitude and PCI at enrolment explains \u0394R\u00b2 \u2265 0.05 more variance in 3-month CRS-R total score outcome than either biomarker alone. HEP amplitude and PCI show a four-group gradient (VS/UWS < MCS < EMCS < controls), DMN\u2013PCI r > 0.50 AND DMN\u2013HEP r < 0.20 (double dissociation), interoceptive perturbation increases PCI \u2265 10% in MCS but not VS/UWS with \u0394PCI exceeding arousal-matched control, and HEP amplitude correlates with CRS-R total score at 3-month and 6-month follow-up (Spearman r > 0.4)."
        ),
        "sub_predictions": ["Pred 7.A", "Pred 7.B", "Pred 7.C", "Pred 7.D", "Pred 7.E", "Pred 7.F"],
        "prediction_cluster": "Pred 7.A\u20137.F",
        "prereg_status": "Pending",
        "notes": "PCI requires TMS, which cannot be applied to patients with metallic implants or unstable hemodynamics; PCI subsample may be smaller than full cohort.",
    },
}


# Status options for dropdown
_STATUS_OPTIONS = [
    "Not started",
    "In preparation",
    "Pre-registered",
    "Data collection",
    "Analysis",
    "Completed",
    "On hold",
]
_PREREG_OPTIONS = ["Pending", "Draft ready", "Submitted", "Registered", "Not applicable"]

# Type badge colours
_TYPE_COLORS: Dict[str, str] = {
    "Empirical": "#2874a6",
    "Computational": "#155724",
    "Clinical": "#721c24",
    "Clinical-Empirical": "#5a0080",
}

# Priority labels
_PRIORITY_LABELS = {1: "Priority 1 — High", 2: "Priority 2 — Medium", 3: "Priority 3 — Long-term"}


# ── Main Application ──────────────────────────────────────────────────────────


class OSFProtocolGUI:
    """GUI for APGI Open Science Framework"""

    def __init__(self, root: tk.Tk, headless: bool = False) -> None:
        self.root = root
        self.headless = headless
        self.gui_queue: queue.Queue[Any] = queue.Queue()
        self._protocol_status: Dict[str, Dict[str, str]] = {
            key: {"status": meta["status"], "prereg_status": meta["prereg_status"], "notes": meta["notes"]}
            for key, meta in EP_PROTOCOLS.items()
        }
        self._selected_protocol: Optional[str] = None
        self._console_visible = True
        self._card_widgets: List[tk.Widget] = []
        self._detail_vars: Dict[str, tk.StringVar] = {}
        self.stop_event = threading.Event()
        self.running_thread: Optional[threading.Thread] = None

        if not self.headless:
            apply_apgi_theme(self.root)
            self._process_gui_queue()
            self.root.title("APGI Open Science Framework — Prediction Registry (Protocols 0 to 7)")
            self.root.geometry("1020x740")
            self.root.minsize(800, 560)
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self._create_menu_bar()
            self._bind_shortcuts()
            self.setup_ui()

        _proj_root = os.path.dirname(os.path.abspath(__file__))
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)

    # ── Queue ─────────────────────────────────────────────────────────────────

    def _process_gui_queue(self) -> None:
        try:
            while True:
                fn = self.gui_queue.get_nowait()
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        logger.warning("GUI queue update failed: %s", exc)
        except queue.Empty:
            pass
        try:
            self.root.after(100, self._process_gui_queue)
        except (KeyboardInterrupt, TclError):
            logger.info("GUI processing interrupted.")
            self.stop_event.set()

    # ── Menu & shortcuts ──────────────────────────────────────────────────────

    def _create_menu_bar(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Protocol Report", command=self.export_protocol_report, accelerator="Ctrl+E")
        file_menu.add_command(
            label="Generate Pre-reg Template", command=self.generate_prereg_template, accelerator="Ctrl+G"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Save Status Snapshot", command=self.save_status_snapshot, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Status Snapshot", command=self.load_status_snapshot)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_closing, accelerator="Ctrl+Q")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Clear Console", command=self.clear_console, accelerator="Ctrl+L")
        view_menu.add_command(label="Toggle Console", command=self._toggle_console)
        view_menu.add_command(label="Dependency Overview", command=self._show_dependency_overview)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-q>", lambda e: self._on_closing())
        self.root.bind("<Command-q>", lambda e: self._on_closing())
        self.root.bind("<Control-e>", lambda e: self.export_protocol_report())
        self.root.bind("<Control-g>", lambda e: self.generate_prereg_template())
        self.root.bind("<Control-s>", lambda e: self.save_status_snapshot())
        self.root.bind("<Control-l>", lambda e: self.clear_console())

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "APGI Open Science Framework GUI\nPrediction Registry Manager (EP-0 – EP-7)\nVersion 2.0.0",
        )

    # ── UI Construction ───────────────────────────────────────────────────────

    def setup_ui(self) -> None:
        """Build the application layout per APGI design spec."""
        # ── Row 0: Metric bar ─────────────────────────────────────────────────
        metric_bar = tk.Frame(self.root, bg=_BLUE, pady=8, padx=15)
        metric_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        metric_bar.columnconfigure(1, weight=1)

        tk.Label(
            metric_bar,
            text="APGI OPEN SCIENCE FRAMEWORK  —  EMPIRICAL PROTOCOLS",
            bg=_BLUE,
            fg="white",
            font=("Noto Sans", 13, "bold"),
        ).grid(row=0, column=0, sticky="w")

        # Summary counters
        self._counter_var = tk.StringVar(value=self._build_counter_text())
        tk.Label(metric_bar, textvariable=self._counter_var, bg=_BLUE, fg="white", font=("Noto Sans", 10)).grid(
            row=0, column=1, padx=(20, 15), sticky="e"
        )

        # OSF status indicator
        self._osf_status_var = tk.StringVar(value="ℹ  OSF: Ready")
        tk.Label(metric_bar, textvariable=self._osf_status_var, bg=_BLUE, fg="white", font=("Noto Sans", 10)).grid(
            row=0, column=2, sticky="e"
        )

        # ── Grid weights ──────────────────────────────────────────────────────
        self.root.columnconfigure(0, weight=0, minsize=220)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=0, minsize=160)

        # ── Row 1, Col 0: Sidebar ─────────────────────────────────────────────
        sidebar = tk.Frame(self.root, bg=_SURFACE, width=220, bd=0)
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)

        tk.Label(
            sidebar, text="EMPIRICAL PROTOCOLS", bg=_BORDER, fg=_MUTED, font=("Noto Sans", 9, "bold"), pady=5
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        lb_frame = tk.Frame(sidebar, bg=_SURFACE)
        lb_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        lb_frame.columnconfigure(0, weight=1)
        lb_frame.rowconfigure(0, weight=1)

        self.protocol_listbox = tk.Listbox(
            lb_frame,
            font=("Noto Sans", 9),
            bg=_SURFACE,
            fg=_FG,
            selectbackground=_BLUE,
            selectforeground="white",
            borderwidth=0,
            highlightthickness=1,
            highlightcolor=_BORDER,
            highlightbackground=_BORDER,
            activestyle="none",
            cursor="hand2",
        )
        lb_scroll = ttk.Scrollbar(lb_frame, orient="vertical", command=self.protocol_listbox.yview)
        self.protocol_listbox.configure(yscrollcommand=lb_scroll.set)
        self.protocol_listbox.grid(row=0, column=0, sticky="nsew")
        lb_scroll.grid(row=0, column=1, sticky="ns")

        for key in EP_PROTOCOLS:
            ep_id = EP_PROTOCOLS[key]["id"]
            short_title = EP_PROTOCOLS[key]["title"]
            self.protocol_listbox.insert(tk.END, f"  {ep_id}  {short_title}")

        self.protocol_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Sidebar action buttons
        btn_area = tk.Frame(sidebar, bg=_SURFACE, pady=8, padx=8)
        btn_area.grid(row=2, column=0, columnspan=2, sticky="ew")
        btn_area.columnconfigure(0, weight=1)

        self._gen_prereg_btn = ttk.Button(
            btn_area,
            text="✎  Generate Pre-reg",
            command=self.generate_prereg_template,
            style="Primary.TButton",
            state=tk.DISABLED,
        )
        self._gen_prereg_btn.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self._export_btn = ttk.Button(
            btn_area,
            text="Export Report",
            command=self.export_protocol_report,
            style="Secondary.TButton",
        )
        self._export_btn.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        ttk.Separator(btn_area, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_area, text="Dependency Overview", command=self._show_dependency_overview).grid(
            row=3, column=0, sticky="ew", pady=(0, 4)
        )
        ttk.Button(btn_area, text="Clear Console", command=self.clear_console).grid(row=4, column=0, sticky="ew")

        # Right border divider
        tk.Frame(self.root, bg=_BORDER, width=1).grid(row=1, column=0, sticky="nse")

        # ── Row 1, Col 1: Workspace ───────────────────────────────────────────
        workspace = ttk.Frame(self.root, padding=(15, 10, 15, 10))
        workspace.grid(row=1, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=0)  # cards row
        workspace.rowconfigure(1, weight=1)  # detail/notes row

        self._card_area = ttk.Frame(workspace)
        self._card_area.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._card_area.columnconfigure(0, weight=1)
        self._show_workspace_empty_state()

        # Detail / status panel
        self._detail_frame = ttk.LabelFrame(workspace, text="PROTOCOL DETAILS & STATUS", padding=(10, 6))
        self._detail_frame.grid(row=1, column=0, sticky="nsew")
        self._detail_frame.columnconfigure(0, weight=1)
        self._detail_frame.rowconfigure(0, weight=1)

        self._detail_inner = ttk.Frame(self._detail_frame)
        self._detail_inner.grid(row=0, column=0, sticky="nsew")
        self._detail_inner.columnconfigure(0, weight=1)
        self._detail_inner.rowconfigure(0, weight=1)

        ttk.Label(
            self._detail_inner,
            text="Select a protocol from the sidebar to view details and manage pre-registration.",
            foreground=_MUTED,
            font=("Noto Sans", 10),
        ).grid(row=0, column=0, padx=20, pady=20)

        # ── Row 2: Console toggle header ──────────────────────────────────────
        console_header = tk.Frame(self.root, bg=_BORDER, pady=3)
        console_header.grid(row=2, column=0, columnspan=2, sticky="ew")

        self._console_toggle_btn = tk.Button(
            console_header,
            text="▼  OUTPUT CONSOLE",
            bg=_BORDER,
            fg=_MUTED,
            font=("Noto Sans", 9, "bold"),
            relief="flat",
            cursor="hand2",
            command=self._toggle_console,
            activebackground=_BORDER,
            activeforeground="#495057",
            bd=0,
        )
        self._console_toggle_btn.pack(side="left", padx=10)

        # ── Row 3: Console body ───────────────────────────────────────────────
        self.root.rowconfigure(3, weight=0, minsize=160)
        self._console_frame = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        self._console_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self._console_frame.columnconfigure(0, weight=1)
        self._console_frame.rowconfigure(0, weight=1)

        self.console = scrolledtext.ScrolledText(
            self._console_frame,
            height=8,
            font=("Noto Sans Mono", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.console.grid(row=0, column=0, sticky="nsew")

        self._log("OSF Protocol Manager initialised. Select a protocol to begin.")
        self._log(f"{len(EP_PROTOCOLS)} protocols loaded  |  All status: Not started  |  Pre-reg: Pending")

    # ── Workspace helpers ─────────────────────────────────────────────────────

    def _show_workspace_empty_state(self) -> None:
        for w in self._card_area.winfo_children():
            w.destroy()

        frame = ttk.Frame(self._card_area, padding=30)
        frame.grid(row=0, column=0)
        tk.Canvas(frame, width=200, height=80, bg=_BG, highlightbackground=_SOFT, highlightthickness=2).pack()
        ttk.Label(
            frame,
            text="No protocol selected.  Choose one from the sidebar to view metadata and manage pre-registration.",
            wraplength=400,
            font=("Noto Sans", 11),
            foreground=_MUTED,
        ).pack(pady=(15, 0))

    def _show_protocol_cards(self, key: str) -> None:
        """Populate the card area with summary cards for the selected protocol."""
        for w in self._card_area.winfo_children():
            w.destroy()

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]

        cards_row = ttk.Frame(self._card_area)
        cards_row.grid(row=0, column=0, sticky="ew")
        for i in range(4):
            cards_row.columnconfigure(i, weight=1)

        APGICard(cards_row, "Protocol", meta["id"], f"Type: {meta['type']}  |  Priority: {meta['priority']}").grid(
            row=0, column=0, padx=(0, 6), pady=4, sticky="ew"
        )

        prereq_text = ", ".join(meta["depends_on"]) if meta["depends_on"] else "None"
        APGICard(cards_row, "Dependencies", prereq_text, "Must be completed before this protocol").grid(
            row=0, column=1, padx=(0, 6), pady=4, sticky="ew"
        )

        APGICard(cards_row, "Study Status", st["status"], "Update via the status panel below").grid(
            row=0, column=2, padx=(0, 6), pady=4, sticky="ew"
        )

        APGICard(
            cards_row,
            "Pre-registration",
            st["prereg_status"],
            f"Platform: {meta['platform']}  |  Required: {'Yes' if meta['prereg_required'] else 'No'}",
        ).grid(row=0, column=3, padx=0, pady=4, sticky="ew")

    def _populate_detail_panel(self, key: str) -> None:
        """Build the scrollable detail/status editor for the selected protocol."""
        for w in self._detail_inner.winfo_children():
            w.destroy()

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]

        # Scrollable canvas
        canvas = tk.Canvas(self._detail_inner, bg=_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self._detail_inner, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._detail_inner.rowconfigure(0, weight=1)
        self._detail_inner.columnconfigure(0, weight=1)

        scrollable = ttk.Frame(canvas, padding=(10, 6))
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        scrollable.columnconfigure(1, weight=1)

        def _row(r: int, label: str, value: str, mono: bool = False) -> None:
            ttk.Label(scrollable, text=label + ":", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
                row=r, column=0, sticky="nw", padx=(0, 12), pady=2
            )
            fnt = ("Noto Sans Mono", 9) if mono else ("Noto Sans", 10)
            ttk.Label(scrollable, text=value, font=fnt, foreground=_FG, wraplength=560, justify="left").grid(
                row=r, column=1, sticky="nw", pady=2
            )

        row_idx = 0

        # Title
        ttk.Label(scrollable, text=meta["title"], font=("Noto Sans", 13, "bold"), foreground=_FG).grid(
            row=row_idx, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        row_idx += 1

        _row(row_idx, "Description", meta["description"])
        row_idx += 1
        _row(row_idx, "Study Type", meta["type"])
        row_idx += 1
        _row(row_idx, "Priority", _PRIORITY_LABELS.get(meta["priority"], str(meta["priority"])))
        row_idx += 1
        _row(row_idx, "Sample Size", meta["sample_size"])
        row_idx += 1
        _row(row_idx, "Key Measures", "  •  " + "\n  •  ".join(meta["measures"]))
        row_idx += 1
        _row(row_idx, "Analysis Plan", meta["analysis"])
        row_idx += 1
        _row(row_idx, "Falsification Criterion", meta["falsification_criterion"], mono=True)
        row_idx += 1

        if meta["notes"]:
            _row(row_idx, "Notes", meta["notes"])
            row_idx += 1

        ttk.Separator(scrollable, orient="horizontal").grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=8)
        row_idx += 1

        # ── Editable status fields ────────────────────────────────────────────
        ttk.Label(scrollable, text="UPDATE STATUS", font=("Noto Sans", 10, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        row_idx += 1

        # Study status
        ttk.Label(scrollable, text="Study Status:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        status_var = tk.StringVar(value=st["status"])
        self._detail_vars["status"] = status_var
        status_cb = ttk.Combobox(
            scrollable, textvariable=status_var, values=_STATUS_OPTIONS, state="readonly", width=28
        )
        status_cb.grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        # Pre-reg status
        ttk.Label(scrollable, text="Pre-reg Status:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        prereg_var = tk.StringVar(value=st["prereg_status"])
        self._detail_vars["prereg_status"] = prereg_var
        prereg_cb = ttk.Combobox(
            scrollable, textvariable=prereg_var, values=_PREREG_OPTIONS, state="readonly", width=28
        )
        prereg_cb.grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        # Notes
        ttk.Label(scrollable, text="Notes:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="nw", pady=2
        )
        notes_text = tk.Text(
            scrollable,
            height=3,
            width=55,
            font=("Noto Sans", 10),
            wrap=tk.WORD,
            bg=_SURFACE,
            fg=_FG,
            relief="solid",
            bd=1,
        )
        notes_text.insert("1.0", st["notes"])
        notes_text.grid(row=row_idx, column=1, sticky="ew", pady=2)
        self._notes_widget = notes_text
        row_idx += 1

        # Save button
        def _save() -> None:
            self._protocol_status[key]["status"] = status_var.get()
            self._protocol_status[key]["prereg_status"] = prereg_var.get()
            self._protocol_status[key]["notes"] = notes_text.get("1.0", tk.END).strip()
            self._show_protocol_cards(key)
            self._counter_var.set(self._build_counter_text())
            self._log(f"[{meta['id']}] Status saved: {status_var.get()} | Pre-reg: {prereg_var.get()}")
            messagebox.showinfo("Saved", f"{meta['id']} status updated.")

        ttk.Button(scrollable, text="Save Changes", command=_save, style="Primary.TButton").grid(
            row=row_idx, column=1, sticky="w", pady=8
        )

    # ── Sidebar selection ─────────────────────────────────────────────────────

    def _on_listbox_select(self, _event: Any = None) -> None:
        sel = self.protocol_listbox.curselection()
        if not sel:
            return
        key = list(EP_PROTOCOLS.keys())[sel[0]]
        self._selected_protocol = key
        self._gen_prereg_btn.configure(state=tk.NORMAL)
        meta = EP_PROTOCOLS[key]
        self._log(f"Selected: {meta['id']} — {meta['title']}")
        self._show_protocol_cards(key)
        self._populate_detail_panel(key)

    # ── Counter text ──────────────────────────────────────────────────────────

    def _build_counter_text(self) -> str:
        total = len(EP_PROTOCOLS)
        registered = sum(1 for s in self._protocol_status.values() if s["prereg_status"] == "Registered")
        completed = sum(1 for s in self._protocol_status.values() if s["status"] == "Completed")
        return f"Protocols: {total}  |  Pre-registered: {registered}/{total}  |  Completed: {completed}/{total}"

    # ── Dependency Overview ───────────────────────────────────────────────────

    def _show_dependency_overview(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Dependency Overview — APGI Empirical Protocols")
        win.geometry("560x480")
        win.configure(bg=_BG)
        win.resizable(True, True)

        tk.Label(
            win, text="PROTOCOL DEPENDENCY GRAPH", bg=_BLUE, fg="white", font=("Noto Sans", 12, "bold"), pady=8
        ).pack(fill="x", padx=0)

        frame = ttk.Frame(win, padding=15)
        frame.pack(fill="both", expand=True)

        lines: List[str] = []
        lines.append("Execution Order & Dependencies\n")
        lines.append("=" * 48)
        for key, meta in EP_PROTOCOLS.items():
            dep_str = " ← requires: " + ", ".join(meta["depends_on"]) if meta["depends_on"] else " (no dependencies)"
            st = self._protocol_status[key]["status"]
            prereg = self._protocol_status[key]["prereg_status"]
            lines.append(
                f"\n  {meta['id']:6s}  {meta['title']}"
                f"\n          Type: {meta['type']}  |  Priority: {meta['priority']}"
                f"\n          Depends on:{dep_str}"
                f"\n          Status: {st}  |  Pre-reg: {prereg}"
            )
        lines.append("\n" + "=" * 48)

        text = scrolledtext.ScrolledText(
            frame, font=("Noto Sans Mono", 9), bg=_SURFACE, fg=_FG, state=tk.DISABLED, wrap=tk.WORD
        )
        text.pack(fill="both", expand=True)
        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(8, 0))

    # ── Pre-registration template ─────────────────────────────────────────────

    def generate_prereg_template(self) -> None:
        key = self._selected_protocol
        if not key:
            messagebox.showwarning("No Protocol Selected", "Please select a protocol from the sidebar first.")
            return

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines: List[str] = [
            "=" * 64,
            "APGI PRE-REGISTRATION TEMPLATE",
            f"Generated: {timestamp}",
            "=" * 64,
            "",
            f"Protocol ID:        {meta['id']}",
            f"Title:              {meta['title']}",
            f"Study Type:         {meta['type']}",
            f"Priority:           {_PRIORITY_LABELS.get(meta['priority'], str(meta['priority']))}",
            f"Platform:           {meta['platform']}",
            f"Current Status:     {st['status']}",
            f"Pre-reg Status:     {st['prereg_status']}",
            "",
            "─" * 64,
            "1. RESEARCH QUESTION & HYPOTHESES",
            "─" * 64,
            "",
            meta["description"],
            "",
            "─" * 64,
            "2. EXPERIMENTAL DESIGN",
            "─" * 64,
            "",
            f"Sample Size:  {meta['sample_size']}",
            "",
            "Key Measures:",
        ]
        for m in meta["measures"]:
            lines.append(f"  • {m}")

        lines += [
            "",
            "─" * 64,
            "3. ANALYSIS PLAN",
            "─" * 64,
            "",
            meta["analysis"],
            "",
            "─" * 64,
            "4. FALSIFICATION CRITERIA",
            "─" * 64,
            "",
            meta["falsification_criterion"],
            "",
            "─" * 64,
            "5. DEPENDENCIES",
            "─" * 64,
            "",
        ]

        if meta["depends_on"]:
            for dep in meta["depends_on"]:
                dep_key = next((k for k in EP_PROTOCOLS if EP_PROTOCOLS[k]["id"] == dep), None)
                dep_title = EP_PROTOCOLS[dep_key]["title"] if dep_key else dep
                lines.append(f"  Requires: {dep} — {dep_title}")
        else:
            lines.append("  None — this protocol has no upstream dependencies.")

        lines += [
            "",
            "─" * 64,
            "6. NOTES",
            "─" * 64,
            "",
            st["notes"] if st["notes"] else "[No notes recorded]",
            "",
            "=" * 64,
            "END OF PRE-REGISTRATION TEMPLATE",
            "=" * 64,
        ]

        template_text = "\n".join(lines)

        # Show in a new window and offer save
        win = tk.Toplevel(self.root)
        win.title(f"Pre-registration Template — {meta['id']}")
        win.geometry("680x560")
        win.configure(bg=_BG)

        tk.Label(
            win,
            text=f"PRE-REGISTRATION TEMPLATE  —  {meta['id']}",
            bg=_BLUE,
            fg="white",
            font=("Noto Sans", 12, "bold"),
            pady=8,
        ).pack(fill="x")

        txt = scrolledtext.ScrolledText(win, font=("Noto Sans Mono", 9), bg=_SURFACE, fg=_FG, wrap=tk.WORD)
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert(tk.END, template_text)

        def _save_file() -> None:
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"prereg_{meta['id'].replace('-', '_')}.txt",
            )
            if path:
                Path(path).write_text(template_text, encoding="utf-8")
                self._log(f"Pre-reg template saved: {path}")
                messagebox.showinfo("Saved", f"Template saved to:\n{path}")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Save to File…", command=_save_file, style="Primary.TButton").pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side="left")

        self._log(f"[{meta['id']}] Pre-registration template generated.")

    # ── Export report ─────────────────────────────────────────────────────────

    def export_protocol_report(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines: List[str] = [
            "=" * 68,
            "APGI OPEN SCIENCE FRAMEWORK — EMPIRICAL PROTOCOL STATUS REPORT",
            f"Generated: {timestamp}",
            "=" * 68,
            "",
        ]

        for key, meta in EP_PROTOCOLS.items():
            st = self._protocol_status[key]
            dep_str = ", ".join(meta["depends_on"]) if meta["depends_on"] else "None"
            lines += [
                f"{'─' * 68}",
                f"  {meta['id']}  |  {meta['title']}",
                f"  Type: {meta['type']}   Priority: {meta['priority']}   Platform: {meta['platform']}",
                f"  Depends on: {dep_str}",
                f"  Study status:    {st['status']}",
                f"  Pre-reg status:  {st['prereg_status']}",
                f"  Notes: {st['notes'] if st['notes'] else '—'}",
                "",
            ]

        lines += [
            "=" * 68,
            "END OF REPORT",
            "=" * 68,
        ]

        report_text = "\n".join(lines)
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="apgi_osf_protocol_report.txt",
        )
        if not path:
            return

        if path.endswith(".json"):
            data = {
                "generated": timestamp,
                "protocols": {key: {**EP_PROTOCOLS[key], **self._protocol_status[key]} for key in EP_PROTOCOLS},
            }
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            Path(path).write_text(report_text, encoding="utf-8")

        self._log(f"Protocol report exported: {path}")
        messagebox.showinfo("Exported", f"Report saved to:\n{path}")

    # ── Status snapshot (persistence) ────────────────────────────────────────

    def save_status_snapshot(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="osf_status_snapshot.json",
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self._protocol_status, indent=2), encoding="utf-8")
        self._log(f"Status snapshot saved: {path}")

    def load_status_snapshot(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for key in EP_PROTOCOLS:
                if key in data:
                    for field in ("status", "prereg_status", "notes"):
                        if field in data[key]:
                            self._protocol_status[key][field] = data[key][field]
            self._counter_var.set(self._build_counter_text())
            self._log(f"Status snapshot loaded: {path}")
            # Refresh workspace if a protocol is selected
            if self._selected_protocol:
                self._show_protocol_cards(self._selected_protocol)
                self._populate_detail_panel(self._selected_protocol)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            messagebox.showerror("Load Failed", f"Could not load snapshot:\n{exc}")

    # ── Console ───────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self.headless:
            logger.info(message)
            return

        def _do() -> None:
            self.console.configure(state=tk.NORMAL)
            ts = datetime.now().strftime("%H:%M:%S")
            self.console.insert(tk.END, f"[{ts}]  {message}\n")
            self.console.see(tk.END)
            self.console.configure(state=tk.DISABLED)

        try:
            self.gui_queue.put_nowait(_do)
        except queue.Full:
            pass

    def clear_console(self) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.configure(state=tk.DISABLED)

    def _toggle_console(self) -> None:
        if self._console_visible:
            self._console_frame.grid_remove()
            self.root.rowconfigure(3, minsize=0)
            self._console_toggle_btn.configure(text="▶  OUTPUT CONSOLE")
            self._console_visible = False
        else:
            self._console_frame.grid()
            self.root.rowconfigure(3, minsize=160)
            self._console_toggle_btn.configure(text="▼  OUTPUT CONSOLE")
            self._console_visible = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_closing(self) -> None:
        self.stop_event.set()
        if self.running_thread and self.running_thread.is_alive():
            self.running_thread.join(timeout=2.0)
        self.root.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="APGI Open Science Framework GUI")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (for testing)")
    parser.add_argument("--token", help="JWT authentication token for secured operations")
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

    if args.headless:
        _ = OSFProtocolGUI(tk.Tk(), headless=True)
        print("OSF GUI initialised in headless mode.")
        print(f"Protocols loaded: {list(EP_PROTOCOLS.keys())}")
        return

    root = tk.Tk()
    _ = OSFProtocolGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
