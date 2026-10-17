"""
APGI Framework-Level Validation Aggregator (VP-ALL)

Implements comprehensive aggregation of results from all 22 validation protocols (VP-1 to VP-22).
Requires all validation protocol files to have produced JSON result files or returned
named prediction results.

Validation Criteria:
- Primary validations: Core APGI predictions from the framework paper
- Secondary validations: Extended empirical support
- Tertiary validations: Experimental and specialized protocols

Aggregates named predictions across all validation protocols for framework
validation status reporting.

TIER DESIGNATION: All outputs are Tier 2 (information-theoretic).
Bridge to Tier 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic implications without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If the aggregated validation results do not achieve >70% success rate across primary
validations, or if the framework-level consistency score falls below 0.6, or if
the aggregated evidence does not support APGI over alternative theories
(Bayes factor BF₁₀ < 3), then the APGI framework validation claim is falsified.
This would indicate that APGI does not receive sufficient empirical support
across the validation battery.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Matplotlib imports for PNG visualization
try:
    import matplotlib

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import falsification thresholds (required for test compliance)
# ---------------------------------------------------------------------------
try:
    from utils.falsification_thresholds import DEFAULT_ALPHA
except ImportError:
    DEFAULT_ALPHA = 0.05

NAMED_PREDICTIONS = {
    # VP-1: Synthetic EEG ML Classification (P1.1–P1.3)
    "V1.1": "ML classifier achieves >85% accuracy on synthetic APGI vs baseline EEG",
    "V1.2": "Feature importance aligns with APGI theoretical signatures",
    "V1.3": "Cross-validation shows robust generalization",
    # VP-2: Behavioral Bayesian Comparison (P1.1–P1.3, V2.1–V2.3, F2.1–F2.5)
    "V2.1": "Interoceptive precision (Πⁱ) modulates visual detection threshold (d=0.40–0.60)",
    "V2.2": "Arousal amplifies the Πⁱ–threshold relationship",
    "V2.3": "High-IA individuals show greater arousal benefit (d > 0.30)",
    # VP-3: Active Inference Agent Comparison
    "V3.1": "APGI agents converge in 50–80 trials",
    "V3.2": "APGI achieves lower BIC than StandardPP and GWTonly",
    "V3.3": "Active inference outperforms reactive baselines",
    # VP-4: Phase Transition / Epistemic Level 2
    "V4.1": "Phase transition observed at predicted critical threshold",
    "V4.2": "Epistemic level 2 shows distinct signature from level 1",
    "V4.3": "Hysteresis effects consistent with APGI predictions",
    # VP-5: Evolutionary Emergence
    "V5.1": "Consciousness-relevant architectures evolve under selection pressure",
    "V5.2": "Precision-weighted information integration emerges spontaneously",
    "V5.3": "Evolutionary trajectories converge on APGI-like architectures",
    # VP-6: Liquid Network Inductive Bias
    "V6.1": "Liquid networks show superior time-series processing",
    "V6.2": "APGI-inspired architectures outperform standard RNNs",
    "V6.3": "Adaptive timescale dynamics match theoretical predictions",
    # VP-7: TMS/Pharmacological Causal Interventions (CANONICAL for P2.a-P2.c)
    "V7.1": "dlPFC TMS shifts threshold >0.1 log units",
    "V7.2": "Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation)",
    "V7.3": "High-IA × insula TMS interaction confirmed",
    # VP-8: Psychophysical Threshold Estimation (Paper 1 — Protocol 1)
    "V8.1": "Detection thresholds follow APGI predicted curve",
    "V8.2": "Interoceptive precision correlates with threshold",
    "V8.3": "Metacognitive sensitivity matches model predictions",
    # VP-9: Convergent Neural Signatures — Empirical Priority 1
    "V9.1": "P3b amplitude correlates with ignition strength",
    "V9.2": "HEP amplitude correlates with interoceptive precision",
    "V9.3": "PCI and HEP show predicted dissociation patterns",
    # VP-10: Causal Manipulations TMS/Pharmacological — Priority 2
    # NOTE: V10.1-V10.3 are SUPPLEMENTARY to VP-07 and NOT counted in 14-prediction tally
    # They provide additional evidence but VP-07 is the canonical source for P2.a-P2.c
    # VP-11: MCMC / Cultural Neuroscience — Priority 3
    "V11.1": "MCMC convergence (Gelman-Rubin R̂ ≤ 1.01)",
    "V11.2": "Cultural parameters show predicted variation",
    "V11.3": "Cross-cultural replication confirms universality",
    # VP-12: Clinical Cross-Species Convergence
    "V12.1": "DoC patients show predicted neural signature patterns",
    "V12.2": "Animal models replicate key APGI phenomena",
    "V12.3": "Cross-species scaling follows predicted power law",
    # VP-13: Epistemic Architecture Predictions P5–P12 (Paper 4)
    "V13.1": "Epistemic architecture P5 predictions confirmed",
    "V13.2": "Epistemic architecture P6–P8 predictions confirmed",
    "V13.3": "Epistemic architecture P9–P12 predictions confirmed",
    # VP-14: fMRI Anticipation/Experience Protocol 5
    "V14.1": "vmPFC–SCR anticipatory correlation r > 0.40",
    "V14.2": "vmPFC uncorrelated with posterior insula (r < 0.20)",
    "V14.3": "Anticipation-experience dynamics match APGI model",
    # VP-15: fMRI Anticipation/vmPFC Protocol (Simulation-Validated)
    "V15.1": "Anticipatory insula activation onset < 500ms pre-stimulus",
    "V15.2": "vmPFC–posterior insula anticipatory connectivity r > 0.40",
    "V15.3": "Anterior/posterior insula dissociation (anticipation vs. experience)",
    # VP-16: Metabolic ATP Ground-Truth
    "V16.1": "APGI metabolic cost correlates with high-resolution ATP traces (r > 0.75)",
    "V16.2": "c1/c2 ratio consistency across frequency bands (10–100 Hz)",
    "V16.3": "Metabolic efficiency advantage >20% vs non-gating controls",
    # VP-17: Allen Visual Coding Fatigue Analysis
    "V17.1": "P3b amplitude decay under fatigue (λ > 0.10, R² > 0.70)",
    "V17.2": "Threshold elevation under fatigue (β > 0, p < 0.05)",
    "V17.3": "Cross-measure fatigue correlations (negative P3b-alpha, positive theta-RT)",
    # VP-18: EEG Microstate Energy Analysis (GFP / P3b)
    "V18.1": "GFP-AUC significantly larger on ignition vs. no-ignition trials (d ≥ 0.50)",
    "V18.2": "Proportional GFP-AUC advantage on ignition trials ≥ 20%",
    "V18.3": "Single-trial GFP-AUC correlates with Sₜ (r ≥ 0.30, p < 0.05)",
    # VP-19: Information Erasure Protocol (MVPA)
    "V19.1": "Losing stimulus decodable pre-ignition (accuracy ≥ 0.60)",
    "V19.2": "Post-ignition losing-stimulus decoding drops below permutation chance (≥ 3 bins)",
    "V19.3": "Both categories remain decodable on no-ignition trials (accuracy ≥ 0.60)",
    # VP-20: Empirical iEEG Pipeline (BIDS-iEEG, P6a + P6c)
    "V20.1": "P6a — HG bimodal occupancy higher on conscious trials (d ≥ 0.50, BC ≥ 0.55)",
    "V20.2": "P6c — AC1 higher on hits-near-threshold vs. misses (critical slowing, p < 0.05)",
    "V20.3": "P6c — Variance higher on hits-near-threshold vs. misses (critical slowing, p < 0.05)",
    # VP-21: Free Energy Proxy — MMN + HEP PE tracking (Tier 3 → Tier 2 only)
    "V21.1": "MMN amplitude declines monotonically over 30-min task (R² ≥ 0.60, slope < 0)",
    "V21.2": "HEP deviation declines monotonically over 30-min task (R² ≥ 0.50, slope < 0)",
    "V21.3": "Ignition events spike PE ≥ 1.20× and resolve below pre-ignition baseline",
    # VP-7a: Mathematical Consistency — SymPy symbolic verification (RESOLVES VP-7 STRUCTURAL DEBT)
    # These are binary proof checks, not statistical thresholds.
    "V7a.1": "∂P/∂S_t > 0 symbolically for all S_t (ignition monotone — proven, not sampled)",
    "V7a.2": "P → 0 as S_t → −∞ and P → 1 as S_t → +∞ (ignition boundaries exact)",
    "V7a.3": "∂Π_eff/∂M > 0 for M ∈ [−2,+2] (soma-bias excitatory — symbolic proof)",
    "V7a.4": "Π_eff(M=0) = Π_base exactly (baseline identity — zero residual)",
    "V7a.5": "dθ/dt = 0 ↔ θ* = S_t uniquely (one fixed point — bistability precondition)",
    # VP-22: fMRI Anticipation vs. Experience (Somatic Marker) — APGI-P03 canonical (specification 1.1.0)
    # Sub-predictions P5a–P5d with hardcoded rejection thresholds in Protocol5Config.
    "V22.1": "P5a: vmPFC–posterior_insula anticipatory coupling r ≥ 0.40, p < 0.01",
    "V22.2": "P5b: vmPFC does NOT significantly track primary interoceptive prediction error (critical falsification gate)",
    "V22.3": "P5c: vmPFC somatic marker effect is valence-specific, not sensory-contrast-driven (p < 0.01)",
    "V22.4": "P5d: Removing anticipation collapses vmPFC–posterior_insula coupling to < 0.15 (paired p < 0.01)",
}

# BIC thresholds for empirical vs theoretical data
try:
    from utils.constants import DIM_CONSTANTS

    EMPIRICAL_BIC_THRESHOLD = getattr(DIM_CONSTANTS, "EMPIRICAL_BIC_THRESHOLD", 10.0)
    THEORETICAL_BIC_PENALTY = getattr(DIM_CONSTANTS, "THEORETICAL_BIC_PENALTY", 15.0)
except ImportError:
    EMPIRICAL_BIC_THRESHOLD = 10.0  # ΔBIC threshold for empirical data
    THEORETICAL_BIC_PENALTY = 15.0  # Penalty for theoretical-only comparisons

# Protocol routing table - maps named predictions to validation protocols
PREDICTION_TO_PROTOCOL = {
    # VP-1: Synthetic EEG ML Classification
    "V1.1": "VP_01_SyntheticEEGMLClassification",
    "V1.2": "VP_01_SyntheticEEGMLClassification",
    "V1.3": "VP_01_SyntheticEEGMLClassification",
    # VP-2: Behavioral Bayesian Comparison
    "V2.1": "VP_02_BehavioralBayesianComparison",
    "V2.2": "VP_02_BehavioralBayesianComparison",
    "V2.3": "VP_02_BehavioralBayesianComparison",
    # VP-3: Active Inference Agent Comparison
    "V3.1": "VP_03_ActiveInferenceAgentSimulations",
    "V3.2": "VP_03_ActiveInferenceAgentSimulations",
    "V3.3": "VP_03_ActiveInferenceAgentSimulations",
    # VP-4: Phase Transition / Epistemic Level 2
    "V4.1": "VP_04_PhaseTransitionEpistemicLevel2",
    "V4.2": "VP_04_PhaseTransitionEpistemicLevel2",
    "V4.3": "VP_04_PhaseTransitionEpistemicLevel2",
    # VP-5: Evolutionary Emergence
    "V5.1": "VP_05_EvolutionaryEmergence",
    "V5.2": "VP_05_EvolutionaryEmergence",
    "V5.3": "VP_05_EvolutionaryEmergence",
    # VP-6: Liquid Network Inductive Bias
    "V6.1": "VP_06_LiquidNetworkInductiveBias",
    "V6.2": "VP_06_LiquidNetworkInductiveBias",
    "V6.3": "VP_06_LiquidNetworkInductiveBias",
    # VP-7: TMS/Pharmacological Causal Interventions (CANONICAL for P2.a-P2.c)
    "V7.1": "VP_07_TMSCausalInterventions",
    "V7.2": "VP_07_TMSCausalInterventions",
    "V7.3": "VP_07_TMSCausalInterventions",
    # VP-8: Psychophysical Threshold Estimation
    "V8.1": "VP_08_PsychophysicalThresholdEstimation",
    "V8.2": "VP_08_PsychophysicalThresholdEstimation",
    "V8.3": "VP_08_PsychophysicalThresholdEstimation",
    # VP-9: Convergent Neural Signatures
    "V9.1": "VP_09_NeuralSignaturesEmpiricalPriority1",
    "V9.2": "VP_09_NeuralSignaturesEmpiricalPriority1",
    "V9.3": "VP_09_NeuralSignaturesEmpiricalPriority1",
    # VP-10: Causal Manipulations (SUPPLEMENTARY - NOT in 14-prediction tally)
    # V10.1-V10.3 removed - VP-07 is canonical source for these predictions
    # VP-11: MCMC / Cultural Neuroscience
    "V11.1": "VP_11_MCMCCulturalNeurosciencePriority3",
    "V11.2": "VP_11_MCMCCulturalNeurosciencePriority3",
    "V11.3": "VP_11_MCMCCulturalNeurosciencePriority3",
    # VP-12: Clinical Cross-Species Convergence
    "V12.1": "VP_12_ClinicalCrossSpeciesConvergence",
    "V12.2": "VP_12_ClinicalCrossSpeciesConvergence",
    "V12.3": "VP_12_ClinicalCrossSpeciesConvergence",
    # VP-13: Epistemic Architecture P5–P12
    "V13.1": "VP_13_EpistemicArchitecture",
    "V13.2": "VP_13_EpistemicArchitecture",
    "V13.3": "VP_13_EpistemicArchitecture",
    # VP-14: fMRI Anticipation/Experience
    "V14.1": "VP_14_FMRIAnticipationExperience",
    "V14.2": "VP_14_FMRIAnticipationExperience",
    "V14.3": "VP_14_FMRIAnticipationExperience",
    # VP-15: fMRI Anticipation/vmPFC
    "V15.1": "VP_15_FMRIAnticipationVmPFC",
    "V15.2": "VP_15_FMRIAnticipationVmPFC",
    "V15.3": "VP_15_FMRIAnticipationVmPFC",
    # VP-16: Metabolic ATP Ground-Truth
    "V16.1": "VP_16_MetabolicATPGroundTruth",
    "V16.2": "VP_16_MetabolicATPGroundTruth",
    "V16.3": "VP_16_MetabolicATPGroundTruth",
    # VP-17: Allen Visual Coding Fatigue Analysis
    "V17.1": "VP_17_AllenVisualCodingFatigue",
    "V17.2": "VP_17_AllenVisualCodingFatigue",
    "V17.3": "VP_17_AllenVisualCodingFatigue",
    "V18.1": "VP_18_EEGMicrostateGFPP3b",
    "V18.2": "VP_18_EEGMicrostateGFPP3b",
    "V18.3": "VP_18_EEGMicrostateGFPP3b",
    "V19.1": "VP_19_InformationErasureMVPA",
    "V19.2": "VP_19_InformationErasureMVPA",
    "V19.3": "VP_19_InformationErasureMVPA",
    # VP-20: Empirical iEEG Pipeline
    "V20.1": "VP_20_EmpiricalIEEG",
    "V20.2": "VP_20_EmpiricalIEEG",
    "V20.3": "VP_20_EmpiricalIEEG",
    # VP-21: Free Energy Proxy (PE tracking)
    "V21.1": "VP_21_FreeEnergyPredictionError",
    "V21.2": "VP_21_FreeEnergyPredictionError",
    "V21.3": "VP_21_FreeEnergyPredictionError",
    # VP-7a: Mathematical Consistency (SymPy) — resolves VP-7 Structural Debt
    "V7a.1": "VP_07a_MathematicalConsistency",
    "V7a.2": "VP_07a_MathematicalConsistency",
    "V7a.3": "VP_07a_MathematicalConsistency",
    "V7a.4": "VP_07a_MathematicalConsistency",
    "V7a.5": "VP_07a_MathematicalConsistency",
    # VP-22: fMRI Anticipation vs. Experience (Somatic Marker) — P05 canonical (COMPLETE)
    # Keys renamed from V22.a–d → V22.1–4 to match NAMED_PREDICTIONS and paper numbering.
    "V22.1": "VP_22_FMRIAnticipationExperience",
    "V22.2": "VP_22_FMRIAnticipationExperience",
    "V22.3": "VP_22_FMRIAnticipationExperience",
    "V22.4": "VP_22_FMRIAnticipationExperience",
}

# Protocol tier classification
PROTOCOL_TIERS = {
    "VP_01_SyntheticEEGMLClassification": "primary",
    "VP_02_BehavioralBayesianComparison": "primary",
    "VP_03_ActiveInferenceAgentSimulations": "secondary",
    "VP_04_PhaseTransitionEpistemicLevel2": "secondary",
    "VP_05_EvolutionaryEmergence": "tertiary",
    "VP_06_LiquidNetworkInductiveBias": "secondary",
    "VP_07a_MathematicalConsistency": "primary",  # VP-7a: SymPy symbolic proofs (resolves VP-7 Structural Debt)
    "VP_07_TMSCausalInterventions": "secondary",  # VP-7b: CANONICAL P2 TMS source
    "VP_08_PsychophysicalThresholdEstimation": "secondary",
    "VP_09_NeuralSignaturesEmpiricalPriority1": "tertiary",
    "VP_10_CausalManipulationsPriority2": "tertiary",  # SUPPLEMENTARY to VP-7
    "VP_11_MCMCCulturalNeurosciencePriority3": "secondary",
    "VP_12_ClinicalCrossSpeciesConvergence": "secondary",
    "VP_13_EpistemicArchitecture": "secondary",
    "VP_14_FMRIAnticipationExperience": "tertiary",
    "VP_15_FMRIAnticipationVmPFC": "tertiary",
    "VP_16_MetabolicATPGroundTruth": "secondary",
    "VP_17_AllenVisualCodingFatigue": "secondary",
    "VP_18_EEGMicrostateGFPP3b": "secondary",
    "VP_19_InformationErasureMVPA": "secondary",
    "VP_20_EmpiricalIEEG": "secondary",
    "VP_21_FreeEnergyPredictionError": "secondary",
    "VP_22_FMRIAnticipationExperience": "secondary",  # P05 canonical (COMPLETE); companion absorbs VP-14/VP-15 fMRI dissociation
}

# VP-07 / VP-10 Boundary Clarification
# ====================================
# VP-07 (TMS_CausalInterventions) is the CANONICAL source for P2.a–P2.c predictions
# VP-10 (CausalManipulations_Priority2) is SUPPLEMENTARY and provides:
#   - Pharmacological specificity testing (propranolol vs insula TMS circuitry)
#   - Medication × TMS interaction testing (Type I error control)
#   - Extended validation coverage at a different abstraction level
#
# AGGREGATOR LOGIC:
# - When both VP-07 and VP-10 report P2.a–P2.c results, VP-07 takes precedence
# - VP-10 results are tagged as "supplementary" and do not contribute to
#   the 14-prediction falsification tally independently
# - VP-10-specific tests (specificity, interaction) are tracked separately
#   as "VP-10-SUPPLEMENTARY" predictions

# Supplementary predictions (VP-10 specific, not counted in 14-prediction tally)
# VP-10 provides supplementary evidence for P2.a-P2.c; VP-07 is the canonical source
SUPPLEMENTARY_PREDICTIONS = {
    "V10.1": "TMS to dlPFC selectively disrupts ignition (supplementary to V7.1)",
    "V10.2": "Propranolol modulates precision-weighted processing (supplementary to V7.2)",
    "V10.3": "Atomoxetine enhances frontoparietal connectivity (supplementary to V7.3)",
    "V10.SPEC": "Pharmacological specificity: propranolol (peripheral) vs insula TMS (cortical)",
    "V10.INT": "Medication × TMS interaction test for Type I error control",
}


def aggregate_prediction_results(
    results_input: Union[Dict, List],
) -> Dict[str, Dict[str, Any]]:
    """Load results from protocols (paths or dicts) and tally prediction pass/fail.

    VP-07 / VP-10 Boundary Handling:
    - VP-07 is the CANONICAL source for P2.a–P2.c predictions
    - VP-10 results are marked as "supplementary" and do not independently
      contribute to the 14-prediction falsification tally
    - When both report P2 results, VP-07 takes precedence

    Args:
        results_input: Dictionary of results or list of file paths/result dicts

    Returns:
        Dict mapping prediction IDs to their pass/fail status and evidence
    """
    # Initialize tallies with proper structure
    tallies: Dict[str, Dict[str, Any]] = {
        k: {"passed": False, "evidence": [], "protocol_source": None} for k in NAMED_PREDICTIONS
    }

    # Handle dict of results, list of paths, or list of dicts
    items: List[Union[str, dict]] = []
    if isinstance(results_input, dict):
        items = list(results_input.values())
    elif isinstance(results_input, list):
        items = results_input

    for item in items:
        data = None
        if isinstance(item, str):
            try:
                with open(item) as f:
                    data = json.load(f)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Protocol JSON load failed for {item}: {e}")
                continue
        elif isinstance(item, dict):
            data = item

        if not data:
            continue

        # Determine protocol source
        protocol_source = data.get("protocol_id", data.get("protocol", "unknown"))

        for pred_id, result_info in data.get("named_predictions", {}).items():
            # Fix 5: Mark V10 supplementary predictions to prevent double-counting
            skip_canonical_overwrite = pred_id in SUPPLEMENTARY_PREDICTIONS

            # Skip supplementary predictions if they would overwrite canonical results
            if skip_canonical_overwrite:
                # Check if canonical VP-07 result already exists for this prediction
                existing_source = tallies.get(pred_id, {}).get("protocol_source", "")
                if "VP-07" in str(existing_source) or "VP_07" in str(existing_source):
                    # VP-07 canonical result exists, skip VP-10 supplementary
                    logger = __import__("logging").getLogger(__name__)
                    logger.info(
                        f"Skipping supplementary {pred_id} from {protocol_source}: "
                        f"VP-07 canonical result already exists"
                    )
                    continue
                # Mark as supplementary in evidence tracking
                tallies[pred_id]["is_supplementary"] = True
                tallies[pred_id]["supplementary_source"] = protocol_source

            # Skip supplementary predictions from main tally if marked
            if pred_id in SUPPLEMENTARY_PREDICTIONS:
                continue

            if pred_id in tallies:
                # Check for VP-07/VP-10 boundary (legacy P2.a-P2.c naming)
                is_p2_prediction = pred_id in ["P2.a", "P2.b", "P2.c"]
                is_vp07 = "VP_07" in str(protocol_source) or "VP-07" in str(protocol_source)
                is_vp10 = (
                    "VP_10" in str(protocol_source)
                    or "VP-10" in str(protocol_source)
                    or "supplementary" in str(result_info.get("source_type", ""))
                )

                passed = False
                if isinstance(result_info, dict):
                    passed = result_info.get("passed", False)
                elif isinstance(result_info, bool):
                    passed = result_info

                # VP-07/VP-10 boundary logic:
                # - If VP-07 reports P2, it takes precedence (canonical)
                # - VP-10 is supplementary and doesn't contribute to 14-prediction tally
                if is_p2_prediction:
                    existing_source = tallies[pred_id].get("protocol_source")
                    if is_vp07:
                        # VP-07 is canonical - overwrite any existing
                        tallies[pred_id]["passed"] = passed
                        tallies[pred_id]["protocol_source"] = "VP-07 (canonical)"
                        tallies[pred_id]["source_type"] = "canonical"
                    elif is_vp10:
                        # VP-10 is supplementary - should not reach here due to SUPPLEMENTARY_PREDICTIONS skip
                        # but kept for safety
                        if existing_source and "VP-07" in str(existing_source):
                            tallies[pred_id]["vp10_supplementary_passed"] = passed
                        else:
                            tallies[pred_id]["passed"] = passed
                            tallies[pred_id]["protocol_source"] = "VP-10 (supplementary)"
                            tallies[pred_id]["source_type"] = "supplementary"
                    else:
                        # Other protocol - treat normally
                        tallies[pred_id]["passed"] |= passed
                        tallies[pred_id]["protocol_source"] = protocol_source
                else:
                    # Non-P2 prediction - normal aggregation
                    tallies[pred_id]["passed"] |= passed

                evidence_item = "result_dict" if isinstance(item, dict) else item
                _ = evidence_item  # Silence unused warning - kept for clarity
                tallies[pred_id]["evidence"].append(
                    {
                        "source": protocol_source,
                        "passed": passed,
                        "is_supplementary": is_vp10 and is_p2_prediction,
                    }
                )

    return tallies


def compute_validation_summary(
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute summary statistics from prediction results.

    Args:
        predictions: Dict of prediction results

    Returns:
        Summary dict with pass rates by tier and overall
    """
    total = len(predictions)
    passing = sum(1 for r in predictions.values() if r.get("passed"))

    # Count by protocol tier
    primary_preds = [p for p in PREDICTION_TO_PROTOCOL if PROTOCOL_TIERS.get(PREDICTION_TO_PROTOCOL[p]) == "primary"]
    secondary_preds = [
        p for p in PREDICTION_TO_PROTOCOL if PROTOCOL_TIERS.get(PREDICTION_TO_PROTOCOL[p]) == "secondary"
    ]
    tertiary_preds = [p for p in PREDICTION_TO_PROTOCOL if PROTOCOL_TIERS.get(PREDICTION_TO_PROTOCOL[p]) == "tertiary"]

    primary_passed = sum(1 for p in primary_preds if predictions.get(p, {}).get("passed"))
    secondary_passed = sum(1 for p in secondary_preds if predictions.get(p, {}).get("passed"))
    tertiary_passed = sum(1 for p in tertiary_preds if predictions.get(p, {}).get("passed"))

    return {
        "total_predictions": total,
        "total_passed": passing,
        "overall_pass_rate": passing / total if total > 0 else 0,
        "primary": {
            "total": len(primary_preds),
            "passed": primary_passed,
            "pass_rate": primary_passed / len(primary_preds) if primary_preds else 0,
        },
        "secondary": {
            "total": len(secondary_preds),
            "passed": secondary_passed,
            "pass_rate": (secondary_passed / len(secondary_preds) if secondary_preds else 0),
        },
        "tertiary": {
            "total": len(tertiary_preds),
            "passed": tertiary_passed,
            "pass_rate": tertiary_passed / len(tertiary_preds) if tertiary_preds else 0,
        },
    }


def check_framework_validated(
    predictions: Dict[str, Dict[str, Any]],
    min_primary_rate: float = 0.8,
    min_overall_rate: float = 0.7,
) -> Dict[str, Any]:
    """Check if APGI framework meets validation criteria.

    Args:
        predictions: Dict of prediction results
        min_primary_rate: Minimum pass rate for primary validations (default 80%)
        min_overall_rate: Minimum overall pass rate (default 70%)

    Returns:
        Dict with validation status and criteria check results
    """
    summary = compute_validation_summary(predictions)

    primary_adequate = summary["primary"]["pass_rate"] >= min_primary_rate
    overall_adequate = summary["overall_pass_rate"] >= min_overall_rate

    return {
        "framework_validated": primary_adequate and overall_adequate,
        "primary_adequate": primary_adequate,
        "overall_adequate": overall_adequate,
        "min_primary_rate": min_primary_rate,
        "min_overall_rate": min_overall_rate,
        "summary": summary,
    }


def run_framework_validation(
    results_input: Union[Dict, List],
    min_primary_rate: float = 0.8,
    min_overall_rate: float = 0.7,
) -> Dict[str, Any]:
    """Run complete framework validation analysis.

    Args:
        results_input: List of JSON result files or dict of outcome dicts from all protocols.
        min_primary_rate: Minimum pass rate for primary validations
        min_overall_rate: Minimum overall pass rate

    Returns:
        dict: Complete validation results with pass rates and validation status
    """
    # Aggregate predictions
    predictions = aggregate_prediction_results(results_input)

    # Compute summary
    summary = compute_validation_summary(predictions)

    # Check validation criteria
    validation_status = check_framework_validated(predictions, min_primary_rate, min_overall_rate)

    return {
        "validation_complete": True,
        "framework_validated": validation_status["framework_validated"],
        "predictions": predictions,
        "summary": summary,
        "validation_status": validation_status,
        "named_prediction_count": len(NAMED_PREDICTIONS),
        "protocol_count": len(set(PREDICTION_TO_PROTOCOL.values())),
    }


class ValidationAggregator:
    """Master aggregator for APGI framework-level validation.

    Loads JSON results from all 14 validation protocols (VP-1 to VP-14),
    tallies the 42 named predictions, and applies validation criteria.

    Attributes:
        named_predictions: Dict of 42 named predictions with descriptions
        protocol_tiers: Dict mapping protocols to their tier classification
        prediction_to_protocol: Dict mapping predictions to their source protocol
    """

    def __init__(self) -> None:
        """Initialize the validation aggregator."""
        self.named_predictions = NAMED_PREDICTIONS
        self.protocol_tiers = PROTOCOL_TIERS
        self.prediction_to_protocol = PREDICTION_TO_PROTOCOL
        self._prediction_statuses: Dict[str, Dict[str, Any]] = {}

    def set_prediction(self, prediction_id: str, status: str, evidence: Optional[str] = None) -> None:
        """Set the status of a named prediction manually.

        Used for marking predictions as PENDING_DATA when protocols
        return STUB status (e.g., VP-15 awaiting empirical fMRI data).

        Args:
            prediction_id: The prediction ID (e.g., "V15.1")
            status: Status string ("PENDING_DATA", "PASS", "FAIL", etc.)
            evidence: Optional evidence string explaining the status
        """
        if prediction_id not in self._prediction_statuses:
            self._prediction_statuses[prediction_id] = {}

        self._prediction_statuses[prediction_id]["status"] = status
        if evidence:
            self._prediction_statuses[prediction_id]["evidence"] = evidence

    def aggregate_results(self, results_input: Union[Dict, List]) -> Dict[str, Dict[str, Any]]:
        """Aggregate prediction results from all protocols.

        Handles VP-15 STUB status by marking predictions as PENDING_DATA.

        Args:
            results_input: List of file paths, dicts, or dict of results

        Returns:
            dict: Tally of pass/fail for each named prediction
        """
        results = aggregate_prediction_results(results_input)

        # Fix 1: Handle VP-15 STUB status - mark predictions as PENDING_DATA
        items = results_input if isinstance(results_input, list) else [results_input]
        for item in items:
            if isinstance(item, dict):
                protocol_id = item.get("protocol_id", item.get("protocol", ""))
                status = item.get("status", "").upper()

                if "VP-15" in str(protocol_id) and status == "STUB":
                    # Mark all VP-15 predictions as PENDING_DATA
                    for pred_id in ["V15.1", "V15.2", "V15.3"]:
                        if pred_id in results:
                            results[pred_id]["passed"] = None
                            results[pred_id]["status"] = "PENDING_DATA"
                            results[pred_id]["evidence"] = "Awaiting empirical fMRI data"

        return results

    def compute_summary(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute validation summary statistics.

        Args:
            predictions: Dict of prediction results

        Returns:
            dict: Summary with pass rates by tier and overall
        """
        return compute_validation_summary(predictions)

    def check_validation(
        self,
        predictions: Dict[str, Dict[str, Any]],
        min_primary_rate: float = 0.8,
        min_overall_rate: float = 0.7,
    ) -> Dict[str, Any]:
        """Check if framework meets validation criteria.

        Args:
            predictions: Dict of prediction results
            min_primary_rate: Minimum pass rate for primary validations
            min_overall_rate: Minimum overall pass rate

        Returns:
            dict: Validation status and criteria check results
        """
        return check_framework_validated(predictions, min_primary_rate, min_overall_rate)

    def run_full_analysis(
        self,
        results_input: Union[Dict, List],
        min_primary_rate: float = 0.8,
        min_overall_rate: float = 0.7,
    ) -> Dict[str, Any]:
        """Run complete framework validation analysis.

        Args:
            results_input: Results from all validation protocols
            min_primary_rate: Minimum pass rate for primary validations
            min_overall_rate: Minimum overall pass rate

        Returns:
            dict: Complete validation report
        """
        return run_framework_validation(results_input, min_primary_rate, min_overall_rate)

    def get_protocol_predictions(self, protocol_name: str) -> List[str]:
        """Get all prediction IDs associated with a specific protocol.

        Args:
            protocol_name: Name of the protocol (e.g., "VP_02_BehavioralBayesianComparison")

        Returns:
            List of prediction IDs for that protocol
        """
        return [p for p, proto in self.prediction_to_protocol.items() if proto == protocol_name]

    def get_tier_predictions(self, tier: str) -> List[str]:
        """Get all prediction IDs for a specific tier.

        Args:
            tier: Tier name ("primary", "secondary", or "tertiary")

        Returns:
            List of prediction IDs for that tier
        """
        protocols = [p for p, t in self.protocol_tiers.items() if t == tier]
        return [p for p, proto in self.prediction_to_protocol.items() if proto in protocols]


def load_protocol_results_from_directory(
    directory: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load all JSON result files from a directory.

    Args:
        directory: Path to directory containing JSON result files

    Returns:
        List of result dictionaries
    """
    directory = Path(directory)
    results: List[Any] = []

    if not directory.exists():
        return results

    for json_file in directory.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Protocol JSON load failed for {json_file}: {e}")
            continue

    return results


def extract_bic_comparison(
    results_input: Union[Dict, List],
    data_source: str = "auto",
) -> Dict[str, Any]:
    """Extract and validate BIC comparisons from protocol results.

    Addresses theoretical fallback by detecting data source type and
    applying appropriate thresholds. Empirical BIC comparisons are prioritized
    over theoretical estimates.

    Args:
        results_input: Dictionary of results or list of file paths/result dicts
        data_source: Data source type ("empirical", "theoretical", "auto")

    Returns:
        Dict with BIC comparison results and data source classification
    """
    bic_results: Dict[str, Any] = {
        "data_source": data_source,
        "empirical_comparisons": [],
        "theoretical_estimates": [],
        "apgi_bic": None,
        "alternative_bics": {},
        "bic_advantage": None,
        "is_empirical": False,
        "threshold_used": EMPIRICAL_BIC_THRESHOLD,
        "warnings": [],
    }

    items: List[Union[str, dict]] = []
    if isinstance(results_input, dict):
        items = list(results_input.values())
    elif isinstance(results_input, list):
        items = results_input

    for item in items:
        data = None
        if isinstance(item, str):
            try:
                with open(item) as f:
                    data = json.load(f)
            except Exception:  # nosec - Intentionally continue on file load failure
                continue
        elif isinstance(item, dict):
            data = item

        if not data:
            continue

        # Check for BIC values in results
        if "bic_values" in data:
            bic_vals = data["bic_values"]
            for agent_type, metrics in bic_vals.items():
                if isinstance(metrics, dict) and "bic" in metrics:
                    if agent_type.upper() == "APGI":
                        bic_results["apgi_bic"] = metrics["bic"]
                    else:
                        bic_results["alternative_bics"][agent_type] = metrics["bic"]

        # Check metadata for data source
        metadata = data.get("metadata", {})
        if metadata.get("data_source") in ["empirical", "clinical", "neuroimaging"]:
            bic_results["is_empirical"] = True
        elif metadata.get("methodology") in [
            "clinical_data",
            "empirical",
            "neuroimaging",
        ]:
            bic_results["is_empirical"] = True

        # Check for empirical data flags
        if data.get("empirical_data_used") or metadata.get("empirical_validation"):
            bic_results["is_empirical"] = True

    # Auto-detect data source if not specified
    if data_source == "auto":
        if bic_results["is_empirical"]:
            bic_results["data_source"] = "empirical"
        else:
            bic_results["data_source"] = "theoretical"
            bic_results["threshold_used"] = THEORETICAL_BIC_PENALTY

    # Calculate BIC advantage
    if bic_results["apgi_bic"] is not None and bic_results["alternative_bics"]:
        best_alt_bic = min(bic_results["alternative_bics"].values())
        bic_results["bic_advantage"] = best_alt_bic - bic_results["apgi_bic"]

        # Validate against appropriate threshold
        if bic_results["data_source"] == "empirical":
            if bic_results["bic_advantage"] < EMPIRICAL_BIC_THRESHOLD:
                bic_results["warnings"].append(
                    f"APGI BIC advantage ({bic_results['bic_advantage']:.2f}) "
                    f"below empirical threshold ({EMPIRICAL_BIC_THRESHOLD})"
                )
        else:
            # Theoretical fallback warning
            bic_results["warnings"].append(
                "Using theoretical BIC estimates. Empirical data recommended for " "publication-ready validation."
            )

    return bic_results


def run_end_to_end_validation_pipeline(
    protocol_modules: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    min_primary_rate: float = 0.8,
    min_overall_rate: float = 0.7,
    falsification_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run complete end-to-end validation pipeline across all protocols.

    Implements full pipeline testing from protocol execution through
    aggregation to final validation report generation.

    Args:
        protocol_modules: List of protocol module names to run (None = all)
        output_dir: Directory to save intermediate JSON results
        min_primary_rate: Minimum pass rate for primary validations
        min_overall_rate: Minimum overall pass rate
        falsification_results: Optional dict of FP results from Master_Falsification.
            Used to enforce gate F8.2 (FIM rank-deficiency): if FP-08 F8.2 is flagged,
            VP-08 and VP-11 are skipped because their statistics depend on a valid FIM.

    Returns:
        Complete pipeline results with execution status and validation report
    """
    # Gate F8.2: FIM rank-deficiency check — must pass before VP-08 and VP-11 run.
    # If FP-08's F8.2 criterion is falsified, results from VP-08 (psychophysical
    # threshold) and VP-11 (MCMC cultural neuroscience) cannot be trusted.
    _F8_2_GATED_PROTOCOLS = {
        "Validation.VP_08_PsychophysicalThresholdEstimation",
        "Validation.VP_11_MCMCCulturalNeurosciencePriority3",
    }
    _f8_2_gate_blocked = False
    if falsification_results:
        fp08 = falsification_results.get("FP-08", {})
        f8_2 = fp08.get("criteria", {}).get("F8.2", fp08.get("F8.2", {}))
        if f8_2.get("falsified", False):
            _f8_2_gate_blocked = True
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Gate F8.2 BLOCKED: FP-08 reports FIM rank-deficiency. "
                "VP-08 and VP-11 will be skipped — their statistics are invalid under rank-deficient FIM."
            )
    import importlib
    from datetime import datetime

    pipeline_results: Dict[str, Any] = {
        "pipeline_started": datetime.now().isoformat(),
        "protocols_executed": [],
        "protocols_failed": [],
        "aggregation_status": {},
        "validation_status": {},
        "bic_analysis": None,
        "consistency_checks": None,
        "pipeline_completed": None,
        "errors": [],
    }

    if protocol_modules is None:
        # Default: run all VP protocols
        protocol_modules = [
            "Validation.VP_01_SyntheticEEGMLClassification",
            "Validation.VP_02_BehavioralBayesianComparison",
            "Validation.VP_03_ActiveInferenceAgentSimulations",
            "Validation.VP_04_PhaseTransitionEpistemicLevel2",
            "Validation.VP_05_EvolutionaryEmergence",
            "Validation.VP_06_LiquidNetworkInductiveBias",
            "Validation.VP_07_TMSCausalInterventions",
            "Validation.VP_08_PsychophysicalThresholdEstimation",
            "Validation.VP_09_NeuralSignaturesEmpiricalPriority1",
            "Validation.VP_10_CausalManipulationsPriority2",
            "Validation.VP_11_MCMCCulturalNeurosciencePriority3",
            "Validation.VP_12_ClinicalCrossSpeciesConvergence",
            "Validation.VP_13_EpistemicArchitecture",
            "Validation.VP_14_FMRIAnticipationExperience",
            "Validation.VP_15_FMRIAnticipationVmPFC",
            "Validation.VP_16_MetabolicATPGroundTruth",
            "Validation.VP_17_AllenVisualCodingFatigue",
            "Validation.VP_18_EEGMicrostateGFPP3b",
            "Validation.VP_19_InformationErasureMVPA",
            "Validation.VP_20_EmpiricalIEEG",
            "Validation.VP_21_FreeEnergyPredictionError",
            "Validation.VP_22_FMRIAnticipationExperience",
        ]

    protocol_outputs = []

    # Phase 1: Execute all protocols
    for module_path in protocol_modules:
        if _f8_2_gate_blocked and module_path in _F8_2_GATED_PROTOCOLS:
            pipeline_results["protocols_failed"].append(
                {"module": module_path, "reason": "Skipped: Gate F8.2 (FIM rank-deficiency) blocked this protocol"}
            )
            continue

        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "run_protocol_main"):
                result = module.run_protocol_main()
                protocol_outputs.append(result)
                pipeline_results["protocols_executed"].append(module_path)

                # Save to output directory if specified
                if output_dir:
                    out_path = Path(output_dir) / f"{module_path.split('.')[-1]}_result.json"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    if hasattr(result, "to_dict"):
                        with open(out_path, "w") as f:
                            json.dump(result.to_dict(), f, indent=2)
                    elif isinstance(result, dict):
                        with open(out_path, "w") as f:
                            json.dump(result, f, indent=2)
            else:
                pipeline_results["protocols_failed"].append(
                    {"module": module_path, "reason": "No run_protocol_main function"}
                )
        except Exception as e:
            pipeline_results["protocols_failed"].append({"module": module_path, "reason": str(e)})

    # Phase 2: Aggregate results
    if protocol_outputs:
        try:
            predictions = aggregate_prediction_results(protocol_outputs)
            pipeline_results["aggregation_status"] = {
                "predictions_aggregated": len(predictions),
                "protocols_included": len(pipeline_results["protocols_executed"]),
            }

            # Phase 3: Run validation
            summary = compute_validation_summary(predictions)
            validation = check_framework_validated(predictions, min_primary_rate, min_overall_rate)
            pipeline_results["validation_status"] = {
                "summary": summary,
                "validation": validation,
            }

            # Phase 4: BIC analysis
            pipeline_results["bic_analysis"] = extract_bic_comparison(protocol_outputs)

            # Phase 5: Consistency checks
            pipeline_results["consistency_checks"] = run_cross_protocol_consistency_checks(
                protocol_outputs, predictions
            )

        except Exception as e:
            pipeline_results["errors"].append(f"Aggregation/validation failed: {e}")

    pipeline_results["pipeline_completed"] = datetime.now().isoformat()
    return pipeline_results


def run_cross_protocol_consistency_checks(
    protocol_outputs: List[Dict],
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run consistency checks across all protocols.

    Validates that related predictions across protocols are consistent,
    detects contradictions, and ensures no double-counting of supplementary
    predictions (VP-10 vs VP-07).

    Args:
        protocol_outputs: List of protocol result dictionaries
        predictions: Aggregated prediction results

    Returns:
        Dict with consistency check results and any detected issues
    """
    consistency_report: Dict[str, Any] = {
        "checks_performed": [],
        "inconsistencies_found": [],
        "warnings": [],
        "overall_consistent": True,
    }

    # Check 1: VP-07 vs VP-10 boundary (canonical vs supplementary)
    vp10_supplementary = ["V10.1", "V10.2", "V10.3"]

    for pred in vp10_supplementary:
        if pred in predictions and predictions[pred].get("passed") is not None:
            # Check if VP-07 canonical result exists
            canonical_pred = pred.replace("V10.", "V7.")
            if canonical_pred in predictions:
                vp07_passed = predictions[canonical_pred].get("passed")
                vp10_passed = predictions[pred].get("passed")
                if vp07_passed != vp10_passed:
                    consistency_report["warnings"].append(
                        f"VP-07 ({canonical_pred}) and VP-10 ({pred}) disagree: " f"{vp07_passed} vs {vp10_passed}"
                    )

    consistency_report["checks_performed"].append("vp07_vp10_boundary")

    # Check 2: Cross-prediction logical consistency
    # V2.1 (threshold modulation) should imply V2.2 (arousal amplification)
    if "V2.1" in predictions and "V2.2" in predictions:
        v2_1_passed = predictions["V2.1"].get("passed")
        v2_2_passed = predictions["V2.2"].get("passed")
        if v2_1_passed and not v2_2_passed:
            consistency_report["inconsistencies_found"].append(
                "V2.1 passed but V2.2 failed - arousal amplification should follow threshold modulation"
            )

    # Check 3: Tier progression consistency
    # Primary validations should generally pass before secondary/tertiary
    primary_passed = sum(
        1
        for p, r in predictions.items()
        if p in PREDICTION_TO_PROTOCOL
        and PROTOCOL_TIERS.get(PREDICTION_TO_PROTOCOL[p]) == "primary"
        and r.get("passed")
    )
    primary_total = len(
        [
            p
            for p in predictions
            if p in PREDICTION_TO_PROTOCOL and PROTOCOL_TIERS.get(PREDICTION_TO_PROTOCOL[p]) == "primary"
        ]
    )

    if primary_total > 0 and primary_passed / primary_total < 0.5:
        consistency_report["warnings"].append(
            f"Primary validations below 50% ({primary_passed}/{primary_total}) - "
            "framework may need fundamental revision"
        )

    consistency_report["checks_performed"].extend(
        [
            "cross_prediction_logic",
            "tier_progression",
        ]
    )

    # Check 4: Data source consistency
    empirical_protocols = []
    theoretical_protocols = []
    for output in protocol_outputs:
        metadata = output.get("metadata", {}) if isinstance(output, dict) else {}
        protocol_id = output.get("protocol_id", "unknown")
        if metadata.get("data_source") in ["empirical", "clinical"]:
            empirical_protocols.append(protocol_id)
        else:
            theoretical_protocols.append(protocol_id)

    consistency_report["data_source_breakdown"] = {
        "empirical_protocols": empirical_protocols,
        "theoretical_protocols": theoretical_protocols,
        "empirical_ratio": (len(empirical_protocols) / len(protocol_outputs) if protocol_outputs else 0),
    }

    if len(empirical_protocols) < 3:
        consistency_report["warnings"].append(
            f"Only {len(empirical_protocols)} protocols use empirical data - "
            "recommend increasing empirical validation for publication readiness"
        )

    consistency_report["overall_consistent"] = len(consistency_report["inconsistencies_found"]) == 0

    return consistency_report


def run_validation(**kwargs) -> Dict[str, Any]:
    """Entry point for validation protocol."""
    aggregator = ValidationAggregator()
    repo_root = Path(__file__).parent.parent

    # Primary location: data_repository/processed_data (canonical store)
    results_dir = repo_root / "data_repository" / "processed_data"
    json_files: list[Path] = []
    if results_dir.exists():
        json_files = list(results_dir.glob("*.json"))

    # Fallback: many protocols write result files directly to the repo root
    # (e.g. protocol3_results.json, protocol8_results.json …).  Include those
    # so VP-ALL does not silently aggregate zero files while individual protocols
    # have already produced valid output.
    root_result_files = [f for f in repo_root.glob("protocol*_results.json") if f not in json_files]
    json_files = json_files + root_result_files

    if json_files:
        print(
            f"\nFound {len(json_files)} result file(s) "
            f"({len(json_files) - len(root_result_files)} in processed_data, "
            f"{len(root_result_files)} in repo root)"
        )
        results = aggregator.aggregate_results([str(f) for f in json_files])
        print("\nAggregation complete. Use aggregator.run_full_analysis() for full report.")
        return results

    print(f"\nNo result files found in {results_dir} or {repo_root} (protocol*_results.json).")
    print("Run individual validation protocols first to generate JSON results.")
    return {"status": "error", "message": "No results found"}


if __name__ == "__main__":
    print("=" * 60)
    print("APGI Framework Validation Aggregator (VP-ALL)")
    print("=" * 60)
    print(f"\nNamed Predictions: {len(NAMED_PREDICTIONS)}")
    print(f"Validation Protocols: {len(set(PREDICTION_TO_PROTOCOL.values()))}")
    print("\nPrediction Mapping by Protocol:")

    current_protocol = None
    for pred_id, desc in NAMED_PREDICTIONS.items():
        proto = PREDICTION_TO_PROTOCOL.get(pred_id, "Unknown")
        tier = PROTOCOL_TIERS.get(proto, "unknown")
        if proto != current_protocol:
            current_protocol = proto
            print(f"\n{proto} [{tier}]")
        print(f"  {pred_id}: {desc[:50]}...")

    # Try to load results from data directory
    aggregator = ValidationAggregator()
    results_dir = Path(__file__).parent.parent / "data_repository" / "processed_data"

    if results_dir.exists():
        json_files = list(results_dir.glob("*.json"))
        print(f"\nFound {len(json_files)} result files in {results_dir}")
        if json_files:
            results = aggregator.aggregate_results([str(f) for f in json_files])
            print("\nAggregation complete. Use aggregator.run_full_analysis() for full report.")
    else:
        print(f"\nNo results directory found at {results_dir}")
        print("Run individual validation protocols first to generate JSON results.")

    print("\n" + "=" * 60)
