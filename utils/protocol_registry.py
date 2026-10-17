"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type


@dataclass
class ProtocolInfo:
    """Information about a falsification protocol."""

    canonical_id: str
    filename: str
    title: str
    description: str
    priority_level: str  # "P1", "P2", "F1", "F2", etc.
    category: str  # "Core", "Validation", "Benchmark", etc.
    implemented: bool = True
    aliases: List[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class PROTOCOL_REGISTRY:
    """
    Central registry for APGI falsification protocols.

    Maps canonical protocol IDs to actual filenames and provides metadata
    for protocol discovery and loading.
    """

    _protocols: Dict[str, ProtocolInfo] = {}

    @classmethod
    def register_protocol(cls, protocol_info: ProtocolInfo) -> None:
        """Register a protocol in the registry."""
        cls._protocols[protocol_info.canonical_id] = protocol_info

        # Also register aliases
        for alias in protocol_info.aliases:
            cls._protocols[alias] = protocol_info

    @classmethod
    def get_protocol(cls, protocol_id: str) -> Optional[ProtocolInfo]:
        """Get protocol information by ID or alias."""
        return cls._protocols.get(protocol_id)

    @classmethod
    def get_filename(cls, protocol_id: str) -> Optional[str]:
        """Get filename for a protocol by ID or alias."""
        protocol = cls.get_protocol(protocol_id)
        return protocol.filename if protocol else None

    @classmethod
    def list_protocols(cls, category: Optional[str] = None) -> List[ProtocolInfo]:
        """List all protocols, optionally filtered by category."""
        protocols = list(cls._protocols.values())

        # Remove duplicates (same protocol with different aliases)
        unique_protocols = {}
        for protocol in protocols:
            if protocol.canonical_id not in unique_protocols:
                unique_protocols[protocol.canonical_id] = protocol

        protocols = list(unique_protocols.values())

        if category:
            protocols = [p for p in protocols if p.category == category]

        return sorted(protocols, key=lambda p: p.canonical_id)

    @classmethod
    def resolve_protocol_file(cls, protocol_id: str, base_dir: Path) -> Optional[Path]:
        """Resolve a protocol ID to an actual file path."""
        filename = cls.get_filename(protocol_id)
        if filename:
            return base_dir / filename
        return None


# Initialize registry with known protocols
def _initialize_registry():
    """Initialize the protocol registry with all known protocols."""

    # ------------------------------------------------------------------
    # Experimental Protocol Specifications (JSON, protocols/ folder)
    # Specification release 1.1.0; numbering follows protocols/MANIFEST.json.
    # ------------------------------------------------------------------
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P00",
            filename="protocols/protocol_0_hep_proxy_validation.json",
            title="HEP Proxy Validation",
            description="Heartbeat-evoked potential as an index of interoceptive gain (Pred 0.A–0.C)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P00", "HEP-Proxy"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P01",
            filename="protocols/protocol_1_cardiac_eeg.json",
            title="Cardiac-Phase EEG Study of Interoceptive Precision Gating",
            description="P3b attention effect, cardiac-phase detection and HEP–P3b coupling (Pred 1.A–1.F)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P01", "EEG-Interoceptive", "HEP-P3b"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P02",
            filename="protocols/protocol_2_somatic_agent_sim.json",
            title="Somatic-Marker Agent Simulations",
            description="Seven-architecture IGT agents and somatic-marker advantages (Pred 2.A–2.G)",
            priority_level="Computational",
            category="ExperimentalProtocol",
            aliases=["P02", "ActiveInference-Sim", "SomaticMarker"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P03",
            filename="protocols/protocol_3_anticipation_fmri.json",
            title="fMRI Anticipation vs Experience",
            description="vmPFC–insula anticipatory coupling and identifiability gate (Pred 3.A–3.D)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P03", "fMRI-Anticipation", "vmPFC-Coupling"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P04",
            filename="protocols/protocol_4_metabolic_crossover.json",
            title="Metabolic-State Crossover",
            description="Allostatic threshold modulation under energy state, two-site d-prime GLMM (Pred 4.A–4.C)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P04", "Metabolic-Crossover"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P05",
            filename="protocols/protocol_5_causal_tms.json",
            title="Causal Neuromodulation of Insular and Frontoparietal Gating",
            description="Right-aINS tFUS/dTMS with modality-matched sham and PCI probes (Pred 5.A–5.E)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P05", "TMS-Insular", "TMS-Causal"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P06",
            filename="protocols/protocol_6_ignition_ieeg.json",
            title="Intracranial EEG All-or-None Ignition",
            description="Bimodal ignition, critical slowing and timescale recovery in iEEG (Pred 6.A–6.E)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P06", "iEEG-Bifurcation", "CriticalSlowing-iEEG"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P07",
            filename="protocols/protocol_7_doc_biomarker.json",
            title="Disorders of Consciousness Biomarker",
            description="Ignition capacity and interoceptive precision as DoC outcome predictors; Pred 7.G primary (Pred 7.A–7.G)",
            priority_level="Clinical",
            category="ExperimentalProtocol",
            aliases=["P07", "DoC", "Consciousness-Disorders"],
        )
    )

    # ------------------------------------------------------------------
    # Legacy protocol template (pre-1.1 benchmarks; not v1.1 decision rules)
    # ------------------------------------------------------------------
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-PROTOCOL-TEMPLATE",
            filename="config/apgi_protocol_template.json",
            title="Legacy APGI Protocol Template (pre-1.1)",
            description="Typed benchmark template retained for reference; superseded by specification 1.1.0",
            priority_level="Reference",
            category="ExperimentalProtocol",
            aliases=["PROTOCOL-TEMPLATE"],
        )
    )

    # ------------------------------------------------------------------
    # Falsification Protocols (FP_01 – FP_12)
    # ------------------------------------------------------------------
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-01",
            filename="Falsification/FP_01_ActiveInference.py",
            title="Active Inference Agents — F1 & F2",
            description="Core falsification protocols for active inference agents",
            priority_level="P1",
            category="Falsification",
            aliases=["P1", "F1", "F2", "ActiveInferenceAgents", "FP_01_ActiveInference"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-02",
            filename="Falsification/FP_02_AgentComparisonConvergenceBenchmark.py",
            title="Agent Comparison & Convergence Benchmark",
            description="Benchmark protocol for comparing agent convergence",
            priority_level="P2",
            category="Falsification",
            aliases=[
                "P2",
                "AgentComparison",
                "ConvergenceBenchmark",
                "IGT",
                "IowaGamblingTask",
                "F2-Iowa",
                "FP_02_AgentComparisonConvergenceBenchmark",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-03",
            filename="Falsification/FP_03_FrameworkLevelMultiProtocol.py",
            title="Framework Level Multi-Protocol",
            description="Multi-protocol validation at framework level",
            priority_level="Integration",
            category="Falsification",
            aliases=[
                "FrameworkLevel",
                "MultiProtocol",
                "Framework",
                "Integration",
                "FP_03_FrameworkLevelMultiProtocol",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-04",
            filename="Falsification/FP_04_PhaseTransitionEpistemicArchitecture.py",
            title="Phase Transition & Bistability (APGI-P06 linked)",
            description="Bistability + critical slowing + hysteresis; linked to APGI-P06 iEEG",
            priority_level="F4",
            category="Falsification",
            aliases=[
                "F4",
                "PhaseTransition",
                "Bistability",
                "CriticalSlowing",
                "FP_04_PhaseTransitionEpistemicArchitecture",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-05",
            filename="Falsification/FP_05_EvolutionaryPlausibility.py",
            title="Evolutionary Plausibility",
            description="Evolutionary plausibility validation",
            priority_level="Validation",
            category="Falsification",
            aliases=["Standard6", "Evolutionary", "EvoPlausibility", "FP_05_EvolutionaryPlausibility"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-06",
            filename="Falsification/FP_06_LiquidNetworkEnergyBenchmark.py",
            title="Neural Network Energy Benchmark",
            description="Energy consumption benchmark for liquid/neural networks",
            priority_level="Benchmark",
            category="Falsification",
            aliases=["EnergyBenchmark", "NeuralEnergy", "FP_06_LiquidNetworkEnergyBenchmark"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-07",
            filename="Falsification/FP_07_MathematicalConsistency.py",
            title="Mathematical Consistency of Equations",
            description="Validation of mathematical consistency in APGI equations",
            priority_level="Validation",
            category="Falsification",
            aliases=["MathematicalConsistency", "Equations", "MathConsistency", "FP_07_MathematicalConsistency"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-08",
            filename="Falsification/FP_08_ParameterSensitivityIdentifiability.py",
            title="Parameter Sensitivity & Identifiability",
            description="Analysis of parameter sensitivity and model identifiability",
            priority_level="Analysis",
            category="Falsification",
            aliases=[
                "ParameterSensitivity",
                "Sensitivity",
                "Identifiability",
                "ParamAnalysis",
                "FP_08_ParameterSensitivityIdentifiability",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-09",
            filename="Falsification/FP_09_NeuralSignaturesP3bHEP.py",
            title="Neural Signatures — EEG P3b & HEP (APGI-P01 linked)",
            description="EEG P3b/HEP signature validation; linked to APGI-P01",
            priority_level="Validation",
            category="Falsification",
            aliases=["NeuralSignatures-EEG", "EEG", "P3b", "HEP", "NeuralSignatures", "FP_09_NeuralSignaturesP3bHEP"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-10",
            filename="Falsification/FP_10_BayesianEstimationMCMC.py",
            title="Bayesian Estimation with MCMC",
            description="MCMC-based Bayesian parameter estimation",
            priority_level="Advanced",
            category="Falsification",
            # NOTE: FP_10_Falsification_BayesianEstimation_ParameterRecovery.py is
            # DEPRECATED — all FP-10 calls route through the canonical MCMC file.
            aliases=["BayesianEstimation-MCMC", "MCMC", "BayesianMCMC", "FP10", "FP_10_BayesianEstimationMCMC"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-11",
            filename="Falsification/FP_11_LiquidNetworkDynamicsEchoState.py",
            title="Liquid Network Dynamics — Echo State",
            description="Liquid network dynamics validation with echo state networks",
            priority_level="Advanced",
            category="Falsification",
            aliases=[
                "LiquidNetworkDynamics",
                "LiquidNetwork",
                "EchoState",
                "LTC",
                "FP_11_LiquidNetworkDynamicsEchoState",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-12",
            filename="Falsification/FP_12_CrossSpeciesScaling.py",
            title="Cross-Species Scaling",
            description="Cross-species validation of APGI predictions",
            priority_level="P12",
            category="Falsification",
            aliases=["P12", "CrossSpecies", "Scaling", "FP_12_CrossSpeciesScaling"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-13",
            filename="Falsification/FP_13_Clinical_CrossSpecies_Convergence.py",
            title="Clinical Cross-Species Convergence",
            description="Tertiary clinical/cross-species convergence falsification stub",
            priority_level="Tertiary",
            category="Falsification",
            implemented=False,
            aliases=[
                "P13",
                "ClinicalCrossSpecies",
                "ClinicalConvergence",
                "FP_13_Clinical_CrossSpecies_Convergence",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-14",
            filename="Falsification/FP_14_fMRI_Anticipation_vmPFC.py",
            title="fMRI Anticipation: vmPFC-like",
            description="Tertiary vmPFC anticipation falsification stub linked to EP-5",
            priority_level="Tertiary",
            category="Falsification",
            implemented=False,
            aliases=["P14", "fMRI-vmPFC-FP", "FP14", "FP_14_fMRI_Anticipation_vmPFC"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-15",
            filename="Falsification/FP_15_AllenVisualCoding_Fatigue.py",
            title="Allen Visual Coding: Fatigue Threshold Dynamics",
            description="Tertiary Allen visual coding fatigue falsification stub linked to VP-17",
            priority_level="Tertiary",
            category="Falsification",
            implemented=False,
            aliases=["P15", "AllenVisualCoding-FP", "FatigueThreshold", "FP_15_AllenVisualCoding_Fatigue"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-ALL",
            filename="Falsification/FP_ALL_Aggregator.py",
            title="Framework-Level Falsification Aggregator",
            description="Aggregates falsification outputs across available FP protocols",
            priority_level="Aggregator",
            category="Falsification",
            aliases=["FP_ALL", "FP_ALL_Aggregator", "FalsificationAggregator", "all-aggregator"],
        )
    )

    # ------------------------------------------------------------------
    # Validation Protocols (VP_00 – VP_22 plus aggregators/internal checks)
    # ------------------------------------------------------------------
    _vp_entries = [
        (
            "VP-00",
            "VP_00_HEPProxyValidation.py",
            "HEP Proxy Validation (EP-0)",
            "Stub for HEP proxy validation; partial falsification coverage in FP-09",
            ["VP00", "HEPProxy", "EP-0", "VP_00_HEPProxyValidation"],
            False,
        ),
        (
            "VP-01",
            "VP_01_SyntheticEEGMLClassification.py",
            "Synthetic EEG ML Classification",
            "ML classification of synthetic EEG with precision-weighting",
            ["VP01", "SyntheticEEG", "MLClassification"],
            True,
        ),
        (
            "VP-02",
            "VP_02_BehavioralBayesianComparison.py",
            "Behavioral Bayesian Model Comparison",
            "Bayesian model comparison on behavioral data",
            ["VP02", "BehavioralBayes", "BayesianComparison"],
            True,
        ),
        (
            "VP-03",
            "VP_03_ActiveInferenceAgentSimulations.py",
            "Active Inference Agent Simulations (APGI-P02 linked)",
            "Agent-based active inference; linked to APGI-P02",
            ["VP03", "ActiveInference", "AgentSimulations"],
            True,
        ),
        (
            "VP-04",
            "VP_04_PhaseTransitionEpistemicLevel2.py",
            "Phase Transition Epistemic Level 2",
            "Phase transition dynamics at epistemic level 2",
            ["VP04", "PhaseTransitionEpistemic"],
        ),
        (
            "VP-05",
            "VP_05_EvolutionaryEmergence.py",
            "Evolutionary Emergence",
            "Evolutionary plausibility of APGI parameters",
            ["VP05", "EvolutionaryEmergence"],
        ),
        (
            "VP-06",
            "VP_06_LiquidNetworkInductiveBias.py",
            "Liquid Network Inductive Bias",
            "Inductive bias testing for LTC networks",
            ["VP06", "LiquidNetworkBias"],
        ),
        (
            "VP-07",
            "VP_07_TMSCausalInterventions.py",
            "TMS Causal Interventions (APGI-P05 linked)",
            "TMS/pharmacological intervention predictions; linked to APGI-P05",
            ["VP07", "TMS", "CausalInterventions"],
        ),
        (
            "VP-07a",
            "VP_07a_MathematicalConsistency.py",
            "Mathematical Consistency — SymPy Symbolic Verification",
            "Internal validation-side mathematical consistency check paired with FP-07",
            ["VP07a", "VP_07a_MathematicalConsistency", "MathConsistency-VP"],
        ),
        (
            "VP-08",
            "VP_08_PsychophysicalThresholdEstimation.py",
            "Psychophysical Threshold Estimation",
            "Threshold estimation from psychophysical data",
            ["VP08", "Psychophysical", "ThresholdEstimation"],
        ),
        (
            "VP-09",
            "VP_09_NeuralSignaturesEmpiricalPriority1.py",
            "Convergent Neural Signatures — Priority 1 (APGI-P01 linked)",
            "Neural signature convergence; linked to APGI-P01 EEG protocol",
            ["VP09", "NeuralSignatures", "Priority1", "HEP-VP"],
        ),
        (
            "VP-10",
            "VP_10_CausalManipulationsPriority2.py",
            "Causal Manipulations — Priority 2 (APGI-P05 linked)",
            "Causal manipulation validation; linked to APGI-P05 TMS",
            ["VP10", "CausalManipulations", "Priority2"],
        ),
        (
            "VP-11",
            "VP_11_MCMCCulturalNeurosciencePriority3.py",
            "MCMC Cultural Neuroscience — Priority 3",
            "MCMC-based cultural neuroscience testing",
            ["VP11", "MCMC-Cultural", "CulturalNeuroscience"],
        ),
        (
            "VP-12",
            "VP_12_ClinicalCrossSpeciesConvergence.py",
            "Clinical & Cross-Species Convergence (APGI-P07 linked)",
            "Clinical/cross-species convergence; linked to APGI-P07 DoC",
            ["VP12", "Clinical", "CrossSpeciesConvergence"],
        ),
        (
            "VP-13",
            "VP_13_EpistemicArchitecture.py",
            "Epistemic Architecture",
            "Epistemic architecture validation",
            ["VP13", "EpistemicArchitecture"],
        ),
        (
            "VP-14",
            "VP_14_FMRIAnticipationExperience.py",
            "fMRI Anticipation vs Experience (APGI-P03 linked)",
            "BOLD tracking of S(t)/M(t); linked to APGI-P03 fMRI",
            ["VP14", "fMRI-Anticipation"],
        ),
        (
            "VP-15",
            "VP_15_FMRIAnticipationVmPFC.py",
            "fMRI Anticipation vmPFC",
            "vmPFC-posterior insula connectivity in anticipation",
            ["VP15", "vmPFC-fMRI"],
        ),
        (
            "VP-16",
            "VP_16_MetabolicATPGroundTruth.py",
            "Metabolic ATP Ground Truth",
            "Metabolic cost validation against ATP measurements",
            ["VP16", "MetabolicATP"],
        ),
        (
            "VP-17",
            "VP_17_AllenVisualCodingFatigue.py",
            "Allen Visual Coding Fatigue",
            "Neural fatigue effects in Allen Brain visual coding data",
            ["VP17", "AllenVisualCoding"],
        ),
        (
            "VP-18",
            "VP_18_EEGMicrostateGFPP3b.py",
            "EEG Microstate GFP P3b",
            "EEG microstate and global field power P3b analysis",
            ["VP18", "EEG-Microstate", "GFP-P3b"],
        ),
        (
            "VP-19",
            "VP_19_InformationErasureMVPA.py",
            "Information Erasure MVPA",
            "MVPA-based information erasure testing",
            ["VP19", "InformationErasure", "MVPA"],
        ),
        (
            "VP-20",
            "VP_20_EmpiricalIEEG.py",
            "Empirical intracranial EEG (APGI-P06 linked)",
            "iEEG bistability and critical slowing; linked to APGI-P06",
            ["VP20", "iEEG", "EmpiricalIEEG"],
        ),
        (
            "VP-21",
            "VP_21_FreeEnergyPredictionError.py",
            "Free Energy Prediction Error",
            "Variational free energy and prediction error validation",
            ["VP21", "FreeEnergy", "PredictionError"],
        ),
        (
            "VP-22",
            "VP_22_FMRIAnticipationExperience.py",
            "Enhanced fMRI Anticipation vs Experience (APGI-P03 linked)",
            "Enhanced anticipation-versus-experience validation; linked to APGI-P03",
            ["VP22", "fMRI-Anticipation-Enhanced", "SomaticMarker-fMRI"],
        ),
        (
            "VP-ALL",
            "VP_ALL_Aggregator.py",
            "Validation Aggregator",
            "Aggregates validation outputs across available VP protocols",
            ["VP_ALL", "VP_ALL_Aggregator", "ValidationAggregator"],
        ),
    ]

    for entry in _vp_entries:
        canon_id, filename, title, description, aliases_raw, *rest = entry  # type: ignore[misc]
        canon_id = str(canon_id)
        filename = str(filename)
        title = str(title)
        description = str(description)
        aliases = list(aliases_raw) if isinstance(aliases_raw, list) else [str(aliases_raw)]
        implemented = bool(rest[0]) if rest else True
        PROTOCOL_REGISTRY.register_protocol(
            ProtocolInfo(
                canonical_id=canon_id,
                filename=f"Validation/{filename}",
                title=title,
                description=description,
                priority_level="Validation",
                category="Validation",
                implemented=implemented,
                aliases=aliases,
            )
        )

    # ------------------------------------------------------------------
    # Theory Modules (Theory/ folder)
    # ------------------------------------------------------------------
    _theory_entries = [
        (
            "T-BayesianEstimation",
            "APGI_Bayesian_Estimation_Framework.py",
            "Bayesian Parameter Estimation Framework",
            "Hierarchical Bayesian modeling, model comparison, parameter recovery",
            ["BayesianFramework", "APGI_Bayesian"],
        ),
        (
            "T-Benchmarking",
            "APGI_Computational_Benchmarking.py",
            "Computational Benchmarking",
            "Compare APGI/FEP/GNW/IIT computational performance",
            ["ComputationalBenchmark", "APGI_Benchmarking"],
        ),
        (
            "T-CrossSpeciesScaling",
            "APGI_Cross_Species_Scaling.py",
            "Cross-Species Scaling Laws",
            "Allostatic scaling laws across human, macaque, mouse, fish",
            ["CrossSpeciesTheory", "APGI_CrossSpecies"],
        ),
        (
            "T-CulturalNeuroscience",
            "APGI_Cultural_Neuroscience.py",
            "Cultural Neuroscience Module",
            "Linguistic and contemplative effects on APGI parameters",
            ["CulturalNeuroscienceTheory", "APGI_Cultural"],
        ),
        (
            "T-Entropy",
            "APGI_Entropy_Implementation.py",
            "Three-Level Entropy Framework",
            "Thermodynamic, Shannon, and variational entropy",
            ["EntropyTheory", "APGI_Entropy"],
        ),
        (
            "T-FalsificationFramework",
            "APGI_Falsification_Framework.py",
            "Popperian Falsification Framework",
            "Core falsification infrastructure used by FP scripts",
            ["FalsificationTheory", "APGI_Falsification"],
        ),
        (
            "T-FractalThreshold",
            "APGI_Fractal_Threshold_Dynamics.py",
            "Fractal Threshold Dynamics",
            "Fractal dynamics of ignition threshold θₜ",
            ["FractalThreshold", "APGI_Fractal"],
        ),
        (
            "T-HierarchicalTiling",
            "APGI_Hierarchical_Temporal_Tiling.py",
            "Hierarchical Temporal Tiling",
            "Temporal integration across hierarchical levels",
            ["TemporalTiling", "APGI_HierarchicalTiling"],
        ),
        (
            "T-BandwidthInfo",
            "APGI_Information_Theoretic_Bandwidth.py",
            "Information-Theoretic Bandwidth (~40 bits/s)",
            "Tier 2 derivation of conscious bandwidth from APGI parameters",
            ["BandwidthInfo", "APGI_Bandwidth"],
        ),
        (
            "T-BifurcationAnalysis",
            "APGI_LNN_Bifurcation_Analysis.py",
            "LNN Bifurcation Analysis (APGI-P06 linked)",
            "Bifurcation analysis of LTC dynamics; linked to APGI-P06",
            ["BifurcationAnalysis", "APGI_Bifurcation"],
        ),
        (
            "T-LiquidNetwork",
            "APGI_Liquid_Network_Implementation.py",
            "Liquid Time-Constant Network Implementation",
            "LTC neurons for APGI dynamics with volatility estimation",
            ["LiquidNetworkTheory", "APGI_LTC"],
        ),
        (
            "T-MultimodalClassifier",
            "APGI_Multimodal_Classifier.py",
            "Multimodal Classifier",
            "Multimodal stimulus classification with APGI",
            ["MultimodalClassifier", "APGI_Classifier"],
        ),
        (
            "T-MultimodalIntegration",
            "APGI_Multimodal_Integration.py",
            "Multimodal Integration",
            "Cross-modal integration with precision-weighting",
            ["MultimodalIntegration", "APGI_Integration"],
        ),
        (
            "T-Neuromodulation",
            "APGI_Neuromodulatory_Control_System.py",
            "Neuromodulatory Control System",
            "ACh, NE, DA, 5-HT effects on APGI parameters",
            ["Neuromodulation", "APGI_Neuromodulation"],
        ),
        (
            "T-OpenScience",
            "APGI_Open_Science_Framework.py",
            "Open Science Framework",
            "Reproducibility, pre-registration, and open data utilities",
            ["OpenScience", "APGI_OpenScience"],
        ),
        (
            "T-ParameterEstimation",
            "APGI_Parameter_Estimation.py",
            "Parameter Estimation",
            "Point estimation of APGI parameters from data",
            ["ParameterEstimationTheory", "APGI_ParamEst"],
        ),
        (
            "T-PsychologicalStates",
            "APGI_Psychological_States.py",
            "Psychological States (51-state space)",
            "51-state psychological state space inference",
            ["PsychStates", "APGI_PsychStates"],
        ),
        (
            "T-SomaticMarker",
            "APGI_Somatic_Marker_Identifiability.py",
            "Somatic Marker Identifiability (APGI-P02 linked)",
            "M̂(c,a) identifiability; validated by APGI-P02",
            ["SomaticMarkerIdentifiability", "APGI_SomaticMarker"],
        ),
        (
            "T-Thermodynamic",
            "APGI_Thermodynamic_Program_Aggregator.py",
            "Thermodynamic Program Aggregator (Tier 1 bridge)",
            "Tier 1 bridge: aggregates thermodynamic constraints",
            ["ThermodynamicAggregator", "APGI_Thermodynamic"],
        ),
        (
            "T-TuringMachine",
            "APGI_Turing_Machine.py",
            "APGI Turing Machine",
            "Computational completeness of APGI state transitions",
            ["TuringMachine", "APGI_Turing"],
        ),
    ]

    for canon_id, filename, title, description, aliases in _theory_entries:
        PROTOCOL_REGISTRY.register_protocol(
            ProtocolInfo(
                canonical_id=canon_id,
                filename=f"Theory/{filename}",
                title=title,
                description=description,
                priority_level="Theory",
                category="Theory",
                aliases=aliases,
            )
        )


# Initialize the registry
_initialize_registry()


def get_protocol_registry() -> Type[PROTOCOL_REGISTRY]:
    """Get the protocol registry instance."""
    return PROTOCOL_REGISTRY


# Convenience functions for external use
def resolve_protocol(protocol_id: str, base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Resolve a protocol ID to its file path.

    Args:
        protocol_id: Canonical protocol ID or alias
        base_dir: Base directory for falsification files (defaults to project root)

    Returns:
        Path to the protocol file, or None if not found
    """
    if base_dir is None:
        from pathlib import Path

        base_dir = Path(__file__).parent.parent

    return PROTOCOL_REGISTRY.resolve_protocol_file(protocol_id, base_dir)


def list_available_protocols(category: Optional[str] = None) -> List[str]:
    """
    List available protocol IDs.

    Args:
        category: Optional category filter

    Returns:
        List of protocol canonical IDs
    """
    protocols = PROTOCOL_REGISTRY.list_protocols(category)
    return [p.canonical_id for p in protocols]


def get_protocol_info(protocol_id: str) -> Optional[ProtocolInfo]:
    """
    Get detailed information about a protocol.

    Args:
        protocol_id: Canonical protocol ID or alias

    Returns:
        ProtocolInfo object, or None if not found
    """
    return PROTOCOL_REGISTRY.get_protocol(protocol_id)
