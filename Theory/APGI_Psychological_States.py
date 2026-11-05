"""

=============================================================================
APGI Psychological State Parameter Library
=============================================================================

Complete parameter mappings for 54 psychological states based on the
Allostatic Precision-Gated Ignition (APGI) framework.

Each state is defined with:
- Pi_e: Exteroceptive precision  [0.1, 15]
- Pi_i_baseline: Baseline interoceptive precision  [0.1, 15]
- Pi_i_eff: Effective interoceptive precision (after somatic modulation)
- theta_t: Ignition threshold (z-score units; negative = lowered, positive = elevated)
- S_t: Accumulated surprise signal (computed: _e|z_e| + _i_eff|z_i|)
- M_ca: Somatic marker value  [-2, +2]
- beta: Somatic influence gain (_som)  [0.3, 0.8]
- z_e: Exteroceptive prediction error magnitude
- z_i: Interoceptive prediction error magnitude

Usage:
    from apgi_state_library import PSYCHOLOGICAL_STATES, get_state, StateCategory

    # Get a specific state
    fear_params = get_state('fear')
    print(f"Fear threshold: {fear_params.theta_t}")

    # Get all states in a category
    aversive_states = get_states_by_category(StateCategory.AVERSIVE_AFFECTIVE)

    # Compare two states

TIER DESIGNATION: All outputs are Tier 3 (algorithmic/mathematical).
Bridge to Tier 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Tier 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If the psychological state parameter mappings fail to produce consistent
state-to-state transitions (transition prediction accuracy < 70%) or if the
parameter ranges do not align with empirical findings from clinical studies
(deviation > 30% from published values), then the APGI psychological state
parameterization claim is falsified. This would indicate that the proposed
parameter mappings do not accurately capture psychological state dynamics.

"""

# =============================================================================

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Matplotlib imports for PNG visualization
try:
    import matplotlib

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class APGIParameters:
    """APGI parameter set with proper type safety"""

    Pi_e: float  # Exteroceptive precision  [0.1, 10]
    Pi_i_baseline: float  # Baseline interoceptive precision  [0.1, 10]
    Pi_i_eff: float  # Effective interoceptive precision (modulated)
    theta_t: float  # Ignition threshold (z-score)
    S_t: float  # Accumulated surprise signal
    M_ca: float  # Somatic marker value  [-2, +2]
    beta: float  # Somatic influence gain (_som)  [0.3, 0.8]
    z_e: float  # Exteroceptive z-score
    z_i: float  # Interoceptive z-score

    def __post_init__(self):
        """Validate parameters are within physiological bounds"""
        if not 0.1 <= self.Pi_e <= 15.0:
            raise ValueError(f"Pi_e must be in [0.1, 15], got {self.Pi_e}")
        if not 0.1 <= self.Pi_i_baseline <= 15.0:
            raise ValueError(f"Pi_i_baseline must be in [0.1, 15], got {self.Pi_i_baseline}")
        if not 0.1 <= self.Pi_i_eff <= 15.0:
            raise ValueError(f"Pi_i_eff must be in [0.1, 15], got {self.Pi_i_eff}")
        if not -2.0 <= self.M_ca <= 2.0:
            raise ValueError(f"M_ca must be in [-2, 2], got {self.M_ca}")
        if not (0.3 <= self.beta <= 0.8):
            raise ValueError(f"_som={self.beta} outside valid range [0.3, 0.8]")

    def compute_ignition_probability(self) -> float:
        """Compute P(ignite) = (S_t - _t) with overflow protection"""
        z = self.S_t - self.theta_t
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            z_exp = np.exp(z)
            return z_exp / (1.0 + z_exp)

    def verify_S_t(self) -> bool:
        """Verify S_t matches the formula: S_t = _e*|z_e| + _i_eff*|z_i|"""
        computed = self.Pi_e * abs(self.z_e) + self.Pi_i_eff * abs(self.z_i)
        return bool(np.isclose(self.S_t, computed, rtol=0.001))

    def verify_Pi_i_eff(self) -> bool:
        """Verify _i_eff matches the formula: _i_eff = _i_baseline * exp(*M)"""
        computed = self.Pi_i_baseline * np.exp(self.beta * self.M_ca)
        computed = np.clip(computed, 0.1, 15.0)
        return bool(np.isclose(self.Pi_i_eff, computed, rtol=0.001))


@dataclass
class PsychologicalState:
    """Extended state representation with metadata"""

    name: str
    parameters: APGIParameters
    category: str
    description: str
    phenomenology: List[str]
    distinguishing_features: Dict[str, str]
    pathological_variant: Optional[str] = None
    temporal_dynamics: Optional[str] = None


class StateCategory(Enum):
    """Categories of psychological states"""

    OPTIMAL_FUNCTIONING = auto()
    POSITIVE_AFFECTIVE = auto()
    COGNITIVE_ATTENTIONAL = auto()
    AVERSIVE_AFFECTIVE = auto()
    PATHOLOGICAL_EXTREME = auto()
    ALTERED_BOUNDARY = auto()
    TRANSITIONAL_CONTEXTUAL = auto()
    UNELABORATED = auto()


def create_apgi_params(
    Pi_e: float,
    Pi_i_baseline: float,
    M_ca: float,
    beta: float,
    z_e: float,
    z_i: float,
    theta_t: float,
) -> APGIParameters:
    """
    Factory function that computes derived parameters automatically.

    Computes:
    - Pi_i_eff = Pi_i_baseline * exp(*M_ca)
    - S_t = _e*|z_e| + _i_eff*|z_i|
    """
    # Compute effective interoceptive precision with somatic modulation (exponential form)
    Pi_i_eff = Pi_i_baseline * np.exp(beta * M_ca)
    Pi_i_eff = np.clip(Pi_i_eff, 0.1, 15.0)

    # Compute accumulated surprise
    S_t = Pi_e * abs(z_e) + Pi_i_eff * abs(z_i)

    return APGIParameters(
        Pi_e=Pi_e,
        Pi_i_baseline=Pi_i_baseline,
        Pi_i_eff=Pi_i_eff,
        theta_t=theta_t,
        S_t=S_t,
        M_ca=M_ca,
        beta=beta,
        z_e=z_e,
        z_i=z_i,
    )


# =============================================================================
# CATEGORY 1: OPTIMAL FUNCTIONING STATES (States 1-4)
# =============================================================================

STATE_01_FLOW = create_apgi_params(
    Pi_e=6.5,  # High precision on task-relevant content
    Pi_i_baseline=1.5,  # Low baseline interoceptive (body recedes)
    M_ca=0.3,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.4,  # Low prediction error (mastery)
    z_i=0.2,  # Minimal interoceptive error
    theta_t=1.8,  # Elevated threshold (effortless filtering)
)

STATE_02_FOCUS = create_apgi_params(
    Pi_e=8.0,  # Very high precision on target
    Pi_i_baseline=1.2,  # Low interoceptive baseline
    M_ca=0.25,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.8,  # Moderate prediction error (challenge)
    z_i=0.3,  # Low interoceptive error
    theta_t=-0.5,  # Actively suppressed threshold (effortful)
)

STATE_03_SERENITY = create_apgi_params(
    Pi_e=1.5,  # Low, broadly distributed precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.7,  # Moderate somatic bias (peaceful embodiment)
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Minimal prediction error
    z_i=0.3,  # Minimal interoceptive error
    theta_t=1.5,  # Elevated threshold (filters trivial content)
)

STATE_04_MINDFULNESS = create_apgi_params(
    Pi_e=3.0,  # Moderate, present-moment precision
    Pi_i_baseline=3.5,  # Elevated interoceptive baseline
    M_ca=0.9,  # Moderate-high somatic bias (embodied presence)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.6,  # Moderate errors (observed without reactivity)
    z_i=0.5,  # Moderate interoceptive errors
    theta_t=0.0,  # Neutral threshold (flexible, non-reactive)
)


# =============================================================================
# CATEGORY 2: POSITIVE AFFECTIVE STATES (States 5-11)
# =============================================================================

STATE_05_AMUSEMENT = create_apgi_params(
    Pi_e=4.0,  # Moderate-high precision on incongruity
    Pi_i_baseline=1.0,  # Low interoceptive (cognitive, not embodied)
    M_ca=-0.1,  # Slightly negative somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=1.2,  # Moderate PE (benign incongruity)
    z_i=0.2,  # Minimal interoceptive error
    theta_t=-0.3,  # Slightly lowered for incongruity resolution
)

STATE_06_JOY = create_apgi_params(
    Pi_e=5.0,  # High precision on reward-relevant content
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Moderate-high somatic bias (warmth, expansion)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=1.0,  # Positive prediction error (better than expected)
    z_i=0.7,  # Moderate interoceptive involvement
    theta_t=-0.8,  # Lowered for positive content
)

STATE_07_PRIDE = create_apgi_params(
    Pi_e=4.5,  # High precision on self-model
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.1,  # High somatic bias (postural expansion)
    beta=0.6,  # Elevated somatic gain
    z_e=1.2,  # Positive self-referential PE
    z_i=0.9,  # Elevated interoceptive (embodied pride)
    theta_t=-0.6,  # Lowered for self-enhancing content
)

STATE_08_ROMANTIC_LOVE_EARLY = create_apgi_params(
    Pi_e=7.5,  # Very high precision on partner
    Pi_i_baseline=4.0,  # High interoceptive baseline
    M_ca=1.8,  # Very high somatic bias
    beta=0.7,  # High somatic gain
    z_e=1.5,  # High PE (partner constantly surprising)
    z_i=1.3,  # High interoceptive involvement
    theta_t=-1.5,  # Very low threshold for partner cues
)

STATE_08B_ROMANTIC_LOVE_SUSTAINED = create_apgi_params(
    Pi_e=5.0,  # Moderate-high precision on partner
    Pi_i_baseline=3.0,  # Moderate interoceptive baseline
    M_ca=1.2,  # High but not extreme somatic bias
    beta=0.6,  # Moderate-high somatic gain
    z_e=0.5,  # Low PE (partner predictable, integrated)
    z_i=0.6,  # Moderate interoceptive involvement
    theta_t=-0.8,  # Lowered but not extreme
)

STATE_09_GRATITUDE = create_apgi_params(
    Pi_e=4.0,  # Moderate-high precision on affiliative confirmation
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.3,  # Low PE (prediction confirmed)
    z_i=0.5,  # Moderate interoceptive (warmth)
    theta_t=-0.4,  # Slightly lowered
)

STATE_10_HOPE = create_apgi_params(
    Pi_e=5.0,  # High precision on specific future outcomes
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.6,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.9,  # Moderate PE (future uncertainty)
    z_i=0.4,  # Low interoceptive error
    theta_t=-0.7,  # Lowered for desired futures
)

STATE_11_OPTIMISM = create_apgi_params(
    Pi_e=3.0,  # Diffuse moderate precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.4,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.4,  # Low PE (positive priors)
    z_i=0.3,  # Low interoceptive error
    theta_t=-0.5,  # Globally slightly lowered
)


# =============================================================================
# CATEGORY 3: COGNITIVE AND ATTENTIONAL STATES (States 12-19)
# =============================================================================

STATE_12_CURIOSITY = create_apgi_params(
    Pi_e=6.0,  # High precision on reducible uncertainty
    Pi_i_baseline=1.0,  # Low interoceptive baseline
    M_ca=-0.2,  # Slightly negative somatic bias (cognitive)
    beta=0.45,  # Slightly low somatic gain
    z_e=1.4,  # Moderate-high PE (novelty)
    z_i=0.2,  # Minimal interoceptive error
    theta_t=-0.9,  # Lowered for novel information
)

STATE_13_BOREDOM = create_apgi_params(
    Pi_e=0.8,  # Collapsed precision on current task
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=-0.3,  # Slightly negative somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.1,  # Very low PE (no surprise)
    z_i=0.2,  # Low interoceptive error
    theta_t=-1.0,  # Lowered for novel inputs (seeking)
)

STATE_14_CREATIVITY = create_apgi_params(
    Pi_e=4.0,  # Moderate precision (loosened high-level)
    Pi_i_baseline=1.0,  # Low interoceptive baseline
    M_ca=-0.3,  # Slightly negative somatic bias (cognitive)
    beta=0.45,  # Slightly low somatic gain
    z_e=1.2,  # Variable PE (tolerated)
    z_i=0.2,  # Minimal interoceptive error
    theta_t=-1.2,  # Lowered for unconventional content
)

STATE_15_INSPIRATION = create_apgi_params(
    Pi_e=8.5,  # Very high precision spike on emergent pattern
    Pi_i_baseline=1.5,  # Low-moderate interoceptive baseline
    M_ca=0.4,  # Low-moderate somatic bias (aha feeling)
    beta=0.5,  # Neutral somatic gain
    z_e=2.0,  # High PE (unexpected convergence)
    z_i=0.4,  # Low interoceptive error
    theta_t=-2.0,  # Acutely suppressed threshold
)

STATE_16_HYPERFOCUS = create_apgi_params(
    Pi_e=9.5,  # Extreme precision on single target
    Pi_i_baseline=0.5,  # Very low interoceptive baseline
    M_ca=-0.8,  # Negative somatic bias (body disappears)
    beta=0.4,  # Low somatic gain
    z_e=0.6,  # Low PE within domain
    z_i=0.1,  # Minimal interoceptive (ignored)
    theta_t=2.5,  # Very elevated for all else
)

STATE_17_FATIGUE = create_apgi_params(
    Pi_e=1.5,  # Globally reduced precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.4,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.3,  # Low PE
    z_i=0.4,  # Moderate interoceptive (tiredness signals)
    theta_t=1.8,  # Elevated (metabolic conservation)
)

STATE_18_DECISION_FATIGUE = create_apgi_params(
    Pi_e=2.5,  # Undifferentiated precision across options
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=0.3,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.8,  # Moderate PE per option
    z_i=0.3,  # Low interoceptive error
    theta_t=1.5,  # Elevated (metabolic conservation)
)

STATE_19_MIND_WANDERING = create_apgi_params(
    Pi_e=0.8,  # Collapsed external precision
    Pi_i_baseline=3.5,  # Elevated interoceptive/DMN baseline
    M_ca=0.6,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.2,  # Low external PE (ignored)
    z_i=0.9,  # Moderate internal PE (narrative unfolds)
    theta_t=1.5,  # High external / Low internal (asymmetric)
)


# =============================================================================
# CATEGORY 4: AVERSIVE AFFECTIVE STATES (States 20-26)
# =============================================================================

STATE_20_FEAR = create_apgi_params(
    Pi_e=8.0,  # Very high precision on threat
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.9,  # Near-maximum somatic bias
    beta=0.75,  # High somatic gain
    z_e=2.5,  # Very high PE (threat detection)
    z_i=2.0,  # High interoceptive PE (body mobilization)
    theta_t=-2.5,  # Strongly suppressed threshold
)

STATE_21_ANXIETY = create_apgi_params(
    Pi_e=6.5,  # High precision on potential threats
    Pi_i_baseline=3.5,  # Elevated interoceptive baseline
    M_ca=1.5,  # High somatic bias
    beta=0.65,  # Elevated somatic gain
    z_e=1.5,  # Moderate-high PE
    z_i=1.3,  # Elevated interoceptive PE
    theta_t=-1.5,  # Lowered for threat-relevant content
)

STATE_22_ANGER = create_apgi_params(
    Pi_e=7.5,  # Very high precision on obstacle/transgressor
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.5,  # High somatic bias (mobilization)
    beta=0.65,  # Elevated somatic gain
    z_e=2.0,  # High PE (goal blockage)
    z_i=1.4,  # Elevated interoceptive PE
    theta_t=-1.2,  # Lowered for target; elevated for mitigation
)

STATE_23_GUILT = create_apgi_params(
    Pi_e=5.0,  # High precision on self-model violations
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=1.3,  # Moderate PE (action-specific)
    z_i=0.9,  # Moderate interoceptive PE
    theta_t=-0.8,  # Lowered for rumination
)

STATE_24_SHAME = create_apgi_params(
    Pi_e=7.0,  # Very high precision on social prediction errors
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.3,  # High somatic bias
    beta=0.6,  # Elevated somatic gain
    z_e=1.8,  # High PE (global self-threat)
    z_i=1.2,  # Elevated interoceptive PE
    theta_t=-1.5,  # Strongly lowered (broad ignition)
)

STATE_25_LONELINESS = create_apgi_params(
    Pi_e=5.5,  # High precision on social absence
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=1.4,  # Moderate-high PE (absence signals)
    z_i=0.9,  # Moderate interoceptive PE
    theta_t=-1.0,  # Lowered for social threat
)

STATE_26_OVERWHELM = create_apgi_params(
    Pi_e=3.0,  # Fragmented, unstable precision
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.2,  # High somatic bias (freeze)
    beta=0.6,  # Elevated somatic gain
    z_e=2.8,  # Very high concurrent PE
    z_i=1.5,  # High interoceptive PE
    theta_t=0.0,  # Chaotic (multiple crossings)
)


# =============================================================================
# CATEGORY 5: PATHOLOGICAL AND EXTREME STATES (States 27-33)
# =============================================================================

STATE_27_DEPRESSION = create_apgi_params(
    Pi_e=2.0,  # Low precision on positive; high on negative
    Pi_i_baseline=1.5,  # Dysregulated interoceptive baseline
    M_ca=0.3,  # Dysregulated somatic (oscillating)
    beta=0.5,  # Neutral somatic gain
    z_e=0.4,  # Blunted PE
    z_i=0.8,  # Moderate interoceptive (allostatic disruption)
    theta_t=1.5,  # Elevated for positive; lowered for negative
)

STATE_28_LEARNED_HELPLESSNESS = create_apgi_params(
    Pi_e=1.5,  # Collapsed precision on controllability
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Extinguished PE (actions don't matter)
    z_i=0.4,  # Low interoceptive PE
    theta_t=2.0,  # Elevated for action initiation
)

STATE_29_PESSIMISTIC_DEPRESSION = create_apgi_params(
    Pi_e=2.5,  # High precision on negative priors
    Pi_i_baseline=2.0,  # Moderate-dysregulated interoceptive
    M_ca=0.7,  # Moderate somatic bias (heaviness)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.3,  # Low PE (positive surprise extinguished)
    z_i=0.6,  # Moderate interoceptive PE
    theta_t=1.8,  # Elevated for positive evidence
)

STATE_30_PANIC = create_apgi_params(
    Pi_e=4.0,  # Moderate external (overwhelmed by internal)
    Pi_i_baseline=5.0,  # Very high interoceptive baseline
    M_ca=2.0,  # Maximum somatic bias
    beta=0.8,  # Maximum somatic gain
    z_e=1.5,  # Moderate external PE
    z_i=3.0,  # Extreme interoceptive PE
    theta_t=-3.0,  # Catastrophically suppressed threshold
)

STATE_31_DISSOCIATION = create_apgi_params(
    Pi_e=2.0,  # Reduced external precision
    Pi_i_baseline=0.5,  # Collapsed interoceptive baseline
    M_ca=-1.5,  # Strongly negative somatic bias (disconnection)
    beta=0.35,  # Low somatic gain
    z_e=0.8,  # Moderate external PE
    z_i=0.1,  # Minimal interoceptive (disconnected)
    theta_t=2.0,  # Elevated for embodiment
)

STATE_32_DEPERSONALIZATION = create_apgi_params(
    Pi_e=3.0,  # Moderate external precision
    Pi_i_baseline=0.8,  # Low interoceptive baseline
    M_ca=-1.2,  # Negative somatic bias (self disconnected)
    beta=0.4,  # Low somatic gain
    z_e=1.0,  # Moderate PE (self-mismatch)
    z_i=0.5,  # Low interoceptive (foreign)
    theta_t=1.5,  # Elevated for self-as-subject
)

STATE_33_DEREALIZATION = create_apgi_params(
    Pi_e=1.5,  # Collapsed external precision
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=-0.8,  # Negative somatic bias
    beta=0.45,  # Slightly low somatic gain
    z_e=1.2,  # Moderate PE (world-mismatch)
    z_i=0.4,  # Low interoceptive PE
    theta_t=1.8,  # Elevated for perceptual integration
)


# =============================================================================
# CATEGORY 6: ALTERED AND BOUNDARY STATES (States 34-39)
# =============================================================================

STATE_34_AWE = create_apgi_params(
    Pi_e=3.5,  # Reduced precision on high-level priors
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Variable, moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=2.8,  # Very high hierarchical PE
    z_i=0.7,  # Moderate interoceptive PE
    theta_t=-1.5,  # Briefly suppressed threshold
)

STATE_35_TRANCE = create_apgi_params(
    Pi_e=1.0,  # Suppressed external precision
    Pi_i_baseline=4.0,  # Very high internal precision
    M_ca=0.4,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Low external PE (ignored)
    z_i=0.6,  # Moderate internal PE
    theta_t=2.0,  # High external / Low internal
)

STATE_36_MEDITATION_FOCUSED = create_apgi_params(
    Pi_e=7.0,  # Very high precision on meditation object
    Pi_i_baseline=3.5,  # Elevated interoceptive (breath-focused)
    M_ca=1.0,  # Moderate-high somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.5,  # Low PE (maintained attention)
    z_i=0.6,  # Moderate interoceptive PE
    theta_t=1.5,  # Elevated for distractions
)

STATE_36B_MEDITATION_OPEN = create_apgi_params(
    Pi_e=3.0,  # Broadly distributed precision
    Pi_i_baseline=3.0,  # Moderate interoceptive baseline
    M_ca=0.7,  # Moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.8,  # Variable PE (observed)
    z_i=0.6,  # Moderate interoceptive PE
    theta_t=0.0,  # Neutral (flexible)
)

STATE_36C_MEDITATION_NONDUAL = create_apgi_params(
    Pi_e=2.0,  # Collapsed content precision
    Pi_i_baseline=1.5,  # Reduced interoceptive baseline
    M_ca=0.5,  # Variable somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Very low PE (no self-reference)
    z_i=0.2,  # Very low interoceptive PE
    theta_t=2.0,  # Elevated for conceptual; lowered for immediate
)

STATE_37_HYPNOSIS = create_apgi_params(
    Pi_e=2.0,  # Suppressed reality-testing precision
    Pi_i_baseline=3.5,  # Elevated for suggested content
    M_ca=0.6,  # Variable (suggestion-dependent)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.3,  # Low PE (suggestions = "true")
    z_i=0.8,  # Moderate interoceptive PE
    theta_t=-1.5,  # Very low for suggestions
)

STATE_38_HYPNAGOGIA = create_apgi_params(
    Pi_e=2.5,  # Unstable external precision
    Pi_i_baseline=4.0,  # Elevated memory-trace precision
    M_ca=0.7,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.6,  # Moderate PE (blended)
    z_i=1.0,  # Elevated interoceptive PE
    theta_t=0.5,  # Unstable (transitional)
)

STATE_39_DEJA_VU = create_apgi_params(
    Pi_e=4.5,  # Anomalously high familiarity precision
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=0.2,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.4,  # Low PE (false confirmation)
    z_i=0.2,  # Low interoceptive PE
    theta_t=-0.8,  # Premature threshold crossing
)


# =============================================================================
# CATEGORY 7: TRANSITIONAL/CONTEXTUAL STATES (States 40-46)
# =============================================================================

STATE_40_MORNING_FLOW = create_apgi_params(
    Pi_e=5.5,  # High precision on routine tasks
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.3,  # Low PE (practiced routines)
    z_i=0.3,  # Low interoceptive PE
    theta_t=1.2,  # Elevated for non-routine
)

STATE_41_EVENING_FATIGUE = create_apgi_params(
    Pi_e=1.2,  # Globally reduced precision
    Pi_i_baseline=3.0,  # Elevated interoceptive (tiredness signals)
    M_ca=1.0,  # Moderate-high somatic bias (heaviness)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.2,  # Low external PE
    z_i=0.7,  # Moderate interoceptive PE
    theta_t=2.2,  # Elevated (metabolic conservation)
)

STATE_42_CREATIVE_INSPIRATION = create_apgi_params(
    Pi_e=8.0,  # Very high precision on novel synthesis
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=0.3,  # Low somatic bias (mild aha)
    beta=0.5,  # Neutral somatic gain
    z_e=2.2,  # High PE (unexpected connection)
    z_i=0.3,  # Low interoceptive PE
    theta_t=-1.8,  # Acutely suppressed threshold
)

STATE_43_ANXIOUS_RUMINATION = create_apgi_params(
    Pi_e=6.0,  # Very high precision on threat possibilities
    Pi_i_baseline=3.5,  # Elevated interoceptive baseline
    M_ca=1.4,  # High somatic bias (sustained tension)
    beta=0.65,  # Elevated somatic gain
    z_e=1.6,  # Moderate-high PE (threat scenarios)
    z_i=1.2,  # Elevated interoceptive PE
    theta_t=-1.2,  # Lowered for threat; elevated for reassurance
)

STATE_44_CALM = create_apgi_params(
    Pi_e=1.8,  # Low, broadly distributed precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Minimal PE
    z_i=0.3,  # Minimal interoceptive PE
    theta_t=1.2,  # Moderately elevated
)

STATE_45_PRODUCTIVE_FOCUS = create_apgi_params(
    Pi_e=7.0,  # High precision on task
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=0.3,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.7,  # Low-moderate PE (manageable challenge)
    z_i=0.3,  # Low interoceptive PE
    theta_t=-0.3,  # Slightly lowered for task
)

STATE_46_SECOND_WIND = create_apgi_params(
    Pi_e=5.5,  # Restored precision
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.9,  # Moderate PE (renewed engagement)
    z_i=0.5,  # Moderate interoceptive PE
    theta_t=-0.8,  # Lowered (catecholamine release)
)


# =============================================================================
# CATEGORY 8: PREVIOUSLY UNELABORATED STATES (States 47-51)
# =============================================================================

STATE_47_HYPERVIGILANCE = create_apgi_params(
    Pi_e=8.5,  # Very high threat-scanning precision
    Pi_i_baseline=4.0,  # Elevated interoceptive baseline
    M_ca=1.7,  # Very high somatic bias
    beta=0.7,  # High somatic gain
    z_e=1.8,  # Moderate-high PE (scanning)
    z_i=1.5,  # Elevated interoceptive PE
    theta_t=-2.0,  # Very low for threat; high for safety
)

STATE_48_SADNESS = create_apgi_params(
    Pi_e=4.5,  # High precision on loss-relevant content
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.9,  # Moderate somatic bias (heaviness)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=1.2,  # Moderate PE (loss detection)
    z_i=0.8,  # Moderate interoceptive PE
    theta_t=-0.6,  # Lowered for loss reminders
)

STATE_49_CHOICE_PARALYSIS = create_apgi_params(
    Pi_e=2.5,  # Equal, undifferentiated precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.9,  # Moderate PE per option
    z_i=0.5,  # Moderate interoceptive PE
    theta_t=1.5,  # Elevated (metabolic conservation)
)

STATE_50_MENTAL_PARALYSIS = create_apgi_params(
    Pi_e=2.0,  # Fragmented, unstable precision
    Pi_i_baseline=3.5,  # Elevated interoceptive baseline
    M_ca=1.3,  # High somatic bias (freeze)
    beta=0.65,  # Elevated somatic gain
    z_e=3.0,  # Very high concurrent PE
    z_i=1.8,  # High interoceptive PE
    theta_t=0.5,  # Chaotic (multiple crossings)
)

STATE_51_CURIOUS_EXPLORATION = create_apgi_params(
    Pi_e=6.5,  # High precision on novel stimuli
    Pi_i_baseline=1.0,  # Low interoceptive baseline
    M_ca=-0.1,  # Slightly negative somatic bias (exteroceptive)
    beta=0.45,  # Slightly low somatic gain
    z_e=1.6,  # Moderate-high PE (novelty)
    z_i=0.2,  # Minimal interoceptive PE
    theta_t=-1.0,  # Lowered for novelty
)


# =============================================================================
# MASTER DICTIONARY OF ALL STATES
# =============================================================================

PSYCHOLOGICAL_STATES: Dict[str, APGIParameters] = {
    # Optimal Functioning States (1-4)
    "flow": STATE_01_FLOW,
    "focus": STATE_02_FOCUS,
    "serenity": STATE_03_SERENITY,
    "mindfulness": STATE_04_MINDFULNESS,
    # Positive Affective States (5-11)
    "amusement": STATE_05_AMUSEMENT,
    "joy": STATE_06_JOY,
    "pride": STATE_07_PRIDE,
    "romantic_love_early": STATE_08_ROMANTIC_LOVE_EARLY,
    "romantic_love_sustained": STATE_08B_ROMANTIC_LOVE_SUSTAINED,
    "gratitude": STATE_09_GRATITUDE,
    "hope": STATE_10_HOPE,
    "optimism": STATE_11_OPTIMISM,
    # Cognitive and Attentional States (12-19)
    "curiosity": STATE_12_CURIOSITY,
    "boredom": STATE_13_BOREDOM,
    "creativity": STATE_14_CREATIVITY,
    "inspiration": STATE_15_INSPIRATION,
    "hyperfocus": STATE_16_HYPERFOCUS,
    "fatigue": STATE_17_FATIGUE,
    "decision_fatigue": STATE_18_DECISION_FATIGUE,
    "mind_wandering": STATE_19_MIND_WANDERING,
    # Aversive Affective States (20-26)
    "fear": STATE_20_FEAR,
    "anxiety": STATE_21_ANXIETY,
    "GAD": STATE_21_ANXIETY,  # Generalized Anxiety Disorder
    "anger": STATE_22_ANGER,
    "guilt": STATE_23_GUILT,
    "shame": STATE_24_SHAME,
    "loneliness": STATE_25_LONELINESS,
    "overwhelm": STATE_26_OVERWHELM,
    # Pathological and Extreme States (27-33)
    "depression": STATE_27_DEPRESSION,
    "MDD": STATE_27_DEPRESSION,  # Major Depressive Disorder
    "learned_helplessness": STATE_28_LEARNED_HELPLESSNESS,
    "pessimistic_depression": STATE_29_PESSIMISTIC_DEPRESSION,
    "panic": STATE_30_PANIC,
    "PTSD": STATE_30_PANIC,  # Post-Traumatic Stress Disorder (panic attacks)
    "dissociation": STATE_31_DISSOCIATION,
    "PTSD_dissociation": STATE_31_DISSOCIATION,  # PTSD with dissociation
    "depersonalization": STATE_32_DEPERSONALIZATION,
    "PTSD_depersonalization": STATE_32_DEPERSONALIZATION,  # PTSD with depersonalization
    "derealization": STATE_33_DEREALIZATION,
    "psychosis": STATE_33_DEREALIZATION,  # Psychosis (extreme reality distortion)
    # Altered and Boundary States (34-39)
    "awe": STATE_34_AWE,
    "trance": STATE_35_TRANCE,
    "meditation_focused": STATE_36_MEDITATION_FOCUSED,
    "meditation_open": STATE_36B_MEDITATION_OPEN,
    "meditation_nondual": STATE_36C_MEDITATION_NONDUAL,
    "hypnosis": STATE_37_HYPNOSIS,
    "hypnagogia": STATE_38_HYPNAGOGIA,
    "deja_vu": STATE_39_DEJA_VU,
    # Transitional/Contextual States (40-46)
    "morning_flow": STATE_40_MORNING_FLOW,
    "evening_fatigue": STATE_41_EVENING_FATIGUE,
    "creative_inspiration": STATE_42_CREATIVE_INSPIRATION,
    "anxious_rumination": STATE_43_ANXIOUS_RUMINATION,
    "calm": STATE_44_CALM,
    "productive_focus": STATE_45_PRODUCTIVE_FOCUS,
    "second_wind": STATE_46_SECOND_WIND,
    # Previously Unelaborated States (47-51)
    "hypervigilance": STATE_47_HYPERVIGILANCE,
    "sadness": STATE_48_SADNESS,
    "choice_paralysis": STATE_49_CHOICE_PARALYSIS,
    "mental_paralysis": STATE_50_MENTAL_PARALYSIS,
    "curious_exploration": STATE_51_CURIOUS_EXPLORATION,
}


# State category mapping
STATE_CATEGORIES: Dict[str, StateCategory] = {
    "flow": StateCategory.OPTIMAL_FUNCTIONING,
    "focus": StateCategory.OPTIMAL_FUNCTIONING,
    "serenity": StateCategory.OPTIMAL_FUNCTIONING,
    "mindfulness": StateCategory.OPTIMAL_FUNCTIONING,
    "amusement": StateCategory.POSITIVE_AFFECTIVE,
    "joy": StateCategory.POSITIVE_AFFECTIVE,
    "pride": StateCategory.POSITIVE_AFFECTIVE,
    "romantic_love_early": StateCategory.POSITIVE_AFFECTIVE,
    "romantic_love_sustained": StateCategory.POSITIVE_AFFECTIVE,
    "gratitude": StateCategory.POSITIVE_AFFECTIVE,
    "hope": StateCategory.POSITIVE_AFFECTIVE,
    "optimism": StateCategory.POSITIVE_AFFECTIVE,
    "curiosity": StateCategory.COGNITIVE_ATTENTIONAL,
    "boredom": StateCategory.COGNITIVE_ATTENTIONAL,
    "creativity": StateCategory.COGNITIVE_ATTENTIONAL,
    "inspiration": StateCategory.COGNITIVE_ATTENTIONAL,
    "hyperfocus": StateCategory.COGNITIVE_ATTENTIONAL,
    "fatigue": StateCategory.COGNITIVE_ATTENTIONAL,
    "decision_fatigue": StateCategory.COGNITIVE_ATTENTIONAL,
    "mind_wandering": StateCategory.COGNITIVE_ATTENTIONAL,
    "fear": StateCategory.AVERSIVE_AFFECTIVE,
    "anxiety": StateCategory.AVERSIVE_AFFECTIVE,
    "GAD": StateCategory.AVERSIVE_AFFECTIVE,  # Generalized Anxiety Disorder
    "anger": StateCategory.AVERSIVE_AFFECTIVE,
    "guilt": StateCategory.AVERSIVE_AFFECTIVE,
    "shame": StateCategory.AVERSIVE_AFFECTIVE,
    "loneliness": StateCategory.AVERSIVE_AFFECTIVE,
    "overwhelm": StateCategory.AVERSIVE_AFFECTIVE,
    "depression": StateCategory.PATHOLOGICAL_EXTREME,
    "MDD": StateCategory.PATHOLOGICAL_EXTREME,  # Major Depressive Disorder
    "learned_helplessness": StateCategory.PATHOLOGICAL_EXTREME,
    "pessimistic_depression": StateCategory.PATHOLOGICAL_EXTREME,
    "panic": StateCategory.PATHOLOGICAL_EXTREME,
    "PTSD": StateCategory.PATHOLOGICAL_EXTREME,  # Post-Traumatic Stress Disorder (panic attacks)
    "dissociation": StateCategory.PATHOLOGICAL_EXTREME,
    "PTSD_dissociation": StateCategory.PATHOLOGICAL_EXTREME,  # PTSD with dissociation
    "depersonalization": StateCategory.PATHOLOGICAL_EXTREME,
    "PTSD_depersonalization": StateCategory.PATHOLOGICAL_EXTREME,  # PTSD with depersonalization
    "derealization": StateCategory.PATHOLOGICAL_EXTREME,
    "psychosis": StateCategory.PATHOLOGICAL_EXTREME,  # Psychosis (extreme reality distortion)
    "awe": StateCategory.ALTERED_BOUNDARY,
    "trance": StateCategory.ALTERED_BOUNDARY,
    "meditation_focused": StateCategory.ALTERED_BOUNDARY,
    "meditation_open": StateCategory.ALTERED_BOUNDARY,
    "meditation_nondual": StateCategory.ALTERED_BOUNDARY,
    "hypnosis": StateCategory.ALTERED_BOUNDARY,
    "hypnagogia": StateCategory.ALTERED_BOUNDARY,
    "deja_vu": StateCategory.ALTERED_BOUNDARY,
    "morning_flow": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "evening_fatigue": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "creative_inspiration": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "anxious_rumination": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "calm": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "productive_focus": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "second_wind": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "hypervigilance": StateCategory.UNELABORATED,
    "sadness": StateCategory.UNELABORATED,
    "choice_paralysis": StateCategory.UNELABORATED,
    "mental_paralysis": StateCategory.UNELABORATED,
    "curious_exploration": StateCategory.UNELABORATED,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_state(name: str) -> APGIParameters:
    """Retrieve parameters for a named psychological state"""
    if name not in PSYCHOLOGICAL_STATES:
        raise KeyError(f"Unknown state: {name}. Available: {list(PSYCHOLOGICAL_STATES.keys())}")
    return PSYCHOLOGICAL_STATES[name]


def get_states_by_category(category: StateCategory) -> Dict[str, APGIParameters]:
    """Retrieve all states belonging to a category"""
    return {name: params for name, params in PSYCHOLOGICAL_STATES.items() if STATE_CATEGORIES.get(name) == category}


def compare_states(state1: str, state2: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Compare two states and return parameter differences.

    Returns:
        Dict mapping parameter names to (state1_value, state2_value, difference)
    """
    p1 = get_state(state1)
    p2 = get_state(state2)

    params = [
        "Pi_e",
        "Pi_i_baseline",
        "Pi_i_eff",
        "theta_t",
        "S_t",
        "M_ca",
        "beta",
        "z_e",
        "z_i",
    ]

    comparison = {}
    for param in params:
        v1 = getattr(p1, param)
        v2 = getattr(p2, param)
        comparison[param] = (v1, v2, v2 - v1)

    return comparison


def find_nearest_state(params: APGIParameters) -> Tuple[str, float]:
    """
    Find the psychological state nearest to given parameters.

    Uses Euclidean distance in normalized parameter space.

    Returns:
        (state_name, distance)
    """
    # Normalization ranges (approximate)
    ranges = {
        "Pi_e": (0.1, 15.0),
        "Pi_i_baseline": (0.1, 15.0),
        "Pi_i_eff": (0.1, 15.0),
        "theta_t": (-3.0, 3.0),
        "S_t": (0.0, 75.0),
        "M_ca": (-2.0, 2.0),
        "beta": (0.3, 0.8),
        "z_e": (0.0, 3.5),
        "z_i": (0.0, 3.5),
    }

    def normalize(value: float, param: str) -> float:
        min_v, max_v = ranges[param]
        return (value - min_v) / (max_v - min_v)

    def distance(p1: APGIParameters, p2: APGIParameters) -> float:
        total = 0.0
        for param in ranges.keys():
            v1 = normalize(getattr(p1, param), param)
            v2 = normalize(getattr(p2, param), param)
            total += (v1 - v2) ** 2
        return np.sqrt(total)

    min_dist = float("inf")
    nearest = None

    for name, state_params in PSYCHOLOGICAL_STATES.items():
        d = distance(params, state_params)
        if d < min_dist:
            min_dist = d
            nearest = name

    return nearest, min_dist


def compute_transition_cost(from_state: str, to_state: str) -> Dict[str, float]:
    """
    Compute the parameter changes required to transition between states.

    Returns:
        Dict mapping parameter names to required change magnitudes
    """
    comparison = compare_states(from_state, to_state)

    # Weight certain parameters as more "costly" to change
    weights = {
        "Pi_e": 1.0,  # Precision shifts are moderate cost
        "Pi_i_baseline": 1.2,  # Interoceptive precision harder to shift
        "Pi_i_eff": 0.8,  # Effective is derived, less "real" cost
        "theta_t": 1.5,  # Threshold changes are effortful
        "S_t": 0.5,  # Accumulated signal is outcome, not lever
        "M_ca": 1.3,  # Somatic bias is sticky
        "beta": 2.0,  # Trait-like, hardest to change
        "z_e": 0.7,  # Environment-dependent
        "z_i": 0.8,  # Body-dependent
    }

    costs = {}
    total_cost = 0.0

    for param, (v1, v2, diff) in comparison.items():
        cost = abs(diff) * weights.get(param, 1.0)
        costs[param] = cost
        total_cost += cost

    costs["total"] = total_cost
    return costs


def validate_transition_plausibility(from_state: str, to_state: str, pathway: List[str]) -> Dict[str, Any]:
    """
    Validate psychological plausibility of a transition pathway.

    Checks:
    - Parameter continuity (no extreme jumps)
    - Valence compatibility (avoid contradictory emotional transitions)
    - Category progression (gradual shifts rather than abrupt changes)
    """
    issues = []
    warnings = []
    score = 100.0

    # Check each transition step
    for i in range(len(pathway) - 1):
        current = pathway[i]
        next_state = pathway[i + 1]

        p_current = get_state(current)
        p_next = get_state(next_state)

        # Check for extreme parameter jumps
        param_changes = {
            "_e": abs(p_next.Pi_e - p_current.Pi_e),
            "_t": abs(p_next.theta_t - p_current.theta_t),
            "M_ca": abs(p_next.M_ca - p_current.M_ca),
        }

        if param_changes["_e"] > 4.0:
            issues.append(f"Large _e jump: {current}  {next_state} (={param_changes['_e']:.1f})")
            score -= 15
        elif param_changes["_e"] > 2.5:
            warnings.append(f"Moderate _e jump: {current}  {next_state} (={param_changes['_e']:.1f})")
            score -= 5

        if param_changes["_t"] > 2.0:
            issues.append(f"Large _t jump: {current}  {next_state} (={param_changes['_t']:.1f})")
            score -= 10
        elif param_changes["_t"] > 1.5:
            warnings.append(f"Moderate _t jump: {current}  {next_state} (={param_changes['_t']:.1f})")
            score -= 3

        # Check valence compatibility
        current_cat = STATE_CATEGORIES.get(current)
        next_cat = STATE_CATEGORIES.get(next_state)

        # Define category compatibility matrix
        incompatible_transitions = [
            (StateCategory.AVERSIVE_AFFECTIVE, StateCategory.OPTIMAL_FUNCTIONING),
            (StateCategory.PATHOLOGICAL_EXTREME, StateCategory.OPTIMAL_FUNCTIONING),
        ]

        for cat1, cat2 in incompatible_transitions:
            if current_cat == cat1 and next_cat == cat2:
                warnings.append(f"Direct aversiveoptimal transition: {current}  {next_state}")
                score -= 8

    return {
        "score": max(0, score),
        "issues": issues,
        "warnings": warnings,
        "plausible": score >= 70,
    }


def get_transition_pathway(from_state: str, to_state: str, validate: bool = True) -> Tuple[List[str], Dict]:
    """
    Suggest intermediate states for gradual transition.

    Uses parameter interpolation to find natural waypoints.
    Optionally validates psychological plausibility.

    Returns:
        Tuple of (pathway_list, validation_dict)
    """
    p1 = get_state(from_state)
    p2 = get_state(to_state)

    # Generate interpolated parameter sets
    pathway = [from_state]

    for alpha in [0.33, 0.66]:
        interpolated = create_apgi_params(
            Pi_e=p1.Pi_e + alpha * (p2.Pi_e - p1.Pi_e),
            Pi_i_baseline=p1.Pi_i_baseline + alpha * (p2.Pi_i_baseline - p1.Pi_i_baseline),
            M_ca=p1.M_ca + alpha * (p2.M_ca - p1.M_ca),
            beta=np.clip(p1.beta + alpha * (p2.beta - p1.beta), 0.3, 0.8),
            z_e=p1.z_e + alpha * (p2.z_e - p1.z_e),
            z_i=p1.z_i + alpha * (p2.z_i - p1.z_i),
            theta_t=p1.theta_t + alpha * (p2.theta_t - p1.theta_t),
        )
        nearest, _ = find_nearest_state(interpolated)
        if nearest not in pathway and nearest != to_state:
            pathway.append(nearest)

    pathway.append(to_state)

    validation = {}
    if validate:
        validation = validate_transition_plausibility(from_state, to_state, pathway)

    return pathway, validation


def get_state_summary(name: str) -> str:
    """Generate a human-readable summary of a state's parameters"""
    params = get_state(name)
    category = STATE_CATEGORIES.get(name, StateCategory.UNELABORATED)
    ignition_prob = params.compute_ignition_probability()

    summary = f"""
State: {name.upper().replace('_', ' ')}
Category: {category.name.replace('_', ' ')}


PRECISION PARAMETERS

  Exteroceptive (_e):        {params.Pi_e:5.2f}  {"" * int(params.Pi_e)}
  Interoceptive baseline:     {params.Pi_i_baseline:5.2f}  {"" * int(params.Pi_i_baseline)}
  Interoceptive effective:    {params.Pi_i_eff:5.2f}  {"" * int(params.Pi_i_eff)}

PRECISION PARAMETERS

  Exteroceptive (z_e):        {params.z_e:5.2f}  {"" * int(params.z_e * 3)}
  Interoceptive (z_i):        {params.z_i:5.2f}  {"" * int(params.z_i * 3)}

THRESHOLD & SOMATIC

  Ignition threshold (_t):   {params.theta_t:+5.2f}  {"" if params.theta_t > 0 else ""} {"" * abs(int(params.theta_t * 2))}
  Somatic marker (M_ca):      {params.M_ca:+5.2f}  {"+" if params.M_ca > 0 else "-"} {"" * abs(int(params.M_ca * 2))}
  Somatic gain ():           {params.beta:5.2f}

DERIVED VALUES

  Accumulated surprise (S_t): {params.S_t:6.2f}
  Ignition probability:       {ignition_prob:6.2%}

Formula verification:
  _i_eff = _i_baseline  exp(_somM):  {"" if params.verify_Pi_i_eff() else ""}
  S_t = _e|z_e| + _i_eff|z_i|:    {"" if params.verify_S_t() else ""}

"""
    return summary


def generate_state_comparison_table(states: List[str]) -> str:
    """Generate a formatted comparison table for multiple states"""
    headers = [
        "State",
        "_e",
        "_i_eff",
        "_t",
        "S_t",
        "M_ca",
        "",
        "z_e",
        "z_i",
        "P(ign)",
    ]

    rows = []
    for name in states:
        p = get_state(name)
        rows.append(
            [
                name[:12],  # Shortened to prevent overflow
                f"{p.Pi_e:.1f}",
                f"{p.Pi_i_eff:.1f}",
                f"{p.theta_t:+.1f}",
                f"{p.S_t:.1f}",
                f"{p.M_ca:+.1f}",
                f"{p.beta:.2f}",
                f"{p.z_e:.1f}",
                f"{p.z_i:.1f}",
                f"{p.compute_ignition_probability():.0%}",
            ]
        )

    # Format table with better spacing
    col_widths = [12, 6, 8, 6, 6, 6, 6, 6, 6, 6]  # Fixed widths for stability

    lines = []
    lines.append("" + "".join("" * w for w in col_widths) + "")
    lines.append("" + "".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + "")
    lines.append("" + "".join("" * w for w in col_widths) + "")

    for row in rows:
        lines.append("" + "".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + "")

    lines.append("" + "".join("" * w for w in col_widths) + "")

    return "\n".join(lines)


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================


def validate_all_states() -> Tuple[int, int, List[str]]:
    """Validate all state parameters and formulas"""
    results: Dict[str, Dict[str, bool]] = {}
    edge_cases: List[Dict[str, List[str]]] = []

    for name, params in PSYCHOLOGICAL_STATES.items():
        checks = {
            "Pi_i_eff_valid": params.verify_Pi_i_eff(),
            "S_t_valid": params.verify_S_t(),
            "ignition_prob_valid": 0.0 <= params.compute_ignition_probability() <= 1.0,
        }

        # Check for edge cases
        edge_case_checks = []
        if params.Pi_e < 0.5 or params.Pi_e > 14.5:
            edge_case_checks.append(f"_e={params.Pi_e:.1f} near boundary")
        if params.theta_t < -2.8 or params.theta_t > 2.3:
            edge_case_checks.append(f"_t={params.theta_t:+.1f} near boundary")
        if params.M_ca < -1.8 or params.M_ca > 1.8:
            edge_case_checks.append(f"M_ca={params.M_ca:+.1f} near boundary")
        if params.compute_ignition_probability() > 0.99:
            edge_case_checks.append("P(ignition)  100% (saturated)")
        if params.compute_ignition_probability() < 0.25:
            edge_case_checks.append("P(ignition) < 25% (suppressed)")

        if edge_case_checks:
            edge_cases.append({name: edge_case_checks})

        results[name] = checks

    # Calculate passed and failed counts
    passed_count = sum(1 for checks in results.values() if all(checks.values()))
    failed_count = len(results) - passed_count
    warnings = [f"{name}: {', '.join(checks)}" for item in edge_cases for name, checks in item.items()]

    return passed_count, failed_count, warnings


def print_validation_report() -> None:
    """Print a validation report for all states"""
    passed_count, failed_count, warnings = validate_all_states()
    results: Dict[str, Dict[str, bool]] = {}
    edge_cases: List[Dict[str, List[str]]] = []

    print("\n" + "=" * 70)
    print("APGI STATE LIBRARY VALIDATION REPORT")
    print("=" * 70)

    all_valid = True
    for name, checks in results.items():
        status = "" if all(checks.values()) else ""
        if not all(checks.values()):
            all_valid = False

        failed = [k for k, v in checks.items() if not v]
        if failed:
            print(f"  {status} {name}: FAILED - {', '.join(failed)}")
        else:
            print(f"  {status} {name}")

    print("-" * 70)
    print(f"Total states: {len(results)}")
    print(f"Overall status: {'ALL VALID ' if all_valid else 'SOME FAILURES '}")

    # Edge case reporting
    if edge_cases:
        print("\n" + "=" * 70)
        print("EDGE CASES AND BOUNDARY WARNINGS")
        print("=" * 70)
        for case in edge_cases:
            for name, warnings in case.items():
                print(f"\n{name}:")
                for warning in warnings:
                    print(f"   {warning}")
    else:
        print("\n No edge cases or boundary violations detected")

    print("=" * 70)


# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================


class PsychStatesModule:
    """TheoryModuleInterface implementation for APGI psychological state library."""

    MODULE_LEVEL: str = "Level 2"

    def compute(self, **kwargs) -> dict:
        passed_count, failed_count, warnings = validate_all_states()
        return {"passed": passed_count, "failed": failed_count, "warnings": warnings}

    def get_falsification_criteria(self) -> dict:
        return {
            "PSY-1": {
                "description": "All canonical states pass thermodynamic validity checks",
                "threshold": "0 failed states out of all registered states",
                "test": "Validate π_i, β, κ ranges and free energy consistency",
            },
            "PSY-2": {
                "description": "State transitions respect predicted free energy ordering",
                "threshold": "≥ 90% of transitions have ΔF in predicted direction",
                "test": "Compute transition cost for all adjacent state pairs",
            },
            "PSY-3": {
                "description": "Parameter nearest-state lookup accuracy > 85%",
                "threshold": "Top-1 accuracy > 85% on held-out parameter vectors",
                "test": "Leave-one-out cross-validation on state library",
            },
        }

    def as_protocol_result(self, protocol_id: str = "T-PsychologicalStates", **kwargs) -> dict:
        try:
            from utils.protocol_schema import PredictionResult, PredictionStatus, ProtocolResult
        except ImportError:
            return self.compute(**kwargs)
        raw = self.compute(**kwargs)
        criteria = self.get_falsification_criteria()
        named = {}
        for key, spec in criteria.items():
            passed = raw.get(key, {}).get("passed", False) if isinstance(raw.get(key), dict) else False
            named[key] = PredictionResult(
                passed=passed,
                status=PredictionStatus.PASSED if passed else PredictionStatus.NOT_EVALUATED,
                sources=[protocol_id],
                metadata=spec,
            )
        return ProtocolResult(
            protocol_id=protocol_id,
            named_predictions=named,
            completion_percentage=100,
            methodology=self.MODULE_LEVEL,
            metadata={"module_level": self.MODULE_LEVEL},
        ).to_dict()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("APGI PSYCHOLOGICAL STATE LIBRARY - DEMONSTRATION")
    print("=" * 70)

    # 1. Basic state retrieval
    print("\n1. BASIC STATE RETRIEVAL")
    print("-" * 40)
    fear_params = get_state("fear")
    print("Fear parameters:")
    print(f"  _e = {fear_params.Pi_e}")
    print(f"  _i_eff = {fear_params.Pi_i_eff}")
    print(f"  _t = {fear_params.theta_t}")
    print(f"  M_ca = {fear_params.M_ca}")
    print(f"  P(ignition) = {fear_params.compute_ignition_probability():.2%}")

    # 2. Category retrieval
    print("\n2. CATEGORY RETRIEVAL")
    print("-" * 40)
    optimal_states = get_states_by_category(StateCategory.OPTIMAL_FUNCTIONING)
    print(f"Optimal functioning states: {list(optimal_states.keys())}")

    # 3. State comparison
    print("\n3. STATE COMPARISON: Fear vs Anxiety")
    print("-" * 40)
    comparison = compare_states("fear", "anxiety")
    print(f"{'Parameter':<15} {'Fear':>8} {'Anxiety':>8} {'':>8}")
    print("-" * 40)
    for param, (v1, v2, diff) in comparison.items():
        print(f"{param:<15} {v1:>8.2f} {v2:>8.2f} {diff:>+8.2f}")

    # 4. State summary
    print("\n4. DETAILED STATE SUMMARY")
    print(get_state_summary("flow"))

    # 5. Comparison table
    print("\n5. AVERSIVE STATES COMPARISON TABLE")
    print(generate_state_comparison_table(["fear", "anxiety", "anger", "guilt", "shame", "loneliness", "overwhelm"]))

    # 6. Find nearest state
    print("\n6. NEAREST STATE DETECTION")
    print("-" * 40)
    test_params = create_apgi_params(Pi_e=7.0, Pi_i_baseline=3.0, M_ca=1.6, beta=0.7, z_e=2.2, z_i=1.8, theta_t=-2.0)
    nearest, distance = find_nearest_state(test_params)
    print(f"Test parameters most similar to: {nearest} (distance: {distance:.3f})")

    # 7. Transition pathway with validation
    print("\n7. STATE TRANSITION PATHWAY")
    print("-" * 40)
    pathway, validation = get_transition_pathway("anxiety", "calm", validate=True)
    print("Suggested pathway from 'anxiety' to 'calm':")
    print(f"  {'  '.join(pathway)}")

    # Display validation results
    if validation:
        print(f"\n  Plausibility Score: {validation['score']:.0f}/100")
        print(f"  Status: {' PLAUSIBLE' if validation['plausible'] else ' QUESTIONABLE'}")

        if validation["issues"]:
            print("\n  Issues:")
            for issue in validation["issues"]:
                print(f"     {issue}")

        if validation["warnings"]:
            print("\n  Warnings:")
            for warning in validation["warnings"]:
                print(f"     {warning}")

    # 8. Transition cost
    print("\n8. TRANSITION COST ANALYSIS")
    print("-" * 40)
    costs = compute_transition_cost("anxiety", "calm")
    print("Cost to transition from 'anxiety' to 'calm':")
    for param, cost in sorted(costs.items(), key=lambda x: -x[1] if x[0] != "total" else 0):
        if param != "total":
            print(f"  {param:<15}: {cost:.2f}")
    print(f"  {'TOTAL':<15}: {costs['total']:.2f}")

    # 9. Full validation
    print("\n9. LIBRARY VALIDATION")
    print_validation_report()

    # 10. All states summary statistics
    print("\n10. LIBRARY STATISTICS")
    print("-" * 40)

    all_pi_e = [p.Pi_e for p in PSYCHOLOGICAL_STATES.values()]
    all_theta = [p.theta_t for p in PSYCHOLOGICAL_STATES.values()]
    all_m_ca = [p.M_ca for p in PSYCHOLOGICAL_STATES.values()]
    all_ignition = [p.compute_ignition_probability() for p in PSYCHOLOGICAL_STATES.values()]

    print(f"Total states: {len(PSYCHOLOGICAL_STATES)}")
    print(f"\n_e range: {min(all_pi_e):.1f} - {max(all_pi_e):.1f} (mean: {np.mean(all_pi_e):.2f})")
    print(f"_t range: {min(all_theta):+.1f} - {max(all_theta):+.1f} (mean: {np.mean(all_theta):+.2f})")
    print(f"M_ca range: {min(all_m_ca):+.1f} - {max(all_m_ca):+.1f} (mean: {np.mean(all_m_ca):+.2f})")
    print(f"P(ignition) range: {min(all_ignition):.0%} - {max(all_ignition):.0%} (mean: {np.mean(all_ignition):.0%})")

    # 11. States by category count
    print("\n11. STATES BY CATEGORY")
    print("-" * 40)
    for category in StateCategory:
        states = get_states_by_category(category)
        print(f"  {category.name:<25}: {len(states):>2} states")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
