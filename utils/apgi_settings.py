"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default_apgi_config.yaml"


class APGISettings(BaseSettings):
    """Immutable-ish APGI runtime settings loaded from YAML + env overrides."""

    model_config = SettingsConfigDict(
        env_prefix="APGI_",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Ensure environment variables override YAML/init-provided defaults.
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # Network architecture
    input_size: int = 64
    hidden_size: int = 128
    num_levels: int = 5  # canonical L0–L4 (APGI-SCI-003)

    # Temporal dynamics
    dt_ms: float = 10.0
    max_window_ms: float = 500.0

    # Threshold parameters
    theta0: float = 0.03
    gamma: float = 0.1
    delta: float = 0.5
    lambda_urg: float = 0.2
    theta_min: float = 0.01
    theta_max: float = 5.0

    # Learning rates
    lr_extero: float = 0.01
    lr_intero: float = 0.01
    lr_somatic: float = 0.1
    lr_policy: float = 0.001
    lr_value: float = 0.001
    lr_precision: float = 0.05

    # Ignition parameters
    alpha_ignition: float = 8.0
    tau_S: float = 0.3
    tau_theta: float = 10.0
    eta_theta: float = 0.01

    # Initial values
    theta_init: float = 0.5
    theta_baseline: float = 0.5
    Pi_e_init: float = 1.0
    Pi_i_init: float = 1.5
    beta_somatic: float = 1.8

    # Stability
    eps: float = 1e-8

    # Refractory period
    max_refractory_ms: float = 50.0

    # Precision parameters
    precision_learning_rate: float = 0.05
    precision_min: float = 0.1
    precision_max: float = 10.0
    precision_history_max: int = 100

    # Neuromodulation (acetylcholine)
    neuromod_ach_baseline: float = 0.5
    neuromod_ach_scaling: float = 1.0

    # Energy dynamics
    energy_depletion_rate: float = 0.1
    energy_min: float = 0.0
    energy_max: float = 1.0

    # Allostatic regulation
    allostatic_increase_rate: float = 0.05
    allostatic_decrease_rate: float = 0.1
    allostatic_min: float = 0.0
    allostatic_max: float = 1.0

    # Monitoring flags
    gradient_monitoring_enabled: bool = True
    performance_tracking_enabled: bool = True
    cost_benefit_gating_enabled: bool = True

    # Physical temperature (thermodynamics)
    use_physical_temperature: bool = True
    boltzmann_constant: float = 1.380649e-23
    temperature_kelvin: float = 310.0
    temperature_normalized: float = 1.0
    energy_scale_factor: float = 1e21
    entropy_scale_factor: float = 1.0

    # Rigorous thermodynamics
    use_rigorous_thermodynamic_entropy: bool = True

    # Additional entropy/thermodynamics settings
    use_shannon_entropy: bool = True
    use_rigorous_variational_fe: bool = False
    cross_level_validation_enabled: bool = True
    cost_benefit_clamp_min: float = -10.0
    cost_benefit_clamp_max: float = 10.0
    cost_benefit_scaling: float = 1.0

    # Time constants
    tau_min: float = 0.01
    tau_max: float = 100.0
    tau_intero_baseline: float = 0.3
    tau_extero_baseline: float = 0.2

    # Refractory cost
    refractory_cost_baseline: float = 0.1
    refractory_cost_scaling: float = 1.0

    # Additional learning parameters
    alpha_broadcast: float = 1.0
    beta_maintenance: float = 0.5
    beta_transition: float = 0.8
    hysteresis: float = 0.1

    # History and memory
    volatility_history_max: int = 100
    entropy_calculation_interval: int = 10

    # Additional neuromodulation
    workspace_sustained_scaling: float = 1.0

    # Additional neural parameters
    gradient_warn_threshold: float = 1000.0
    gradient_clip_value: float = 1.0
    reservoir_scaling: float = 1.0
    reservoir_sparsity: float = 0.5

    # Optional provenance
    config_path: Optional[str] = Field(default=None, description="Resolved YAML path used")

    @classmethod
    def from_yaml(cls, path: Path) -> "APGISettings":
        data: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    data = loaded
        # Let env vars override YAML (BaseSettings handles env on init)
        settings = cls(**data)
        settings.config_path = str(path)
        return settings


def resolve_apgi_config_path(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return explicit
    env_path = os.environ.get("APGI_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH
