"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class NetworkConfig(BaseModel):
    input_size: int = Field(64, gt=0)
    hidden_size: int = Field(128, gt=0)
    num_levels: int = Field(5, gt=0)  # canonical L0–L4 (APGI-SCI-003)


class TemporalConfig(BaseModel):
    dt_ms: float = Field(10.0, gt=0)
    max_window_ms: float = Field(500.0, gt=0)


class ThresholdConfig(BaseModel):
    theta0: float = Field(0.03, gt=0)
    gamma: float = Field(0.1, gt=0)
    delta: float = Field(0.5, ge=0, le=1)
    lambda_urg: float = Field(0.2, ge=0, le=1)
    theta_min: float = Field(0.01, gt=0)
    theta_max: float = Field(5.0, gt=0)

    @field_validator("theta_max")
    @classmethod
    def validate_theta_range(cls, v: float, info: Any) -> float:
        if "theta_min" in info.data and v <= info.data["theta_min"]:
            raise ValueError("theta_max must be greater than theta_min")
        return v


class PrecisionConfig(BaseModel):
    tau_min: float = Field(0.01, gt=0)
    tau_max: float = Field(10.0, gt=0)
    tau_intero_baseline: float = Field(0.1, gt=0)
    tau_extero_baseline: float = Field(0.05, gt=0)
    precision_min: float = Field(0.01, gt=0)
    precision_max: float = Field(10.0, gt=0)
    learning_rate: float = Field(0.01, gt=0)


class MetabolicConfig(BaseModel):
    alpha_broadcast: float = Field(1.0, ge=0)
    beta_maintenance: float = Field(0.5, ge=0)
    energy_depletion_rate: float = Field(0.5, ge=0)
    energy_min: float = Field(0.0, ge=0)
    energy_max: float = Field(1.0, gt=0)


class IgnitionConfig(BaseModel):
    alpha: float = Field(8.0, gt=0)
    tau_S: float = Field(0.3, gt=0)
    tau_theta: float = Field(10.0, gt=0)
    eta_theta: float = Field(0.01, gt=0)
    theta_init: float = Field(0.5, gt=0)
    theta_baseline: float = Field(0.5, gt=0)


class APGISystemConfig(BaseModel):
    version: str = "1.3.0"
    network: NetworkConfig = NetworkConfig()  # type: ignore[call-arg]
    temporal: TemporalConfig = TemporalConfig()  # type: ignore[call-arg]
    thresholds: ThresholdConfig = ThresholdConfig()  # type: ignore[call-arg]
    precision: PrecisionConfig = PrecisionConfig()  # type: ignore[call-arg]
    metabolic: MetabolicConfig = MetabolicConfig()  # type: ignore[call-arg]
    ignition: IgnitionConfig = IgnitionConfig()  # type: ignore[call-arg]

    # Generic learning rates
    learning_rates: Dict[str, float] = {
        "extero": 0.01,
        "intero": 0.01,
        "somatic": 0.1,
        "policy": 0.001,
        "value": 0.001,
        "precision": 0.05,
    }

    # Falsification Thresholds
    falsification: Dict[str, Union[float, int, str]] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "APGISystemConfig":
        path = Path(path)
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        return cls(**data)

    def save(self, path: Union[str, Path]):
        path = Path(path)
        data = self.model_dump()
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
