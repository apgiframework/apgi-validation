# APGI Validation Framework

A computational framework for validating and falsifying Active Inference and Predictive Processing models through systematic experimentation.

This repository contains all code needed to reproduce the validation protocols, falsification tests, and computational benchmarks presented in our work on the Allostatic Precision-Gated Ignition (APGI) theory framework.

---

## Data Availability

All empirical data used for validation (synthetic EEG, fMRI vmPFC recordings, cross-cultural datasets) is organized in the `data_repository/` directory:

- `empirical_data/` - Raw and processed experimental data
- `metadata/` - Dataset documentation and codebooks
- `dashboard_data/` - Pre-computed results for dashboard visualization
- `codebooks/` - Variable dictionaries and data specifications

Pre-computed validation results and performance profiles are available in `apgi_outputs/` for rapid reproduction of figures and statistical analyses.

---

## Usage

### Installation

First, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/apgi-validation.git
cd apgi-validation
```

We recommend creating a virtual environment for this project:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required Python packages. `requirements.txt` includes
`requirements-protocols.txt`, so this single command installs everything:

```bash
python3 -m pip install -r requirements.txt
```

To install only the core CLI and GUIs, without the heavier protocol stack
(torch, PyMC, MNE):

```bash
python3 -m pip install .
```

Or as an editable install with the protocol extras and dev tooling:

```bash
python3 -m pip install -e '.[protocols,dev]'
```

### Required Environment Variables

Copy the template and generate a master key:

```bash
cp .env.example .env
```

```bash
python3 -c "from cryptography.fernet import Fernet; print('APGI_MASTER_KEY=' + Fernet.generate_key().decode())" >> .env
```

`APGI_MASTER_KEY` is the only key you must supply. It encrypts the pickle,
backup-HMAC and JWT keys, each of which is minted automatically into `.keys/`
on first run. Keep it safe: if it is lost, existing key files cannot be
decrypted, and the framework will refuse to start rather than silently
replacing them. Recover by restoring the key, or by rotating:

```bash
python3 -m utils.key_rotation_manager --rotate
```

For development and CI you can skip the master key entirely by allowing an
ephemeral one — never do this in production:

```bash
export APGI_ALLOW_EPHEMERAL_MASTER_KEY=1
```

Authentication is enforced by default. GUIs require `--token`; for local
single-user work, set `APGI_ENFORCE_AUTH=0` to get an auto-generated
24-hour development token.

### Quick Start

Run the CLI to see available commands:

```bash
python3 main.py --help
```

Run a formal model simulation with default parameters:

```bash
python3 main.py formal-model --plot
```

Run with custom parameters:

```bash
python3 main.py formal-model --simulation-steps 500 --dt 0.01 --output-file results.csv
```

---

## Theory Modules

The theoretical foundation of APGI is implemented in the `Theory/` directory:

### Core Components

- **`APGI_Bayesian_Estimation_Framework.py`** - Bayesian parameter estimation with precision-weighted prediction errors
- **`APGI_Computational_Benchmarking.py`** - Performance profiling and scaling analysis
- **`APGI_Cross_Species_Scaling.py`** - Cross-species neural dynamics comparison
- **`APGI_Cultural_Neuroscience.py`** - Cultural variation in predictive processing
- **`APGI_Entropy_Implementation.py`** - Information-theoretic entropy measures for APGI signals
- **`APGI_Falsification_Framework.py`** - Falsification protocol infrastructure
- **`APGI_Fractal_Threshold_Dynamics.py`** - Fractal and DFA-based threshold dynamics analysis
- **`APGI_Hierarchical_Temporal_Tiling.py`** - Hierarchical temporal tiling for multi-scale integration
- **`APGI_Information_Theoretic_Bandwidth.py`** - Information-theoretic bandwidth and capacity analysis
- **`APGI_LNN_Bifurcation_Analysis.py`** - Liquid neural network bifurcation analysis
- **`APGI_Liquid_Network_Implementation.py`** - Liquid-state network dynamics implementation
- **`APGI_Multimodal_Classifier.py`** - Multimodal signal classification
- **`APGI_Multimodal_Integration.py`** - Cross-modal integration protocols
- **`APGI_Neuromodulatory_Control_System.py`** - Neuromodulatory gain-control mechanisms
- **`APGI_Open_Science_Framework.py`** - Open science reproducibility utilities
- **`APGI_Parameter_Estimation.py`** - Maximum likelihood and Bayesian parameter estimation
- **`APGI_Psychological_States.py`** - Psychological state inference from APGI signals
- **`APGI_Somatic_Marker_Identifiability.py`** - Somatic marker identifiability analysis
- **`APGI_Thermodynamic_Program_Aggregator.py`** - Thermodynamic program aggregation utilities
- **`APGI_Turing_Machine.py`** - Turing-complete APGI computation model

All theory modules can be run independently or through the unified CLI in `main.py`.

---

## Protocol Specification (v1.1.0)

`protocols/` holds the eight registered experimental protocols (Protocol 0–7) at
specification release **1.1.0**, byte-identical to the copy in
[apgi-framework](https://github.com/apgiframework/apgi-framework/tree/main/protocols).
`protocols/CHANGELOG.md` lists every change from 1.0 and the source-lock
requirements that remain open; `protocols/MANIFEST.json` pins each file's SHA-256.

| ID | File | Registered predictions |
|----|------|------------------------|
| APGI-P00 | `protocol_0_hep_proxy_validation.json` | Pred 0.A–0.C |
| APGI-P01 | `protocol_1_cardiac_eeg.json` | Pred 1.A–1.F |
| APGI-P02 | `protocol_2_somatic_agent_sim.json` | Pred 2.A–2.G |
| APGI-P03 | `protocol_3_anticipation_fmri.json` | Pred 3.A–3.D |
| APGI-P04 | `protocol_4_metabolic_crossover.json` | Pred 4.A–4.C |
| APGI-P05 | `protocol_5_causal_tms.json` | Pred 5.A–5.E |
| APGI-P06 | `protocol_6_ignition_ieeg.json` | Pred 6.A–6.E |
| APGI-P07 | `protocol_7_doc_biomarker.json` | Pred 7.A–7.G (7.G primary) |

Validate the release (schema, cross-field arithmetic, exact Protocol 0–7
membership and manifest checksums); CI runs the same check:

```bash
make protocols-check   # python3 scripts/validate_protocols.py
```

Rules the validation code follows under 1.1.0:

- **Three-way verdicts.** Each registered prediction is CONFIRMED, FALSIFIED or
  INDETERMINATE (`utils.protocol_loader.classify_prediction`). A failure to confirm
  is not falsification. The simulations here never apply a registered falsifying
  rule to empirical data, so they report a registered prediction as confirmed
  (synthetic) or INDETERMINATE, never as falsified.
- **κ is uncalibrated.** `apgi_parameters.kappa.value` is `null`; it is the ATP/bit
  energy–information bridge, never the ignition steepness (`gamma_sig`).
- **No legacy defaults.** `KAPPA = 100`, `ALPHA = 0.3` and `BETA = 0.7` are not
  registered values; read thresholds and cohort sizes from the protocol JSON.
- **Status, not validation.** Every protocol is `pre_collection`; passing the
  validator checks structure, not scientific validity or regulatory approval.

The pre-1.1 typed-benchmark template now lives at
`config/apgi_protocol_template.json` for reference only.

---

## Validation Protocols

Systematic validation protocols are implemented in the `Validation/` directory:

### Protocol 1: Synthetic EEG Classification

**File:** `VP_01_SyntheticEEG_MLClassification.py`

Validates APGI predictions using machine learning classification of synthetic EEG signals. Generates synthetic P3b waveforms with varying latency and amplitude, then tests classification accuracy.

Run:

```bash
python3 Validation/VP_01_SyntheticEEG_MLClassification.py
```

### Protocol 2: Behavioral Bayesian Comparison

**File:** `VP_02_Behavioral_BayesianComparison.py`

Compares APGI model predictions against behavioral data using Bayesian model comparison metrics.

### Protocol 3: Active Inference Agent Simulations

**File:** `VP_03_ActiveInference_AgentSimulations.py`

Runs agent-based simulations to validate active inference predictions in controlled environments.

### Additional Protocols

- **`BayesianModelComparison_ParameterRecovery.py`** - Parameter recovery validation
- **`BayesianModelComparison_PredictiveAccuracy.py`** - Predictive accuracy assessment
- **`BayesianModelComparison_RobustnessChecks.py`** - Robustness validation
- **`CrossValidation_Empirical_Generalizability.py`** - Cross-validation protocols
- **`CrossValidation_ParameterStability.py`** - Parameter stability analysis
- **`Master_Validation.py`** - Orchestrates all validation protocols
- **`Multiverse_Analysis_Robustness.py`** - Multiverse robustness analysis
- **`VP_03_ActiveInference_AgentSimulations_Protocol3.py`** - Extended agent simulations
- **`VP_04_CausalIntervention_TMS.py`** - TMS intervention validation

Run all validations:

```bash
python3 Validation/Master_Validation.py
```

---

## Falsification Protocols

Rigorous falsification tests are implemented in the `Falsification/` directory:

### Core Falsification Tests

- **`FP_01_ActiveInference.py`** - Active inference model falsification
- **`FP_02_AgentComparison_ConvergenceBenchmark.py`** - Multi-agent convergence benchmarks
- **`FP_03_FrameworkLevel_MultiProtocol.py`** - Framework-level multi-protocol testing
- **`FP_04_PhaseTransition_EpistemicArchitecture.py`** - Phase transition analysis
- **`FP_05_SurvivalAnalysis_TimeToIgnition.py`** - Survival analysis of ignition timing
- **`FP_06_Perturbation_Resilience.py`** - Perturbation resilience testing
- **`FP_07_Diversity_BehavioralSpaceCoverage.py`** - Behavioral space coverage analysis
- **`FP_08_MetaLearning_FewShotAdaptation.py`** - Meta-learning adaptation tests
- **`FP_09_TemporalDynamics_SequenceSensitivity.py`** - Temporal dynamics validation
- **`FP_10_Adversarial_AttackResistance.py`** - Adversarial robustness testing
- **`FP_11_ResourceEfficiency_ComputationalCost.py`** - Resource efficiency benchmarks
- **`FP_12_Falsification_Aggregator.py`** - Aggregates falsification results

Run the falsification aggregator:

```bash
python3 Falsification/FP_12_Falsification_Aggregator.py
```

---

## Testing

Comprehensive test suite with property-based testing:

```bash
# Run all tests
python3 -m pytest

# Run specific test categories
python3 -m pytest tests/test_cli_integration_comprehensive.py      # CLI integration
python3 -m pytest tests/test_utility_modules_comprehensive.py      # Utility modules
python3 -m pytest tests/test_file_io_real.py                       # File I/O
python3 -m pytest tests/test_data_pipeline_end_to_end.py           # Data pipeline
python3 -m pytest tests/test_property_based_comprehensive.py       # Property-based tests
python3 -m pytest tests/test_validation_falsification_protocols_individual.py  # Protocols
python3 -m pytest tests/test_performance_regression.py             # Performance
python3 -m pytest tests/test_concurrent_config_access.py           # Concurrency
```

Run with coverage:

```bash
python3 -m pytest --cov=utils --cov=Theory --cov=Validation --cov=Falsification
```

---

## GUI Applications

Launch graphical interfaces for interactive exploration:

```bash
python3 Theory_GUI.py         # Theory module GUI
python3 Validation_GUI.py     # Validation protocols GUI
python3 Falsification_GUI.py  # Falsification tests GUI
python3 Tests_GUI.py          # Test runner GUI
python3 Utils_GUI.py          # Utilities GUI
```

---

## Configuration

Configuration files are in the `config/` directory:

- **`default.yaml`** - Default configuration settings
- **`config_schema.json`** - Configuration validation schema
- **`profiles/`** - Environment-specific profiles (ADHD, anxiety-disorder, research-default)
- **`versions/`** - Version-specific configurations

Load a specific profile:

```bash
python3 main.py formal-model --params config/profiles/adhd.yaml
```

### APGI System API

Like every APGI application, this repository talks to the hosted API at
`https://api.apgiframework.com`. Use `utils/apgi_api_client.py` rather than hardcoding a URL:

```python
from utils.apgi_api_client import get_client

client = get_client()          # reads APGI_API_URL / APGI_API_TOKEN
client.meta()                  # citation, licence and release-state identifiers
```

Set `APGI_API_URL` only for staging or a local API (plain `http` is accepted for loopback hosts only).

---

## Project Structure

```text
apgi-validation/
├── Theory/                    # Theoretical framework implementations
├── Validation/                # Validation protocols
├── Falsification/             # Falsification test protocols
├── utils/                     # Utility modules
│   ├── errors.py              # Typed error taxonomy
│   ├── security_audit_logger.py  # Security auditing
│   ├── cache_manager.py       # Cache governance
│   ├── data_protection.py     # Data protection workflows
│   └── ...
├── protocols/                 # Protocol specification release 1.1.0 (JSON + schema)
├── scripts/                   # validate_protocols.py
├── config/                    # Configuration files
├── data_repository/           # Data storage
│   ├── empirical_data/
│   ├── metadata/
│   └── codebooks/
├── tests/                     # Comprehensive test suite
├── docs/                      # Documentation
└── apgi_outputs/              # Output directory
    └── performance_profiles/
```

---

## Security Features

- Secure module loading with path validation
- Persistent audit logging (`SecurityAuditLogger`)
- Dependency vulnerability scanning (pip-audit integration)
- Thread-safe configuration access
- Deny-by-default security context
- Automated SAST/DAST in CI pipeline (Bandit, Semgrep, Safety)
- Typed error taxonomy with domain-specific exceptions
- Data protection workflows with secure deletion

---

## Documentation

Additional documentation is available in `docs/`:

- **`APGI-Design-Guide.md`** - UI/UX design specifications
- **`APGI-Equations.md`** - Complete mathematical formalization
- **`APGI-Parameter-Specifications.md`** - Parameter specifications
- **`APGI-Empirical-Credibility-Roadmap.md`** - Empirical validation roadmap
- **`Compliance-Matrix.md`** - Regulatory compliance matrix
- **`APGI-Falsification-Criteria.md`** - Falsification criteria

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{apgi_validation_framework,
  title={APGI Validation Framework: A Computational Framework for Validating Active Inference Models},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/apgi-validation}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## About

The APGI Framework provides a systematic, scientifically rigorous approach to validating and falsifying computational models of active inference and predictive processing. It combines theoretical formalization, computational benchmarking, and empirical validation through a unified CLI interface with comprehensive testing and security features.
