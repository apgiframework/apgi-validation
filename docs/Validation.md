# APGI Theory Framework

A comprehensive computational framework for implementing the **Allostatic Precision-Gated Ignition (APGI)** theory, which models psychological state dynamics through surprise accumulation, ignition cascades, and multimodal integration.

The APGI Theory Framework provides:

- **Formal mathematical models** of surprise accumulation and ignition dynamics
- **Multimodal integration** protocols for psychological state assessment
- **Parameter estimation** tools using Bayesian inference
- **Validation protocols** for model verification
- **CLI interfaces** for interactive exploration

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd apgi-theory

# Install dependencies
pip install -r requirements.txt

# Run the main CLI
python main.py --help
```

### Basic Usage

```bash
# Run formal model simulation
python main.py formal-model --simulation-steps 1000

# Execute multimodal integration
python main.py multimodal --input-data data_repository/raw_data/sample.csv

# Estimate parameters from data
python main.py estimate-params --data-file data_repository/raw_data/experimental.csv

# Run validation protocols
python main.py validate --protocol 3

# Run falsification protocols
python main.py falsify --protocol 1

# Cross-species scaling analysis
python main.py cross-species --species human

# Launch GUI
python3 main.py gui --gui-type validation

# View framework information
python3 main.py info
```

## CLI Commands Reference

### Core Commands

| Command | Description | Status |
| :--- | :--- | :--- |
| `python3 main.py formal_model` | Run formal model simulations | WORKING |
| `python3 main.py multimodal` | Execute multimodal integration | WORKING |
| `python3 main.py estimate-params` | Perform parameter estimation | WORKING |
| `python3 main.py validate` | Run validation protocols | WORKING |
| `python3 main.py falsify` | Execute falsification tests | WORKING |

### Protocol Commands

| Command | Description | Status |
| :--- | :--- | :--- |
| `python3 main.py validate --protocol N` | Run specific validation protocol (1-22) | WORKING |
| `python3 main.py validate --all-protocols` | Run all validation protocols | WORKING |
| `python3 main.py falsify --protocol N` | Run specific falsification protocol (1-12) | WORKING |
| `python3 main.py validate-neural-signatures` | Validate neural signature protocols | WORKING |
| `python3 main.py validate-causal-manipulations` | Validate causal manipulation protocols | WORKING |
| `python3 main.py validate-clinical-convergence` | Validate clinical convergence protocols | WORKING |

### Utility Commands

| Command | Description | Status |
| :--- | :--- | :--- |
| `python3 main.py info` | Show framework information | WORKING |
| `python3 main.py config --show` | Show current configuration | WORKING |
| `python3 main.py config --set key=value` | Set configuration value | WORKING |
| `python3 main.py config --reset` | Reset configuration to defaults | WORKING |
| `python3 main.py logs --tail 20` | View recent log entries | WORKING |
| `python3 main.py logs --follow` | Follow logs in real-time | WORKING |
| `python3 main.py logs --export logs.json` | Export logs to file | WORKING |
| `python3 main.py analyze-logs --level ERROR` | Analyze log patterns | WORKING |
| `python3 main.py performance` | Show performance metrics | WORKING |
| `python3 main.py monitor-performance` | Monitor command performance | WORKING |
| `python3 main.py visualize --input-file data.csv` | Generate visualizations | WORKING |
| `python3 main.py process-data --input-file data.csv` | Process data files | WORKING |

### Data & Cache Management

| Command | Description | Status |
| :--- | :--- | :--- |
| `python3 main.py export-data --input-file X --output-file Y` | Export data | WORKING |
| `python3 main.py import-data --input-file X --output-file Y` | Import data | WORKING |
| `python3 main.py cache --action clear` | Clear cache | WORKING |
| `python3 main.py dashboard` | Generate dashboards | WORKING |

### Backup & Restore

| Command | Description | Status |
| :--- | :--- | :--- |
| `python3 main.py backup --components "config,logs"` | Create backup | WORKING |
| `python3 main.py restore-backup --backup-id X` | Restore from backup | WORKING |
| `python3 main.py list-backups --limit 10` | List backups | WORKING |
| `python3 main.py delete-backup --backup-id X` | Delete backup | WORKING |
| `python3 main.py cleanup-backups --keep 5` | Clean old backups | WORKING |

### Advanced Validation Commands

| Command | Description | Status |
| :--- | :--- | :--- |
| `python3 main.py validate-quantitative-fits` | Validate quantitative model fits | WORKING |
| `python3 main.py validate-open-science` | Validate open science framework compliance | WORKING |
| `python3 main.py bayesian-estimation --method mcmc` | Run Bayesian estimation | WORKING |

### Error Management

| Command | Description | Status |
| :--- | :--- | :--- |
| `python3 main.py errors --category CONFIGURATION` | View error history | WORKING |
| `python3 main.py test-errors --test-config` | Test error handling system | WORKING |

### GUI

| Command | Description | Status |
| :--- | :--- | :--- |
| `python3 main.py gui --gui-type validation` | Launch validation GUI | WORKING |
| `python3 main.py gui --gui-type falsification` | Launch falsification GUI | WORKING |
| `python3 main.py gui --gui-type theory` | Launch theory GUI | WORKING |
| `python3 main.py gui --gui-type utils` | Launch utilities GUI | WORKING |
| `python3 Tests_GUI.py` | Launch test GUI directly | WORKING |

## 🔬 Core Components

### 1. Formal Model (`apgi_core/equations.py`)

Implements dynamical system for accumulated surprise and ignition:

  • Surprise accumulation with timescale τ_S
  • Threshold adaptation with timescale τ_θ
  • Ignition cascades and reset dynamics
  • Metabolic and arousal modulation
  • 51 psychological states with domain-specific thresholds
  • Neuromodulator mapping (ACh, NE, DA, 5-HT)

### 2. Full Dynamic Model (`apgi_core.full_model`)

Extended dynamical system with complete implementations:

  • Corrected parameter ranges
  • Complete measurement equations (HEP, P3b)
  • Π vs Π̂ distinction for anxiety modeling
  • Psychiatric profiles (GAD, MDD, Psychosis)

### 3. Multimodal Integration (`Theory/APGI_Multimodal_Integration.py`)

Integrates multiple data modalities:

  • Physiological signals (HRV, GSR, EEG, fMRI)
  • Behavioral metrics
  • Self-report measures
  • Real-time state estimation
  • Cross-modal fusion algorithms
  • APGI normalizer and core integration classes

### 4. Parameter Estimation (`Theory/APGI_Parameter_Estimation.py`)

Bayesian parameter inference:

  • Markov Chain Monte Carlo (MCMC) sampling
  • Posterior distribution analysis
  • Model comparison metrics
  • Uncertainty quantification
  • Neural signal generators (HEP, P3b waveforms)

### 5. Validation Protocols (`Validation/` - 22 Protocols)

Comprehensive testing suite:

| Protocol | Description | Tier |
| -------- | ----------- | ---- |
| VP_01 | Synthetic EEG ML Classification | Primary |
| VP_02 | Behavioral Bayesian Comparison | Primary |
| VP_03 | Active Inference Agent Simulations | Primary |
| VP_04 | Phase Transition Epistemic Tier 2 | Primary |
| VP_05 | Evolutionary Emergence | Secondary |
| VP_06 | Liquid Network Inductive Bias | Secondary |
| VP_07 | TMS Causal Interventions (canonical P2.a–P2.c) | Primary |
| VP_08 | Psychophysical Threshold Estimation | Primary |
| VP_09 | Neural Signatures Empirical Priority 1 | Primary |
| VP_10 | Causal Manipulations Priority 2 (supplementary V10.a–V10.c) | Secondary |
| VP_11 | MCMC Cultural Neuroscience Priority 3 | Secondary |
| VP_12 | Clinical Cross-Species Convergence | Secondary |
| VP_13 | Epistemic Architecture | Secondary |
| VP_14 | fMRI Anticipation Experience (P5a–P5d) | Primary |
| VP_15 | fMRI Anticipation vmPFC (P5e–P5h stubs) | Secondary |
| VP_16 | Metabolic ATP Ground-Truth (synthetic; qBOLD/PET pending) | Secondary |
| VP_17 | Allen Visual Coding Fatigue (synthetic; allensdk path optional) | Secondary |
| VP_18 | EEG Microstate GFP / P3b Energy | Secondary |
| VP_19 | Information Erasure MVPA | Secondary |
| VP_20 | Empirical iEEG Pipeline (BIDS-iEEG) | Secondary |
| VP_21 | Free Energy Prediction Error (MMN + HEP) | Secondary |
| VP_22 | fMRI Anticipation Experience — canonical P5a–P5d owner | Primary |

### 6. Falsification Protocols (`Falsification/` - 12 Protocols)

Model falsification testing. **FP-03 must run after all other FP protocols** (it performs
meta-level cross-protocol consistency checks). FP-05 and FP-08 require prerequisite
data (genome_data from VP-05, APGIAgent from VP-03) which are injected automatically
by `Master_Falsification.run_falsification()`.

Gate: **F8.2** (FP-08 FIM rank-deficiency) blocks VP-08 and VP-11 if flagged —
pass `falsification_results` to `run_end_to_end_validation_pipeline()`.

| Protocol | Description | Key Criteria |
| -------- | ----------- | ------------ |
| FP_01 | Active Inference — MCMC + threshold dynamics | F1.1–F1.6 |
| FP_02 | Agent Comparison Convergence Benchmark | F2.1–F2.5 |
| FP_03 | Framework-Level Multi-Protocol Consistency (**runs last**) | F3.1–F3.6 |
| FP_04 | Phase Transition Epistemic Architecture | F4.1–F4.5 |
| FP_05 | Evolutionary Plausibility (needs genome_data) | F5.1–F5.6 |
| FP_06 | Liquid Network Energy Benchmark | F6.1–F6.2 |
| FP_07 | Mathematical Consistency (algebra + numerics) | F7.x |
| FP_08 | Parameter Sensitivity & Identifiability (**Gate F8.2**) | F8.1–F8.x |
| FP_09 | Neural Signatures P3b / HEP | F9.x |
| FP_10 | Bayesian Estimation MCMC (PyMC v5; pymc3 deprecated) | F10.x |
| FP_11 | Liquid Network Dynamics Echo State | F11.x |
| FP_12 | Cross-Species Scaling | F12.x |

### System Requirements

- Python 3.8+
- 8GB+ RAM recommended
- 2GB+ disk space

See `requirements.txt` for complete list. Key dependencies include:

- `numpy`, `scipy`, `pandas` - Scientific computing
- `torch`, `scikit-learn` - Machine learning
- `pymc`, `arviz` - Bayesian modeling
- `matplotlib`, `seaborn` - Visualization

## 📖 Detailed Usage

### Command Line Interface

The unified CLI provides access to all framework components:

```bash
python3 main.py [COMMAND] [OPTIONS]

Commands:
  formal-model              Run formal model simulations
  multimodal                Execute multimodal integration
  estimate-params           Perform parameter estimation
  cross-species             Cross-species scaling analysis
  validate                  Run validation protocols
  falsify                   Execute falsification tests
  config                    Manage configuration settings
  logs                      View log files
  analyze-logs              Analyze log patterns
  info                      Show framework information
  backup                    Create backups
  restore-backup            Restore from backups
  list-backups              List backups
  cleanup-backups           Clean old backups
  delete-backup             Delete backup
  performance               Show performance metrics
  monitor-performance       Monitor command performance
  process-data              Process data files
  visualize                 Generate visualizations
  export-data               Export data
  import-data               Import data
  cache                     Cache management
  dashboard                 Generate dashboards
  gui                       Launch GUI interfaces
  validate-neural-signatures       Validate neural signatures
  validate-causal-manipulations    Validate causal manipulations
  validate-clinical-convergence    Validate clinical convergence
  validate-quantitative-fits       Validate quantitative fits
  validate-open-science            Validate open science framework
  bayesian-estimation             Run Bayesian estimation
  errors                    View error history
  test-errors               Test error handling
```

### Configuration Management

Configuration files in `config/` allow customization of:

- Model parameters
- Simulation settings
- Data paths
- Logging preferences
- Visualization options

### Logging Framework

Comprehensive logging system with:

- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Rotating log files
- Structured output formats
- Performance metrics tracking

## 🧪 Validation and Testing

### Running Validation

```bash
# Run all validation protocols
python3 main.py validate --all-protocols

# Run specific validation protocol (1-22)
python3 main.py validate --protocol 3

# Run validation with output directory
python3 main.py validate --protocol 1 --output-dir results/

# Run validation in parallel
python3 main.py validate --all-protocols --parallel
```

### Falsification Testing

```bash
# List available falsification protocols
python3 main.py falsify

# Execute specific falsification protocol (1-12)
python3 main.py falsify --protocol 1

# Run falsification with output file
python3 main.py falsify --protocol 2 --output-file results.json

# Run comprehensive falsification
python3 main.py falsify --comprehensive
```

## 📊 Example Workflows

### 1. Basic Simulation

```python
from apgi_core.equations import SurpriseIgnitionSystem

# Initialize system
system = SurpriseIgnitionSystem()

# Run simulation
dt = 0.01
steps = 1000

def input_generator(t):
    return {
        'Pi_e': 1.0,      # Exteroceptive precision
        'Pi_i': 1.0,      # Interoceptive precision
        'eps_e': 1.0,     # Exteroceptive prediction error
        'eps_i': 0.5,     # Interoceptive prediction error
        'beta': 1.2       # Somatic bias
    }

results = system.simulate(duration=steps*dt, dt=dt, input_generator=input_generator)
```

### 2. Parameter Estimation

```python
from Theory.APGI_Parameter_Estimation import ParameterEstimator
import pandas as pd

# Load data
data = pd.read_csv('data_repository/raw_data/experiment.csv')

# Estimate parameters
estimator = ParameterEstimator()
results = estimator.estimate_parameters(
    data,
    method='mcmc',
    iterations=1000
)
```

### 3. Multimodal Integration

```python
from Theory.APGI_Multimodal_Integration import APGINormalizer, APGICoreIntegration

# Initialize normalizer with configuration
config = {
    'exteroceptive': {'mean': 0, 'std': 1},
    'interoceptive': {'mean': 0, 'std': 1},
    'somatic': {'mean': 0, 'std': 1}
}
normalizer = APGINormalizer(config)

# Initialize integration
integration = APGICoreIntegration(normalizer)

# Process multimodal data
result = integration.integrate_multimodal({
    'exteroceptive': eeg_data,
    'interoceptive': pupil_data,
    'somatic': eda_data
})

print(f"Accumulated surprise: {result['S_t']}")
print(f"Ignition probability: {result['P_ignition']}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all validation protocols pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the details below:

```text
MIT License

Copyright (c) 2026 APGI Framework

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📚 References

- [Primary APGI Theory Paper]
- [Mathematical Foundations]
- [Validation Studies]

## 🆘 Support

For issues and questions:

1. Check the logs directory for error messages
2. Review validation protocol outputs
3. Consult the troubleshooting section in documentation
4. Create an issue with detailed error information

## 🔄 Version History

- **v1.0.0** - Initial framework release
- **v1.1.0** - Added multimodal integration
- **v1.2.0** - Enhanced parameter estimation
- **v1.3.0** - Unified CLI and configuration system
