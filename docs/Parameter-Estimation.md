# APGI Parameter Estimation Protocol

This script implements a hierarchical Bayesian framework for estimating parameters of the Allostatic Precision-Gated Ignition (APGI) model, which integrates interoceptive and exteroceptive processing. The protocol synthesizes realistic neural and behavioral data, fits a hierarchical Bayesian model, and validates parameter recovery across multiple modalities.

## Core Components

### 1. Neural Signal Generator (`NeuralSignalGenerator` Class)

Generates biologically plausible neural and physiological signals:

- **Heartbeat-Evoked Potential (HEP)**: Simulates EEG waveforms following heartbeats
  - Adds realistic noise (1/f pink noise) and 50Hz line noise
  - Amplitude modulated by interoceptive precision (Pi_i)

- **P3b Component**: Simulates the P3b event-related potential
  - Models both P3b and earlier P2 components
  - Includes physiological noise

- **Pupil Response**: Generates pupil dilation responses
  - Includes blink artifacts and slow drifts
  - Modulated by interoceptive precision

### 2. APGI Dynamics Engine (`APGIDynamics` Class)

Implements the core computational model:

- **Surprise Accumulation**: Integrates interoceptive and exteroceptive prediction errors
  - Implements: dS/dt = -S/τ + f(Πₑ·|εₑ|, β·Πᵢ·|εᵢ|)
  - Tracks surprise (S) over time

- **Ignition Probability**: Computes probability of conscious perception
  - Implements: B(t) = σ(α(S(t) - θ(t)))
  - Sigmoidal function with gain parameter α

### 3. Synthetic Dataset Generation (`generate_synthetic_dataset` Function)

Creates realistic multimodal datasets with known ground truth parameters:

- **Task 1: Detection Psychometric Curve**
  - Varies stimulus intensity
  - Simulates binary detection responses

- **Task 2: Heartbeat Detection**
  - Generates HEP waveforms and pupil responses
  - Simulates behavioral d-prime for heartbeat detection

- **Task 3: Oddball P3b**
  - Simulates ERP responses to interoceptive and exteroceptive deviants
  - Captures neural signatures of prediction error processing

### 4. Hierarchical Bayesian Model (`build_apgi_model` Function)

Implements a probabilistic model in PyMC with:

- **Hierarchical Structure**:
  - Group-level hyperpriors for population parameters
  - Subject-level parameters with non-centered parameterization
  - Parameter constraints for biological plausibility

- **Likelihood Functions**:
  1. Detection psychometric curve (binomial likelihood)
  2. Heartbeat detection d-prime (normal likelihood)
  3. HEP amplitude (neural prior, normal likelihood)
  4. Pupil dilation response (normal likelihood)
  5. P3b ratio between intero/extero conditions (normal likelihood)

### 5. Validation Framework

#### Parameter Recovery (`validate_parameter_recovery`)

- Computes correlation between true and recovered parameters
- Calculates RMSE and credible interval coverage
- Implements threshold-based passing criteria for each parameter

#### Predictive Validity (`assess_predictive_validity`)

- Tests model predictions on held-out behavioral tasks:
  - Emotional Stroop interference
  - Continuous Performance Test (CPT) lapses
  - Body Vigilance Scale scores

#### Test-Retest Reliability (`assess_test_retest_reliability`)

- Simulates test-retest reliability over time
- Computes intraclass correlation coefficients (ICCs)
- Assesses stability of parameter estimates

## Key Features

1. **Multimodal Integration**: Combines behavioral, EEG, and pupillometry data
2. **Hierarchical Bayesian Modeling**: Accounts for both within- and between-subject variability
3. **Parameter Recovery**: Validates ability to accurately recover ground truth parameters
4. **Predictive Validity**: Tests generalizability to novel tasks
5. **Test-Retest Reliability**: Assesses stability of parameter estimates
6. **Computational Efficiency**: Vectorized operations for performance

## Dependencies

- Python 3.7+
- NumPy
- PyMC
- ArviZ
- SciPy
- Matplotlib
- scikit-learn

## Usage

1. Generate synthetic dataset:

   ```python
   subjects, true_params = generate_synthetic_dataset(n_subjects=100)
   ```

2. Build and sample from the model:

   ```python
   model = build_apgi_model(subjects)
   with model:
       trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)
   ```

3. Validate parameter recovery:

   ```python
   recovery_results, falsified = validate_parameter_recovery(true_params, trace, n_subjects=100)
   ```

4. Assess predictive validity:

   ```python
   validity_results = assess_predictive_validity(subjects, trace)
   ```

## Model Parameters

| Parameter | Description | Constraints |
| --- | --- | --- |
| θ₀ | Perceptual threshold | [0.25, 0.85] |
| Πᵢ | Interoceptive precision | [0.4, 2.8] |
| β | Interoceptive bias | [0.6, 2.3] |
| α | Gain parameter | [2.5, 7.5] |

## Visualization Templates for Recovered Parameter Distributions

To aid clinical interpretation of psychiatric profiles, the following visualization templates are provided for recovered parameter distributions:

### Violin Plot Template

```python
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# Create violin plots for clinical interpretation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# θ₀ (Threshold) - Clinical interpretation: barrier to conscious access
az.plot_violin(trace, var_names=['theta_0'], ax=axes[0, 0])
axes[0, 0].axvline(x=0.55, color='red', linestyle='--', label='Population mean')
axes[0, 0].set_title('θ₀: Perceptual Threshold\n(Higher = More difficult conscious access)')
axes[0, 0].set_xlabel('Threshold (arbitrary units)')

# Πᵢ (Interoceptive Precision) - Clinical interpretation: body awareness
az.plot_violin(trace, var_names=['Pi_i'], ax=axes[0, 1])
axes[0, 1].axvline(x=1.0, color='green', linestyle='--', label='Normative')
axes[0, 1].set_title('Πᵢ: Interoceptive Precision\n(Higher = Better body awareness)')
axes[0, 1].set_xlabel('Precision (1/variance)')

# β (Somatic Bias) - Clinical interpretation: emotional modulation
az.plot_violin(trace, var_names=['beta'], ax=axes[1, 0])
axes[1, 0].axvline(x=0.55, color='orange', linestyle='--', label='Typical anxiety')
axes[1, 0].axvline(x=0.35, color='blue', linestyle='--', label='Typical alexithymia')
axes[1, 0].set_title('β: Somatic Bias\n(Higher = Stronger emotional modulation)')
axes[1, 0].set_xlabel('Bias parameter')

# α (Gain) - Clinical interpretation: ignition sharpness
az.plot_violin(trace, var_names=['alpha'], ax=axes[1, 1])
axes[1, 1].axvline(x=5.0, color='purple', linestyle='--', label='Sharp transition')
axes[1, 1].set_title('α: Ignition Gain\n(Higher = Sharper conscious transition)')
axes[1, 1].set_xlabel('Sigmoid steepness')

plt.tight_layout()
plt.savefig('clinical_parameter_profile.png', dpi=300)
```

### Clinical Profile Interpretation Guide

| Parameter | Typical Range | Clinical Marker | Condition Implication |
| --------- | ------------- | --------------- | -------------------- |
| **θ₀ > 0.65** | Elevated threshold | Reduced conscious access | Depression, dissociation |
| **θ₀ < 0.45** | Lowered threshold | Hyper-accessibility | Mania, anxiety |
| **Πᵢ > 1.5** | High interoception | Enhanced body awareness | Alexithymia (protective) |
| **Πᵢ < 0.8** | Low interoception | Blunted body signals | Depression, somatization |
| **β > 0.65** | High somatic bias | Emotional over-influence | GAD, panic |
| **β < 0.40** | Low somatic bias | Emotional under-influence | Alexithymia, psychopathy |
| **α > 6.0** | Sharp ignition | All-or-none consciousness | PTSD (hypervigilance) |
| **α < 3.5** | Graded ignition | Fuzzy transitions | Brain fog, fatigue |

### Parameter Correlation Matrix Template

```python
# Show parameter interdependencies for differential diagnosis
def plot_parameter_correlation_matrix(trace):
    """Create correlation matrix for APGI parameters."""
    import pandas as pd

    # Extract posterior samples
    data = {
        'θ₀ (Threshold)': trace.posterior['theta_0'].values.flatten(),
        'Πᵢ (Precision)': trace.posterior['Pi_i'].values.flatten(),
        'β (Somatic Bias)': trace.posterior['beta'].values.flatten(),
        'α (Gain)': trace.posterior['alpha'].values.flatten()
    }
    df = pd.DataFrame(data)

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True)
    plt.title('APGI Parameter Intercorrelations\n(For Differential Diagnosis)')
    plt.tight_layout()
    plt.savefig('parameter_correlation_matrix.png', dpi=300)

plot_parameter_correlation_matrix(trace)
```

### Group Comparison Template (Patient vs Control)

```python
# Compare patient group to healthy controls
def plot_group_comparison(trace_patient, trace_control, param='theta_0'):
    """Violin plot comparing patient and control groups."""
    fig, ax = plt.subplots(figsize=(8, 5))

    patient_data = trace_patient.posterior[param].values.flatten()
    control_data = trace_control.posterior[param].values.flatten()

    data_to_plot = [control_data, patient_data]
    positions = [1, 2]

    parts = ax.violinplot(data_to_plot, positions, showmeans=True, showmedians=True)

    # Color code
    parts['bodies'][0].set_facecolor('lightgreen')  # Control
    parts['bodies'][1].set_facecolor('lightcoral')    # Patient

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Healthy Controls', 'Patient Group'])
    ax.set_ylabel(f'{param} Value')
    ax.set_title(f'Group Comparison: {param}')

    # Effect size annotation
    from scipy import stats
    d = (np.mean(patient_data) - np.mean(control_data)) / np.sqrt(
        (np.std(patient_data)**2 + np.std(control_data)**2) / 2)
    ax.text(0.5, 0.95, f'Cohen\'s d = {d:.2f}', transform=ax.transAxes,
            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    return fig
```

---

## Output

The script generates several outputs:

1. Parameter recovery plots (true vs. recovered values) with violin distributions
2. Posterior distributions for all parameters (clinical interpretation overlays)
3. Convergence diagnostics (R-hat, effective sample size)
4. Predictive validity metrics
5. Test-retest reliability statistics
