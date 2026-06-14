# zonal-mio-tcell-synapse
Code and synthetic outputs for zonal mechanics-information alignment simulations of T-cell immunological synapse organization.
# Zonal MIO T-cell Synapse Simulation

Code and synthetic outputs for zonal mechanics-information alignment simulations of T-cell immunological synapse organization.

## Overview

This repository contains the code, synthetic outputs, and figure files associated with a computational simulation of **zonal mechanics-information alignment** at the T-cell immunological synapse.

The simulation generates time-resolved synthetic synaptic fields over a registered circular contact domain partitioned into distal, peripheral, and central zones. For each synthetic synapse, three fields are generated:

* local signaling informativeness, (\tilde I(\mathbf r,t))
* mechanical permissivity, (\mu(\mathbf r,t))
* carrier density, (C(\mathbf r,t))

The analysis compares conventional amplitude-based summaries with zone-resolved alignment descriptors across four synthetic synaptic regimes:

1. productive activation
2. checkpoint-mediated suppression
3. unstable engagement
4. assembly failure

The purpose of the simulation is to test whether zonal alignment descriptors retain regime-level structure when gross amplitude summaries are weakly informative.

## Repository structure

```text
zonal-mio-tcell-synapse/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── zonal_mio_simulation.py
│
├── figures/
│   ├── main/
│   └── supplementary/
│
├── outputs/
│   ├── amplitude_overlap/
│   └── structured_regime_control/
│
└── docs/
```

## Main files

### Code

```text
zonal_mio_simulation.py
```

This script generates synthetic synaptic fields, extracts descriptor families, performs cross-validated model comparison, and exports figures and CSV outputs.

### Main figures

```text
figures/main/combined_model_comparison_AB.png
figures/main/combined_model_comparison_AB.pdf
figures/main/combined_alignment_trajectories_side_by_side_AB_spaced.png
figures/main/combined_alignment_trajectories_side_by_side_AB_spaced.pdf
figures/main/combined_spatial_fields_AB_spaced.png
figures/main/combined_spatial_fields_AB_spaced.pdf
```

### Supplementary figures

```text
figures/supplementary/03_feature_space_pca.png
figures/supplementary/03_feature_space_pca.pdf
figures/supplementary/05_confusion_matrices.png
figures/supplementary/05_confusion_matrices.pdf
figures/supplementary/01_spatial_montage.png
figures/supplementary/01_spatial_montage.pdf
```

### Output tables

Each simulation scenario contains:

```text
parameters.json
synthetic_features.csv
representative_timeseries.csv
model_comparison_folds.csv
model_comparison_summary.csv
incremental_value.csv
```

## Synthetic scenarios

### 1. Amplitude-overlap challenge

The amplitude-overlap scenario harmonizes gross signal amplitude, integrated signal, contact area, and total carrier mass across regimes. This limits classification based on total signal intensity or total carrier abundance.

This scenario tests whether zonal mechanics-information alignment retains regime-level information beyond gross amplitude summaries.

Code argument:

```bash
--scenario amplitude_overlap
```

### 2. Structured-regime control

The structured-regime control provides a secondary synthetic regime set for testing whether the qualitative descriptor ranking is preserved under a related synthetic construction.

In the code, this scenario is called:

```bash
--scenario separable
```

In the manuscript and figure captions, it is referred to as:

```text
structured-regime control
```

## Installation

The code was tested with Python 3.13.

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
pillow
```

## Reproducing the analysis

### Quick test

Use this command to check that the script runs correctly:

```bash
python zonal_mio_simulation.py --mode quick --scenario amplitude_overlap
```

### Standard amplitude-overlap analysis

```bash
python zonal_mio_simulation.py --mode standard --scenario amplitude_overlap
```

### Standard structured-regime control

```bash
python zonal_mio_simulation.py --mode standard --scenario separable
```

### Run both scenarios

```bash
python zonal_mio_simulation.py --mode standard --scenario both
```

## Descriptor families

The simulation compares four descriptor families.

### Amplitude-only descriptors

Gross signal summaries and contact area.

### Amplitude plus carrier-mass descriptors

Amplitude summaries plus total carrier mass.

### Alignment-only descriptors

Zone-resolved alignment means, temporal alignment variance, early-middle-late zonal alignment summaries, zonal contrast, alignment-order structure, carrier persistence, and inward carrier routing.

### Combined descriptors

Amplitude, carrier-mass, and alignment descriptors.

## Model comparison

The primary classifier is logistic regression. Random forest is included as a nonlinear sensitivity model.

Performance is evaluated using repeated stratified cross-validation and reported using:

* balanced accuracy
* macro-AUROC
* log loss

## Representative standard-run results

In the amplitude-overlap challenge, amplitude-based summaries were weakly discriminative, whereas alignment-based descriptors separated the synthetic regimes.

Representative logistic-regression macro-AUROC values:

| Descriptor family        | Macro-AUROC |
| ------------------------ | ----------- |
| Amplitude only           | 0.577       |
| Amplitude + carrier mass | 0.647       |
| Alignment only           | 1.000       |
| Combined                 | 1.000       |

In the structured-regime control, the same qualitative ranking was preserved:

| Descriptor family        | Macro-AUROC |
| ------------------------ | ----------- |
| Amplitude only           | 0.501       |
| Amplitude + carrier mass | 0.497       |
| Alignment only           | 1.000       |
| Combined                 | 1.000       |

## Interpretation

This repository provides a controlled synthetic analysis of descriptor behavior. It does not contain empirical live-cell imaging data and should not be interpreted as experimental validation of the biological framework.

The simulation tests whether the proposed zonal alignment descriptors preserve regime-level structure when gross amplitude summaries are weakly informative. Empirical validation will require synapse-resolved imaging data with paired measurements of signaling output, mechanical proxies, cytoskeletal organization, adhesion, and carrier dynamics.

## Data availability

No empirical human, animal, or patient-derived data are included in this repository. All fields are synthetically generated from declared parametric rules. Generated feature tables, model-comparison outputs, simulation parameters, and figures are included for reproducibility.

## Code availability

The Python code used to generate the synthetic synaptic regimes, extract amplitude and zonal alignment descriptors, perform cross-validated model comparison, and reproduce all computational figures is provided in this repository.

## Citation

If you use this code, synthetic simulation framework, or generated outputs, please cite the associated manuscript and this repository.

## License

Code is released under the MIT License.

Synthetic outputs and figures are released under CC BY 4.0 unless otherwise specified.
