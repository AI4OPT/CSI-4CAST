# CSI-4CAST: Channel State Information Forecasting

CSI-4CAST is a comprehensive framework for generating and evaluating Channel State Information (CSI) prediction models using 3GPP TR 38.901 channel models. The repository provides tools for large-scale dataset generation and model evaluation with support for both high-performance computing environments ([Phoenix HPC](https://pace.gatech.edu/phoenix-cluster/)) and direct execution on local machines.

## Repository Structure

```
CSI-4CAST/
├── README.md                    # Project documentation
├── LICENSE                      # License information
├── environment.yml              # Conda environment configuration
├── pyproject.toml              # Python project configuration
├── scripts/                    # SLURM job scripts
│   ├── data_gen.sh             # actual data generation script (#TODO: will be deleted in the final submission)
│   └── data_gen_template.sh    # Template for custom configurations
├── src/                        # Source code
│   ├── data/                   # Data generation module
│   │   ├── csi_simulator.py    # CSI simulation using Sionna
│   │   └── generator.py        # Dataset generation pipeline
│   ├── testing/                # Model evaluation module
│   └── utils/                  # Utility functions
│       ├── data_utils.py       # Constants and data handling utilities
│       ├── dirs.py             # Directory path management
│       └── norm_utils.py       # Data normalization utilities
└── z_artifacts/               # Generated artifacts and outputs
```

## Core Modules

### 1. Data Generation Module (`src/data`)

The data generation module provides a complete pipeline for creating realistic CSI datasets using 3GPP channel models.

#### Key Components:

- **`csi_simulator.py`**: Configures and implements the CSI simulator based on [Sionna's](https://github.com/NVlabs/sionna) 3GPP TR 38.901 channel model implementation. The simulator generates realistic channel responses for various propagation scenarios including different channel models, delay spreads, and mobility conditions.

- **`data_utils.py`**: Defines all simulation parameters and constants following the specifications detailed in the research paper. This includes antenna configurations, OFDM parameters, subcarrier arrangements, and dataset organization structures.

- **`generator.py`**: Employs the CSI simulator to generate comprehensive datasets including:
  - Training datasets for model development
  - Regular testing datasets for standard evaluation
  - Generalization testing datasets for robustness assessment

#### Dataset Generation

The generator creates three types of CSI data files for each channel configuration:
- `H_U_hist.pt`: Uplink historical CSI data (model input)
- `H_U_pred.pt`: Uplink prediction target CSI data
- `H_D_pred.pt`: Downlink prediction target CSI data (for cross-link scenarios)

**Data Dimensions:**
- **Antennas**: 32 (4×4×2 dual-polarized BS antenna array)
- **Time slots**: 20 total (16 historical + 4 prediction)
- **Subcarriers**: 300 each for uplink and downlink (750 total with gap)
- **Channel models**: A, C, D (regular) / A, B, C, D, E (generalization)
- **Delay spreads**: 30-400 nanoseconds
- **Mobility scenarios**: 1-45 m/s

### 2. Testing Module (`src/testing`)

The testing module provides evaluation frameworks for CSI prediction models (working in progress).

## Usage

The CSI-4CAST framework is designed to be flexible and compatible with various computing environments, from local development machines to large-scale HPC clusters.

### High-Performance Computing (Recommended for Large Datasets)

For generating complete datasets efficiently, the framework is optimized for high-performance computing environments like [Phoenix HPC](https://docs.pace.gatech.edu/phoenix_cluster/gettingstarted_phnx/) at Georgia Tech, with built-in support for parallel processing and distributed computation.

#### Using SLURM Job Arrays

A template script is provided in `scripts/data_gen_template.sh` for custom configurations. To generate datasets:

```bash
# Submit job to SLURM scheduler
sbatch scripts/data_gen.sh
```

The script supports various generation modes through command-line arguments:

```bash
python3 -m src.data.generator --is_train              # Generate training data
python3 -m src.data.generator                         # Generate regular test data
python3 -m src.data.generator --is_gen                # Generate generalization test data
```

**Array Size Recommendations:**
- 1-20 for generalization testing (510 parameter combinations)
- 1-9 for training and regular testing (27 parameter combinations each)
- Memory allocation should align with the batch size configuration in `src/utils/data_utils.py`

### Local Machine Execution (Fully Compatible)

The framework is fully compatible with local machines and can be executed directly without any HPC infrastructure. This makes it accessible for development, testing, and smaller-scale dataset generation:

```bash
# Local execution examples
python3 -m src.data.generator --debug --is_train      # Debug mode (recommended for local)
python3 -m src.data.generator --debug                 # Debug test dataset
python3 -m src.data.generator --debug --is_gen        # Debug generalization dataset
```

## Environment Setup

```bash
module load mamba/[mamba_version]
mamba env create -f env.yml
mamba activate csi-4cast-env
```

## Key Features

- **Cross-Platform Compatibility**: Runs seamlessly on local machines and HPC clusters
- **Scalable Generation**: Parallel processing support for large-scale dataset creation
- **Multiple Scenarios**: Comprehensive coverage of 3GPP channel models and mobility conditions
- **Memory Efficient**: Batch-wise processing with automatic garbage collection
- **Reproducible**: Deterministic generation with configurable random seeds
- **HPC Optimized**: Native SLURM integration for cluster computing
- **Debug Support**: Lightweight modes for development and testing
- **Flexible Deployment**: No infrastructure dependencies for basic usage

## Citation

If you use this framework in your research, please cite the corresponding paper:

```bibtex
[Citation information to be added]
```

## License

This project is licensed under the terms specified in the LICENSE file.