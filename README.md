# CSI-4CAST: Channel State Information Forecasting

CSI-4CAST is a comprehensive framework for generating and evaluating Channel State Information (CSI) prediction models using 3GPP TR 38.901 channel models. The repository provides tools for large-scale dataset generation, model training, and comprehensive evaluation with support for both high-performance computing environments ([Phoenix HPC](https://pace.gatech.edu/phoenix-cluster/)) and direct execution on local machines.

## Repository Structure

```
CSI-4CAST/
├── README.md                    # Project documentation
├── LICENSE                      # License information
├── env.yml                      # Conda environment configuration
├── pyproject.toml              # Python project configuration and linting rules
├── scripts/                    # SLURM job scripts and templates
│   ├── data_gen_template.sh    # Template for data generation jobs
│   ├── cp.slurm               # Model training job script
│   ├── testing.slurm          # Model testing job script
│   ├── noise.slurm            # Noise degree testing script
│   └── computational_overhead.slurm  # Computational overhead testing
├── src/                        # Source code
│   ├── data/                   # Data generation module
│   │   ├── csi_simulator.py    # CSI simulation using Sionna
│   │   └── generator.py        # Dataset generation pipeline
│   ├── cp/                     # Channel Prediction (model training) module
│   │   ├── main.py            # Training entry point
│   │   ├── config/            # Training configuration management
│   │   │   └── config.py      # Configuration file generator
│   │   ├── dataset/           # PyTorch Lightning data modules
│   │   │   └── data_module.py # Data loading and preprocessing
│   │   ├── models/            # Model architectures
│   │   │   ├── __init__.py    # Model registry (PREDICTORS class)
│   │   │   ├── common/        # Shared model components
│   │   │   │   ├── base.py    # BaseCSIModel class
│   │   │   │   ├── activation.py # Custom activation functions
│   │   │   │   ├── dataembedding.py # Data embedding layers
│   │   │   │   ├── mlp.py     # Multi-layer perceptron components
│   │   │   │   ├── normalizer.py # Normalization layers
│   │   │   │   └── resblocks.py # Residual block components
│   │   │   ├── baseline_models/ # Baseline model implementations
│   │   │   │   ├── cnn.py     # CNN-based predictor
│   │   │   │   ├── llm4cp.py  # LLM-based predictor
│   │   │   │   ├── np.py      # No-prediction baseline
│   │   │   │   ├── rnn.py     # RNN-based predictor
│   │   │   │   └── stemgnn.py # STEM-GNN predictor
│   │   │   ├── model_fdd.py   # FDD-specific model architecture
│   │   │   └── model_tdd.py   # TDD-specific model architecture
│   │   └── loss/              # Loss functions
│   │       └── loss.py        # Custom loss implementations
│   ├── noise/                  # Noise modeling and testing module
│   │   ├── noise.py           # Noise generation functions
│   │   ├── noise_degree.py    # Noise parameter calibration
│   │   ├── noise_testing.py   # Noise testing utilities
│   │   └── results/           # Noise calibration results
│   │       ├── decide_nd.json # Noise degree mapping
│   │       └── snr.csv        # SNR measurement results
│   ├── testing/                # Model evaluation module
│   │   ├── config.py          # Testing configuration
│   │   ├── get_models.py      # Model loading utilities
│   │   ├── computational_overhead/ # Performance profiling
│   │   │   ├── main.py        # Computational overhead testing
│   │   │   └── utils.py       # Profiling utilities
│   │   ├── prediction_performance/ # Prediction accuracy testing
│   │   │   ├── main.py        # Performance testing entry point
│   │   │   └── test_unit.py   # Individual test units
│   │   ├── results/           # Result processing and analysis
│   │   │   ├── main.py        # Results processing pipeline
│   │   │   ├── analysis_df.py # Statistical analysis
│   │   │   ├── check_completion.py # Test completion verification
│   │   │   └── gather_results.py # Result aggregation
│   │   └── vis/               # Visualization module
│   │       ├── main.py        # Visualization entry point
│   │       ├── line.py        # Line plot generation
│   │       ├── radar.py       # Radar plot generation
│   │       ├── table.py       # Table generation
│   │       └── violin.py      # Violin plot generation
│   └── utils/                  # Utility functions
│       ├── data_utils.py      # Constants and data handling utilities
│       ├── dirs.py            # Directory path management
│       ├── norm_utils.py      # Data normalization utilities
│       ├── main_utils.py      # General utilities
│       ├── model_utils.py     # Model-related utilities
│       ├── real_n_complex.py  # Complex number handling
│       ├── time_utils.py      # Time formatting utilities
│       └── vis_utils.py       # Visualization utilities
└── z_artifacts/               # Generated artifacts and outputs
    ├── config/                # Generated configuration files
    ├── data/                  # Generated datasets
    ├── outputs/               # Training and testing outputs
    └── weights/               # Trained model checkpoints
```

## Core Modules

### 1. Data Generation Module (`src/data`)

The data generation module provides a complete pipeline for creating realistic CSI datasets using 3GPP channel models.

#### Key Components:

- **[`csi_simulator.py`](src/data/csi_simulator.py)**: Configures and implements the CSI simulator based on [Sionna's](https://github.com/NVlabs/sionna) 3GPP TR 38.901 channel model implementation. The simulator generates realistic channel responses for various propagation scenarios including different channel models, delay spreads, and mobility conditions.

- **[`data_utils.py`](src/utils/data_utils.py)**: Defines all simulation parameters and constants following the specifications detailed in the research paper. This includes antenna configurations, OFDM parameters, subcarrier arrangements, and dataset organization structures.

- **[`generator.py`](src/data/generator.py)**: Employs the CSI simulator to generate comprehensive datasets including:
  - Training datasets for model development
  - Regular testing datasets for standard and robustness evaluation
  - Generalization testing datasets for generalization evaluation

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

### 2. Channel Prediction Module (`src/cp`)

The channel prediction module provides a comprehensive framework for training CSI prediction models using PyTorch Lightning.

#### Key Components:

- **[`main.py`](src/cp/main.py)**: Training entry point that orchestrates the entire training process
- **[`config/config.py`](src/cp/config/config.py)**: Configuration management system for training parameters, model settings, and hyperparameters
- **[`dataset/data_module.py`](src/cp/dataset/data_module.py)**: PyTorch Lightning data modules for efficient data loading and preprocessing
- **[`models/`](src/cp/models/)**: Model architectures including:
  - **[`common/base.py`](src/cp/models/common/base.py)**: BaseCSIModel class that all models inherit from
  - **[`baseline_models/`](src/cp/models/baseline_models/)**: Implementation of various baseline models (CNN, RNN, LLM4CP, STEM-GNN)
  - **[`model_fdd.py`](src/cp/models/model_fdd.py)** and **[`model_tdd.py`](src/cp/models/model_tdd.py)**: Specialized architectures for FDD and TDD scenarios
- **[`loss/loss.py`](src/cp/loss/loss.py)**: Custom loss functions optimized for CSI prediction tasks

### 3. Noise Module (`src/noise`)

The noise module handles realistic noise modeling and parameter calibration for comprehensive testing scenarios.

#### Key Components:

- **[`noise.py`](src/noise/noise.py)**: Core noise generation functions implementing various realistic noise types
- **[`noise_degree.py`](src/noise/noise_degree.py)**: Noise parameter calibration system that maps target SNRs to appropriate noise parameters
- **[`noise_testing.py`](src/noise/noise_testing.py)**: Noise testing utilities and configurations
- **[`results/decide_nd.json`](src/noise/results/decide_nd.json)**: Pre-calibrated noise degree mapping for different noise types

### 4. Testing Module (`src/testing`)

The testing module provides comprehensive evaluation frameworks for CSI prediction models across multiple dimensions.

#### Key Components:

- **[`config.py`](src/testing/config.py)**: Testing configuration including model lists, scenarios, noise types, and job allocation settings
- **[`get_models.py`](src/testing/get_models.py)**: Model loading utilities with checkpoint path management
- **[`computational_overhead/`](src/testing/computational_overhead/)**: Performance profiling for measuring model computational requirements
- **[`prediction_performance/`](src/testing/prediction_performance/)**: Accuracy evaluation across thousands of testing scenarios
- **[`results/`](src/testing/results/)**: Result processing pipeline including completion checking, data aggregation, and statistical analysis
- **[`vis/`](src/testing/vis/)**: Comprehensive visualization suite generating line plots, radar charts, violin plots, and tables

## Usage Guide

The CSI-4CAST framework is designed to be flexible and compatible with various computing environments, from local development machines to large-scale HPC clusters.

### Environment Setup

```bash
module load mamba/[mamba_version]
mamba env create -f env.yml
mamba activate csi-4cast-env
```

### 1. Data Generation

The code related to data generation is in the [`src/data`](src/data) folder and [`src/utils/data_utils.py`](src/utils/data_utils.py) file.

#### Define Constants

The [`data_utils.py`](src/utils/data_utils.py) file defines all constants which configure the Sionna simulator and data generation process. It is critical to understand and adjust these constants based on your setting before running any code.

#### Generate Data

For high-performance computing, use the template in [`scripts/data_gen_template.sh`](scripts/data_gen_template.sh):

```bash
python3 -m src.data.generator --is_train              # Generate training data, typical array size is 1-9
python3 -m src.data.generator                         # Generate regular test data, typical array size is 1
python3 -m src.data.generator --is_gen                # Generate generalization test data, typical array size is 1-20
```

For local/single-node execution, use debug mode for minimal datasets:

```bash
python3 -m src.data.generator --debug --is_train      # Debug mode: minimal training data
python3 -m src.data.generator --debug                 # Debug mode: minimal test data
python3 -m src.data.generator --debug --is_gen        # Debug mode: minimal generalization data
```

#### Obtain Normalization Stats

After data generation, compute normalization statistics using [`src/utils/norm_utils.py`](src/utils/norm_utils.py):

```bash
python3 -m src.utils.norm_utils
```

The normalization stats will be saved in `z_refer/data/stats/[fdd/tdd]/normalization_stats.pkl`.

### 2. Model Training

The model training framework is built on PyTorch Lightning and located in the [`src/cp`](src/cp) folder.

#### Define Models

Models should be defined under [`src/cp/models`](src/cp/models) folder, inherit from `BaseCSIModel` in [`src/cp/models/common/base.py`](src/cp/models/common/base.py), and be registered in the `PREDICTORS` class in [`src/cp/models/__init__.py`](src/cp/models/__init__.py). See [`src/cp/models/baseline_models/rnn.py`](src/cp/models/baseline_models/rnn.py) for an example implementation.

#### Configure Training

Configure the training process in [`src/cp/config/config.py`](src/cp/config/config.py), then generate configuration files:

```bash
python3 -m src.cp.config.config --model [model_name] --output-dir [output_dir] --is_U2D [True/False] --config-file [yaml/json]
```

Default output directory: `z_artifacts/config/cp/[model_name]/`

#### Train Models

```bash
python3 -m src.cp.main --hparams_csi_pred [config_file]
```

For HPC clusters, use [`scripts/cp.slurm`](scripts/cp.slurm). Training outputs are saved in `z_artifacts/outputs/[TDD/FDD]/[model_name]/[date_time]/` with checkpoints in `ckpts/` and TensorBoard logs in `tb_logs/`.

View training progress:
```bash
tensorboard --logdir [output_directory]/tb_logs
```

### 3. Noise Degree Testing

Since realistic noise types cannot be directly defined by SNRs, calibrate noise parameters first:

```bash
python3 -m src.noise.noise_degree
```

Results are saved in `z_artifacts/outputs/noise/noise_degree/[date_time]/decide_nd.json` and copied to [`src/noise/results/decide_nd.json`](src/noise/results/decide_nd.json).

### 4. Model Testing

The model evaluation framework in [`src/testing`](src/testing) provides comprehensive assessment across multiple dimensions.

#### Configure Testing

Configure models and checkpoint paths in [`src/testing/config.py`](src/testing/config.py). Ensure checkpoints conform to the `get_ckpt_path` function in [`src/testing/get_models.py`](src/testing/get_models.py). Default checkpoint path: `z_artifacts/weights/[tdd/fdd]/[model_name]/model.ckpt`.

#### Computational Overhead Testing

```bash
python3 -m src.testing.computational_overhead.main
```

Results saved in `z_artifacts/outputs/testing/computational_overhead/[date_time]/` for all configured models.

#### Prediction Performance Testing

For HPC clusters using SLURM array jobs (recommended), use [`scripts/testing.slurm`](scripts/testing.slurm) with array size matching `JOBS_PER_MODEL` in [`src/testing/config.py`](src/testing/config.py).

For local execution:
```bash
python3 -m src.testing.prediction_performance.main --model [model_name]
```

Results saved in `z_artifacts/outputs/testing/prediction_performance/[model_name]/[full_test/slice_i]/[date_time]/`.

#### Results Processing

Process all testing results with comprehensive analysis:

```bash
python3 -m src.testing.results.main
```

This performs three steps:
1. Check completion status of testing models
2. Gather and aggregate all results into CSV files
3. Post-process results for scenario-wise distributions based on NMSE and SE metrics

Results saved in:
- `z_artifacts/outputs/testing/results/completion_reports/[date_time]/`
- `z_artifacts/outputs/testing/results/gather/[date_time]/`
- `z_artifacts/outputs/testing/results/analysis/[nmse/se]/[date_time]/`

#### Visualization

Generate comprehensive visualizations (line plots, radar plots, violin plots, tables):

```bash
python3 -m src.testing.vis.main
```

Results saved in `z_artifacts/outputs/testing/vis/[date_time]/[line/radar/violin/table]/`.


## Citation

If you use this framework in your research, please cite the corresponding paper:

```bibtex
[Citation information to be added]
```

## License

This project is licensed under the terms specified in the LICENSE file.