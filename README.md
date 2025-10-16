# CSI-4CAST: Channel State Information Forecasting

CSI-4CAST is a comprehensive framework for generating and evaluating Channel State Information (CSI) prediction models using 3GPP TR 38.901 channel models. The repository provides tools for large-scale dataset generation, model training, and comprehensive evaluation with support for both high-performance computing environments ([Phoenix HPC](https://pace.gatech.edu/phoenix-cluster/)) and direct execution on local machines.

This framework is developed as part of our research paper [**CSI-4CAST: A Hybrid Deep Learning Model for CSI Prediction with Comprehensive Robustness and Generalization Testing**](https://arxiv.org/abs/2510.12996).  (A BibTeX entry for citation is provided at the end of this page.) The corresponding datasets are publicly available on our [Hugging Face Dataset](https://huggingface.co/CSI-4CAST).

## Repository Structure

```
CSI-4CAST/
├── README.md                    # Project documentation
├── LICENSE                      # License information
├── env.yml                      # Conda environment configuration
├── pyproject.toml              # Python project configuration and linting rules
├── scripts/                    # SLURM job scripts and templates
│   ├── data_gen_template.sh    # Template for data generation jobs
│   ├── cp_template.sh         # Template for model training jobs
│   ├── testing_template.sh    # Template for testing jobs
│   └── outs/                  # Job output logs
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
│   │   │   │   └── base.py    # BaseCSIModel class
│   │   │   └── baseline_models/ # Baseline model implementations
│   │   │       ├── np.py      # No-prediction baseline
│   │   │       └── rnn.py     # RNN-based predictor
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
    │   └── cp/                # Channel prediction configurations
    ├── data/                  # Generated datasets (created during data generation)
    ├── outputs/               # Training and testing outputs
    │   ├── [TDD/FDD]/         # Training outputs by scenario
    │   ├── noise/             # Noise calibration results
    │   └── testing/           # Testing results and analysis
    │       ├── computational_overhead/ # Performance profiling results
    │       ├── prediction_performance/ # Accuracy testing results
    │       ├── results/       # Processed analysis results
    │       └── vis/           # Generated visualizations
    └── weights/               # Trained model checkpoints
        ├── fdd/               # FDD scenario model weights
        └── tdd/               # TDD scenario model weights
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
  - **[`__init__.py`](src/cp/models/__init__.py)**: PREDICTORS registry for model selection
  - **[`common/base.py`](src/cp/models/common/base.py)**: BaseCSIModel class that all models inherit from
  - **[`baseline_models/`](src/cp/models/baseline_models/)**: Implementation of baseline models (NP, RNN)
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

The normalization stats will be saved in `z_artifacts/data/stats/[fdd/tdd]/normalization_stats.pkl`.

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

For HPC clusters, use [`scripts/cp_template.sh`](scripts/cp_template.sh). Training outputs are saved in `z_artifacts/outputs/[TDD/FDD]/[model_name]/[date_time]/` with checkpoints in `ckpts/` and TensorBoard logs in `tb_logs/`.

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

For HPC clusters using SLURM array jobs (recommended), use [`scripts/testing.slurm`](scripts/testing.slurm) or [`scripts/testing_template.sh`](scripts/testing_template.sh) with array size matching `JOBS_PER_MODEL` in [`src/testing/config.py`](src/testing/config.py).

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


## Sample Outputs

To better illustrate the usage of the framework, sample outputs are provided in the [`z_artifacts/`](z_artifacts/) directory. These examples demonstrate the complete workflow from configuration to final visualization results.

### Configuration Files

- **[`config/cp/rnn/`](z_artifacts/config/cp/rnn/)**: Sample configuration files for RNN model training
  - [`fdd_rnn.yaml`](z_artifacts/config/cp/rnn/fdd_rnn.yaml): FDD scenario RNN configuration
  - [`tdd_rnn.yaml`](z_artifacts/config/cp/rnn/tdd_rnn.yaml): TDD scenario RNN configuration

### Noise Calibration Results

- **[`noise/noise_degree/`](z_artifacts/outputs/noise/noise_degree/)**: Noise parameter calibration outputs
  - [`decide_nd.json`](z_artifacts/outputs/noise/noise_degree/20250928_155913/decide_nd.json): Calibrated noise degree mappings for different noise types
  - [`snr.csv`](z_artifacts/outputs/noise/noise_degree/20250928_155913/snr.csv): SNR measurement results across noise parameters

### Model Training Results

- **[`TDD/RNN/`](z_artifacts/outputs/TDD/RNN/)**: Sample training output for RNN model in TDD scenario
  - [`config_copy.yaml`](z_artifacts/outputs/TDD/RNN/2025-09-28_15-07-23/config_copy.yaml): Training configuration backup
  - [`tb_logs/`](z_artifacts/outputs/TDD/RNN/2025-09-28_15-07-23/tb_logs/): TensorBoard logs for training monitoring

### Testing Performance Results

The [`testing/`](z_artifacts/outputs/testing/) directory contains comprehensive evaluation results for both NP baseline and RNN models:

#### Raw Testing Data
- **[`computational_overhead/`](z_artifacts/outputs/testing/computational_overhead/)**: Performance profiling results
  - [`computational_overhead.csv`](z_artifacts/outputs/testing/computational_overhead/20250928_164946/computational_overhead.csv): FLOPs, inference time, and parameter counts

- **[`prediction_performance/`](z_artifacts/outputs/testing/prediction_performance/)**: Prediction accuracy results
  - **[`NP/full_test/`](z_artifacts/outputs/testing/prediction_performance/NP/full_test/)**: NP baseline results obtained via local execution mode
  - **[`RNN/slice_*/`](z_artifacts/outputs/testing/prediction_performance/RNN/)**: RNN results obtained via SLURM job slices (20 parallel jobs)

#### Processed Analysis Results
- **[`results/`](z_artifacts/outputs/testing/results/)**: Consolidated and analyzed testing data
  - [`completion_reports/`](z_artifacts/outputs/testing/results/completion_reports/): Testing completion status verification
  - [`gather/`](z_artifacts/outputs/testing/results/gather/): Consolidated raw results from all models and slices
  - [`analysis/`](z_artifacts/outputs/testing/results/analysis/): Statistical analysis with rankings and distributions
    - [`nmse/`](z_artifacts/outputs/testing/results/analysis/nmse/): NMSE-based performance analysis
    - [`se/`](z_artifacts/outputs/testing/results/analysis/se/): Spectral efficiency-based performance analysis

#### Visualization Results
- **[`vis/`](z_artifacts/outputs/testing/vis/)**: Comprehensive visualization suite
  - [`line/`](z_artifacts/outputs/testing/vis/20250928_194940/line/): Line plots showing performance across different conditions
    - [`generalization/`](z_artifacts/outputs/testing/vis/20250928_194940/line/generalization/): Out-of-distribution performance
    - [`regular/`](z_artifacts/outputs/testing/vis/20250928_194940/line/regular/): In-distribution performance  
    - [`robustness/`](z_artifacts/outputs/testing/vis/20250928_194940/line/robustness/): Performance under noise conditions
  - [`radar/`](z_artifacts/outputs/testing/vis/20250928_194940/radar/): Multi-dimensional performance comparison
    - [`combined_radar_fdd.pdf`](z_artifacts/outputs/testing/vis/20250928_194940/radar/combined_radar_fdd.pdf): FDD scenario radar plot
    - [`combined_radar_tdd.pdf`](z_artifacts/outputs/testing/vis/20250928_194940/radar/combined_radar_tdd.pdf): TDD scenario radar plot
  - [`table/`](z_artifacts/outputs/testing/vis/20250928_194940/table/): Performance summary tables by channel model and delay spread
  - [`violin/`](z_artifacts/outputs/testing/vis/20250928_194940/violin/): Distribution analysis across scenarios

### Key Insights from Sample Results

The provided sample outputs demonstrate:
- **Execution Modes**: NP baseline uses local full_test mode while RNN uses distributed SLURM slices. The current testing framework supports both modes.
- **Comprehensive Evaluation**: Testing covers regular, robustness, and generalization scenarios.
- **Multi-Metric Analysis**: Both NMSE and spectral efficiency (SE) metrics are evaluated.
- **Rich Visualizations**: Multiple plot types provide different perspectives on model performance.
- **Scalable Framework**: The structure supports easy extension to additional models and scenarios.

**For more comprehensive results and detailed analysis, please refer to the corresponding research paper.**


## Citation

If you use this framework in your research, please cite the corresponding paper:

```bibtex
@misc{cheng2025csi4casthybriddeeplearning,
      title={CSI-4CAST: A Hybrid Deep Learning Model for CSI Prediction with Comprehensive Robustness and Generalization Testing}, 
      author={Sikai Cheng and Reza Zandehshahvar and Haoruo Zhao and Daniel A. Garcia-Ulloa and Alejandro Villena-Rodriguez and Carles Navarro Manchón and Pascal Van Hentenryck},
      year={2025},
      eprint={2510.12996},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.12996}, 
}
```

## License

This project is licensed under the terms specified in the LICENSE file.
