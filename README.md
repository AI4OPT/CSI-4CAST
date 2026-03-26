# CSI-4CAST: Channel State Information Forecasting

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

CSI-4CAST is a comprehensive framework for generating and evaluating Channel State Information (CSI) prediction models using 3GPP TR 38.901 channel models. The repository provides tools for large-scale dataset generation, model training, and comprehensive evaluation with support for both high-performance computing environments ([Phoenix HPC](https://pace.gatech.edu/phoenix-cluster/)) and direct execution on local machines.

This framework is developed as part of our research paper [**CSI-4CAST: A Hybrid Deep Learning Model for CSI Prediction with Comprehensive Robustness and Generalization Testing**](https://arxiv.org/abs/2510.12996). (A BibTeX entry for citation is provided at the end of this page.) The corresponding datasets are publicly available on our [Hugging Face organization](https://huggingface.co/CSI-4CAST).

## Updates

> [!IMPORTANT]
> - **Model weights are now public**: all model weights used in the experiments are available in [CSI-4CAST/weights](https://huggingface.co/CSI-4CAST/weights), including the proposed models and the baseline models.
> - **The tuning workflow is now included**: the reusable tuning infrastructure is in [`ahpt/`](ahpt/) and the CSI prediction tuning entry points are in [`src/cp/tune/`](src/cp/tune/).
> - **All ablation models are included**: the ablation study implementations are available under [`src/cp/models/ablation/`](src/cp/models/ablation/).
> - **All baselines are included**: the repository now contains statistical baselines in [`src/cp/models/baseline/statistical/`](src/cp/models/baseline/statistical/) and learning-based baselines in [`src/cp/models/baseline/learning/`](src/cp/models/baseline/learning/), together with the no-prediction baseline in [`src/cp/models/baseline/np.py`](src/cp/models/baseline/np.py).

## Repository Structure

```
CSI-4CAST/
├── README.md                         # Project documentation
├── LICENSE                           # License information
├── env.yml                           # Conda environment configuration
├── pyproject.toml                    # Project configuration and linting rules
├── ahpt/                             # Reusable hyperparameter tuning utilities
│   └── base/
│       ├── config.py                 # Tuning configuration dataclasses
│       ├── submit_jobs.py            # SLURM job submission helpers
│       ├── tune_obj.py               # Optuna objective base classes
│       ├── tune_runner.py            # Study orchestration utilities
│       ├── tune_space.py             # Search-space helpers
│       └── utils.py                  # Shared tuning utilities
├── scripts/                          # HPC job scripts and templates
│   ├── computational_overhead.sh
│   ├── cp_template.sh
│   ├── data_gen_template.sh
│   ├── nd.sh
│   ├── param_estimation.sh
│   ├── resync.sh
│   ├── testing.sh
│   └── outs/                         # Example job logs
├── src/                              # Source code
│   ├── data/                         # Data generation module
│   ├── cp/                           # Channel prediction and tuning module
│   │   ├── main.py                   # Training entry point
│   │   ├── config/                   # Training configuration management
│   │   ├── dataset/                  # Data modules
│   │   ├── loss/                     # Loss functions
│   │   ├── tune/                     # Tuning entry points and monitoring
│   │   │   ├── main.py
│   │   │   ├── monitor.py
│   │   │   ├── submit.py
│   │   │   ├── tune_obj.py
│   │   │   ├── tune_runner.py
│   │   │   └── worker.py
│   │   └── models/                   # Model registry and implementations
│   │       ├── __init__.py
│   │       ├── common/               # Shared model components
│   │       ├── proposed/             # Proposed CSI-4CAST models
│   │       ├── ablation/             # Ablation study variants
│   │       └── baseline/             # Baseline models
│   │           ├── np.py             # No-prediction baseline
│   │           ├── learning/         # CNN, RNN, STEMGNN, LLM4CP
│   │           └── statistical/      # AR, PAD, Wiener
│   ├── noise/                        # Noise modeling and calibration
│   ├── testing/                      # Evaluation and visualization
│   └── utils/                        # Shared utilities
└── z_artifacts/                      # Generated artifacts and outputs used by the codebase
    ├── config/                       # Generated configuration files
    ├── data/                         # Generated datasets and normalization stats
    ├── outputs/                      # Training, tuning, noise, and testing outputs
    └── weights/                      # Trained checkpoints organized by scenario/model
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
- **[`tune/`](src/cp/tune/)**: CSI prediction tuning workflow including SLURM submission, Optuna workers, and dashboard monitoring
- **[`models/`](src/cp/models/)**: Model architectures including:
  - **[`__init__.py`](src/cp/models/__init__.py)**: PREDICTORS registry for model selection
  - **[`common/base.py`](src/cp/models/common/base.py)**: BaseCSIModel class that all models inherit from
  - **[`proposed/`](src/cp/models/proposed/)**: Proposed CSI-4CAST models for FDD and TDD
  - **[`ablation/`](src/cp/models/ablation/)**: Ablation models for denoiser, ARL, IDFT, embedding, and predictor studies
  - **[`baseline/`](src/cp/models/baseline/)**: Baseline implementations including NP, AR, PAD, Wiener, CNN, RNN, STEMGNN, and LLM4CP
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

Models should be defined under [`src/cp/models`](src/cp/models), inherit from `BaseCSIModel` in [`src/cp/models/common/base.py`](src/cp/models/common/base.py), and be registered in the `PREDICTORS` class in [`src/cp/models/__init__.py`](src/cp/models/__init__.py). See [`src/cp/models/baseline/learning/rnn.py`](src/cp/models/baseline/learning/rnn.py) for an example implementation.

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

### 3. Hyperparameter Tuning

The hyperparameter tuning workflow combines the reusable tuning utilities in [`ahpt/`](ahpt/) with the CSI prediction-specific integration in [`src/cp/tune/`](src/cp/tune/). Each study is defined by a tuning YAML that points to a base training config, a search space module, and an output directory.

#### Example Tuning Config

Use [`src/cp/models/ablation/predictor/lstm_replace/tuning.yaml`](src/cp/models/ablation/predictor/lstm_replace/tuning.yaml) as a reference. This file defines:

- the base training config in [`config.yaml`](src/cp/models/ablation/predictor/lstm_replace/config.yaml)
- the target model `ABL_LSTM_REPLACE_PRED`
- the search space in `src.cp.models.ablation.predictor.lstm_replace.tune_space`
- Optuna study settings such as `n_trials`, pruning, and early stopping
- the output directory `z_artifacts/outputs/ahpt/ablation/predictor/lstm_replace`

#### Launch a Tuning Study

```bash
python3 -m src.cp.tune.main \
    --config src/cp/models/ablation/predictor/lstm_replace/tuning.yaml \
    --num_workers 3
```

You can also override SLURM resource settings at submission time:

```bash
python3 -m src.cp.tune.main \
    --config src/cp/models/ablation/predictor/lstm_replace/tuning.yaml \
    --num_workers 3 \
    --time 08:00:00 \
    --mem 64G
```

Each study writes its artifacts under `z_artifacts/outputs/ahpt/ablation/...`, including the Optuna database, best configs, and study history.

#### Monitor Tuning Results

To inspect the discovered studies without launching dashboards:

```bash
python3 -m src.cp.tune.monitor --list-only
```

To launch Optuna dashboards for the latest study databases:

```bash
python3 -m src.cp.tune.monitor --base-port 9000
```

The monitor scans `z_artifacts/outputs/ahpt/ablation/` for `study.db` files, keeps the latest run for each ablation, and serves one dashboard per study on consecutive ports.

### 4. Noise Degree Testing

Since realistic noise types cannot be directly defined by SNRs, calibrate noise parameters first:

```bash
python3 -m src.noise.noise_degree
```

Results are saved in `z_artifacts/outputs/noise/noise_degree/[date_time]/decide_nd.json` and copied to [`src/noise/results/decide_nd.json`](src/noise/results/decide_nd.json).

### 5. Model Testing

The model evaluation framework in [`src/testing`](src/testing) provides comprehensive assessment across multiple dimensions.

#### Configure Testing

Configure models and checkpoint paths in [`src/testing/config.py`](src/testing/config.py). Ensure checkpoints conform to the `get_ckpt_path` function in [`src/testing/get_models.py`](src/testing/get_models.py). Default checkpoint path: `z_artifacts/weights/[tdd/fdd]/[model_name]/model.ckpt`. The published experiment weights are also available from [CSI-4CAST/weights](https://huggingface.co/CSI-4CAST/weights).

#### Computational Overhead Testing

```bash
python3 -m src.testing.computational_overhead.main
```

Results saved in `z_artifacts/outputs/testing/computational_overhead/[date_time]/` for all configured models.

#### Prediction Performance Testing

For HPC clusters using SLURM array jobs, use [`scripts/testing.sh`](scripts/testing.sh) with array size matching `JOBS_PER_MODEL` in [`src/testing/config.py`](src/testing/config.py).

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
