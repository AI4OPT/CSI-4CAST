"""Configuration Management for CSI Prediction Experiments.

This module provides a comprehensive configuration system for CSI prediction experiments
using dataclasses. It supports hierarchical configuration with separate classes for
different aspects of the experiment (data, model, training, optimization, etc.).

The configuration system supports:
- JSON and YAML serialization/deserialization
- Type hints and validation
- Default values for all parameters
- Automatic experiment naming and output directory creation
- Model-specific parameter configuration

Key Configuration Classes:
- DataConfig: Data loading and preprocessing settings
- ModelConfig: Model architecture and parameters
- TrainingConfig: Training procedure settings (epochs, callbacks, etc.)
- OptimizerConfig: Optimizer settings (Adam, SGD, etc.)
- SchedulerConfig: Learning rate scheduler settings
- LossConfig: Loss function configuration
- ExperimentConfig: Top-level configuration combining all components

Usage:
    # Create default configuration
    config = ExperimentConfig()

    # Load from file
    config = ExperimentConfig.fromYaml('config.yaml')
    config = ExperimentConfig.fromJson('config.json')

    # Save configuration
    config.saveYaml('output_config.yaml')
    config.saveJson('output_config.json')

    # Command line usage
    python3 -m src.cp.config.config --model RNN --is_U2D
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.utils.dirs import DIR_DATA, DIR_OUTPUTS


@dataclass
class DataConfig:
    """Data-related configuration for CSI prediction experiments.

    This class contains all parameters related to data loading, preprocessing,
    and data loader configuration.

    Attributes:
        dir_dataset (str): Path to the dataset directory containing CSI data files
        batch_size (int): Batch size for training and validation data loaders
        num_workers (int): Number of worker processes for data loading
        is_U2D (bool): Whether to use U2D (FDD) scenario (True) or TDD scenario (False)
        shuffle (bool): Whether to shuffle training data
        train_ratio (float): Ratio of data to use for training (0.0-1.0)
        is_separate_antennas (bool): Critical parameter controlling antenna processing strategy

    Antenna Processing Strategy (is_separate_antennas):
        This parameter determines how multi-antenna CSI data is processed during training:

        When True (Separate Antennas):
            - Each antenna's CSI is treated as an independent sample
            - Data shape transformation: [batch, antennas, time, freq] → [batch*antennas, time, freq*2]
            - Complex values converted to real [real, imag] representation
            - Effective batch size increases by factor of num_antennas
            - Use case: When antennas have independent channel characteristics
            - Data preprocessing: collect_fn_separate_antennas() in src.utils.data_utils
            - Memory efficient for models that don't need spatial correlation

        When False (Gather Antennas):
            - All antennas processed together maintaining spatial structure
            - Data shape preserved: [batch, antennas, time, freq] (complex-valued)
            - Preserves spatial correlation between antennas
            - Use case: When leveraging antenna array spatial properties
            - Data preprocessing: collect_fn_gather_antennas() in src.utils.data_utils
            - Required for models that exploit spatial diversity

        Note: This parameter must be consistent between DataConfig and ModelConfig
              to ensure proper data flow through the training pipeline.

    """

    dir_dataset: str = DIR_DATA
    batch_size: int = 16
    num_workers: int = 1
    is_U2D: bool = False
    shuffle: bool = True
    train_ratio: float = 0.9
    is_separate_antennas: bool = True


@dataclass
class ModelConfig:
    """Model-related configuration for CSI prediction experiments.

    This class contains parameters for model architecture, initialization,
    and checkpoint management.

    Attributes:
        name (str): Model name (e.g., "RNN_TDD", "CNN_FDD")
        is_separate_antennas (bool): Must match DataConfig.is_separate_antennas for consistency
        params (dict): Model-specific parameters (hidden dims, layers, etc.)
        checkpoint_path (str, optional): Path to checkpoint file for resuming training

    Model-Data Consistency:
        The is_separate_antennas parameter must match the corresponding parameter in DataConfig.
        This ensures that the data preprocessing strategy aligns with the model's expected input format:

        - If True: Model expects input shape [batch*antennas, time, features*2] (real-valued)
        - If False: Model expects input shape [batch, antennas, time, features] (complex-valued)

        Mismatch between DataConfig and ModelConfig will result in shape incompatibility errors.

    """

    name: str = "RNN_TDD"
    is_separate_antennas: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass
class TrainingConfig:
    """Training-related configuration for CSI prediction experiments.

    This class contains all parameters related to the training process including
    epochs, callbacks, monitoring, and logging settings.

    Attributes:
        num_epochs (int): Maximum number of training epochs
        gradient_clip_val (float, optional): Gradient clipping value for stability
        accumulate_grad_batches (int): Number of batches for gradient accumulation
        check_val_every_n_epoch (int): Frequency of validation checks

        Early Stopping:
            early_stopping (bool): Whether to enable early stopping
            early_stopping_patience (int): Number of epochs to wait before stopping
            early_stopping_min_delta (float): Minimum change to qualify as improvement
            early_stopping_mode (str): "min" or "max" for monitoring metric

        Model Checkpointing:
            save_top_k (int): Number of best checkpoints to save
            monitor_metric (str): Metric to monitor for checkpointing
            monitor_mode (str): "min" or "max" for monitoring metric

        Logging:
            log_every_n_steps (int): Frequency of logging during training
            enable_progress_bar (bool): Whether to show progress bars
            enable_model_summary (bool): Whether to show model summary

    """

    num_epochs: int = 500
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = 1
    check_val_every_n_epoch: int = 1

    # Early stopping parameters
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0001
    early_stopping_mode: str = "min"

    # Model checkpoint parameters
    save_top_k: int = 1
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"

    # Logging parameters
    log_every_n_steps: int = 50
    enable_progress_bar: bool = True
    enable_model_summary: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration for CSI prediction experiments.

    This class contains parameters for the optimization algorithm used during training.
    Supports various PyTorch optimizers (Adam, SGD, AdamW, etc.).

    Attributes:
        name (str): Name of the optimizer class from torch.optim
        params (dict): Optimizer-specific parameters (learning rate, weight decay, etc.)
            Default Adam parameters:
                - lr: Learning rate (0.0005)
                - weight_decay: L2 regularization coefficient (0.0001)
                - eps: Term added to denominator for numerical stability (1e-08)
                - betas: Coefficients for computing running averages ([0.9, 0.999])

    """

    name: str = "Adam"
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "eps": 1e-08,
            "betas": [0.9, 0.999],
        }
    )


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration for CSI prediction experiments.

    This class contains parameters for the learning rate scheduling strategy.
    Supports various PyTorch schedulers (ReduceLROnPlateau, StepLR, CosineAnnealingLR, etc.).

    Attributes:
        name (str): Name of the scheduler class from torch.optim.lr_scheduler
        params (dict): Scheduler-specific parameters
            Default ReduceLROnPlateau parameters:
                - mode: "min" for metrics where lower is better
                - factor: Factor by which learning rate is reduced (0.1)
                - patience: Number of epochs with no improvement before reducing (10)
                - threshold: Threshold for measuring new optimum (0.0001)
                - cooldown: Number of epochs to wait before resuming operation (5)
                - min_lr: Minimum learning rate (1e-06)

    """

    name: str = "ReduceLROnPlateau"
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "threshold": 0.0001,
            "cooldown": 5,
            "min_lr": 1e-06,
        }
    )


@dataclass
class LossConfig:
    """Loss function configuration for CSI prediction experiments.

    This class specifies which loss function to use and its parameters.
    Supports various loss functions defined in src.cp.loss.loss module.

    Attributes:
        name (str): Name of the loss function (NMSE, MSE, Huber, etc.)
        params (dict): Loss-specific parameters (empty dict for most losses)

    Examples:
                - For Huber loss: {"beta": 1.0}
                - For NMSE/MSE: {} (no parameters needed)

    """

    name: str = "NMSE"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    seed: int = 42
    deterministic: bool = False

    # Hardware
    accelerator: str = "auto"  # 'gpu', 'cpu', 'tpu', 'auto'
    devices: int = 1
    precision: Literal["16-mixed", "bf16", "32", "64"] = "32"  # '16', '32', 'bf16'

    # experiment metadata
    prefix: str = "TDD"
    model_name: str = "RNN"
    experiment_name: str = field(default="")
    output_dir: str = field(default="")

    def __post_init__(self):
        """Set default values for experiment_name and output_dir if they're empty."""
        if not self.experiment_name:
            self.experiment_name = f"{self.prefix}_{self.model_name}"
        if not self.output_dir:
            self.output_dir = str(Path(DIR_OUTPUTS) / self.prefix / self.model_name)

    @classmethod
    def fromDict(cls, dict_config: dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        # Handle nested configurations
        if "data" in dict_config and isinstance(dict_config["data"], dict):
            dict_config["data"] = DataConfig(**dict_config["data"])
        if "model" in dict_config and isinstance(dict_config["model"], dict):
            dict_config["model"] = ModelConfig(**dict_config["model"])
        if "training" in dict_config and isinstance(dict_config["training"], dict):
            dict_config["training"] = TrainingConfig(**dict_config["training"])
        if "optimizer" in dict_config and isinstance(dict_config["optimizer"], dict):
            dict_config["optimizer"] = OptimizerConfig(**dict_config["optimizer"])
        if "scheduler" in dict_config and isinstance(dict_config["scheduler"], dict):
            dict_config["scheduler"] = SchedulerConfig(**dict_config["scheduler"])
        if "loss" in dict_config and isinstance(dict_config["loss"], dict):
            dict_config["loss"] = LossConfig(**dict_config["loss"])

        return cls(**dict_config)

    @classmethod
    def fromJson(cls, path_json: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path_json) as f:
            dict_config = json.load(f)
        return cls.fromDict(dict_config)

    @classmethod
    def fromYaml(cls, path_yaml: str) -> "ExperimentConfig":
        """Load config from YAML file."""

        # Register a constructor for the python/tuple tag to convert to list
        def tuple_constructor(loader, node):
            return list(loader.construct_sequence(node))

        yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)

        with open(path_yaml) as f:
            dict_config = yaml.safe_load(f)

        return cls.fromDict(dict_config)

    def toDict(self) -> dict[str, Any]:
        """Convert config to dictionary."""

        def _toDict(obj) -> Any:
            if hasattr(obj, "__dict__"):
                return {k: _toDict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [_toDict(item) for item in obj]
            else:
                return obj

        return _toDict(self)

    def saveJson(self, path_json: str):
        """Save config to JSON file."""
        Path(path_json).parent.mkdir(parents=True, exist_ok=True)
        with open(path_json, "w") as f:
            json.dump(self.toDict(), f, indent=4)

    def saveYaml(self, path_yaml: str):
        """Save config to YAML file."""
        Path(path_yaml).parent.mkdir(parents=True, exist_ok=True)
        with open(path_yaml, "w") as f:
            yaml.dump(self.toDict(), f, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.utils.data_utils import NUM_SUBCARRIERS, PRED_LEN

    parser = argparse.ArgumentParser(description="Experiment configuration management")
    parser.add_argument("--model", "-m", type=str, default="RNN", help="Model name")
    parser.add_argument(
        "--output-dir", "-o", type=str, default="z_artifacts/config/cp", help="Output directory for config files"
    )
    parser.add_argument("--is_U2D", "-u", action="store_true", help="Use U2D (FDD) configuration")
    parser.add_argument("--config-file", "-c", type=str, default="yaml", help="Path to config file (JSON or YAML)")
    args = parser.parse_args()

    config = ExperimentConfig()

    # U2D
    is_U2D = args.is_U2D
    scenario = "FDD" if is_U2D else "TDD"
    config.data.is_U2D = is_U2D

    # model configuration
    model_name = args.model

    if model_name == "RNN":
        config.model.name = f"{model_name}_{scenario}"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {
            "dim_data": NUM_SUBCARRIERS * 2,
            "rnn_hidden_dim": NUM_SUBCARRIERS * 4,
            "rnn_num_layers": 4,
            "pred_len": PRED_LEN,
        }

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # make sure the data and model configurations are consistent
    config.data.is_separate_antennas = config.model.is_separate_antennas

    # set experiment metadata
    config.prefix = scenario
    config.model_name = model_name
    config.experiment_name = f"{config.prefix}_{config.model_name}"
    config.output_dir = str(Path(DIR_OUTPUTS) / config.prefix / config.model_name)

    # save the config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.config_file == "yaml":
        config.saveYaml(str(output_dir / config.model_name.lower() / f"{config.experiment_name.lower()}.yaml"))
    elif args.config_file == "json":
        config.saveJson(str(output_dir / config.model_name.lower() / f"{config.experiment_name.lower()}.json"))
    else:
        raise ValueError(f"Unsupported config file format: {args.config_file}. Use 'yaml' or 'json'.")
