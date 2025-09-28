import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.utils.dirs import DIR_DATA, DIR_OUTPUTS


@dataclass
class DataConfig:
    """Data-related configuration."""

    dir_dataset: str = DIR_DATA
    batch_size: int = 16
    num_workers: int = 1
    is_U2D: bool = False
    shuffle: bool = True
    train_ratio: float = 0.9
    is_separate_antennas: bool = True


@dataclass
class ModelConfig:
    """Model-related configuration."""

    name: str = "RNN_TDD"
    is_separate_antennas: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass
class TrainingConfig:
    """Training-related configuration."""

    num_epochs: int = 500
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = 1
    check_val_every_n_epoch: int = 1

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0001
    early_stopping_mode: str = "min"

    # Model checkpoint
    save_top_k: int = 1
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"

    # Logging
    log_every_n_steps: int = 50
    enable_progress_bar: bool = True
    enable_model_summary: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

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
    """Learning rate scheduler configuration."""

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
    """Loss function configuration."""

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
    model_name: str = "CNN"
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
