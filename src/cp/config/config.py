import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

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

    name: str = "LLM4CP"
    is_separate_antennas: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training-related configuration."""

    num_epochs: int = 500
    gradient_clip_val: Optional[float] = None
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
    params: Dict[str, Any] = field(
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
    params: Dict[str, Any] = field(
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
    params: Dict[str, Any] = field(default_factory=dict)


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
    def fromDict(cls, dict_config: Dict[str, Any]) -> "ExperimentConfig":
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
        with open(path_json, "r") as f:
            dict_config = json.load(f)
        return cls.fromDict(dict_config)

    @classmethod
    def fromYaml(cls, path_yaml: str) -> "ExperimentConfig":
        """Load config from YAML file."""

        # Register a constructor for the python/tuple tag to convert to list
        def tuple_constructor(loader, node):
            return list(loader.construct_sequence(node))

        yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)

        with open(path_yaml, "r") as f:
            dict_config = yaml.safe_load(f)

        return cls.fromDict(dict_config)

    def toDict(self) -> Dict[str, Any]:
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

    from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, PRED_LEN, TOT_ANTENNAS

    parser = argparse.ArgumentParser(description="Experiment configuration management")
    parser.add_argument("--model", "-m", type=str, default="CNN", help="Model name")
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

    if model_name == "CNN":
        config.model.name = "CNN"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {}

    elif model_name == "LLM4CP":
        config.model.name = "LLM4CP"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {
            "gpt_type": "gpt2",
            "d_ff": 768,
            "d_model": 768,
            "gpt_layers": 6,
            "pred_len": 4,
            "prev_len": 16,
            "mlp": 0,
            "res_layers": 4,
            "K": 300,
            "UQh": 1,
            "UQv": 1,
            "BQh": 1,
            "BQv": 1,
            "patch_size": 4,
            "stride": 1,
            "res_dim": 64,
            "embed": "timeF",
            "freq": "h",
            "dropout": 0.1,
        }

    elif model_name == "MODEL":
        config.model.name = "MODEL"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {
            # data
            "hist_len": HIST_LEN,
            "pred_len": PRED_LEN,
            "dim_data": NUM_SUBCARRIERS * 2,
            "dim_model": 768,
            # denoiser
            "denoiser_num_filters_2d": 3,
            "denoiser_filter_size_2d": 3,
            "denoiser_filter_size_1d": 3,
            "denoiser_activation": "tanh",
            # ARL
            "arl_is_U2D": is_U2D,  # TDD as default
            "arl_temporal_proj_num_layers": 2,
            "arl_temporal_proj_hidden_dim": 256,
            "arl_temporal_proj_is_arl": False,
            "arl_temporal_proj_output_activation_name": "none",
            "arl_temporal_proj_arl_operation": "add",
            "arl_subcarrier_proj_num_layers": 2,
            "arl_subcarrier_proj_hidden_dim": 256,
            "arl_subcarrier_proj_is_arl": True,
            "arl_subcarrier_proj_output_activation_name": "sigmoid",
            "arl_subcarrier_proj_arl_operation": "add",
            # Shuffle Embedding
            "embedding_num_res_layers": 4,
            "embedding_res_dim": 64,
            "embedding_res_groups": 4,
            "embedding_embed": "timeF",
            "embedding_freq": "h",
            "embedding_dropout": 0.1,
            # TransformerPredictor
            "transformer_num_layers": 2,
            "transformer_num_heads": 4,
            "transformer_hidden_dim": 1024,
            "transformer_dropout_prob": 0.1,
        }

    elif model_name == "DeepAR":
        config.model.name = "DeepAR"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {
            "dim_data": NUM_SUBCARRIERS * 2,
            "encoder_hidden_dim": 1024,
            "encoder_num_layers": 2,
            "encoder_output_activation": "none",
            "lstm_hidden_dim": 1024,
            "lstm_num_layers": 4,
            "lstm_dropout": 0.1,
            "mean_layer_hidden_dim": 1024,
            "mean_layer_num_layers": 1,
            "mean_layer_output_activation": "none",
            "std_layer_hidden_dim": 1024,
            "std_layer_num_layers": 1,
            "std_layer_output_activation": "softplus",
            "pred_len": PRED_LEN,
            "is_sample": True,
        }

    elif model_name == "RNN":
        config.model.name = "RNN"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {
            "dim_data": NUM_SUBCARRIERS * 2,
            "rnn_hidden_dim": NUM_SUBCARRIERS * 4,
            "rnn_num_layers": 4,
            "pred_len": PRED_LEN,
        }

    elif model_name == "STEMGNN":
        config.model.name = "STEMGNN"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {
            "embedded_dim": NUM_SUBCARRIERS * 2,
            "window_size": HIST_LEN,
            "horizon": PRED_LEN,
            "n_stacks": 2,
            "multi_layer": 5,
            "dropout_rate": 0.5,
            "leaky_rate": 0.2,
        }

    elif model_name == "GRU":
        config.model.name = "GRU"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {
            "features": NUM_SUBCARRIERS * 2,
            "input_size": NUM_SUBCARRIERS * 2,
            "hidden_size": NUM_SUBCARRIERS * 4,
            "num_layers": 4,
            "pred_len": PRED_LEN,
        }

    elif model_name == "TFT":
        config.model.name = "TFT"
        config.model.is_separate_antennas = True
        config.model.checkpoint_path = None
        config.model.params = {
            "dim_data": NUM_SUBCARRIERS * 2,
            "hidden_dim": 256,
            "lstm_layers": 2,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "pred_len": PRED_LEN,
            "context_length": HIST_LEN,
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
