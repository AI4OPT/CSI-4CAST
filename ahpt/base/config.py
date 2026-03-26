"""Configuration classes for hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning experiments."""

    seed: int = 42

    # Study settings
    study_name: str = "tuning"
    direction: Literal["minimize", "maximize"] = "minimize"
    is_U2D: bool = False

    # Checkpoint loading
    start_from_checkpoint: bool = False
    checkpoint_path: str | None = None

    # Optimization
    n_trials: int = 200
    timeout: float | None = None

    # Pruning
    enable_pruning: bool = True
    pruning_warmup_steps: int = 3
    pruning_warmup_trials: int = 10
    pruning_interval_steps: int = 1

    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5

    # Per-trial early stopping
    trial_early_stopping: bool = True
    trial_patience: int = 10
    trial_min_delta: float = 0.0001
    trial_min_epochs: int = 3
    trial_max_epochs: int = 5
    trial_monitor: str = "val_loss"
    trial_monitor_mode: str = "min"

    # Resource limits per trial
    max_trial_time: int | None = 21600

    # Target models
    target_models: list[str] = field(default_factory=list)

    # Search space
    tune_space_file: str = ""
    tune_model_params: bool = True
    tune_optimizer_params: bool = True
    tune_training_params: bool = True
    tune_loss_params: bool = False
    tune_scheduler_params: bool = False
    tune_data_params: bool = False
    tune_train_ratio: bool = False

    # Sampler
    sampler_n_startup_trials: int = 20

    # Output
    output_dir: str = ""
    save_best_configs: bool = True
    save_study_history: bool = True

    # Parallel execution
    n_jobs: int = 1

    # Logging
    verbose: bool = True
    log_level: str = "INFO"

    # Base experiment config
    base_config_path: str = ""

    @classmethod
    def from_dict(cls, dict_config: dict[str, Any]) -> TuningConfig:
        """Build a config from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict_config.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path_yaml: str) -> TuningConfig:
        """Load a config from a YAML file."""
        with open(path_yaml, encoding="utf-8") as f:
            dict_config = yaml.safe_load(f)
        return cls.from_dict(dict_config)

    @classmethod
    def from_json(cls, path_json: str) -> TuningConfig:
        """Load a config from a JSON file."""
        with open(path_json, encoding="utf-8") as f:
            dict_config = json.load(f)
        return cls.from_dict(dict_config)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a plain dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def save_yaml(self, path_yaml: str) -> None:
        """Write the config to a YAML file."""
        Path(path_yaml).parent.mkdir(parents=True, exist_ok=True)
        with open(path_yaml, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def save_json(self, path_json: str) -> None:
        """Write the config to a JSON file."""
        Path(path_json).parent.mkdir(parents=True, exist_ok=True)
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

@dataclass
class StudyResult:
    """Result of a hyperparameter tuning study."""

    study_name: str
    best_value: float
    best_params: dict[str, Any]
    best_trial_number: int
    n_trials: int
    best_config_path: str | None = None
    study_history_path: str | None = None
    top_k_config_paths: list[str] | None = None

    def save(self, output_path: str) -> None:
        """Persist the study summary to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "study_name": self.study_name,
                    "best_value": self.best_value,
                    "best_params": self.best_params,
                    "best_trial_number": self.best_trial_number,
                    "n_trials": self.n_trials,
                    "best_config_path": self.best_config_path,
                    "study_history_path": self.study_history_path,
                    "top_k_config_paths": self.top_k_config_paths,
                },
                f,
                indent=4,
            )

    @classmethod
    def load(cls, path: str) -> StudyResult:
        """Load a study summary from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
