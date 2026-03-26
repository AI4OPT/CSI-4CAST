"""Concrete CP tuning objective — bridges ahpt base classes with CP config/models."""

import importlib

from ahpt.base.tune_obj import BaseTuningObjective
import torch

from src.cp.config.config import ExperimentConfig
from src.cp.dataset.data_module import TrainValDataModule
from src.cp.models import PREDICTORS


class CPTuningObjective(BaseTuningObjective):
    """Optuna objective for CSI prediction tuning trials."""

    def _setup_base_config(self):
        return ExperimentConfig.from_yaml(self.tuning_config.base_config_path)

    def _setup_hyperparameter_space(self):
        module = importlib.import_module(self.tuning_config.tune_space_file)
        return module.HyperparameterSpace()

    def _setup_data_module(self, data_cfg):
        return TrainValDataModule(data_cfg)

    def _load_model(self, config, device: torch.device):
        model_name = f"{config.model.name}_{config.prefix}"
        model_class = getattr(PREDICTORS, model_name)

        if self.tuning_config.start_from_checkpoint and self.tuning_config.checkpoint_path:
            model = model_class.load_from_checkpoint(checkpoint_path=self.tuning_config.checkpoint_path)
        else:
            model = model_class(config)

        return model.to(device)
