"""Base abstract class for hyperparameter search spaces.

This provides a reusable framework that can be inherited and customized
by specific implementations for different domains.
"""

from abc import ABC, abstractmethod
from typing import Any

import optuna


class BaseHyperparameterSpace(ABC):
    """Abstract base class for hyperparameter search spaces.

    This class provides template implementations for common hyperparameter
    suggestions. Subclasses should override methods to customize the search space
    for their specific use case.
    """

    @staticmethod
    def suggest_optimizer_params(trial: optuna.Trial, optimizer_name: str) -> dict[str, Any]:
        """Suggest optimizer hyperparameters.

        Subclasses should override this to customize optimizer parameter ranges.

        Args:
            trial: Optuna trial object
            optimizer_name: Name of the optimizer

        Returns:
            Dictionary of optimizer parameters

        """
        params = {}

        if optimizer_name == "Adam" or optimizer_name == "AdamW":
            params["lr"] = trial.suggest_float("optimizer_lr", 1e-5, 1e-2, log=True)
            params["weight_decay"] = trial.suggest_float("optimizer_weight_decay", 1e-8, 1e-2, log=True)

        return params

    @staticmethod
    def suggest_scheduler_params(trial: optuna.Trial, scheduler_name: str) -> dict[str, Any]:
        """Suggest scheduler hyperparameters.

        Subclasses should override this to customize scheduler parameter ranges.

        Args:
            trial: Optuna trial object
            scheduler_name: Name of the scheduler

        Returns:
            Dictionary of scheduler parameters

        """
        params = {}

        if scheduler_name == "ReduceLROnPlateau":
            params["mode"] = "min"
            params["factor"] = trial.suggest_float("scheduler_factor", 0.1, 0.9)
            params["patience"] = trial.suggest_int("scheduler_patience", 5, 50)
            params["threshold"] = trial.suggest_float("scheduler_threshold", 1e-6, 1e-3, log=True)
            params["cooldown"] = trial.suggest_int("scheduler_cooldown", 0, 10)
            params["min_lr"] = trial.suggest_float("scheduler_min_lr", 1e-8, 1e-5, log=True)
        elif scheduler_name == "CosineAnnealingWarmupRestarts":
            params["warmup_epochs"] = trial.suggest_int("scheduler_warmup_epochs", 10, 100)
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")

        return params

    @staticmethod
    @abstractmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
        """Suggest model hyperparameters.

        This method MUST be overridden by subclasses as model parameters are
        highly domain-specific.

        Args:
            trial: Optuna trial object
            model_name: Name of the model

        Returns:
            Dictionary of model parameters

        """
        pass

    @staticmethod
    @abstractmethod
    def suggest_training_params(trial: optuna.Trial) -> dict[str, Any]:
        """Suggest training hyperparameters.

        This method MUST be overridden by subclasses as training parameters
        are domain-specific.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of training parameters

        """
        pass

    @staticmethod
    def suggest_loss_params(trial: optuna.Trial, loss_name: str) -> dict[str, Any]:
        """Suggest loss function hyperparameters.

        Subclasses can override this to customize loss parameter ranges.

        Args:
            trial: Optuna trial object
            loss_name: Name of the loss function

        Returns:
            Dictionary of loss parameters

        """
        params = {}
        # Most loss functions don't have hyperparameters
        return params

    @staticmethod
    def suggest_optimizer_name(trial: optuna.Trial) -> str:
        """Suggest optimizer name.

        Subclasses can override this to customize available optimizers.

        Args:
            trial: Optuna trial object

        Returns:
            Name of the suggested optimizer

        """
        return trial.suggest_categorical("optimizer_name", ["Adam", "AdamW"])

    @staticmethod
    def suggest_scheduler_name(trial: optuna.Trial) -> str:
        """Suggest scheduler name.

        Subclasses can override this to customize available schedulers.

        Args:
            trial: Optuna trial object

        Returns:
            Name of the suggested scheduler

        """
        return trial.suggest_categorical("scheduler_name", ["ReduceLROnPlateau", "CosineAnnealingWarmupRestarts"])

    @staticmethod
    def suggest_loss_name(trial: optuna.Trial) -> str:
        """Suggest loss function name.

        Subclasses can override this to customize available loss functions.

        Args:
            trial: Optuna trial object

        Returns:
            Name of the suggested loss function

        """
        return trial.suggest_categorical("loss_name", ["NMSE", "MSE"])

    @staticmethod
    def suggest_model_name(trial: optuna.Trial, available_models: list[str]) -> str:
        """Suggest model name from available models.

        Subclasses can override this to customize model selection logic.

        Args:
            trial: Optuna trial object
            available_models: List of available model names

        Returns:
            Name of the suggested model

        """
        return trial.suggest_categorical("model_name", available_models)
