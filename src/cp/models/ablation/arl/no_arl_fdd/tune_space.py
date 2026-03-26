"""Tune space for no-ARL ablation (FDD)."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_fdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    """Tuning space for the no-ARL FDD ablation."""

    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        """Suggest model parameters for this ablation."""
        return get_fdd_base_model_params()
