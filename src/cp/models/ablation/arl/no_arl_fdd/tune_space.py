"""Tune space for no-ARL ablation (FDD)."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_fdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        return get_fdd_base_model_params()
