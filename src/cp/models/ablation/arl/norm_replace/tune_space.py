"""Tune space for ARL norm-replacement ablation (TDD)."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_tdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        return get_tdd_base_model_params()
