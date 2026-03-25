"""Tune space for FDD no-subcarrier ablation."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_fdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        params = get_fdd_base_model_params()
        params["arl_is_U2D"] = False
        return params
