"""Tune space for no-denoiser ablation (TDD)."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_tdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    """Tuning space for the no-denoiser ablation."""

    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        """Suggest model parameters for this ablation."""
        params = get_tdd_base_model_params()
        params["denoiser_num_filters_2d"] = 1
        return params
