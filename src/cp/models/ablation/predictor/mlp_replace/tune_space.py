"""Tune space for MLP predictor replacement (TDD)."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_tdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    """Tuning space for the MLP-predictor ablation."""

    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        """Suggest model parameters for this ablation."""
        params = get_tdd_base_model_params()
        params["predictor_mlp_num_layers"] = trial.suggest_int("predictor_mlp_num_layers", 2, 6)
        params["predictor_mlp_hidden_dim"] = trial.suggest_categorical(
            "predictor_mlp_hidden_dim", [128, 256, 512, 1024]
        )
        return params
