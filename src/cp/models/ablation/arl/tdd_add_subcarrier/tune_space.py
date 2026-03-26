"""Tune space for TDD add-subcarrier ablation."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_tdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    """Tuning space for the add-subcarrier ARL ablation."""

    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        """Suggest model parameters for this ablation."""
        params = get_tdd_base_model_params()
        # enable subcarrier ARL for this ablation
        params["arl_is_U2D"] = True

        # simple search space for subcarrier ARL
        params["arl_subcarrier_proj_num_layers"] = trial.suggest_int("arl_subcarrier_proj_num_layers", 1, 3)
        params["arl_subcarrier_proj_hidden_dim"] = trial.suggest_categorical(
            "arl_subcarrier_proj_hidden_dim", [256, 512, 1024]
        )
        params["arl_subcarrier_proj_output_activation_name"] = trial.suggest_categorical(
            "arl_subcarrier_proj_output_activation_name", ["tanh", "relu"]
        )
        params["arl_subcarrier_proj_arl_operation"] = trial.suggest_categorical(
            "arl_subcarrier_proj_arl_operation", ["multiply", "add"]
        )

        return params
