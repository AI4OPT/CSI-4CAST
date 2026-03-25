"""Tune space for MLP embedding replacement (TDD)."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_tdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        params = get_tdd_base_model_params()
        params["embedding_mlp_num_layers"] = trial.suggest_int("embedding_mlp_num_layers", 2, 6)
        params["embedding_mlp_hidden_dim"] = trial.suggest_categorical(
            "embedding_mlp_hidden_dim", [128, 256, 512, 1024]
        )
        params["embedding_mlp_dropout"] = trial.suggest_categorical("embedding_mlp_dropout", [0.0, 0.1, 0.2])
        return params
