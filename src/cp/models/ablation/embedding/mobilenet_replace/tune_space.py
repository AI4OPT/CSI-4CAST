"""Tune space for MobileNet embedding replacement (TDD)."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_tdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        params = get_tdd_base_model_params()
        params["embedding_mobilenet_num_blocks"] = trial.suggest_int("embedding_mobilenet_num_blocks", 2, 8)
        params["embedding_mobilenet_base_channels"] = trial.suggest_categorical(
            "embedding_mobilenet_base_channels", [16, 24, 32, 48, 64, 96, 128]
        )
        params["embedding_mobilenet_expand_ratio"] = trial.suggest_categorical(
            "embedding_mobilenet_expand_ratio", [1, 2, 3]
        )
        params["embedding_mobilenet_kernel_size"] = trial.suggest_categorical("embedding_mobilenet_kernel_size", [3, 5])
        params["embedding_mobilenet_use_se"] = trial.suggest_categorical("embedding_mobilenet_use_se", [True])
        params["embedding_mobilenet_use_hs"] = trial.suggest_categorical("embedding_mobilenet_use_hs", [True])
        params["embedding_mobilenet_se_ratio"] = trial.suggest_categorical("embedding_mobilenet_se_ratio", [2, 4, 8])
        return params
