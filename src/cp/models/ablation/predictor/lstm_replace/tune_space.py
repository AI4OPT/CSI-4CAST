"""Tune space for LSTM predictor replacement (TDD)."""

import optuna

from src.cp.models.ablation.base import AblationHyperparameterSpace, get_tdd_base_model_params


class HyperparameterSpace(AblationHyperparameterSpace):
    """Tuning space for the LSTM-predictor ablation."""

    @staticmethod
    def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
        """Suggest model parameters for this ablation."""
        params = get_tdd_base_model_params()
        params["predictor_lstm_num_layers"] = trial.suggest_int("predictor_lstm_num_layers", 1, 4)
        params["predictor_lstm_hidden_dim"] = trial.suggest_categorical(
            "predictor_lstm_hidden_dim", [128, 256, 512, 1024]
        )
        params["predictor_lstm_dropout"] = trial.suggest_categorical("predictor_lstm_dropout", [0.0, 0.1, 0.2])
        params["predictor_lstm_bidirectional"] = trial.suggest_categorical(
            "predictor_lstm_bidirectional", [False, True]
        )
        return params
