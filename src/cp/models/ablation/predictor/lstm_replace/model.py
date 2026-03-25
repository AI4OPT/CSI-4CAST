"""Ablation A5c: replace Transformer predictor with LSTM predictor (TDD)."""

import torch.nn as nn

from src.cp.models.ablation.base import AblationLightningModel, AblationTDDModel


class LSTMPredictor(nn.Module):
    """Replace the Transformer predictor with an LSTM.

    Dimensions (default: dim_model=768, hidden_dim=512, bidirectional=False):
        input:  (B, L, dim_model)        e.g. (B, 16, 768)
        lstm:   (B, L, hidden_dim * D)   e.g. (B, 16, 512)  where D=2 if bidirectional else 1
        proj:   (B, L, dim_model)        e.g. (B, 16, 768)
    """

    def __init__(self, dim_model: int, num_layers: int = 2, hidden_dim: int = 512,
                 dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=dim_model, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=recurrent_dropout, bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, dim_model)

    def forward(self, x):
        # x: (B, L, dim_model)
        x, _ = self.lstm(x)  # -> (B, L, hidden_dim * D)
        return self.proj(x)  # -> (B, L, dim_model)


class Model(AblationTDDModel):
    def _build_transformer(self) -> nn.Module:
        p = self._p
        return LSTMPredictor(
            dim_model=self.dim_model,
            num_layers=p.get("predictor_lstm_num_layers", 2),
            hidden_dim=p.get("predictor_lstm_hidden_dim", 512),
            dropout=p.get("predictor_lstm_dropout", 0.1),
            bidirectional=p.get("predictor_lstm_bidirectional", False),
        )


class LSTM_REPLACE_PRED_TDD(AblationLightningModel):
    model_class = Model
    model_display_name = "LSTM_REPLACE_PRED"
