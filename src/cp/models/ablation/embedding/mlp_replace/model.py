"""Ablation A4a: replace ShuffleNet embedding with branch MLP embedding (TDD)."""

import torch.nn as nn

from src.cp.models.ablation.base import AblationLightningModel, AblationTDDModel
from src.cp.models.common.dataembedding import DataEmbedding


class MLPEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        hist_len: int,
        dim_data: int,
        embed: str,
        freq: str,
        dropout: float,
        num_layers: int = 3,
        hidden_dim: int = 512,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()

        # dim notation used in this module:
        # B: batch_size
        # L: hist_len (sequence length in time)
        # D: dim_data (per-timestep feature size, typically 2 * NUM_SUBCARRIERS)
        # M: dim_model (model/embedding size)

        self.delay_branch = self._build_branch(dim_data, num_layers, hidden_dim, mlp_dropout)
        self.freq_branch = self._build_branch(dim_data, num_layers, hidden_dim, mlp_dropout)

        self.embedding = DataEmbedding(dim_data, dim_model, embed, freq, dropout)
        self.predict_linear_pre = nn.Linear(hist_len, hist_len)

    def _build_branch(self, dim_data: int, num_layers: int, hidden_dim: int, mlp_dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = dim_data

        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(mlp_dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, dim_data))
        return nn.Sequential(*layers)

    def forward(self, x_delay, x_freq):
        # x_delay, x_freq: [B, L, D]
        x_delay = self.delay_branch(x_delay)  # [B, L, D]
        x_freq = self.freq_branch(x_freq)  # [B, L, D]

        x = x_delay + x_freq  # [B, L, D]
        x = self.embedding(x)  # [B, L, M]
        x = self.predict_linear_pre(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, M]
        return x


class Model(AblationTDDModel):
    def _build_embedding(self) -> nn.Module:
        p = self._p
        return MLPEmbedding(
            dim_model=self.dim_model,
            hist_len=self.hist_len,
            dim_data=self.dim_data,
            embed=p.get("embedding_embed", "timeF"),
            freq=p.get("embedding_freq", "h"),
            dropout=p.get("embedding_dropout", 0.1),
            num_layers=p.get("embedding_mlp_num_layers", 3),
            hidden_dim=p.get("embedding_mlp_hidden_dim", 512),
            mlp_dropout=p.get("embedding_mlp_dropout", 0.1),
        )


class MLP_REPLACE_EMBED_TDD(AblationLightningModel):
    model_class = Model
    model_display_name = "MLP_REPLACE_EMBED"
