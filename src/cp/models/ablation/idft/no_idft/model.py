"""Ablation A2: remove IDFT delay branch, use frequency branch only (TDD).

Without IDFT there is no delay-domain signal, so:
  - ARL: only frequency-side temporal (and optionally subcarrier) projections
  - Embedding: only RB_freq ShuffleNet branch, no RB_delay, no delay+freq sum
  - forward: single-path (no delay/freq split)
"""

from einops import rearrange
import torch.nn as nn

from src.cp.models.ablation.base import (
    AblationLightningModel,
    AblationTDDModel,
    batch_denormalize,
    batch_normalizer,
)
from src.cp.models.common.dataembedding import DataEmbedding
from src.cp.models.proposed.model_tdd import (
    AdaptiveReweightingLayer,
    ShuffleBlockCA,
)


class FreqOnlyARL(nn.Module):
    """ARL that only processes the frequency branch (no IDFT / no delay branch)."""

    def __init__(
        self,
        hist_len=16,
        dim_data=600,
        is_U2D=False,
        temporal_proj_num_layers=4,
        temporal_proj_hidden_dim=512,
        temporal_proj_is_arl=False,
        temporal_proj_output_activation_name="none",
        temporal_proj_arl_operation="multiply",
        subcarrier_proj_num_layers=2,
        subcarrier_proj_hidden_dim=128,
        subcarrier_proj_is_arl=True,
        subcarrier_proj_output_activation_name="tanh",
        subcarrier_proj_arl_operation="multiply",
    ):
        """Initialize frequency-only ARL without IDFT."""
        super().__init__()
        self.is_U2D = is_U2D

        self.temporal_proj_freq = AdaptiveReweightingLayer(
            in_dim=hist_len,
            out_dim=hist_len,
            num_layers=temporal_proj_num_layers,
            hidden_dim=temporal_proj_hidden_dim,
            is_arl=temporal_proj_is_arl,
            output_activation_name=temporal_proj_output_activation_name,
            arl_operation=temporal_proj_arl_operation,
        )
        if is_U2D:
            self.subcarrier_proj_freq = AdaptiveReweightingLayer(
                in_dim=dim_data,
                out_dim=dim_data,
                num_layers=subcarrier_proj_num_layers,
                hidden_dim=subcarrier_proj_hidden_dim,
                is_arl=subcarrier_proj_is_arl,
                output_activation_name=subcarrier_proj_output_activation_name,
                arl_operation=subcarrier_proj_arl_operation,
            )

    def forward(self, x):
        """Process input through frequency-only ARL."""
        x = x.permute(0, 2, 1)
        x = self.temporal_proj_freq(x)
        x = x.permute(0, 2, 1)
        if self.is_U2D:
            x = self.subcarrier_proj_freq(x)
        return x


class FreqOnlyEmbedding(nn.Module):
    """ShuffleNet embedding that only has the frequency branch."""

    def __init__(
        self,
        dim_model: int,
        num_res_layers: int = 4,
        res_dim: int = 64,
        res_groups: int = 4,
        res_ca_ratio: int = 2,
        hist_len: int = 16,
        dim_data: int = 600,
        embed: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        """Initialize frequency-only ShuffleNet embedding."""
        super().__init__()

        layers: list[nn.Module] = [nn.Conv2d(2, res_dim, 3, 1, 1)]
        for _ in range(num_res_layers):
            layers.append(ShuffleBlockCA(in_channels=res_dim, groups=res_groups, ca_ratio=res_ca_ratio))
        layers.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        self.RB_freq = nn.Sequential(*layers)

        self.embedding = DataEmbedding(dim_data, dim_model, embed, freq, dropout)
        self.predict_linear_pre = nn.Linear(hist_len, hist_len)

    def forward(self, x_freq):
        """Embed frequency input through the ShuffleNet branch."""
        x_freq = rearrange(x_freq, "b l (k o) -> b o l k", o=2)
        x_freq = self.RB_freq(x_freq)
        x = rearrange(x_freq, "b o l k -> b l (k o)", o=2)
        x = self.embedding(x)
        x = self.predict_linear_pre(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class Model(AblationTDDModel):
    """TDD ablation model without IDFT."""

    def _build_arl(self) -> nn.Module:
        return FreqOnlyARL(**self._arl_kwargs())

    def _build_embedding(self) -> nn.Module:
        p = self._p
        return FreqOnlyEmbedding(
            dim_model=self.dim_model,
            num_res_layers=p.get("embedding_num_res_layers", 4),
            res_dim=p.get("embedding_res_dim", 64),
            res_groups=p.get("embedding_res_groups", 4),
            res_ca_ratio=p.get("embedding_ca_ratio", 2),
            hist_len=self.hist_len,
            dim_data=self.dim_data,
            embed=p.get("embedding_embed", "timeF"),
            freq=p.get("embedding_freq", "h"),
            dropout=p.get("embedding_dropout", 0.1),
        )

    def forward(self, x):
        """Run the no-IDFT prediction pipeline."""
        x = self.denoiser(x)
        x, mean, std = batch_normalizer(x)
        x = self.arl(x)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.mlp(x)
        x = batch_denormalize(x, mean, std)
        return x


class NO_IDFT_TDD(AblationLightningModel):
    """Ablation: TDD model without IDFT."""

    model_class = Model
    model_display_name = "NO_IDFT"
