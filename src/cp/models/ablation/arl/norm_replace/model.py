"""Ablation A3c: replace ARL with branch-wise normalization (TDD)."""

import torch
import torch.nn as nn

from src.cp.models.ablation.base import AblationLightningModel, AblationTDDModel
from src.utils.real_n_complex import complex_to_real_flat, real_flat_to_complex


class NormReplaceARL(nn.Module):
    """Apply temporal LayerNorm to delay/frequency branches after data shaping."""

    def __init__(self, hist_len: int):
        super().__init__()
        # Match original ARL temporal projection normalization axis:
        # normalize over L on tensors shaped [B, D, L].
        self.norm_delay = nn.LayerNorm(hist_len)
        self.norm_freq = nn.LayerNorm(hist_len)

    def _apply_norm(self, x, norm_module):
        # x shape entering this helper is always [B, L, D].
        # Apply LayerNorm over temporal dimension L:
        # [B, L, D] -> [B, D, L] -> LN(L) -> [B, L, D].
        x = x.permute(0, 2, 1)
        x = norm_module(x)
        return x.permute(0, 2, 1)

    def forward(self, x):
        # Input from AblationTDDModel ARL stage:
        # x: [B, L, D], where D = 2*K and K is number of complex subcarriers.
        x_complex = real_flat_to_complex(x)
        # real_flat_to_complex: [B, L, 2*K] -> [B, L, K] (complex).
        x_delay = torch.fft.ifft(x_complex, dim=2)
        # IFFT is applied along subcarrier axis K, same as baseline ARL.
        # x_delay remains complex with shape [B, L, K].
        x_delay = complex_to_real_flat(x_delay)
        # complex_to_real_flat: [B, L, K] -> [B, L, 2*K] == [B, L, D].

        # Frequency branch keeps the normalized input representation unchanged.
        # x_freq: [B, L, D].
        x_freq = x

        # Branch-wise normalization, both outputs kept in [B, L, D].
        x_delay = self._apply_norm(x_delay, self.norm_delay)
        x_freq = self._apply_norm(x_freq, self.norm_freq)

        return x_delay, x_freq


class Model(AblationTDDModel):
    def _build_arl(self) -> nn.Module:
        return NormReplaceARL(hist_len=self.hist_len)


class NORM_REPLACE_ARL_TDD(AblationLightningModel):
    model_class = Model
    model_display_name = "NORM_REPLACE_ARL"
