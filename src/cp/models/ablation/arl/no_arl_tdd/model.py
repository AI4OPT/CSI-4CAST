"""Ablation A3a: remove ARL learnable projections, keep data shaping only (TDD)."""

import torch
import torch.nn as nn

from src.cp.models.ablation.base import AblationLightningModel, AblationTDDModel
from src.utils.real_n_complex import complex_to_real_flat, real_flat_to_complex


class PassThroughARL(nn.Module):
    """Only perform frequency/delay data shaping, with no learnable layers."""

    def forward(self, x):
        x_complex = real_flat_to_complex(x)
        x_delay = torch.fft.ifft(x_complex, dim=2)
        x_delay = complex_to_real_flat(x_delay)

        x_freq = x
        return x_delay, x_freq


class Model(AblationTDDModel):
    def _build_arl(self) -> nn.Module:
        return PassThroughARL()


class NO_ARL_TDD(AblationLightningModel):
    model_class = Model
    model_display_name = "NO_ARL"
