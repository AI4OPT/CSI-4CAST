"""Ablation A3b: remove ARL learnable projections, keep data shaping only (FDD)."""

import torch
import torch.nn as nn

from src.cp.models.ablation.base import AblationFDDModel, AblationLightningModel
from src.utils.real_n_complex import complex_to_real_flat, real_flat_to_complex


class PassThroughARL(nn.Module):
    """Only perform frequency/delay data shaping, with no learnable layers."""

    def forward(self, x):
        """Pass input through frequency/delay shaping only."""
        x_complex = real_flat_to_complex(x)
        x_delay = torch.fft.ifft(x_complex, dim=2)
        x_delay = complex_to_real_flat(x_delay)

        x_freq = x
        return x_delay, x_freq


class Model(AblationFDDModel):
    """FDD ablation model without ARL."""

    def _build_arl(self) -> nn.Module:
        return PassThroughARL()


class NO_ARL_FDD(AblationLightningModel):
    """Ablation: FDD model without ARL."""

    model_class = Model
    model_display_name = "NO_ARL"
