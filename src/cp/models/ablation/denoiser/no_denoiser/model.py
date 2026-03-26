"""Ablation A1: remove denoiser by using identity mapping (TDD)."""

from src.cp.models.ablation.base import AblationLightningModel, AblationTDDModel


class Model(AblationTDDModel):
    """denoiser_num_filters_2d=1 in params triggers nn.Identity()."""

    pass


class NO_DENOISER_TDD(AblationLightningModel):
    """Ablation: TDD model without denoiser."""

    model_class = Model
    model_display_name = "NO_DENOISER"
