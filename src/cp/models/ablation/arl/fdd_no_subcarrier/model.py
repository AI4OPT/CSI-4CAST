"""Ablation A3e: remove subcarrier projection from FDD ARL."""

from src.cp.models.ablation.base import AblationFDDModel, AblationLightningModel


class Model(AblationFDDModel):
    """arl_is_U2D=False in params disables subcarrier projection."""

    pass


class NO_SUBCARRIER_ARL_FDD(AblationLightningModel):
    """Ablation: FDD model without subcarrier ARL."""

    model_class = Model
    model_display_name = "NO_SUBCARRIER_ARL"
