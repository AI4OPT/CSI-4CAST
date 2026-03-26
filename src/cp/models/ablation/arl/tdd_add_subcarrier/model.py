"""Ablation A3d: add subcarrier projection to TDD ARL."""

from src.cp.models.ablation.base import AblationLightningModel, AblationTDDModel


class Model(AblationTDDModel):
    """arl_is_U2D=True in params enables subcarrier projection."""

    pass


class ADD_SUBCARRIER_ARL_TDD(AblationLightningModel):
    """Ablation: TDD model with added subcarrier ARL."""

    model_class = Model
    model_display_name = "ADD_SUBCARRIER_ARL"
