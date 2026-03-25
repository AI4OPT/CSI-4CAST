"""Ablation A5a: replace Transformer predictor with a simple MLP (TDD).

Dimensions (default: dim_model=768, hidden_dim=1024):
    input:  (B, L, dim_model)   e.g. (B, 16, 768)
    MLP:    (B, L, dim_model)   e.g. (B, 16, 768)  — applied per time step
    output: (B, L, dim_model)   e.g. (B, 16, 768)
"""

import torch.nn as nn

from src.cp.models.ablation.base import AblationLightningModel, AblationTDDModel
from src.cp.models.common.mlp import MLP


class Model(AblationTDDModel):
    def _build_transformer(self) -> nn.Module:
        p = self._p
        return MLP(
            in_dim=self.dim_model,
            out_dim=self.dim_model,
            num_layers=p.get("predictor_mlp_num_layers", 3),
            hidden_dim=p.get("predictor_mlp_hidden_dim", 1024),
            output_activation=nn.Identity(),
        )


class MLP_REPLACE_PRED_TDD(AblationLightningModel):
    model_class = Model
    model_display_name = "MLP_REPLACE_PRED"
