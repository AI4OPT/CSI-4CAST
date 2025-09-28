import logging
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn

from src.cp.models import PREDICTORS
from src.utils.dirs import DIR_WEIGHTS


logger = logging.getLogger(__name__)


def get_ckpt_path(model_name: str, scenario: str) -> str | None:
    """For TDD and FDD, return the best ckpt path for each model_name"""
    if model_name == "NP":
        return None
    else:
        return str(Path(DIR_WEIGHTS) / scenario.lower() / model_name.lower() / "model.ckpt")


def _get_eval_model(model_name: str, device: torch.device, scenario: str, ckpt_path: str | None = None):
    model_class_name = f"{model_name}_{scenario}"
    model_class = getattr(PREDICTORS, model_class_name)
    model = model_class.load_from_checkpoint(ckpt_path) if "NP" not in model_class_name else model_class()

    model.to(device)
    model.eval()

    return model


def get_eval_model(model_name: str, device: torch.device, scenario: str):
    ckpt_path = get_ckpt_path(model_name=model_name, scenario=scenario)
    logger.info(f"Loading model {model_name} for scenario {scenario} from checkpoint: {ckpt_path}\n")
    return _get_eval_model(model_name=model_name, device=device, scenario=scenario, ckpt_path=ckpt_path)


def get_eval_model_with_imputer(model_name: str, device: torch.device, scenario: str):
    """Get a model with ForwardBackwardImputer for handling package drop noise"""
    model = get_eval_model(model_name, device, scenario)
    logger.info(f"Wrapping Lightning model {model_name} with ForwardBackwardImputer for package drop handling\n")
    return LightningModelWithImputer(model)


def wrap_model_with_imputer(model: pl.LightningModule | nn.Module):
    """Wrap an existing model with the ForwardBackwardImputer for package drop handling"""
    # Check if this is a Lightning module (has trainer attribute or inherits from pl.LightningModule)
    if isinstance(model, pl.LightningModule):
        return LightningModelWithImputer(model)
    else:
        return ModelWithImputer(model)


class ForwardBackwardImputer(nn.Module):
    def __init__(self, atol=1e-6, rtol=1e-3):
        super().__init__()
        self.atol = atol
        self.rtol = rtol

    def _impute_missing(self, x):
        zeros = torch.zeros_like(x)
        close_mask = torch.isclose(x, zeros, atol=1e-6, rtol=1e-3)
        mask = close_mask.all(dim=-1)

        B, L, _ = x.shape
        device = x.device

        # Create an index matrix [[0,1,2,…,T-1], …] of shape [B, L]
        idx = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)

        # Zero-out the positions that are missing (so they don't advance the "last valid" index)
        idx = torch.where(mask, torch.zeros_like(idx), idx)  # idx[b, t] = t if not missing, else = 0

        # For each t, take the maximum index seen so far in the row
        # cummax yields ("last valid index ≤ t") for each position
        idx_fwd = torch.cummax(idx, dim=1).values  # shape [B, T]

        # We need a batch index matrix [0,0,0…;1,1,1…;…] of shape [B, T]
        b = torch.arange(B, device=device).unsqueeze(1).repeat(1, L)

        x_fwd = x[b, idx_fwd]  # fill forward, x_fwd.shape = [B, T, D]

        still_missing = mask & (idx_fwd == 0)  # still missing after forward fill

        if still_missing.any():
            # Backward fill

            # Reverse the time axis indices: [T-1, T-2, …, 0]
            idx_rev = torch.arange(L - 1, -1, -1, device=device).unsqueeze(0).repeat(B, 1)

            # Reverse the mask so "leading" slots become "trailing" slots
            mask_rev = mask.flip(dims=[1])

            # Zero-out positions that are missing in the reversed view
            idx_rev = torch.where(mask_rev, torch.full_like(idx_rev, L - 1), idx_rev)
            # cummax on reversed gives us "next valid index ≥ t" in the original order
            idx_bwd = L - 1 - torch.cummax(idx_rev, dim=1).values

            x_bwd = x[b, idx_bwd]

            x_filled = torch.where(still_missing.unsqueeze(-1), x_bwd, x_fwd)

        else:
            x_filled = x_fwd

        return x_filled

    def forward(self, x):
        """- if model.is_separate_antennas: data shape is [batch_size, hist_len, num_subcarriers*2]
        - else: data shape is [batch_size, 2, num_antennas, hist_len, num_subcarriers]
        convert to [batch_size, hist_len, num_subcarriers*2] for imputation
        """
        *batch_dims, L, D = x.shape
        x_flat = x.reshape(-1, L, D)  # Flatten batch dimensions
        x_imp_flat = self._impute_missing(x_flat)
        x_imp = x_imp_flat.reshape(*batch_dims, L, D)  # Reshape back to original batch dimensions
        return x_imp


class ModelWithImputer(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.imputer = ForwardBackwardImputer(atol=1e-6, rtol=1e-3)
        self.name = self.base_model.name
        self.is_separate_antennas = self.base_model.is_separate_antennas

    def forward(self, x, *args, **kwargs):
        # Handle missing values
        if not self.training:
            x = self.imputer(x)
        # Forward through the base model
        return self.base_model(x, *args, **kwargs)


class LightningModelWithImputer(nn.Module):
    """Wrapper for PyTorch Lightning modules with imputer capability"""

    def __init__(self, lightning_model):
        super().__init__()
        self.lightning_model = lightning_model
        self.imputer = ForwardBackwardImputer(atol=1e-6, rtol=1e-3)
        self.name = self.lightning_model.name
        self.is_separate_antennas = self.lightning_model.is_separate_antennas

        # Set the Lightning model to eval mode for inference
        self.lightning_model.eval()

    def forward(self, x, *args, **kwargs):
        # Handle missing values during inference
        if not self.training:
            x = self.imputer(x)

        # Forward through the Lightning model
        with torch.no_grad():  # Ensure no gradients during inference
            return self.lightning_model(x, *args, **kwargs)

    def to(self, device):
        """Move both the Lightning model and imputer to device"""
        self.lightning_model.to(device)
        self.imputer.to(device)
        return super().to(device)

    def eval(self):
        """Set both models to eval mode"""
        self.lightning_model.eval()
        return super().eval()

    def train(self, mode=True):
        """Set training mode - note: Lightning model should stay in eval mode for inference"""
        if mode:
            logger.warning(
                "LightningModelWithImputer is designed for inference only. Lightning model remains in eval mode."
            )
        # Only set the imputer to training mode, keep Lightning model in eval
        self.imputer.train(mode)
        return super().train(mode)
