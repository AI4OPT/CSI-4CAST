"""Model Loading and Imputation Utilities for Testing.

This module provides utilities for loading trained models and wrapping them with
imputation capabilities for handling missing data (package drop noise).

Key Components:
- Model loading from checkpoints with automatic device management
- Forward-backward imputation for handling missing CSI data
- Wrapper classes for both PyTorch and Lightning models
- Specialized handling for package drop noise scenarios

Classes:
    ForwardBackwardImputer: Handles missing data imputation
    ModelWithImputer: Wrapper for regular PyTorch models
    LightningModelWithImputer: Wrapper for PyTorch Lightning models

Functions:
    get_eval_model: Load model from checkpoint for evaluation
    get_eval_model_with_imputer: Load model with imputation capability
    wrap_model_with_imputer: Wrap existing model with imputer
"""

import logging
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn

from src.cp.models import PREDICTORS
from src.utils.dirs import DIR_WEIGHTS


logger = logging.getLogger(__name__)


def get_ckpt_path(model_name: str, scenario: str) -> str | None:
    """Get checkpoint path for trained models.

    Args:
        model_name (str): Model identifier (e.g., "RNN", "NP")
        scenario (str): Scenario type ("TDD" or "FDD")

    Returns:
        str | None: Path to checkpoint file, None for baseline models (NP)

    """
    if model_name == "NP":
        # Baseline models don't need checkpoints
        return None
    else:
        # Construct path: DIR_WEIGHTS/scenario/model/model.ckpt
        return str(Path(DIR_WEIGHTS) / scenario.lower() / model_name.lower() / "model.ckpt")


def _get_eval_model(model_name: str, device: torch.device, scenario: str, ckpt_path: str | None = None):
    """Load and prepare model for evaluation (internal function).

    Args:
        model_name (str): Model identifier
        device (torch.device): Target device for model
        scenario (str): Scenario type ("TDD" or "FDD")
        ckpt_path (str | None): Path to checkpoint file

    Returns:
        Model instance ready for evaluation

    """
    # Construct full model class name (e.g., "RNN_TDD", "NP_FDD")
    model_class_name = f"{model_name}_{scenario}"
    model_class = getattr(PREDICTORS, model_class_name)

    # Load from checkpoint if not a baseline model, otherwise create new instance
    model = model_class.load_from_checkpoint(ckpt_path) if "NP" not in model_class_name else model_class()

    # Move to target device and set to evaluation mode
    model.to(device)
    model.eval()

    return model


def get_eval_model(model_name: str, device: torch.device, scenario: str):
    """Load trained model for evaluation.

    Args:
        model_name (str): Model identifier (e.g., "RNN", "NP")
        device (torch.device): Target device for model
        scenario (str): Scenario type ("TDD" or "FDD")

    Returns:
        Model instance ready for evaluation

    """
    ckpt_path = get_ckpt_path(model_name=model_name, scenario=scenario)
    logger.info(f"Loading model {model_name} for scenario {scenario} from checkpoint: {ckpt_path}\n")
    return _get_eval_model(model_name=model_name, device=device, scenario=scenario, ckpt_path=ckpt_path)


def get_eval_model_with_imputer(model_name: str, device: torch.device, scenario: str):
    """Load model with imputation capability for package drop noise.

    Args:
        model_name (str): Model identifier
        device (torch.device): Target device for model
        scenario (str): Scenario type ("TDD" or "FDD")

    Returns:
        LightningModelWithImputer: Model wrapped with imputation capability

    """
    model = get_eval_model(model_name, device, scenario)
    logger.info(f"Wrapping Lightning model {model_name} with ForwardBackwardImputer for package drop handling\n")
    return LightningModelWithImputer(model)


def wrap_model_with_imputer(model: pl.LightningModule | nn.Module):
    """Wrap existing model with imputation capability.

    Args:
        model: PyTorch or Lightning model to wrap

    Returns:
        Wrapped model with ForwardBackwardImputer for handling missing data

    """
    # Check if this is a Lightning module and use appropriate wrapper
    if isinstance(model, pl.LightningModule):
        return LightningModelWithImputer(model)
    else:
        return ModelWithImputer(model)


class ForwardBackwardImputer(nn.Module):
    """Forward-backward imputation for handling missing CSI data.

    This class implements a two-stage imputation strategy:
    1. Forward fill: Use last valid observation to fill missing values
    2. Backward fill: For remaining missing values at the beginning, use next valid observation

    Args:
        atol (float): Absolute tolerance for zero detection
        rtol (float): Relative tolerance for zero detection

    """

    def __init__(self, atol=1e-6, rtol=1e-3):
        """Initialize the ForwardBackwardImputer.

        Args:
            atol (float): Absolute tolerance for zero detection
            rtol (float): Relative tolerance for zero detection

        """
        super().__init__()
        self.atol = atol
        self.rtol = rtol

    def _impute_missing(self, x):
        """Core imputation logic using forward-backward fill strategy.

        Args:
            x (torch.Tensor): Input tensor [batch_size, time_len, features]

        Returns:
            torch.Tensor: Imputed tensor with missing values filled

        """
        # Identify missing values (close to zero across all features)
        zeros = torch.zeros_like(x)
        close_mask = torch.isclose(x, zeros, atol=1e-6, rtol=1e-3)
        mask = close_mask.all(dim=-1)  # [batch_size, time_len]

        B, L, _ = x.shape
        device = x.device

        # === FORWARD FILL STAGE ===
        # Create time index matrix [[0,1,2,…,T-1], …] for each batch
        idx = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)

        # Zero out indices where data is missing (prevents advancing "last valid" index)
        idx = torch.where(mask, torch.zeros_like(idx), idx)

        # Find last valid index up to each time step using cumulative maximum
        idx_fwd = torch.cummax(idx, dim=1).values  # [batch_size, time_len]

        # Create batch index matrix for advanced indexing
        b = torch.arange(B, device=device).unsqueeze(1).repeat(1, L)

        # Forward fill: use last valid observation for each time step
        x_fwd = x[b, idx_fwd]

        # Check which positions are still missing after forward fill
        still_missing = mask & (idx_fwd == 0)

        if still_missing.any():
            # === BACKWARD FILL STAGE ===
            # For remaining missing values (at the beginning), use backward fill

            # Create reversed time indices
            idx_rev = torch.arange(L - 1, -1, -1, device=device).unsqueeze(0).repeat(B, 1)

            # Reverse the missing mask
            mask_rev = mask.flip(dims=[1])

            # Apply reversed mask to indices
            idx_rev = torch.where(mask_rev, torch.full_like(idx_rev, L - 1), idx_rev)

            # Find next valid index using cummax on reversed indices
            idx_bwd = L - 1 - torch.cummax(idx_rev, dim=1).values

            # Backward fill: use next valid observation
            x_bwd = x[b, idx_bwd]

            # Combine forward and backward fills
            x_filled = torch.where(still_missing.unsqueeze(-1), x_bwd, x_fwd)
        else:
            x_filled = x_fwd

        return x_filled

    def forward(self, x):
        """Apply imputation to input tensor.

        Handles different input shapes:
        - Separate antennas: [batch_size, hist_len, num_subcarriers*2]
        - Gather antennas: [batch_size, num_antennas, hist_len, num_subcarriers]

        Args:
            x (torch.Tensor): Input tensor with potential missing values

        Returns:
            torch.Tensor: Imputed tensor with same shape as input

        """
        # Extract dimensions and flatten batch dimensions for processing
        *batch_dims, L, D = x.shape
        x_flat = x.reshape(-1, L, D)  # Flatten to [total_batch, time, features]

        # Apply imputation to flattened tensor
        x_imp_flat = self._impute_missing(x_flat)

        # Reshape back to original batch dimensions
        x_imp = x_imp_flat.reshape(*batch_dims, L, D)
        return x_imp


class ModelWithImputer(nn.Module):
    """Wrapper for regular PyTorch models with imputation capability.

    Args:
        base_model (nn.Module): The base model to wrap with imputation

    """

    def __init__(self, base_model: nn.Module):
        """Initialize the ModelWithImputer.

        Args:
            base_model (nn.Module): The base model to wrap with imputation

        """
        super().__init__()
        self.base_model = base_model
        self.imputer = ForwardBackwardImputer(atol=1e-6, rtol=1e-3)
        # Inherit properties from base model
        self.name = self.base_model.name
        self.is_separate_antennas = self.base_model.is_separate_antennas

    def forward(self, x, *args, **kwargs):
        """Forward pass with optional imputation during inference.

        Args:
            x (torch.Tensor): Input tensor
            *args: Additional positional arguments for base model
            **kwargs: Additional keyword arguments for base model

        Returns:
            Model output after optional imputation and base model forward pass

        """
        # Apply imputation only during inference (not training)
        if not self.training:
            x = self.imputer(x)

        # Forward through the base model
        return self.base_model(x, *args, **kwargs)


class LightningModelWithImputer(nn.Module):
    """Wrapper for PyTorch Lightning modules with imputation capability.

    This wrapper is specifically designed for inference with Lightning models,
    keeping the Lightning model in eval mode while allowing the imputer to
    switch between training/eval modes as needed.

    Args:
        lightning_model: PyTorch Lightning model to wrap

    """

    def __init__(self, lightning_model):
        """Initialize the LightningModelWithImputer.

        Args:
            lightning_model: PyTorch Lightning model to wrap

        """
        super().__init__()
        self.lightning_model = lightning_model
        self.imputer = ForwardBackwardImputer(atol=1e-6, rtol=1e-3)
        # Inherit properties from Lightning model
        self.name = self.lightning_model.name
        self.is_separate_antennas = self.lightning_model.is_separate_antennas

        # Set the Lightning model to eval mode for inference
        self.lightning_model.eval()

    def forward(self, x, *args, **kwargs):
        """Forward pass with imputation and gradient-free Lightning model inference.

        Args:
            x (torch.Tensor): Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Model output after imputation and Lightning model forward pass

        """
        # Apply imputation during inference only
        if not self.training:
            x = self.imputer(x)

        # Forward through Lightning model without gradients (inference only)
        with torch.no_grad():
            return self.lightning_model(x, *args, **kwargs)

    def to(self, device):
        """Move both Lightning model and imputer to specified device.

        Args:
            device: Target device

        Returns:
            Self for method chaining

        """
        self.lightning_model.to(device)
        self.imputer.to(device)
        return super().to(device)

    def eval(self):
        """Set both models to evaluation mode."""
        self.lightning_model.eval()
        return super().eval()

    def train(self, mode=True):
        """Set training mode with Lightning model kept in eval mode.

        Args:
            mode (bool): Training mode flag

        Note:
            Lightning model remains in eval mode for inference-only usage.
            Only the imputer switches between train/eval modes.

        """
        if mode:
            logger.warning(
                "LightningModelWithImputer is designed for inference only. Lightning model remains in eval mode."
            )
        # Only set the imputer to training mode, keep Lightning model in eval
        self.imputer.train(mode)
        return super().train(mode)
