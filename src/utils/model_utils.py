"""Model utility functions for loading and managing PyTorch Lightning models.

This module provides utilities for:
- Loading models from checkpoints or configuration
- Finding the best checkpoint based on validation loss
- Managing model initialization and device placement

The utilities support both checkpoint-based loading (for inference or continued training)
and fresh initialization (for new training runs).
"""

import logging
import re
from pathlib import Path

import lightning.pytorch as pl
import torch

from src.cp.config.config import ExperimentConfig
from src.cp.models import PREDICTORS


# Configure logger for this module
logger = logging.getLogger(__name__)


def _best_ckpt_path(ckpt_dir: Path) -> Path:
    """Find the checkpoint file with the lowest validation loss in a directory.

    Searches through all .ckpt files in the given directory, extracts validation
    loss values from filenames following the pattern '*-val_loss=XXXXX.ckpt',
    and returns the path to the checkpoint with the minimum validation loss.

    This is useful for automatically selecting the best performing model from
    a training run for inference or evaluation.

    Args:
        ckpt_dir (Path): Directory containing checkpoint files

    Returns:
        Path: Path to the checkpoint file with the lowest validation loss

    Raises:
        FileNotFoundError: If no .ckpt files are found or no files match the
                          expected naming pattern '*-val_loss=XXX.ckpt'

    """
    # Collect all checkpoint files in the directory
    all_ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not all_ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir!r}")

    # Extract validation loss values from checkpoint filenames
    # Expected format: "epoch=XXX-step=XXXXXX-val_loss=X.XXXXXX.ckpt"
    ckpt_pairs = []
    pattern = re.compile(r"val_loss=([0-9]*\.?[0-9]+)")  # Matches decimal numbers

    for checkpoint_path in all_ckpts:
        match = pattern.search(checkpoint_path.stem)
        if match:
            # Extract validation loss value from filename
            val_loss_value = float(match.group(1))
            ckpt_pairs.append((val_loss_value, checkpoint_path))
        else:
            # Skip files that don't follow the expected naming convention
            logger.debug(f"Skipping checkpoint file with unexpected name: {checkpoint_path.name}")
            continue

    if not ckpt_pairs:
        raise FileNotFoundError(
            f"No checkpoint filenames in {ckpt_dir!r} match the expected pattern '*-val_loss=XXX.ckpt'"
        )

    # Select the checkpoint with the minimum validation loss
    _, best_checkpoint_path = min(ckpt_pairs, key=lambda tup: tup[0])
    logger.info(f"Selected best checkpoint: {best_checkpoint_path.name} with val_loss={min(ckpt_pairs)[0]:.6f}")

    return best_checkpoint_path


def load_model(config: ExperimentConfig, device: torch.device) -> pl.LightningModule:
    """Load a PyTorch Lightning model from checkpoint or initialize from configuration.

    This function provides flexible model loading that supports:
    1. Loading from a specific checkpoint file (.ckpt)
    2. Auto-selecting the best checkpoint from a directory
    3. Fresh initialization from configuration (for new training)

    The function automatically handles model placement on the specified device
    and provides appropriate logging for tracking model loading.

    Args:
        config (ExperimentConfig): Experiment configuration containing model settings
        device (torch.device): Target device for model placement (CPU/GPU)

    Returns:
        pl.LightningModule: Loaded or initialized PyTorch Lightning model

    Raises:
        FileNotFoundError: If specified checkpoint path doesn't exist or is invalid
        AttributeError: If the specified model class is not found in PREDICTORS

    """
    model_cfg = config.model

    # Dynamically get the model class from the PREDICTORS registry
    model_class = getattr(PREDICTORS, model_cfg.name)
    logger.info(f"Using model class: {model_class.__name__}")

    # Determine loading strategy based on checkpoint configuration
    if model_cfg.checkpoint_path is not None:
        ckpt_path = Path(model_cfg.checkpoint_path)

        if ckpt_path.is_file() and ckpt_path.suffix == ".ckpt":
            # Load from specific checkpoint file
            checkpoint_to_load = ckpt_path
            logger.info(f"Loading from specific checkpoint: {checkpoint_to_load}")

        elif ckpt_path.is_dir():
            # Auto-select best checkpoint from directory
            checkpoint_to_load = _best_ckpt_path(ckpt_path)
            logger.info(f"Auto-selected best checkpoint from directory: {checkpoint_to_load}")

        else:
            raise FileNotFoundError(f"Checkpoint path '{model_cfg.checkpoint_path}' is not a valid file or directory")

        # Load model from selected checkpoint
        model = model_class.load_from_checkpoint(checkpoint_path=str(checkpoint_to_load))
        logger.info(f"Successfully loaded model from checkpoint: {checkpoint_to_load.name}")

    else:
        # Initialize fresh model from configuration
        model = model_class(config)
        logger.info("Initialized new model from configuration (no checkpoint provided)")

    # Move model to specified device (GPU/CPU)
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    return model
