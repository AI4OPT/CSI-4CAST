import logging
import re
from pathlib import Path

import lightning.pytorch as pl
import torch

from src.cp.config.config import ExperimentConfig
from src.cp.models import PREDICTORS


logger = logging.getLogger(__name__)


def _best_ckpt_path(ckpt_dir: Path) -> Path:
    """In ckpt_dir, find all files named like "*-val_loss=XXXXX.ckpt",
    parse out the val_loss number, and return the Path for the smallest val_loss.

    Raises:
      FileNotFoundError if no valid filenames are found.

    """
    # 1) gather all .ckpt files
    all_ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not all_ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir!r}")

    # 2) build a list of tuples (val_loss_value, path), but skip any file
    #    whose name does NOT match the pattern '-val_loss=NUM.ckpt'
    ckpt_pairs = []
    pattern = re.compile(r"val_loss=([0-9]*\.?[0-9]+)")  # captures e.g. "0.0298"
    for p in all_ckpts:
        m = pattern.search(p.stem)
        if m:
            # parse string "0.0298" → float 0.0298
            val = float(m.group(1))
            ckpt_pairs.append((val, p))
        else:
            # didn’t find 'val_loss=…' in the filename → skip it
            continue

    if not ckpt_pairs:
        raise FileNotFoundError(f"No checkpoint filenames in {ckpt_dir!r} match '*-val_loss=XXX.ckpt'")

    # 3) pick the tuple whose val_loss is minimal
    _, best_path = min(ckpt_pairs, key=lambda tup: tup[0])
    return best_path


def load_model(config: ExperimentConfig, device: torch.device) -> pl.LightningModule:
    """Load model based on the configuration."""
    model_cfg = config.model
    model_class = getattr(PREDICTORS, model_cfg.name)

    if model_cfg.checkpoint_path is not None:
        ckpt_path = Path(model_cfg.checkpoint_path)
        if ckpt_path.is_file() and ckpt_path.suffix == ".ckpt":
            checkpoint_to_load = ckpt_path
        elif ckpt_path.is_dir():
            checkpoint_to_load = _best_ckpt_path(ckpt_path)
        else:
            raise FileNotFoundError(f"Checkpoint path {model_cfg.checkpoint_path} is not a valid file or directory.")

        model = model_class.load_from_checkpoint(checkpoint_path=str(checkpoint_to_load))
        logger.info(f"Loaded model from checkpoint: {checkpoint_to_load!s}")
    else:
        model = model_class(config)
        logger.info("Initialized model from configuration without checkpoint.")

    model.to(device)
    return model
