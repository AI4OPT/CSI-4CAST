import logging

import lightning.pytorch as pl
import torch

from src.cp.config.config import LossConfig, OptimizerConfig, SchedulerConfig
from src.cp.loss.loss import LOSS


logger = logging.getLogger(__name__)


class BaseCSIModel(pl.LightningModule):
    """Base class for all CSI prediction models using PyTorch Lightning."""

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
        loss_config: LossConfig,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model name for tracking
        self.name = "BaseModel"
        self.is_separate_antennas = True

        # Loss function setup
        self._setup_loss()

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

    def _setup_loss(self):
        loss_cfg = self.hparams.loss_config  # type: ignore[attr-defined]
        self.criterion = getattr(LOSS, loss_cfg.name)(**loss_cfg.params)

    def forward(self, x):
        """Forward pass - to be implemented by child classes."""
        raise NotImplementedError("Forward method must be implemented by child class")

    def training_step(self, batch, batch_idx):
        """Training step."""
        hist, target = batch
        pred = self(hist)
        loss = self.criterion(pred, target)

        # Logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        hist, target = batch
        pred = self(hist)
        loss = self.criterion(pred, target)

        # Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""

        optimizer_cfg = self.hparams.optimizer_config  # type: ignore[attr-defined]
        scheduler_cfg = self.hparams.scheduler_config  # type: ignore[attr-defined]

        # Get the optimizer class
        optimizer_class = getattr(torch.optim, optimizer_cfg.name)

        # Filter optimizer parameters to only include valid ones for this optimizer
        valid_optimizer_params = self._filter_optimizer_params(optimizer_class, optimizer_cfg.params)

        optimizer = optimizer_class(self.parameters(), **valid_optimizer_params)
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_cfg.name)(optimizer, **scheduler_cfg.params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _filter_optimizer_params(self, optimizer_class, params):
        """
        Filter optimizer parameters to only include valid ones for the given optimizer class.

        Args:
            optimizer_class: The optimizer class (e.g., torch.optim.SGD)
            params: Dictionary of parameters from config

        Returns:
            Dictionary of filtered parameters
        """
        import inspect

        # Get the optimizer's __init__ signature
        signature = inspect.signature(optimizer_class.__init__)
        valid_param_names = set(signature.parameters.keys())

        # Always exclude 'self' and 'params' (first argument after self)
        valid_param_names.discard("self")
        valid_param_names.discard("params")

        # Filter the parameters
        filtered_params = {k: v for k, v in params.items() if k in valid_param_names}

        # Log any filtered out parameters
        filtered_out = set(params.keys()) - set(filtered_params.keys())
        if filtered_out:
            logger.warning(f"Filtered out invalid {optimizer_class.__name__} parameters: {filtered_out}")
            logger.info(f"Valid {optimizer_class.__name__} parameters: {sorted(valid_param_names)}")
            logger.info(f"Using parameters: {filtered_params}")

        return filtered_params

    def on_train_start(self):
        """Log model architecture and parameters at the start of training."""
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.log_dict(
            {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            },
            on_step=False,
            on_epoch=True,
        )

        # Log hyperparameters to TensorBoard
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore[attr-defined]

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log learning rate
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("learning_rate", current_lr, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        hist, _ = batch if isinstance(batch, tuple) else (batch, None)
        return self(hist)
