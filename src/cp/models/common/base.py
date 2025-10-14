"""Base model class for CSI prediction using PyTorch Lightning.

This module provides the foundational BaseCSIModel class that serves as the parent
class for all CSI (Channel State Information) prediction models in the system.
It handles common functionality like training/validation loops, optimizer configuration,
loss function setup, and logging.
"""

import logging

import lightning.pytorch as pl
import torch

from src.cp.config.config import LossConfig, OptimizerConfig, SchedulerConfig
from src.cp.loss.loss import LOSS


# Configure logger for this module
logger = logging.getLogger(__name__)


class BaseCSIModel(pl.LightningModule):
    """Base class for all CSI prediction models using PyTorch Lightning.

    This class provides the common infrastructure for CSI prediction models including:
    - Training and validation loops with automatic logging
    - Configurable optimizers and learning rate schedulers
    - Loss function setup and management
    - Model parameter tracking and hyperparameter logging
    - Prediction step for inference

    All CSI prediction models should inherit from this class and implement
    the forward() method with their specific architecture.

    Attributes:
        name (str): Model name for identification and logging
        is_separate_antennas (bool): Flag indicating if model processes antennas separately
        criterion: Loss function instance configured from loss_config
        train_losses (list): Historical training losses (for compatibility)
        val_losses (list): Historical validation losses (for compatibility)

    """

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
        loss_config: LossConfig,
        **kwargs,
    ):
        """Initialize the base CSI model.

        Args:
            optimizer_config (OptimizerConfig): Configuration for the optimizer including
                name and parameters (e.g., learning_rate, weight_decay)
            scheduler_config (SchedulerConfig): Configuration for the learning rate
                scheduler including name and parameters
            loss_config (LossConfig): Configuration for the loss function including
                name and parameters
            **kwargs: Additional keyword arguments passed to parent class

        """
        super().__init__()
        # Save all hyperparameters for logging and checkpointing
        self.save_hyperparameters()

        # Model identification attributes
        self.name = "BaseModel"  # Override in child classes
        self.is_separate_antennas = True  # Flag for antenna processing strategy

        # Initialize loss function based on configuration
        self._setup_loss()

        # Legacy attributes for backward compatibility
        # Note: Actual loss tracking is handled by PyTorch Lightning
        self.train_losses = []
        self.val_losses = []

    def _setup_loss(self):
        """Set up the loss function based on configuration.

        Dynamically creates the loss function instance using the name and parameters
        specified in the loss configuration. The loss function is retrieved from
        the LOSS module and instantiated with the provided parameters.

        """
        loss_cfg = self.hparams.loss_config  # type: ignore[attr-defined]
        # Dynamically get the loss class and instantiate with parameters
        self.criterion = getattr(LOSS, loss_cfg.name)(**loss_cfg.params)

    def forward(self, x):
        """Perform forward pass through the model.

        This method must be implemented by all child classes to define the
        specific architecture and forward computation of the model.

        Args:
            x (torch.Tensor): Input tensor containing historical CSI data
                Expected shape depends on the specific model implementation

        Returns:
            torch.Tensor: Predicted CSI values for future time steps

        Raises:
            NotImplementedError: This method must be implemented by child classes

        """
        raise NotImplementedError("Forward method must be implemented by child class")

    def training_step(self, batch, batch_idx):
        """Execute a single training step.

        Processes one batch of training data by performing forward pass,
        computing loss, and logging metrics.

        Args:
            batch (tuple): Training batch containing (historical_data, target_data)
            batch_idx (int): Index of the current batch within the epoch

        Returns:
            torch.Tensor: Computed loss value for this batch

        """
        # Unpack batch data
        hist, target = batch

        # Forward pass through the model
        pred = self(hist)

        # Compute loss between predictions and targets
        loss = self.criterion(pred, target)

        # Log training loss (averaged over epoch, shown in progress bar)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Execute a single validation step.

        Processes one batch of validation data by performing forward pass
        and computing loss. No gradient computation is performed.

        Args:
            batch (tuple): Validation batch containing (historical_data, target_data)
            batch_idx (int): Index of the current batch within the validation set

        Returns:
            torch.Tensor: Computed validation loss for this batch

        """
        # Unpack batch data
        hist, target = batch

        # Forward pass through the model (no gradients computed)
        pred = self(hist)

        # Compute validation loss
        loss = self.criterion(pred, target)

        # Log validation loss (averaged over epoch, shown in progress bar)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Sets up the optimizer and scheduler based on the configurations provided
        during initialization. Automatically filters optimizer parameters to ensure
        only valid parameters are passed to each optimizer type.

        Returns:
            dict: Dictionary containing optimizer and lr_scheduler configurations
                - optimizer: The configured optimizer instance
                - lr_scheduler: Dictionary with scheduler configuration including
                  monitoring metric and update frequency

        """
        # Extract configurations from saved hyperparameters
        optimizer_cfg = self.hparams.optimizer_config  # type: ignore[attr-defined]
        scheduler_cfg = self.hparams.scheduler_config  # type: ignore[attr-defined]

        # Dynamically get the optimizer class from torch.optim
        optimizer_class = getattr(torch.optim, optimizer_cfg.name)

        # Filter parameters to ensure compatibility with the chosen optimizer
        valid_optimizer_params = self._filter_optimizer_params(optimizer_class, optimizer_cfg.params)

        # Create optimizer instance with model parameters
        optimizer = optimizer_class(self.parameters(), **valid_optimizer_params)

        # Create learning rate scheduler instance
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_cfg.name)(optimizer, **scheduler_cfg.params)

        # Return configuration dictionary for PyTorch Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor for scheduler decisions
                "interval": "epoch",  # Update frequency
                "frequency": 1,  # Update every epoch
            },
        }

    def _filter_optimizer_params(self, optimizer_class, params):
        """Filter optimizer parameters to only include valid ones for the given optimizer class.

        Different optimizers accept different parameters (e.g., SGD accepts momentum,
        but Adam doesn't). This method inspects the optimizer's signature and filters
        the provided parameters to prevent runtime errors.

        Args:
            optimizer_class (type): The optimizer class (e.g., torch.optim.SGD, torch.optim.Adam)
            params (dict): Dictionary of parameters from configuration

        Returns:
            dict: Dictionary containing only valid parameters for the specified optimizer

        """
        import inspect

        # Inspect the optimizer's constructor to get valid parameter names
        signature = inspect.signature(optimizer_class.__init__)
        valid_param_names = set(signature.parameters.keys())

        # Remove standard parameters that are handled separately
        valid_param_names.discard("self")  # Constructor self parameter
        valid_param_names.discard("params")  # Model parameters (handled separately)

        # Filter configuration parameters to only include valid ones
        filtered_params = {k: v for k, v in params.items() if k in valid_param_names}

        # Log information about parameter filtering for debugging
        filtered_out = set(params.keys()) - set(filtered_params.keys())
        if filtered_out:
            logger.warning(f"Filtered out invalid {optimizer_class.__name__} parameters: {filtered_out}")
            logger.info(f"Valid {optimizer_class.__name__} parameters: {sorted(valid_param_names)}")
            logger.info(f"Using parameters: {filtered_params}")

        return filtered_params

    def on_train_start(self):
        """Execute callback at the start of training.

        Logs important model information including parameter counts and
        hyperparameters for monitoring and reproducibility.

        """
        # Calculate model complexity metrics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Log parameter counts for model analysis
        self.log_dict(
            {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            },
            on_step=False,
            on_epoch=True,
        )

        # Log hyperparameters for experiment tracking and reproducibility
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore[attr-defined]

    def on_train_epoch_end(self):
        """Execute callback at the end of each training epoch.

        Logs the current learning rate for monitoring the learning rate
        schedule progression.

        """
        # Log current learning rate for monitoring scheduler behavior
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("learning_rate", current_lr, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx):
        """Execute a single prediction step during inference.

        Processes input data to generate predictions without computing loss.
        Handles both tuple batches (with targets) and single tensor batches.

        Args:
            batch (torch.Tensor or tuple): Input batch, either a single tensor
                or tuple of (historical_data, target_data)
            batch_idx (int): Index of the current batch

        Returns:
            torch.Tensor: Model predictions for the input batch

        """
        # Extract historical data, ignoring targets if present
        hist, _ = batch if isinstance(batch, tuple) else (batch, None)

        # Generate predictions using the model's forward pass
        return self(hist)
