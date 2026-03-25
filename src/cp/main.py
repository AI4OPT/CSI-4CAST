"""Channel Prediction Model Training Script.

This module provides the main training entry point for CSI prediction models using PyTorch Lightning.
It handles configuration loading, model initialization, data preparation, and training orchestration.

The training process includes:
- Configuration management and validation
- Model and data module setup
- Callback configuration (checkpointing, early stopping)
- TensorBoard logging setup
- Training execution with automatic resumption support

Usage:
    python3 -m src.cp.main --hparams_csi_pred [config_file]

Example:
    python3 -m src.cp.main --hparams_csi_pred z_artifacts/config/cp/rnn/tdd_rnn.yaml

The script automatically:
- Sets up GPU/CPU device detection
- Configures reproducible training with seeds
- Creates output directories
- Logs training progress and hyperparameters
- Saves model checkpoints and TensorBoard logs

"""

import lightning.pytorch as pl
import torch

from src.cp.dataset.data_module import TrainValDataModule
from src.utils.main_utils import (
    arg_parser,
    make_checkpoint_callback,
    make_config,
    make_early_stopping_callback,
    make_logger,
    make_output_dir,
    make_tensorboard_logger,
)
from src.utils.model_utils import load_model


# Set PyTorch matmul precision for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")

# Parse command line arguments
parser = arg_parser()
args = parser.parse_args()

# Load experiment configuration from file
config = make_config(args.hparams_csi_pred)

# Set random seed for reproducible training
pl.seed_everything(config.seed, workers=True)

# Create output directory structure
output_dir = make_output_dir(config)

# Initialize logging system
logger = make_logger(output_dir)
logger.info(f"load config from - {args.hparams_csi_pred}")
logger.info(f"output directory - {output_dir}")
logger.info(f"seed - {config.seed}")

# Setup compute device (GPU/CPU) and log device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
logger.info("device - {} | {}".format(device, device_name if torch.cuda.is_available() else "CPU"))

# Initialize model from configuration
model = load_model(config, device=device)
logger.info(f"model - {model!s}")

# Setup data module for training and validation
datamodule = TrainValDataModule(config.data)
logger.info("setup data module")

# Configure training callbacks
ckpt_cb = make_checkpoint_callback(config=config, output_dir=output_dir)  # Model checkpointing
earlystop_cb = make_early_stopping_callback(config=config)  # Early stopping
tb_logger = make_tensorboard_logger(config=config, output_dir=output_dir)  # TensorBoard logging

# Initialize PyTorch Lightning trainer with all configurations
training_cfg = config.training
trainer = pl.Trainer(
    # Basic training configuration
    max_epochs=training_cfg.num_epochs,
    accelerator=config.accelerator,  # GPU/CPU/TPU selection
    devices=config.devices,  # Number of devices to use
    precision=config.precision,  # Mixed precision training
    # Callbacks for monitoring and control
    callbacks=[ckpt_cb, earlystop_cb],
    logger=tb_logger,
    # Training behavior
    log_every_n_steps=training_cfg.log_every_n_steps,
    gradient_clip_val=training_cfg.gradient_clip_val,  # Gradient clipping for stability
    accumulate_grad_batches=training_cfg.accumulate_grad_batches,  # Gradient accumulation
    check_val_every_n_epoch=training_cfg.check_val_every_n_epoch,  # Validation frequency
    # Display options
    enable_progress_bar=training_cfg.enable_progress_bar,
    enable_model_summary=training_cfg.enable_model_summary,
)

# Execute training with optional checkpoint resumption
if config.model.checkpoint_path is not None:
    logger.info(f"Resuming training from checkpoint: {config.model.checkpoint_path}")
    trainer.fit(model, datamodule, ckpt_path=config.model.checkpoint_path)
else:
    logger.info("Starting training from scratch...")
    trainer.fit(model, datamodule)

# Log training completion and output locations
logger.info(f"Training complete. Checkpoint saved at: {ckpt_cb.best_model_path}")
logger.info(f"TensorBoard logs saved at: {tb_logger.log_dir}")
logger.info(f"Output directory: {output_dir}")
logger.info(f"Run 'tensorboard --logdir {tb_logger.log_dir}' to view TensorBoard logs.")
