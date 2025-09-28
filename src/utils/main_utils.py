"""Main utility functions for experiment setup and management.

This module provides essential utilities for setting up and managing machine learning
experiments, including:
- Command line argument parsing
- Configuration file loading and validation
- Output directory creation with timestamping
- Logging setup with custom formatters
- PyTorch Lightning callback creation (checkpointing, early stopping, logging)

These utilities are used across different experiment scripts to ensure consistent
setup and configuration management.
"""

import argparse
import datetime
import logging
import os
from pathlib import Path

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.cp.config.config import ExperimentConfig


def arg_parser():
    """Create and configure the command line argument parser.

    Sets up the standard command line interface for CSI prediction experiments,
    allowing users to specify custom configuration files.

    Returns:
        argparse.ArgumentParser: Configured argument parser with CSI prediction options

    """
    parser = argparse.ArgumentParser(description="CSI Prediction Experiment Runner")
    parser.add_argument(
        "--hparams_csi_pred",
        "-hcp",
        type=str,
        default="z_artifacts/config/cp/rnn/tdd_rnn.yaml",
        help="Path to the CSI prediction hyperparameters configuration file (YAML or JSON)",
    )
    return parser


def make_config(path_config: str):
    """Load experiment configuration from file.

    Supports both JSON and YAML configuration file formats. The configuration
    contains all experiment settings including model architecture, training
    parameters, data paths, and logging options.

    Args:
        path_config (str): Path to the configuration file (.json or .yaml)

    Returns:
        ExperimentConfig: Loaded and validated configuration object

    Raises:
        ValueError: If the configuration file format is not supported

    """
    if path_config.endswith(".json"):
        config = ExperimentConfig.fromJson(path_config)
    elif path_config.endswith(".yaml"):
        config = ExperimentConfig.fromYaml(path_config)
    else:
        raise ValueError(f"Unsupported config file format: {path_config}. Use .json or .yaml")
    return config


def make_output_dir(config: ExperimentConfig) -> Path:
    """Create timestamped output directory and save configuration copy.

    Creates a new output directory with the current timestamp to ensure each
    experiment run has a unique output location. Also saves a copy of the
    configuration for reproducibility and experiment tracking.

    Args:
        config (ExperimentConfig): Experiment configuration containing output directory path

    Returns:
        Path: Path to the created timestamped output directory

    """
    # Create timestamped directory name for this experiment run
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(config.output_dir) / cur_time

    # Create the directory (including parent directories if needed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configuration for reproducibility
    path_config_copy = output_dir / "config_copy.yaml"
    config.saveYaml(str(path_config_copy))

    return output_dir


class VisibleColorFormatter(logging.Formatter):
    """Custom logging formatter with color coding and enhanced location information.

    This formatter enhances log readability by:
    - Color coding log levels (red for warnings/errors, default for info)
    - Including clickable file paths with line numbers
    - Showing function names for better debugging context
    - Using ANSI color codes for terminal output

    The formatter is designed to work well in development environments where
    colored terminal output improves log readability and debugging efficiency.
    """

    def format(self, record):
        """Format a log record with colors and enhanced location info.

        Args:
            record (LogRecord): The log record to format

        Returns:
            str: Formatted log message with colors and location information

        """
        # ANSI color codes
        RESET = "\033[0m"

        # Apply color coding based on log level
        if record.levelname == "INFO":
            level_color = ""
            levelname = ""  # Don't show [INFO] prefix to reduce clutter
        else:
            level_color = "\033[91m"  # Red for warnings and errors
            levelname = f"{level_color}[{record.levelname}]{RESET}"

        # Create clickable file location (relative path + line number)
        location_color = "\033[94m"  # Blue for file locations
        relpath = os.path.relpath(record.pathname, os.getcwd())
        location = f"{location_color}{relpath}:{record.lineno}{RESET}"

        # Add function name for debugging context
        func_color = "\033[96m"  # Cyan for function names
        funcname = f"{func_color}({record.funcName}){RESET}"

        # Combine all components into final log message
        message = record.getMessage()
        return f"{levelname}[{location}{funcname}]{message}"


def make_logger(output_dir: Path):
    """Set up logging with both file and console output.

    Creates a dual logging setup:
    - Plain text logging to file for permanent record keeping
    - Colored console logging for better development experience

    The file logger captures all output in a clean format suitable for
    analysis, while the console logger uses colors and enhanced formatting
    for better readability during development and debugging.

    Args:
        output_dir (Path): Directory where the log file will be created

    Returns:
        logging.Logger: Configured logger instance

    """
    # Create log file path
    log_file = output_dir / "result.log"

    # Create custom formatter for colored terminal output
    formatter = VisibleColorFormatter()

    # Set up dual handlers: file (plain) and console (colored)
    handlers = [
        logging.FileHandler(log_file),  # Plain text for file storage
        logging.StreamHandler(),  # Colored output for terminal
    ]

    # Apply color formatting only to terminal output
    # File output remains plain for easier parsing and analysis
    handlers[1].setFormatter(formatter)

    # Configure basic logging settings
    logging.basicConfig(level=logging.INFO, handlers=handlers)

    # Return logger for this module
    logger = logging.getLogger(__name__)
    return logger


def make_checkpoint_callback(config: ExperimentConfig, output_dir: Path):
    """Create a PyTorch Lightning checkpoint callback for model saving.

    Configures automatic model checkpointing based on validation metrics.
    The callback saves the best performing models according to the specified
    monitoring metric and also maintains the most recent checkpoint.

    Args:
        config (ExperimentConfig): Configuration containing checkpoint settings
        output_dir (Path): Base output directory for the experiment

    Returns:
        ModelCheckpoint: Configured checkpoint callback

    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "ckpts",  # Subdirectory for checkpoints
        filename="{epoch:03d}-{step:06d}-{val_loss:.6f}",  # Descriptive filename format
        monitor=config.training.monitor_metric,  # Metric to track (e.g., 'val_loss')
        mode=config.training.monitor_mode,  # 'min' for loss, 'max' for accuracy
        save_top_k=config.training.save_top_k,  # Number of best models to keep
        save_last=True,  # Always save the most recent checkpoint
    )
    return checkpoint_callback


def make_tensorboard_logger(config: ExperimentConfig, output_dir: Path):
    """Create a PyTorch Lightning TensorBoard logger for experiment tracking.

    Sets up TensorBoard logging for monitoring training progress, visualizing
    metrics, and tracking hyperparameters. The logger creates timestamped
    subdirectories to organize different experiment runs.

    Args:
        config (ExperimentConfig): Experiment configuration (currently unused but kept for API consistency)
        output_dir (Path): Base output directory for the experiment

    Returns:
        TensorBoardLogger: Configured TensorBoard logger

    """
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,  # Base directory for TensorBoard logs
        name="tb_logs",  # Subdirectory name for TensorBoard files
        default_hp_metric=False,  # Disable default hyperparameter metric logging
        version=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),  # Unique version ID
    )
    return tb_logger


def make_early_stopping_callback(config: ExperimentConfig):
    """Create a PyTorch Lightning early stopping callback to prevent overfitting.

    Monitors a specified metric and stops training when no improvement is observed
    for a given number of epochs (patience). This helps prevent overfitting and
    saves computational resources.

    Args:
        config (ExperimentConfig): Configuration containing early stopping parameters

    Returns:
        EarlyStopping: Configured early stopping callback

    """
    early_stopping_callback = EarlyStopping(
        monitor=config.training.monitor_metric,  # Metric to monitor (e.g., 'val_loss')
        patience=config.training.early_stopping_patience,  # Epochs to wait for improvement
        min_delta=config.training.early_stopping_min_delta,  # Minimum change to qualify as improvement
        mode=config.training.early_stopping_mode,  # 'min' for loss, 'max' for accuracy
        verbose=True,  # Print early stopping messages
    )
    return early_stopping_callback
