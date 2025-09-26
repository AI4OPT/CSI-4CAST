import argparse
import datetime
import logging
import os
from pathlib import Path

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.cp.config.config import ExperimentConfig


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_csi_pred", "-hcp", type=str, default="src/cp/config/default_config.yaml")
    return parser


def make_config(path_config: str):
    if path_config.endswith(".json"):
        config = ExperimentConfig.fromJson(path_config)
    elif path_config.endswith(".yaml"):
        config = ExperimentConfig.fromYaml(path_config)
    else:
        raise ValueError(f"Unsupported config file format: {path_config}")
    return config


def make_output_dir(config: ExperimentConfig) -> Path:
    # make output directory
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(config.output_dir) / cur_time
    output_dir.mkdir(parents=True, exist_ok=True)
    # save the configuration
    path_config_copy = output_dir / "config_copy.yaml"
    config.saveYaml(str(path_config_copy))
    return output_dir


class VisibleColorFormatter(logging.Formatter):
    def format(self, record):
        RESET = "\033[0m"

        # Level-name coloring stays the same
        if record.levelname == "INFO":
            level_color = ""
            levelname = ""
        else:
            level_color = "\033[91m"
            levelname = f"{level_color}[{record.levelname}]{RESET}"

        # New: combine full path + line into one linkified segment
        location_color = "\033[94m"  # pick your color
        relpath = os.path.relpath(record.pathname, os.getcwd())
        location = f"{location_color}{relpath}:{record.lineno}{RESET}"

        # If you still want funcName, append it after:
        func_color = "\033[96m"
        funcname = f"{func_color}({record.funcName}){RESET}"

        message = record.getMessage()
        return f"{levelname}[{location}{funcname}]{message}"


def make_logger(output_dir: Path):
    log_file = output_dir / "result.log"

    formatter = VisibleColorFormatter()

    handlers = [
        logging.FileHandler(log_file),  # plain log file
        logging.StreamHandler(),  # color to terminal
    ]

    # Only apply color formatter to terminal
    handlers[1].setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=handlers)

    logger = logging.getLogger(__name__)
    return logger


def make_checkpoint_callback(config: ExperimentConfig, output_dir: Path):
    """Create a PyTorch Lightning checkpoint callback."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "ckpts",
        filename="{epoch:03d}-{step:06d}-{val_loss:.6f}",
        monitor=config.training.monitor_metric,
        mode=config.training.monitor_mode,
        save_top_k=config.training.save_top_k,
        save_last=True,
    )
    return checkpoint_callback


def make_tensorboard_logger(config: ExperimentConfig, output_dir: Path):
    """Create a PyTorch Lightning TensorBoard logger."""
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tb_logs",
        default_hp_metric=False,
        version=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    return tb_logger


def make_early_stopping_callback(config: ExperimentConfig):
    """Create a PyTorch Lightning early stopping callback."""
    early_stopping_callback = EarlyStopping(
        monitor=config.training.monitor_metric,
        patience=config.training.early_stopping_patience,
        min_delta=config.training.early_stopping_min_delta,
        mode=config.training.early_stopping_mode,
        verbose=True,
    )
    return early_stopping_callback
