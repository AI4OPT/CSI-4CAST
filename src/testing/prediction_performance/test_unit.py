"""Test Unit for Single Combination Performance Evaluation.

This module implements the core testing logic for evaluating model performance
on a single combination of parameters (scenario, noise type, channel model, etc.).
It handles data loading, noise application, model evaluation, and metric computation.

Key Features:
- Loads and preprocesses CSI data for specific test scenarios
- Applies various noise types (vanilla, phase, burst, package drop)
- Supports both separate and gather antenna processing modes
- Computes multiple performance metrics (NMSE, MSE, SE)
- Handles result aggregation and CSV output
- Efficient memory management with cleanup

The test_unit function is the main entry point called by the orchestration script
for each parameter combination that needs to be evaluated.
"""

from collections.abc import Callable
import gc
import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.data_utils import CSIDataset, collect_fn_gather_antennas, collect_fn_separate_antennas, load_data
from src.utils.norm_utils import denormalize_output, normalize_input


logger = logging.getLogger(__name__)


def test_unit(
    # data input
    scenario: str,
    is_gen: bool,
    is_U2D: bool,
    dir_data: Path,
    batch_size: int,
    device: torch.device,
    # model
    list_models: list[pl.LightningModule | nn.Module],
    # criterion
    criterion_nmse: nn.Module,
    criterion_mse: nn.Module,
    criterion_se: nn.Module,
    # scenario
    cm: str,  # channel model
    ds: float,  # delay spread
    ms: int,  # min speed
    # noise
    noise_type: str,
    noise_func: Callable,
    noise_degree: int | float,  # could be snr, or the precentage for packagedrop
    df_path: Path | str,
):
    """Evaluate model performance on a single parameter combination.

    This function performs complete evaluation including data loading, preprocessing,
    noise application, model inference, metric computation, and result storage.

    Args:
        # Data parameters
        scenario (str): Test scenario ("TDD" or "FDD")
        is_gen (bool): Whether to use generalization test data
        is_U2D (bool): Whether to use U2D (FDD) data format
        dir_data (Path): Root directory containing test data
        batch_size (int): Batch size for data loading
        device (torch.device): Compute device (CPU/GPU)

        # Model parameters
        list_models (list): List of models to evaluate (typically single model)

        # Loss functions
        criterion_nmse (nn.Module): NMSE loss function
        criterion_mse (nn.Module): MSE loss function
        criterion_se (nn.Module): SE loss function

        # Scenario parameters
        cm (str): Channel model identifier ("A", "C", "D", etc.)
        ds (float): Delay spread in seconds (30e-9, 100e-9, 300e-9)
        ms (int): Minimum speed in km/h (1, 10, 30)

        # Noise parameters
        noise_type (str): Type of noise ("vanilla", "phase", "burst", "packagedrop")
        noise_func (Callable): Noise generation function
        noise_degree (int | float): Noise strength (SNR for most types, drop rate for packagedrop)
        df_path (Path | str): Path to CSV file for storing results

    Returns:
        pd.DataFrame: Results dataframe with computed metrics

    Note:
        Results include mean and std for NMSE, MSE, SE, and SE0 metrics computed
        across all batches and prediction time steps.

    """
    # === DATA LOADING ===
    # Load historical CSI data for model input
    h_hist = load_data(
        dir_data=dir_data,
        list_cm=[cm],
        list_ds=[ds],
        list_ms=[ms],
        is_train=False,  # Use test data
        is_gen=is_gen,  # Generalization vs regular test data
        is_hist=True,  # Load historical (input) data
        is_U2D=is_U2D,  # Data format flag
    )

    # Load prediction target CSI data
    h_pred = load_data(
        dir_data=dir_data,
        list_cm=[cm],
        list_ds=[ds],
        list_ms=[ms],
        is_train=False,  # Use test data
        is_gen=is_gen,  # Generalization vs regular test data
        is_hist=False,  # Load prediction (target) data
        is_U2D=is_U2D,  # Data format flag
    )

    # === DATA PREPROCESSING ===
    # Normalize both historical and prediction data to standard range
    h_hist, h_pred = normalize_input(h_hist, h_pred, is_U2D=is_U2D)

    # === NOISE APPLICATION ===
    # Apply noise to historical data (targets remain clean for evaluation)
    if noise_type == "vanilla":
        # Apply Gaussian noise individually to each sample (matches training behavior)
        for i in range(len(h_hist)):
            h_hist[i] = h_hist[i] + noise_func(h_hist[i], noise_degree)
    else:
        # For other noise types (phase, burst, packagedrop), apply batch-wise
        h_hist = h_hist + noise_func(h_hist, noise_degree)

    # === DATASET AND DATALOADER SETUP ===
    # Create dataset combining historical inputs and prediction targets
    test_dataset = CSIDataset(h_hist, h_pred)

    # Create dataloader for "gather antennas" mode (default)
    # Data shape: [batch_size, num_antennas, hist_len, num_subcarriers]
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic order for reproducible results
        num_workers=0,  # Single-threaded for stability
        collate_fn=collect_fn_gather_antennas,
    )

    # Create dataloader for "separate antennas" mode
    # Data shape: [batch_size, hist_len, num_subcarriers*num_antennas]
    test_dataloader_separate_antennas = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collect_fn_separate_antennas,
    )

    # === MODEL EVALUATION LOOP ===
    list_records = []

    for model in list_models:
        # Initialize metric storage for current model
        list_nmse = []
        list_mse = []
        list_se = []
        list_se0 = []

        # Select appropriate dataloader based on model's antenna processing mode
        dataloader = test_dataloader_separate_antennas if model.is_separate_antennas else test_dataloader

        # Process all test batches
        for hist, target in dataloader:
            # Move data to compute device
            hist, target = hist.to(device), target.to(device)
            # Optional debug logging (commented for performance)
            # logger.info(f"model - {model.name} | hist.shape - {hist.shape} | target.shape - {target.shape}")

            # Generate predictions
            pred = model(hist)
            # Optional debug logging (commented for performance)
            # logger.info(f"model - {model.name} | pred.shape - {pred.shape}")

            # === METRIC COMPUTATION ===
            # Denormalize both prediction and target to original scale for fair evaluation
            pred_orig, target_orig = denormalize_output(pred, target, is_U2D=is_U2D)

            # Compute performance metrics on original (denormalized) scale
            nmse = criterion_nmse(pred_orig, target_orig, by_step=True)
            mse = criterion_mse(pred_orig, target_orig, by_step=True)
            se, se0 = criterion_se(pred_orig, target_orig, by_step=True)

            # Store metrics (move to CPU to free GPU memory)
            list_nmse.append(nmse.detach().cpu())
            list_mse.append(mse.detach().cpu())
            list_se.append(se.detach().cpu())
            list_se0.append(se0.detach().cpu())

        # === STATISTICAL AGGREGATION ===
        # Compute mean metrics across all batches and time steps
        nmse_mean = np.mean(list_nmse, axis=0)
        mse_mean = np.mean(list_mse, axis=0)
        se_mean = np.mean(list_se, axis=0)
        se0_mean = np.mean(list_se0, axis=0)

        # Compute standard deviation metrics across all batches and time steps
        nmse_std = np.std(list_nmse, axis=0)
        mse_std = np.std(list_mse, axis=0)
        se_std = np.std(list_se, axis=0)
        se0_std = np.std(list_se0, axis=0)

        # === RECORD CREATION ===
        # Create comprehensive result record for current model and parameter combination
        list_records.append(
            {
                # Model identification
                "model": model.name,
                # Test scenario parameters
                "cm": cm,  # Channel model
                "ds": ds,  # Delay spread
                "ms": ms,  # Minimum speed
                "is_gen": is_gen,  # Generalization test flag
                "is_U2D": is_U2D,  # Data format flag
                "scenario": scenario,  # TDD/FDD scenario
                # Noise parameters
                "noise_type": noise_type,  # Type of applied noise
                "noise_degree": noise_degree,  # Noise strength/degree
                # Performance metrics (mean and std across time steps)
                "nmse_mean": nmse_mean,
                "nmse_std": nmse_std,
                "mse_mean": mse_mean,
                "mse_std": mse_std,
                "se_mean": se_mean,
                "se_std": se_std,
                "se0_mean": se0_mean,
                "se0_std": se0_std,
            }
        )

    # === RESULT STORAGE ===
    # Ensure df_path is a Path object
    if isinstance(df_path, str):
        df_path = Path(df_path)

    # Load existing results or create new dataframe
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        # Create new dataframe if file doesn't exist
        df = pd.DataFrame()

    # Convert new records to dataframe and append to existing results
    df_new = pd.DataFrame(list_records)
    df = pd.concat([df, df_new], ignore_index=True)  # type: ignore

    # Save updated results to CSV
    df.to_csv(df_path, index=False)

    # === MEMORY CLEANUP ===
    # Delete large data structures to free memory
    del h_hist, h_pred
    del test_dataset, test_dataloader, test_dataloader_separate_antennas

    # Force GPU and system memory cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return df_new
