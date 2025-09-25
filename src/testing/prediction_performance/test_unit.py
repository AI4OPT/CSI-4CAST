import gc
import logging
from pathlib import Path
from typing import Callable, List

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.data_utils import CSIDataset, collect_fn_gather_antennas, collect_fn_separate_antennas, load_data
from src.utils.normalization import denormalize_output, normalize_input

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
    list_models: List[pl.LightningModule | nn.Module],
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
    # load data
    h_hist = load_data(
        dir_data=dir_data,
        list_cm=[cm],
        list_ds=[ds],
        list_ms=[ms],
        is_train=False,
        is_gen=is_gen,
        is_hist=True,
        is_U2D=is_U2D,
    )

    h_pred = load_data(
        dir_data=dir_data,
        list_cm=[cm],
        list_ds=[ds],
        list_ms=[ms],
        is_train=False,
        is_gen=is_gen,
        is_hist=False,
        is_U2D=is_U2D,
    )

    # normalize the data
    h_hist, h_pred = normalize_input(h_hist, h_pred, is_U2D=is_U2D)

    # add the noise sample-by-sample (matching training behavior)
    if noise_type == "vanilla":
        # Apply noise individually to each sample to match training behavior
        for i in range(len(h_hist)):
            h_hist[i] = h_hist[i] + noise_func(h_hist[i], noise_degree)
    else:
        # For other noise types, keep the original batch-wise application
        h_hist = h_hist + noise_func(h_hist, noise_degree)

    # dataset and dataloader
    test_dataset = CSIDataset(h_hist, h_pred)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collect_fn_gather_antennas,  # use the default collate function
    )

    test_dataloader_separate_antennas = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collect_fn_separate_antennas,
    )

    # testing
    list_records = []

    for model in list_models:
        list_nmse = []
        list_mse = []
        list_se = []
        list_se0 = []

        if model.is_separate_antennas:
            dataloader = test_dataloader_separate_antennas
        else:
            dataloader = test_dataloader

        for hist, target in dataloader:
            hist, target = hist.to(device), target.to(device)
            # logger.info(f"model - {model.name} | hist.shape - {hist.shape} | target.shape - {target.shape}")

            pred = model(hist)
            # logger.info(f"model - {model.name} | pred.shape - {pred.shape}")

            # Denormalize both prediction and target to original scale before computing loss metrics
            pred_orig, target_orig = denormalize_output(pred, target, is_U2D=is_U2D)

            # Compute loss metrics on original scale
            nmse = criterion_nmse(pred_orig, target_orig, by_step=True)
            mse = criterion_mse(pred_orig, target_orig, by_step=True)
            se, se0 = criterion_se(pred_orig, target_orig, by_step=True)

            list_nmse.append(nmse.detach().cpu())
            list_mse.append(mse.detach().cpu())
            list_se.append(se.detach().cpu())
            list_se0.append(se0.detach().cpu())

        nmse_mean = np.mean(list_nmse, axis=0)
        mse_mean = np.mean(list_mse, axis=0)
        se_mean = np.mean(list_se, axis=0)
        se0_mean = np.mean(list_se0, axis=0)

        nmse_std = np.std(list_nmse, axis=0)
        mse_std = np.std(list_mse, axis=0)
        se_std = np.std(list_se, axis=0)
        se0_std = np.std(list_se0, axis=0)

        list_records.append(
            {
                "model": model.name,
                "cm": cm,
                "ds": ds,
                "ms": ms,
                "is_gen": is_gen,
                "is_U2D": is_U2D,
                "scenario": scenario,
                "noise_type": noise_type,
                "noise_degree": noise_degree,
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

    # add the records to the dataframe
    if isinstance(df_path, str):
        df_path = Path(df_path)

    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    df_new = pd.DataFrame(list_records)
    df = pd.concat([df, df_new], ignore_index=True)  # type: ignore
    df.to_csv(df_path, index=False)

    # Memory cleanup
    del h_hist, h_pred
    del test_dataset, test_dataloader, test_dataloader_separate_antennas
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return df_new
