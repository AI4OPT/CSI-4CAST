"""PyTorch Lightning Data Module for CSI Prediction.

This module implements a comprehensive data loading and preprocessing pipeline for CSI prediction
models using PyTorch Lightning. It handles subset-stratified data splitting, normalization,
noise addition, and efficient data loading for training and validation.

Key Features:
- Subset-stratified splitting: Ensures all 27 parameter combinations are represented in both
  training and validation sets while preventing data leakage
- Automatic data normalization using pre-computed statistics
- Additional White Gaussian Noise during training and validation
- Flexible antenna handling (separate vs. gathered antennas)
- Memory-efficient data loading with proper collation functions

The data module loads CSI data from multiple scenarios defined by:
- Channel models: A, C, D (3 models)
- Delay spreads: 30ns, 100ns, 300ns (3 values)
- Mobility speeds: 1, 10, 30 m/s (3 values)
Total: 3 * 3 * 3 = 27 parameter combinations

Usage:
    from src.cp.dataset.data_module import TrainValDataModule

    data_module = TrainValDataModule(data_config)
    data_module.setup()
"""

from itertools import product
import logging
from pathlib import Path
import random

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from src.cp.config.config import DataConfig
from src.noise.noise import gen_vanilla_noise_snr
from src.utils.data_utils import (
    LIST_CHANNEL_MODEL,
    LIST_DELAY_SPREAD,
    LIST_MIN_SPEED_TRAIN,
    SNR_RANGE_GAUSSIAN_NOISE_TRAIN,
    CSIDataset,
    _load_data,
    collect_fn_gather_antennas,
    collect_fn_separate_antennas,
)
from src.utils.norm_utils import normalize_input


logger = logging.getLogger(__name__)


class TrainValDataModule(pl.LightningDataModule):
    """PyTorch Lightning Data Module for CSI Training and Validation.

    This data module implements subset-stratified data splitting to ensure that all 27
    parameter combinations (channel model * delay spread * mobility speed) are represented
    proportionally in both training and validation sets. This approach prevents data leakage
    while ensuring representative validation performance.

    The module handles:
    - Data loading from multiple CSI scenarios
    - Subset-stratified train/validation splitting
    - Data normalization using pre-computed statistics
    - Additional White Gaussian Noise for realistic training conditions
    - Flexible data loading with configurable batch sizes and collation functions

    Args:
        data_config (DataConfig): Configuration object containing data loading parameters

    """

    def __init__(self, data_config: DataConfig):
        """Initialize the data module with configuration.

        Args:
            data_config (DataConfig): Data configuration containing:
                - dir_dataset: Path to dataset directory
                - batch_size: Batch size for data loaders
                - train_ratio: Ratio of data to use for training (0.0-1.0)
                - is_U2D: Whether to use U2D (FDD) or TDD scenario
                - is_separate_antennas: Whether to process antennas separately
                - shuffle: Whether to shuffle training data

        """
        super().__init__()
        self.data_cfg = data_config

    def setup(self, stage=None):
        """Set up datasets for training and validation with subset-stratified splitting."""
        self._setup_subset_stratified_split()

    def _setup_subset_stratified_split(self):
        """Load data with subset-stratified splitting.

        This method implements a sophisticated data splitting strategy where each of the 27
        parameter combinations (subsets) contributes proportionally to both training and
        validation sets. This ensures:

        1. Representative validation: All scenarios are represented in validation
        2. No data leakage: Training and validation come from the same distributions
        3. Balanced evaluation: Each scenario contributes equally to validation metrics

        The splitting process:
        1. Load each of the 27 subsets individually
        2. Split each subset according to train_ratio using consistent random seeding
        3. Aggregate all training splits and all validation splits
        4. Apply normalization and additional White Gaussian Noise to the aggregated datasets
        """
        H_hist_train_list = []
        H_pred_train_list = []
        H_hist_val_list = []
        H_pred_val_list = []

        total_subsets = 0
        total_train_samples = 0
        total_val_samples = 0

        logger.info(f"🔄 Loading data with subset-stratified splitting (train_ratio={self.data_cfg.train_ratio})")

        # Process each subset individually
        for cm, ds, ms in product(LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN):
            # Load this specific subset
            H_hist_subset = _load_data(
                dir_data=Path(self.data_cfg.dir_dataset),
                cm=cm,
                ds=ds,
                ms=ms,
                is_train=True,
                is_gen=False,
                is_hist=True,
                is_U2D=self.data_cfg.is_U2D,
            )

            H_pred_subset = _load_data(
                dir_data=Path(self.data_cfg.dir_dataset),
                cm=cm,
                ds=ds,
                ms=ms,
                is_train=True,
                is_gen=False,
                is_hist=False,
                is_U2D=self.data_cfg.is_U2D,
            )

            # Split this subset according to train_ratio
            subset_size = len(H_hist_subset)
            train_size = int(subset_size * self.data_cfg.train_ratio)

            # Use consistent indexing for reproducible splits
            indices = torch.randperm(subset_size, generator=torch.Generator().manual_seed(42))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Split the subset
            H_hist_train_subset = H_hist_subset[train_indices]
            H_pred_train_subset = H_pred_subset[train_indices]
            H_hist_val_subset = H_hist_subset[val_indices]
            H_pred_val_subset = H_pred_subset[val_indices]

            # Add to respective lists
            H_hist_train_list.append(H_hist_train_subset)
            H_pred_train_list.append(H_pred_train_subset)
            H_hist_val_list.append(H_hist_val_subset)
            H_pred_val_list.append(H_pred_val_subset)

            # Track statistics
            total_subsets += 1
            total_train_samples += len(H_hist_train_subset)
            total_val_samples += len(H_hist_val_subset)

            logger.info(
                f"📊 Subset CM={cm}, DS={round(ds * 1e9)}ns, MS={ms}: "
                f"{subset_size} total → {len(H_hist_train_subset)} train, {len(H_hist_val_subset)} val"
            )

            # Free memory for this subset's original data
            del (
                H_hist_subset,
                H_pred_subset,
                H_hist_train_subset,
                H_pred_train_subset,
                H_hist_val_subset,
                H_pred_val_subset,
            )

        # Concatenate all training data
        H_hist_train = torch.cat(H_hist_train_list, dim=0)
        H_pred_train = torch.cat(H_pred_train_list, dim=0)

        # Concatenate all validation data
        H_hist_val = torch.cat(H_hist_val_list, dim=0)
        H_pred_val = torch.cat(H_pred_val_list, dim=0)

        # Free memory by deleting intermediate lists
        del H_hist_train_list, H_pred_train_list, H_hist_val_list, H_pred_val_list

        logger.info("✅ Stratified split completed:")
        logger.info(f"   📈 {total_subsets} subsets processed")
        logger.info(f"   🏋️ Training: {total_train_samples} samples from ALL subsets")
        logger.info(f"   🔍 Validation: {total_val_samples} samples from ALL subsets")
        logger.info(f"   📏 Data shapes: hist={H_hist_train[0].shape}, pred={H_pred_train[0].shape}")

        # Normalize the input data
        H_hist_train, H_pred_train = normalize_input(H_hist_train, H_pred_train, is_U2D=self.data_cfg.is_U2D)
        H_hist_val, H_pred_val = normalize_input(H_hist_val, H_pred_val, is_U2D=self.data_cfg.is_U2D)

        assert H_pred_train is not None, "H_pred_train should not be None"
        assert H_pred_val is not None, "H_pred_val should not be None"

        # Add noise to both training and validation data (realistic scenario)
        logger.info("🔊 Adding noise to training data...")
        for i in range(len(H_hist_train)):
            snr_val = random.uniform(*SNR_RANGE_GAUSSIAN_NOISE_TRAIN)
            H_hist_train[i] = H_hist_train[i] + gen_vanilla_noise_snr(H_hist_train[i], SNR=snr_val)

        logger.info("🔊 Adding noise to validation data...")
        for i in range(len(H_hist_val)):
            snr_val = random.uniform(*SNR_RANGE_GAUSSIAN_NOISE_TRAIN)
            H_hist_val[i] = H_hist_val[i] + gen_vanilla_noise_snr(H_hist_val[i], SNR=snr_val)

        # Create datasets
        self.train_dataset = CSIDataset(H_hist_train, H_pred_train)
        self.val_dataset = CSIDataset(H_hist_val, H_pred_val)

        logger.info(f"🎯 Final datasets: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")

    def train_dataloader(self):
        """Create training data loader.

        Returns:
            DataLoader: PyTorch DataLoader for training data with appropriate
                       collation function based on antenna configuration

        """
        # Select collation function based on antenna processing mode
        collect_fn = collect_fn_separate_antennas if self.data_cfg.is_separate_antennas else collect_fn_gather_antennas

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=self.data_cfg.shuffle,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on some systems
            collate_fn=collect_fn,
        )

    def val_dataloader(self):
        """Create validation data loader.

        Returns:
            DataLoader: PyTorch DataLoader for validation data with appropriate
                       collation function based on antenna configuration

        """
        # Select collation function based on antenna processing mode
        collect_fn = collect_fn_separate_antennas if self.data_cfg.is_separate_antennas else collect_fn_gather_antennas

        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on some systems
            collate_fn=collect_fn,
        )

    def get_data_shapes(self):
        """Get the shapes of data tensors from the training dataset.

        This method is useful for model initialization and debugging to understand
        the expected input and output tensor dimensions.

        Returns:
            tuple: (history_shape, prediction_shape) where:
                - history_shape: Shape of historical CSI input tensors
                - prediction_shape: Shape of prediction target tensors
                - Returns (None, None) if dataset is not yet set up

        """
        if not hasattr(self, "train_dataset") or self.train_dataset is None or len(self.train_dataset) == 0:
            return None, None

        # Get a batch from the training loader to determine shapes
        loader = self.train_dataloader()
        hist_batch, pred_batch = next(iter(loader))

        return hist_batch.shape, pred_batch.shape
