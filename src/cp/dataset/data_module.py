import logging
import random
from pathlib import Path

import lightning.pytorch as pl
import torch
from src.noise.noise import gen_vanilla_noise_snr
from src.utils.normalization import normalize_input
from torch.utils.data import DataLoader

from src.cp.config.config import DataConfig
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


logger = logging.getLogger(__name__)


class TrainValDataModule(pl.LightningDataModule):
    def __init__(self, data_config: DataConfig):
        super().__init__()
        self.data_cfg = data_config

    def setup(self, stage=None):
        """Setup datasets for training and validation with subset-aware splitting."""
        # Use subset-stratified splitting to ensure all 27 subsets are represented in both train and val
        self._setup_subset_stratified_split()

    def _setup_subset_stratified_split(self):
        """Load data with subset-stratified splitting.
        Each of the 27 subsets contributes proportionally to both train and validation sets.
        This ensures representative validation without data leakage.
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
        for cm in LIST_CHANNEL_MODEL:
            for ds in LIST_DELAY_SPREAD:
                for ms in LIST_MIN_SPEED_TRAIN:
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

        # Concatenate all training data
        H_hist_train = torch.cat(H_hist_train_list, dim=0)
        H_pred_train = torch.cat(H_pred_train_list, dim=0)

        # Concatenate all validation data
        H_hist_val = torch.cat(H_hist_val_list, dim=0)
        H_pred_val = torch.cat(H_pred_val_list, dim=0)

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
        if self.data_cfg.is_separate_antennas:
            collate_fn = collect_fn_separate_antennas
        else:
            collate_fn = collect_fn_gather_antennas

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=self.data_cfg.shuffle,
            num_workers=0,  # fix num_workers to 0 to avoid BUS ERROR
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        if self.data_cfg.is_separate_antennas:
            collate_fn = collect_fn_separate_antennas
        else:
            collate_fn = collect_fn_gather_antennas

        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=0,  # fix num_workers to 0 to avoid BUS ERROR
            collate_fn=collate_fn,
        )

    def get_data_shapes(self):
        """Return (history_shape, pred_shape) of a single sample from train_dataset.
        If train_dataset is not yet set up or empty, returns (None, None).
        """
        if not hasattr(self, "train_dataset") or self.train_dataset is None or len(self.train_dataset) == 0:
            return None, None

        # Grab one sample (use __getitem__(0))
        loader = self.train_dataloader()
        hist_batch, pred_batch = next(iter(loader))
        # Get shapes
        return hist_batch.shape, pred_batch.shape


if __name__ == "__main__":
    # Example usage
    data_config = DataConfig()
    data_module = TrainValDataModule(data_config)
    data_module.setup()

    print("✅ Subset-stratified data module created successfully!")
    print(f"   🏋️ Training samples: {len(data_module.train_dataset)}")
    print(f"   🔍 Validation samples: {len(data_module.val_dataset)}")
    print("   📊 All 27 subsets represented in both train and validation")
s
