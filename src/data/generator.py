"""
CSI Dataset Generator Module

This module provides functionality for generating large-scale CSI (Channel State Information)
datasets using 3GPP TR 38.901 channel models. It supports parallel processing through SLURM
job arrays and generates data for training, regular testing and generalization evaluation.

The generator creates three types of CSI data:
1. Uplink historical CSI (H_U_hist.pt) - for model input
2. Uplink prediction CSI (H_U_pred.pt) - for prediction targets  
3. Downlink prediction CSI (H_D_pred.pt) - for cross-link prediction

Key Features:
- Batch-wise generation to manage memory usage
- Support for multiple channel models (A, B, C, D, E)
- Configurable delay spreads and mobility scenarios
- Automatic data organization for training/testing splits
- Memory-efficient processing with garbage collection

Usage:
    python3 -m src.data.generator.py --is_train              # Generate training data
    python3 -m src.data.generator.py                         # Generate regular test data
    python3 -m src.data.generator.py --is_gen                # Generate generalization test data
    python3 -m src.data.generator.py --debug --is_train      # Debug mode: minimal training data
    python3 -m src.data.generator.py --debug                 # Debug mode: minimal test data
    python3 -m src.data.generator.py --debug --is_gen        # Debug mode: minimal generalization data
"""

import gc
from pathlib import Path
from typing import Any, List, Tuple

from tqdm import tqdm

from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, PRED_LEN, TOT_ANTENNAS


def check_exist(dir_folder: Path) -> bool:
    """
    Check if all required CSI data files exist in the specified directory.
    
    Args:
        dir_folder (Path): Directory path to check for CSI data files
    
    Returns:
        bool: True if all three CSI data files exist (H_U_hist.pt, H_U_pred.pt, H_D_pred.pt),
              False otherwise
    
    Note:
        This function checks for the standard CSI file naming convention:
        - H_U_hist.pt: Uplink historical CSI data
        - H_U_pred.pt: Uplink prediction target CSI data  
        - H_D_pred.pt: Downlink prediction target CSI data
    """
    path_h_u_hist = dir_folder / "H_U_hist.pt"
    path_h_u_pred = dir_folder / "H_U_pred.pt"
    path_h_d_pred = dir_folder / "H_D_pred.pt"

    if path_h_u_hist.exists() and path_h_u_pred.exists() and path_h_d_pred.exists():
        return True
    else:
        return False


def get_slice(
    list_all_combs: List[Tuple[Any, ...]],
    num_total_comb: int,
    array_id: int,
    num_chunks: int,
) -> List[Tuple[Any, ...]]:
    """
    Split a list of combinations into chunks for parallel processing.
    
    This function is designed to work with SLURM job arrays, where each array task
    processes a subset of all possible parameter combinations.
    
    Args:
        list_all_combs (List[Tuple[Any, ...]]): List of all parameter combinations
        num_total_comb (int): Total number of combinations
        array_id (int): Current array task ID (1-indexed)
        num_chunks (int): Total number of chunks/array tasks
    
    Returns:
        List[Tuple[Any, ...]]: Subset of combinations for the current array task
    
    Note:
        - If num_chunks == 1, returns the entire list
        - The last chunk may contain more items if total combinations don't divide evenly
        - Array IDs are expected to be 1-indexed (SLURM convention)
    """
    if num_chunks == 1:
        return list_all_combs

    else:
        chunk_size = num_total_comb // num_chunks
        start_idx = (array_id - 1) * chunk_size
        end_idx = start_idx + chunk_size if array_id < num_chunks else num_total_comb
        return list_all_combs[start_idx:end_idx]


def gen_csi(
    csi_simulator,
    dir_folder: Path,
    batch_size: int,
    num_repeat: int,
) -> None:
    """
    Generate CSI dataset using the provided CSI simulator and save to disk.
    
    This function generates a complete CSI dataset for a specific channel configuration,
    splitting the data into uplink historical, uplink prediction, and downlink prediction
    components. The generated data is saved as PyTorch tensor files.
    
    Args:
        csi_simulator: CSI_Simulator instance configured for specific channel conditions
        dir_folder (Path): Directory where generated CSI files will be saved
        batch_size (int): Number of CSI samples to generate per batch
        num_repeat (int): Total number of CSI samples to generate
    
    Returns:
        None
    
    Raises:
        AssertionError: If num_repeat is not divisible by batch_size
    
    Generated Files:
        - H_U_hist.pt: Uplink historical CSI [num_repeat, 32, 16, 300]
        - H_U_pred.pt: Uplink prediction CSI [num_repeat, 32, 4, 300]
        - H_D_pred.pt: Downlink prediction CSI [num_repeat, 32, 4, 300]
    
    Data Dimensions:
        - NUM_ANTENNAS = 32 = 4 * 4 * 2 (dual polarized BS antenna array)
        - HIST_LEN (16) + PRED_LEN (4) = 20 total time slots
        - NUM_SUBCARRIERS (300) subcarriers for each UL and DL
        - Total subcarriers = 300 (UL) + 150 (gap) + 300 (DL) = 750
    
    Note:
        The function processes data in batches to manage memory usage and includes
        garbage collection to prevent memory leaks during large dataset generation.
    """

    assert num_repeat % batch_size == 0, "num_repeat must be divisible by batch_size"
    num_batches = num_repeat // batch_size

    # Initialize tensors to store all generated CSI data
    h_u_hist_all = torch.zeros((num_repeat, TOT_ANTENNAS, HIST_LEN, NUM_SUBCARRIERS), dtype=torch.cfloat)
    h_u_pred_all = torch.zeros((num_repeat, TOT_ANTENNAS, PRED_LEN, NUM_SUBCARRIERS), dtype=torch.cfloat)
    h_d_pred_all = torch.zeros((num_repeat, TOT_ANTENNAS, PRED_LEN, NUM_SUBCARRIERS), dtype=torch.cfloat)

    # Generate CSI data in batches
    for i in tqdm(range(num_batches), desc=str(dir_folder), total=num_batches):
        # Generate full CSI tensor: (batch_size, 32 antennas, 20 time slots, 750 subcarriers)
        h = csi_simulator()  # Shape: [batch_size, NUM_ANTENNAS, HIST_LEN+PRED_LEN, TOT_SUBCARRIERS]

        # Extract uplink subcarriers (first NUM_SUBCARRIERS)
        h_u = h[:, :, :, :NUM_SUBCARRIERS]  # UL subcarriers
        h_u_hist = h_u[:, :, :HIST_LEN, :]  # UL historical data
        h_u_pred = h_u[:, :, HIST_LEN:, :]  # UL prediction targets

        # Extract downlink subcarriers (last NUM_SUBCARRIERS)
        h_d = h[:, :, :, -NUM_SUBCARRIERS:]  # DL subcarriers
        h_d_pred = h_d[:, :, HIST_LEN:, :]  # DL prediction targets

        # Store batch data in the complete tensors
        h_u_hist_all[i * batch_size : (i + 1) * batch_size, :, :, :] = h_u_hist
        h_u_pred_all[i * batch_size : (i + 1) * batch_size, :, :, :] = h_u_pred
        h_d_pred_all[i * batch_size : (i + 1) * batch_size, :, :, :] = h_d_pred

        # Clean up memory after each batch
        del h, h_u, h_u_hist, h_u_pred, h_d, h_d_pred
        gc.collect()

    # Save the generated CSI data to disk
    torch.save(h_u_hist_all, dir_folder / "H_U_hist.pt")
    torch.save(h_u_pred_all, dir_folder / "H_U_pred.pt")
    torch.save(h_d_pred_all, dir_folder / "H_D_pred.pt")

    # Final memory cleanup
    del h_u_hist_all, h_u_pred_all, h_d_pred_all
    gc.collect()

    return


if __name__ == "__main__":
    # Import required modules for dataset generation
    import argparse
    import os
    import random
    from itertools import product

    import numpy as np
    import tensorflow as tf
    import torch

    from src.data.csi_simulator import CSI_Config, CSI_Simulator
    from src.utils.data_utils import (
        BATCH_SIZE,
        BATCH_SIZE_DEBUG,
        LIST_CHANNEL_MODEL,
        LIST_CHANNEL_MODEL_GEN,
        LIST_DELAY_SPREAD,
        LIST_DELAY_SPREAD_GEN,
        LIST_MIN_SPEED_TEST,
        LIST_MIN_SPEED_TEST_GEN,
        LIST_MIN_SPEED_TRAIN,
        NUM_REPEAT_TEST,
        NUM_REPEAT_TEST_DEBUG,
        NUM_REPEAT_TRAIN,
        NUM_REPEAT_TRAIN_DEBUG,
        make_folder_name,
    )
    from src.utils.dirs import DIR_DATA

    # Set random seeds for reproducible dataset generation
    # This ensures that the same CSI data is generated across different runs
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    torch.manual_seed(SEED)

    # Get SLURM job array parameters for parallel processing
    # SLURM_ARRAY_TASK_ID: Current array task ID (1-indexed)
    # SLURM_ARRAY_TASK_MAX: Maximum array task ID (total number of chunks)
    # Defaults to "1" if not running under SLURM (single-node execution)
    ARRAY_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))
    NUM_CHUNKS = int(os.getenv("SLURM_ARRAY_TASK_MAX", "1"))
    print(f"Running chunk {ARRAY_ID} of {NUM_CHUNKS}")

    # Parse command line arguments to determine dataset type
    parser = argparse.ArgumentParser(description="Generate CSI dataset")
    parser.add_argument("--is_train", action="store_true", help="Generate training dataset")
    parser.add_argument("--is_gen", action="store_true", help="Generate testing dataset for generalization")
    parser.add_argument("--debug", action="store_true", help="Debug mode: use small batch size and sample counts for quick testing")
    args = parser.parse_args()

    # Set parameters based on debug mode
    # Debug mode uses minimal resources for quick testing and development
    if args.debug:
        batch_size = BATCH_SIZE_DEBUG
        num_repeat_train = NUM_REPEAT_TRAIN_DEBUG
        num_repeat_test = NUM_REPEAT_TEST_DEBUG
        print("DEBUG MODE: Using minimal batch size and sample counts")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples per scenario: {num_repeat_train}")
        print(f"  Testing samples per scenario: {num_repeat_test}")
    else:
        batch_size = BATCH_SIZE
        num_repeat_train = NUM_REPEAT_TRAIN
        num_repeat_test = NUM_REPEAT_TEST

    # Create output directory for generated datasets
    dir_output = Path(DIR_DATA)
    dir_output.mkdir(parents=True, exist_ok=True)

    # Generate regular dataset (training or testing)
    # Regular datasets use a limited set of channel conditions for controlled evaluation
    if not args.is_gen:
        # Determine dataset parameters based on training/testing mode
        if args.is_train:
            mode = "train"
            list_min_speed = LIST_MIN_SPEED_TRAIN  # Training mobility scenarios
            num_repeat = num_repeat_train          # Number of training samples per scenario (debug-aware)
        else:
            mode = "test"
            list_min_speed = LIST_MIN_SPEED_TEST    # Testing mobility scenarios
            num_repeat = num_repeat_test            # Number of testing samples per scenario (debug-aware)

        # Create directory structure: data/{train|test}/regular/
        dir_mode = dir_output / mode / "regular"
        dir_mode.mkdir(parents=True, exist_ok=True)

        # Generate all combinations of channel parameters for regular dataset
        # Cartesian product of: channel_models × delay_spreads × min_speeds
        list_all_comb = list(
            product(
                LIST_CHANNEL_MODEL,    # ['A', 'C', 'D'] - 3 channel models
                LIST_DELAY_SPREAD,     # [30e-9, 100e-9, 300e-9] - 3 delay spreads
                list_min_speed,        # [1, 10, 30] - 3 mobility scenarios
            )
        )
        num_total_comb = len(LIST_CHANNEL_MODEL) * len(LIST_DELAY_SPREAD) * len(list_min_speed)

        # Distribute combinations across SLURM array tasks for parallel processing
        # Each task processes a subset of all parameter combinations
        slice_comb = get_slice(
            list_all_combs=list_all_comb,
            num_total_comb=num_total_comb,
            array_id=ARRAY_ID,
            num_chunks=NUM_CHUNKS,
        )

        # Process each parameter combination assigned to this array task
        for cm, ds, ms in tqdm(
            slice_comb,
            total=len(slice_comb),
            desc=f"{mode} | {ARRAY_ID}/{NUM_CHUNKS}",
        ):
            # Create standardized folder name for this parameter combination
            folder_name = make_folder_name(cm, ds, ms)
            dir_folder = dir_mode / folder_name
            dir_folder.mkdir(parents=True, exist_ok=True)

            print()
            print("Generating CSI for {} | channel model {} | delay spread {} | min speed {}".format(mode, cm, ds, ms))
            
            # Skip generation if all required files already exist
            if check_exist(dir_folder):
                print("{} | {} already exists, skipping...".format(mode, dir_folder))
                continue
            print()

            # Initialize CSI simulator with current parameter combination
            csi_config = CSI_Config(batch_size=batch_size, cdl_model=cm, delay_spread=ds, min_speed=ms)
            csi_config = csi_config.getConfig()

            # Create simulator instance with the configuration
            csi_simulator = CSI_Simulator(csi_config)

            # Generate CSI dataset for this parameter combination
            # This creates H_U_hist.pt, H_U_pred.pt, and H_D_pred.pt files
            gen_csi(
                csi_simulator=csi_simulator,
                dir_folder=dir_folder,
                batch_size=batch_size,
                num_repeat=num_repeat,
            )

            # print(f"{mode} | {dir_folder}")

    else:
        # Generate generalization dataset for testing model robustness
        # Generalization dataset uses extended parameter ranges to evaluate
        # model performance on unseen channel conditions
        dir_gen = dir_output / "test" / "generalization"
        dir_gen.mkdir(parents=True, exist_ok=True)

        # Generate all combinations for generalization testing
        # Extended parameter ranges compared to regular dataset:
        # - More channel models: ['A', 'B', 'C', 'D', 'E'] (5 models)
        # - More delay spreads: [30e-9, 50e-9, 100e-9, 200e-9, 300e-9, 400e-9] (6 values)
        # - More mobility scenarios: [*range(3, 46, 3), 1, 10]] (17 values)
        list_all_comb = list(
            product(
                LIST_CHANNEL_MODEL_GEN,   # Extended channel model list
                LIST_DELAY_SPREAD_GEN,    # Extended delay spread range
                LIST_MIN_SPEED_TEST_GEN,  # Extended mobility range
            )
        )
        num_total_comb = len(LIST_CHANNEL_MODEL_GEN) * len(LIST_DELAY_SPREAD_GEN) * len(LIST_MIN_SPEED_TEST_GEN)

        # Distribute generalization combinations across array tasks
        slice_comb = get_slice(
            list_all_combs=list_all_comb,
            num_total_comb=num_total_comb,
            array_id=ARRAY_ID,
            num_chunks=NUM_CHUNKS,
        )

        # Process each generalization parameter combination
        for cm, ds, ms in tqdm(
            slice_comb,
            total=len(slice_comb),
            desc=f"generalization | {ARRAY_ID}/{NUM_CHUNKS}",
        ):
            # Create folder for this generalization test scenario
            folder_name = make_folder_name(cm, ds, ms)
            dir_folder = dir_gen / folder_name
            dir_folder.mkdir(parents=True, exist_ok=True)

            print()
            print(
                "Generating CSI for {} | channel model {} | delay spread {} | min speed {}".format(
                    "test gen", cm, ds, ms
                )
            )
            
            # Skip if generalization data already exists for this scenario
            if check_exist(dir_folder):
                print(f"generalization | {dir_folder} already exists, skipping...")
                continue
            print()

            # Configure simulator for generalization test scenario
            csi_config = CSI_Config(batch_size=batch_size, cdl_model=cm, delay_spread=ds, min_speed=ms)
            csi_config = csi_config.getConfig()

            # Create simulator for this generalization scenario
            csi_simulator = CSI_Simulator(csi_config)

            # Generate generalization test data
            # Note: Uses test sample count (not training count) for consistent test set size
            gen_csi(
                csi_simulator=csi_simulator,
                dir_folder=dir_folder,
                batch_size=batch_size,
                num_repeat=num_repeat_test,  # Same number of samples as regular test set (debug-aware)
            )
