from itertools import product

import torch

from src.noise.noise_testing import Noise
from src.utils.data_utils import (
    LIST_CHANNEL_MODEL,
    LIST_CHANNEL_MODEL_GEN,
    LIST_DELAY_SPREAD,
    LIST_DELAY_SPREAD_GEN,
    LIST_MIN_SPEED_TEST,
    LIST_MIN_SPEED_TEST_GEN,
)


# Model configurations
LIST_MODELS = ["MODEL", "LLM4CP", "CNN", "RNN", "NP", "STEMGNN"]  # Consistent order for testing

# Scenario and noise configurations
LIST_SCENARIOS = ["TDD", "FDD"]
LIST_NOISE_TYPES = ["phase", "burst", "vanilla", "packagedrop"]

# Testing configurations
BATCH_SIZE = 1

# Array job configurations
JOBS_PER_MODEL = 20


def create_all_combinations():
    """Create all combinations for both regular and generation testing.
    This is the same for all models.

    Returns:
        list: All combinations as tuples of (is_gen, scenario, noise_type, noise_degree, cm, ds, ms)

    """
    noise = Noise()
    all_combinations = []

    # Regular testing (all noise types)
    for scenario in LIST_SCENARIOS:
        is_gen = False
        list_cm = LIST_CHANNEL_MODEL
        list_ds = LIST_DELAY_SPREAD
        list_ms = LIST_MIN_SPEED_TEST

        for noise_type in LIST_NOISE_TYPES:
            list_noise_degree = (
                getattr(noise, f"list_{noise_type}_snr")
                if noise_type != "packagedrop"
                else getattr(noise, f"list_{noise_type}_nd")
            )

            for noise_degree, cm, ds, ms in product(list_noise_degree, list_cm, list_ds, list_ms):
                all_combinations.append((scenario, is_gen, noise_type, noise_degree, cm, ds, ms))

    # Generation testing (vanilla noise only)
    for scenario in LIST_SCENARIOS:
        is_gen = True
        list_cm = LIST_CHANNEL_MODEL_GEN
        list_ds = LIST_DELAY_SPREAD_GEN
        list_ms = LIST_MIN_SPEED_TEST_GEN

        noise_type = "vanilla"
        list_noise_degree = getattr(noise, f"list_{noise_type}_snr")

        for noise_degree, cm, ds, ms in product(list_noise_degree, list_cm, list_ds, list_ms):
            all_combinations.append((scenario, is_gen, noise_type, noise_degree, cm, ds, ms))

    return all_combinations


def slice_combinations(list_all_combs, slice_info):
    """Slice the combinations list based on array job allocation.

    Args:
        list_all_combs: List of all combinations
        slice_info: Tuple of (slice_idx, total_jobs)

    Returns:
        Sliced list of combinations for this specific array job

    """
    slice_idx, total_jobs = slice_info
    total_combs = len(list_all_combs)

    # Calculate slice boundaries
    combs_per_job = total_combs // total_jobs
    remainder = total_combs % total_jobs

    # Distribute remainder among first jobs
    if slice_idx < remainder:
        start_idx = slice_idx * (combs_per_job + 1)
        end_idx = start_idx + combs_per_job + 1
    else:
        start_idx = remainder * (combs_per_job + 1) + (slice_idx - remainder) * combs_per_job
        end_idx = start_idx + combs_per_job

    return list_all_combs[start_idx:end_idx]


def log_gpu_memory_usage(logger=None):
    """Log current GPU memory usage if available.

    Args:
        logger: Logger instance to use (optional)

    """
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

            memory_info = (
                f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB"
            )

            if logger:
                logger.info(memory_info)
            else:
                print(memory_info)
        else:
            if logger:
                logger.info("GPU not available")
            else:
                print("GPU not available")
    except ImportError:
        if logger:
            logger.warning("PyTorch not available for memory monitoring")
        else:
            print("PyTorch not available for memory monitoring")
