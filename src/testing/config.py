"""Configuration for CSI Prediction Model Testing.

This module contains configuration parameters and utility functions for comprehensive
testing of CSI prediction models. It defines testing scenarios, model lists, noise
configurations, and job allocation settings for distributed testing on HPC clusters.

Key Components:
- Model configurations: List of models to test (MODEL, LLM4CP, CNN, RNN, NP, STEMGNN)
- Scenario configurations: TDD and FDD testing scenarios
- Noise configurations: Various noise types (phase, burst, vanilla, packagedrop)
- Job allocation: SLURM array job configuration for parallel testing
- Combination generation: Creates all testing parameter combinations

The module supports both regular testing (27 combinations per scenario) and
generalization testing (510 combinations per scenario) across different:
- Channel models: A, C, D (regular) / A, B, C, D, E (generalization)
- Delay spreads: 30-400 nanoseconds
- Mobility speeds: 1-45 m/s
- Noise types: Realistic noise models calibrated to target SNRs

Usage:
    from src.testing.config import LIST_MODELS, create_all_combinations

    # Get all testing combinations
    all_combinations = create_all_combinations()

    # Slice for array job
    job_combinations = slice_combinations(all_combinations, (job_id, total_jobs))
"""

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
    """Create all testing combinations for both regular and generalization testing.

    This function generates the complete set of testing parameter combinations used
    across all models. It creates combinations for both regular testing (27 combinations
    per scenario) and generalization testing (510 combinations per scenario).

    The combinations include:
    - Regular testing: All noise types (phase, burst, vanilla, packagedrop)
    - Generalization testing: Only vanilla noise for robustness evaluation
    - Both TDD and FDD scenarios
    - All channel models, delay spreads, and mobility speeds

    Returns:
        list: All combinations as tuples of (scenario, is_gen, noise_type, noise_degree, cm, ds, ms)
            - scenario: "TDD" or "FDD"
            - is_gen: Boolean indicating generalization testing
            - noise_type: Type of noise ("phase", "burst", "vanilla", "packagedrop")
            - noise_degree: Noise parameter (SNR or noise degree)
            - cm: Channel model ("A", "B", "C", "D", "E")
            - ds: Delay spread in seconds (30e-9 to 400e-9)
            - ms: Mobility speed in m/s (1 to 45)

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
    """Slice the combinations list based on SLURM array job allocation.

    This function divides the complete list of testing combinations into smaller
    chunks for parallel processing on HPC clusters. It ensures balanced distribution
    of work across array jobs while handling remainder combinations properly.

    Args:
        list_all_combs (list): Complete list of testing combinations
        slice_info (tuple): Tuple of (slice_idx, total_jobs)
            - slice_idx: Current array job index (0-based)
            - total_jobs: Total number of array jobs

    Returns:
        list: Subset of combinations assigned to this specific array job

    Note:
        The function distributes remainder combinations among the first jobs
        to ensure all combinations are processed exactly once.

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
    """Log current GPU memory usage for monitoring and debugging.

    This utility function logs GPU memory statistics including allocated memory,
    reserved memory, and peak memory usage. Useful for monitoring memory usage
    during testing and identifying potential memory leaks.

    Args:
        logger (logging.Logger, optional): Logger instance to use for output.
            If None, prints to stdout.

    Note:
        - Requires CUDA to be available for GPU memory monitoring
        - Falls back gracefully if PyTorch or CUDA is not available
        - Memory values are reported in GB for readability

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
