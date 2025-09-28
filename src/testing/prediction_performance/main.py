"""Prediction Performance Testing Main Script.

This script orchestrates large-scale prediction performance evaluation across multiple models,
scenarios, and noise conditions. It supports both local testing and distributed cluster execution
using SLURM array jobs for efficient parallel processing.

Key Features:
- Supports multiple models (RNN, NP baseline, etc.)
- Tests across TDD/FDD scenarios with various noise conditions
- Efficient memory management with CPU/GPU model transfers
- SLURM array job support for distributed testing
- Automatic result aggregation and CSV output

Execution Modes:
1. Local Mode: Test single model with all combinations
   Usage: python3 -m src.testing.prediction_performance.main --model RNN

2. Cluster Mode: Distributed testing using SLURM array jobs
   Usage: sbatch scripts/testing_template.slurm (automatically uses SLURM_ARRAY_TASK_ID)

The script automatically handles:
- Model loading and device management
- Noise generation and application
- Performance metric computation (NMSE, MSE, SE)
- Result storage and memory cleanup

Output Structure:
- z_artifacts/outputs/testing/prediction_performance/[date_time]/
"""

from src.noise.noise_testing import Noise
from src.testing.config import (
    BATCH_SIZE,
    JOBS_PER_MODEL,
    LIST_MODELS,
    create_all_combinations,
    log_gpu_memory_usage,
    slice_combinations,
)


def get_array_mapping(array_id):
    """Map SLURM array task ID to specific model and slice assignment.

    Distributes array jobs across models with JOBS_PER_MODEL jobs per model:
    - Jobs 1-20: First model in LIST_MODELS
    - Jobs 21-40: Second model in LIST_MODELS
    - And so on...

    Args:
        array_id (int): SLURM array task ID (1-based)

    Returns:
        tuple: (model_name, slice_info) where slice_info is (slice_idx, total_slices)

    Raises:
        ValueError: If array_id is outside valid range

    """
    total_models = len(LIST_MODELS)
    max_array_id = total_models * JOBS_PER_MODEL

    # Validate array ID is within expected range
    if not 1 <= array_id <= max_array_id:
        raise ValueError(f"Array ID {array_id} is out of range. Valid range: 1-{max_array_id}")

    # Convert to 0-based index for calculations
    array_idx = array_id - 1

    # Determine which model this array ID belongs to
    model_idx = array_idx // JOBS_PER_MODEL
    model_name = LIST_MODELS[model_idx]

    # Determine slice index within this model's job allocation
    slice_idx = array_idx % JOBS_PER_MODEL

    # Return slice info as (current_slice, total_slices)
    slice_info = (slice_idx, JOBS_PER_MODEL)

    return model_name, slice_info


if __name__ == "__main__":
    import argparse
    import gc
    import os
    from pathlib import Path

    import torch
    from tqdm import tqdm

    from src.cp.loss.loss import MSELoss, NMSELoss, SELoss
    from src.testing.get_models import get_eval_model, wrap_model_with_imputer
    from src.testing.prediction_performance.test_unit import test_unit
    from src.utils.dirs import DIR_DATA, DIR_OUTPUTS
    from src.utils.main_utils import make_logger
    from src.utils.time_utils import get_current_time

    # === ARGUMENT PARSING AND MODE DETECTION ===
    parser = argparse.ArgumentParser(description="Run prediction performance testing")
    parser.add_argument(
        "--model", "-m", type=str, help="Specific model to test (if not provided, uses SLURM array mode)"
    )
    args = parser.parse_args()

    # Determine execution mode based on arguments
    if args.model:
        # LOCAL MODE: Test specific model with all combinations
        model_name = args.model
        if model_name not in LIST_MODELS:
            raise ValueError(f"Model {model_name} not found in LIST_MODELS: {LIST_MODELS}")
        slice_info = None
        mode = "local"
    else:
        # CLUSTER MODE: Use SLURM array task ID for distributed processing
        array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        model_name, slice_info = get_array_mapping(array_id)
        mode = "cluster"

    # === OUTPUT DIRECTORY SETUP ===
    dir_outputs = Path(DIR_OUTPUTS) / "testing" / "prediction_performance"
    cur_time = get_current_time()

    # Create mode-specific output directory structure
    if mode == "local":
        dir_outputs = dir_outputs / f"{model_name}" / "full_test" / cur_time
    else:
        # slice_info is guaranteed to be not None in cluster mode
        assert slice_info is not None, "slice_info should not be None in cluster mode"
        dir_outputs = dir_outputs / f"{model_name}" / f"slice_{slice_info[0] + 1}" / cur_time

    dir_outputs.mkdir(parents=True, exist_ok=True)
    df_path = dir_outputs / "result.csv"

    # === LOGGING AND DEVICE SETUP ===
    logger = make_logger(dir_outputs)

    # Setup compute device (prefer GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    logger.info("device - {} | {}".format(device, device_name if torch.cuda.is_available() else "CPU"))

    # Log execution mode and configuration
    if mode == "local":
        logger.info(f"Running in local mode - model: {model_name} (full test, no slicing)")
    else:
        # slice_info is guaranteed to be not None in cluster mode
        assert slice_info is not None, "slice_info should not be None in cluster mode"
        logger.info(
            f"Running in cluster mode - Array ID {array_id}: model: {model_name}, "
            f"slice: {slice_info[0] + 1} of {slice_info[1]} total jobs"
        )

    # === LOSS FUNCTIONS AND NOISE SETUP ===
    # Initialize loss functions on target device
    criterion_nmse = NMSELoss().to(device)
    logger.info("NMSE loss function initialized")

    criterion_mse = MSELoss().to(device)
    logger.info("MSE loss function initialized")

    criterion_se = SELoss(SNR=10).to(device)
    logger.info("SE loss function initialized")

    # Initialize noise generator for adding various noise types
    noise = Noise()

    # === COMBINATION GENERATION AND ASSIGNMENT ===
    # Create all possible test combinations (scenario, noise type, parameters, etc.)
    list_all_combs = create_all_combinations()

    # Assign combinations based on execution mode
    if mode == "local":
        # LOCAL MODE: Process all combinations without slicing
        list_assigned_combs = list_all_combs
        tot_combs = len(list_assigned_combs)
        logger.info(f"Processing all {tot_combs} combinations for {model_name} (local mode)")
    else:
        # CLUSTER MODE: Process only assigned slice of combinations
        list_assigned_combs = slice_combinations(list_all_combs, slice_info)
        tot_combs = len(list_assigned_combs)
        tot_all_combs = len(list_all_combs)
        logger.info(f"Processing {tot_combs} out of {tot_all_combs} total combinations for {model_name} (cluster mode)")

    # === MODEL LOADING AND MEMORY OPTIMIZATION ===
    # Load all required models once and keep on CPU for efficient GPU memory management
    logger.info("Loading all models and keeping on CPU...")
    cpu_device = torch.device("cpu")

    # Load both TDD and FDD variants of the model
    model_TDD = get_eval_model(model_name=model_name, device=device, scenario="TDD").to(cpu_device)
    model_FDD = get_eval_model(model_name=model_name, device=device, scenario="FDD").to(cpu_device)

    # Create wrapped versions with imputation capability for package drop noise
    model_imputer_TDD = wrap_model_with_imputer(model_TDD).to(cpu_device)
    model_imputer_FDD = wrap_model_with_imputer(model_FDD).to(cpu_device)

    logger.info("All models loaded and kept on CPU for efficient GPU memory management")
    log_gpu_memory_usage(logger)

    # === COMBINATION PROCESSING OPTIMIZATION ===
    # Sort combinations by scenario and noise_type to minimize GPU model transfers
    # Combinations format: (scenario, is_gen, noise_type, noise_degree, cm, ds, ms)
    list_assigned_combs.sort(key=lambda x: (x[0], x[2]))  # Sort by scenario, then noise_type

    # === MAIN TESTING LOOP ===
    for scenario, is_gen, noise_type, nd, cm, ds, ms in tqdm(
        list_assigned_combs,
        total=tot_combs,
        desc=f"Testing assigned combinations | {model_name}",
    ):
        # Validate parameter types for safety
        assert isinstance(nd, (int, float))
        assert isinstance(cm, str)
        assert isinstance(ds, float)
        assert isinstance(ms, int)

        # Select appropriate model based on scenario and noise type
        if noise_type == "packagedrop":
            # Use imputation-capable models for package drop noise
            current_model = model_imputer_TDD if scenario == "TDD" else model_imputer_FDD
        else:
            # Use standard models for other noise types
            current_model = model_TDD if scenario == "TDD" else model_FDD

        # Move selected model to GPU for testing (efficient memory management)
        current_model.to(device)
        current_model.eval()

        # Convert scenario to data format flag
        is_U2D = scenario == "FDD"

        # Execute testing for current combination
        df = test_unit(
            # === Data parameters ===
            scenario=scenario,
            is_gen=is_gen,
            is_U2D=is_U2D,
            dir_data=Path(DIR_DATA),
            batch_size=BATCH_SIZE,
            device=device,
            # === Model parameters ===
            list_models=[current_model],  # Single model wrapped in list
            # === Loss functions ===
            criterion_nmse=criterion_nmse,
            criterion_mse=criterion_mse,
            criterion_se=criterion_se,
            # === Scenario parameters ===
            cm=cm,
            ds=ds,
            ms=ms,
            # === Noise parameters ===
            noise_type=noise_type,
            noise_func=getattr(noise, noise_type),
            noise_degree=nd,
            df_path=df_path,  # Single CSV file for all combinations in this slice
        )

        # Move model back to CPU to free GPU memory for next iteration
        current_model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === FINAL CLEANUP ===
    # Delete all model references to free memory
    del model_TDD, model_FDD, model_imputer_TDD, model_imputer_FDD
    logger.info("Cleaned up all models (TDD, FDD, and wrapped versions)")

    # Force memory cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    logger.info(f"Completed memory cleanup for model: {model_name}")
    log_gpu_memory_usage(logger)

    logger.info(f"Testing completed successfully for {model_name} in {mode} mode")
