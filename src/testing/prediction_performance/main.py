from src.testing.config import (
    BATCH_SIZE,
    JOBS_PER_MODEL,
    LIST_MODELS,
    create_all_combinations,
    log_gpu_memory_usage,
    slice_combinations,
)
from src.testing.noise.noise_testing import Noise


def get_array_mapping(array_id):
    """Map SLURM array task ID to specific parameter combination.

    Dynamically assigns array IDs based on the models in list_model_name:
    - Jobs 1-10: list_model_name[0]
    - Jobs 11-20: list_model_name[1]
    - Jobs 21-30: list_model_name[2]
    - And so on...

    Returns the model name and slice info for the given array_id.
    """
    total_models = len(LIST_MODELS)
    max_array_id = total_models * JOBS_PER_MODEL

    if not 1 <= array_id <= max_array_id:
        raise ValueError(f"Array ID {array_id} is out of range. Valid range: 1-{max_array_id}")

    # Convert to 0-based index
    array_idx = array_id - 1

    # Determine which model this array ID belongs to
    model_idx = array_idx // JOBS_PER_MODEL
    model_name = LIST_MODELS[model_idx]

    # Determine slice within this model's jobs
    slice_idx = array_idx % JOBS_PER_MODEL

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

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run prediction performance testing")
    parser.add_argument(
        "--model", "-m", type=str, help="Specific model to test (if not provided, uses SLURM array mode)"
    )
    args = parser.parse_args()

    # Determine mode: standalone or cluster
    if args.model:
        # Standalone mode - test specific model without slicing
        model_name = args.model
        if model_name not in LIST_MODELS:
            raise ValueError(f"Model {model_name} not found in LIST_MODELS: {LIST_MODELS}")
        slice_info = None
        mode = "local"
    else:
        # Cluster mode - use SLURM array task ID
        array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        model_name, slice_info = get_array_mapping(array_id)
        mode = "cluster"

    # make output directory
    dir_outputs = Path(DIR_OUTPUTS) / "testing" / "prediction_performance"
    cur_time = get_current_time()

    if mode == "local":
        dir_outputs = dir_outputs / f"{model_name}" / "full_test" / cur_time
    else:
        # slice_info is guaranteed to be not None in cluster mode
        assert slice_info is not None, "slice_info should not be None in cluster mode"
        dir_outputs = dir_outputs / f"{model_name}" / f"slice_{slice_info[0] + 1}" / cur_time

    dir_outputs.mkdir(parents=True, exist_ok=True)

    df_path = dir_outputs / "result.csv"

    # make logger
    logger = make_logger(dir_outputs)

    # make the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    logger.info("device - {} | {}".format(device, device_name if torch.cuda.is_available() else "CPU"))

    # Log mode and configuration
    if mode == "local":
        logger.info(f"Running in local mode - model: {model_name} (full test, no slicing)")
    else:
        # slice_info is guaranteed to be not None in cluster mode
        assert slice_info is not None, "slice_info should not be None in cluster mode"
        logger.info(
            f"Running in cluster mode - Array ID {array_id}: model: {model_name}, "
            f"slice: {slice_info[0] + 1} of {slice_info[1]} total jobs"
        )

    # make criterion
    criterion_nmse = NMSELoss().to(device)
    logger.info("NMSE is initialized")

    # MSE loss
    criterion_mse = MSELoss().to(device)
    logger.info("MSE is initialized")

    # SE loss
    criterion_se = SELoss(SNR=10).to(device)
    logger.info("SE is initialized")

    noise = Noise()

    # Create all combinations for this model
    list_all_combs = create_all_combinations()

    # Handle combinations based on mode
    if mode == "local":
        # Use all combinations without slicing
        list_assigned_combs = list_all_combs
        tot_combs = len(list_assigned_combs)
        logger.info(f"Processing all {tot_combs} combinations for {model_name} (local mode)")
    else:
        # Slice combinations for this array job
        list_assigned_combs = slice_combinations(list_all_combs, slice_info)
        tot_combs = len(list_assigned_combs)
        tot_all_combs = len(list_all_combs)
        logger.info(f"Processing {tot_combs} out of {tot_all_combs} total combinations for {model_name} (cluster mode)")

    # Load all models once and keep on CPU - much more efficient!
    logger.info("Loading all models and keeping on CPU...")
    cpu_device = torch.device("cpu")
    model_TDD = get_eval_model(model_name=model_name, device=device, scenario="TDD").to(cpu_device)
    model_FDD = get_eval_model(model_name=model_name, device=device, scenario="FDD").to(cpu_device)

    # Create wrapped versions for packagedrop noise
    model_imputer_TDD = wrap_model_with_imputer(model_TDD).to(cpu_device)
    model_imputer_FDD = wrap_model_with_imputer(model_FDD).to(cpu_device)

    logger.info("All models loaded and kept on CPU")
    log_gpu_memory_usage(logger)

    # Sort combinations by scenario and noise_type to optimize GPU transfers
    # Combinations format: (scenario, is_gen, noise_type, noise_degree, cm, ds, ms)
    list_assigned_combs.sort(key=lambda x: (x[0], x[2]))  # Sort by scenario, then noise_type

    for scenario, is_gen, noise_type, nd, cm, ds, ms in tqdm(
        list_assigned_combs,
        total=tot_combs,
        desc=f"Testing assigned combinations | {model_name}",
    ):
        assert isinstance(nd, (int, float))
        assert isinstance(cm, str)
        assert isinstance(ds, float)
        assert isinstance(ms, int)

        # Select the right model based on scenario and noise type
        if noise_type == "packagedrop":
            current_model = model_imputer_TDD if scenario == "TDD" else model_imputer_FDD
        else:
            current_model = model_TDD if scenario == "TDD" else model_FDD

        # Move model to GPU for testing
        current_model.to(device)
        current_model.eval()

        is_U2D = scenario == "FDD"

        # Use single results file per slice - much simpler for completion checking
        df = test_unit(
            # data
            scenario=scenario,
            is_gen=is_gen,
            is_U2D=is_U2D,
            dir_data=Path(DIR_DATA),
            batch_size=BATCH_SIZE,
            device=device,
            # model
            list_models=[current_model],  # Pass single model as list
            # criterion
            criterion_nmse=criterion_nmse,
            criterion_mse=criterion_mse,
            criterion_se=criterion_se,
            # scenario
            cm=cm,
            ds=ds,
            ms=ms,
            # noise
            noise_type=noise_type,
            noise_func=getattr(noise, noise_type),
            noise_degree=nd,
            df_path=df_path,  # Single file for all combinations in this slice
        )

        # Move model back to CPU to free GPU memory
        current_model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final cleanup - delete all models
    del model_TDD, model_FDD, model_imputer_TDD, model_imputer_FDD
    logger.info("Cleaned up all models (TDD, FDD, and wrapped versions)")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    logger.info(f"Cleaned up memory for model: {model_name}")
    log_gpu_memory_usage(logger)
