"""Prediction Performance Testing Main Script.

Each SLURM array job evaluates ONE (model, scenario, test_type) setting on a
slice of its sub-combinations.

The array ID is read from SLURM_ARRAY_TASK_ID (cluster) or passed via
``--array-id`` (local).  Use ``python3 -m src.testing.config`` to see which
array IDs correspond to which settings.

Output directory layout::

    prediction_performance/{model}/{scenario}/{test_type}/slice_{idx}/{timestamp}/result.csv
"""

if __name__ == "__main__":
    import gc
    import os
    from pathlib import Path

    import torch
    from tqdm import tqdm

    from src.cp.loss.loss import MSELoss, NMSELoss, SELoss
    from src.noise.noise_testing import Noise
    from src.testing.config import (
        BATCH_SIZE,
        CPU_ONLY_MODELS,
        create_combinations_for_setting,
        get_array_mapping,
        log_gpu_memory_usage,
        slice_combinations,
    )
    from src.testing.get_models import get_eval_model, wrap_model_with_imputer
    from src.testing.prediction_performance.test_unit import test_unit
    from src.utils.dirs import DIR_DATA, DIR_OUTPUTS
    from src.utils.main_utils import make_logger
    from src.utils.time_utils import get_current_time

    # === RESOLVE ARRAY ID ===
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_id is None:
        raise RuntimeError("SLURM_ARRAY_TASK_ID not set. This script must be run as a SLURM array job.")
    array_id = int(slurm_id)
    model_name, scenario, test_type, slice_info = get_array_mapping(array_id)

    # === OUTPUT DIRECTORY ===
    dir_outputs = Path(DIR_OUTPUTS) / "testing" / "prediction_performance"
    cur_time = get_current_time()
    dir_outputs = dir_outputs / model_name / scenario / test_type / f"slice_{slice_info[0]}" / cur_time
    dir_outputs.mkdir(parents=True, exist_ok=True)
    df_path = dir_outputs / "result.csv"

    # === LOGGING / DEVICE ===
    logger = make_logger(dir_outputs)
    if model_name in CPU_ONLY_MODELS:
        device = torch.device("cpu")
        device_name = "CPU"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"device - {device} | {device_name}")
    logger.info(
        f"array_id={array_id}, model={model_name}, scenario={scenario}, "
        f"test_type={test_type}, slice={slice_info[0]} of {slice_info[1]}"
    )

    # === LOSS FUNCTIONS / NOISE ===
    criterion_nmse = NMSELoss().to(device)
    criterion_mse = MSELoss().to(device)
    criterion_se = SELoss(SNR=10).to(device)
    noise = Noise()

    # === COMBINATIONS ===
    all_combs = create_combinations_for_setting(scenario, test_type)
    list_assigned_combs = slice_combinations(all_combs, slice_info)

    tot_combs = len(list_assigned_combs)
    logger.info(f"Processing {tot_combs} combinations (total for {test_type}: {len(all_combs)})")

    # === MODEL LOADING (single model, single scenario) ===
    cpu_device = torch.device("cpu")
    model = get_eval_model(model_name=model_name, device=device, scenario=scenario).to(cpu_device)
    model_imputer = wrap_model_with_imputer(model).to(cpu_device)
    logger.info(f"Model loaded: {model_name} for {scenario}")
    log_gpu_memory_usage(logger)

    # Sort by noise_type to minimize model-swap overhead
    list_assigned_combs.sort(key=lambda x: x[2])

    # === MAIN TESTING LOOP ===
    for _scenario, is_gen, noise_type, nd, cm, ds, ms in tqdm(
        list_assigned_combs,
        total=tot_combs,
        desc=f"Testing {model_name}|{scenario}|{test_type}",
    ):
        assert isinstance(nd, (int, float))
        assert isinstance(cm, str)
        assert isinstance(ds, float)
        assert isinstance(ms, int)

        current_model = model_imputer if noise_type == "packagedrop" else model
        current_model.to(device)
        current_model.eval()

        is_U2D = scenario == "FDD"

        test_unit(
            scenario=scenario,
            is_gen=is_gen,
            is_U2D=is_U2D,
            dir_data=Path(DIR_DATA),
            batch_size=BATCH_SIZE,
            device=device,
            list_models=[current_model],
            criterion_nmse=criterion_nmse,
            criterion_mse=criterion_mse,
            criterion_se=criterion_se,
            cm=cm,
            ds=ds,
            ms=ms,
            noise_type=noise_type,
            noise_func=getattr(noise, noise_type),
            noise_degree=nd,
            df_path=df_path,
        )

        current_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === CLEANUP ===
    del model, model_imputer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"Completed: {model_name}|{scenario}|{test_type}")
    log_gpu_memory_usage(logger)
