"""Computational Overhead Analysis for CSI Prediction Models.

This script measures key performance metrics for all registered models:
- Model complexity (parameter counts, FLOPs)
- Runtime performance (inference and training time)
- Inference efficiency (GPU/CPU energy and memory footprint)
- Cross-scenario performance comparison (TDD vs FDD)

Output Structure:
- computational_overhead.csv: Detailed metrics table
- computational_overhead.json: Raw results dictionary
- result.log: Detailed execution log
- Console summary: Formatted results table

Usage:
    python3 -m src.testing.computational_overhead.main
"""

import torch

from src.testing.computational_overhead.utils import (
    count_flops,
    count_total_parameters,
    count_trainable_parameters,
    estimate_gpu_energy_per_inference_nvml,
    get_input_data_for_model,
    get_model_device_memory_bytes,
    get_model_param_memory_bytes,
    measure_cpu_package_energy_per_inference_rapl,
    measure_model_time,
    measure_peak_cpu_rss_bytes,
    measure_peak_gpu_memory_inference,
)
from src.testing.config import CPU_ONLY_MODELS


TDD_MODELS = [
    "AR",
    "CNN",
    "LLM4CP",
    "MODEL",
    "NP",
    "PAD",
    "RNN",
    "STEMGNN",
    "WIENER",
    "ABL_NO_DENOISER",
    "ABL_NO_IDFT",
    "ABL_NO_ARL",
    "ABL_NORM_REPLACE_ARL",
    "ABL_ADD_SUBCARRIER_ARL",
    "ABL_MLP_REPLACE_EMBED",
    "ABL_MOBILENET_REPLACE_EMBED",
    "ABL_MLP_REPLACE_PRED",
    "ABL_LSTM_REPLACE_PRED",
]
FDD_MODELS = ["CNN", "LLM4CP", "MODEL", "NP", "RNN", "STEMGNN", "WIENER", "ABL_NO_ARL", "ABL_NO_SUBCARRIER_ARL"]

WARMUP_ITERATIONS = 50
MEASURE_ITERATIONS = 100

if __name__ == "__main__":
    import gc
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from tabulate import tabulate

    from src.testing.get_models import get_eval_model
    from src.utils.dirs import DIR_OUTPUTS
    from src.utils.main_utils import make_logger
    from src.utils.time_utils import get_current_time

    dir_output = Path(DIR_OUTPUTS) / "testing" / "computational_overhead" / get_current_time()
    dir_output.mkdir(parents=True, exist_ok=True)
    logger = make_logger(dir_output)

    GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CPU_DEVICE = torch.device("cpu")
    device_name = torch.cuda.get_device_name(0) if GPU_DEVICE.type == "cuda" else "CPU"
    logger.info(f"GPU device: {GPU_DEVICE} | {device_name}")

    BATCH_SIZE = 1

    res_dict: dict[str, dict] = {}
    df_results = pd.DataFrame()

    model_scenario_pairs: list[tuple[str, str]] = []
    for m in TDD_MODELS:
        model_scenario_pairs.append((m, "TDD"))
    for m in FDD_MODELS:
        model_scenario_pairs.append((m, "FDD"))

    for model_name, scenario in model_scenario_pairs:
        device = CPU_DEVICE if model_name in CPU_ONLY_MODELS else GPU_DEVICE

        try:
            model = get_eval_model(model_name=model_name, device=device, scenario=scenario)
        except Exception as e:
            logger.warning(f"Failed to load {model_name} for {scenario}: {e}")
            continue

        logger.info(f"Loaded {model_name} | {scenario} | device={device}")

        input_data = get_input_data_for_model(model, BATCH_SIZE, device=device)

        # --- Complexity ---
        try:
            total_params = count_total_parameters(model)
            trainable_params = count_trainable_parameters(model)
            flops, flops_error = count_flops(model, input_data)
            logger.info(f"Params: {total_params:,} total, {trainable_params:,} trainable | FLOPs: {flops:,}")
            if flops_error:
                logger.warning(f"FLOPs error for {model_name}: {flops_error}")
        except Exception as e:
            logger.warning(f"Failed complexity metrics for {model_name}: {e}")
            total_params = trainable_params = flops = np.nan

        # --- Timing ---
        list_inference_time = measure_model_time(
            model=model,
            mode="inference",
            device=device,
            input_data=input_data,
            warmup_iterations=WARMUP_ITERATIONS,
            tot_iterations=WARMUP_ITERATIONS + MEASURE_ITERATIONS,
            desc=f"Inference {model_name}|{scenario}",
        )
        list_training_time = measure_model_time(
            model=model,
            mode="training",
            device=device,
            input_data=input_data,
            warmup_iterations=WARMUP_ITERATIONS,
            tot_iterations=WARMUP_ITERATIONS + MEASURE_ITERATIONS,
            desc=f"Training {model_name}|{scenario}",
        )

        inference_time_avg = np.mean(list_inference_time) if list_inference_time else np.nan
        inference_time_std = np.std(list_inference_time) if list_inference_time else np.nan
        training_time_avg = np.mean(list_training_time) if list_training_time else np.nan
        training_time_std = np.std(list_training_time) if list_training_time else np.nan

        # --- Static memory ---
        model_memory = get_model_param_memory_bytes(model)
        model_device_memory = get_model_device_memory_bytes(model)

        # --- Peak GPU memory ---
        try:
            peak_gpu_memory = measure_peak_gpu_memory_inference(
                model=model,
                input_data=input_data,
                device=device,
                warmup_iterations=WARMUP_ITERATIONS,
                measure_iterations=MEASURE_ITERATIONS,
            )
        except Exception as e:
            logger.warning(f"Failed peak GPU memory for {model_name}: {e}")
            peak_gpu_memory = {
                "baseline_allocated_bytes": np.nan,
                "baseline_reserved_bytes": np.nan,
                "peak_allocated_bytes": np.nan,
                "peak_reserved_bytes": np.nan,
                "peak_allocated_delta_bytes": np.nan,
                "peak_reserved_delta_bytes": np.nan,
            }

        # --- Peak CPU RSS ---
        _model_ref, _input_ref = model, input_data

        def run_single_inference(_m=_model_ref, _inp=_input_ref, _dev=device) -> None:
            _m.eval()
            with torch.inference_mode():
                _ = _m(_inp)
                if _dev.type == "cuda":
                    torch.cuda.synchronize(_dev)

        try:
            peak_cpu_rss = measure_peak_cpu_rss_bytes(
                run_single_inference,
                warmup_iterations=WARMUP_ITERATIONS,
                measure_iterations=MEASURE_ITERATIONS,
            )
        except Exception as e:
            logger.warning(f"Failed peak CPU RSS for {model_name}: {e}")
            peak_cpu_rss = None

        # --- Energy ---
        gpu_energy = estimate_gpu_energy_per_inference_nvml(
            model=model,
            input_data=input_data,
            device=device,
            warmup_iterations=WARMUP_ITERATIONS,
            measure_iterations=MEASURE_ITERATIONS,
        )
        if gpu_energy.get("error"):
            logger.debug(f"GPU energy note for {model_name}: {gpu_energy['error']}")

        cpu_energy = measure_cpu_package_energy_per_inference_rapl(
            run_single_inference,
            warmup_iterations=WARMUP_ITERATIONS,
            measure_iterations=MEASURE_ITERATIONS,
        )
        if cpu_energy is None:
            cpu_energy = {
                "available": False,
                "energy_per_inference_J": np.nan,
                "avg_power_W": np.nan,
                "duration_s": np.nan,
                "energy_source": "",
                "error": "rapl_not_available",
            }
        elif cpu_energy.get("error"):
            logger.debug(f"CPU energy note for {model_name}: {cpu_energy['error']}")

        peak_cpu_rss_baseline = peak_cpu_rss["rss_baseline_bytes"] if peak_cpu_rss else np.nan
        peak_cpu_rss_peak = peak_cpu_rss["rss_peak_bytes"] if peak_cpu_rss else np.nan
        peak_cpu_rss_delta = peak_cpu_rss["rss_peak_delta_bytes"] if peak_cpu_rss else np.nan

        res_key = f"{scenario}:{model_name}:bs{BATCH_SIZE}"
        res_dict[res_key] = {
            "scenario": scenario,
            "model": model_name,
            "batch_size": BATCH_SIZE,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": flops,
            "inference_time_avg": inference_time_avg,
            "inference_time_std": inference_time_std,
            "training_time_avg": training_time_avg,
            "training_time_std": training_time_std,
            "model_param_memory_bytes": model_memory,
            "model_device_memory_bytes": model_device_memory,
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "peak_cpu_rss_bytes": peak_cpu_rss if peak_cpu_rss is not None else {},
            "gpu_energy_per_inference": gpu_energy,
            "cpu_energy_per_inference": cpu_energy,
        }

        new_row = pd.DataFrame(
            {
                "Scenario": [scenario],
                "Model": [model_name],
                "Batch_Size": [BATCH_SIZE],
                "Total_Params": [total_params],
                "Trainable_Params": [trainable_params],
                "Total_Params_M": [total_params / 1e6],
                "Trainable_Params_M": [trainable_params / 1e6],
                "FLOPS": [flops],
                "MFLOPS": [flops / 1e6],
                "GFLOPS": [flops / 1e9],
                "Inference_Time_Avg_ms": [inference_time_avg * 1000],
                "Inference_Time_Std_ms": [inference_time_std * 1000],
                "Training_Time_Avg_ms": [training_time_avg * 1000],
                "Training_Time_Std_ms": [training_time_std * 1000],
                "Inference_Time_Avg_s": [inference_time_avg],
                "Inference_Time_Std_s": [inference_time_std],
                "Training_Time_Avg_s": [training_time_avg],
                "Training_Time_Std_s": [training_time_std],
                "Model_Param_Bytes": [model_memory["param_bytes"]],
                "Model_Buffer_Bytes": [model_memory["buffer_bytes"]],
                "Model_ParamPlusBuffer_Bytes": [model_memory["param_plus_buffer_bytes"]],
                "Model_CPU_Bytes": [model_device_memory["cpu_bytes"]],
                "Model_GPU_Bytes": [model_device_memory["gpu_bytes"]],
                "Model_ParamPlusBuffer_MiB": [model_memory["param_plus_buffer_bytes"] / (1024**2)],
                "Peak_GPU_Allocated_Bytes": [peak_gpu_memory["peak_allocated_bytes"]],
                "Peak_GPU_Reserved_Bytes": [peak_gpu_memory["peak_reserved_bytes"]],
                "Peak_GPU_Allocated_Delta_Bytes": [peak_gpu_memory["peak_allocated_delta_bytes"]],
                "Peak_GPU_Reserved_Delta_Bytes": [peak_gpu_memory["peak_reserved_delta_bytes"]],
                "Peak_GPU_Allocated_MiB": [peak_gpu_memory["peak_allocated_bytes"] / (1024**2)],
                "Peak_CPU_RSS_Bytes": [peak_cpu_rss_peak],
                "Peak_CPU_RSS_Delta_Bytes": [peak_cpu_rss_delta],
                "RSS_Baseline_Bytes": [peak_cpu_rss_baseline],
                "Peak_CPU_RSS_MiB": [peak_cpu_rss_peak / (1024**2)],
                "Peak_CPU_RSS_Delta_MiB": [peak_cpu_rss_delta / (1024**2)],
                "GPU_Energy_Per_Inference_J": [gpu_energy["energy_per_inference_J"]],
                "GPU_Avg_Power_W": [gpu_energy["avg_power_W"]],
                "GPU_Idle_Power_W": [gpu_energy["idle_power_W"]],
                "GPU_Energy_Available": [gpu_energy["available"]],
                "GPU_Energy_Error": [gpu_energy["error"]],
                "CPU_Energy_Per_Inference_J": [cpu_energy["energy_per_inference_J"]],
                "CPU_Avg_Power_W": [cpu_energy["avg_power_W"]],
                "CPU_Energy_Available": [cpu_energy["available"]],
                "CPU_Energy_Source": [cpu_energy["energy_source"]],
                "CPU_Energy_Error": [cpu_energy["error"]],
            }
        )

        df_results = pd.concat([df_results, new_row], ignore_index=True)

        logger.info(
            f"Done {model_name}|{scenario}: "
            f"Inf {inference_time_avg * 1000:.2f}±{inference_time_std * 1000:.2f}ms, "
            f"Train {training_time_avg * 1000:.2f}±{training_time_std * 1000:.2f}ms, "
            f"GPU energy {gpu_energy['energy_per_inference_J']:.6f}J/inf, "
            f"CPU energy {cpu_energy['energy_per_inference_J']:.6f}J/inf"
        )

        del model, input_data, _model_ref, _input_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # --- Save results ---
    logger.info("Saving results...")

    with open(dir_output / "computational_overhead.json", "w") as f:
        json.dump(res_dict, f, indent=4)
    logger.info(f"JSON saved: {dir_output / 'computational_overhead.json'}")

    if not df_results.empty:
        df_results.to_csv(dir_output / "computational_overhead.csv", index=False)
        logger.info(f"CSV saved: {dir_output / 'computational_overhead.csv'}")

        display_cols = [
            "Scenario",
            "Model",
            "Total_Params_M",
            "GFLOPS",
            "Inference_Time_Avg_ms",
            "Training_Time_Avg_ms",
            "Model_ParamPlusBuffer_MiB",
            "Peak_GPU_Allocated_MiB",
            "Peak_CPU_RSS_Delta_MiB",
            "GPU_Energy_Per_Inference_J",
            "CPU_Energy_Per_Inference_J",
        ]
        if all(col in df_results.columns for col in display_cols):
            display_df = df_results[display_cols].round(3)
            logger.info("\n" + tabulate(display_df.values, headers=display_cols, tablefmt="psql"))

    logger.info("Computational overhead analysis completed!")
