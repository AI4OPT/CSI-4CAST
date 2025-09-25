import torch

from src.testing.computational_overhead.utils import (
    count_flops,
    count_total_parameters,
    count_trainable_parameters,
    get_input_data_for_model,
    measure_model_time,
)
from src.testing.config import LIST_MODELS as LIST_MODEL_NAMES


WARM_UP_ITERATIONS = 50
TOT_ITERATIONS = 100

# Scenarios to test
LIST_SCENARIOS = ["TDD", "FDD"]

if __name__ == "__main__":
    import gc
    import json
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from tabulate import tabulate

    from src.testing.get_models import get_eval_model
    from src.utils.dirs import DIR_OUTPUTS
    from src.utils.main_utils import make_logger

    dir_output = Path(DIR_OUTPUTS) / "testing" / "computational_overhead" / datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_output.mkdir(parents=True, exist_ok=True)

    # Make logger
    logger = make_logger(dir_output)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    logger.info("Using device - {} | {}".format(DEVICE, device_name if DEVICE.type == "cuda" else "CPU"))

    BATCH_SMALL = 1

    res_dict: dict[str, dict] = {}
    df_results = pd.DataFrame()

    for scenario in LIST_SCENARIOS:
        # for batch_size in [BATCH_SMALL]:
        for batch_size in [BATCH_SMALL]:
            logger.info(f"Testing batch size: {batch_size}")

            for model_name in LIST_MODEL_NAMES:
                model = get_eval_model(model_name=model_name, device=DEVICE, scenario=scenario)
                logger.info(f"Model loaded - {model_name} for {scenario}")
                logger.info(f"Testing {model_name} with batch size {batch_size}")

                # Get appropriate input data for the model
                input_data = get_input_data_for_model(model, batch_size, device=DEVICE)

                # Count parameters and FLOPs
                try:
                    total_params = count_total_parameters(model)
                    trainable_params = count_trainable_parameters(model)
                    flops, flops_error = count_flops(model, input_data)
                    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
                    logger.info(f"FLOPs: {flops:,}")
                    if flops_error:
                        logger.warning(f"FLOPs computation error for {model_name}: {flops_error}")
                except Exception as e:
                    logger.warning(f"Failed to count parameters/FLOPs for {model_name}: {e}")
                    total_params = np.nan
                    trainable_params = np.nan
                    flops = np.nan

                # Measure inference and training time using helper function
                list_inference_time = measure_model_time(
                    model=model,
                    mode="inference",
                    device=DEVICE,
                    input_data=input_data,
                    desc=f"Inference time for {model_name} | {scenario} | batch_size {batch_size}",
                )

                list_training_time = measure_model_time(
                    model=model,
                    mode="training",
                    device=DEVICE,
                    input_data=input_data,
                    desc=f"Training time for {model_name} | {scenario} | batch_size {batch_size}",
                )

                # Calculate statistics
                if list_inference_time and list_training_time:
                    inference_time_avg = np.mean(list_inference_time)
                    inference_time_std = np.std(list_inference_time)
                    training_time_avg = np.mean(list_training_time)
                    training_time_std = np.std(list_training_time)

                    # Store raw results for JSON
                    res_dict[model_name] = {
                        "total_params": total_params,
                        "trainable_params": trainable_params,
                        "flops": flops,
                        "inference_time_avg": inference_time_avg,
                        "inference_time_std": inference_time_std,
                        "training_time_avg": training_time_avg,
                        "training_time_std": training_time_std,
                    }

                    # Create row for DataFrame
                    new_row = pd.DataFrame(
                        {
                            "Scenario": [scenario],
                            "Model": [model_name],
                            "Batch_Size": [batch_size],
                            "Total_Params": [total_params],
                            "Trainable_Params": [trainable_params],
                            "Total_Params_M": [total_params / 1e6],  # In millions
                            "Trainable_Params_M": [trainable_params / 1e6],  # In millions
                            "FLOPS": [flops],
                            "MFLOPS": [flops / 1e6],  # In millions
                            "GFLOPS": [flops / 1e9],  # In billions
                            "Inference_Time_Avg_ms": [inference_time_avg * 1000],
                            "Inference_Time_Std_ms": [inference_time_std * 1000],
                            "Training_Time_Avg_ms": [training_time_avg * 1000],
                            "Training_Time_Std_ms": [training_time_std * 1000],
                            "Inference_Time_Avg_s": [inference_time_avg],
                            "Inference_Time_Std_s": [inference_time_std],
                            "Training_Time_Avg_s": [training_time_avg],
                            "Training_Time_Std_s": [training_time_std],
                        }
                    )

                    # Append to single results DataFrame
                    df_results = pd.concat([df_results, new_row], ignore_index=True)

                    logger.info(
                        f"Completed {model_name}: "
                        f"Inference {inference_time_avg * 1000:.2f}±{inference_time_std * 1000:.2f}ms, "
                        f"Training {training_time_avg * 1000:.2f}±{training_time_std * 1000:.2f}ms"
                    )
                else:
                    logger.warning(f"Skipping {model_name} due to timing errors")

                # Clean up CUDA memory after each model test
                logger.debug(f"Cleaning up memory after {model_name}")
                del model, input_data
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                # Optional: force garbage collection for thorough cleanup
                gc.collect()

        logger.info(f"Completed testing for scenario: {scenario}")

    # Save all results
    logger.info("Saving results...")

    # Save raw results as JSON
    with open(dir_output / "computational_overhead.json", "w") as f:
        json.dump(res_dict, f, indent=4)
    logger.info(f"Raw results saved to: {dir_output / 'computational_overhead.json'}")

    # Save single consolidated DataFrame
    if not df_results.empty:
        df_results.to_csv(dir_output / "computational_overhead.csv", index=False)
        logger.info(f"Results saved to: {dir_output / 'computational_overhead.csv'}")

        # Display summary
        logger.info("Results summary:")
        display_cols = [
            "Scenario",
            "Model",
            "Batch_Size",
            "Total_Params_M",
            "GFLOPS",
            "Inference_Time_Avg_ms",
            "Training_Time_Avg_ms",
        ]
        if all(col in df_results.columns for col in display_cols):
            display_df = df_results[display_cols].round(3)
            logger.info("\n" + tabulate(display_df.values, headers=display_cols, tablefmt="psql"))

    logger.info("Computational overhead analysis completed!")
