"""Computational Overhead Analysis for CSI Prediction Models.

This script performs comprehensive computational overhead analysis for CSI prediction models,
measuring key performance metrics including:
- Model complexity (parameter counts, FLOPs)
- Runtime performance (inference and training time)
- Cross-scenario performance comparison (TDD vs FDD)

The analysis generates detailed reports in multiple formats (CSV, JSON, console tables)
for performance comparison and optimization guidance.

Key Features:
- Automated benchmarking across all registered models
- Statistical timing analysis with warmup periods
- Memory-efficient testing with cleanup between models
- Device-agnostic testing (CPU/GPU with proper synchronization)
- Comprehensive result logging and export

Output Structure:
- computational_overhead.csv: Detailed metrics table
- computational_overhead.json: Raw results dictionary
- result.log: Detailed execution log with timing information
- Console summary: Formatted results table for quick review

Usage:
    python3 -m src.testing.computational_overhead.main

"""

import torch

from src.testing.computational_overhead.utils import (
    count_flops,
    count_total_parameters,
    count_trainable_parameters,
    get_input_data_for_model,
    measure_model_time,
)
from src.testing.config import LIST_MODELS as LIST_MODEL_NAMES


# Timing configuration constants
# These values balance measurement accuracy with execution time
WARM_UP_ITERATIONS = 50  # Iterations to stabilize CUDA kernels and caches
TOT_ITERATIONS = 100  # Total iterations including warmup for statistical reliability

# Scenarios to test - covers both TDD and FDD communication modes
LIST_SCENARIOS = ["TDD", "FDD"]

if __name__ == "__main__":
    # Import required libraries for data processing and output formatting
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

    # Create timestamped output directory for this analysis run
    # Structure: DIR_OUTPUTS/testing/computational_overhead/YYYYMMDD_HHMMSS/
    dir_output = Path(DIR_OUTPUTS) / "testing" / "computational_overhead" / get_current_time()
    dir_output.mkdir(parents=True, exist_ok=True)

    # Initialize logger for detailed execution tracking
    # Logs both to file (result.log) and console with color formatting
    logger = make_logger(dir_output)

    # Device detection and configuration
    # Automatically selects CUDA if available for optimal performance
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    logger.info("Using device - {} | {}".format(DEVICE, device_name if DEVICE.type == "cuda" else "CPU"))

    # Batch size configuration for testing
    # Small batch size (1) used for consistent timing measurements
    BATCH_SMALL = 1

    # Initialize result storage structures
    res_dict: dict[str, dict] = {}  # Raw results for JSON export
    df_results = pd.DataFrame()  # Structured results for CSV export and analysis

    # Main testing loop - iterate through all scenarios and models
    for scenario in LIST_SCENARIOS:
        # Batch size loop (currently fixed to BATCH_SMALL for consistency)
        for batch_size in [BATCH_SMALL]:
            logger.info(f"Testing batch size: {batch_size}")

            # Test each model in the current scenario configuration
            for model_name in LIST_MODEL_NAMES:
                # Load model in evaluation mode for the specific scenario
                model = get_eval_model(model_name=model_name, device=DEVICE, scenario=scenario)
                logger.info(f"Model loaded - {model_name} for {scenario}")
                logger.info(f"Testing {model_name} with batch size {batch_size}")

                # Generate appropriate synthetic input data based on model architecture
                # Different models may expect different input formats (complex vs real tensors)
                input_data = get_input_data_for_model(model, batch_size, device=DEVICE)

                # Measure model complexity metrics (parameters and computational cost)
                try:
                    # Count model parameters for memory usage estimation
                    total_params = count_total_parameters(model)
                    trainable_params = count_trainable_parameters(model)

                    # Count FLOPs (Floating Point Operations) for computational cost analysis
                    flops, flops_error = count_flops(model, input_data)

                    # Log complexity metrics for immediate feedback
                    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
                    logger.info(f"FLOPs: {flops:,}")

                    # Handle FLOP computation errors (some models may not be supported)
                    if flops_error:
                        logger.warning(f"FLOPs computation error for {model_name}: {flops_error}")
                except Exception as e:
                    # Graceful handling of complexity measurement failures
                    logger.warning(f"Failed to count parameters/FLOPs for {model_name}: {e}")
                    total_params = np.nan
                    trainable_params = np.nan
                    flops = np.nan

                # Measure runtime performance for both inference and training modes
                # Uses statistical timing with warmup periods for accurate measurements

                # Inference timing - model in eval mode with no gradient computation
                list_inference_time = measure_model_time(
                    model=model,
                    mode="inference",
                    device=DEVICE,
                    input_data=input_data,
                    desc=f"Inference time for {model_name} | {scenario} | batch_size {batch_size}",
                )

                # Training timing - model in train mode with gradient tracking
                # Note: Only measures forward pass time, not backward pass
                list_training_time = measure_model_time(
                    model=model,
                    mode="training",
                    device=DEVICE,
                    input_data=input_data,
                    desc=f"Training time for {model_name} | {scenario} | batch_size {batch_size}",
                )

                # Calculate statistical measures from timing data
                # Only proceed if both timing measurements succeeded
                if list_inference_time and list_training_time:
                    # Compute mean and standard deviation for inference timing
                    inference_time_avg = np.mean(list_inference_time)
                    inference_time_std = np.std(list_inference_time)

                    # Compute mean and standard deviation for training timing
                    training_time_avg = np.mean(list_training_time)
                    training_time_std = np.std(list_training_time)

                    # Store raw numerical results for JSON export
                    # Preserves original precision for detailed analysis
                    res_dict[model_name] = {
                        "total_params": total_params,
                        "trainable_params": trainable_params,
                        "flops": flops,
                        "inference_time_avg": inference_time_avg,
                        "inference_time_std": inference_time_std,
                        "training_time_avg": training_time_avg,
                        "training_time_std": training_time_std,
                    }

                    # Create structured DataFrame row for CSV export and analysis
                    # Includes multiple unit conversions for different use cases
                    new_row = pd.DataFrame(
                        {
                            # Experiment configuration
                            "Scenario": [scenario],
                            "Model": [model_name],
                            "Batch_Size": [batch_size],
                            # Model complexity metrics (raw counts)
                            "Total_Params": [total_params],
                            "Trainable_Params": [trainable_params],
                            # Model complexity metrics (human-readable millions)
                            "Total_Params_M": [total_params / 1e6],
                            "Trainable_Params_M": [trainable_params / 1e6],
                            # Computational cost metrics (multiple scales)
                            "FLOPS": [flops],  # Raw FLOP count
                            "MFLOPS": [flops / 1e6],  # Millions of FLOPs
                            "GFLOPS": [flops / 1e9],  # Billions of FLOPs
                            # Timing metrics in milliseconds (common for reporting)
                            "Inference_Time_Avg_ms": [inference_time_avg * 1000],
                            "Inference_Time_Std_ms": [inference_time_std * 1000],
                            "Training_Time_Avg_ms": [training_time_avg * 1000],
                            "Training_Time_Std_ms": [training_time_std * 1000],
                            # Timing metrics in seconds (for precise calculations)
                            "Inference_Time_Avg_s": [inference_time_avg],
                            "Inference_Time_Std_s": [inference_time_std],
                            "Training_Time_Avg_s": [training_time_avg],
                            "Training_Time_Std_s": [training_time_std],
                        }
                    )

                    # Append results to consolidated DataFrame for analysis
                    # Uses ignore_index=True to maintain continuous row numbering
                    df_results = pd.concat([df_results, new_row], ignore_index=True)

                    # Log completion summary with key timing metrics
                    # Format: mean±std for statistical significance
                    logger.info(
                        f"Completed {model_name}: "
                        f"Inference {inference_time_avg * 1000:.2f}±{inference_time_std * 1000:.2f}ms, "
                        f"Training {training_time_avg * 1000:.2f}±{training_time_std * 1000:.2f}ms"
                    )
                else:
                    # Handle cases where timing measurements failed
                    logger.warning(f"Skipping {model_name} due to timing errors")

                # Memory management - critical for preventing OOM in sequential testing
                logger.debug(f"Cleaning up memory after {model_name}")

                # Explicitly delete large objects to free memory immediately
                del model, input_data

                # Clear CUDA cache to free GPU memory for next model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                # Force garbage collection to ensure thorough cleanup
                # Particularly important when testing multiple large models sequentially
                gc.collect()

        # Log completion of scenario testing
        logger.info(f"Completed testing for scenario: {scenario}")

    # Results export and summary generation
    logger.info("Saving results...")

    # Export raw numerical results as JSON for programmatic access
    # Preserves full precision and nested structure for detailed analysis
    with open(dir_output / "computational_overhead.json", "w") as f:
        json.dump(res_dict, f, indent=4)
    logger.info(f"Raw results saved to: {dir_output / 'computational_overhead.json'}")

    # Export structured results as CSV for spreadsheet analysis and plotting
    if not df_results.empty:
        df_results.to_csv(dir_output / "computational_overhead.csv", index=False)
        logger.info(f"Results saved to: {dir_output / 'computational_overhead.csv'}")

        # Generate and display formatted summary table for immediate review
        logger.info("Results summary:")

        # Select key metrics for summary display (most important for quick comparison)
        display_cols = [
            "Scenario",  # Testing scenario (TDD/FDD)
            "Model",  # Model name
            "Batch_Size",  # Batch size used
            "Total_Params_M",  # Model size in millions of parameters
            "GFLOPS",  # Computational cost in billions of FLOPs
            "Inference_Time_Avg_ms",  # Average inference time in milliseconds
            "Training_Time_Avg_ms",  # Average training time in milliseconds
        ]

        # Verify all required columns exist before displaying
        if all(col in df_results.columns for col in display_cols):
            # Round values for readability in console output
            display_df = df_results[display_cols].round(3)
            # Use PostgreSQL-style table formatting for professional appearance
            logger.info("\n" + tabulate(display_df.values, headers=display_cols, tablefmt="psql"))

    # Final completion message
    logger.info("Computational overhead analysis completed!")
