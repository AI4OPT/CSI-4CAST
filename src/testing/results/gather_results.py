"""Results gathering module for CSI prediction testing experiments.

This module provides functionality to gather and consolidate test results from
multiple sources (cluster slices and local full tests) based on completion status.
It handles the complexity of determining which data source to use for each model
and consolidates results into a unified DataFrame for analysis.

Key Features:
- Intelligent source selection based on completion status
- Support for both cluster slice and local full_test data sources
- Array string parsing and flattening for prediction step data
- Consolidated CSV output with comprehensive metadata
- Data validation and error handling
- Summary statistics generation

Typical Usage:
    # Get completion status first
    completion_result = check_testing_completion()

    # Gather results based on completion status
    consolidated_df = gather_all_results(
        model_completion=completion_result['model_completion'],
        verbose=True
    )

    # Save consolidated results
    save_path = save_consolidated_results(consolidated_df)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.testing.config import JOBS_PER_MODEL
from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_latest_datetime_folder


def gather_all_results(
    model_completion: dict,
    base_prediction_performance: Path = Path(DIR_OUTPUTS) / "testing" / "prediction_performance",
    verbose: bool = True,
) -> pd.DataFrame:
    """Gather results based on model completion status.

    This is the main function for intelligently gathering test results from multiple
    sources. For each model, it determines the best data source (cluster slices vs
    local full_test) based on completion status and timestamp information, then
    consolidates all results into a unified DataFrame.

    The function handles the complexity of:
    - Choosing between slice and full_test data sources per model
    - Loading data from multiple timestamp directories
    - Adding metadata columns for traceability
    - Handling missing or corrupted files gracefully
    - Providing detailed progress reporting

    Args:
        model_completion: Dictionary from check_testing_completion() containing:
            - model_complete: Whether the model has completed testing
            - use_source: Which data source to use ("slice" or "full_test")
            - Timestamp information for decision making
        base_prediction_performance: Path to the base testing results directory
                                   Expected structure: base_dir/MODEL_NAME/slice_X/ or full_test/
        verbose: Whether to print detailed progress information

    Returns:
        pd.DataFrame: Consolidated DataFrame with results from all completed models.
                     Contains original test result columns plus metadata:
                     - model_name: Name of the source model
                     - slice_number: Slice number or "full_test"
                     - timestamp: Directory timestamp of the data source
                     - data_source: "slice" or "full_test"

    Raises:
        ValueError: If no valid result files are found from completed models

    """
    all_dataframes = []

    if verbose:
        print(f"Gathering results based on completion status from {base_prediction_performance}")
        print(f"Processing {len(model_completion)} models")

    for model_name, status in model_completion.items():
        if not status["model_complete"]:
            if verbose:
                print(f"Skipping {model_name}: Not complete")
            continue

        use_source = status["use_source"]
        if verbose:
            print(f"Processing {model_name}: Using {use_source} data")

        if use_source == "slice":
            # Gather slice results for this model
            model_dir = base_prediction_performance / model_name
            for slice_number in range(1, JOBS_PER_MODEL + 1):
                slice_dir = model_dir / f"slice_{slice_number}"
                if not slice_dir.exists():
                    continue

                latest_timestamp_dir = get_latest_datetime_folder(slice_dir)
                if latest_timestamp_dir is None:
                    continue

                result_file = latest_timestamp_dir / "result.csv"
                if not result_file.exists():
                    continue

                try:
                    df = pd.read_csv(result_file)
                    if len(df) > 0:
                        df["model_name"] = model_name
                        df["slice_number"] = slice_number
                        df["timestamp"] = latest_timestamp_dir.name
                        df["data_source"] = "slice"
                        all_dataframes.append(df)
                        if verbose:
                            print(f"  Loaded {len(df)} records from slice_{slice_number}")
                except Exception as e:
                    if verbose:
                        print(f"  Error reading slice_{slice_number}: {e}")

        elif use_source == "full_test":
            # Gather full_test results for this model
            full_test_dir = base_prediction_performance / model_name / "full_test"
            if not full_test_dir.exists():
                continue

            latest_timestamp_dir = get_latest_datetime_folder(full_test_dir)
            if latest_timestamp_dir is None:
                continue

            result_file = latest_timestamp_dir / "result.csv"
            if not result_file.exists():
                continue

            try:
                df = pd.read_csv(result_file)
                if len(df) > 0:
                    df["model_name"] = model_name
                    df["slice_number"] = "full_test"
                    df["timestamp"] = latest_timestamp_dir.name
                    df["data_source"] = "full_test"
                    all_dataframes.append(df)
                    if verbose:
                        print(f"  Loaded {len(df)} records from full_test")
            except Exception as e:
                if verbose:
                    print(f"  Error reading full_test: {e}")

    if not all_dataframes:
        raise ValueError("No valid result files found from completed models")

    # Concatenate all dataframes
    consolidated_df = pd.concat(all_dataframes, ignore_index=True)

    if verbose:
        print(f"\nSuccessfully gathered {len(consolidated_df)} total records from {len(all_dataframes)} sources")
        print(f"Columns: {list(consolidated_df.columns)}")
        print(f"Unique models: {consolidated_df['model'].unique()}")
        print(f"Unique scenarios: {consolidated_df['scenario'].unique()}")
        print(f"Unique noise types: {consolidated_df['noise_type'].unique()}")
        print(f"Data sources: {consolidated_df['data_source'].value_counts().to_dict()}")

    return consolidated_df


def parse_array_string(df: pd.DataFrame, list_columns: list[str]) -> pd.DataFrame:
    """Parse string representations of arrays in specified columns and flatten by prediction steps.

    This function handles the conversion of array-like string columns (e.g., "[1.2 3.4 5.6]")
    into individual rows for each prediction step. This is essential for proper analysis
    of time-series prediction results where each prediction step needs to be analyzed separately.

    The function:
    1. Parses string representations of numpy arrays back to actual arrays
    2. Creates separate rows for each prediction step
    3. Preserves all other columns from the original DataFrame
    4. Adds a 'pred_step' column indicating the prediction step index

    Args:
        df: Input DataFrame with array string columns
        list_columns: List of column names containing array strings to parse
                     (e.g., ["nmse_mean", "nmse_std", "se_mean", "se_std"])

    Returns:
        pd.DataFrame: Flattened DataFrame where each row represents one prediction step
                     from one test combination. The DataFrame will have more rows than
                     the input (multiplied by the number of prediction steps).

    Example:
        Input row: {"model": "A", "nmse_mean": "[0.1 0.2 0.3 0.4]", "scenario": "TDD"}
        Output rows:
            {"model": "A", "nmse_mean": 0.1, "scenario": "TDD", "pred_step": 0}
            {"model": "A", "nmse_mean": 0.2, "scenario": "TDD", "pred_step": 1}
            {"model": "A", "nmse_mean": 0.3, "scenario": "TDD", "pred_step": 2}
            {"model": "A", "nmse_mean": 0.4, "scenario": "TDD", "pred_step": 3}

    """
    # Parse string representations of arrays in specified columns
    for column in list_columns:
        df[column] = df[column].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))  # type: ignore
    # flatten the arrays to ensure they are 1D
    flatten_records = []
    for _, row in df.iterrows():
        for step in range(len(row[list_columns[0]])):
            record = row.drop(list_columns).to_dict()
            record["pred_step"] = step
            record.update({col: row[col][step] for col in list_columns})
            flatten_records.append(record)
    # Create a new DataFrame from the flattened records
    df = pd.DataFrame(flatten_records)
    return df


def save_consolidated_results(df: pd.DataFrame, output_dir: Path = Path(DIR_OUTPUTS) / "testing" / "results"):
    """Save the consolidated DataFrame to a CSV file with timestamped directory.

    This function saves the consolidated results to a CSV file in a structured
    directory format. It creates the necessary directory structure and provides
    a standardized location for downstream analysis tools to find the results.

    Args:
        df: Consolidated DataFrame containing all gathered test results
        output_dir: Base directory path to save the CSV file. The actual file
                   will be saved as output_dir/consolidated_results.csv

    Returns:
        Path: Full path to the saved CSV file

    Side Effects:
        - Creates the output directory if it doesn't exist
        - Overwrites any existing consolidated_results.csv file
        - Prints confirmation message with save location

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "consolidated_results.csv", index=False)
    print(f"Consolidated results saved to {output_dir}")

    return output_dir / "consolidated_results.csv"


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get a summary of the consolidated data.

    Args:
        df: Consolidated DataFrame

    Returns:
        Dictionary with summary statistics

    """
    summary = {
        "total_records": len(df),
        "unique_models": df["model"].unique().tolist(),
        "unique_scenarios": df["scenario"].unique().tolist(),
        "unique_noise_types": df["noise_type"].unique().tolist(),
        "unique_cms": df["cm"].unique().tolist(),
        "ds_range": (df["ds"].min(), df["ds"].max()),
        "ms_range": (df["ms"].min(), df["ms"].max()),
        "noise_degree_range": (df["noise_degree"].min(), df["noise_degree"].max()),
        "is_gen_values": df["is_gen"].unique().tolist(),
        "is_U2D_values": df["is_U2D"].unique().tolist(),
        "records_per_model": df["model"].value_counts().to_dict(),
        "records_per_scenario": df["scenario"].value_counts().to_dict(),
        "records_per_noise_type": df["noise_type"].value_counts().to_dict(),
        "slices_per_model": df.groupby("model_name")["slice_number"].nunique().to_dict(),
    }

    return summary
