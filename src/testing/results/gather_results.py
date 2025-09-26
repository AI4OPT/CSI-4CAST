import datetime
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
    For each model, use the appropriate data source (slice or full_test) based on completion status.

    Args:
        model_completion: Dictionary from check_testing_completion containing completion status
        base_prediction_performance: Path to the base testing results directory
        verbose: Whether to print detailed information

    Returns:
        Consolidated DataFrame with results from appropriate sources

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
    # Parse string representations of arrays in specified columns.
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
    """Save the consolidated DataFrame to a CSV file.

    Args:
        df: Consolidated DataFrame
        output_dir: Path to save the CSV file

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
