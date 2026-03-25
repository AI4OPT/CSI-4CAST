"""Results gathering module for CSI prediction testing experiments.

Scans the new directory layout and consolidates result CSVs into a single
DataFrame for downstream analysis and visualization.

Directory layout::

    prediction_performance/{model}/{scenario}/{test_type}/slice_{idx}/{timestamp}/result.csv

For each (model, scenario, test_type) in LIST_SETTINGS we:
  1. Scan all slice_* subdirs.
  2. In each slice, find the latest timestamp folder, read result.csv.
  3. Concat all CSVs into one DataFrame.

Usage::

    python3 -m src.testing.results.gather_results
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.testing.config import LIST_SETTINGS
from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_latest_datetime_folder


BASE_DIR = Path(DIR_OUTPUTS) / "testing" / "prediction_performance"


def gather_setting_results(
    model: str,
    scenario: str,
    test_type: str,
    base_dir: Path = BASE_DIR,
) -> pd.DataFrame | None:
    """Read and concat all slice result CSVs for one (model, scenario, test_type).

    Returns None if no result files are found.
    """
    setting_dir = base_dir / model / scenario / test_type
    if not setting_dir.exists():
        return None

    frames: list[pd.DataFrame] = []
    for child in sorted(setting_dir.iterdir()):
        if not (child.is_dir() and child.name.startswith("slice_")):
            continue
        latest = get_latest_datetime_folder(child)
        if latest is None:
            continue
        result_file = latest / "result.csv"
        if not result_file.exists():
            continue
        try:
            df = pd.read_csv(result_file)
            if len(df) > 0:
                df["slice"] = child.name
                df["timestamp"] = latest.name
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def gather_all_results(
    base_dir: Path = BASE_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Gather results for every setting in LIST_SETTINGS.

    Returns a single consolidated DataFrame.  Raises ValueError if nothing
    was found at all.
    """
    all_frames: list[pd.DataFrame] = []

    for model, scenario, test_type in LIST_SETTINGS:
        df = gather_setting_results(model, scenario, test_type, base_dir)
        if df is None:
            if verbose:
                print(f"  No results: {model}/{scenario}/{test_type}")
            continue
        if verbose:
            print(f"  {model}/{scenario}/{test_type}: {len(df)} rows")
        all_frames.append(df)

    if not all_frames:
        raise ValueError("No result files found for any setting in LIST_SETTINGS")

    consolidated = pd.concat(all_frames, ignore_index=True)

    if verbose:
        print(f"\nTotal rows gathered: {len(consolidated)}")
        print(f"Columns: {list(consolidated.columns)}")
        if "model" in consolidated.columns:
            print(f"Models: {sorted(consolidated['model'].unique())}")
        if "scenario" in consolidated.columns:
            print(f"Scenarios: {sorted(consolidated['scenario'].unique())}")

    return consolidated


def parse_array_string(df: pd.DataFrame, list_columns: list[str]) -> pd.DataFrame:
    """Parse string array columns and flatten by prediction step.

    Converts columns containing string representations of arrays
    (e.g., "[0.1 0.2 0.3 0.4]") into individual rows per prediction step.
    """
    for column in list_columns:
        df[column] = df[column].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    flatten_records = []
    for _, row in df.iterrows():
        for step in range(len(row[list_columns[0]])):
            record = row.drop(list_columns).to_dict()
            record["pred_step"] = step
            record.update({col: row[col][step] for col in list_columns})
            flatten_records.append(record)

    return pd.DataFrame(flatten_records)


def save_consolidated_results(
    df: pd.DataFrame,
    output_dir: Path = Path(DIR_OUTPUTS) / "testing" / "results",
) -> Path:
    """Save consolidated DataFrame to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "consolidated_results.csv"
    df.to_csv(path, index=False)
    print(f"Saved to {path}")
    return path


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict of the consolidated data."""
    summary: dict = {
        "total_records": len(df),
    }
    for col in ["model", "scenario", "noise_type", "cm"]:
        if col in df.columns:
            summary[f"unique_{col}"] = sorted(df[col].unique().tolist())
            summary[f"records_per_{col}"] = df[col].value_counts().to_dict()
    for col in ["ds", "ms", "noise_degree"]:
        if col in df.columns:
            summary[f"{col}_range"] = (float(df[col].min()), float(df[col].max()))
    return summary


if __name__ == "__main__":
    print("Gathering results from all settings...")
    print("-" * 60)
    result_df = gather_all_results(verbose=True)
    save_consolidated_results(result_df)
    print("\nSummary:")
    for k, v in get_data_summary(result_df).items():
        print(f"  {k}: {v}")
