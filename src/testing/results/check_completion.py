"""Testing completion status checker for CSI prediction experiments.

Checks whether each (model, scenario, test_type) setting in LIST_SETTINGS
has produced the expected number of result rows.

Directory layout produced by prediction_performance/main.py::

    prediction_performance/{model}/{scenario}/{test_type}/slice_{idx}/{timestamp}/result.csv

For each setting we:
  1. Scan all slice_* subdirs.
  2. In each slice, find the latest timestamp folder, read result.csv.
  3. Sum rows across ALL slices (independent of JOBS_PER_* constants).
  4. Compare total against SCENARIOS_PER_TEST_TYPE[test_type].

Usage::

    python3 -m src.testing.results.check_completion
"""

from pathlib import Path

import pandas as pd

from src.testing.config import LIST_SETTINGS, SCENARIOS_PER_TEST_TYPE
from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_current_time, get_latest_datetime_folder


BASE_DIR = Path(DIR_OUTPUTS) / "testing" / "prediction_performance"


def _count_rows_in_slice(slice_dir: Path) -> int:
    """Read result.csv from the latest timestamp folder in *slice_dir*."""
    latest = get_latest_datetime_folder(slice_dir)
    if latest is None:
        return 0
    result_file = latest / "result.csv"
    if not result_file.exists():
        return 0
    try:
        return len(pd.read_csv(result_file))
    except Exception:
        return 0


def count_rows_for_setting(
    model: str,
    scenario: str,
    test_type: str,
    base_dir: Path = BASE_DIR,
) -> tuple[int, int]:
    """Sum result rows across all slices for one setting.

    Returns:
        (completed_rows, num_slices_found)
    """
    setting_dir = base_dir / model / scenario / test_type
    if not setting_dir.exists():
        return 0, 0

    total_rows = 0
    num_slices = 0
    for child in sorted(setting_dir.iterdir()):
        if child.is_dir() and child.name.startswith("slice_"):
            total_rows += _count_rows_in_slice(child)
            num_slices += 1

    return total_rows, num_slices


def check_testing_completion(
    base_dir: Path = BASE_DIR,
    verbose: bool = True,
    save_report: bool = False,
) -> pd.DataFrame:
    """Check completion status for every setting in LIST_SETTINGS.

    Returns a DataFrame with one row per setting containing columns:
        model, scenario, test_type, expected, completed, slices, status, pct
    """
    rows: list[dict] = []

    for model, scenario, test_type in LIST_SETTINGS:
        expected = SCENARIOS_PER_TEST_TYPE[test_type]
        completed, num_slices = count_rows_for_setting(model, scenario, test_type, base_dir)

        if completed == 0:
            status = "not_started"
        elif completed >= expected:
            status = "completed"
        else:
            status = "in_progress"

        rows.append(
            {
                "model": model,
                "scenario": scenario,
                "test_type": test_type,
                "expected": expected,
                "completed": completed,
                "slices": num_slices,
                "status": status,
                "pct": round(completed / expected * 100, 1) if expected > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows)

    if verbose:
        _print_report(df)

    if save_report:
        report_dir = Path(DIR_OUTPUTS) / "testing" / "results" / "completion_reports" / get_current_time()
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "completion_status.csv"
        df.to_csv(path, index=False)
        print(f"Report saved to {path}")

    return df


def _print_report(df: pd.DataFrame) -> None:
    """Print a human-readable completion report."""
    n_total = len(df)
    n_done = (df["status"] == "completed").sum()
    n_prog = (df["status"] == "in_progress").sum()
    n_none = (df["status"] == "not_started").sum()

    print("=" * 90)
    print("PREDICTION PERFORMANCE COMPLETION STATUS")
    print("=" * 90)
    print(f"Settings: {n_total}  |  Completed: {n_done}  |  In-progress: {n_prog}  |  Not started: {n_none}")
    print()
    print(df.to_string(index=False))
    print()

    models = sorted(df["model"].unique())
    print("PER-MODEL SUMMARY:")
    print("-" * 60)
    for m in models:
        sub = df[df["model"] == m]
        done = (sub["status"] == "completed").sum()
        total_expected = sub["expected"].sum()
        total_completed = sub["completed"].sum()
        print(f"  {m:<10}  {done}/{len(sub)} settings done   ({total_completed}/{total_expected} rows)")
    print()

    all_complete = (df["status"] == "completed").all()
    print(f"All complete: {all_complete}")


if __name__ == "__main__":
    check_testing_completion(verbose=True, save_report=False)
