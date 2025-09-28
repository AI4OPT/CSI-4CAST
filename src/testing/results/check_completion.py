"""Testing completion status checker for CSI prediction experiments.

This module provides comprehensive functionality to check the completion status of
CSI prediction testing jobs. It can analyze both cluster-based slice testing and
local full testing modes to determine which models have completed their evaluation.

Key Features:
- Scan cluster slice directories for completion status
- Scan local full_test directories for completion status
- Count completed combinations vs expected combinations
- Generate detailed completion reports with statistics
- Support for both individual slice analysis and overall model analysis
- Automatic detection of latest timestamp directories

Typical Usage:
    # Check completion status with detailed reporting
    completion_result = check_testing_completion(verbose=True, save_report=True)

    # Check if all models are complete
    if completion_result['all_complete']:
        print("All testing is complete!")
    else:
        print(f"Incomplete models: {len(completion_result['model_completion'])}")
"""

from pathlib import Path

import pandas as pd

from src.testing.config import (
    JOBS_PER_MODEL,
    LIST_MODELS,
    create_all_combinations,
    slice_combinations,
)
from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_current_time, get_latest_datetime_folder


def count_completed_combinations(slice_dir: Path) -> int:
    """Count the number of completed combinations in a slice directory.

    This function examines a slice directory to determine how many test combinations
    have been completed by counting rows in the result.csv file from the most recent
    timestamp directory. This is used to track progress of cluster-based testing jobs.

    Args:
        slice_dir: Path to the slice directory (e.g., testing1/MODEL3/slice_1/)
                  Expected structure: slice_dir/TIMESTAMP/result.csv

    Returns:
        int: Number of completed combinations (rows in result.csv).
             Returns 0 if:
             - Directory doesn't exist
             - No timestamp directories found
             - No result.csv file found
             - File is empty or corrupted

    """
    # Check if the slice directory exists
    if not slice_dir.exists():
        return 0

    # Find the most recent timestamp directory within the slice
    latest_timestamp_dir = get_latest_datetime_folder(slice_dir)
    if latest_timestamp_dir is None:
        return 0

    result_file = latest_timestamp_dir / "result.csv"
    if not result_file.exists():
        return 0

    try:
        # Count rows in CSV file (excluding header)
        df = pd.read_csv(result_file)
        return len(df)
    except Exception:
        # If file is empty or corrupted, return 0
        return 0


def count_completed_combinations_full_test(full_test_dir: Path) -> int:
    """Count the number of completed combinations in a full_test directory.

    This function examines a full_test directory to determine how many test combinations
    have been completed by counting rows in the result.csv file from the most recent
    timestamp directory. This is used to track progress of local full testing jobs.

    Args:
        full_test_dir: Path to the full_test directory
                      (e.g., testing/prediction_performance/MODEL3/full_test/)
                      Expected structure: full_test_dir/TIMESTAMP/result.csv

    Returns:
        int: Number of completed combinations (rows in result.csv).
             Returns 0 if:
             - Directory doesn't exist
             - No timestamp directories found
             - No result.csv file found
             - File is empty or corrupted

    """
    # Check if the full_test directory exists
    if not full_test_dir.exists():
        return 0

    # Find the most recent timestamp directory within full_test
    latest_timestamp_dir = get_latest_datetime_folder(full_test_dir)
    if latest_timestamp_dir is None:
        return 0

    result_file = latest_timestamp_dir / "result.csv"
    if not result_file.exists():
        return 0

    try:
        # Count rows in CSV file (excluding header)
        df = pd.read_csv(result_file)
        return len(df)
    except Exception:
        # If file is empty or corrupted, return 0
        return 0


def get_expected_combinations_for_slice(slice_idx: int, total_jobs: int) -> int:
    """Get the expected number of combinations for a specific slice.

    This function calculates how many test combinations should be present in a
    specific slice based on the slice index and total number of jobs. It uses
    the same slicing logic as the test execution to determine expected counts.

    Args:
        slice_idx: 0-based slice index (e.g., 0 for slice_1, 1 for slice_2)
        total_jobs: Total number of jobs per model (JOBS_PER_MODEL)

    Returns:
        int: Expected number of combinations for this slice.
             This represents how many rows should be in result.csv when complete.

    """
    # Generate all possible test combinations
    all_combinations = create_all_combinations()
    # Apply slicing logic to get combinations for this specific slice
    sliced_combinations = slice_combinations(all_combinations, (slice_idx, total_jobs))
    return len(sliced_combinations)


def scan_local_completion_status(
    base_prediction_performance: Path = Path(DIR_OUTPUTS) / "testing" / "prediction_performance",
) -> list[dict]:
    """Scan local mode full_test directories and collect completion status.

    This function examines all model directories for local full_test completion status.
    It checks for the presence of result.csv files and counts completed combinations
    to determine the completion percentage and status of each model's full testing.

    Local full_test mode runs all test combinations in a single job, unlike cluster
    mode which splits combinations across multiple slice jobs.

    Args:
        base_prediction_performance: Base directory containing model test results.
                                   Expected structure: base_dir/MODEL_NAME/full_test/

    Returns:
        list[dict]: List of dictionaries, one per model, containing:
            - model_name: Name of the model
            - test_type: "local_full_test"
            - status: "not_started", "started_no_results", "in_progress", "completed", or "error"
            - complete_num: Number of completed combinations
            - expected_num: Expected number of combinations (all combinations)
            - incomplete_num: Number of remaining combinations
            - percentage: Completion percentage (0.0-100.0)
            - latest_timestamp: Name of latest timestamp directory or "N/A"
            - dir_exists: Whether the full_test directory exists

    """
    results = []

    # Check if the base testing directory exists
    if not base_prediction_performance.exists():
        print(f"Output directory does not exist: {base_prediction_performance}")
        return results

    # Iterate through all configured models to check their full_test status
    for model_name in LIST_MODELS:
        full_test_dir = base_prediction_performance / model_name / "full_test"

        # Initialize result entry with default values
        result_entry = {
            "model_name": model_name,
            "test_type": "local_full_test",  # Distinguish from cluster slice testing
            "status": "not_started",
            "complete_num": 0,
            "expected_num": 0,
            "incomplete_num": 0,
            "percentage": 0.0,
            "latest_timestamp": "N/A",
            "dir_exists": full_test_dir.exists(),
        }

        try:
            if full_test_dir.exists():
                # Get expected combinations (all combinations for full test)
                all_combinations = create_all_combinations()
                expected_count = len(all_combinations)
                result_entry["expected_num"] = expected_count

                # Find latest timestamp directory and get its name
                latest_timestamp_dir = get_latest_datetime_folder(full_test_dir)
                if latest_timestamp_dir is not None:
                    result_entry["latest_timestamp"] = latest_timestamp_dir.name

                # Count completed combinations by reading result.csv
                completed_count = count_completed_combinations_full_test(full_test_dir)
                result_entry["complete_num"] = completed_count
                result_entry["incomplete_num"] = expected_count - completed_count

                # Calculate percentage
                if expected_count > 0:
                    result_entry["percentage"] = (completed_count / expected_count) * 100.0

                # Update status based on completion
                if latest_timestamp_dir is None:
                    result_entry["status"] = "not_started"
                elif completed_count == 0:
                    result_entry["status"] = "started_no_results"
                elif completed_count == expected_count:
                    result_entry["status"] = "completed"
                else:
                    result_entry["status"] = "in_progress"

        except Exception as e:
            result_entry["status"] = "error"
            result_entry["error_message"] = str(e)

        results.append(result_entry)

    return results


def scan_completion_status() -> list[dict]:
    """Scan all output directories and collect completion status.

    Returns:
        List of dictionaries with completion status for each slice

    """
    results = []
    base_prediction_performance = Path(DIR_OUTPUTS) / "testing" / "prediction_performance"

    if not base_prediction_performance.exists():
        print(f"Output directory does not exist: {base_prediction_performance}")
        return results

    # Scan for all model slice directories
    for model_name in LIST_MODELS:
        for slice_idx in range(JOBS_PER_MODEL):
            slice_name = f"slice_{slice_idx + 1}"
            slice_dir = base_prediction_performance / model_name / slice_name

            # Initialize result entry
            result_entry = {
                "model_name": model_name,
                "test_type": "cluster_slice",
                "slice_id": slice_idx + 1,  # 1-based for display
                "slice_name": slice_name,
                "slice_path": f"{model_name}/{slice_name}",
                "status": "not_started",
                "complete_num": 0,
                "expected_num": 0,
                "incomplete_num": 0,
                "percentage": 0.0,
                "latest_timestamp": "N/A",
                "dir_exists": slice_dir.exists(),
            }

            try:
                if slice_dir.exists():
                    # Get expected combinations for this slice
                    expected_count = get_expected_combinations_for_slice(slice_idx, JOBS_PER_MODEL)
                    result_entry["expected_num"] = expected_count

                    # Find latest timestamp directory and get its name
                    latest_timestamp_dir = get_latest_datetime_folder(slice_dir)
                    if latest_timestamp_dir is not None:
                        result_entry["latest_timestamp"] = latest_timestamp_dir.name

                    # Count completed combinations by reading result.csv
                    completed_count = count_completed_combinations(slice_dir)
                    result_entry["complete_num"] = completed_count
                    result_entry["incomplete_num"] = expected_count - completed_count

                    # Calculate percentage
                    if expected_count > 0:
                        result_entry["percentage"] = (completed_count / expected_count) * 100.0

                    # Update status based on completion
                    if latest_timestamp_dir is None:
                        result_entry["status"] = "not_started"
                    elif completed_count == 0:
                        result_entry["status"] = "started_no_results"
                    elif completed_count == expected_count:
                        result_entry["status"] = "completed"
                    else:
                        result_entry["status"] = "in_progress"

            except Exception as e:
                result_entry["status"] = "error"
                result_entry["error_message"] = str(e)

            results.append(result_entry)

    return results


def create_completion_table(results: list[dict]) -> pd.DataFrame:
    """Create a formatted completion table from results.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with completion status

    """
    df = pd.DataFrame(results)

    # Select and reorder columns for display
    display_columns = [
        "model_name",
        "test_type",
        "slice_id",
        "status",
        "complete_num",
        "incomplete_num",
        "expected_num",
        "percentage",
        "latest_timestamp",
    ]

    # Handle missing slice_id column for local tests
    if "slice_id" not in df.columns:
        df["slice_id"] = "N/A"

    df = df[display_columns].copy()

    # Format percentage to 1 decimal place
    df["percentage"] = df["percentage"].round(1)

    return df  # type: ignore


def print_summary_statistics(results: list[dict]):
    """Print summary statistics across all models and slices.

    Args:
        results: List of result dictionaries

    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_slices = len(results)
    completed_slices = sum(1 for r in results if r["status"] == "completed")
    in_progress_slices = sum(1 for r in results if r["status"] == "in_progress")
    not_started_slices = sum(1 for r in results if r["status"] == "not_started")
    error_slices = sum(1 for r in results if r["status"] == "error")

    print(f"Total slices: {total_slices}")
    print(f"Completed: {completed_slices} ({completed_slices / total_slices * 100:.1f}%)")
    print(f"In progress: {in_progress_slices} ({in_progress_slices / total_slices * 100:.1f}%)")
    print(f"Not started: {not_started_slices} ({not_started_slices / total_slices * 100:.1f}%)")
    print(f"Errors: {error_slices} ({error_slices / total_slices * 100:.1f}%)")

    # Per-model statistics
    print("\nPER-MODEL STATISTICS:")
    print("-" * 50)
    for model_name in LIST_MODELS:
        model_results = [r for r in results if r["model_name"] == model_name]
        if not model_results:
            continue

        print(f"{model_name}:")

        # Separate slice and local results
        slice_results = [r for r in model_results if r.get("test_type") == "cluster_slice"]
        local_results = [r for r in model_results if r.get("test_type") == "local_full_test"]

        if slice_results:
            slice_completed = sum(1 for r in slice_results if r["status"] == "completed")
            slice_total_complete = sum(r["complete_num"] for r in slice_results)
            slice_total_expected = sum(r["expected_num"] for r in slice_results)
            print(f"  Slice tests: {slice_completed}/{len(slice_results)} slices completed")
            print(f"  Slice combinations: {slice_total_complete}/{slice_total_expected}")
            if slice_total_expected > 0:
                print(f"  Slice completion: {slice_total_complete / slice_total_expected * 100:.1f}%")

        if local_results:
            local_completed = sum(1 for r in local_results if r["status"] == "completed")
            local_total_complete = sum(r["complete_num"] for r in local_results)
            local_total_expected = sum(r["expected_num"] for r in local_results)
            print(f"  Local tests: {local_completed}/{len(local_results)} full tests completed")
            print(f"  Local combinations: {local_total_complete}/{local_total_expected}")
            if local_total_expected > 0:
                print(f"  Local completion: {local_total_complete / local_total_expected * 100:.1f}%")

        print()


def save_results(results: list[dict], output_dir: Path):
    """Save results to CSV and create a summary report.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save results

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    df = pd.DataFrame(results)
    timestamp = get_current_time()
    csv_path = output_dir / f"completion_status_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    # Save summary table
    summary_df = create_completion_table(results)
    summary_path = output_dir / f"completion_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")


def check_testing_completion(
    base_prediction_performance: Path = Path(DIR_OUTPUTS) / "testing" / "prediction_performance",
    verbose: bool = True,
    save_report: bool = False,
) -> dict:
    """Check the completion status of all testing jobs.

    Args:
        base_prediction_performance: Base directory containing test results
        verbose: Whether to print detailed status information
        save_report: Whether to save completion report to disk

    Returns:
        Dictionary containing:
        - 'all_complete': bool, whether all models are complete
        - 'model_completion': dict, completion status for each model
        - 'completion_results': list, detailed results for all slices/tests
        - 'summary': dict, summary statistics

    """
    if verbose:
        print("Prediction Performance Testing Completion Checker")
        print("=" * 60)
        print(f"Base output directory: {base_prediction_performance}")
        print(f"Total models: {len(LIST_MODELS)}")

    all_results = []

    if verbose:
        print(f"Jobs per model: {JOBS_PER_MODEL}")
        print(f"Total slices to check: {len(LIST_MODELS) * JOBS_PER_MODEL}")
        print("\nScanning slice directories...")

    slice_results = scan_completion_status()
    all_results.extend(slice_results)

    if verbose:
        print("\nScanning local full_test directories...")
    local_results = scan_local_completion_status()
    all_results.extend(local_results)

    if not all_results:
        if verbose:
            print("No results found!")
        return {
            "all_complete": False,
            "model_completion": {},
            "completion_results": [],
            "summary": {},
        }

    # Analyze completion status per model
    model_completion = {}
    for model_name in LIST_MODELS:
        model_results = [r for r in all_results if r["model_name"] == model_name]

        # Check slice completion
        slice_results_model = [r for r in model_results if r.get("test_type") == "cluster_slice"]
        slice_complete = all(r["status"] == "completed" for r in slice_results_model) if slice_results_model else False

        # Check local full_test completion
        local_results_model = [r for r in model_results if r.get("test_type") == "local_full_test"]
        local_complete = all(r["status"] == "completed" for r in local_results_model) if local_results_model else False

        # Get latest timestamps for decision making
        slice_latest_timestamp = None
        local_latest_timestamp = None

        if slice_complete and slice_results_model:
            slice_latest_timestamp = max(
                r["latest_timestamp"] for r in slice_results_model if r["latest_timestamp"] != "N/A"
            )

        if local_complete and local_results_model:
            local_latest_timestamp = max(
                r["latest_timestamp"] for r in local_results_model if r["latest_timestamp"] != "N/A"
            )

        # Determine which source to use (prefer more recent)
        use_source = None
        if slice_complete and local_complete:
            # Both complete, choose based on timestamp
            if slice_latest_timestamp and local_latest_timestamp:
                use_source = "local" if local_latest_timestamp > slice_latest_timestamp else "slice"
            elif local_latest_timestamp:
                use_source = "local"
            elif slice_latest_timestamp:
                use_source = "slice"
        elif local_complete:
            use_source = "local"
        elif slice_complete:
            use_source = "slice"

        model_completion[model_name] = {
            "slice_complete": slice_complete,
            "local_complete": local_complete,
            "model_complete": slice_complete or local_complete,
            "use_source": use_source,
            "slice_latest_timestamp": slice_latest_timestamp,
            "local_latest_timestamp": local_latest_timestamp,
        }

    # Check if all models are complete
    all_complete = all(status["model_complete"] for status in model_completion.values())

    # Create summary statistics
    summary = {
        "total_models": len(LIST_MODELS),
        "complete_models": sum(1 for status in model_completion.values() if status["model_complete"]),
        "slice_only_complete": sum(
            1 for status in model_completion.values() if status["slice_complete"] and not status["local_complete"]
        ),
        "local_only_complete": sum(
            1 for status in model_completion.values() if status["local_complete"] and not status["slice_complete"]
        ),
        "both_complete": sum(
            1 for status in model_completion.values() if status["slice_complete"] and status["local_complete"]
        ),
        "all_testing_complete": all_complete,
    }

    if verbose:
        # Display completion table
        print("\nCOMPLETION STATUS TABLE:")
        print("-" * 120)
        completion_df = create_completion_table(all_results)
        print(completion_df.to_string(index=False))

        # Print summary statistics
        print_summary_statistics(all_results)

        # Print model-specific completion status
        print("\nMODEL COMPLETION SUMMARY:")
        print("-" * 50)
        for model_name, status in model_completion.items():
            print(f"{model_name}: Complete={status['model_complete']}, Use={status['use_source']}")

        print(f"\nOverall Testing Complete: {all_complete}")

    if save_report:
        cur_time = get_current_time()
        output_dir = Path(DIR_OUTPUTS) / "testing" / "results" / "completion_reports" / cur_time
        save_results(all_results, output_dir)

    return {
        "all_complete": all_complete,
        "model_completion": model_completion,
        "completion_results": all_results,
        "summary": summary,
    }
