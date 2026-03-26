"""Main orchestration script for CSI prediction testing results.

Workflow:
  1. Check completion status for all settings in LIST_SETTINGS.
  2. If all complete, gather and consolidate result CSVs.
  3. Parse array columns and flatten by prediction step.
  4. Save consolidated results and run analysis.

Usage::

    python3 -m src.testing.results.main
    python3 -m src.testing.results.main --quiet --no-save
    python3 -m src.testing.results.main --baseline-only
"""

from pathlib import Path
import sys

from src.testing.results.check_completion import check_testing_completion
from src.testing.results.gather_results import (
    gather_all_results,
    get_data_summary,
    parse_array_string,
    save_consolidated_results,
)
from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_current_time


BASE_DIR = Path(DIR_OUTPUTS) / "testing" / "prediction_performance"
BASELINE_AND_PROPOSED_MODELS = (
    "MODEL",
    "LLM4CP",
    "RNN",
    "CNN",
    "STEMGNN",
    "PAD",
    "AR",
    "WIENER",
    "NP",
)


def filter_to_baseline_models(
    consolidated_df,
    verbose: bool = True,
):
    """Keep only the proposed model and baseline models."""
    if "model" not in consolidated_df.columns:
        raise ValueError("Cannot apply baseline-only mode because the consolidated data has no 'model' column")

    filtered_df = consolidated_df[consolidated_df["model"].isin(BASELINE_AND_PROPOSED_MODELS)].copy()

    if filtered_df.empty:
        raise ValueError("Baseline-only mode removed all rows; check model names in the consolidated results")

    if verbose:
        removed_models = sorted(set(consolidated_df["model"].unique()) - set(filtered_df["model"].unique()))
        print("\n--- Step 3b: Filtering models (baseline-only mode) ---")
        print(f"Keeping models: {list(BASELINE_AND_PROPOSED_MODELS)}")
        print(f"Removed models: {removed_models if removed_models else 'None'}")
        print(f"Rows after filtering: {len(filtered_df)}")

    return filtered_df


def main(
    base_dir: Path = BASE_DIR,
    save_results: bool = True,
    verbose: bool = True,
    baseline_only: bool = False,
) -> dict:
    """Run the complete results workflow: check -> gather -> analyse."""
    if verbose:
        print("=" * 80)
        print("PREDICTION PERFORMANCE TESTING - RESULTS WORKFLOW")
        print("=" * 80)

    # --- Step 1: completion check ---
    if verbose:
        print("\n--- Step 1: Checking completion status ---")
    completion_df = check_testing_completion(base_dir=base_dir, verbose=verbose, save_report=save_results)
    all_complete = (completion_df["status"] == "completed").all()

    if not all_complete:
        incomplete = completion_df[completion_df["status"] != "completed"]
        if verbose:
            print("\nTesting is NOT complete. Cannot gather results.")
            print(f"{len(incomplete)} settings still incomplete.")
        return {
            "completion_df": completion_df,
            "consolidated_df": None,
            "results_saved_to": None,
            "summary": None,
            "message": "Testing not complete",
        }

    # --- Step 2: gather results ---
    if verbose:
        print("\n--- Step 2: Gathering results ---")
    try:
        consolidated_df = gather_all_results(base_dir=base_dir, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"Error gathering results: {e}")
        return {
            "completion_df": completion_df,
            "consolidated_df": None,
            "results_saved_to": None,
            "summary": None,
            "message": f"Error gathering results: {e}",
        }

    # --- Step 3: parse array columns ---
    if verbose:
        print("\n--- Step 3: Parsing array columns ---")
    try:
        array_cols = ["nmse_mean", "nmse_std", "se_mean", "se_std", "se0_mean", "se0_std"]
        existing = [c for c in array_cols if c in consolidated_df.columns]
        if existing:
            consolidated_df = parse_array_string(consolidated_df, existing)
            if verbose:
                print(f"Parsed columns: {existing}  ->  {len(consolidated_df)} rows after flattening")
    except Exception as e:
        if verbose:
            print(f"Warning: array parsing failed: {e}")

    # --- Step 3b: optional model filtering ---
    if baseline_only:
        try:
            consolidated_df = filter_to_baseline_models(consolidated_df, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Error applying baseline-only mode: {e}")
            return {
                "completion_df": completion_df,
                "consolidated_df": None,
                "results_saved_to": None,
                "summary": None,
                "message": f"Error applying baseline-only mode: {e}",
            }

    # --- Step 4: save ---
    results_saved_to = None
    if save_results:
        if verbose:
            print("\n--- Step 4: Saving consolidated results ---")
        try:
            out_dir = Path(DIR_OUTPUTS) / "testing" / "results" / "gather" / get_current_time()
            results_saved_to = save_consolidated_results(consolidated_df, output_dir=out_dir)
        except Exception as e:
            if verbose:
                print(f"Error saving: {e}")

    # --- Step 5: analysis ---
    if verbose:
        print("\n--- Step 5: Running analysis ---")
    try:
        if results_saved_to:
            from src.testing.results.analysis_df import CSIResultsAnalyzer

            analyzer = CSIResultsAnalyzer(results_path=results_saved_to)
            analyzer.run_analysis(metric_types=["nmse", "se"], save_results=save_results)
            if verbose:
                print("Analysis completed.")
    except Exception as e:
        if verbose:
            print(f"Analysis error: {e}")

    # --- Step 6: summary ---
    summary = None
    try:
        summary = get_data_summary(consolidated_df)
        if verbose:
            print("\n--- Data summary ---")
            for k, v in summary.items():
                print(f"  {k}: {v}")
    except Exception as e:
        if verbose:
            print(f"Summary error: {e}")

    if verbose:
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED")
        print("=" * 80)

    return {
        "completion_df": completion_df,
        "consolidated_df": consolidated_df,
        "results_saved_to": results_saved_to,
        "summary": summary,
        "baseline_only": baseline_only,
        "message": "Success",
    }


if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    save = "--no-save" not in sys.argv
    baseline_only = "--baseline-only" in sys.argv

    try:
        result = main(verbose=verbose, save_results=save, baseline_only=baseline_only)
        if result["consolidated_df"] is not None:
            print(f"\nProcessed {len(result['consolidated_df'])} records")
        else:
            print(f"\n{result['message']}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
