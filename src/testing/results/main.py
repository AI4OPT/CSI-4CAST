"""Main script for testing completion checking and results gathering.

This script implements the complete workflow:
1. Check completion status of all testing jobs
2. Determine which models are complete (either full_test or all slices)
3. If all models are complete, gather results from appropriate sources
4. Process and save consolidated results

Usage:
    python main.py

"""

import sys
from pathlib import Path

from src.testing.results.analysis_df import CSIResultsAnalyzer
from src.testing.results.check_completion import check_testing_completion
from src.testing.results.gather_results import (
    gather_all_results,
    get_data_summary,
    parse_array_string,
    save_consolidated_results,
)
from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_current_time


def main(
    base_prediction_performance: Path = Path(DIR_OUTPUTS) / "testing" / "prediction_performance",
    save_results: bool = True,
    verbose: bool = True,
) -> dict:
    """Main function to orchestrate testing completion check and results gathering.

    Args:
        base_prediction_performance: Path to the base testing results directory
        save_results: Whether to save consolidated results to disk
        verbose: Whether to print detailed progress information

    Returns:
        Dictionary containing:
        - 'completion_status': Results from completion checking
        - 'consolidated_df': Consolidated DataFrame (if testing complete)
        - 'results_saved_to': Path where results were saved (if applicable)
        - 'summary': Data summary statistics (if applicable)

    """
    if verbose:
        print("=" * 80)
        print("PREDICTION PERFORMANCE TESTING - MAIN WORKFLOW")
        print("=" * 80)
        print(f"Base testing path: {base_prediction_performance}")
        print(f"Current time: {get_current_time()}")
        print()

    # Step 1: Check completion status
    if verbose:
        print("#############################################")
        print("### STEP 1: Checking completion status... ###")
        print("#############################################")
        print("-" * 50)

    completion_result = check_testing_completion(
        base_prediction_performance=base_prediction_performance,
        verbose=verbose,
        save_report=save_results,
    )

    if verbose:
        print("\nCompletion check finished.")
        print(f"All testing complete: {completion_result['all_complete']}")
        print(
            f"Complete models: {completion_result['summary']['complete_models']}/{completion_result['summary']['total_models']}"
        )

    # Step 2: Check if all testing is complete
    if not completion_result["all_complete"]:
        if verbose:
            print("\nTesting is not complete yet. Cannot proceed with results gathering.")
            print("Incomplete models:")
            for model_name, status in completion_result["model_completion"].items():
                if not status["model_complete"]:
                    print(
                        f"  - {model_name}: slice_complete={status['slice_complete']}, local_complete={status['local_complete']}"
                    )

        return {
            "completion_status": completion_result,
            "consolidated_df": None,
            "results_saved_to": None,
            "summary": None,
            "message": "Testing not complete - cannot gather results",
        }

    # Step 3: Gather results from completed models
    if verbose:
        print("\n" + "=" * 80)
        print("##########################################################")
        print("### STEP 2: All models complete - gathering results... ###")
        print("##########################################################")
        print("-" * 50)

    try:
        consolidated_df = gather_all_results(
            model_completion=completion_result["model_completion"],
            base_prediction_performance=base_prediction_performance,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"Error gathering results: {e}")
        return {
            "completion_status": completion_result,
            "consolidated_df": None,
            "results_saved_to": None,
            "summary": None,
            "message": f"Error gathering results: {e}",
        }

    # Step 4: Process array columns
    if verbose:
        print("\n" + "-" * 50)
        print("############################################")
        print("### STEP 3: Processing array columns... ###")
        print("############################################")

    try:
        # Parse array string columns to flatten the data
        list_columns = ["nmse_mean", "nmse_std", "se_mean", "se_std", "se0_mean", "se0_std"]

        # Check which columns actually exist in the dataframe
        existing_list_columns = [col for col in list_columns if col in consolidated_df.columns]

        if existing_list_columns:
            if verbose:
                print(f"Parsing array columns: {existing_list_columns}")
            consolidated_df = parse_array_string(df=consolidated_df, list_columns=existing_list_columns)
        else:
            if verbose:
                print("No array columns found to parse")

    except Exception as e:
        if verbose:
            print(f"Warning: Error parsing array columns: {e}")
            print("Proceeding without array parsing...")

    # Step 5: Save results
    results_saved_to = None
    if save_results:
        if verbose:
            print("\n" + "-" * 50)
            print("##############################################")
            print("### STEP 4: Saving consolidated results... ###")
            print("##############################################")

        try:
            output_dir = Path(DIR_OUTPUTS) / "testing" / "results" / "gather" / get_current_time()
            results_saved_to = save_consolidated_results(consolidated_df, output_dir=output_dir)
            if verbose:
                print(f"Results saved to: {results_saved_to}")
        except Exception as e:
            if verbose:
                print(f"Error saving results: {e}")

    # Step 6: Run analysis on consolidated results
    if verbose:
        print("\n" + "-" * 50)
        print("###########################################################")
        print("### STEP 5: Running analysis on consolidated results... ###")
        print("###########################################################")

    try:
        # Initialize analyzer with the saved results path
        if results_saved_to:
            analyzer = CSIResultsAnalyzer(results_path=results_saved_to)
            analyzer.run_analysis(metric_types=["nmse", "se"], save_results=save_results)
            if verbose:
                print("Analysis completed successfully!")
        else:
            if verbose:
                print("Skipping analysis - no results file saved")
    except Exception as e:
        if verbose:
            print(f"Error running analysis: {e}")

    # Step 7: Generate summary
    if verbose:
        print("\n" + "-" * 50)
        print("##########################################")
        print("### STEP 6: Generating data summary... ###")
        print("##########################################")

    try:
        summary = get_data_summary(consolidated_df)

        if verbose:
            print("\n" + "=" * 50)
            print("DATA SUMMARY")
            print("=" * 50)
            for key, value in summary.items():
                print(f"{key}: {value}")

            print(f"\nConsolidated DataFrame shape: {consolidated_df.shape}")
            print("\nFirst few rows:")
            print(consolidated_df.head())

    except Exception as e:
        if verbose:
            print(f"Error generating summary: {e}")
        summary = None

    if verbose:
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)

    return {
        "completion_status": completion_result,
        "consolidated_df": consolidated_df,
        "results_saved_to": results_saved_to,
        "summary": summary,
        "message": "Success - all testing complete and results gathered",
    }


if __name__ == "__main__":
    # Parse command line arguments (simple implementation)
    verbose = True
    save_results = True

    if len(sys.argv) > 1:
        if "--quiet" in sys.argv:
            verbose = False
        if "--no-save" in sys.argv:
            save_results = False

    # Run the main workflow
    try:
        result = main(verbose=verbose, save_results=save_results)

        if result["consolidated_df"] is not None:
            print(f"\nSUCCESS: Processed {len(result['consolidated_df'])} records")
            if result["results_saved_to"]:
                print(f"Results saved to: {result['results_saved_to']}")
        else:
            print(f"\nSTATUS: {result['message']}")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
