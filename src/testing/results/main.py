"""Main orchestration script for CSI prediction testing workflow.

This script implements the complete end-to-end workflow for CSI prediction testing
analysis, from completion checking to final result analysis. It serves as the main
entry point for processing completed test results and generating comprehensive
performance analysis reports.

Workflow Overview:
1. **Completion Checking**: Scan all model directories to determine completion status
2. **Source Selection**: Intelligently choose between cluster slices and local full_test data
3. **Data Gathering**: Consolidate results from appropriate sources into unified DataFrame
4. **Data Processing**: Parse array columns and flatten prediction step data
5. **Result Saving**: Save consolidated results with proper timestamping
6. **Analysis Execution**: Run comprehensive ranking and statistical analysis
7. **Summary Generation**: Create detailed data summaries and reports

Key Features:
- Automatic detection of completed vs incomplete models
- Intelligent data source selection (slice vs full_test)
- Robust error handling with graceful degradation
- Comprehensive progress reporting and logging
- Timestamped output directories for result organization
- Integration with downstream analysis tools
- Support for both verbose and quiet execution modes

Usage:
    # Basic usage with default settings
    python3 -m src.testing.results.main

    # Programmatic usage
    from src.testing.results.main import main
    result = main(verbose=True, save_results=True)

Output Structure:
    - Completion reports: z_artifacts/outputs/testing/results/completion_reports/
    - Consolidated results: z_artifacts/outputs/testing/results/gather/
    - Analysis outputs: z_artifacts/outputs/testing/results/analysis/

Dependencies:
    - src.testing.results.check_completion: For completion status checking
    - src.testing.results.gather_results: For result gathering and consolidation
    - src.testing.results.analysis_df: For comprehensive performance analysis
"""

from pathlib import Path
import sys

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
    """Orchestrate function for the complete CSI testing analysis workflow.

    This function coordinates the entire process from completion checking to final
    analysis, implementing a robust pipeline that handles various edge cases and
    provides comprehensive error reporting. It serves as the primary interface
    for both programmatic and command-line usage.

    The function implements a multi-stage workflow:
    1. **Completion Assessment**: Determines which models have finished testing
    2. **Data Validation**: Ensures all required models are complete before proceeding
    3. **Intelligent Gathering**: Selects optimal data sources per model
    4. **Data Processing**: Handles array parsing and prediction step flattening
    5. **Result Persistence**: Saves consolidated data with proper organization
    6. **Analysis Integration**: Triggers comprehensive performance analysis
    7. **Summary Generation**: Provides detailed statistics and insights

    Error Handling:
    - Graceful handling of incomplete testing scenarios
    - Robust file I/O with detailed error reporting
    - Continuation on non-critical errors with user notification
    - Comprehensive logging for troubleshooting

    Args:
        base_prediction_performance: Path to the base testing results directory.
                                   Expected structure: base_dir/MODEL_NAME/slice_X/ or full_test/
                                   Defaults to standard output directory location.
        save_results: Whether to save consolidated results and analysis outputs to disk.
                     If False, results are returned but not persisted.
        verbose: Whether to print detailed progress information, timing, and statistics.
                If False, only critical messages and errors are displayed.

    Returns:
        dict: Comprehensive results dictionary containing:
            - 'completion_status': Full results from check_testing_completion()
                - 'all_complete': Boolean indicating if all models finished
                - 'model_completion': Per-model completion details
                - 'completion_results': Detailed slice/test status information
                - 'summary': Aggregate completion statistics
            - 'consolidated_df': Unified DataFrame with all test results (if complete)
            - 'results_saved_to': Path to saved consolidated results file (if saved)
            - 'summary': Data summary statistics including coverage and distributions
            - 'message': Human-readable status message describing the outcome

    Raises:
        Exception: Critical errors that prevent workflow completion are propagated
                  after logging. Non-critical errors are handled gracefully.

    Example:
        >>> result = main(verbose=True, save_results=True)
        >>> if result['consolidated_df'] is not None:
        ...     print(f"Processed {len(result['consolidated_df'])} records")
        ... else:
        ...     print(f"Status: {result['message']}")

    """
    # Print workflow header and configuration information
    if verbose:
        print("=" * 80)
        print("PREDICTION PERFORMANCE TESTING - MAIN WORKFLOW")
        print("=" * 80)
        print(f"Base testing path: {base_prediction_performance}")
        print(f"Current time: {get_current_time()}")
        print()

    # ============================================================================
    # STEP 1: COMPLETION STATUS ASSESSMENT
    # ============================================================================
    # Check which models have completed testing (either all slices or full_test)
    if verbose:
        print("#############################################")
        print("### STEP 1: Checking completion status... ###")
        print("#############################################")
        print("-" * 50)

    # Perform comprehensive completion analysis across all models
    completion_result = check_testing_completion(
        base_prediction_performance=base_prediction_performance,
        verbose=verbose,
        save_report=save_results,  # Save detailed completion report if requested
    )

    # Report completion status summary
    if verbose:
        print("\nCompletion check finished.")
        print(f"All testing complete: {completion_result['all_complete']}")
        print(
            f"Complete models: {completion_result['summary']['complete_models']}/{completion_result['summary']['total_models']}"  # noqa: E501
        )

    # ============================================================================
    # STEP 2: COMPLETION VALIDATION
    # ============================================================================
    # Ensure all models are complete before proceeding to data gathering
    if not completion_result["all_complete"]:
        if verbose:
            print("\nTesting is not complete yet. Cannot proceed with results gathering.")
            print("Incomplete models:")
            # List which models are still incomplete and why
            for model_name, status in completion_result["model_completion"].items():
                if not status["model_complete"]:
                    print(
                        f"  - {model_name}: slice_complete={status['slice_complete']}, local_complete={status['local_complete']}"  # noqa: E501
                    )

        # Return early with incomplete status information
        return {
            "completion_status": completion_result,
            "consolidated_df": None,
            "results_saved_to": None,
            "summary": None,
            "message": "Testing not complete - cannot gather results",
        }

    # ============================================================================
    # STEP 3: INTELLIGENT DATA GATHERING
    # ============================================================================
    # All models are complete - proceed with gathering results from optimal sources
    if verbose:
        print("\n" + "=" * 80)
        print("##########################################################")
        print("### STEP 2: All models complete - gathering results... ###")
        print("##########################################################")
        print("-" * 50)

    try:
        # Intelligently gather results from appropriate sources per model
        consolidated_df = gather_all_results(
            model_completion=completion_result["model_completion"],
            base_prediction_performance=base_prediction_performance,
            verbose=verbose,
        )
    except Exception as e:
        # Handle data gathering errors gracefully
        if verbose:
            print(f"Error gathering results: {e}")
        return {
            "completion_status": completion_result,
            "consolidated_df": None,
            "results_saved_to": None,
            "summary": None,
            "message": f"Error gathering results: {e}",
        }

    # ============================================================================
    # STEP 4: DATA PROCESSING AND TRANSFORMATION
    # ============================================================================
    # Parse array string columns and flatten prediction step data
    if verbose:
        print("\n" + "-" * 50)
        print("############################################")
        print("### STEP 3: Processing array columns... ###")
        print("############################################")

    try:
        # Define expected array columns that may need parsing
        list_columns = ["nmse_mean", "nmse_std", "se_mean", "se_std", "se0_mean", "se0_std"]

        # Check which array columns actually exist in the consolidated DataFrame
        existing_list_columns = [col for col in list_columns if col in consolidated_df.columns]

        if existing_list_columns:
            if verbose:
                print(f"Parsing array columns: {existing_list_columns}")
            # Transform array strings into individual rows per prediction step
            consolidated_df = parse_array_string(df=consolidated_df, list_columns=existing_list_columns)
        else:
            if verbose:
                print("No array columns found to parse")

    except Exception as e:
        # Array parsing is not critical - continue without it if it fails
        if verbose:
            print(f"Warning: Error parsing array columns: {e}")
            print("Proceeding without array parsing...")

    # ============================================================================
    # STEP 5: RESULT PERSISTENCE
    # ============================================================================
    # Save consolidated results to disk with proper timestamping
    results_saved_to = None
    if save_results:
        if verbose:
            print("\n" + "-" * 50)
            print("##############################################")
            print("### STEP 4: Saving consolidated results... ###")
            print("##############################################")

        try:
            # Create timestamped output directory for consolidated results
            output_dir = Path(DIR_OUTPUTS) / "testing" / "results" / "gather" / get_current_time()
            results_saved_to = save_consolidated_results(consolidated_df, output_dir=output_dir)
            if verbose:
                print(f"Results saved to: {results_saved_to}")
        except Exception as e:
            # Saving errors are not critical - continue with in-memory results
            if verbose:
                print(f"Error saving results: {e}")

    # ============================================================================
    # STEP 6: COMPREHENSIVE ANALYSIS EXECUTION
    # ============================================================================
    # Run detailed performance analysis on the consolidated results
    if verbose:
        print("\n" + "-" * 50)
        print("###########################################################")
        print("### STEP 5: Running analysis on consolidated results... ###")
        print("###########################################################")

    try:
        # Run comprehensive ranking and statistical analysis if results were saved
        if results_saved_to:
            # Initialize analyzer with the consolidated results
            analyzer = CSIResultsAnalyzer(results_path=results_saved_to)
            # Perform analysis for both NMSE and SE metrics
            analyzer.run_analysis(metric_types=["nmse", "se"], save_results=save_results)
            if verbose:
                print("Analysis completed successfully!")
        else:
            if verbose:
                print("Skipping analysis - no results file saved")
    except Exception as e:
        # Analysis errors are not critical - results are still available
        if verbose:
            print(f"Error running analysis: {e}")

    # ============================================================================
    # STEP 7: SUMMARY GENERATION AND REPORTING
    # ============================================================================
    # Generate comprehensive data summary and statistics
    if verbose:
        print("\n" + "-" * 50)
        print("##########################################")
        print("### STEP 6: Generating data summary... ###")
        print("##########################################")

    try:
        # Generate detailed summary statistics about the consolidated data
        summary = get_data_summary(consolidated_df)

        if verbose:
            print("\n" + "=" * 50)
            print("DATA SUMMARY")
            print("=" * 50)
            # Display all summary statistics
            for key, value in summary.items():
                print(f"{key}: {value}")

            print(f"\nConsolidated DataFrame shape: {consolidated_df.shape}")
            print("\nFirst few rows:")
            print(consolidated_df.head())

    except Exception as e:
        # Summary generation errors are not critical
        if verbose:
            print(f"Error generating summary: {e}")
        summary = None

    # Print workflow completion message
    if verbose:
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)

    # Return comprehensive results dictionary
    return {
        "completion_status": completion_result,
        "consolidated_df": consolidated_df,
        "results_saved_to": results_saved_to,
        "summary": summary,
        "message": "Success - all testing complete and results gathered",
    }


if __name__ == "__main__":
    # ============================================================================
    # COMMAND LINE INTERFACE
    # ============================================================================
    # Parse command line arguments with default settings
    verbose = True  # Default to detailed output
    save_results = True  # Default to saving results

    # Simple argument parsing for common options
    if len(sys.argv) > 1:
        if "--quiet" in sys.argv:
            verbose = False  # Suppress detailed output
        if "--no-save" in sys.argv:
            save_results = False  # Skip saving results to disk

    # ============================================================================
    # WORKFLOW EXECUTION
    # ============================================================================
    # Execute the main workflow with error handling
    try:
        result = main(verbose=verbose, save_results=save_results)

        # Report final status based on results
        if result["consolidated_df"] is not None:
            print(f"\nSUCCESS: Processed {len(result['consolidated_df'])} records")
            if result["results_saved_to"]:
                print(f"Results saved to: {result['results_saved_to']}")
        else:
            # Testing was incomplete or other issue occurred
            print(f"\nSTATUS: {result['message']}")

    except Exception as e:
        # Handle any unrecoverable errors
        print(f"\nERROR: {e}")
        sys.exit(1)
