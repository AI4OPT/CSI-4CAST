"""Main visualization orchestrator for model analysis.

This module serves as the central entry point for generating comprehensive
visualization outputs from experimental results. It coordinates
the execution of all visualization types (line plots, radar plots, violin
plots, and performance tables) to provide a complete analytical overview.

Visualization Pipeline:
    1. Line Plots: Performance trends across variables (regular, robustness, generalization)
    2. Radar Plots: Multi-dimensional model comparison (accuracy + efficiency)
    3. Violin Plots: Rank distribution analysis showing performance consistency
    4. Performance Tables: Detailed numerical comparisons for delay spread and channel model

Key Features:
    - Automatic input discovery from latest analysis results
    - Organized output structure with timestamped directories
    - Comprehensive coverage of all analysis dimensions
    - Consistent styling and formatting across all visualization types
    - High-resolution outputs suitable for publication

The module integrates results from:
    - Consolidated experimental results (CSV)
    - SE and NMSE analysis pipelines
    - Computational overhead measurements

Output Organization:
- z_artifacts/outputs/testing/vis/[date_time]/line
- z_artifacts/outputs/testing/vis/[date_time]/radar
- z_artifacts/outputs/testing/vis/[date_time]/violin
- z_artifacts/outputs/testing/vis/[date_time]/table

Usage:
    python3 -m src.testing.vis.main
"""

from pathlib import Path

# Import visualization modules
from src.testing.vis.line import plot_line
from src.testing.vis.radar import plot_radar
from src.testing.vis.table import plot_table
from src.testing.vis.violin import plot_violin
from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_current_time, get_latest_datetime_folder


# =============================================================================
# MAIN VISUALIZATION ORCHESTRATION FUNCTION
# =============================================================================


def vis_main(
    input_path_consolidated_results: Path | None = None,
    input_dir_analysis_nmse: Path | None = None,
    input_dir_analysis_se: Path | None = None,
    input_dir_computational_overhead: Path | None = None,
):
    """Visualization pipeline orchestrator for model analysis.

    This function coordinates the generation of all visualization types from
    experimental results, creating a comprehensive analytical output
    that covers performance trends, multi-dimensional comparisons, ranking
    distributions, and detailed numerical tables.

    The function automatically discovers the latest analysis results if no
    specific inputs are provided, ensuring seamless integration with the
    analysis pipeline. All outputs are organized in a timestamped directory
    structure for easy identification and comparison.

    Parameters
    ----------
    input_path_consolidated_results : Path, optional
        Path to consolidated_results.csv. If None, uses latest from gather directory.
    input_dir_analysis_nmse : Path, optional
        Directory containing NMSE analysis results. If None, uses latest from analysis/nmse.
    input_dir_analysis_se : Path, optional
        Directory containing SE analysis results. If None, uses latest from analysis/se.
    input_dir_computational_overhead : Path, optional
        Directory containing computational overhead data. If None, uses latest available.

    Generated Visualizations
    -----------------------
    Line Plots:
        - Regular performance under standard conditions
        - Robustness analysis under noise conditions
        - Generalization performance beyond training ranges

    Radar Plots:
        - Multi-dimensional model comparison
        - Combined accuracy and efficiency metrics
        - FDD and TDD scenario variants

    Violin Plots:
        - Model ranking distribution analysis
        - Scenario-specific ranking patterns

    Performance Tables:
        - Detailed numerical comparisons
        - Channel model and delay spread analysis
        - Regular vs generalization breakdowns

    Output Structure
    ---------------
    DIR_OUTPUTS/testing/vis/{timestamp}/
    ├── line/           # Performance trend plots
    ├── radar/          # Multi-dimensional comparisons
    ├── violin/         # Ranking distribution plots
    └── table/          # Numerical performance tables

    Notes
    -----
    This function provides a one-stop solution for generating publication-ready
    visualizations from experimental data. All outputs use consistent
    styling and high-resolution formats suitable for research papers and presentations.

    The automatic input discovery ensures the function works seamlessly with the
    standard analysis pipeline workflow.

    """
    # =================================================================
    # DIRECTORY STRUCTURE AND INPUT DISCOVERY
    # =================================================================

    # Set up standard directory structure
    dir_testing = Path(DIR_OUTPUTS) / "testing"
    dir_computational_overhead = dir_testing / "computational_overhead"
    dir_results = dir_testing / "results"
    dir_analysis = dir_results / "analysis"
    dir_gather = dir_results / "gather"

    # Create timestamped output directory for this visualization run
    dir_vis = dir_testing / "vis" / get_current_time()
    dir_vis.mkdir(parents=True, exist_ok=True)

    # Discover consolidated results file (latest if not specified)
    if input_path_consolidated_results is None:
        latest_gather_dir = get_latest_datetime_folder(dir_gather)
        if latest_gather_dir is None:
            raise ValueError("No gather directory found - run analysis pipeline first")
        path_consolidated_results = latest_gather_dir / "consolidated_results.csv"
        if not path_consolidated_results.exists():
            raise FileNotFoundError(f"Consolidated results file not found: {path_consolidated_results}")
    else:
        path_consolidated_results = input_path_consolidated_results

    # Discover NMSE analysis directory (latest if not specified)
    if input_dir_analysis_nmse is None:
        latest_analysis_nmse_dir = get_latest_datetime_folder(dir_analysis / "nmse")
        if latest_analysis_nmse_dir is None:
            raise ValueError("No NMSE analysis directory found - run NMSE analysis first")
        dir_analysis_nmse = latest_analysis_nmse_dir
    else:
        dir_analysis_nmse = input_dir_analysis_nmse

    # Discover SE analysis directory (latest if not specified)
    if input_dir_analysis_se is None:
        latest_analysis_se_dir = get_latest_datetime_folder(dir_analysis / "se")
        if latest_analysis_se_dir is None:
            raise ValueError("No SE analysis directory found - run SE analysis first")
        dir_analysis_se = latest_analysis_se_dir
    else:
        dir_analysis_se = input_dir_analysis_se

    # Discover computational overhead directory (latest if not specified)
    if input_dir_computational_overhead is None:
        latest_computational_overhead_dir = get_latest_datetime_folder(dir_computational_overhead)
        if latest_computational_overhead_dir is None:
            raise ValueError("No computational overhead directory found - run overhead analysis first")
        dir_computational_overhead_final = latest_computational_overhead_dir
    else:
        dir_computational_overhead_final = input_dir_computational_overhead

    # =================================================================
    # LINE PLOT GENERATION
    # =================================================================

    # Create line plots showing performance trends across variables
    dir_line = dir_vis / "line"
    dir_line.mkdir(parents=True, exist_ok=True)

    print("" + "=" * 60)
    print("GENERATING LINE PLOTS - Performance Trends Analysis")
    print("=" * 60)
    print(f"Input: {path_consolidated_results}")
    print(f"Output: {dir_line}")
    print("Generating: Regular, Robustness, and Generalization line plots...")
    plot_line(path_consolidated_results, dir_line)
    print("Line plots completed successfully!\n")

    # =================================================================
    # RADAR PLOT GENERATION
    # =================================================================

    # Create radar plots for multi-dimensional model comparison
    dir_radar = dir_vis / "radar"
    dir_radar.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING RADAR PLOTS - Multi-Dimensional Model Comparison")
    print("=" * 60)
    print(f"Computational data: {dir_computational_overhead_final}")
    print(f"SE analysis: {dir_analysis_se}")
    print(f"NMSE analysis: {dir_analysis_nmse}")
    print(f"Output: {dir_radar}")
    print("Generating: Combined accuracy and efficiency radar plots...")
    plot_radar(dir_computational_overhead_final, dir_analysis_se, dir_analysis_nmse, dir_radar)
    print("Radar plots completed successfully!\n")

    # =================================================================
    # VIOLIN PLOT GENERATION
    # =================================================================

    # Create violin plots showing model ranking distributions
    dir_violin = dir_vis / "violin"
    dir_violin.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING VIOLIN PLOTS - Ranking Distribution Analysis")
    print("=" * 60)
    print(f"SE analysis: {dir_analysis_se}")
    print(f"NMSE analysis: {dir_analysis_nmse}")
    print(f"Output: {dir_violin}")
    print("Generating: Model ranking distribution violin plots...")
    plot_violin(dir_analysis_se, dir_analysis_nmse, dir_violin)
    print("Violin plots completed successfully!\n")

    # =================================================================
    # PERFORMANCE TABLE GENERATION
    # =================================================================

    # Create detailed performance comparison tables
    dir_table = dir_vis / "table"
    dir_table.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING PERFORMANCE TABLES - Detailed Numerical Analysis")
    print("=" * 60)
    print(f"Input: {path_consolidated_results}")
    print(f"Output: {dir_table}")
    print("Generating: SE/NMSE tables for channel models and delay spreads...")
    plot_table(path_consolidated_results, dir_table)
    print("Performance tables completed successfully!\n")

    # =================================================================
    # VISUALIZATION PIPELINE COMPLETION
    # =================================================================

    print("=" * 60)
    print("VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"All visualizations saved to: {dir_vis}")
    print("")
    print("Generated outputs:")
    print(f"  • Line plots: {dir_line}")
    print(f"  • Radar plots: {dir_radar}")
    print(f"  • Violin plots: {dir_violin}")
    print(f"  • Performance tables: {dir_table}")
    print("")
    print("Visualization pipeline ready for analysis and publication!")
    print("=" * 60)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    # Execute main visualization pipeline with automatic input discovery
    vis_main()
