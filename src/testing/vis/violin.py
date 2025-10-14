"""Violin plot visualization for CSI model rank distributions.

This module creates violin plots that visualize the distribution of model rankings
across different experimental scenarios. Violin plots combine the benefits of
box plots and kernel density estimation to show both summary statistics and
the full distribution shape of model performance rankings.

Key Features:
    - Rank distribution visualization with kernel density estimation
    - Dual y-axis design showing both rank distributions and rank-1 percentages
    - Automatic model sorting by performance
    - Color-coded violin plots with consistent model styling
    - Support for different scenario categories (regular, robustness, generalization)
    - Interactive legend with comprehensive model and metric information

Visualization Components:
    - Violin plots: Show full rank distribution shape for each model
    - Box plots: Display quartiles and outliers within violins
    - Line plot: Rank-1 percentage overlay on secondary y-axis
    - Legend: Model identification with performance indicators

The violin plots provide insights into:
    - Consistency of model performance (narrow vs wide distributions)
    - Performance ceiling (how often models achieve top ranks)
    - Performance floor (worst-case ranking scenarios)
    - Distribution skewness (bias toward good or poor performance)
"""

from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.dirs import DIR_OUTPUTS
from src.utils.vis_utils import get_display_name, plt_styles, set_plot_style, vis_config


# =============================================================================
# VIOLIN PLOT CREATION FUNCTIONS
# =============================================================================


def create_violin_plot(
    csv_path: Path,
    output_dir: Path | None = None,
    metric_type: str | None = None,
    figsize: tuple = vis_config["figsize_single"],
) -> None:
    """Generate violin plots showing model rank distribution patterns.

    This function creates comprehensive violin plots that visualize how model
    rankings are distributed across different experimental scenarios. Each violin
    shows the full distribution of ranks achieved by a model, providing insights
    into performance consistency and reliability.

    Plot Features:
        - Violin shapes: Kernel density estimation of rank distributions
        - Box plots: Quartile information embedded within violins
        - Dual y-axes: Rank scale (inverted) + rank-1 percentage overlay
        - Model sorting: Ordered by mean performance for easy comparison
        - Color coding: Consistent with other visualizations

    Data Processing:
        1. Load rank distribution data from CSV
        2. Expand rank counts into individual rank entries
        3. Create violin plot data structure
        4. Generate dual-axis visualization with overlays
        5. Apply consistent styling and export

    Parameters
    ----------
    csv_path : Path
        Path to rank_distributions.csv file containing model ranking data.
        Expected columns: model, scenario_category, scenario, rank_1, rank_2, ...,
        rank_N, total_scenarios, mean_rank, rank_1_pct.
    output_dir : Path, optional
        Directory for saving violin plots. If None, uses CSV file's directory.
    metric_type : str, optional
        Metric type identifier ('nmse' or 'se') for output organization.
    figsize : tuple, default=vis_config["figsize_single"]
        Figure dimensions (width, height) in inches.

    Output
    ------
    Saves violin plots as PDF files with naming pattern:
    violin_{scenario_category}_{scenario}.pdf

    Notes
    -----
    The violin plots reveal important performance characteristics:
    - Wide violins: Inconsistent performance across scenarios
    - Narrow violins: Consistent performance
    - Skewed violins: Bias toward good or poor performance
    - High rank-1 percentage: Frequent best-model performance

    The inverted y-axis (rank 1 at top) aligns with intuitive "better = higher"
    visualization convention.

    """
    # =================================================================
    # DATA LOADING AND PREPARATION
    # =================================================================

    # Load rank distribution data from CSV file
    df = pd.read_csv(csv_path)

    # Set up output directory structure
    output_path = Path(csv_path).parent if output_dir is None else Path(output_dir)

    # Organize output by metric type if specified
    if metric_type:
        output_path = output_path / metric_type

    output_path.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # SCENARIO PROCESSING LOOP
    # =================================================================

    # Process each unique scenario combination separately
    scenario_combinations = df[["scenario_category", "scenario"]].drop_duplicates()

    # Generate violin plot for each scenario combination
    for _, row in scenario_combinations.iterrows():
        scenario_category = row["scenario_category"]
        scenario = row["scenario"]

        # Extract data subset for current scenario
        subset = df[(df["scenario_category"] == scenario_category) & (df["scenario"] == scenario)].copy()

        if len(subset) == 0:
            continue

        # Order models by performance (best to worst ranking)
        models_sorted = subset.sort_values("mean_rank")["model"].tolist()  # type: ignore

        # Alternative sorting approach (commented out):
        # Could sort by rank-1 percentage instead of mean rank
        # This would prioritize models that frequently achieve best performance
        # over models with consistent but not necessarily top performance

        # =============================================================
        # VIOLIN PLOT DATA PREPARATION
        # =============================================================

        # Convert rank counts to individual rank entries for violin plotting
        violin_data = []

        # Expand rank distribution counts into individual data points
        for _, model_row in subset.iterrows():
            model = model_row["model"]
            # Convert rank counts to individual rank entries for KDE
            for rank in range(1, len(models_sorted) + 1):
                rank_col = f"rank_{rank}"
                if rank_col in model_row and not pd.isna(model_row[rank_col]):  # type: ignore
                    count = int(model_row[rank_col])
                    # Create 'count' individual rank entries for violin shape
                    for _ in range(count):
                        violin_data.append({"model": model, "rank": rank})

        if not violin_data:
            print(f"No data for {scenario_category}_{scenario}, skipping...")
            continue

        violin_df = pd.DataFrame(violin_data)

        # =============================================================
        # DUAL-AXIS VIOLIN PLOT CREATION
        # =============================================================

        # Set up primary axis for violin plots and secondary for rank-1 percentage
        plt.clf()
        set_plot_style()
        _, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()  # Create second y-axis for line plot

        model_colors = []
        for model in models_sorted:
            # Convert RGB values to hex or use tuple
            color = plt_styles[model]["color"]
            model_colors.append(color)

        # Generate violin plots with model-specific colors and sorting
        sns.violinplot(
            data=violin_df,
            x="model",
            y="rank",
            ax=ax1,
            hue="model",
            hue_order=models_sorted,
            palette=model_colors,
            order=models_sorted,
            alpha=0.5,  # Make violin more transparent
            inner="box",  # show quartile lines instead of box
            cut=0,  # Truncate KDE at data limits, no extension beyond actual rank range
        )

        # =============================================================
        # RANK-1 PERCENTAGE OVERLAY PREPARATION
        # =============================================================

        # Extract rank-1 percentage data for line plot overlay
        line_data_x = []
        line_data_y = []
        for model in models_sorted:
            model_subset = subset[subset["model"] == model]
            if len(model_subset) > 0:
                rank_1_pct = float(model_subset["rank_1_pct"].iloc[0])  # type: ignore
                line_data_x.append(model)
                line_data_y.append(rank_1_pct)

        # Plot rank-1 percentage as prominent line overlay
        line_positions = range(len(models_sorted))
        ax2.plot(
            line_positions,
            line_data_y,
            marker="o",
            linestyle="-",
            linewidth=4,
            markersize=10,
            color="crimson",
            alpha=1.0,
            label="Rank-1 %",
        )

        # Configure secondary y-axis for rank-1 percentage display
        max_rank = violin_df["rank"].max() if len(violin_df) > 0 else len(models_sorted)
        ax2.set_ylim(-20, 110)
        ax2.set_ylabel(r"$\mathbf{P}_{\mathrm{rank1}}$", fontweight="bold", color="crimson")
        ax2.set_yticks(range(0, 101, 20))  # only show 0-100
        ax2.tick_params(axis="y", labelcolor="crimson")

        # =============================================================
        # PLOT CUSTOMIZATION AND STYLING
        # =============================================================

        # Configure primary axis (violin plots)
        ax1.set_xlabel("Model", fontweight="bold")
        ax1.set_ylabel(r"$\mathrm{RankScore}$", fontweight="bold")

        # Apply model display names and rotation for readability
        ax1.set_xticklabels([get_display_name(model) for model in models_sorted])
        ax1.tick_params(axis="x", rotation=45)

        # Invert y-axis: rank 1 (best) at top, higher ranks at bottom
        ax1.invert_yaxis()

        # Configure rank axis with integer ticks and appropriate limits
        max_rank = violin_df["rank"].max() if len(violin_df) > 0 else len(models_sorted)
        ax1.set_ylim(max_rank + 1, 0.5)
        ax1.set_yticks(range(1, max_rank + 1))

        # Enable grid for easier rank reading
        ax1.grid(True, alpha=0.3, axis="y")

        # =============================================================
        # LEGEND CREATION AND LAYOUT
        # =============================================================

        # Build comprehensive legend including models and rank-1 percentage
        legend_elements = []
        for i, model in enumerate(models_sorted):
            color = model_colors[i]
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=get_display_name(model)))

        # Include rank-1 percentage line in legend
        legend_elements.append(
            Line2D([0], [0], color="crimson", marker="o", linestyle="-", linewidth=4, markersize=10, label="Rank-1 %")
        )

        # Configure legend layout: horizontal arrangement in two rows
        ncols_for_two_rows = (len(legend_elements) + 1) // 2  # Ceiling division for 2 rows

        ax1.legend(
            handles=legend_elements,
            title="Models & Metrics",
            bbox_to_anchor=(0.0, 0.02, 1.0, 0.14),  # (x, y, width, height) - taller for two long rows
            loc="center",
            ncol=ncols_for_two_rows,  # Arrange to create exactly 2 long rows
            mode="expand",  # Expand to fill the bbox
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,  # Smaller legend text
            title_fontsize=12,  # Smaller title
            markerscale=1.2,  # Smaller legend markers
            handlelength=2.0,  # Shorter legend lines
            columnspacing=1.2,  # Slightly more space between columns for readability
            handletextpad=0.5,  # Less padding between marker and text
        )

        # Note: Summary statistics removed per user preferences

        # Apply layout optimization
        plt.tight_layout()

        # =============================================================
        # OUTPUT AND CLEANUP
        # =============================================================

        # Export violin plot as high-resolution PDF
        filename = f"violin_{scenario_category}_{scenario}.pdf"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved violin plot: {save_path}")


def _plot_violin(
    dir_analysis: Path,
    metric_type: str,
    dir_output: Path | None = None,
) -> None:
    """Create violin plots for model rank distributions.

    Args:
        dir_analysis: Path to analysis directory (containing timestamp subdirectories)
        metric_type: Type of metric ('nmse' or 'se')
        dir_output: Output directory. If None, uses default structure

    """
    if dir_output is None:
        dir_output = Path(DIR_OUTPUTS) / "testing" / "vis" / "violin"

    # Get the rank distribution file
    df_rank_distribution = dir_analysis / "rank_distributions.csv"
    if not df_rank_distribution.exists():
        raise FileNotFoundError(f"Rank distributions file not found: {df_rank_distribution}")

    create_violin_plot(df_rank_distribution, dir_output, metric_type)


def plot_violin(
    dir_se_analysis: Path,
    dir_nmse_analysis: Path,
    dir_output: Path | None = None,
) -> None:
    """Create comprehensive violin plots for both SE and NMSE analysis results.

    This is the main entry point for violin plot generation, orchestrating the
    creation of rank distribution visualizations for both spectral efficiency
    and NMSE metrics. It processes analysis results from both metrics and
    generates organized violin plot outputs.

    The function creates violin plots that reveal:
        - Model ranking consistency across scenarios
        - Performance distribution patterns
        - Frequency of achieving top rankings
        - Comparative model reliability

    Parameters
    ----------
    ----------
    dir_se_analysis : Path
        Directory containing SE analysis results with rank_distributions.csv.
    dir_nmse_analysis : Path
        Directory containing NMSE analysis results with rank_distributions.csv.
    dir_output : Path, optional
        Output directory for violin plots. If None, uses default structure
        under DIR_OUTPUTS/testing/vis/violin.

    Output Structure
    ----------------
    dir_output/
    ├── se/                          # SE metric violin plots
    │   ├── violin_regular_TDD.pdf
    │   ├── violin_regular_FDD.pdf
    │   ├── violin_robustness_TDD.pdf
    │   ├── violin_robustness_FDD.pdf
    │   ├── violin_generalization_TDD.pdf
    │   └── violin_generalization_FDD.pdf
    └── nmse/                        # NMSE metric violin plots
        ├── violin_regular_TDD.pdf
        ├── violin_regular_FDD.pdf
        ├── violin_robustness_TDD.pdf
        ├── violin_robustness_FDD.pdf
        ├── violin_generalization_TDD.pdf
        └── violin_generalization_FDD.pdf

    Notes
    -----
    This function provides a comprehensive view of model ranking distributions
    across all experimental conditions, complementing the numerical tables and
    line plots with distribution-focused visualizations that reveal performance
    consistency and reliability patterns.

    The violin plots are particularly useful for:
    - Identifying models with consistent vs variable performance
    - Understanding performance ceiling and floor for each model
    - Comparing ranking distribution shapes across models
    - Assessing the reliability of model rankings

    """
    if dir_output is None:
        dir_output = Path(DIR_OUTPUTS) / "testing" / "vis" / "violin"

    print("Creating SE violin plots...")
    _plot_violin(dir_se_analysis, "se", dir_output)

    print("Creating NMSE violin plots...")
    _plot_violin(dir_nmse_analysis, "nmse", dir_output)

    print(f"All violin plots saved to: {dir_output}")
