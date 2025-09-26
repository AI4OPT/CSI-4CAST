"""Violin plot visualization for CSI model rank distributions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from src.utils.dirs import DIR_OUTPUTS
from src.utils.vis_utils import get_display_name, plt_styles, set_plot_style, vis_config


def create_violin_plot(
    csv_path: Path,
    output_dir: Path | None = None,
    metric_type: str | None = None,
    figsize: tuple = vis_config["figsize_single"],
) -> None:
    """Create violin plots for model rank distributions.

    Args:
        csv_path: Path to the rank distributions CSV file
        output_dir: Directory to save the plots. If None, saves in same directory as CSV
        metric_type: Type of metric ('nmse' or 'se') for subfolder organization
        figsize: Figure size for the plots

    """
    # Load the data
    df = pd.read_csv(csv_path)

    # Create output directory if not provided
    output_path = Path(csv_path).parent if output_dir is None else Path(output_dir)

    # Add metric type subfolder if specified
    if metric_type:
        output_path = output_path / metric_type

    output_path.mkdir(parents=True, exist_ok=True)

    # Get unique scenario combinations
    scenario_combinations = df[["scenario_category", "scenario"]].drop_duplicates()

    # Create plots for each scenario combination
    for _, row in scenario_combinations.iterrows():
        scenario_category = row["scenario_category"]
        scenario = row["scenario"]

        # Filter data for this scenario combination
        subset = df[(df["scenario_category"] == scenario_category) & (df["scenario"] == scenario)].copy()

        if len(subset) == 0:
            continue

        # Sort models by mean_rank (best to worst) - for SE, lower rank number is better
        models_sorted = subset.sort_values("mean_rank")["model"].tolist()  # type: ignore

        # # Calculate rank-1 percentage for each model
        # subset = subset.copy()
        # subset["rank_1_percentage"] = (subset["rank_1"] / subset["total_scenarios"]) * 100

        # # Sort models by rank-1 percentage (highest to lowest)
        # models_sorted = subset.sort_values("rank_1_percentage", ascending=False)["model"].tolist()

        # Prepare data for violin plot
        # We need to create individual rank entries for each scenario
        violin_data = []

        for _, model_row in subset.iterrows():
            model = model_row["model"]
            # Create rank entries based on rank counts
            for rank in range(1, len(models_sorted) + 1):
                rank_col = f"rank_{rank}"
                if rank_col in model_row and not pd.isna(model_row[rank_col]):  # type: ignore
                    count = int(model_row[rank_col])
                    # Add 'count' number of entries with this rank for this model
                    for _ in range(count):
                        violin_data.append({"model": model, "rank": rank})

        if not violin_data:
            print(f"No data for {scenario_category}_{scenario}, skipping...")
            continue

        violin_df = pd.DataFrame(violin_data)

        # Create violin plot with dual y-axes
        plt.clf()
        set_plot_style()
        _, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()  # Create second y-axis for line plot

        model_colors = []
        for model in models_sorted:
            # Convert RGB values to hex or use tuple
            color = plt_styles[model]["color"]
            model_colors.append(color)

        # Create violin plot with sorted model order and custom colors
        # Set cut=0 to truncate KDE at data limits (no extension beyond 1-6)
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

        # Prepare data for line plot (rank_1_pct)
        line_data_x = []
        line_data_y = []
        for model in models_sorted:
            model_subset = subset[subset["model"] == model]
            if len(model_subset) > 0:
                rank_1_pct = float(model_subset["rank_1_pct"].iloc[0])  # type: ignore
                line_data_x.append(model)
                line_data_y.append(rank_1_pct)

        # Create line plot on second y-axis with vivid styling
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

        # Align y-axes: 100% should align with rank 1 (top), 0% should align with rank 6 (bottom)
        max_rank = violin_df["rank"].max() if len(violin_df) > 0 else len(models_sorted)
        ax2.set_ylim(-20, 110)
        ax2.set_ylabel(r"$\mathbf{P}_{\mathrm{rank1}}$", fontweight="bold", color="crimson")
        ax2.set_yticks(range(0, 101, 20))  # only show 0–100
        ax2.tick_params(axis="y", labelcolor="crimson")

        # Customize plot
        ax1.set_xlabel("Model", fontweight="bold")
        ax1.set_ylabel(r"$\mathrm{RankScore}$", fontweight="bold")

        # Set x-tick labels to display names
        ax1.set_xticklabels([get_display_name(model) for model in models_sorted])
        ax1.tick_params(axis="x", rotation=45)

        # Invert y-axis so rank 1 (best) is at the top
        ax1.invert_yaxis()

        # Set y-axis ticks to integer ranks
        max_rank = violin_df["rank"].max() if len(violin_df) > 0 else len(models_sorted)
        ax1.set_ylim(max_rank + 1, 0.5)
        ax1.set_yticks(range(1, max_rank + 1))

        # Add grid for better readability
        ax1.grid(True, alpha=0.3, axis="y")

        # Create legend with model colors and line plot
        legend_elements = []
        for i, model in enumerate(models_sorted):
            color = model_colors[i]
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=get_display_name(model)))

        # Add line plot to legend
        legend_elements.append(
            Line2D([0], [0], color="crimson", marker="o", linestyle="-", linewidth=4, markersize=10, label="Rank-1 %")
        )

        # Place legend horizontally across the bottom within the figure area in two long rows
        # Calculate columns for exactly 2 rows
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

        # Remove summary statistics text as requested

        # Use tight layout since legend is now within the figure area
        plt.tight_layout()

        # Save plot
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
    """Create violin plots for both SE and NMSE analysis results.

    Args:
        dir_se_analysis: Path to SE analysis directory
        dir_nmse_analysis: Path to NMSE analysis directory
        dir_output: Output directory. If None, uses default structure

    """
    if dir_output is None:
        dir_output = Path(DIR_OUTPUTS) / "testing" / "vis" / "violin"

    print("Creating SE violin plots...")
    _plot_violin(dir_se_analysis, "se", dir_output)

    print("Creating NMSE violin plots...")
    _plot_violin(dir_nmse_analysis, "nmse", dir_output)

    print(f"All violin plots saved to: {dir_output}")
