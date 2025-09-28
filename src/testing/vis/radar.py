"""Radar plot visualization for combined SE and NMSE analysis results.

This module creates comprehensive radar plots that combine multiple performance
dimensions into a single visualization, allowing for holistic model comparison
across computational efficiency and prediction accuracy metrics.

Key Features:
    - Multi-dimensional performance visualization (SE, NMSE, computational metrics)
    - Automatic data scaling and normalization for fair comparison
    - Smart legend positioning and color coding
    - Support for both FDD and TDD scenarios
    - Offset capability to avoid center clustering

Radar Plot Dimensions:
    Performance Metrics (rank-based):
        - SE Generalization: Out-of-distribution spectral efficiency performance
        - NMSE Generalization: Out-of-distribution NMSE performance
        - SE Regular: In-distribution spectral efficiency performance
        - NMSE Regular: In-distribution NMSE performance
        - SE Robustness: Performance under noise conditions
        - NMSE Robustness: NMSE under noise conditions

    Computational Metrics (efficiency-based):
        - FLOPs: Computational complexity (lower is better)
        - Inference Speed: Processing time (lower is better)
        - Total Params: Model size (lower is better)

The radar visualization enables identification of models that achieve optimal
trade-offs between prediction accuracy and computational efficiency.
"""

from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.dirs import DIR_OUTPUTS
from src.utils.vis_utils import get_display_name, plt_styles, set_plot_style, vis_config


# =============================================================================
# MAIN RADAR PLOT CREATION FUNCTIONS
# =============================================================================


def create_combined_radar_plot(
    dir_computational_overhead: Path,
    dir_se_analysis: Path,
    dir_nmse_analysis: Path,
    output_dir: Path | None = None,
    figsize: tuple = vis_config["figsize_single"],
    is_offset: bool = True,
) -> None:
    """Create comprehensive radar plots combining performance and computational metrics.

    This function generates radar plots that provide a holistic view of model performance
    by combining accuracy metrics (SE, NMSE) across different scenarios with computational
    efficiency metrics (FLOPs, inference time, parameters). This enables identification
    of models with optimal accuracy-efficiency trade-offs.

    Data Integration Process:
        1. Load computational overhead data (FLOPs, inference time, parameters)
        2. Load SE rank distributions across scenarios (regular, robustness, generalization)
        3. Load NMSE rank distributions across scenarios
        4. Normalize and scale all metrics for fair comparison
        5. Generate radar plots for FDD and TDD scenarios separately

    Scaling Strategy:
        - Performance metrics: Convert ranks to scores (higher = better)
        - Computational metrics: Convert to improvement percentages (higher = more efficient)
        - Apply offset to prevent center clustering for better visualization

    Parameters
    ----------
    dir_computational_overhead : Path
        Directory containing computational_overhead.csv with model efficiency metrics.
    dir_se_analysis : Path
        Directory containing SE rank distribution CSV files from analysis pipeline.
    dir_nmse_analysis : Path
        Directory containing NMSE rank distribution CSV files from analysis pipeline.
    output_dir : Path, optional
        Directory to save radar plots. If None, uses default output structure.
    figsize : tuple, default=vis_config["figsize_single"]
        Figure size (width, height) for the radar plots.
    is_offset : bool, default=True
        Whether to apply offset to radar values to avoid center clustering,
        improving visualization clarity.

    Notes
    -----
    The function creates separate radar plots for FDD and TDD scenarios since
    these represent different communication modes with potentially different
    performance characteristics. Models are compared across 9 dimensions:
    - 6 performance dimensions (SE/NMSE x regular/robustness/generalization)
    - 3 computational dimensions (FLOPs/inference time/parameters)

    Output files are saved as PDF with high resolution (300 DPI) for publication use.

    """
    # =================================================================
    # DATA LOADING AND PREPARATION
    # =================================================================

    # Load computational efficiency metrics
    overhead_path = dir_computational_overhead / "computational_overhead.csv"
    overhead_df = pd.read_csv(overhead_path)

    # Load spectral efficiency performance rankings
    se_rank_path = dir_se_analysis / "rank_distributions.csv"
    se_rank_df = pd.read_csv(se_rank_path)
    se_rank_df["metric_type"] = "SE"

    # Load NMSE performance rankings
    nmse_rank_path = dir_nmse_analysis / "rank_distributions.csv"
    nmse_rank_df = pd.read_csv(nmse_rank_path)
    nmse_rank_df["metric_type"] = "NMSE"

    # Set up output directory structure
    if output_dir is None:
        output_dir = Path(DIR_OUTPUTS) / "testing" / "vis" / "radar"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify models present in both SE and NMSE data
    se_models = set(se_rank_df["model"].unique())
    nmse_models = set(nmse_rank_df["model"].unique())
    models = sorted(list(se_models & nmse_models))

    # =================================================================
    # SCENARIO-SPECIFIC RADAR PLOT GENERATION
    # =================================================================

    # Generate separate radar plots for each communication scenario
    scenarios = ["FDD", "TDD"]

    for scenario in scenarios:
        # Extract scenario-specific data from all sources
        scenario_se_df = se_rank_df[se_rank_df["scenario"] == scenario].copy()
        scenario_nmse_df = nmse_rank_df[nmse_rank_df["scenario"] == scenario].copy()
        scenario_overhead_df = overhead_df[overhead_df["Scenario"] == scenario].copy()

        # =============================================================
        # MODEL-SPECIFIC METRIC EXTRACTION AND PROCESSING
        # =============================================================

        # Process each model's performance across all dimensions
        radar_data = {}

        for model in models:
            model_overhead = scenario_overhead_df[scenario_overhead_df["Model"] == model]

            if len(model_overhead) == 0:
                continue

            model_data = {}

            # Get computational efficiency metrics
            gflops = model_overhead["GFLOPS"].iloc[0]  # type: ignore
            inference_time = model_overhead["Inference_Time_Avg_ms"].iloc[0]  # type: ignore
            total_params = model_overhead["Total_Params"].iloc[0]  # type: ignore

            # Extract SE performance across scenario categories
            se_model_data = scenario_se_df[scenario_se_df["model"] == model]
            se_regular = se_model_data[se_model_data["scenario_category"] == "regular"]
            se_robust = se_model_data[se_model_data["scenario_category"] == "robustness"]
            se_gen = se_model_data[se_model_data["scenario_category"] == "generalization"]

            # Extract NMSE performance across scenario categories
            nmse_model_data = scenario_nmse_df[scenario_nmse_df["model"] == model]
            nmse_regular = nmse_model_data[nmse_model_data["scenario_category"] == "regular"]
            nmse_robust = nmse_model_data[nmse_model_data["scenario_category"] == "robustness"]
            nmse_gen = nmse_model_data[nmse_model_data["scenario_category"] == "generalization"]

            # Store raw computational efficiency values
            model_data["gflops"] = gflops
            model_data["inference_time"] = inference_time
            model_data["total_params"] = total_params

            # Store SE performance rankings (lower rank = better performance)
            model_data["se_regular_mean_rank"] = se_regular["mean_rank"].iloc[0]  # type: ignore
            model_data["se_robust_mean_rank"] = se_robust["mean_rank"].iloc[0]  # type: ignore
            model_data["se_gen_mean_rank"] = se_gen["mean_rank"].iloc[0]  # type: ignore

            # Store NMSE performance rankings (lower rank = better performance)
            model_data["nmse_regular_mean_rank"] = nmse_regular["mean_rank"].iloc[0]  # type: ignore
            model_data["nmse_robust_mean_rank"] = nmse_robust["mean_rank"].iloc[0]  # type: ignore
            model_data["nmse_gen_mean_rank"] = nmse_gen["mean_rank"].iloc[0]  # type: ignore

            radar_data[model] = model_data

        # Filter out models with incomplete data across all metrics
        radar_data = {k: v for k, v in radar_data.items() if not any(pd.isna(val) for val in v.values())}

        if not radar_data:
            print(f"No valid data for scenario {scenario}, skipping...")
            continue

        # =============================================================
        # METRIC NORMALIZATION AND SCALING
        # =============================================================

        # Convert computational metrics to improvement percentages
        gflops_values = [data["gflops"] for data in radar_data.values()]
        inference_time_values = [data["inference_time"] for data in radar_data.values()]
        total_params_values = [data["total_params"] for data in radar_data.values()]

        max_gflops = max(gflops_values)
        max_inference_time = max(inference_time_values)
        max_total_params = max(total_params_values)

        # Transform all metrics to radar plot scale (higher values = better)
        radar_plot_data = {}
        n_models = len(radar_data) + 1

        for model in radar_data:
            data = radar_data[model]

            # Efficiency metrics: scale improvement percentages to radar range
            gflops_improvement = (max_gflops - data["gflops"]) / max_gflops * (n_models - 1)
            inference_improvement = (max_inference_time - data["inference_time"]) / max_inference_time * (n_models - 1)
            total_params_improvement = (max_total_params - data["total_params"]) / max_total_params * (n_models - 1)

            # Performance metrics: convert ranks to scores (lower rank → higher score)
            se_regular_score = n_models - data["se_regular_mean_rank"]
            se_robust_score = n_models - data["se_robust_mean_rank"]
            se_gen_score = n_models - data["se_gen_mean_rank"]

            nmse_regular_score = n_models - data["nmse_regular_mean_rank"]
            nmse_robust_score = n_models - data["nmse_robust_mean_rank"]
            nmse_gen_score = n_models - data["nmse_gen_mean_rank"]

            radar_plot_data[model] = {
                "GFLOPS": gflops_improvement,
                "Inference_Time": inference_improvement,
                "Total_Params": total_params_improvement,
                "SE_Generalization": se_gen_score,
                "NMSE_Generalization": nmse_gen_score,
                "SE_Regular": se_regular_score,
                "NMSE_Regular": nmse_regular_score,
                "SE_Robustness": se_robust_score,
                "NMSE_Robustness": nmse_robust_score,
            }

        # =============================================================
        # RADAR PLOT GENERATION
        # =============================================================

        # Generate the actual radar visualization
        _create_combined_radar_plot(
            radar_plot_data,
            scenario,
            output_dir,
            figsize,
            is_offset=is_offset,
        )


def _color_category_labels(ax, categories: list, value_keys: list) -> None:
    """Color the category labels based on whether they represent rank or percentage metrics."""
    # Define which metrics are rank-based vs percentage-based
    rank_metrics = {
        "SE_Generalization",
        "NMSE_Generalization",
        "SE_Regular",
        "NMSE_Regular",
        "SE_Robustness",
        "NMSE_Robustness",
    }

    # Colors for different metric types
    rank_color = "#1f77b4"  # Blue for rank-based metrics
    percentage_color = "#ff7f0e"  # Orange for percentage-based metrics

    # Get the x-axis labels (category labels)
    labels = ax.get_xticklabels()

    for label, value_key in zip(labels, value_keys, strict=False):
        if value_key in rank_metrics:
            label.set_color(rank_color)
            label.set_fontweight("bold")
        else:
            label.set_color(percentage_color)
            label.set_fontweight("bold")


def _create_combined_radar_plot(
    radar_plot_data: dict,
    scenario: str,
    output_dir: Path,
    figsize: tuple,
    is_offset: bool = True,
) -> None:
    """Create a combined radar plot for SE and NMSE metrics with specific axis order."""
    # Set up the radar plot with specific axis order
    categories = []
    value_keys = []

    # Performance metrics in the requested order
    categories.extend(
        [
            "Generalization\nSE",
            "Generalization\nNMSE",
            "Regular\nSE",
            "Regular\nNMSE",
            "Robustness\nSE",
            "Robustness\nNMSE",
        ]
    )
    value_keys.extend(
        ["SE_Generalization", "NMSE_Generalization", "SE_Regular", "NMSE_Regular", "SE_Robustness", "NMSE_Robustness"]
    )

    # Computational metrics
    categories.extend(["FLOPs", "Inference\nSpeed", "Total\nParams"])
    value_keys.extend(["GFLOPS", "Inference_Time", "Total_Params"])

    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Create the plot
    plt.clf()
    set_plot_style()
    _, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    # Calculate offset to avoid center clustering (if enabled)
    all_values = []
    for data in radar_plot_data.values():
        all_values.extend(data.values())

    offset = 1.0 if is_offset else 0.0

    # Plot data for each model
    for model, data in radar_plot_data.items():
        values = [data[key] + offset for key in value_keys]
        values += values[:1]  # Complete the circle

        # Get model color
        if model in plt_styles:
            color = plt_styles[model]["color"]
            linestyle = plt_styles[model]["linestyle"]
            marker = plt_styles[model]["marker"]
        else:
            color = [0.5, 0.5, 0.5]  # Default gray
            linestyle = "-"
            marker = "o"

        # Plot the model
        ax.plot(
            angles,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=get_display_name(model),
            marker=marker,
            markersize=6,
        )
        ax.fill(angles, values, color=color, alpha=0.2)

    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)  # Show axis labels

    # Move labels outside the plot area to prevent overlap
    ax.tick_params(axis="x", pad=25)

    # Set label distance from center to push them further out
    ax.set_rlabel_position(0)  # type: ignore

    # Set axis limits and ticks
    if all_values:
        max_value = max(all_values)
        ax.set_ylim(0, max_value + offset + 0.1)

        if is_offset:
            # Set custom radial ticks to show the original scale when offset is used
            tick_positions = [offset + i for i in range(int(max_value) + 2)]
            tick_labels = [str(i) for i in range(int(max_value) + 2)]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)

        # Remove radial tick labels
        ax.set_yticklabels([])
    else:
        ax.set_ylim(0, 2)
        ax.set_yticklabels([])

    # Add grid lines
    ax.grid(True)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Adjust layout to accommodate labels outside the plot
    plt.tight_layout(pad=3.0)

    # Save plot
    filename = f"combined_radar_{scenario.lower()}.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined radar plot: {save_path}")


def plot_radar(
    dir_computational_overhead: Path,
    dir_se_analysis: Path,
    dir_nmse_analysis: Path,
    output_dir: Path | None = None,
    is_offset: bool = True,
) -> None:
    """Create comprehensive radar plots combining SE, NMSE, and computational metrics.

    This is the main entry point for radar plot generation, orchestrating the
    creation of multi-dimensional performance visualizations that enable holistic
    model comparison across accuracy and efficiency dimensions.

    The function serves as a high-level interface that calls create_combined_radar_plot()
    with the provided parameters, handling default directory resolution and ensuring
    consistent radar plot generation across the analysis pipeline.

    Parameters
    ----------
    dir_computational_overhead : Path
        Directory containing computational_overhead.csv with model efficiency data.
    dir_se_analysis : Path
        Directory containing SE analysis results and rank distributions.
    dir_nmse_analysis : Path
        Directory containing NMSE analysis results and rank distributions.
    output_dir : Path, optional
        Output directory for radar plots. If None, uses default structure.
    is_offset : bool, default=True
        Whether to apply offset to radar values to prevent center clustering.

    Notes
    -----
    This function creates radar plots for both FDD and TDD scenarios, providing
    a comprehensive view of model performance across multiple dimensions:
    - Performance accuracy (SE/NMSE across different test conditions)
    - Computational efficiency (FLOPs, inference time, model parameters)

    The resulting visualizations enable identification of models with optimal
    trade-offs between prediction accuracy and computational efficiency.

    """
    create_combined_radar_plot(
        dir_computational_overhead=dir_computational_overhead,
        dir_se_analysis=dir_se_analysis,
        dir_nmse_analysis=dir_nmse_analysis,
        output_dir=output_dir,
        is_offset=is_offset,
    )
