"""Radar plot visualization for combined SE and NMSE analysis results."""

from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.dirs import DIR_OUTPUTS
from src.utils.vis_utils import get_display_name, plt_styles, set_plot_style, vis_config


def create_combined_radar_plot(
    dir_computational_overhead: Path,
    dir_se_analysis: Path,
    dir_nmse_analysis: Path,
    output_dir: Path | None = None,
    figsize: tuple = vis_config["figsize_single"],
    is_offset: bool = True,
) -> None:
    """Create combined radar plots comparing model performance across SE, NMSE, and computational metrics.

    Args:
        dir_computational_overhead: Directory containing computational_overhead.csv
        dir_se_analysis: Directory containing SE rank distribution CSV files
        dir_nmse_analysis: Directory containing NMSE rank distribution CSV files
        output_dir: Directory to save plots. If None, saves in se_nmse_radar directory
        figsize: Figure size for the plots
        is_offset: Whether to apply offset to avoid center clustering (default: True)

    """
    # Load computational overhead data
    overhead_path = dir_computational_overhead / "computational_overhead.csv"
    overhead_df = pd.read_csv(overhead_path)

    # Load SE rank distribution data (find latest timestamp folder)
    se_rank_path = dir_se_analysis / "rank_distributions.csv"
    se_rank_df = pd.read_csv(se_rank_path)
    se_rank_df["metric_type"] = "SE"

    # Load NMSE rank distribution data (find latest timestamp folder)
    nmse_rank_path = dir_nmse_analysis / "rank_distributions.csv"
    nmse_rank_df = pd.read_csv(nmse_rank_path)
    nmse_rank_df["metric_type"] = "NMSE"

    # Create output directory
    if output_dir is None:
        output_dir = Path(DIR_OUTPUTS) / "testing" / "vis" / "radar"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique models (exclude NP model)
    se_models = set(se_rank_df["model"].unique())
    nmse_models = set(nmse_rank_df["model"].unique())
    models = sorted([model for model in (se_models & nmse_models) if model != "NP"])

    # Process data for each scenario (FDD and TDD)
    scenarios = ["FDD", "TDD"]

    for scenario in scenarios:
        # Filter data for this scenario
        scenario_se_df = se_rank_df[se_rank_df["scenario"] == scenario].copy()
        scenario_nmse_df = nmse_rank_df[nmse_rank_df["scenario"] == scenario].copy()
        scenario_overhead_df = overhead_df[overhead_df["Scenario"] == scenario].copy()

        # Calculate metrics for each model
        radar_data = {}

        for model in models:
            model_overhead = scenario_overhead_df[scenario_overhead_df["Model"] == model]

            if len(model_overhead) == 0:
                continue

            model_data = {}

            # Extract computational metrics
            gflops = model_overhead["GFLOPS"].iloc[0]  # type: ignore
            inference_time = model_overhead["Inference_Time_Avg_ms"].iloc[0]  # type: ignore
            total_params = model_overhead["Total_Params"].iloc[0]  # type: ignore

            # Get SE performance metrics
            se_model_data = scenario_se_df[scenario_se_df["model"] == model]
            se_regular = se_model_data[se_model_data["scenario_category"] == "regular"]
            se_robust = se_model_data[se_model_data["scenario_category"] == "robustness"]
            se_gen = se_model_data[se_model_data["scenario_category"] == "generalization"]

            # Get NMSE performance metrics
            nmse_model_data = scenario_nmse_df[scenario_nmse_df["model"] == model]
            nmse_regular = nmse_model_data[nmse_model_data["scenario_category"] == "regular"]
            nmse_robust = nmse_model_data[nmse_model_data["scenario_category"] == "robustness"]
            nmse_gen = nmse_model_data[nmse_model_data["scenario_category"] == "generalization"]

            # Store computational metrics
            model_data["gflops"] = gflops
            model_data["inference_time"] = inference_time
            model_data["total_params"] = total_params

            # Store SE mean ranks
            model_data["se_regular_mean_rank"] = se_regular["mean_rank"].iloc[0]  # type: ignore
            model_data["se_robust_mean_rank"] = se_robust["mean_rank"].iloc[0]  # type: ignore
            model_data["se_gen_mean_rank"] = se_gen["mean_rank"].iloc[0]  # type: ignore

            # Store NMSE mean ranks
            model_data["nmse_regular_mean_rank"] = nmse_regular["mean_rank"].iloc[0]  # type: ignore
            model_data["nmse_robust_mean_rank"] = nmse_robust["mean_rank"].iloc[0]  # type: ignore
            model_data["nmse_gen_mean_rank"] = nmse_gen["mean_rank"].iloc[0]  # type: ignore

            radar_data[model] = model_data

        # Remove models with missing data
        radar_data = {k: v for k, v in radar_data.items() if not any(pd.isna(val) for val in v.values())}

        if not radar_data:
            print(f"No valid data for scenario {scenario}, skipping...")
            continue

        # Calculate percentage improvements for computational metrics
        gflops_values = [data["gflops"] for data in radar_data.values()]
        inference_time_values = [data["inference_time"] for data in radar_data.values()]
        total_params_values = [data["total_params"] for data in radar_data.values()]

        max_gflops = max(gflops_values)
        max_inference_time = max(inference_time_values)
        max_total_params = max(total_params_values)

        # Create final data for radar plot
        radar_plot_data = {}
        n_models = len(radar_data) + 1

        for model in radar_data:
            data = radar_data[model]

            # Computational metrics: percentage improvement scaled by (num_models - 1)
            gflops_improvement = (max_gflops - data["gflops"]) / max_gflops * (n_models - 1)
            inference_improvement = (max_inference_time - data["inference_time"]) / max_inference_time * (n_models - 1)
            total_params_improvement = (max_total_params - data["total_params"]) / max_total_params * (n_models - 1)

            # Performance metrics: use num_models - mean_rank (smaller rank = better = higher value)
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

        # Create radar plot
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
    """Create combined SE and NMSE radar plots.

    Args:
        dir_computational_overhead: Directory containing computational_overhead.csv
        dir_se_analysis: Directory containing SE rank distribution CSV files
        dir_nmse_analysis: Directory containing NMSE rank distribution CSV files
        output_dir: Directory to save plots. If None, saves in se_nmse_radar directory
        is_offset: Whether to apply offset to avoid center clustering (default: True)

    """
    create_combined_radar_plot(
        dir_computational_overhead=dir_computational_overhead,
        dir_se_analysis=dir_se_analysis,
        dir_nmse_analysis=dir_nmse_analysis,
        output_dir=output_dir,
        is_offset=is_offset,
    )
