"""Unified line plotting module for both NMSE and SE visualizations.

This module provides comprehensive line plotting functionality for model
performance analysis. It combines NMSE and SE visualization capabilities with
advanced features including:

- Broken axis support for handling large performance gaps (e.g., NP vs other models)
- Automatic legend positioning to avoid data overlap
- Generalization background highlighting for delay spread and speed analysis
- Support for multiple variable types (channel model, delay spread, speed, etc.)
- Consistent styling and formatting across all plots

Key Features:
    - Handles both regular and broken axis layouts automatically
    - Smart y-axis scaling (log for NMSE, linear for SE)
    - Background coloring for generalization vs regular ranges
    - Automatic tick formatting and positioning

Output Structure:
- z_artifacts/outputs/testing/vis/[date_time]/line/
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import pandas as pd

from src.utils.data_utils import (
    LIST_CHANNEL_MODEL,
    LIST_DELAY_SPREAD,
    LIST_MIN_SPEED_TEST,
    LIST_MIN_SPEED_TEST_GEN,
)
from src.utils.vis_utils import (
    get_display_name,
    order_models_for_scenario,
    plt_styles,
    reorder_legend_entries,
    set_plot_style,
    vis_config,
)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Mapping from variable names to their display labels for axes
VAR_NAME_TO_LABEL = {
    "cm": "Channel Model",
    "ds": "Delay Spread (ns)",
    "ms": "Speed (m/s)",
    "pred_step": "Prediction Step",
}


def aggregate_mean_std(df_group, mean_col: str, std_col: str):
    """Aggregate per-row means and standard deviations into overall statistics.

    This function properly handles the aggregation of multiple measurements where each
    row contains a mean and standard deviation. It computes the overall mean and
    the correct combined standard deviation using the formula for combining
    independent samples with known means and standard deviations.

    Mathematical Background:
        For samples with means μᵢ and standard deviations σᵢ:
        - Overall mean: μ = Σμᵢ / n
        - Overall variance: σ² = Σ(σᵢ² + (μᵢ - μ)²) / n
        - Overall std: std = √σ²

    Parameters
    ----------
    df_group : pd.DataFrame
        A grouped subset of the dataframe containing multiple rows with
        individual means and standard deviations to aggregate.
    mean_col : str
        Name of the column containing individual means.
    std_col : str
        Name of the column containing individual standard deviations.

    Returns
    -------
    pd.Series
        Series with aggregated statistics:
        - '{mean_col}_agg': Overall mean across all rows
        - '{std_col}_agg': Properly computed overall standard deviation

    Notes
    -----
    This is crucial for properly combining results from multiple experimental
    runs or conditions, ensuring statistical validity of the aggregated metrics.

    """
    means = df_group[mean_col].values
    stds = df_group[std_col].values

    overall_mean = means.mean()
    overall_var = np.mean(stds**2 + (means - overall_mean) ** 2)
    overall_std = np.sqrt(overall_var)

    return pd.Series({f"{mean_col}_agg": overall_mean, f"{std_col}_agg": overall_std})


def _log_plain_formatter(y, _pos):
    """Format logarithmic scale tick labels as plain decimal numbers.

    This formatter prevents matplotlib from using scientific notation (e.g., 1e-2)
    on log scale axes, instead displaying clean decimal numbers (e.g., 0.01).
    This improves readability of performance metrics like NMSE values.

    Parameters
    ----------
    y : float
        The tick value to format.
    _pos : int
        Tick position (unused, required by matplotlib formatter interface).

    Returns
    -------
    str
        Formatted tick label as plain decimal string, with trailing zeros
        and decimal points removed for cleaner appearance.

    Examples
    --------
    >>> _log_plain_formatter(0.01, 0)
    '0.01'
    >>> _log_plain_formatter(0.1000, 0)
    '0.1'
    >>> _log_plain_formatter(1.0, 0)
    '1'

    """
    if y == 0 or np.isnan(y):
        return "0"
    # print as plain decimal (no scientific), trim trailing zeros
    return f"{y:.10f}".rstrip("0").rstrip(".")


def get_xlabel(df: pd.DataFrame, var_1: str) -> str:
    """Generate appropriate x-axis label based on the plotted variable and data context.

    This function provides context-aware labeling, particularly for noise_degree
    variables where the label depends on the type of noise being analyzed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data being plotted.
    var_1 : str
        The variable being plotted on the x-axis (e.g., 'cm', 'ds', 'ms', 'noise_degree').

    Returns
    -------
    str
        Human-readable label for the x-axis.

    Examples
    --------
    For noise_degree with packagedrop noise: returns "Missing Probability"
    For noise_degree with phase noise: returns "SNR"
    For 'cm': returns "Channel Model"
    For 'ds': returns "Delay Spread (ns)"

    """
    if var_1 == "noise_degree":
        noise_type = df["noise_type"].unique()[0]
        return "Missing Probability" if noise_type == "packagedrop" else "SNR"
    else:
        return VAR_NAME_TO_LABEL[var_1]


def setup_axis_ticks(ax, df: pd.DataFrame, var_1: str):
    """Configure x-axis ticks and labels based on the variable type.

    This function handles specialized tick configuration for different variable types,
    ensuring appropriate spacing, formatting, and limits for each data type.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to configure.
    df : pd.DataFrame
        DataFrame containing the data being plotted.
    var_1 : str
        The variable type being plotted ('cm', 'ds', etc.).

    Notes
    -----
    - Channel models ('cm'): Uses alphabetical labels (A, B, C, D, E)
    - Delay spread ('ds'): Uses nanosecond values with proper spacing and limits
    - Other variables: Uses default matplotlib behavior

    The function automatically adjusts axis limits and tick positions to provide
    optimal visualization for each data type.

    """
    if var_1 == "cm":
        cm_labels = sorted(df["cm"].unique().tolist())
        positions = list(range(len(cm_labels)))
        ax.xaxis.set_major_locator(FixedLocator(positions))
        ax.xaxis.set_major_formatter(FixedFormatter(cm_labels))
    elif var_1 == "ds":
        ds_ticks = [0, 50e-9, 100e-9, 150e-9, 200e-9, 250e-9, 300e-9, 350e-9, 400e-9, 450e-9]
        ax.set_xticks(ds_ticks)
        ax.xaxis.set_major_locator(FixedLocator(ds_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(int(ds * 1e9)) for ds in ds_ticks]))
        ds_range = sorted(list(df["ds"].unique()))
        ax.set_xlim(0, max(ds_range) + 30e-9)  # add a little margin to the right


def get_broken_axis_outlier_model(df_agg: pd.DataFrame, gap_threshold: float = 2.0) -> str | None:
    """Return the model that should be isolated in a broken-axis plot.

    The function compares the top two model curves after aggregating each
    model by the minimum and maximum plotted value. A broken axis is used
    only when the strongest model stays sufficiently separated from the
    second one across the full line.

    Args:
        df_agg: Aggregated plotting dataframe.
        gap_threshold: Minimum ratio required between the top line and the
            second line before using a broken axis.

    Returns:
        The outlier model name if a broken axis is warranted, otherwise
        ``None``.

    """
    if gap_threshold <= 1:
        gap_threshold = 1.0

    if "model" not in df_agg.columns or "obj_mean_agg" not in df_agg.columns:
        return None

    model_stats = (
        df_agg.groupby("model")["obj_mean_agg"].agg(["min", "max"]).dropna().sort_values("max", ascending=False)
    )
    if len(model_stats) < 2:
        return None

    top_model = str(model_stats.index[0])
    top_line_min = float(model_stats.iloc[0]["min"])
    second_line_max = float(model_stats.iloc[1]["max"])

    if top_line_min <= 0:
        return None

    if second_line_max <= 0:
        ratio = np.inf
    else:
        ratio = top_line_min / second_line_max

    return top_model if ratio >= gap_threshold else None


def should_use_broken_axis(df_agg: pd.DataFrame, gap_threshold: float = 2.0) -> bool:
    """Return whether the plot should use a broken y-axis.

    Args:
        df_agg: Aggregated plotting dataframe.
        gap_threshold: Minimum separation ratio required to treat one
            model as an outlier line.

    Returns:
        ``True`` when a broken axis should be used, otherwise ``False``.

    """
    return get_broken_axis_outlier_model(df_agg, gap_threshold=gap_threshold) is not None


def find_best_legend_position(ax_top, ax_bottom):
    """Find the least obstructive legend position across two axes.

    The function approximates the legend footprint in each corner of the
    top and bottom axes and chooses the location that overlaps the fewest
    plotted points.

    Returns:
        Tuple of ``(axis, location)`` where ``axis`` is the chosen axis
        object and ``location`` is a matplotlib legend location string.

    """
    # Define possible legend locations
    locations = ["upper right", "upper left", "lower right", "lower left"]

    def get_data_density_score(ax, location):
        """Calculate how much data would be covered by legend at given location."""
        if not ax.lines and not ax.collections:  # No data plotted
            return 0

        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Define legend box regions (approximate)
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        if location == "upper right":
            box_x = (xlim[0] + 0.7 * x_range, xlim[1])
            box_y = (ylim[0] + 0.7 * y_range, ylim[1])
        elif location == "upper left":
            box_x = (xlim[0], xlim[0] + 0.3 * x_range)
            box_y = (ylim[0] + 0.7 * y_range, ylim[1])
        elif location == "lower right":
            box_x = (xlim[0] + 0.7 * x_range, xlim[1])
            box_y = (ylim[0], ylim[0] + 0.3 * y_range)
        else:  # lower left
            box_x = (xlim[0], xlim[0] + 0.3 * x_range)
            box_y = (ylim[0], ylim[0] + 0.3 * y_range)

        # Count data points in legend box region
        density_score = 0
        for line in ax.lines:
            xdata, ydata = line.get_data()
            # Count points in legend box region
            in_box = (xdata >= box_x[0]) & (xdata <= box_x[1]) & (ydata >= box_y[0]) & (ydata <= box_y[1])
            density_score += np.sum(in_box)

        return density_score

    # Test all combinations of axis and location
    best_score = float("inf")
    best_axis = ax_bottom
    best_location = "upper right"

    for ax in [ax_top, ax_bottom]:
        for location in locations:
            score = get_data_density_score(ax, location)
            if score < best_score:
                best_score = score
                best_axis = ax
                best_location = location

    return best_axis, best_location


def add_axis_break_indicators(ax_top, ax_bottom, break_size: float = 0.02):
    """Add diagonal break markers between two stacked axes."""
    # Add break lines
    d = break_size  # size of break indicator

    # Top subplot - bottom break
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    # Bottom subplot - top break
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def set_labels_ticks_and_titles(ax, df: pd.DataFrame, obj: str, var_1: str, var_2: str | None = None):
    """Set labels, ticks, legend, and grid styling for one plot.

    Args:
        ax: Target matplotlib axis.
        df: Dataframe used to infer axis labels and tick ranges.
        obj: Objective name plotted on the y-axis.
        var_1: Primary variable plotted on the x-axis.
        var_2: Optional secondary grouping variable.

    """
    # set y-label and y-ticks
    ax.set_ylabel(obj.upper(), fontweight="bold")

    # Choose scale based on objective type
    if obj.lower() == "se":
        # For SE, use linear scale since values are close together (around 0.9-1.0)
        ax.set_yscale("linear")
    else:
        # For NMSE and other metrics, use log scale
        ax.set_yscale("log")

        # Apply log scale with plain decimal formatting
        # Get the current y-axis limits to determine appropriate ticks
        y_min, y_max = ax.get_ylim()

        # Create list of desired tick values
        tick_candidates = []
        for exp in range(-3, 3):  # Cover range from 0.001 to 100
            base_val = 10**exp
            tick_candidates.extend([base_val, 2 * base_val, 4 * base_val])

        # Filter to only show ticks within our data range
        valid_ticks = [tick for tick in tick_candidates if y_min <= tick <= y_max]

        # Set exactly these ticks and no others
        ax.set_yticks(valid_ticks)
        ax.set_yticklabels([_log_plain_formatter(tick, None) for tick in valid_ticks])

        # Completely disable minor ticks
        ax.yaxis.set_minor_locator(plt.NullLocator())  # type: ignore
        ax.yaxis.set_minor_formatter(plt.NullFormatter())  # type: ignore
        ax.tick_params(which="minor", length=0)

    # set x-label using centralized function
    xlabel = get_xlabel(df, var_1)
    ax.set_xlabel(xlabel, fontweight="bold")

    # set x-ticks using centralized function
    setup_axis_ticks(ax, df, var_1)

    # set legend
    ax.legend()

    # set grid
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)


def add_generalization_background(ax, df: pd.DataFrame, var_1: str):
    """Add background colors and vertical dashed lines for regular vs generalization ranges.

    Args:
        ax: Matplotlib axis object
        df: DataFrame containing the data
        var_1: Variable being plotted on x-axis (ds, ms, etc.)

    """
    if var_1 == "ds":
        # Delay spread: 30e-9 to 300e-9 is regular (light green), 300e-9+ is generalization (light red)
        ds_values = sorted(df["ds"].unique())
        if len(ds_values) > 1:
            x_min, x_max = min(ds_values), max(ds_values)

            # Regular range: 30e-9 to 300e-9
            regular_end = 300e-9
            if x_max > regular_end:
                # Add light green background for regular range
                ax.axvspan(
                    x_min,
                    regular_end,
                    color=vis_config["color_regular_bg"],
                    alpha=vis_config["alpha_background"],
                    zorder=0,
                )
                # Add light red background for generalization range
                ax.axvspan(
                    regular_end,
                    x_max,
                    color=vis_config["color_generalization_bg"],
                    alpha=vis_config["alpha_background"],
                    zorder=0,
                )
            else:
                # All values are in regular range
                ax.axvspan(
                    x_min, x_max, color=vis_config["color_regular_bg"], alpha=vis_config["alpha_background"], zorder=0
                )

        # Add vertical dashed lines at in-distribution (regular) delay spread values
        regular_ds_values = LIST_DELAY_SPREAD  # [30e-9, 100e-9, 300e-9]
        for ds_val in regular_ds_values:
            if ds_val in ds_values:  # Only add line if this value is actually in the data
                ax.axvline(
                    x=ds_val,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    zorder=2,
                )

    elif var_1 == "ms":
        # Min speed: 1 to 30 is regular (light green), 30+ is generalization (light red)
        ms_values = sorted(df["ms"].unique())
        if len(ms_values) > 1:
            x_min, x_max = min(ms_values), max(ms_values)

            # Regular range: 1 to 30
            regular_end = 30
            if x_max > regular_end:
                # Add light green background for regular range
                ax.axvspan(
                    x_min,
                    regular_end,
                    color=vis_config["color_regular_bg"],
                    alpha=vis_config["alpha_background"],
                    zorder=0,
                )
                # Add light red background for generalization range
                ax.axvspan(
                    regular_end,
                    x_max,
                    color=vis_config["color_generalization_bg"],
                    alpha=vis_config["alpha_background"],
                    zorder=0,
                )
            else:
                # All values are in regular range
                ax.axvspan(
                    x_min, x_max, color=vis_config["color_regular_bg"], alpha=vis_config["alpha_background"], zorder=0
                )

        # Add vertical dashed lines at in-distribution (regular) min speed values
        regular_ms_values = LIST_MIN_SPEED_TEST  # [1, 10, 30]
        for ms_val in regular_ms_values:
            if ms_val in ms_values:  # Only add line if this value is actually in the data
                ax.axvline(
                    x=ms_val,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    zorder=2,
                )


# =============================================================================
# MAIN VISUALIZATION CLASS
# =============================================================================


class Vis:
    """Main visualization class for model performance analysis.

    This class provides comprehensive line plotting capabilities for analyzing
    model performance across different scenarios, metrics, and variables. It handles
    data filtering, aggregation, and visualization with support for advanced features
    like broken axes and generalization background highlighting.

    Key Capabilities:
        - Multi-variable line plots (channel model, delay spread, speed, etc.)
        - Automatic broken axis detection and layout
        - Error bar visualization with proper statistical aggregation
        - Generalization vs regular range highlighting
        - Consistent styling across all model comparisons

    Attributes
    ----------
    df : pd.DataFrame
        Copy of the input consolidated results dataframe.

    Methods
    -------
    get_filtered_data(...) : Filter data based on multiple criteria
    default_line_plot(...) : Create line plots with full feature support
    _plot_model_data(...) : Internal method for plotting individual model data

    """

    def __init__(self, df: pd.DataFrame):
        """Initialize the line-plot visualization helper.

        Parameters
        ----------
        df : pd.DataFrame
            The consolidated results dataframe.

        """
        self.df = df.copy()
        self.setup_plotting()

    def setup_plotting(self):
        """Set up the shared plotting style for line figures."""
        set_plot_style()
        plt.style.use("default")  # Use default style with custom parameters

    def _plot_model_data(self, ax, model_data, model, var_1, var_2, is_std):
        """Plot one model's aggregated curve on the target axis.

        Args:
            ax: Target matplotlib axis.
            model_data: Aggregated dataframe for one model.
            model: Model identifier.
            var_1: Primary x-axis variable.
            var_2: Optional secondary grouping variable.
            is_std: Whether to include standard deviation error bars.

        """
        if var_2 is not None:
            list_var_2_values = sorted(list(model_data[var_2].unique()))
            list_alpha_levels = np.linspace(1, 0.3, len(list_var_2_values))
            dict_var_2_to_alpha = dict(zip(list_var_2_values, list_alpha_levels, strict=False))

            for var_2_value in model_data[var_2].unique():
                subset = model_data[model_data[var_2] == var_2_value]
                if is_std:
                    ax.errorbar(
                        subset[var_1],
                        subset["obj_mean_agg"],
                        yerr=subset["obj_std_agg"],
                        label=f"{get_display_name(model)} ({var_2}={var_2_value})",
                        alpha=dict_var_2_to_alpha[var_2_value],
                        **plt_styles[model],
                    )
                else:
                    ax.plot(
                        subset[var_1],
                        subset["obj_mean_agg"],
                        label=f"{get_display_name(model)} ({var_2}={var_2_value})",
                        alpha=dict_var_2_to_alpha[var_2_value],
                        **plt_styles[model],
                    )
        else:
            if is_std:
                ax.errorbar(
                    model_data[var_1],
                    model_data["obj_mean_agg"],
                    yerr=model_data["obj_std_agg"],
                    label=get_display_name(model),
                    **plt_styles[model],
                )
            else:
                ax.plot(
                    model_data[var_1],
                    model_data["obj_mean_agg"],
                    label=get_display_name(model),
                    **plt_styles[model],
                )

    def get_filtered_data(
        self,
        list_models: list[str],
        list_scenarios: list[str],
        list_noise_types: list[str],
        list_is_gen: list[bool],
        list_cm: list[str],
        list_ds: list[float],
        list_ms: list[int],
        list_pred_steps: list[int],
    ) -> pd.DataFrame:
        """Filter data based on specified criteria.

        Args:
            list_models: List of model names to include
            list_scenarios: List of scenarios to include
            list_noise_types: List of noise types to include
            list_is_gen: List of booleans to include
            list_cm: List of channel models to include
            list_ds: List of delay spreads to include
            list_ms: List of min speeds to include
            list_pred_steps: List of prediction steps to include

        Returns:
            Filtered DataFrame

        """
        df_filtered = self.df.copy()
        df_filtered = df_filtered[
            (
                (df_filtered["model"].isin(list_models))
                & df_filtered["scenario"].isin(list_scenarios)
                & df_filtered["noise_type"].isin(list_noise_types)
                & df_filtered["is_gen"].isin(list_is_gen)
                & (df_filtered["cm"].isin(list_cm))
                & (df_filtered["ds"].isin(list_ds))
                & (df_filtered["ms"].isin(list_ms))
                & (df_filtered["pred_step"].isin(list_pred_steps))
            )
        ]
        return df_filtered  # type: ignore

    def default_line_plot(
        self,
        df: pd.DataFrame,
        obj: str,  # nmse, se, se0
        var_1: str,  # cm, ds, ms, noise_degree, pred_step
        var_2: str | None = None,  # ms
        output_dir: str | None = None,
        is_std: bool | None = True,  # whether to plot std
        use_broken_axis: bool | None = None,  # whether to use broken axis (auto-detect if None)
        broken_axis_gap_threshold: float = 2.0,  # largest/second-largest ratio threshold
        add_generalization_bg: bool | None = False,  # whether to add generalization background colors
    ):
        """Create one line plot with optional broken-axis handling.

        The plot can show one curve per model or several grouped curves
        when ``var_2`` is provided. It also supports automatic broken-axis
        detection and optional highlighting of regular versus
        generalization ranges.

        Args:
            df: Raw dataframe to aggregate and plot.
            obj: Objective to plot on the y-axis, such as ``nmse`` or ``se``.
            var_1: Primary variable to plot on the x-axis.
            var_2: Optional secondary grouping variable.
            output_dir: Output path prefix used when saving the figure.
            is_std: Whether to include standard deviation error bars.
            use_broken_axis: Whether to force or disable broken-axis mode.
                If ``None``, the function decides automatically.
            broken_axis_gap_threshold: Ratio threshold used by the
                broken-axis detector.
            add_generalization_bg: Whether to add background shading for
                regular and generalization ranges.

        """
        if df.empty:
            print(f"Skipping empty line plot for obj={obj}, var_1={var_1}, var_2={var_2}")
            return

        scenario = str(df["scenario"].iloc[0]) if "scenario" in df.columns and not df.empty else None

        # # make a summary json to store the metadata of the plot
        # summary = {
        #     "obj": obj,
        #     "var_1": var_1,
        #     "var_2": var_2,
        #     "models": df["model"].unique().tolist(),
        #     "cm": df["cm"].unique().tolist(),
        #     "ds": df["ds"].unique().tolist(),
        #     "ms": df["ms"].unique().tolist(),
        #     "noise_types": df["noise_type"].unique().tolist(),
        #     "is_gen": df["is_gen"].unique().tolist(),
        #     "scenarios": df["scenario"].unique().tolist(),
        # }

        # Determine the column names based on the objective type
        mean_col = f"{obj}_mean"
        std_col = f"{obj}_std"

        # compute the mean and std
        df["obj_mean"] = df[mean_col]
        df["obj_std"] = df[std_col]

        group_cols = ["model", var_1]
        if var_2 is not None:
            group_cols.append(var_2)

        df_agg = (
            df.groupby(group_cols, group_keys=False)
            .apply(
                lambda g: aggregate_mean_std(g, mean_col="obj_mean", std_col="obj_std"),
                include_groups=False,  # type: ignore the pandas include_groups
            )
            .reset_index()
        )

        # Create the plot
        if var_1 == "cm":
            cm_order = ["A", "B", "C", "D", "E"]
            cm_to_idx = {cm: i for i, cm in enumerate(cm_order)}
            df_agg[var_1] = df_agg[var_1].map(cm_to_idx)  # type: ignore

        # Check if we should use broken axis based on largest-vs-second-largest line gap.
        if use_broken_axis is None:
            use_broken_axis = should_use_broken_axis(df_agg, gap_threshold=broken_axis_gap_threshold)

        outlier_model = (
            get_broken_axis_outlier_model(df_agg, gap_threshold=broken_axis_gap_threshold) if use_broken_axis else None
        )
        if use_broken_axis and outlier_model is None:
            print(
                "Warning: use_broken_axis=True but no model exceeds the largest/second-largest "
                f"ratio threshold ({broken_axis_gap_threshold}). Using regular plot."
            )
            use_broken_axis = False

        set_plot_style()
        if use_broken_axis:
            # Create subplots with shared x-axis - use same width as regular plots
            fig, (ax_top, ax_bottom) = plt.subplots(
                2,
                1,
                figsize=vis_config["figsize_single"],
                sharex=True,
                gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08},
            )

            assert outlier_model is not None

            # Separate outlier line and remaining lines
            df_top = df_agg[df_agg["model"] == outlier_model]
            df_bottom = df_agg[df_agg["model"] != outlier_model]

            # Plot outlier in top subplot
            ordered_top_models = order_models_for_scenario(df_top["model"].unique().tolist(), scenario)
            for model in ordered_top_models:
                model_data = df_top[df_top["model"] == model]
                self._plot_model_data(ax_top, model_data, model, var_1, var_2, is_std)

            # Plot remaining models in bottom subplot
            ordered_bottom_models = order_models_for_scenario(df_bottom["model"].unique().tolist(), scenario)
            for model in ordered_bottom_models:
                model_data = df_bottom[df_bottom["model"] == model]
                self._plot_model_data(ax_bottom, model_data, model, var_1, var_2, is_std)

            if obj.lower() == "se":
                # SE stays on linear scale.
                ax_top.set_yscale("linear")
                ax_bottom.set_yscale("linear")
            else:
                # NMSE and related objectives stay on log scale.
                ax_top.set_yscale("log")
                ax_bottom.set_yscale("log")

                # For top axis, keep only low/high ticks to reduce clutter.
                y_min, y_max = ax_top.get_ylim()
                ax_top.set_yticks([y_min, y_max])
                ax_top.set_yticklabels([f"{y_min:.3f}", f"{y_max:.3f}"])
                ax_top.yaxis.set_minor_formatter(NullFormatter())  # Hide minor labels to avoid clutter

                # For bottom axis, keep plain decimal major log ticks.
                y_min_bottom, y_max_bottom = ax_bottom.get_ylim()
                tick_candidates_bottom = []
                for exp in range(-3, 3):  # Cover range from 0.001 to 100
                    base_val = 10**exp
                    tick_candidates_bottom.extend([base_val, 2 * base_val, 4 * base_val])
                valid_ticks_bottom = [tick for tick in tick_candidates_bottom if y_min_bottom <= tick <= y_max_bottom]
                ax_bottom.set_yticks(valid_ticks_bottom)
                ax_bottom.set_yticklabels([_log_plain_formatter(tick, None) for tick in valid_ticks_bottom])
                ax_bottom.yaxis.set_minor_locator(plt.NullLocator())  # type: ignore
                ax_bottom.yaxis.set_minor_formatter(plt.NullFormatter())  # type: ignore
                ax_bottom.tick_params(which="minor", length=0)

                # Force refresh of the top axis to ensure log scale is applied
                ax_top.relim()
                ax_top.autoscale_view()

            # Add axis break indicators
            add_axis_break_indicators(ax_top, ax_bottom)

            # Remove x-axis labels from top plot
            ax_top.set_xlabel("")
            ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            # Set labels for bottom plot using centralized functions
            xlabel = get_xlabel(df, var_1)
            ax_bottom.set_xlabel(xlabel, fontweight="bold")
            ax_bottom.set_ylabel(obj.upper(), fontweight="bold")
            setup_axis_ticks(ax_bottom, df, var_1)

            # Setup grids (no y-label for top plot to avoid duplication)
            ax_top.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax_bottom.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Automatically find best legend position
            handles_top, labels_top = ax_top.get_legend_handles_labels()
            handles_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()

            # Use smart positioning to find the best axis and location for legend
            best_axis, best_location = find_best_legend_position(ax_top, ax_bottom)
            legend_handles, legend_labels = reorder_legend_entries(
                handles_top + handles_bottom,
                labels_top + labels_bottom,
                scenario,
            )
            best_axis.legend(legend_handles, legend_labels, loc=best_location)

            # Add generalization background if requested
            if add_generalization_bg and var_1 in ["ds", "ms"]:
                add_generalization_background(ax_top, df, var_1)
                add_generalization_background(ax_bottom, df, var_1)

            # Adjust layout manually for broken axis - optimize space usage
            fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.08, hspace=0.08)

        else:
            # Regular single plot
            fig, ax = plt.subplots(figsize=vis_config["figsize_single"])

            ordered_models = order_models_for_scenario(df_agg["model"].unique().tolist(), scenario)
            for model in ordered_models:
                model_data = df_agg[df_agg["model"] == model]
                self._plot_model_data(ax, model_data, model, var_1, var_2, is_std)

            # Set labels and ticks
            set_labels_ticks_and_titles(ax, df, obj, var_1, var_2)

            # Add generalization background if requested
            if add_generalization_bg and var_1 in ["ds", "ms"]:
                add_generalization_background(ax, df, var_1)

            # Only use tight_layout for regular plots
            plt.tight_layout()
        if output_dir is not None:
            output_fig = output_dir + "_res.pdf"
            plt.savefig(output_fig, dpi=300)
            print(f"Plot saved to {output_fig}")
            # output_summary = output_dir + "_summary.json"
            # with open(output_summary, "w") as f:
            #     json.dump(summary, f, indent=4)
        else:
            plt.show()
        plt.close(fig)


# Constants for prediction steps
LIST_PRED_STEP = [0, 1, 2, 3]


def plot_regular(
    path_consolidated_results: Path,
    dir_output: Path,
    broken_axis_gap_threshold: float = 2.0,
) -> Path:
    """Create line plots for regular test settings.

    Args:
        path_consolidated_results: Path to the consolidated results CSV.
        dir_output: Output directory for generated figures.
        broken_axis_gap_threshold: Ratio threshold used by the broken-axis
            detector.

    Returns:
        The output directory containing the generated plots.

    """
    objectives = ["nmse", "se"]
    df = pd.read_csv(path_consolidated_results)
    vis = Vis(df)

    # Extract models from data instead of using config
    dir_outputs_regular = dir_output
    dir_outputs_regular.mkdir(parents=True, exist_ok=True)

    list_is_gen = [False]

    for scenario in ["TDD", "FDD"]:
        list_scenarios = [scenario]
        models = order_models_for_scenario(
            df[df["scenario"] == scenario]["model"].unique().tolist(),
            scenario,
        )
        for noise_type in ["vanilla"]:
            list_noise_types = [noise_type]
            for obj in objectives:
                dir_outputs_obj = dir_outputs_regular / scenario / noise_type / obj
                dir_outputs_obj.mkdir(parents=True, exist_ok=True)
                for var_1 in ["noise_degree"]:
                    sub_df = vis.get_filtered_data(
                        list_models=models,
                        list_scenarios=list_scenarios,
                        list_noise_types=list_noise_types,
                        list_is_gen=list_is_gen,
                        list_cm=LIST_CHANNEL_MODEL,
                        list_ds=LIST_DELAY_SPREAD,
                        list_ms=LIST_MIN_SPEED_TEST,
                        list_pred_steps=LIST_PRED_STEP,
                    )

                    vis.default_line_plot(
                        df=sub_df,
                        obj=obj,
                        var_1=var_1,
                        output_dir=str(dir_outputs_obj / f"{scenario}_{noise_type}_{obj}_{var_1}_no_std"),
                        is_std=False,
                        broken_axis_gap_threshold=broken_axis_gap_threshold,
                    )

    return dir_output


def plot_robustness(
    path_consolidated_results: Path,
    dir_output: Path,
    broken_axis_gap_threshold: float = 2.0,
) -> Path:
    """Create line plots for robustness test settings.

    Args:
        path_consolidated_results: Path to the consolidated results CSV.
        dir_output: Output directory for generated figures.
        broken_axis_gap_threshold: Ratio threshold used by the broken-axis
            detector.

    Returns:
        The output directory containing the generated plots.

    """
    objectives = ["nmse", "se"]
    df = pd.read_csv(path_consolidated_results)
    vis = Vis(df)

    # Extract models from data instead of using config
    dir_outputs_rob = dir_output
    dir_outputs_rob.mkdir(parents=True, exist_ok=True)

    list_is_gen = [False]

    for scenario in ["TDD", "FDD"]:
        list_scenarios = [scenario]
        models = order_models_for_scenario(
            df[df["scenario"] == scenario]["model"].unique().tolist(),
            scenario,
        )
        for noise_type in ["phase", "burst", "packagedrop"]:
            list_noise_types = [noise_type]
            for obj in objectives:
                dir_outputs_obj = dir_outputs_rob / scenario / noise_type / obj
                dir_outputs_obj.mkdir(parents=True, exist_ok=True)
                for var_1 in ["noise_degree"]:
                    sub_df = vis.get_filtered_data(
                        list_models=models,
                        list_scenarios=list_scenarios,
                        list_noise_types=list_noise_types,
                        list_is_gen=list_is_gen,
                        list_cm=LIST_CHANNEL_MODEL,
                        list_ds=LIST_DELAY_SPREAD,
                        list_ms=LIST_MIN_SPEED_TEST,
                        list_pred_steps=LIST_PRED_STEP,
                    )

                    vis.default_line_plot(
                        df=sub_df,
                        obj=obj,
                        var_1=var_1,
                        output_dir=str(dir_outputs_obj / f"{scenario}_{noise_type}_{obj}_{var_1}_no_std"),
                        is_std=False,
                        broken_axis_gap_threshold=broken_axis_gap_threshold,
                    )

    return dir_output


def plot_generalization(
    path_consolidated_results: Path,
    dir_output: Path,
    broken_axis_gap_threshold: float = 2.0,
) -> Path:
    """Create line plots for generalization test settings.

    Args:
        path_consolidated_results: Path to the consolidated results CSV.
        dir_output: Output directory for generated figures.
        broken_axis_gap_threshold: Ratio threshold used by the broken-axis
            detector.

    Returns:
        The output directory containing the generated plots.

    """
    objectives = ["nmse", "se"]
    df = pd.read_csv(path_consolidated_results)
    vis = Vis(df)

    # Extract models from data instead of using config
    # Interpolation plots
    for gen_var in ["ms"]:
        dir_outputs_interpolation = dir_output / gen_var
        dir_outputs_interpolation.mkdir(parents=True, exist_ok=True)

        if gen_var == "ms":
            list_channel_model_interpolation = LIST_CHANNEL_MODEL
            list_delay_spread_interpolation = LIST_DELAY_SPREAD
            list_min_speed_interpolation = LIST_MIN_SPEED_TEST_GEN
        else:
            raise ValueError("gen_var must be one of ['ms']")

        list_is_gen = [True]

        for scenario in ["TDD", "FDD"]:
            list_scenarios = [scenario]
            models = order_models_for_scenario(
                df[df["scenario"] == scenario]["model"].unique().tolist(),
                scenario,
            )
            for noise_type in ["vanilla"]:
                list_noise_types = [noise_type]
                for obj in objectives:
                    dir_outputs_obj = dir_outputs_interpolation / scenario / noise_type / obj
                    dir_outputs_obj.mkdir(parents=True, exist_ok=True)
                    for var_1 in [gen_var]:
                        sub_df = vis.get_filtered_data(
                            list_models=models,
                            list_scenarios=list_scenarios,
                            list_noise_types=list_noise_types,
                            list_is_gen=list_is_gen,
                            list_cm=list_channel_model_interpolation,
                            list_ds=list_delay_spread_interpolation,
                            list_ms=list_min_speed_interpolation,
                            list_pred_steps=LIST_PRED_STEP,
                        )

                        vis.default_line_plot(
                            df=sub_df,
                            obj=obj,
                            var_1=var_1,
                            output_dir=str(dir_outputs_obj / f"{scenario}_{noise_type}_{obj}_{var_1}_no_std"),
                            is_std=False,
                            broken_axis_gap_threshold=broken_axis_gap_threshold,
                            add_generalization_bg=(
                                var_1 in ["ds", "ms"]
                            ),  # Add background for delay spread and min speed
                        )

    return dir_output


def plot_line(
    path_consolidated_results: Path,
    dir_output: Path,
    broken_axis_gap_threshold: float = 2.0,
) -> None:
    """Create the full line-plot visualization suite.

    Args:
        path_consolidated_results: Path to the consolidated results CSV.
        dir_output: Root output directory for all line-plot artifacts.
        broken_axis_gap_threshold: Ratio threshold used by the broken-axis
            detector.

    """
    dir_output.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each plot type
    dir_regular = plot_regular(
        path_consolidated_results,
        dir_output / "regular",
        broken_axis_gap_threshold=broken_axis_gap_threshold,
    )
    dir_robustness = plot_robustness(
        path_consolidated_results,
        dir_output / "robustness",
        broken_axis_gap_threshold=broken_axis_gap_threshold,
    )
    dir_generalization = plot_generalization(
        path_consolidated_results,
        dir_output / "generalization",
        broken_axis_gap_threshold=broken_axis_gap_threshold,
    )

    print(f"Regular plots saved to {dir_regular}")
    print(f"Robustness plots saved to {dir_robustness}")
    print(f"Generalization plots saved to {dir_generalization}")
