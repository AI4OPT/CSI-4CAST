"""Visualization utilities and styling configurations for plotting.

This module provides comprehensive plotting utilities for creating consistent,
professional visualizations in the CSI prediction project. It includes:

- Standardized plot styling and formatting constants
- Color schemes optimized for scientific publications
- Model-specific styling configurations
- Background highlighting for different data ranges (training vs generalization)
- Utility functions for consistent plot appearance

The styling follows best practices for scientific visualization with:
- High contrast colors for accessibility
- Appropriate font sizes for readability
- Consistent line styles and markers for different models
- Professional color palette suitable for publications
"""

import matplotlib.pyplot as plt


# =============================================================================
# Plot Styling Constants
# =============================================================================
# These constants ensure consistent appearance across all visualizations

# Figure dimensions optimized for readability and publication
CONST_FIGSIZE = (10, 8)

# Font sizes following scientific publication standards
CONST_LABEL_FONTSIZE = 18  # Axis labels
CONST_TICK_FONTSIZE_MAJOR = 16  # Major tick labels
CONST_TICK_FONTSIZE_MINOR = 4  # Minor tick labels
CONST_LEGEND_FONTSIZE = 12  # Legend text
CONST_TITLE_FONTSIZE = 20  # Plot titles

# Line and marker styling for clear visualization
CONST_LINEWIDTH = 3  # Line thickness for visibility
CONST_MARKERSIZE = 12  # Marker size for data points

# Background colors for highlighting different data ranges
CONST_COLOR_REGULAR_BG = "#f7fcf5"  # Very light green for training range
CONST_COLOR_GENERALIZATION_BG = "#fee0d2"  # Very light orange for generalization range
CONST_ALPHA_BACKGROUND = 0.3  # Transparency for background highlighting


def set_plot_style():
    """Apply standardized plot styling to matplotlib.

    Configures matplotlib's global parameters (rcParams) to ensure consistent
    appearance across all plots in the project. The styling emphasizes:
    - Bold, readable fonts for scientific publications
    - High contrast colors for accessibility
    - Appropriate sizing for different plot elements
    - Professional appearance suitable for papers and presentations

    This function should be called once at the beginning of any plotting script
    to ensure consistent styling throughout the visualization.

    Example:
        >>> set_plot_style()
        >>> plt.figure(figsize=CONST_FIGSIZE)
        >>> # Your plotting code here...

    """
    plt.rcParams.update(
        {
            # Title styling - bold and large for emphasis
            "axes.titlesize": CONST_TITLE_FONTSIZE,
            "axes.titleweight": "bold",
            # Axis label styling - bold and readable
            "axes.labelsize": CONST_LABEL_FONTSIZE,
            "axes.labelweight": "bold",
            "axes.labelcolor": "black",  # Ensure high contrast
            # Tick label styling
            "xtick.labelsize": CONST_TICK_FONTSIZE_MAJOR,
            "ytick.labelsize": CONST_TICK_FONTSIZE_MAJOR,
            "xtick.minor.size": CONST_TICK_FONTSIZE_MINOR,
            "ytick.minor.size": CONST_TICK_FONTSIZE_MINOR,
            "xtick.color": "black",  # High contrast tick labels
            "ytick.color": "black",
            # Legend and general font styling
            "legend.fontsize": CONST_LEGEND_FONTSIZE,
            "font.weight": "bold",  # Bold text for better readability
            # Line and marker defaults
            "lines.linewidth": CONST_LINEWIDTH,
            "lines.markersize": CONST_MARKERSIZE,
        }
    )


# =============================================================================
# Visualization Configuration Dictionary
# =============================================================================
# Centralized configuration for easy access to styling parameters

vis_config = {
    "figsize_single": CONST_FIGSIZE,  # Standard figure size
    "color_regular_bg": CONST_COLOR_REGULAR_BG,  # Training range background
    "color_generalization_bg": CONST_COLOR_GENERALIZATION_BG,  # Generalization range background
    "alpha_background": CONST_ALPHA_BACKGROUND,  # Background transparency
}

# =============================================================================
# Model Display Configuration
# =============================================================================
# Mapping from internal model names to display-friendly labels

# Model name mapping for consistent display across visualizations
model_display_names = {
    "NP": "NP",
    "AR": "AR",
    "WIENER": "Wiener",
    "PAD": "PAD",
    "RNN": "RNN",
    "STEMGNN": "StemGNN",
    "CNN": "CNN",
    "MODEL": "CSI-4CAST",
    "LLM4CP": "LLM4CP",
    "ABL_NO_DENOISER": "No CNN",
    "ABL_NO_IDFT": "No IDFT",
    "ABL_NO_ARL": "No ACL",
    "ABL_NORM_REPLACE_ARL": "Norm Replace ACL",
    "ABL_ADD_SUBCARRIER_ARL": "Add Subcarrier ACL",
    "ABL_NO_SUBCARRIER_ARL": "No Subcarrier ACL",
    "ABL_MLP_REPLACE_EMBED": "MLP Replace ShuffleNet",
    "ABL_MOBILENET_REPLACE_EMBED": "MobileNet Replace ShuffleNet",
    "ABL_MLP_REPLACE_PRED": "MLP Replace Transformer",
    "ABL_LSTM_REPLACE_PRED": "LSTM Replace Transformer",
    "NO_DENOISER": "No CNN",
    "NO_IDFT": "No IDFT",
    "NO_ARL": "No ACL",
    "NORM_REPLACE_ARL": "Norm Replace ACL",
    "ADD_SUBCARRIER_ARL": "Add Subcarrier ACL",
    "NO_SUBCARRIER_ARL": "No Subcarrier ACL",
    "MLP_REPLACE_EMBED": "MLP Replace ShuffleNet",
    "MOBILENET_REPLACE_EMBED": "MobileNet Replace ShuffleNet",
    "MLP_REPLACE_PRED": "MLP Replace Transformer",
    "LSTM_REPLACE_PRED": "LSTM Replace Transformer",
}


PAPER_MODEL_ORDER_BY_SCENARIO = {
    "TDD": ("NP", "AR", "WIENER", "PAD", "CNN", "STEMGNN", "RNN", "LLM4CP", "MODEL"),
    "FDD": ("NP", "WIENER", "CNN", "STEMGNN", "RNN", "LLM4CP", "MODEL"),
}

_BASELINE_DISPLAY_TO_MODEL = {
    "NP": "NP",
    "AR": "AR",
    "Wiener": "WIENER",
    "PAD": "PAD",
    "CNN": "CNN",
    "StemGNN": "STEMGNN",
    "RNN": "RNN",
    "LLM4CP": "LLM4CP",
    "CSI-4CAST": "MODEL",
}


def get_display_name(model_name: str) -> str:
    """Get the display-friendly name for a model.

    Converts internal model names to display-friendly versions suitable for
    plot legends, titles, and labels. If a model name is not found in the
    mapping, returns the original name as fallback.

    Args:
        model_name (str): Internal model name (e.g., from config files)

    Returns:
        str: Display-friendly model name for use in plots and labels

    Example:
        >>> display_name = get_display_name("RNN")
        >>> print(display_name)  # "RNN"
        >>>
        >>> unknown_name = get_display_name("UnknownModel")
        >>> print(unknown_name)  # "UnknownModel" (fallback)

    """
    return model_display_names.get(model_name, model_name)


def order_models_for_scenario(models: list[str], scenario: str | None) -> list[str]:
    """Return models ordered to match the paper's scenario-specific presentation."""
    if scenario is None:
        return sorted(models)

    preferred_order = PAPER_MODEL_ORDER_BY_SCENARIO.get(scenario.upper())
    if preferred_order is None:
        return sorted(models)

    model_set = set(models)
    ordered_models = [model for model in preferred_order if model in model_set]
    remaining_models = sorted(model for model in models if model not in set(ordered_models))
    return ordered_models + remaining_models


def reorder_legend_entries(handles: list, labels: list[str], scenario: str | None) -> tuple[list, list[str]]:
    """Reorder legend entries to match the paper order for the given scenario."""
    if scenario is None:
        return handles, labels

    preferred_order = PAPER_MODEL_ORDER_BY_SCENARIO.get(scenario.upper())
    if preferred_order is None:
        return handles, labels

    preferred_rank = {model: idx for idx, model in enumerate(preferred_order)}
    indexed_entries = []
    fallback_index = len(preferred_rank)

    for idx, (handle, label) in enumerate(zip(handles, labels, strict=False)):
        base_label = label.split(" (", 1)[0]
        model_name = _BASELINE_DISPLAY_TO_MODEL.get(base_label, base_label)
        rank = preferred_rank.get(model_name, fallback_index + idx)
        indexed_entries.append((rank, idx, handle, label))

    indexed_entries.sort(key=lambda item: (item[0], item[1]))
    ordered_handles = [item[2] for item in indexed_entries]
    ordered_labels = [item[3] for item in indexed_entries]
    return ordered_handles, ordered_labels


# =============================================================================
# Model-Specific Plotting Styles
# =============================================================================
# Each model has a unique combination of color, line style, and marker
# to ensure clear distinction in multi-model comparisons

plt_styles = {
    "NP": {"color": [8 / 255, 48 / 255, 107 / 255], "linestyle": "--", "marker": "d"},
    "AR": {"color": [128 / 255, 128 / 255, 128 / 255], "linestyle": "--", "marker": "h"},
    "WIENER": {"color": [169 / 255, 204 / 255, 227 / 255], "linestyle": ":", "marker": "v"},
    "PAD": {"color": [84 / 255, 39 / 255, 143 / 255], "linestyle": "-", "marker": "<"},
    "RNN": {"color": [254 / 255, 217 / 255, 118 / 255], "linestyle": "-", "marker": "^"},
    "STEMGNN": {"color": [116 / 255, 196 / 255, 118 / 255], "linestyle": "-", "marker": "o"},
    "CNN": {"color": [241 / 255, 105 / 255, 19 / 255], "linestyle": "-", "marker": ">"},
    "MODEL": {"color": [197 / 255, 180 / 255, 227 / 255], "linestyle": "-", "marker": "o"},
    "LLM4CP": {"color": [33 / 255, 113 / 255, 181 / 255], "linestyle": "-", "marker": "s"},
    "ABL_NO_DENOISER": {"color": [8 / 255, 48 / 255, 107 / 255], "linestyle": "--", "marker": "d"},
    "ABL_NO_IDFT": {"color": [128 / 255, 128 / 255, 128 / 255], "linestyle": "--", "marker": "h"},
    "ABL_NO_ARL": {"color": [169 / 255, 204 / 255, 227 / 255], "linestyle": ":", "marker": "v"},
    "ABL_NORM_REPLACE_ARL": {"color": [254 / 255, 217 / 255, 118 / 255], "linestyle": "-", "marker": "^"},
    "ABL_ADD_SUBCARRIER_ARL": {"color": [116 / 255, 196 / 255, 118 / 255], "linestyle": "-", "marker": "o"},
    "ABL_NO_SUBCARRIER_ARL": {"color": [241 / 255, 105 / 255, 19 / 255], "linestyle": "-", "marker": ">"},
    "ABL_MLP_REPLACE_EMBED": {"color": [33 / 255, 113 / 255, 181 / 255], "linestyle": "-", "marker": "s"},
    "ABL_MOBILENET_REPLACE_EMBED": {"color": [84 / 255, 39 / 255, 143 / 255], "linestyle": "-", "marker": "<"},
    "ABL_MLP_REPLACE_PRED": {"color": [0 / 255, 0 / 255, 0 / 255], "linestyle": ":", "marker": "P"},
    "ABL_LSTM_REPLACE_PRED": {"color": [203 / 255, 24 / 255, 29 / 255], "linestyle": "-.", "marker": "X"},
    "NO_DENOISER": {"color": [8 / 255, 48 / 255, 107 / 255], "linestyle": "--", "marker": "d"},
    "NO_IDFT": {"color": [128 / 255, 128 / 255, 128 / 255], "linestyle": "--", "marker": "h"},
    "NO_ARL": {"color": [169 / 255, 204 / 255, 227 / 255], "linestyle": ":", "marker": "v"},
    "NORM_REPLACE_ARL": {"color": [254 / 255, 217 / 255, 118 / 255], "linestyle": "-", "marker": "^"},
    "ADD_SUBCARRIER_ARL": {"color": [116 / 255, 196 / 255, 118 / 255], "linestyle": "-", "marker": "o"},
    "NO_SUBCARRIER_ARL": {"color": [241 / 255, 105 / 255, 19 / 255], "linestyle": "-", "marker": ">"},
    "MLP_REPLACE_EMBED": {"color": [33 / 255, 113 / 255, 181 / 255], "linestyle": "-", "marker": "s"},
    "MOBILENET_REPLACE_EMBED": {"color": [84 / 255, 39 / 255, 143 / 255], "linestyle": "-", "marker": "<"},
    "MLP_REPLACE_PRED": {"color": [0 / 255, 0 / 255, 0 / 255], "linestyle": ":", "marker": "P"},
    "LSTM_REPLACE_PRED": {"color": [203 / 255, 24 / 255, 29 / 255], "linestyle": "-.", "marker": "X"},
}
