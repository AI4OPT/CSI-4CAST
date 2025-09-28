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
    "NP": "NP",  # Neural Process model
    "RNN": "RNN",  # Recurrent Neural Network model
    # Add more model mappings here as needed
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


# =============================================================================
# Model-Specific Plotting Styles
# =============================================================================
# Each model has a unique combination of color, line style, and marker
# to ensure clear distinction in multi-model comparisons

plt_styles = {
    "NP": {
        "color": [8 / 255, 48 / 255, 107 / 255],  # Dark blue - professional and distinct
        "linestyle": "--",  # Dashed line for easy identification
        "marker": "d",  # Diamond marker
    },
    "RNN": {
        "color": [254 / 255, 217 / 255, 118 / 255],  # Warm yellow - high contrast with NP
        "linestyle": "-",  # Solid line
        "marker": "^",  # Triangle marker
    },
    # Additional model styles can be added here following the same pattern
    # Ensure each model has a unique combination of visual properties
}
