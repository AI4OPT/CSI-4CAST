import matplotlib.pyplot as plt


CONST_FIGSIZE = (10, 8)
CONST_LABEL_FONTSIZE = 18
CONST_TICK_FONTSIZE_MAJOR = 16
CONST_TICK_FONTSIZE_MINOR = 4
CONST_LEGEND_FONTSIZE = 12
CONST_TITLE_FONTSIZE = 20
CONST_LINEWIDTH = 3
CONST_MARKERSIZE = 12

# Background colors for generalization plots
CONST_COLOR_REGULAR_BG = "#f7fcf5"  # Very light green for regular range
CONST_COLOR_GENERALIZATION_BG = "#fee0d2"  # Very light orange for generalization range
CONST_ALPHA_BACKGROUND = 0.3  # Transparency for background colors


def set_plot_style():
    plt.rcParams.update(
        {
            "axes.titlesize": CONST_TITLE_FONTSIZE,
            "axes.titleweight": "bold",  # Make titles bold
            "axes.labelsize": CONST_LABEL_FONTSIZE,
            "axes.labelweight": "bold",  # Make axis labels bold
            "xtick.labelsize": CONST_TICK_FONTSIZE_MAJOR,
            "ytick.labelsize": CONST_TICK_FONTSIZE_MAJOR,
            "xtick.minor.size": CONST_TICK_FONTSIZE_MINOR,
            "ytick.minor.size": CONST_TICK_FONTSIZE_MINOR,
            "legend.fontsize": CONST_LEGEND_FONTSIZE,
            "font.weight": "bold",  # Make all text bold by default
            "axes.labelcolor": "black",  # Ensure labels are black and bold
            "xtick.color": "black",  # Make x-tick labels black
            "ytick.color": "black",  # Make y-tick labels black
            "lines.linewidth": CONST_LINEWIDTH,
            "lines.markersize": CONST_MARKERSIZE,
        }
    )


vis_config = {
    "figsize_single": CONST_FIGSIZE,
    "color_regular_bg": CONST_COLOR_REGULAR_BG,
    "color_generalization_bg": CONST_COLOR_GENERALIZATION_BG,
    "alpha_background": CONST_ALPHA_BACKGROUND,
}

# Model name mapping for display labels
model_display_names = {
    "MODEL": "CSI-4CAST",
    # Add more mappings as needed
}


def get_display_name(model_name: str) -> str:
    """Get the display name for a model, with LaTeX formatting if specified."""
    return model_display_names.get(model_name, model_name)


plt_styles = {
    "NP": {"color": [8 / 255, 48 / 255, 107 / 255], "linestyle": "--", "marker": "d"},
    "RNN": {
        "color": [254 / 255, 217 / 255, 118 / 255],
        "linestyle": "-",
        "marker": "^",
    },
    "PAD": {"color": [84 / 255, 39 / 255, 143 / 255], "linestyle": "-", "marker": "<"},
    "STEMGNN": {"color": [116 / 255, 196 / 255, 118 / 255], "linestyle": "-", "marker": "o"},
    "CNN": {"color": [241 / 255, 105 / 255, 19 / 255], "linestyle": "-", "marker": ">"},
    "MODEL1": {
        "color": [197 / 255, 180 / 255, 227 / 255],
        "linestyle": "-",
        "marker": "o",
    },  # Light purple
    "MODEL": {
        "color": [197 / 255, 180 / 255, 227 / 255],
        "linestyle": "-",
        "marker": "o",
    },  # Light purple
    "LLM4CP": {
        "color": [33 / 255, 113 / 255, 181 / 255],
        "linestyle": "-",
        "marker": "s",
    },
    "DeepAR": {
        "color": [233 / 255, 196 / 255, 106 / 255],
        "linestyle": "--",
        "marker": "h",
    },  # Warm pastel yellow-orange
    # "GRU": {
    #     "color": [169 / 255, 204 / 255, 227 / 255],
    #     "linestyle": ":",
    #     "marker": "v",
    # },  # Soft sky blue
    # "STEMGNN": {
    #     "color": [178 / 255, 200 / 255, 187 / 255],
    #     "linestyle": "-.",
    #     "marker": "|",
    # },  # Muted green-gray
}
