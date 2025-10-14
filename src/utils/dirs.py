"""Directory path constants for the project.

This module defines all the standard directory paths used throughout the project
for data storage, outputs, model weights, and external dependencies. Using
centralized path constants ensures consistency and makes path management easier.
"""

# Main storage directory for all project artifacts
DIR_STORAGE = "z_artifacts"

# Data directory for datasets, preprocessed data, and statistics
DIR_DATA = DIR_STORAGE + "/data"

# Output directory for experiment results, logs, and visualizations
DIR_OUTPUTS = DIR_STORAGE + "/outputs"

# Directory for storing trained model weights and checkpoints
DIR_WEIGHTS = DIR_STORAGE + "/weights"
