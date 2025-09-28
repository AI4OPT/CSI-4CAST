"""Time and datetime utilities for experiment management.

This module provides utilities for handling timestamps and datetime-based
file/directory operations commonly used in machine learning experiments:

- Standardized timestamp formatting for consistent naming
- Directory discovery based on datetime patterns
- Experiment run organization by timestamp

The module uses a consistent time format (YYYYMMDD_HHMMSS) across all
utilities to ensure proper sorting and identification of experiment runs.
"""

import datetime
from pathlib import Path


# Standard time format used throughout the project for consistent timestamp formatting
# Format: YYYYMMDD_HHMMSS (e.g., "20250928_143022" for Sep 28, 2025 at 2:30:22 PM)
TIME_FORMAT = "%Y%m%d_%H%M%S"


def get_current_time():
    """Get current timestamp in standardized format.

    Returns the current date and time as a formatted string using the
    project's standard time format. This is commonly used for creating
    unique identifiers for experiment runs, log files, and output directories.

    Returns:
        str: Current timestamp in format YYYYMMDD_HHMMSS

    Example:
        >>> timestamp = get_current_time()
        >>> print(timestamp)  # "20250928_143022"

    """
    return datetime.datetime.now().strftime(TIME_FORMAT)


def get_latest_datetime_folder(parent_dir: Path) -> Path | None:
    """Get the most recent folder that matches the standard datetime naming pattern.

    Searches through a parent directory for subdirectories that follow the
    project's standard datetime naming convention (YYYYMMDD_HHMMSS)
    returns the path to the most recently created one.

    This is useful for finding the latest experiment run or output directory
    when multiple timestamped directories exist.

    Args:
        parent_dir (Path): Parent directory to search for datetime-named folders

    Returns:
        Path | None: Path to the most recent datetime folder, or None if:
                    - The parent directory doesn't exist
                    - No folders match the datetime pattern

    Example:
        >>> parent = Path("experiments/outputs")
        >>> latest = get_latest_datetime_folder(parent)
        >>> if latest:
        ...     print(f"Latest run: {latest.name}")  # "20250928_143022"

    """
    # Check if parent directory exists
    if not parent_dir.exists():
        return None

    # Collect all folders that match the datetime naming pattern
    datetime_folders = []
    for folder in parent_dir.iterdir():
        if folder.is_dir():
            try:
                # Attempt to parse folder name as datetime using standard format
                datetime.datetime.strptime(folder.name, TIME_FORMAT)
                datetime_folders.append(folder)
            except ValueError:
                # Skip folders that don't match the expected datetime format
                continue

    # Return None if no valid datetime folders found
    if not datetime_folders:
        return None

    # Sort folders by their datetime (newest first) and return the most recent
    datetime_folders.sort(key=lambda p: datetime.datetime.strptime(p.name, TIME_FORMAT), reverse=True)

    return datetime_folders[0]
