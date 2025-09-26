import datetime
from pathlib import Path
from typing import Optional


TIME_FORMAT = "%Y%m%d_%H%M%S"


def get_current_time():
    return datetime.datetime.now().strftime(TIME_FORMAT)


def get_latest_datetime_folder(parent_dir: Path) -> Path | None:
    """Get the latest folder that matches the TIME_FORMAT pattern (YYYYMMDD_HHMMSS).

    Args:
        parent_dir: Parent directory to search in

    Returns:
        Path to the latest datetime folder, or None if no valid folders found

    """
    if not parent_dir.exists():
        return None

    # Find all folders that match the time format
    datetime_folders = []
    for folder in parent_dir.iterdir():
        if folder.is_dir():
            try:
                # Try to parse the folder name as a datetime
                datetime.datetime.strptime(folder.name, TIME_FORMAT)
                datetime_folders.append(folder)
            except ValueError:
                # Skip folders that don't match the time format
                continue

    if not datetime_folders:
        return None

    # Sort by the datetime in the folder name (newest first)
    datetime_folders.sort(key=lambda p: datetime.datetime.strptime(p.name, TIME_FORMAT), reverse=True)

    return datetime_folders[0]
