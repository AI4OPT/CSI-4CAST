"""Shared utility helpers for AHPT modules.

This module contains lightweight helpers used across the tuning
infrastructure. The functions here are intentionally small and free of
project-specific side effects so they can be reused by submission,
runner, and configuration code.
"""

import datetime


TIME_FORMAT = "%Y%m%d_%H%M%S"


def get_current_time() -> str:
    """Return the current timestamp in the project time format.

    The returned string is used for naming run directories, job scripts,
    and other generated artifacts that should remain sortable by time.
    """
    return datetime.datetime.now().strftime(TIME_FORMAT)
