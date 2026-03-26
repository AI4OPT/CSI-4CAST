"""Configuration for CSI Prediction Model Testing.

This module defines:
- LIST_SETTINGS: The master list of (model, scenario, test_type) evaluation entries.
- Per-test-type combination creators and expected counts.
- SLURM array ID mapping so each entry gets a contiguous range of job IDs.
- A __main__ block that prints every setting and its array ID range.

Run directly to see the mapping table::

    python3 -m src.testing.config
"""

from itertools import product

import torch

from src.noise.noise_testing import Noise
from src.utils.data_utils import (
    LIST_CHANNEL_MODEL,
    LIST_CHANNEL_MODEL_GEN,
    LIST_DELAY_SPREAD,
    LIST_DELAY_SPREAD_GEN,
    LIST_MIN_SPEED_TEST,
    LIST_MIN_SPEED_TEST_GEN,
)


# ============================================================================
# Scenario / noise constants
# ============================================================================

LIST_SCENARIOS = ["TDD", "FDD"]
LIST_NOISE_TYPES = ["phase", "burst", "vanilla", "packagedrop"]
LIST_TEST_TYPES = ["regular", "robustness", "generalization"]

BATCH_SIZE = 1
CPU_ONLY_MODELS = {"AR", "WIENER", "PAD", "NP"}


# ============================================================================
# Expected combination count per *single* scenario for each test type.
# Used to verify evaluation completeness.
# ============================================================================

# Regular: vanilla noise, standard data grid
#   6 SNR x 3 CM x 3 DS x 3 MS = 162
SCENARIOS_PER_REGULAR = 162

# Robustness: (phase 4 + burst 4 + packagedrop 10) x 3 CM x 3 DS x 3 MS = 486
SCENARIOS_PER_ROBUSTNESS = 486

# Generalization: vanilla noise, expanded data grid
#   6 SNR x 5 CM x 6 DS x 17 MS = 3060
SCENARIOS_PER_GENERALIZATION = 3060

SCENARIOS_PER_TEST_TYPE: dict[str, int] = {
    "regular": SCENARIOS_PER_REGULAR,
    "robustness": SCENARIOS_PER_ROBUSTNESS,
    "generalization": SCENARIOS_PER_GENERALIZATION,
}

# ============================================================================
# SLURM jobs per test type.
# Balanced so each job processes about 162 combinations:
#   regular        162 / 1  = 162
#   robustness     486 / 3  = 162
#   generalization 3060 / 19 ≈ 161
# ============================================================================

JOBS_PER_REGULAR = 1
JOBS_PER_ROBUSTNESS = 3
JOBS_PER_GENERALIZATION = 19

JOBS_PER_TEST_TYPE: dict[str, int] = {
    "regular": JOBS_PER_REGULAR,
    "robustness": JOBS_PER_ROBUSTNESS,
    "generalization": JOBS_PER_GENERALIZATION,
}

# ============================================================================
# Settings list -- the master evaluation roster
# ============================================================================


TDD_ONLY_MODELS = ["PAD", "AR"]
DUAL_SCENARIO_MODELS = ["CNN", "LLM4CP", "MODEL", "NP", "RNN", "STEMGNN", "WIENER"]
TRAINED_MODEL_SCENARIOS = ["FDD", "TDD"]

ABLATION_TDD_MODELS = [
    "ABL_NO_DENOISER",
    "ABL_NO_IDFT",
    "ABL_NO_ARL",
    "ABL_NORM_REPLACE_ARL",
    "ABL_ADD_SUBCARRIER_ARL",
    "ABL_MLP_REPLACE_EMBED",
    "ABL_MOBILENET_REPLACE_EMBED",
    "ABL_MLP_REPLACE_PRED",
    "ABL_LSTM_REPLACE_PRED",
]

ABLATION_FDD_MODELS = ["ABL_NO_ARL", "ABL_NO_SUBCARRIER_ARL"]

# (model_name, scenario, test_type)
LIST_SETTINGS: list[tuple[str, str, str]] = [
    *((model_name, "TDD", test_type) for model_name, test_type in product(TDD_ONLY_MODELS, LIST_TEST_TYPES)),
    *product(DUAL_SCENARIO_MODELS, TRAINED_MODEL_SCENARIOS, LIST_TEST_TYPES),
    *((model_name, "TDD", test_type) for model_name, test_type in product(ABLATION_TDD_MODELS, LIST_TEST_TYPES)),
    *((model_name, "FDD", test_type) for model_name, test_type in product(ABLATION_FDD_MODELS, LIST_TEST_TYPES)),
]

# ============================================================================
# Combination creation functions
# ============================================================================


def create_regular_combinations(scenario: str) -> list[tuple]:
    """Regular testing: vanilla noise, standard data grid."""
    noise = Noise()
    combos = []
    for nd, cm, ds, ms in product(noise.list_vanilla_snr, LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TEST):
        combos.append((scenario, False, "vanilla", nd, cm, ds, ms))
    return combos


def create_robustness_combinations(scenario: str) -> list[tuple]:
    """Robustness testing: phase / burst / packagedrop noise, standard data grid."""
    noise = Noise()
    combos = []
    for noise_type in ["phase", "burst", "packagedrop"]:
        list_nd = (
            getattr(noise, f"list_{noise_type}_snr")
            if noise_type != "packagedrop"
            else getattr(noise, f"list_{noise_type}_nd")
        )
        for nd, cm, ds, ms in product(list_nd, LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TEST):
            combos.append((scenario, False, noise_type, nd, cm, ds, ms))
    return combos


def create_generalization_combinations(scenario: str) -> list[tuple]:
    """Generalization testing: vanilla noise, expanded data grid."""
    noise = Noise()
    combos = []
    for nd, cm, ds, ms in product(
        noise.list_vanilla_snr, LIST_CHANNEL_MODEL_GEN, LIST_DELAY_SPREAD_GEN, LIST_MIN_SPEED_TEST_GEN
    ):
        combos.append((scenario, True, "vanilla", nd, cm, ds, ms))
    return combos


CREATE_COMBINATIONS_FN: dict = {
    "regular": create_regular_combinations,
    "robustness": create_robustness_combinations,
    "generalization": create_generalization_combinations,
}


def create_combinations_for_setting(scenario: str, test_type: str) -> list[tuple]:
    """Create the sub-combination list for a specific (scenario, test_type)."""
    return CREATE_COMBINATIONS_FN[test_type](scenario)


# ============================================================================
# SLURM array ID mapping
# ============================================================================


def get_setting_array_ranges() -> list[tuple[str, str, str, int, int]]:
    """Return (model, scenario, test_type, start_id, end_id) for every setting.

    Array IDs are 1-based (SLURM convention).
    """
    ranges: list[tuple[str, str, str, int, int]] = []
    current_id = 1
    for model_name, scenario, test_type in LIST_SETTINGS:
        n_jobs = JOBS_PER_TEST_TYPE[test_type]
        start = current_id
        end = current_id + n_jobs - 1
        ranges.append((model_name, scenario, test_type, start, end))
        current_id = end + 1
    return ranges


def get_array_range_for_setting(model_name: str, scenario: str, test_type: str) -> tuple[int, int]:
    """Return the 1-based (start, end) array ID range for one setting line."""
    for m, s, t, start, end in get_setting_array_ranges():
        if m == model_name and s == scenario and t == test_type:
            return start, end
    raise ValueError(f"Setting ({model_name}, {scenario}, {test_type}) not found")


def get_total_array_jobs() -> int:
    """Total number of SLURM array jobs across all settings."""
    return sum(JOBS_PER_TEST_TYPE[t] for _, _, t in LIST_SETTINGS)


def get_array_mapping(array_id: int) -> tuple[str, str, str, tuple[int, int]]:
    """Map a 1-based SLURM array task ID to its setting and slice info.

    Returns:
        (model_name, scenario, test_type, (slice_idx, total_slices))

    """
    for m, s, t, start, end in get_setting_array_ranges():
        if start <= array_id <= end:
            n_jobs = JOBS_PER_TEST_TYPE[t]
            slice_idx = array_id - start
            return m, s, t, (slice_idx, n_jobs)
    total = get_total_array_jobs()
    raise ValueError(f"Array ID {array_id} out of range. Valid range: 1-{total}")


# ============================================================================
# Slicing helper
# ============================================================================


def slice_combinations(list_all_combs: list, slice_info: tuple[int, int]) -> list:
    """Divide a combination list into a balanced slice for one SLURM job."""
    slice_idx, total_jobs = slice_info
    total_combs = len(list_all_combs)
    combs_per_job = total_combs // total_jobs
    remainder = total_combs % total_jobs

    if slice_idx < remainder:
        start_idx = slice_idx * (combs_per_job + 1)
        end_idx = start_idx + combs_per_job + 1
    else:
        start_idx = remainder * (combs_per_job + 1) + (slice_idx - remainder) * combs_per_job
        end_idx = start_idx + combs_per_job

    return list_all_combs[start_idx:end_idx]


# ============================================================================
# Utility
# ============================================================================


def log_gpu_memory_usage(logger=None):
    """Log current GPU memory usage for monitoring."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            msg = f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB"
        else:
            msg = "GPU not available"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    except ImportError:
        msg = "PyTorch not available for memory monitoring"
        if logger:
            logger.warning(msg)
        else:
            print(msg)


# ============================================================================
# CLI: print settings table with array ID ranges
# ============================================================================

if __name__ == "__main__":
    total_jobs = get_total_array_jobs()
    ranges = get_setting_array_ranges()

    print("=" * 100)
    print("TESTING SETTINGS AND SLURM ARRAY ID RANGES")
    print("=" * 100)
    print(f"Total settings (lines) : {len(LIST_SETTINGS)}")
    print(f"Total SLURM array jobs : {total_jobs}")
    print(
        f"JOBS_PER_REGULAR={JOBS_PER_REGULAR}  |  "
        f"JOBS_PER_ROBUSTNESS={JOBS_PER_ROBUSTNESS}  |  "
        f"JOBS_PER_GENERALIZATION={JOBS_PER_GENERALIZATION}"
    )
    print(
        f"SCENARIOS_PER_REGULAR={SCENARIOS_PER_REGULAR}  |  "
        f"SCENARIOS_PER_ROBUSTNESS={SCENARIOS_PER_ROBUSTNESS}  |  "
        f"SCENARIOS_PER_GENERALIZATION={SCENARIOS_PER_GENERALIZATION}"
    )
    print()

    row_data: list[tuple[str, str, str, str, str, str, str, str]] = []
    for i, (model_name, scenario, test_type, start, end) in enumerate(ranges):
        n_jobs = JOBS_PER_TEST_TYPE[test_type]
        n_combos = SCENARIOS_PER_TEST_TYPE[test_type]
        combos_per_job = n_combos / n_jobs
        row_data.append(
            (
                str(i),
                model_name,
                scenario,
                test_type,
                str(n_jobs),
                f"{start}-{end}",
                str(n_combos),
                f"{combos_per_job:.1f}",
            )
        )

    headers = ("Line", "Model", "Scenario", "Test Type", "Jobs", "Array Range", "Combos", "Combos/Job")
    widths = [max(len(header), *(len(row[idx]) for row in row_data)) for idx, header in enumerate(headers)]

    header = (
        f"{headers[0]:>{widths[0]}}  "
        f"{headers[1]:<{widths[1]}}  "
        f"{headers[2]:<{widths[2]}}  "
        f"{headers[3]:<{widths[3]}}  "
        f"{headers[4]:>{widths[4]}}  "
        f"{headers[5]:<{widths[5]}}  "
        f"{headers[6]:>{widths[6]}}  "
        f"{headers[7]:>{widths[7]}}"
    )
    print(header)
    print("-" * len(header))

    for row in row_data:
        print(
            f"{row[0]:>{widths[0]}}  "
            f"{row[1]:<{widths[1]}}  "
            f"{row[2]:<{widths[2]}}  "
            f"{row[3]:<{widths[3]}}  "
            f"{row[4]:>{widths[4]}}  "
            f"{row[5]:<{widths[5]}}  "
            f"{row[6]:>{widths[6]}}  "
            f"{row[7]:>{widths[7]}}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':>{widths[0]}}  {'':<{widths[1]}}  {'':<{widths[2]}}  {'':<{widths[3]}}  {total_jobs!s:>{widths[4]}}"
    )
    print()
    print("Usage:  sbatch --array=START-END scripts/testing.sh")
