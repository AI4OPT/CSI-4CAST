from __future__ import annotations

from pathlib import Path

import torch


class NormalizationStats:
    """Container for complex data normalization statistics."""

    def __init__(self, mean_r: torch.Tensor, std_r: torch.Tensor, mean_i: torch.Tensor, std_i: torch.Tensor):
        """
        Initialize normalization statistics for complex data.

        Args:
            mean_r, std_r: Real part statistics (tensors)
            mean_i, std_i: Imaginary part statistics (tensors)
        """
        self.mean_r = mean_r
        self.std_r = std_r
        self.mean_i = mean_i
        self.std_i = std_i

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean_r": self.mean_r.item(),
            "std_r": self.std_r.item(),
            "mean_i": self.mean_i.item(),
            "std_i": self.std_i.item(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> NormalizationStats:
        """Create from dictionary."""
        return cls(
            mean_r=torch.tensor(data["mean_r"]),
            std_r=torch.tensor(data["std_r"]),
            mean_i=torch.tensor(data["mean_i"]),
            std_i=torch.tensor(data["std_i"]),
        )


def compute_normalization_stats(H_data: torch.Tensor, eps: float = 1e-8) -> NormalizationStats:
    """
    Compute normalization statistics from complex training data.

    Args:
        H_data: Training data tensor (must be complex)
        eps: Small epsilon to avoid division by zero

    Returns:
        NormalizationStats object containing computed statistics
    """
    if not H_data.is_complex():
        raise ValueError("H_data must be a complex tensor")

    # Complex data: compute separate stats for real and imaginary parts
    H_r = H_data.real
    H_i = H_data.imag

    mean_r = H_r.mean()
    std_r = H_r.std(unbiased=False)
    mean_i = H_i.mean()
    std_i = H_i.std(unbiased=False)

    print(
        f"Computed complex normalization stats: "
        f"Real(mean={mean_r.item():.6f}, std={std_r.item():.6f}), "
        f"Imag(mean={mean_i.item():.6f}, std={std_i.item():.6f})"
    )

    return NormalizationStats(mean_r=mean_r, std_r=std_r, mean_i=mean_i, std_i=std_i)


"""
load the normalization stats
"""


def load_normalization_stats(dir_data: Path, is_U2D: bool) -> NormalizationStats:
    """
    Load normalization statistics from file.

    Args:
        file_path: Path to load the statistics from

    Returns:
        NormalizationStats object
    """
    file_path = dir_data / "stats" / ("fdd" if is_U2D else "tdd") / "normalization_stats.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Normalization stats file not found: {file_path}")

    with open(file_path, "rb") as f:
        stats_dict = pickle.load(f)

    stats = NormalizationStats.from_dict(stats_dict)
    # logger.info(f"Loaded normalization statistics from {file_path}")

    return stats


if __name__ == "__main__":
    import pickle

    from src.utils.data_utils import LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN, load_data
    from src.utils.dirs import DIR_DATA

    output_dir = Path(DIR_DATA) / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    """
    compute the normalization stats for TDD
    """

    tdd_dir = output_dir / "tdd"
    tdd_dir.mkdir(parents=True, exist_ok=True)

    H_hist = load_data(
        dir_data=Path(DIR_DATA),
        list_cm=LIST_CHANNEL_MODEL,
        list_ds=LIST_DELAY_SPREAD,
        list_ms=LIST_MIN_SPEED_TRAIN,
        is_train=True,
        is_gen=False,
        is_hist=True,
        is_U2D=False,
    )

    tdd_stats = compute_normalization_stats(H_data=H_hist)
    with open(tdd_dir / "normalization_stats.pkl", "wb") as f:
        pickle.dump(tdd_stats.to_dict(), f)

    """
    compute the normalization stats for FDD
    """

    fdd_dir = output_dir / "fdd"
    fdd_dir.mkdir(parents=True, exist_ok=True)

    H_hist = load_data(
        dir_data=Path(DIR_DATA),
        list_cm=LIST_CHANNEL_MODEL,
        list_ds=LIST_DELAY_SPREAD,
        list_ms=LIST_MIN_SPEED_TRAIN,
        is_train=True,
        is_gen=False,
        is_hist=True,
        is_U2D=True,
    )

    fdd_stats = compute_normalization_stats(H_data=H_hist)
    with open(fdd_dir / "normalization_stats.pkl", "wb") as f:
        pickle.dump(fdd_stats.to_dict(), f)
