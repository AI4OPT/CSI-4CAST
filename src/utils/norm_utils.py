from __future__ import annotations

import pickle
from pathlib import Path

import torch

from src.utils.data_utils import LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN, TOT_ANTENNAS, load_data
from src.utils.dirs import DIR_DATA


class NormalizationStats:
    """Container for complex data normalization statistics."""

    def __init__(self, mean_r: torch.Tensor, std_r: torch.Tensor, mean_i: torch.Tensor, std_i: torch.Tensor):
        """Initialize normalization statistics for complex data.

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
    """Compute normalization statistics from complex training data.

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
    """Load normalization statistics from file.

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


def get_normalization_stats(is_U2D: bool) -> NormalizationStats:
    """Get normalization statistics from file.

    Args:
        dir_data: Path to the data directory
        is_U2D: True if U2D, False if TDD

    """
    dir_output = Path(DIR_DATA) / "stats" / ("fdd" if is_U2D else "tdd")
    dir_output.mkdir(parents=True, exist_ok=True)

    H_hist = load_data(
        dir_data=Path(DIR_DATA),
        list_cm=LIST_CHANNEL_MODEL,
        list_ds=LIST_DELAY_SPREAD,
        list_ms=LIST_MIN_SPEED_TRAIN,
        is_train=True,
        is_gen=False,
        is_hist=True,
        is_U2D=is_U2D,
    )

    stats = compute_normalization_stats(H_data=H_hist)
    with open(dir_output / "normalization_stats.pkl", "wb") as f:
        pickle.dump(stats.to_dict(), f)

    return stats


def normalize_input(
    H_hist: torch.Tensor,
    H_pred: torch.Tensor | None,
    eps: float = 1e-8,
    is_U2D: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize complex input tensors using precomputed statistics

    Args:
        H_hist: History tensor to normalize (must be complex)
        H_pred: Prediction tensor to normalize (can be None, must be complex if provided)
        eps: Small epsilon to avoid division by zero
        is_U2D: FDD scenario is_U2D=True, TDD scenario is_U2D=False

    Returns:
        H_hist_norm: Normalized history tensor
        H_pred_norm: Normalized prediction tensor

    """
    # Validate that input tensors are complex
    if not H_hist.is_complex():
        raise ValueError("H_hist must be a complex tensor")

    if H_pred is not None and not H_pred.is_complex():
        raise ValueError("H_pred must be a complex tensor if provided")

    # load the normalization stats
    stats = load_normalization_stats(dir_data=Path(DIR_DATA), is_U2D=is_U2D)

    H_hist_r = H_hist.real
    H_hist_i = H_hist.imag

    H_hist_r_norm = (H_hist_r - stats.mean_r) / (stats.std_r + eps)
    H_hist_i_norm = (H_hist_i - stats.mean_i) / (stats.std_i + eps)
    H_hist_norm = torch.complex(H_hist_r_norm, H_hist_i_norm)

    if H_pred is not None:
        H_pred_r = H_pred.real
        H_pred_i = H_pred.imag

        H_pred_r_norm = (H_pred_r - stats.mean_r) / (stats.std_r + eps)
        H_pred_i_norm = (H_pred_i - stats.mean_i) / (stats.std_i + eps)
        H_pred_norm = torch.complex(H_pred_r_norm, H_pred_i_norm)
    else:
        H_pred_norm = None

    return H_hist_norm, H_pred_norm


def denormalize_output(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    eps: float = 1e-8,
    is_U2D: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denormalize tensors back to original scale, handling both real and complex formats

    Args:
        pred_norm: Normalized prediction tensor (can be real or complex)
        target_norm: Normalized target tensor (can be real or complex)
        eps: Small epsilon (same as used in normalization)
        is_U2D: FDD scenario is_U2D=True, TDD scenario is_U2D=False

    Returns:
        pred_orig: Prediction tensor in original scale (complex)
        target_orig: Target tensor in original scale (complex)

    """
    # Load the same normalization stats used for normalization
    stats = load_normalization_stats(dir_data=Path(DIR_DATA), is_U2D=is_U2D)

    # Helper function to convert real tensor back to complex if needed and restore original shape
    def _convert_to_complex_and_restore_shape(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_complex():
            return tensor
        else:
            # Handle real tensor format from collect_fn_separate_antennas
            # Input: [batch_size * num_antennas, time_steps, num_subcarriers*2] real
            # Output: [batch_size, num_antennas, time_steps, num_subcarriers] complex

            batch_times_antennas, time_steps, features = tensor.shape
            assert features % 2 == 0, f"Last dimension must be even for real tensor conversion, got {features}"

            num_subcarriers = features // 2
            batch_size = batch_times_antennas // TOT_ANTENNAS

            # Reshape to [batch_size, num_antennas, time_steps, num_subcarriers, 2]
            tensor_reshaped = tensor.view(batch_size, TOT_ANTENNAS, time_steps, num_subcarriers, 2)

            # Convert to complex: [batch_size, num_antennas, time_steps, num_subcarriers]
            return torch.complex(tensor_reshaped[..., 0], tensor_reshaped[..., 1])

    # Convert to complex format and restore original shape if needed
    pred_complex = _convert_to_complex_and_restore_shape(pred_norm)
    target_complex = _convert_to_complex_and_restore_shape(target_norm)

    # Denormalize prediction
    pred_r_norm = pred_complex.real
    pred_i_norm = pred_complex.imag
    pred_r_orig = pred_r_norm * (stats.std_r + eps) + stats.mean_r
    pred_i_orig = pred_i_norm * (stats.std_i + eps) + stats.mean_i
    pred_orig = torch.complex(pred_r_orig, pred_i_orig)

    # Denormalize target
    target_r_norm = target_complex.real
    target_i_norm = target_complex.imag
    target_r_orig = target_r_norm * (stats.std_r + eps) + stats.mean_r
    target_i_orig = target_i_norm * (stats.std_i + eps) + stats.mean_i
    target_orig = torch.complex(target_r_orig, target_i_orig)

    return pred_orig, target_orig


if __name__ == "__main__":
    tdd_stats = get_normalization_stats(is_U2D=False)
    fdd_stats = get_normalization_stats(is_U2D=True)

    print(f"TDD stats: {tdd_stats.to_dict()}")
    print(f"FDD stats: {fdd_stats.to_dict()}")
