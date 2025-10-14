"""Normalization utilities for complex CSI data processing.

This module provides comprehensive normalization functionality for Channel State Information (CSI)
data, which is inherently complex-valued. The normalization process is crucial for neural network
training stability and performance.

Key features:
- Separate normalization statistics for real and imaginary parts of complex data
- Persistent storage and loading of normalization statistics
- Support for both FDD (U2D) and TDD scenarios
- Batch processing with proper handling of complex tensor formats
- Denormalization for converting model outputs back to original scale

The normalization follows the standard z-score normalization: (x - mean) / std
but applied separately to real and imaginary components of complex CSI data.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import torch

from src.utils.data_utils import LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN, TOT_ANTENNAS, load_data
from src.utils.dirs import DIR_DATA


class NormalizationStats:
    """Container for complex data normalization statistics.

    This class encapsulates the normalization statistics (mean and standard deviation)
    for both real and imaginary parts of complex CSI data. It provides methods for
    serialization and deserialization to enable persistent storage of computed statistics.

    The separate treatment of real and imaginary parts is important because CSI data
    exhibits different statistical properties in these components, and proper normalization
    requires handling them independently.
    """

    def __init__(self, mean_r: torch.Tensor, std_r: torch.Tensor, mean_i: torch.Tensor, std_i: torch.Tensor):
        """Initialize normalization statistics for complex data.

        Args:
            mean_r (torch.Tensor): Mean of the real part across the training dataset
            std_r (torch.Tensor): Standard deviation of the real part
            mean_i (torch.Tensor): Mean of the imaginary part across the training dataset
            std_i (torch.Tensor): Standard deviation of the imaginary part

        """
        self.mean_r = mean_r
        self.std_r = std_r
        self.mean_i = mean_i
        self.std_i = std_i

    def to_dict(self) -> dict:
        """Convert normalization statistics to dictionary for serialization.

        Converts tensor statistics to scalar values for JSON/pickle serialization.
        This enables saving computed statistics to disk for later use.

        Returns:
            dict: Dictionary containing scalar values of all statistics

        """
        return {
            "mean_r": self.mean_r.item(),  # Convert tensor to scalar
            "std_r": self.std_r.item(),
            "mean_i": self.mean_i.item(),
            "std_i": self.std_i.item(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> NormalizationStats:
        """Create NormalizationStats instance from dictionary.

        Reconstructs the statistics object from serialized dictionary data,
        converting scalar values back to tensors.

        Args:
            data (dict): Dictionary containing serialized statistics

        Returns:
            NormalizationStats: Reconstructed statistics object

        """
        return cls(
            mean_r=torch.tensor(data["mean_r"]),  # Convert scalar back to tensor
            std_r=torch.tensor(data["std_r"]),
            mean_i=torch.tensor(data["mean_i"]),
            std_i=torch.tensor(data["std_i"]),
        )


def compute_normalization_stats(H_data: torch.Tensor, eps: float = 1e-8) -> NormalizationStats:
    """Compute normalization statistics from complex training data.

    Calculates mean and standard deviation for both real and imaginary parts
    of the complex CSI training data. These statistics are essential for
    z-score normalization which helps neural network training convergence.

    The function uses population standard deviation (unbiased=False) to ensure
    consistency with the normalization process during training and inference.

    Args:
        H_data (torch.Tensor): Complex-valued training data tensor
                              Shape: [batch_size, time_steps, num_features]
        eps (float, optional): Small epsilon value (unused in this function,
                              kept for API consistency). Defaults to 1e-8.

    Returns:
        NormalizationStats: Object containing computed statistics for both
                           real and imaginary parts

    Raises:
        ValueError: If H_data is not a complex tensor

    """
    if not H_data.is_complex():
        raise ValueError("H_data must be a complex tensor for CSI normalization")

    # Extract real and imaginary components
    H_r = H_data.real
    H_i = H_data.imag

    # Compute statistics across all dimensions (global normalization)
    mean_r = H_r.mean()  # Global mean of real part
    std_r = H_r.std(unbiased=False)  # Population std of real part
    mean_i = H_i.mean()  # Global mean of imaginary part
    std_i = H_i.std(unbiased=False)  # Population std of imaginary part

    # Log computed statistics for verification
    print(
        f"Computed complex normalization stats: "
        f"Real(mean={mean_r.item():.6f}, std={std_r.item():.6f}), "
        f"Imag(mean={mean_i.item():.6f}, std={std_i.item():.6f})"
    )

    return NormalizationStats(mean_r=mean_r, std_r=std_r, mean_i=mean_i, std_i=std_i)


# =============================================================================
# Normalization Statistics Loading and Management
# =============================================================================
# The following functions handle persistent storage and retrieval of
# normalization statistics, supporting both FDD and TDD scenarios.


def load_normalization_stats(dir_data: Path, is_U2D: bool) -> NormalizationStats:
    """Load normalization statistics from persistent storage.

    Loads previously computed and saved normalization statistics from disk.
    The statistics are scenario-specific (FDD vs TDD) and stored in separate
    directories to avoid cross-contamination between different channel models.

    Args:
        dir_data (Path): Base data directory containing statistics subdirectories
        is_U2D (bool): Scenario flag - True for FDD (U2D), False for TDD

    Returns:
        NormalizationStats: Loaded normalization statistics object

    Raises:
        FileNotFoundError: If the statistics file doesn't exist for the specified scenario

    """
    # Construct path to scenario-specific statistics file
    scenario_dir = "fdd" if is_U2D else "tdd"
    file_path = dir_data / "stats" / scenario_dir / "normalization_stats.pkl"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Normalization stats file not found: {file_path}. Please run normalization statistics computation first."
        )

    # Load serialized statistics from pickle file
    with open(file_path, "rb") as f:
        stats_dict = pickle.load(f)

    # Reconstruct NormalizationStats object from dictionary
    stats = NormalizationStats.from_dict(stats_dict)
    # Note: Logging commented out to avoid verbose output during frequent loading
    # logger.info(f"Loaded normalization statistics from {file_path}")

    return stats


def get_normalization_stats(is_U2D: bool) -> NormalizationStats:
    """Compute and save normalization statistics from training data.

    This function loads the complete training dataset for the specified scenario,
    computes normalization statistics, and saves them for future use. This is
    typically run once during data preprocessing.

    The function ensures proper directory structure and handles the complete
    data loading and statistics computation pipeline.

    Args:
        is_U2D (bool): Scenario flag - True for FDD (U2D), False for TDD

    Returns:
        NormalizationStats: Computed normalization statistics

    Note:
        This function loads the entire training dataset into memory, so ensure
        sufficient RAM is available. The computed statistics are automatically
        saved to disk for future loading via load_normalization_stats().

    """
    # Create output directory for scenario-specific statistics
    scenario_dir = "fdd" if is_U2D else "tdd"
    dir_output = Path(DIR_DATA) / "stats" / scenario_dir
    dir_output.mkdir(parents=True, exist_ok=True)

    # Load complete training dataset for statistics computation
    print(f"Loading training data for {scenario_dir.upper()} scenario...")
    H_hist = load_data(
        dir_data=Path(DIR_DATA),
        list_cm=LIST_CHANNEL_MODEL,  # All channel models
        list_ds=LIST_DELAY_SPREAD,  # All delay spreads
        list_ms=LIST_MIN_SPEED_TRAIN,  # Training speed ranges only
        is_train=True,  # Training data
        is_gen=False,  # Not generated/synthetic data
        is_hist=True,  # Historical data (input)
        is_U2D=is_U2D,  # Scenario selection
    )

    # Compute normalization statistics from loaded data
    print(f"Computing normalization statistics for {H_hist.shape[0]} samples...")
    stats = compute_normalization_stats(H_data=H_hist)

    # Save statistics to disk for future use
    stats_file = dir_output / "normalization_stats.pkl"
    with open(stats_file, "wb") as f:
        pickle.dump(stats.to_dict(), f)

    print(f"Normalization statistics saved to: {stats_file}")
    return stats


def normalize_input(
    H_hist: torch.Tensor,
    H_pred: torch.Tensor | None,
    eps: float = 1e-8,
    is_U2D: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize complex input tensors using precomputed statistics.

    Applies z-score normalization to complex CSI data using previously computed
    statistics. The normalization is applied separately to real and imaginary
    parts to preserve the complex structure while ensuring proper scaling.

    This function is typically called during data loading to ensure consistent
    input scaling for neural network training and inference.

    Args:
        H_hist (torch.Tensor): Historical CSI data tensor (must be complex)
                              Shape: [batch_size, time_steps, num_features]
        H_pred (torch.Tensor | None): Prediction target tensor (must be complex if provided)
                                     Shape: [batch_size, time_steps, num_features]
        eps (float, optional): Small epsilon to avoid division by zero. Defaults to 1e-8.
        is_U2D (bool, optional): Scenario flag - True for FDD, False for TDD. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            - H_hist_norm: Normalized historical data
            - H_pred_norm: Normalized prediction data (None if H_pred was None)

    Raises:
        ValueError: If input tensors are not complex-valued
        FileNotFoundError: If normalization statistics are not found for the specified scenario

    """
    # Validate input tensor types
    if not H_hist.is_complex():
        raise ValueError("H_hist must be a complex tensor for CSI normalization")

    if H_pred is not None and not H_pred.is_complex():
        raise ValueError("H_pred must be a complex tensor if provided")

    # Load scenario-specific normalization statistics
    stats = load_normalization_stats(dir_data=Path(DIR_DATA), is_U2D=is_U2D)

    # Normalize historical data
    H_hist_r = H_hist.real
    H_hist_i = H_hist.imag

    # Apply z-score normalization: (x - mean) / (std + eps)
    H_hist_r_norm = (H_hist_r - stats.mean_r) / (stats.std_r + eps)
    H_hist_i_norm = (H_hist_i - stats.mean_i) / (stats.std_i + eps)
    H_hist_norm = torch.complex(H_hist_r_norm, H_hist_i_norm)

    # Normalize prediction data if provided
    if H_pred is not None:
        H_pred_r = H_pred.real
        H_pred_i = H_pred.imag

        # Apply same normalization to prediction targets
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
    """Denormalize model outputs back to original CSI scale.

    Converts normalized model predictions and targets back to the original CSI scale
    using the inverse of the z-score normalization. This function handles both
    complex and real tensor formats, automatically converting real tensors back
    to complex format when needed.

    This is essential for:
    - Computing meaningful loss values in original scale
    - Evaluating model performance with interpretable metrics
    - Comparing predictions with ground truth data

    Args:
        pred_norm (torch.Tensor): Normalized model predictions
                                 (can be real or complex format)
        target_norm (torch.Tensor): Normalized target values
                                   (can be real or complex format)
        eps (float, optional): Small epsilon value (same as used in normalization).
                              Defaults to 1e-8.
        is_U2D (bool, optional): Scenario flag - True for FDD, False for TDD.
                                Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - pred_orig: Predictions in original CSI scale (complex format)
            - target_orig: Targets in original CSI scale (complex format)

    Note:
        The function automatically handles conversion from real tensor format
        (used by some models) back to complex format with proper shape restoration.

    """
    # Load the same normalization statistics used during normalization
    stats = load_normalization_stats(dir_data=Path(DIR_DATA), is_U2D=is_U2D)

    def _convert_to_complex_and_restore_shape(tensor: torch.Tensor) -> torch.Tensor:
        """Convert real tensor format back to complex and restore original shape.

        Some models output real tensors where complex data is represented as
        concatenated real and imaginary parts. This function converts them back
        to proper complex format.
        """
        if tensor.is_complex():
            # Already in complex format, no conversion needed
            return tensor
        else:
            # Handle real tensor format from models using separate antenna processing
            # Input format: [batch_size * num_antennas, time_steps, num_subcarriers*2]
            # Output format: [batch_size, num_antennas, time_steps, num_subcarriers] complex

            batch_times_antennas, time_steps, features = tensor.shape
            assert features % 2 == 0, f"Last dimension must be even for real-to-complex conversion, got {features}"

            # Calculate original dimensions
            num_subcarriers = features // 2
            batch_size = batch_times_antennas // TOT_ANTENNAS

            # Reshape to separate real and imaginary parts
            # [batch_size, num_antennas, time_steps, num_subcarriers, 2]
            tensor_reshaped = tensor.view(batch_size, TOT_ANTENNAS, time_steps, num_subcarriers, 2)

            # Convert to complex format: [batch_size, num_antennas, time_steps, num_subcarriers]
            return torch.complex(tensor_reshaped[..., 0], tensor_reshaped[..., 1])

    # Ensure both tensors are in complex format with correct shapes
    pred_complex = _convert_to_complex_and_restore_shape(pred_norm)
    target_complex = _convert_to_complex_and_restore_shape(target_norm)

    # Denormalize predictions using inverse z-score: x_orig = x_norm * std + mean
    pred_r_norm = pred_complex.real
    pred_i_norm = pred_complex.imag
    pred_r_orig = pred_r_norm * (stats.std_r + eps) + stats.mean_r
    pred_i_orig = pred_i_norm * (stats.std_i + eps) + stats.mean_i
    pred_orig = torch.complex(pred_r_orig, pred_i_orig)

    # Denormalize targets using the same transformation
    target_r_norm = target_complex.real
    target_i_norm = target_complex.imag
    target_r_orig = target_r_norm * (stats.std_r + eps) + stats.mean_r
    target_i_orig = target_i_norm * (stats.std_i + eps) + stats.mean_i
    target_orig = torch.complex(target_r_orig, target_i_orig)

    return pred_orig, target_orig


# =============================================================================
# Main Execution: Compute and Display Normalization Statistics
# =============================================================================
# This section is executed when the module is run directly, typically during
# initial data preprocessing to compute and save normalization statistics.

if __name__ == "__main__":
    print("Computing normalization statistics for both TDD and FDD scenarios...")
    print("=" * 70)

    # Compute statistics for TDD scenario
    print("\n[1/2] Processing TDD scenario...")
    tdd_stats = get_normalization_stats(is_U2D=False)

    # Compute statistics for FDD scenario
    print("\n[2/2] Processing FDD scenario...")
    fdd_stats = get_normalization_stats(is_U2D=True)

    # Display computed statistics for verification
    print("\n" + "=" * 70)
    print("COMPUTED NORMALIZATION STATISTICS:")
    print("=" * 70)
    print(f"\nTDD Statistics: {tdd_stats.to_dict()}")
    print(f"FDD Statistics: {fdd_stats.to_dict()}")
    print("\nStatistics computation completed successfully!")
