"""No-Prediction Baseline Model for CSI Prediction.

This module implements a simple baseline model that performs no actual prediction.
Instead, it repeats the last observed CSI value for all future time steps. This
serves as a naive baseline for comparison with more sophisticated prediction models.

The model is useful for:
- Establishing baseline performance metrics
- Validating that more complex models actually learn useful patterns
- Providing a simple reference point for model comparison
- Testing the evaluation pipeline with a known-simple model

Usage:
    from src.cp.models.baseline_models.np import NOPREDICTMODEL

    model = NOPREDICTMODEL(pred_len=4)
    predictions = model(historical_csi)  # Simply repeats last observation
"""

import torch.nn as nn

from src.utils.data_utils import PRED_LEN


class NOPREDICTMODEL(nn.Module):
    """No-Prediction Baseline Model.

    This model implements the simplest possible "prediction" strategy: repeating
    the last observed CSI value for all future time steps. It serves as a naive
    baseline to establish the minimum performance threshold that any learning-based
    model should exceed.

    The model assumes that the channel remains constant, which is obviously unrealistic
    but provides a useful lower bound for comparison. In scenarios with slow-fading
    channels or very short prediction horizons, this baseline might be surprisingly
    effective.

    Args:
        pred_len (int): Number of future time steps to predict. Default: PRED_LEN
        *args: Additional positional arguments (ignored, for compatibility)
        **kwargs: Additional keyword arguments (ignored, for compatibility)

    Attributes:
        is_separate_antennas (bool): Always True, indicating antenna-wise processing
        name (str): Model identifier "NP" (No Prediction)
        pred_len (int): Number of prediction steps

    """

    def __init__(self, pred_len=PRED_LEN, *args, **kwargs):
        """Initialize the No-Prediction baseline model.

        Args:
            pred_len (int): Number of future time steps to predict
            *args: Additional arguments (ignored for compatibility with other models)
            **kwargs: Additional keyword arguments (ignored for compatibility)

        """
        super().__init__()

        # Model configuration
        self.is_separate_antennas = True  # Process antennas separately
        self.name = "NP"  # Model identifier

        # Prediction configuration
        self.pred_len = pred_len  # Number of time steps to repeat

    def __str__(self) -> str:
        """Return string representation of the model."""
        return self.name

    def forward(self, x):
        """Forward pass: repeat the last observed CSI value.

        This method implements the core "prediction" logic by simply repeating
        the last time step of the input sequence for the specified number of
        prediction steps. This represents a zero-order hold assumption.

        Args:
            x (torch.Tensor): Input CSI sequence with shape (batch_size, seq_len, feature_dim)
                - batch_size: Number of samples in the batch
                - seq_len: Number of historical time steps
                - feature_dim: CSI feature dimension (e.g., subcarriers × 2 for complex)

        Returns:
            torch.Tensor: Predicted CSI sequence with shape (batch_size, pred_len, feature_dim)
                The last time step from input is repeated pred_len times

        Example:
            >>> model = NOPREDICTMODEL(pred_len=4)
            >>> input_seq = torch.randn(32, 16, 600)  # 32 samples, 16 history, 600 features
            >>> predictions = model(input_seq)  # Shape: (32, 4, 600)

        """
        # Extract the last time step from the input sequence
        # Using [:, [-1], :] preserves the time dimension for proper broadcasting
        last_step = x[:, [-1], :]  # Shape: (batch_size, 1, feature_dim)

        # Repeat the last step for all prediction time steps
        # This implements a zero-order hold prediction strategy
        return last_step.repeat(1, self.pred_len, 1)  # Shape: (batch_size, pred_len, feature_dim)
