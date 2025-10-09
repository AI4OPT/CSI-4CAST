"""Loss Functions for CSI Prediction Models.

This module implements various loss functions specifically designed for Channel State Information (CSI)
prediction tasks. The loss functions handle both complex and real-valued tensors and support different
tensor formats used throughout the framework.

Key Features:
- Support for complex-valued CSI data
- Flexible tensor format handling (4D complex, 3D real)
- Step-wise loss computation for temporal analysis
- Automatic dimension detection and reduction
- Spectral Efficiency (SE) computation for communication performance evaluation

Loss Functions:
- NMSELoss: Normalized Mean Squared Error
- MSELoss: Mean Squared Error
- HuberLoss: Huber Loss (robust to outliers)
- SELoss: Spectral Efficiency Loss

Usage:
    from src.cp.loss.loss import LOSS

    # Initialize loss function
    criterion = getattr(LOSS, "NMSE")()

    # Compute loss
    loss = criterion(predictions, targets)

    # Compute step-wise loss
    step_losses = criterion(predictions, targets, by_step=True)
"""

from einops import rearrange
import torch
from torch import nn

from src.utils.data_utils import TOT_ANTENNAS


class NMSELoss(nn.Module):
    """Normalized Mean Squared Error (NMSE) Loss.

    NMSE is commonly used in CSI prediction tasks as it normalizes the MSE by the signal power,
    making it scale-invariant and suitable for comparing performance across different scenarios.

    Formula: NMSE = E[|x - x_hat|²] / E[|x|²]

    Supports both complex and real-valued tensors with automatic format detection.
    """

    def __init__(self, **kwargs):
        """Initialize NMSE loss function."""
        super().__init__()

    def nmse(self, x_hat, x):
        """Compute NMSE between predictions and targets.

        Args:
            x_hat (torch.Tensor): Predicted values
            x (torch.Tensor): Target values

        Returns:
            torch.Tensor: NMSE values with same shape as input (except last dimension)

        """
        # Handle both complex and real tensors
        if torch.is_complex(x):
            power = torch.sum(x.abs() ** 2, dim=-1, keepdim=True)
            mse = torch.sum((x - x_hat).abs() ** 2, dim=-1, keepdim=True)
        else:
            power = torch.sum(x**2, dim=-1, keepdim=True)
            mse = torch.sum((x - x_hat) ** 2, dim=-1, keepdim=True)
        nmse = mse / power
        return nmse

    def forward(self, x_hat, x, by_step: bool = False, step_dim: int | None = None):
        """Forward pass of NMSE loss.

        Args:
            x_hat (torch.Tensor): Predicted CSI values
            x (torch.Tensor): Target CSI values
            by_step (bool): If True, return NMSE for each prediction step separately
            step_dim (int, optional): Dimension representing time steps. Auto-detected if None.
                For [batch_size, num_antennas, time_step, num_subcarriers]: step_dim=2
                For [batch_size * num_antennas, time_step, num_subcarriers*2]: step_dim=1

        Returns:
            torch.Tensor: NMSE loss value(s)
                - If by_step=False: Single scalar loss value
                - If by_step=True: Tensor with loss for each time step

        """
        if step_dim is None:
            # Auto-detect step dimension based on tensor shape
            if x_hat.dim() == 4:  # [batch_size, num_antennas, time_step, num_subcarriers]
                step_dim = 2
            elif x_hat.dim() == 3:  # [batch_size * num_antennas, time_step, num_subcarriers*2]
                step_dim = 1
            else:
                step_dim = 1  # fallback

        nmse = self.nmse(x_hat, x)

        if by_step:
            # Return step-wise NMSE by averaging over all dimensions except step_dim
            x_dim = x_hat.dim()
            axis_to_reduce = tuple(i for i in range(x_dim) if i != step_dim)
            return torch.mean(nmse, dim=axis_to_reduce)
        else:
            # Return overall average NMSE
            return torch.mean(nmse)


class MSELoss(nn.Module):
    """Mean Squared Error (MSE) Loss.

    Standard MSE loss function that handles both complex and real-valued tensors.
    Unlike NMSE, this is not normalized by signal power.

    Formula: MSE = E[|x - x_hat|²]

    Supports both complex and real-valued tensors with automatic format detection.
    """

    def __init__(self, **kwargs):
        """Initialize MSE loss function."""
        super().__init__()

    def mse(self, x_hat, x):
        """Compute MSE between predictions and targets.

        Args:
            x_hat (torch.Tensor): Predicted values
            x (torch.Tensor): Target values

        Returns:
            torch.Tensor: MSE values with same shape as input (except last dimension)

        """
        # Handle both complex and real tensors
        if torch.is_complex(x):
            mse = torch.sum((x - x_hat).abs() ** 2, dim=-1, keepdim=True)
        else:
            mse = torch.sum((x - x_hat) ** 2, dim=-1, keepdim=True)
        return mse

    def forward(self, x_hat, x, by_step: bool = False, step_dim: int | None = None):
        """Forward pass of MSE loss.

        Args:
            x_hat (torch.Tensor): Predicted CSI values
            x (torch.Tensor): Target CSI values
            by_step (bool): If True, return MSE for each prediction step separately
            step_dim (int, optional): Dimension representing time steps. Auto-detected if None.
                For [batch_size, num_antennas, time_step, num_subcarriers]: step_dim=2
                For [batch_size * num_antennas, time_step, num_subcarriers*2]: step_dim=1

        Returns:
            torch.Tensor: MSE loss value(s)
                - If by_step=False: Single scalar loss value
                - If by_step=True: Tensor with loss for each time step

        """
        if step_dim is None:
            # Auto-detect step dimension based on tensor shape
            if x_hat.dim() == 4:  # [batch_size, num_antennas, time_step, num_subcarriers]
                step_dim = 2
            elif x_hat.dim() == 3:  # [batch_size * num_antennas, time_step, num_subcarriers*2]
                step_dim = 1
            else:
                step_dim = 1  # fallback

        mse = self.mse(x_hat, x)

        if by_step:
            # Return step-wise MSE by averaging over all dimensions except step_dim
            x_dim = x_hat.dim()
            axis_to_reduce = tuple(i for i in range(x_dim) if i != step_dim)
            return torch.mean(mse, dim=axis_to_reduce)
        else:
            # Return overall average MSE
            return torch.mean(mse)


class HuberLoss(nn.Module):
    """Huber Loss (Smooth L1 Loss).

    Huber loss is robust to outliers, combining the best properties of L1 and L2 losses.
    It's quadratic for small errors and linear for large errors.

    Args:
        beta (float): The threshold at which to change between L1 and L2 loss. Default: 1.0

    """

    def __init__(self, beta=1.0):
        """Initialize Huber loss function.

        Args:
            beta (float): Threshold parameter for switching between L1 and L2 behavior

        """
        super().__init__()
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, x_hat, x, by_step: bool = False, step_dim: int = 1):
        """Forward pass of Huber loss.

        Args:
            x_hat (torch.Tensor): Predicted values
            x (torch.Tensor): Target values
            by_step (bool): If True, return loss for each prediction step separately
            step_dim (int): Dimension representing time steps. Default: 1

        Returns:
            torch.Tensor: Huber loss value(s)
                - If by_step=False: Single scalar loss value
                - If by_step=True: Tensor with loss for each time step

        """
        huber = self.loss_fn(x_hat, x)

        if by_step:
            # Return step-wise Huber loss by averaging over all dimensions except step_dim
            x_dim = x_hat.dim()
            axis_to_reduce = tuple(i for i in range(x_dim) if i != step_dim)  # Fixed bug: was 'x != step_dim'
            return torch.mean(huber, dim=axis_to_reduce)
        else:
            # Return overall average Huber loss
            return torch.mean(huber)


class LOSS:
    """Registry class for all available loss functions.

    Usage:
        criterion = getattr(LOSS, "NMSE")()
        criterion = getattr(LOSS, "MSE")()
        criterion = getattr(LOSS, "Huber")(beta=0.5)
    """

    NMSE = NMSELoss
    MSE = MSELoss
    Huber = HuberLoss


class SELoss(nn.Module):
    """Spectral Efficiency (SE) Loss.

    Computes the spectral efficiency metric for communication systems, which measures
    the data transmission rate per unit bandwidth. This is particularly important for
    evaluating CSI prediction performance in terms of actual communication performance.

    Formula: SE = log₂(1 + |h^H * h_hat|² / (||h_hat||² * σ²))

    where:
    - h is the true channel
    - h_hat is the predicted channel
    - σ² is the noise variance (computed from SNR)

    Args:
        SNR (int): Signal-to-noise ratio in dB. Default: 10

    """

    def __init__(self, SNR: int = 10, **kwargs):
        """Initialize SE loss function.

        Args:
            SNR (int): Signal-to-noise ratio in dB for noise variance calculation
            **kwargs: Additional arguments (unused)

        """
        super().__init__()
        self.SNR = SNR
        self.sigma2 = 10 ** (-SNR / 10)  # Convert SNR from dB to linear scale

    def forward(self, x_hat, x, by_step: bool = False):
        """Forward pass of SE loss.

        Args:
            x_hat (torch.Tensor): Predicted CSI values
            x (torch.Tensor): Target CSI values
            by_step (bool): If True, return SE for each prediction step separately

        Returns:
            tuple: (se_pred, se_true) - Spectral efficiency using predicted and true channels

        """
        se_pred = self._compute_se(x_hat, x)  # SE with predicted channel
        se_true = self._compute_se(x, x)  # SE with true channel (upper bound)

        if by_step:
            # Return step-wise SE by averaging over batch dimension
            se_pred = se_pred.mean(dim=0)
            se_true = se_true.mean(dim=0)
        else:
            # Return overall average SE
            se_pred = se_pred.mean()
            se_true = se_true.mean()

        return se_pred, se_true

    def _compute_se(self, x_hat, x):
        """Compute spectral efficiency for given channel estimates.

        Handles different input formats and converts to complex tensors for SE computation.
        Supported formats:
        - [batch_size, num_antennas, pred_len, num_subcarriers] complex (complex-valued format)
        - [batch_size * num_antennas, pred_len, num_subcarriers*2] real (real-valued format)

        Args:
            x_hat (torch.Tensor): Channel estimates (predicted or true)
            x (torch.Tensor): True channel values

        Returns:
            torch.Tensor: Spectral efficiency values [batch_size, pred_len]

        """
        if torch.is_complex(x_hat):
            # Complex-valued tensor format: [batch_size, num_antennas, pred_len, num_subcarriers]
            if x_hat.dim() == 4:  # [batch_size, num_antennas, pred_len, num_subcarriers] complex
                # Rearrange to [batch_size, pred_len, num_antennas, num_subcarriers] for SE computation
                x_hat = x_hat.permute(0, 2, 1, 3)
                x = x.permute(0, 2, 1, 3)
            else:
                raise ValueError(f"Unsupported complex tensor shape: {x_hat.shape}")

        elif x_hat.dim() == 3:  # [batch_size * num_antennas, pred_len, num_subcarriers*2] real tensor
            # Reshape to [batch_size, pred_len, num_antennas, num_subcarriers*2]
            x_hat = rearrange(x_hat, "(b nt) l k -> b l nt k", nt=TOT_ANTENNAS)
            x = rearrange(x, "(b nt) l k -> b l nt k", nt=TOT_ANTENNAS)

            # Convert real-valued representation to complex-valued tensor
            # [batch_size, pred_len, num_antennas, num_subcarriers] complex tensor
            x_hat_pair = rearrange(x_hat, "b l nt (k o) -> b l nt k o", o=2)
            x_hat = torch.complex(x_hat_pair[..., 0], x_hat_pair[..., 1])

            x_pair = rearrange(x, "b l nt (k o) -> b l nt k o", o=2)
            x = torch.complex(x_pair[..., 0], x_pair[..., 1])

        else:
            raise ValueError(f"Unsupported tensor shape: {x_hat.shape}")

        # Compute spectral efficiency
        # Inner product: h^H * h_hat for each subcarrier
        inner_product = torch.sum(torch.conj(x_hat) * x, dim=2)  # [batch_size, pred_len, num_subcarriers]

        # Power of predicted channel: ||h_hat||²
        power_hat = torch.sum(x_hat.abs().pow(2), dim=2)  # [batch_size, pred_len, num_subcarriers]

        # SE formula: log₂(1 + |h^H * h_hat|² / (||h_hat||² * σ²))
        numerator = inner_product.abs().pow(2)
        denominator = power_hat * self.sigma2
        se_k = torch.log2(1 + numerator / denominator)  # SE per subcarrier

        # Average over subcarriers to get SE per time step
        se = se_k.mean(dim=-1)  # [batch_size, pred_len]

        return se
