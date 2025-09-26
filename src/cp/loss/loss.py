import torch
from einops import rearrange
from torch import nn

from src.utils.data_utils import TOT_ANTENNAS


class NMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def nmse(self, x_hat, x):
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
        """Here the step means the prediction step
        - by_step: if True, return the NMSE by step, otherwise return the average NMSE
        - step_dim: the dimension of the step, auto-detected if None
          For [batch_size, num_antennas, time_step, num_subcarriers]: step_dim=2
          For [batch_size * num_antennas, time_step, num_subcarriers*2]: step_dim=1
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
            x_dim = x_hat.dim()
            axis_to_reduce = tuple(i for i in range(x_dim) if i != step_dim)
            return torch.mean(nmse, dim=axis_to_reduce)
        else:
            return torch.mean(nmse)


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def mse(self, x_hat, x):
        # Handle both complex and real tensors
        if torch.is_complex(x):
            mse = torch.sum((x - x_hat).abs() ** 2, dim=-1, keepdim=True)
        else:
            mse = torch.sum((x - x_hat) ** 2, dim=-1, keepdim=True)
        return mse

    def forward(self, x_hat, x, by_step: bool = False, step_dim: int | None = None):
        """Here the step means the prediction step
        - by_step: if True, return the MSE by step, otherwise return the average MSE
        - step_dim: the dimension of the step, auto-detected if None
          For [batch_size, num_antennas, time_step, num_subcarriers]: step_dim=2
          For [batch_size * num_antennas, time_step, num_subcarriers*2]: step_dim=1
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
            x_dim = x_hat.dim()
            axis_to_reduce = tuple(i for i in range(x_dim) if i != step_dim)
            return torch.mean(mse, dim=axis_to_reduce)
        else:
            return torch.mean(mse)


class HuberLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, x_hat, x, by_step: bool = False, step_dim: int = 1):
        huber = self.loss_fn(x_hat, x)
        if by_step:
            x_dim = x_hat.dim()
            axis_to_reduce = tuple(i for i in range(x_dim) if x != step_dim)
            return torch.mean(huber, dim=axis_to_reduce)
        else:
            return torch.mean(huber)


class LOSS:
    NMSE = NMSELoss
    MSE = MSELoss
    Huber = HuberLoss


class SELoss(nn.Module):
    def __init__(self, SNR: int = 10, **kwargs):
        super().__init__()
        self.SNR = SNR
        self.sigma2 = 10 ** (-SNR / 10)

    def forward(self, x_hat, x, by_step: bool = False):
        se_pred = self._compute_se(x_hat, x)
        se_true = self._compute_se(x, x)
        if by_step:
            se_pred = se_pred.mean(dim=0)
            se_true = se_true.mean(dim=0)
        else:
            se_pred = se_pred.mean()
            se_true = se_true.mean()
        return se_pred, se_true

    def _compute_se(self, x_hat, x):
        """Handle different input formats and convert to complex tensors for SE computation
        Supports:
        - [batch_size, num_antennas, pred_len, num_subcarriers] complex (new format)
        - [batch_size * num_antennas, pred_len, num_subcarriers*2] real (old format)

        """
        if torch.is_complex(x_hat):
            # New complex tensor format: [batch_size, num_antennas, pred_len, num_subcarriers]
            if x_hat.dim() == 4:  # [batch_size, num_antennas, pred_len, num_subcarriers] complex
                # Rearrange to [batch_size, pred_len, num_antennas, num_subcarriers]
                x_hat = x_hat.permute(0, 2, 1, 3)
                x = x.permute(0, 2, 1, 3)
            else:
                raise ValueError(f"Unsupported complex tensor shape: {x_hat.shape}")

        elif x_hat.dim() == 3:  # [batch_size * num_antennas, pred_len, num_subcarriers*2] real tensor
            # [batch_size, pred_len, num_antennas, num_subcarriers*2]
            x_hat = rearrange(x_hat, "(b nt) l k -> b l nt k", nt=TOT_ANTENNAS)
            # [batch_size, pred_len, num_antennas, num_subcarriers*2]
            x = rearrange(x, "(b nt) l k -> b l nt k", nt=TOT_ANTENNAS)

            # [batch_size, pred_len, num_antennas, num_subcarriers] complex tensor
            x_hat_pair = rearrange(x_hat, "b l nt (k o) -> b l nt k o", o=2)
            x_hat = torch.complex(x_hat_pair[..., 0], x_hat_pair[..., 1])

            # [batch_size, pred_len, num_antennas, num_subcarriers] complex tensor
            x_pair = rearrange(x, "b l nt (k o) -> b l nt k o", o=2)
            x = torch.complex(x_pair[..., 0], x_pair[..., 1])

        else:
            raise ValueError(f"Unsupported tensor shape: {x_hat.shape}")

        inner_product = torch.sum(torch.conj(x_hat) * x, dim=2)  # batch_size, pred_len, num_subcarriers
        power_hat = torch.sum(x_hat.abs().pow(2), dim=2)  # batch_size, pred_len, num_subcarriers

        numerator = inner_product.abs().pow(2)
        denominator = power_hat * self.sigma2
        se_k = torch.log2(1 + numerator / denominator)
        se = se_k.mean(dim=-1)

        return se
