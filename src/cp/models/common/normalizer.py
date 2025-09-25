import torch


def batch_normalizer(x: torch.Tensor, eps: float = 1e-6):
    """
    Standardize `x` by batch-wise mean and std.
    Args:
      x: input tensor.
      eps: small constant to avoid division by zero.
    Returns:
      x_norm: standardized tensor
      mean:   tensor of means (broadcastable to x)
      std:    tensor of stds  (broadcastable to x)
    """

    mean = x.mean()
    std = x.std(unbiased=False)
    std = std.clamp(min=eps)

    x_norm = (x - mean) / std
    return x_norm, mean, std


def batch_denormalize(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Reverse the standardization: x = x_norm * std + mean
    """
    return x_norm * std + mean
