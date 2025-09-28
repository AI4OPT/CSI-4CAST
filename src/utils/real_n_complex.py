"""Utilities for converting between real and complex tensor representations.

This module provides functions to convert between different tensor formats used
for representing complex-valued data in neural networks:

1. Complex tensors: Native PyTorch complex tensors (torch.complex64/128)
2. Real flat tensors: Real tensors where complex data is represented as
   concatenated real and imaginary parts in the last dimension

These conversions are essential when working with models that don't natively
support complex operations, allowing complex CSI data to be processed using
real-valued neural network layers.

The module uses einops for efficient tensor rearrangement operations.
"""

import torch
from einops import rearrange


def real_flat_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert a flat real tensor representation to complex tensor format.

    Takes a real tensor where complex values are represented as concatenated
    real and imaginary parts in the last dimension, and converts it to a
    proper complex tensor. This is useful when working with models that
    output real tensors but represent complex data.

    The conversion assumes the last dimension contains pairs of (real, imaginary)
    values, so the dimension size must be even.

    Args:
        x (torch.Tensor): Real tensor of shape [B, L, D] where D = K * 2
                         B = batch size, L = sequence length, K = complex features
                         The last dimension D contains alternating real and imaginary values

    Returns:
        torch.Tensor: Complex tensor of shape [B, L, K] where each element
                     is a complex number formed from consecutive real/imaginary pairs

    Example:
        >>> x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # [1+2j, 3+4j]
        >>> result = real_flat_to_complex(x)
        >>> print(result.shape)  # torch.Size([1, 1, 2])
        >>> print(result)        # tensor([[[1.+2.j, 3.+4.j]]])

    """
    # Reshape last dimension from D to (K, 2) where 2 represents [real, imag]
    x_pair = rearrange(x, "b l (k o) -> b l k o", o=2)

    # Create complex tensor from real and imaginary parts
    # x_pair[..., 0] = real parts, x_pair[..., 1] = imaginary parts
    return torch.complex(x_pair[..., 0], x_pair[..., 1])


def complex_to_real_flat(x_complex: torch.Tensor) -> torch.Tensor:
    """Convert a complex tensor to flat real tensor representation.

    Takes a complex tensor and converts it to a real tensor where complex
    values are represented as concatenated real and imaginary parts in the
    last dimension. This is useful when feeding complex data to models that
    only support real-valued operations.

    The resulting tensor has twice the size in the last dimension, with
    alternating real and imaginary components.

    Args:
        x_complex (torch.Tensor): Complex tensor of shape [B, L, K]
                                 B = batch size, L = sequence length, K = complex features

    Returns:
        torch.Tensor: Real tensor of shape [B, L, K * 2] where the last dimension
                     contains alternating real and imaginary values:
                     [real_1, imag_1, real_2, imag_2, ..., real_K, imag_K]

    Example:
        >>> x = torch.tensor([[[1.+2.j, 3.+4.j]]])
        >>> result = complex_to_real_flat(x)
        >>> print(result.shape)  # torch.Size([1, 1, 4])
        >>> print(result)        # tensor([[[1., 2., 3., 4.]]])

    """
    # Convert complex tensor to real tensor with explicit real/imaginary parts
    # Result shape: [B, L, K, 2] where last dim is [real, imag]
    x_pair = torch.view_as_real(x_complex)

    # Flatten the last two dimensions to get alternating real/imaginary values
    # Final shape: [B, L, K*2] with pattern [real_1, imag_1, real_2, imag_2, ...]
    return rearrange(x_pair, "b l k o -> b l (k o)")
