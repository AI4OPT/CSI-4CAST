import torch
from einops import rearrange


def real_flat_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a flat real tensor into complex, including rearrange.

    Args:
        x: real tensor of shape [B, L, D], where D = K * 2.

    Returns:
        Complex tensor of shape [B, L, K].
    """
    # split D into (K, 2)
    x_pair = rearrange(x, "b l (k o) -> b l k o", o=2)
    # form complex
    return torch.complex(x_pair[..., 0], x_pair[..., 1])


def complex_to_real_flat(x_complex: torch.Tensor) -> torch.Tensor:
    """
    Convert a complex tensor back to flat real, including rearrange.

    Args:
        x_complex: complex tensor of shape [B, L, K].

    Returns:
        Real tensor of shape [B, L, K * 2].
    """
    # get real-pair tensor of shape [B, L, K, 2]
    x_pair = torch.view_as_real(x_complex)
    # flatten back to D = K*2
    return rearrange(x_pair, "b l k o -> b l (k o)")
