"""Vector AR baseline (per subcarrier, across antennas) for TDD prediction."""

from pathlib import Path

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from src.cp.models.baseline.statistical.param_estimation import load_ar_parameters
from src.utils.data_utils import NUM_SUBCARRIERS, PRED_LEN, TOT_ANTENNAS


class ARMODEL(nn.Module):
    """Per-subcarrier vector AR(p) baseline model.

    Input shape:
        [batch_size, num_antennas, hist_len, num_subcarriers] (complex)
    Output shape:
        [batch_size, num_antennas, pred_len, num_subcarriers] (complex)
    """

    def __init__(
        self,
        order: int = 2,
        pred_len: int = PRED_LEN,
        num_antennas: int = TOT_ANTENNAS,
        num_subcarriers: int = NUM_SUBCARRIERS,
        param_path: str | Path | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the AR model with order and optional parameters."""
        super().__init__()
        self.name = "AR"
        self.is_separate_antennas = False

        self.order = int(order)
        self.pred_len = int(pred_len)
        self.num_antennas = int(num_antennas)
        self.num_subcarriers = int(num_subcarriers)

        self.register_buffer("coeff", torch.empty(0, dtype=torch.complex64), persistent=True)
        self.register_buffer("mean", torch.empty(0, dtype=torch.complex64), persistent=True)  # [K, N]
        self._params_loaded = False

        if param_path is not None:
            self.load_parameters(param_path)

    def __str__(self) -> str:
        """Return the model name."""
        return self.name

    @property
    def params_loaded(self) -> bool:
        """Return whether AR coefficients have been loaded."""
        return self._params_loaded

    def load_parameters(self, param_path: str | Path) -> None:
        """Load AR coefficients estimated offline."""
        params = load_ar_parameters(param_path)

        coeff_np = np.asarray(params["coeff"])
        mean_np = np.asarray(params["mean"])
        order = int(params["order"])
        num_antennas = int(params["num_antennas"])
        num_subcarriers = int(params["num_subcarriers"])

        expected_last_dim = num_antennas * order
        if coeff_np.shape != (num_subcarriers, num_antennas, expected_last_dim):
            raise ValueError(
                f"Invalid AR coeff shape {coeff_np.shape}; expected "
                f"({num_subcarriers}, {num_antennas}, {expected_last_dim})"
            )
        if mean_np.shape != (num_subcarriers, num_antennas):
            raise ValueError(f"Invalid AR mean shape {mean_np.shape}; expected ({num_subcarriers}, {num_antennas})")

        self.order = order
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers

        coeff_tensor = torch.from_numpy(coeff_np.astype(np.complex64))
        mean_tensor = torch.from_numpy(mean_np.astype(np.complex64))
        self.coeff = coeff_tensor.to(device=self.coeff.device)
        self.mean = mean_tensor.to(device=self.mean.device)
        self._params_loaded = True

    def _validate_input(self, x: torch.Tensor) -> None:
        if not torch.is_complex(x):
            raise TypeError(f"{self.name} expects complex input, got dtype={x.dtype}")
        if x.ndim != 4:
            raise ValueError(
                f"{self.name} expects input shape [B, N, L, K], got shape={tuple(x.shape)} with ndim={x.ndim}"
            )
        _, n, l, k = x.shape
        if n != self.num_antennas:
            raise ValueError(f"{self.name} expected num_antennas={self.num_antennas}, got {n}")
        if k != self.num_subcarriers:
            raise ValueError(f"{self.name} expected num_subcarriers={self.num_subcarriers}, got {k}")
        if l < self.order:
            raise ValueError(f"{self.name} requires hist_len >= order ({self.order}), got hist_len={l}")
        if not self._params_loaded:
            raise RuntimeError(
                f"{self.name} parameters are not loaded. "
                "Estimate and save AR coefficients first, then pass param_path or call load_parameters()."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run recursive AR rollout for pred_len future steps."""
        self._validate_input(x)
        x = x.to(dtype=torch.complex64)

        # state: [B, K, T, N]
        state = rearrange(x, "b n l k -> b k l n")  # [B, K, L, N]
        coeff = self.coeff.to(device=x.device, dtype=x.dtype)  # [K, N, N*order]
        mean = self.mean.to(device=x.device, dtype=x.dtype)  # [K, N]

        # Predict residual dynamics around the training-set mean.
        centered_state = state - mean.unsqueeze(0).unsqueeze(2)

        preds: list[torch.Tensor] = []
        for _ in range(self.pred_len):
            reg_parts = [centered_state[:, :, -lag, :] for lag in range(1, self.order + 1)]  # each [B, K, N]
            phi = torch.cat(reg_parts, dim=-1)  # [B, K, N*order]
            pred_centered = torch.einsum("knd,bkd->bkn", coeff, phi)  # [B, K, N]
            pred = pred_centered + mean.unsqueeze(0)  # [B, K, N]
            preds.append(pred)
            centered_state = torch.cat((centered_state, pred_centered.unsqueeze(2)), dim=2)

        y = torch.stack(preds, dim=2)  # [B, K, pred_len, N]
        return rearrange(y, "b k l n -> b n l k")
