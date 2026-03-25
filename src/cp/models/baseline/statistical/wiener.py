"""Wiener (LMMSE) baseline with per-subcarrier linear mapping for FDD."""

from pathlib import Path

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from src.cp.models.baseline.statistical.param_estimation import load_wiener_parameters
from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, PRED_LEN, TOT_ANTENNAS


class WIENERMODEL(nn.Module):
    """Per-subcarrier Wiener model from history CSI to future CSI.

    Input shape:
        [batch_size, num_antennas, hist_len, num_subcarriers] (complex)
    Output shape:
        [batch_size, num_antennas, pred_len, num_subcarriers] (complex)
    """

    def __init__(
        self,
        hist_len: int = HIST_LEN,
        pred_len: int = PRED_LEN,
        num_antennas: int = TOT_ANTENNAS,
        num_subcarriers: int = NUM_SUBCARRIERS,
        param_path: str | Path | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.name = "WIENER"
        self.is_separate_antennas = False

        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.num_antennas = int(num_antennas)
        self.num_subcarriers = int(num_subcarriers)

        self.register_buffer("weights", torch.empty(0, dtype=torch.complex64), persistent=True)
        self.register_buffer("mu_x", torch.empty(0, dtype=torch.complex64), persistent=True)
        self.register_buffer("mu_y", torch.empty(0, dtype=torch.complex64), persistent=True)
        self._params_loaded = False

        if param_path is not None:
            self.load_parameters(param_path)

    def __str__(self) -> str:
        return self.name

    @property
    def params_loaded(self) -> bool:
        """Return whether Wiener parameters have been loaded."""
        return self._params_loaded

    def load_parameters(self, param_path: str | Path) -> None:
        """Load Wiener parameters estimated offline."""
        params = load_wiener_parameters(param_path)

        weights_np = np.asarray(params["weights"])
        mu_x_np = np.asarray(params["mu_x"])
        mu_y_np = np.asarray(params["mu_y"])
        num_antennas = int(params["num_antennas"])
        hist_len = int(params["hist_len"])
        pred_len = int(params["pred_len"])
        num_subcarriers = int(params["num_subcarriers"])

        d_x = num_antennas * hist_len
        d_y = num_antennas * pred_len
        if weights_np.shape != (num_subcarriers, d_y, d_x):
            raise ValueError(
                f"Invalid Wiener weight shape {weights_np.shape}; expected ({num_subcarriers}, {d_y}, {d_x})"
            )
        if mu_x_np.shape != (num_subcarriers, d_x):
            raise ValueError(f"Invalid Wiener mu_x shape {mu_x_np.shape}; expected ({num_subcarriers}, {d_x})")
        if mu_y_np.shape != (num_subcarriers, d_y):
            raise ValueError(f"Invalid Wiener mu_y shape {mu_y_np.shape}; expected ({num_subcarriers}, {d_y})")

        self.num_antennas = num_antennas
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.num_subcarriers = num_subcarriers

        self.weights = torch.from_numpy(weights_np.astype(np.complex64)).to(device=self.weights.device)
        self.mu_x = torch.from_numpy(mu_x_np.astype(np.complex64)).to(device=self.mu_x.device)
        self.mu_y = torch.from_numpy(mu_y_np.astype(np.complex64)).to(device=self.mu_y.device)
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
        if l != self.hist_len:
            raise ValueError(f"{self.name} expected hist_len={self.hist_len}, got {l}")
        if k != self.num_subcarriers:
            raise ValueError(f"{self.name} expected num_subcarriers={self.num_subcarriers}, got {k}")
        if not self._params_loaded:
            raise RuntimeError(
                f"{self.name} parameters are not loaded. "
                "Estimate and save Wiener parameters first, then pass param_path or call load_parameters()."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Wiener mapping independently on each subcarrier."""
        self._validate_input(x)
        x = x.to(dtype=torch.complex64)

        x_flat = rearrange(x, "b n l k -> b k (n l)")  # [B, K, d_x]
        weights = self.weights.to(device=x.device, dtype=x.dtype)  # [K, d_y, d_x]
        mu_x = self.mu_x.to(device=x.device, dtype=x.dtype)  # [K, d_x]
        mu_y = self.mu_y.to(device=x.device, dtype=x.dtype)  # [K, d_y]

        centered = x_flat - mu_x.unsqueeze(0)  # [B, K, d_x]
        y_flat = torch.einsum("kod,bkd->bko", weights, centered) + mu_y.unsqueeze(0)  # [B, K, d_y]

        return rearrange(y_flat, "b k (n l) -> b n l k", n=self.num_antennas, l=self.pred_len)
