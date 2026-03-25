"""Parameter estimation utilities for per-subcarrier Wiener (LMMSE) mapping.

The implemented Wiener baseline uses one linear map per subcarrier:

    y_k = W_k (x_k - mu_x_k) + mu_y_k

where:
    - x_k is flattened UL history: [num_antennas * hist_len]
    - y_k is flattened DL target: [num_antennas * pred_len]
    - W_k has shape [num_antennas * pred_len, num_antennas * hist_len]
"""

import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def _validate_xy(x_hist: np.ndarray, y_target: np.ndarray) -> tuple[int, int, int, int, int]:
    """Validate Wiener estimation inputs and return dimensions."""
    if x_hist.ndim != 4:
        raise ValueError(
            "x_hist must have shape [num_samples, num_antennas, hist_len, num_subcarriers], "
            f"got ndim={x_hist.ndim}"
        )
    if y_target.ndim != 4:
        raise ValueError(
            "y_target must have shape [num_samples, num_antennas, pred_len, num_subcarriers], "
            f"got ndim={y_target.ndim}"
        )

    s1, n1, hist_len, k1 = x_hist.shape
    s2, n2, pred_len, k2 = y_target.shape
    if s1 != s2 or n1 != n2 or k1 != k2:
        raise ValueError(
            "x_hist and y_target must align on [num_samples, num_antennas, num_subcarriers], "
            f"got {x_hist.shape} vs {y_target.shape}"
        )
    return s1, n1, hist_len, pred_len, k1


def estimate_wiener_parameters(
    x_hist: np.ndarray, y_target: np.ndarray, ridge_lambda: float = 1e-4
) -> dict[str, np.ndarray | int]:
    """Estimate per-subcarrier Wiener parameters from training pairs.

    Args:
        x_hist: Complex input history [num_samples, num_antennas, hist_len, num_subcarriers].
        y_target: Complex target [num_samples, num_antennas, pred_len, num_subcarriers].
        ridge_lambda: Ridge regularization coefficient for stable inversion.

    Returns:
        Dictionary containing:
            - "weights": [num_subcarriers, num_antennas*pred_len, num_antennas*hist_len] complex64
            - "mu_x": [num_subcarriers, num_antennas*hist_len] complex64
            - "mu_y": [num_subcarriers, num_antennas*pred_len] complex64
            - "num_antennas", "hist_len", "pred_len", "num_subcarriers": scalar ints
    """
    if ridge_lambda < 0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")

    x_hist = np.asarray(x_hist)
    y_target = np.asarray(y_target)
    if not np.iscomplexobj(x_hist) or not np.iscomplexobj(y_target):
        raise TypeError(f"Inputs must be complex, got x_hist={x_hist.dtype}, y_target={y_target.dtype}")
    num_samples, num_antennas, hist_len, pred_len, num_subcarriers = _validate_xy(x_hist, y_target)
    logger.info(
        "Wiener fit started | x_hist_shape=%s | x_dtype=%s | y_target_shape=%s | ridge_lambda=%.3e",
        x_hist.shape,
        x_hist.dtype,
        y_target.shape,
        ridge_lambda,
    )

    d_x = num_antennas * hist_len
    d_y = num_antennas * pred_len
    eye = np.eye(d_x, dtype=np.complex128)

    weights = np.zeros((num_subcarriers, d_y, d_x), dtype=np.complex128)
    mu_x = np.zeros((num_subcarriers, d_x), dtype=np.complex128)
    mu_y = np.zeros((num_subcarriers, d_y), dtype=np.complex128)

    for k in range(num_subcarriers):
        # Upcast one subcarrier slice to complex128 for numerical precision in the solve.
        xk = x_hist[:, :, :, k].reshape(num_samples, d_x).astype(np.complex128)
        yk = y_target[:, :, :, k].reshape(num_samples, d_y).astype(np.complex128)

        mu_x_k = np.mean(xk, axis=0)
        mu_y_k = np.mean(yk, axis=0)
        x_centered = xk - mu_x_k
        y_centered = yk - mu_y_k

        gram = x_centered.conj().T @ x_centered + ridge_lambda * eye  # [d_x, d_x]
        rhs = x_centered.conj().T @ y_centered  # [d_x, d_y]
        w_t = np.linalg.solve(gram, rhs)  # [d_x, d_y]

        weights[k] = w_t.T  # [d_y, d_x]
        mu_x[k] = mu_x_k
        mu_y[k] = mu_y_k

    params = {
        "weights": weights.astype(np.complex64),
        "mu_x": mu_x.astype(np.complex64),
        "mu_y": mu_y.astype(np.complex64),
        "num_antennas": np.int64(num_antennas),
        "hist_len": np.int64(hist_len),
        "pred_len": np.int64(pred_len),
        "num_subcarriers": np.int64(num_subcarriers),
    }
    logger.info(
        "Wiener fit completed | weights_shape=%s | mu_x_shape=%s | mu_y_shape=%s",
        params["weights"].shape,
        params["mu_x"].shape,
        params["mu_y"].shape,
    )
    return params


def _predict_wiener(
    x_hist: np.ndarray,
    weights: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    num_antennas: int,
    pred_len: int,
) -> np.ndarray:
    """Apply Wiener prediction in numpy.

    Args:
        x_hist: [S, N, L, K]
        weights: [K, d_y, d_x]
        mu_x: [K, d_x]
        mu_y: [K, d_y]
        num_antennas: N
        pred_len: P

    Returns:
        Predicted array [S, N, pred_len, K].
    """
    s, n, l, num_k = x_hist.shape
    dtype = np.complex64
    weights = weights.astype(dtype, copy=False)
    mu_x = mu_x.astype(dtype, copy=False)
    mu_y = mu_y.astype(dtype, copy=False)
    d_x = n * l

    preds = np.zeros((s, num_antennas, pred_len, num_k), dtype=dtype)
    for ki in range(num_k):
        xk = x_hist[:, :, :, ki].reshape(s, d_x).astype(dtype)  # [S, d_x]
        yk = (xk - mu_x[ki]) @ weights[ki].T + mu_y[ki]  # [S, d_y]
        preds[:, :, :, ki] = yk.reshape(s, num_antennas, pred_len)

    return preds


def _compute_error(
    pred: np.ndarray, target: np.ndarray, metric: str = "mse", eps: float = 1e-12
) -> float:
    """Compute scalar validation error."""
    metric = metric.lower()
    if metric == "mse":
        return float(np.mean(np.abs(pred - target) ** 2))
    if metric == "nmse":
        num = float(np.sum(np.abs(pred - target) ** 2))
        den = float(np.sum(np.abs(target) ** 2))
        return num / max(den, eps)
    raise ValueError(f"Unsupported metric '{metric}'. Supported: 'mse', 'nmse'")


def save_wiener_parameters(path: str | Path, params: dict[str, np.ndarray | int]) -> Path:
    """Save Wiener parameter dictionary to disk as compressed NPZ."""
    required = {"weights", "mu_x", "mu_y", "num_antennas", "hist_len", "pred_len", "num_subcarriers"}
    missing = required.difference(params.keys())
    if missing:
        raise ValueError(f"Missing Wiener parameter keys: {sorted(missing)}")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": np.asarray(params["weights"]),
        "mu_x": np.asarray(params["mu_x"]),
        "mu_y": np.asarray(params["mu_y"]),
        "num_antennas": np.int64(params["num_antennas"]),
        "hist_len": np.int64(params["hist_len"]),
        "pred_len": np.int64(params["pred_len"]),
        "num_subcarriers": np.int64(params["num_subcarriers"]),
    }
    if "train_error" in params:
        payload["train_error"] = np.float64(params["train_error"])
    if "val_error" in params:
        payload["val_error"] = np.float64(params["val_error"])
    if "selected_metric" in params:
        payload["selected_metric"] = np.asarray(params["selected_metric"])

    np.savez_compressed(out_path, **payload)
    logger.info(
        "Wiener params saved | path=%s | weights_shape=%s",
        out_path,
        np.asarray(params["weights"]).shape,
    )
    return out_path


def estimate_and_save_wiener_parameters(
    x_hist: np.ndarray,
    y_target: np.ndarray,
    path: str | Path,
    ridge_lambda: float = 1e-4,
    val_x_hist: np.ndarray | None = None,
    val_y_target: np.ndarray | None = None,
    metric: str = "nmse",
) -> tuple[dict[str, np.ndarray | int], Path]:
    """Estimate Wiener parameters and save them to disk.

    If val_x_hist and val_y_target are provided, computes and logs
    train/validation errors using the specified metric.
    """
    logger.info("Wiener estimate_and_save started | path=%s", path)
    params = estimate_wiener_parameters(x_hist=x_hist, y_target=y_target, ridge_lambda=ridge_lambda)

    weights = np.asarray(params["weights"])
    mu_x = np.asarray(params["mu_x"])
    mu_y = np.asarray(params["mu_y"])
    num_antennas = int(params["num_antennas"])
    pred_len = int(params["pred_len"])

    train_pred = _predict_wiener(x_hist, weights, mu_x, mu_y, num_antennas, pred_len)
    train_err = _compute_error(train_pred, y_target, metric=metric)
    params["train_error"] = np.float64(train_err)
    params["selected_metric"] = np.asarray(metric, dtype="<U8")
    logger.info("Wiener train_%s=%.6e", metric, train_err)

    if val_x_hist is not None and val_y_target is not None:
        val_x_hist = np.asarray(val_x_hist)
        val_y_target = np.asarray(val_y_target)
        val_pred = _predict_wiener(val_x_hist, weights, mu_x, mu_y, num_antennas, pred_len)
        val_err = _compute_error(val_pred, val_y_target, metric=metric)
        params["val_error"] = np.float64(val_err)
        logger.info("Wiener val_%s=%.6e", metric, val_err)

    out_path = save_wiener_parameters(path=path, params=params)
    logger.info("Wiener estimate_and_save completed | out_path=%s", out_path)
    return params, out_path


def load_wiener_parameters(path: str | Path) -> dict[str, np.ndarray | int]:
    """Load Wiener parameter dictionary from NPZ."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Wiener parameter file not found: {in_path}")

    with np.load(in_path) as data:
        params = {
            "weights": data["weights"],
            "mu_x": data["mu_x"],
            "mu_y": data["mu_y"],
            "num_antennas": int(data["num_antennas"]),
            "hist_len": int(data["hist_len"]),
            "pred_len": int(data["pred_len"]),
            "num_subcarriers": int(data["num_subcarriers"]),
        }
        if "train_error" in data:
            params["train_error"] = float(data["train_error"])
        if "val_error" in data:
            params["val_error"] = float(data["val_error"])
        if "selected_metric" in data:
            params["selected_metric"] = str(np.asarray(data["selected_metric"]).item())

    weights = np.asarray(params["weights"])
    if weights.ndim != 3:
        raise ValueError(f"Invalid Wiener weights shape in {in_path}: {weights.shape}, expected 3D [K, dy, dx]")
    logger.info(
        "Wiener params loaded | path=%s | weights_shape=%s | mu_x_shape=%s | mu_y_shape=%s",
        in_path,
        weights.shape,
        np.asarray(params["mu_x"]).shape,
        np.asarray(params["mu_y"]).shape,
    )
    return params
