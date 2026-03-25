"""Parameter estimation utilities for per-subcarrier vector AR models.

The implemented AR baseline uses a vector AR(p) process across antennas for each
subcarrier independently:

    h_t[k] = A_1[k] h_{t-1}[k] + ... + A_p[k] h_{t-p}[k]

where:
    - h_t[k] has shape [num_antennas]
    - A_j[k] has shape [num_antennas, num_antennas]
"""

import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def _to_sorted_unique_orders(order_candidates: list[int] | tuple[int, ...]) -> list[int]:
    """Normalize, validate, and sort AR order candidates."""
    if len(order_candidates) == 0:
        raise ValueError("order_candidates must contain at least one integer order")
    orders = sorted(set(int(o) for o in order_candidates))
    if any(o < 1 for o in orders):
        raise ValueError(f"All order candidates must be >= 1, got {orders}")
    return orders


def _validate_hist(hist: np.ndarray, order: int) -> tuple[int, int, int, int]:
    """Validate AR estimation input and return dimensions."""
    if hist.ndim != 4:
        raise ValueError(
            f"hist must have shape [num_samples, num_antennas, hist_len, num_subcarriers], got ndim={hist.ndim}"
        )

    num_samples, num_antennas, hist_len, num_subcarriers = hist.shape
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if hist_len <= order:
        raise ValueError(f"hist_len must be > order, got hist_len={hist_len}, order={order}")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    return num_samples, num_antennas, hist_len, num_subcarriers


def _validate_target(target: np.ndarray, num_samples: int, num_antennas: int, num_subcarriers: int) -> int:
    """Validate AR validation target shape and return pred_len."""
    if target.ndim != 4:
        raise ValueError(
            f"target must have shape [num_samples, num_antennas, pred_len, num_subcarriers], got ndim={target.ndim}"
        )
    s, n, pred_len, k = target.shape
    if s != num_samples or n != num_antennas or k != num_subcarriers:
        raise ValueError(
            "target shape must align with hist on [num_samples, num_antennas, num_subcarriers], "
            f"got hist=[{num_samples}, {num_antennas}, *, {num_subcarriers}] and target={target.shape}"
        )
    if pred_len < 1:
        raise ValueError(f"pred_len must be >= 1, got {pred_len}")
    return pred_len


def _rollout_ar_predictions(
    hist: np.ndarray, coeff: np.ndarray, order: int, pred_len: int, mean: np.ndarray | None = None
) -> np.ndarray:
    """Run AR rollout predictions.

    Args:
        hist: [S, N, L, K]
        coeff: [K, N, N*order]
        order: AR order
        pred_len: prediction horizon
        mean: Optional [num_subcarriers, num_antennas] mean used for centering.

    Returns:
        Predicted tensor with shape [S, N, pred_len, K].
    """
    s, n, _, num_k = hist.shape
    dtype = np.complex64
    coeff = coeff.astype(dtype, copy=False)  # [K, N, N*order]
    if mean is None:
        mean = np.zeros((num_k, n), dtype=dtype)
    else:
        mean = np.asarray(mean, dtype=dtype)
        if mean.shape != (num_k, n):
            raise ValueError(f"mean must have shape ({num_k}, {n}), got {mean.shape}")

    preds = np.zeros((s, n, pred_len, num_k), dtype=dtype)

    for ki in range(num_k):
        # Work on one subcarrier at a time to avoid huge transposed copies.
        state_k = hist[:, :, :, ki].transpose(0, 2, 1).astype(dtype, copy=True)  # [S, T, N]
        centered_k = state_k - mean[ki][None, None, :]
        coeff_k = coeff[ki]  # [N, N*order]
        mean_k = mean[ki]  # [N]

        for step in range(pred_len):
            reg_parts = [centered_k[:, -lag, :] for lag in range(1, order + 1)]  # each [S, N]
            phi = np.concatenate(reg_parts, axis=-1)  # [S, N*order]
            pred_centered = phi @ coeff_k.T  # [S, N]
            preds[:, :, step, ki] = pred_centered + mean_k[None, :]
            centered_k = np.concatenate((centered_k, pred_centered[:, np.newaxis, :]), axis=1)

    return preds


def _compute_error(pred: np.ndarray, target: np.ndarray, metric: str = "mse", eps: float = 1e-12) -> float:
    """Compute scalar validation error."""
    metric = metric.lower()
    if metric == "mse":
        return float(np.mean(np.abs(pred - target) ** 2))
    if metric == "nmse":
        num = float(np.sum(np.abs(pred - target) ** 2))
        den = float(np.sum(np.abs(target) ** 2))
        return num / max(den, eps)
    raise ValueError(f"Unsupported metric '{metric}'. Supported: 'mse', 'nmse'")


def estimate_ar_parameters(
    hist: np.ndarray, order: int = 2, ridge_lambda: float = 1e-4, mean_center: bool = True
) -> dict[str, np.ndarray | int]:
    """Estimate per-subcarrier vector AR(p) coefficients from training history.

    Args:
        hist: Complex array with shape [num_samples, num_antennas, hist_len, num_subcarriers].
        order: AR order p.
        ridge_lambda: Ridge regularization coefficient for stable inversion.
        mean_center: If True, estimate on centered histories and save mean for add-back.

    Returns:
        Dictionary containing:
            - "coeff": [num_subcarriers, num_antennas, num_antennas * order] complex64
            - "order": scalar int
            - "num_antennas": scalar int
            - "num_subcarriers": scalar int
    """
    if ridge_lambda < 0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")

    hist = np.asarray(hist)
    if not np.iscomplexobj(hist):
        raise TypeError(f"hist must be complex, got dtype={hist.dtype}")
    num_samples, num_antennas, hist_len, num_subcarriers = _validate_hist(hist, order)
    logger.info(
        "AR fit started | hist_shape=%s | hist_dtype=%s | order=%d | ridge_lambda=%.3e | mean_center=%s",
        hist.shape,
        hist.dtype,
        order,
        ridge_lambda,
        mean_center,
    )
    d_reg = num_antennas * order

    # Mean in input dtype (complex64) to avoid full-array upcast.
    mean_nk = np.mean(hist, axis=(0, 2)) if mean_center else np.zeros((num_antennas, num_subcarriers), dtype=hist.dtype)

    coeff = np.zeros((num_subcarriers, num_antennas, d_reg), dtype=np.complex128)
    eye = np.eye(d_reg, dtype=np.complex128)

    for k in range(num_subcarriers):
        # Upcast one subcarrier slice to complex128 for numerical precision in the solve.
        x = (hist[:, :, :, k] - mean_nk[:, k][None, :, None]).astype(np.complex128)

        regressors: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for t in range(order, hist_len):
            past = x[:, :, t - order : t][:, :, ::-1]  # [S, N, p]
            phi_t = np.transpose(past, (0, 2, 1)).reshape(num_samples, d_reg)  # [S, pN]
            y_t = x[:, :, t]  # [S, N]
            regressors.append(phi_t)
            targets.append(y_t)

        phi = np.concatenate(regressors, axis=0)  # [M, pN]
        y = np.concatenate(targets, axis=0)  # [M, N]

        gram = phi.conj().T @ phi + ridge_lambda * eye  # [pN, pN]
        rhs = phi.conj().T @ y  # [pN, N]
        a_t = np.linalg.solve(gram, rhs)  # [pN, N]
        coeff[k] = a_t.T  # [N, pN]

    params = {
        "coeff": coeff.astype(np.complex64),
        "mean": mean_nk.T.astype(np.complex64),  # [K, N]
        "order": np.int64(order),
        "num_antennas": np.int64(num_antennas),
        "num_subcarriers": np.int64(num_subcarriers),
    }
    logger.info(
        "AR fit completed | coeff_shape=%s | mean_shape=%s",
        params["coeff"].shape,
        params["mean"].shape,
    )
    return params


def estimate_ar_parameters_with_order_selection(
    train_hist: np.ndarray,
    val_hist: np.ndarray,
    val_target: np.ndarray,
    order_candidates: list[int] | tuple[int, ...],
    train_target: np.ndarray | None = None,
    ridge_lambda: float = 1e-4,
    metric: str = "mse",
    mean_center: bool = True,
    checkpoint_path: str | Path | None = None,
    concat_target: bool = False,
) -> dict[str, np.ndarray | int | float]:
    """Estimate AR parameters and select order by minimal validation error.

    This function:
    1. Fits one AR model per candidate order on train_hist.
    2. Runs rollout prediction on val_hist for horizon = val_target.shape[2].
    3. Selects the order with minimal validation error.
    4. Returns the best-order parameters plus selection metadata.

    If checkpoint_path is provided, saves the best-so-far parameters after each
    candidate is evaluated so that progress is not lost if the job is killed.

    Args:
        train_hist: Training history [S_train, N, L, K].
        val_hist: Validation history [S_val, N, L, K].
        val_target: Validation target [S_val, N, P, K].
        train_target: Optional training target [S_train, N, P_train, K] for train-error logging.
        order_candidates: Candidate AR orders to evaluate, e.g., [1, 2, 3, 4].
        ridge_lambda: Ridge regularization.
        metric: Validation metric: "mse" or "nmse".
        checkpoint_path: If provided, save best-so-far params to this path after each candidate.
        concat_target: If True and train_target is provided, concatenate train_hist
            and train_target along the time axis before fitting. This gives more
            regression pairs and supports higher orders. Validation still uses
            val_hist only.

    Returns:
        Dictionary compatible with save/load, containing:
            - coeff, order, num_antennas, num_subcarriers
            - best_val_error (float)
            - selected_metric (str)
            - candidate_orders (int array)
            - candidate_train_errors (float array; NaN if train_target not provided)
            - candidate_errors (float array)
    """
    if ridge_lambda < 0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")

    train_hist = np.asarray(train_hist)
    val_hist = np.asarray(val_hist)
    val_target = np.asarray(val_target)

    # Optionally concatenate train_hist + train_target for estimation.
    if concat_target and train_target is not None:
        train_target = np.asarray(train_target)
        fit_hist = np.concatenate([train_hist, train_target], axis=2)
        logger.info(
            "AR concat_target enabled | train_hist=%s + train_target=%s -> fit_hist=%s",
            train_hist.shape,
            train_target.shape,
            fit_hist.shape,
        )
    else:
        fit_hist = train_hist

    orders = _to_sorted_unique_orders(order_candidates)
    max_order = max(orders)
    logger.info(
        "AR order selection started | candidates=%s | metric=%s | ridge_lambda=%.3e | mean_center=%s",
        orders,
        metric,
        ridge_lambda,
        mean_center,
    )

    fit_s, fit_n, fit_l, fit_k = _validate_hist(fit_hist, max_order)
    val_s = val_hist.shape[0]
    val_n = val_hist.shape[1]
    val_l = val_hist.shape[2]
    val_k = val_hist.shape[3]
    if fit_n != val_n or fit_k != val_k:
        raise ValueError(
            "train and val must match on [num_antennas, num_subcarriers], "
            f"got fit={fit_hist.shape}, val={val_hist.shape}"
        )
    pred_len = _validate_target(val_target, val_s, val_n, val_k)
    train_pred_len = None
    if train_target is not None:
        train_target = np.asarray(train_target)
        train_pred_len = _validate_target(train_target, train_hist.shape[0], fit_n, fit_k)

    errors: list[float] = []
    train_errors: list[float] = []
    best_err = float("inf")
    best_params: dict[str, np.ndarray | int] = {}

    for order in orders:
        if fit_l <= order:
            raise ValueError(f"order={order} is too large for fit_hist with hist_len={fit_l}")
        if val_l < order:
            raise ValueError(f"order={order} requires val_hist_len >= {order}, got {val_l}")
        params = estimate_ar_parameters(
            hist=fit_hist, order=order, ridge_lambda=ridge_lambda, mean_center=mean_center
        )
        coeff = np.asarray(params["coeff"])
        mean = np.asarray(params["mean"])
        pred = _rollout_ar_predictions(hist=val_hist, coeff=coeff, order=order, pred_len=pred_len, mean=mean)
        err = _compute_error(pred=pred, target=val_target, metric=metric)
        if train_target is not None and train_pred_len is not None:
            pred_train = _rollout_ar_predictions(
                hist=train_hist, coeff=coeff, order=order, pred_len=train_pred_len, mean=mean
            )
            train_err = _compute_error(pred=pred_train, target=train_target, metric=metric)
        else:
            train_err = float("nan")
        errors.append(err)
        train_errors.append(train_err)
        logger.info(
            "AR candidate evaluated | order=%d | train_%s=%.6e | val_%s=%.6e", order, metric, train_err, metric, err
        )

        if err < best_err or (err == best_err and order < int(best_params.get("order", order + 1))):
            best_err = err
            best_params = params
            logger.info("AR new best | order=%d | val_%s=%.6e", order, metric, err)

            if checkpoint_path is not None:
                ckpt = dict(best_params)
                ckpt["best_val_error"] = np.float64(best_err)
                ckpt["selected_metric"] = np.asarray(metric, dtype="<U8")
                ckpt["candidate_orders"] = np.asarray(orders[: len(errors)], dtype=np.int64)
                ckpt["candidate_train_errors"] = np.asarray(train_errors, dtype=np.float64)
                ckpt["candidate_errors"] = np.asarray(errors, dtype=np.float64)
                save_ar_parameters(path=checkpoint_path, params=ckpt)
                logger.info("AR checkpoint saved | order=%d | path=%s", order, checkpoint_path)

    best_params["best_val_error"] = np.float64(best_err)
    best_params["selected_metric"] = np.asarray(metric, dtype="<U8")
    best_params["candidate_orders"] = np.asarray(orders, dtype=np.int64)
    best_params["candidate_train_errors"] = np.asarray(train_errors, dtype=np.float64)
    best_params["candidate_errors"] = np.asarray(errors, dtype=np.float64)
    logger.info(
        "AR order selection completed | selected_order=%d | best_val_%s=%.6e",
        int(best_params["order"]),
        metric,
        float(best_params["best_val_error"]),
    )
    return best_params


def save_ar_parameters(path: str | Path, params: dict[str, object]) -> Path:
    """Save AR parameter dictionary to disk as compressed NPZ."""
    required = {"coeff", "mean", "order", "num_antennas", "num_subcarriers"}
    missing = required.difference(params.keys())
    if missing:
        raise ValueError(f"Missing AR parameter keys: {sorted(missing)}")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "coeff": np.asarray(params["coeff"]),
        "mean": np.asarray(params["mean"]),
        "order": np.int64(params["order"]),
        "num_antennas": np.int64(params["num_antennas"]),
        "num_subcarriers": np.int64(params["num_subcarriers"]),
    }

    # Optional order-selection metadata.
    if "best_val_error" in params:
        payload["best_val_error"] = np.float64(params["best_val_error"])
    if "selected_metric" in params:
        payload["selected_metric"] = np.asarray(params["selected_metric"])
    if "candidate_orders" in params:
        payload["candidate_orders"] = np.asarray(params["candidate_orders"], dtype=np.int64)
    if "candidate_train_errors" in params:
        payload["candidate_train_errors"] = np.asarray(params["candidate_train_errors"], dtype=np.float64)
    if "candidate_errors" in params:
        payload["candidate_errors"] = np.asarray(params["candidate_errors"], dtype=np.float64)

    np.savez_compressed(out_path, **payload)
    logger.info(
        "AR params saved | path=%s | coeff_shape=%s | order=%d",
        out_path,
        payload["coeff"].shape,
        int(payload["order"]),
    )
    return out_path


def estimate_and_save_ar_parameters(
    hist: np.ndarray,
    path: str | Path,
    order: int = 2,
    ridge_lambda: float = 1e-4,
    val_hist: np.ndarray | None = None,
    val_target: np.ndarray | None = None,
    train_target: np.ndarray | None = None,
    order_candidates: list[int] | tuple[int, ...] | None = None,
    metric: str = "mse",
    mean_center: bool = True,
    concat_target: bool = False,
) -> tuple[dict[str, object], Path]:
    """Estimate AR parameters and save them to disk.

    Selection behavior:
    - If order_candidates, val_hist, and val_target are all provided, select best
      order via validation and save only the best-order parameters.
    - Otherwise, fit the single specified order.

    If concat_target is True and train_target is provided, the training history
    and target are concatenated along the time axis before fitting, giving more
    regression pairs and supporting higher orders.
    """
    has_order_candidates = order_candidates is not None
    has_val_hist = val_hist is not None
    has_val_target = val_target is not None
    has_val_inputs = has_val_hist and has_val_target
    has_any_selection_inputs = has_order_candidates or has_val_hist or has_val_target

    if has_any_selection_inputs and not (has_order_candidates and has_val_inputs):
        raise ValueError(
            "To enable order selection, provide all of: order_candidates, val_hist, val_target. "
            "Otherwise provide none of them."
        )

    if has_order_candidates and has_val_inputs:
        logger.info("AR estimate_and_save using validation-based order selection (concat_target=%s)", concat_target)
        params = estimate_ar_parameters_with_order_selection(
            train_hist=hist,
            val_hist=val_hist,
            val_target=val_target,
            train_target=train_target,
            order_candidates=order_candidates,
            ridge_lambda=ridge_lambda,
            metric=metric,
            mean_center=mean_center,
            checkpoint_path=path,
            concat_target=concat_target,
        )
    else:
        if concat_target and train_target is not None:
            hist = np.concatenate([hist, np.asarray(train_target)], axis=2)
            logger.info("AR concat_target enabled (single order) | fit_hist=%s", hist.shape)
        logger.info("AR estimate_and_save using single fixed order=%d", order)
        params = estimate_ar_parameters(hist=hist, order=order, ridge_lambda=ridge_lambda, mean_center=mean_center)

    coeff = np.asarray(params["coeff"])
    mean = np.asarray(params["mean"])
    selected_order = int(params["order"])

    if train_target is not None:
        train_pred = _rollout_ar_predictions(
            hist=hist,
            coeff=coeff,
            order=selected_order,
            pred_len=train_target.shape[2],
            mean=mean,
        )
        train_err = _compute_error(train_pred, train_target, metric=metric)
        logger.info("AR train_%s=%.6e | order=%d", metric, train_err, selected_order)

    if val_hist is not None and val_target is not None:
        val_pred = _rollout_ar_predictions(
            hist=val_hist,
            coeff=coeff,
            order=selected_order,
            pred_len=val_target.shape[2],
            mean=mean,
        )
        val_err = _compute_error(val_pred, val_target, metric=metric)
        logger.info("AR val_%s=%.6e | order=%d", metric, val_err, selected_order)

    out_path = save_ar_parameters(path=path, params=params)
    logger.info("AR estimate_and_save completed | out_path=%s", out_path)
    return params, out_path


def load_ar_parameters(path: str | Path) -> dict[str, np.ndarray | int | float]:
    """Load AR parameter dictionary from NPZ."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"AR parameter file not found: {in_path}")

    with np.load(in_path) as data:
        params = {
            "coeff": data["coeff"],
            "mean": data["mean"] if "mean" in data else None,
            "order": int(data["order"]),
            "num_antennas": int(data["num_antennas"]),
            "num_subcarriers": int(data["num_subcarriers"]),
        }
        if "best_val_error" in data:
            params["best_val_error"] = float(data["best_val_error"])
        if "selected_metric" in data:
            params["selected_metric"] = str(np.asarray(data["selected_metric"]).item())
        if "candidate_orders" in data:
            params["candidate_orders"] = data["candidate_orders"]
        if "candidate_train_errors" in data:
            params["candidate_train_errors"] = data["candidate_train_errors"]
        if "candidate_errors" in data:
            params["candidate_errors"] = data["candidate_errors"]

    coeff = np.asarray(params["coeff"])
    if coeff.ndim != 3:
        raise ValueError(f"Invalid AR coeff shape in {in_path}: {coeff.shape}, expected 3D [K, N, N*order]")
    if params["mean"] is None:
        # Backward compatibility for old parameter files without mean.
        k, n, _ = coeff.shape
        params["mean"] = np.zeros((k, n), dtype=np.complex64)
        logger.warning("AR params file missing mean; using zero mean for backward compatibility: %s", in_path)
    else:
        mean = np.asarray(params["mean"])
        k, n, _ = coeff.shape
        if mean.shape != (k, n):
            raise ValueError(f"Invalid AR mean shape in {in_path}: {mean.shape}, expected ({k}, {n})")
    logger.info(
        "AR params loaded | path=%s | coeff_shape=%s | mean_shape=%s | order=%d",
        in_path,
        coeff.shape,
        np.asarray(params["mean"]).shape,
        int(params["order"]),
    )
    return params
