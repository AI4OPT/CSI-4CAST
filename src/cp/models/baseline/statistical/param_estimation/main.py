"""Entry point for statistical baseline parameter estimation.

This script centralizes train/validation data preparation shared by AR and Wiener
baselines, then calls the corresponding estimation utilities.

Data preprocessing mirrors the learning-model training pipeline
(TrainValDataModule) so that AR/Wiener operate in the same data domain:
  1. Subset-stratified train/val split (same seed & ratio)
  2. Z-score normalization via ``normalize_input``
  3. Per-sample AWGN augmentation (same SNR range as learning training)
"""

from itertools import product
import logging
from pathlib import Path
import random

import numpy as np
import torch

from src.cp.models.baseline.statistical.param_estimation.ar import estimate_and_save_ar_parameters
from src.cp.models.baseline.statistical.param_estimation.wiener import estimate_and_save_wiener_parameters
from src.noise.noise import gen_vanilla_noise_snr
from src.utils.data_utils import (
    LIST_CHANNEL_MODEL,
    LIST_DELAY_SPREAD,
    LIST_MIN_SPEED_TRAIN,
    SNR_RANGE_GAUSSIAN_NOISE_TRAIN,
    _load_data,
)
from src.utils.dirs import DIR_DATA, DIR_OUTPUTS, DIR_WEIGHTS
from src.utils.main_utils import make_logger
from src.utils.norm_utils import normalize_input
from src.utils.time_utils import get_current_time


logger = logging.getLogger(__name__)


def _parse_order_candidates(raw: str | None) -> list[int] | None:
    """Parse comma-separated order candidates like '1,2,3,4'."""
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def _prepare_train_val_arrays(
    dir_data: str | Path,
    train_ratio: float,
    seed: int,
    list_cm: list[str],
    list_ds: list[float],
    list_ms: list[int],
    *,
    is_u2d_target: bool = False,
    is_u2d_hist: bool = False,
    num_load: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare train/val arrays matching the learning-model data pipeline.

    Steps mirror ``TrainValDataModule``:
      1. Per-subset deterministic split (same seed).
      2. ``normalize_input`` (z-score on real/imag independently).
      3. Per-sample AWGN at random SNR in ``SNR_RANGE_GAUSSIAN_NOISE_TRAIN``.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    dir_data = Path(dir_data)
    train_hist_list: list[torch.Tensor] = []
    train_target_list: list[torch.Tensor] = []
    val_hist_list: list[torch.Tensor] = []
    val_target_list: list[torch.Tensor] = []

    for cm, ds, ms in product(list_cm, list_ds, list_ms):
        hist_subset = _load_data(
            dir_data=dir_data,
            cm=cm,
            ds=ds,
            ms=ms,
            is_train=True,
            is_gen=False,
            is_hist=True,
            is_U2D=is_u2d_hist,
            num_load=num_load,
        )
        target_subset = _load_data(
            dir_data=dir_data,
            cm=cm,
            ds=ds,
            ms=ms,
            is_train=True,
            is_gen=False,
            is_hist=False,
            is_U2D=is_u2d_target,
            num_load=num_load,
        )

        subset_size = len(hist_subset)
        train_size = int(subset_size * train_ratio)
        if train_size <= 0 or train_size >= subset_size:
            raise ValueError(
                f"Invalid split for subset (cm={cm}, ds={ds}, ms={ms}): "
                f"subset_size={subset_size}, train_size={train_size}, train_ratio={train_ratio}"
            )

        # Mirrors TrainValDataModule deterministic per-subset splitting.
        indices = torch.randperm(subset_size, generator=torch.Generator().manual_seed(seed))
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        train_hist_list.append(hist_subset[train_idx])
        train_target_list.append(target_subset[train_idx])
        val_hist_list.append(hist_subset[val_idx])
        val_target_list.append(target_subset[val_idx])

    train_hist = torch.cat(train_hist_list, dim=0)
    train_target = torch.cat(train_target_list, dim=0)
    val_hist = torch.cat(val_hist_list, dim=0)
    val_target = torch.cat(val_target_list, dim=0)
    del train_hist_list, train_target_list, val_hist_list, val_target_list

    # ---- Normalize (matches TrainValDataModule / test_unit) ----
    is_u2d = is_u2d_hist or is_u2d_target
    train_hist, train_target = normalize_input(train_hist, train_target, is_U2D=is_u2d)
    val_hist, val_target = normalize_input(val_hist, val_target, is_U2D=is_u2d)
    assert train_target is not None and val_target is not None
    logger.info("Normalized train & val data (is_U2D=%s)", is_u2d)

    # ---- AWGN augmentation (matches TrainValDataModule) ----
    for i in range(len(train_hist)):
        snr_val = random.uniform(*SNR_RANGE_GAUSSIAN_NOISE_TRAIN)
        train_hist[i] = train_hist[i] + gen_vanilla_noise_snr(train_hist[i], SNR=snr_val)
    for i in range(len(val_hist)):
        snr_val = random.uniform(*SNR_RANGE_GAUSSIAN_NOISE_TRAIN)
        val_hist[i] = val_hist[i] + gen_vanilla_noise_snr(val_hist[i], SNR=snr_val)
    logger.info(
        "Added AWGN to train (%d) & val (%d) hist | SNR range %s dB",
        len(train_hist),
        len(val_hist),
        SNR_RANGE_GAUSSIAN_NOISE_TRAIN,
    )

    # ---- Convert to numpy ----
    train_hist_np = train_hist.detach().cpu().numpy().astype(np.complex64)
    train_target_np = train_target.detach().cpu().numpy().astype(np.complex64)
    val_hist_np = val_hist.detach().cpu().numpy().astype(np.complex64)
    val_target_np = val_target.detach().cpu().numpy().astype(np.complex64)
    return train_hist_np, train_target_np, val_hist_np, val_target_np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate and save AR/Wiener baseline parameters.")
    parser.add_argument("--model", type=str, required=True, choices=["ar", "wiener"], help="Baseline model.")
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["tdd", "fdd"],
        help="Scenario for estimation. AR defaults to tdd; Wiener defaults to fdd. "
        "Use --scenario tdd to estimate Wiener for TDD.",
    )
    parser.add_argument("--dir-data", type=str, default=DIR_DATA, help="Root dataset directory.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio from train/regular data.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument(
        "--num-load",
        type=int,
        default=-1,
        help="Optional number of samples to load per subset for debugging; -1 loads all.",
    )
    parser.add_argument("--ridge-lambda", type=float, default=1e-4, help="Ridge regularization coefficient.")

    # AR-specific arguments
    parser.add_argument("--order", type=int, default=2, help="Single AR order when not using order selection.")
    parser.add_argument(
        "--order-candidates",
        type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated AR order candidates for selection, e.g. '1,2,3,4'.",
    )
    parser.add_argument("--metric", type=str, default="nmse", choices=["mse", "nmse"], help="AR validation metric.")
    parser.add_argument(
        "--mean-centering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mean-centering + add-back for AR estimation/inference.",
    )
    # Optional output path override
    parser.add_argument("--output", type=str, default="", help="Optional output NPZ path.")
    parser.add_argument(
        "--dir-output-base",
        type=str,
        default=str(Path(DIR_OUTPUTS) / "cp" / "baseline" / "statistical" / "param_estimation"),
        help="Base directory for timestamped run outputs and logs.",
    )
    args = parser.parse_args()

    # Create timestamped output directory and logger (result.log) per run.
    dir_output = Path(args.dir_output_base) / get_current_time()
    dir_output.mkdir(parents=True, exist_ok=True)
    make_logger(dir_output)
    log_path = dir_output / "result.log"

    logger.info("Run output dir: %s", dir_output)
    logger.info("Log file path: %s", log_path)
    logger.info(
        "Starting parameter estimation | model=%s | train_ratio=%.3f | seed=%d | ridge_lambda=%.3e",
        args.model,
        args.train_ratio,
        args.seed,
        args.ridge_lambda,
    )

    order_candidates = _parse_order_candidates(args.order_candidates)
    if order_candidates is not None and len(order_candidates) == 0:
        raise ValueError("--order-candidates was provided but empty after parsing.")

    # Resolve scenario defaults: AR → tdd, Wiener → fdd (backward compat).
    scenario = args.scenario
    if scenario is None:
        scenario = "tdd" if args.model == "ar" else "fdd"
    if args.model == "ar" and scenario != "tdd":
        raise ValueError("AR is only supported for TDD scenario.")
    is_fdd = scenario == "fdd"
    logger.info("Resolved scenario: %s (is_fdd=%s)", scenario, is_fdd)

    if args.model == "ar":
        default_output = Path(DIR_WEIGHTS) / "tdd" / "ar_new" / "params.npz"
        output_path = Path(args.output) if args.output else default_output

        train_hist, train_target, val_hist, val_target = _prepare_train_val_arrays(
            dir_data=args.dir_data,
            train_ratio=args.train_ratio,
            seed=args.seed,
            list_cm=LIST_CHANNEL_MODEL,
            list_ds=LIST_DELAY_SPREAD,
            list_ms=LIST_MIN_SPEED_TRAIN,
            is_u2d_hist=False,
            is_u2d_target=False,  # AR in TDD uses UL future target.
            num_load=args.num_load,
        )
        logger.info(
            "AR data prepared | train_hist=%s | train_target=%s | val_hist=%s | val_target=%s",
            train_hist.shape,
            train_target.shape,
            val_hist.shape,
            val_target.shape,
        )
        if order_candidates is not None:
            logger.info(
                "AR order selection enabled | candidates=%s | metric=%s | concat_target=True",
                order_candidates,
                args.metric,
            )
        else:
            logger.info("AR single-order mode | order=%d", args.order)

        params, out_path = estimate_and_save_ar_parameters(
            hist=train_hist,
            path=output_path,
            order=args.order,
            ridge_lambda=args.ridge_lambda,
            val_hist=val_hist if order_candidates is not None else None,
            val_target=val_target if order_candidates is not None else None,
            train_target=train_target if order_candidates is not None else None,
            order_candidates=order_candidates,
            metric=args.metric,
            mean_center=args.mean_centering,
            concat_target=True,
        )

        logger.info("Saved AR parameters to: %s", out_path)
        logger.info(
            "Estimated coeff shape=%s | selected_order=%d | mean_centering=%s",
            np.asarray(params["coeff"]).shape,
            int(params["order"]),
            args.mean_centering,
        )
        if "best_val_error" in params:
            logger.info(
                "Selection metric=%s | best_val_error=%.6e",
                params.get("selected_metric"),
                float(params["best_val_error"]),
            )
        if order_candidates is not None:
            cand_orders = np.asarray(params.get("candidate_orders"), dtype=np.int64)
            cand_train = np.asarray(params.get("candidate_train_errors"), dtype=np.float64)
            cand_val = np.asarray(params.get("candidate_errors"), dtype=np.float64)

            logger.info("AR order-selection metrics:")
            for i, p in enumerate(cand_orders):
                logger.info(
                    f"order={int(p)} | train_{args.metric}={cand_train[i]:.6e} | val_{args.metric}={cand_val[i]:.6e}"
                )
            logger.info(
                f"selected_order={int(params['order'])} | best_val_{args.metric}={float(params['best_val_error']):.6e}"
            )

    else:
        default_output = Path(DIR_WEIGHTS) / scenario / "wiener" / "params.npz"
        output_path = Path(args.output) if args.output else default_output

        train_hist, train_target, val_hist, val_target = _prepare_train_val_arrays(
            dir_data=args.dir_data,
            train_ratio=args.train_ratio,
            seed=args.seed,
            list_cm=LIST_CHANNEL_MODEL,
            list_ds=LIST_DELAY_SPREAD,
            list_ms=LIST_MIN_SPEED_TRAIN,
            is_u2d_hist=False,
            is_u2d_target=is_fdd,  # FDD: UL→DL, TDD: UL→UL.
            num_load=args.num_load,
        )
        logger.info(
            "Wiener data prepared | train_hist=%s | train_target=%s | val_hist=%s | val_target=%s",
            train_hist.shape,
            train_target.shape,
            val_hist.shape,
            val_target.shape,
        )

        params, out_path = estimate_and_save_wiener_parameters(
            x_hist=train_hist,
            y_target=train_target,
            path=output_path,
            ridge_lambda=args.ridge_lambda,
            val_x_hist=val_hist,
            val_y_target=val_target,
            metric=args.metric,
        )

        logger.info("Saved Wiener parameters to: %s", out_path)
        logger.info(
            "Estimated Wiener shapes | weights=%s | mu_x=%s | mu_y=%s",
            np.asarray(params["weights"]).shape,
            np.asarray(params["mu_x"]).shape,
            np.asarray(params["mu_y"]).shape,
        )
        logger.info(
            "Wiener metric=%s | train_error=%.6e | val_error=%.6e",
            params.get("selected_metric", args.metric),
            float(params.get("train_error", float("nan"))),
            float(params.get("val_error", float("nan"))),
        )
