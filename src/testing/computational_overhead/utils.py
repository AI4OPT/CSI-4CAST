import time
from contextlib import nullcontext

import lightning.pytorch as pl
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from tqdm import tqdm

from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, TOT_ANTENNAS


def count_trainable_parameters(model: torch.nn.Module | pl.LightningModule) -> int:
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module | pl.LightningModule) -> int:
    """Count total number of parameters in the model."""
    return sum(param.nelement() for param in model.parameters())


def count_flops(model: torch.nn.Module | pl.LightningModule, input_tensor: torch.Tensor) -> tuple[int, str | None]:
    """Count FLOPs for a single forward pass."""

    if model.name == "NP":
        return 0, None

    def parse_flops(flops: str) -> int:
        """Parse FLOPs string and return integer value."""
        if "T" in flops:
            return int(float(flops.replace("T", "").strip()) * 1e12)
        elif "G" in flops:
            return int(float(flops.replace("G", "").strip()) * 1e9)
        elif "M" in flops:
            return int(float(flops.replace("M", "").strip()) * 1e6)
        elif "K" in flops:
            return int(float(flops.replace("K", "").strip()) * 1e3)
        raise ValueError(f"Invalid FLOPs string: {flops}")

    try:
        flops, _, _ = get_model_profile(
            model=model,
            input_shape=tuple(input_tensor.shape),
            print_profile=False,
        )
        return parse_flops(flops), None
    except Exception as e:
        print(f"⚠️ Could not compute FLOPs: {e}")
        return -1, str(e)  # Return both -1 and the exception message


def measure_model_time(
    model: torch.nn.Module | pl.LightningModule,
    mode: str,
    device: torch.device,
    input_data: torch.Tensor,
    *,
    warmup_iterations: int = 50,
    tot_iterations: int = 100,
    desc: str | None = None,
) -> list[float]:
    """
    Benchmark forward-pass runtime for model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to benchmark.
    mode : str
        Either "inference" or "training". When "inference" the
        model is put into eval mode and the forward pass is wrapped with
        torch.no_grad. Otherwise the model is put into train mode and
        gradients are tracked (forward pass only).
    device : torch.device
        Device on which to allocate synthetic inputs.
    input_data : torch.Tensor
        Input tensor to pass to the model each iteration.
    warmup_iterations : int, default=50
        Number of initial iterations ignored when reporting runtimes. These
        iterations help stabilise caches / CUDA kernels.
    tot_iterations : int, default=100
        Total number of iterations to run **including** warm-up.
    desc : str | None
        Optional description string for the progress bar shown by tqdm.

    Returns
    -------
    list[float]
        Execution times per iteration in seconds (post warm-up).
    """
    assert mode in {"inference", "training"}, "mode must be 'inference' or 'training'"

    # Configure model state and choose appropriate context manager.
    if mode == "inference":
        model.eval()
        ctx = torch.no_grad()
    else:
        model.train()
        ctx = nullcontext()  # gradients are required for training mode

    times: list[float] = []
    bar_desc = desc or f"{mode.title()} | {getattr(model, 'name', 'Model')} | timing"

    with ctx:
        for i in tqdm(range(tot_iterations), desc=bar_desc, leave=False):
            # Warm-up iterations (not timed)
            if i < warmup_iterations:
                _ = model(input_data)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                continue

            start = time.perf_counter()
            _ = model(input_data)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return times


def get_input_data_for_model(
    model: torch.nn.Module | pl.LightningModule, batch_size: int, device: torch.device
) -> torch.Tensor:
    """
    Get appropriate input data for the model based on its architecture.

    Args:
        model: The model to get input data for
        batch_size: Batch size to use
        device: Device to create the tensor on

    Returns:
        Input tensor with appropriate shape and dtype (complex for gather_antennas, real for separate_antennas)
    """
    if hasattr(model, "is_separate_antennas") and model.is_separate_antennas:
        # For models that process antennas separately: [batch_size * num_antennas, hist_len, num_subcarriers*2] real
        shape = (batch_size * TOT_ANTENNAS, HIST_LEN, NUM_SUBCARRIERS * 2)
        return torch.randn(*shape, device=device, dtype=torch.float32)
    else:
        # For models that process all antennas together: [batch_size, num_antennas, hist_len, num_subcarriers] complex
        shape = (batch_size, TOT_ANTENNAS, HIST_LEN, NUM_SUBCARRIERS)
        real_part = torch.randn(*shape, device=device)
        imag_part = torch.randn(*shape, device=device)
        return torch.complex(real_part, imag_part)


def measure_inference_time_stats(
    model: torch.nn.Module | pl.LightningModule,
    input_data: torch.Tensor,
    device: torch.device,
    num_iterations: int = 10,
    warmup_iterations: int = 5,
) -> dict[str, float]:
    """
    Measure inference time statistics (simplified version for quick measurements).

    Args:
        model: Model to benchmark
        input_data: Input tensor for the model
        device: Device to run on
        num_iterations: Number of timing iterations
        warmup_iterations: Number of warmup iterations (not timed)

    Returns:
        Dictionary with timing statistics (mean, std in seconds)
    """
    times = measure_model_time(
        model=model,
        mode="inference",
        device=device,
        input_data=input_data,
        warmup_iterations=warmup_iterations,
        tot_iterations=warmup_iterations + num_iterations,
        desc=None,  # No progress bar for quick measurements
    )

    return {
        "mean": sum(times) / len(times) if times else 0.0,
        "std": torch.tensor(times).std().item() if len(times) > 1 else 0.0,
        "min": min(times) if times else 0.0,
        "max": max(times) if times else 0.0,
    }


def compute_all_metrics(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 1,
    inference_iterations: int = 10,
    warmup_iterations: int = 5,
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Compute all computational metrics for a model.

    Args:
        model: Model to analyze
        device: Device to run on
        batch_size: Batch size for computations
        inference_iterations: Number of iterations for timing
        warmup_iterations: Number of warmup iterations

    Returns:
        Tuple of (metrics_dict, errors_dict) where errors_dict contains any exception messages
    """
    metrics = {}
    errors = {}

    # Get input data
    sample_input = get_input_data_for_model(model, batch_size, device)

    # Count parameters
    metrics["total_params"] = count_total_parameters(model)
    metrics["trainable_params"] = count_trainable_parameters(model)
    metrics["total_params_M"] = metrics["total_params"] / 1e6  # In millions
    metrics["trainable_params_M"] = metrics["trainable_params"] / 1e6  # In millions

    # Count FLOPs
    metrics["flops"], flops_error = count_flops(model, sample_input)
    if flops_error:
        errors["flops_error"] = flops_error

    if metrics["flops"] > 0:
        metrics["mflops"] = metrics["flops"] / 1e6  # In millions
        metrics["gflops"] = metrics["flops"] / 1e9  # In billions
    else:
        metrics["mflops"] = -1
        metrics["gflops"] = -1

    # Measure inference time
    try:
        timing_stats = measure_inference_time_stats(
            model, sample_input, device, num_iterations=inference_iterations, warmup_iterations=warmup_iterations
        )

        # Add timing metrics with appropriate prefixes
        metrics["inference_time_mean_ms"] = timing_stats["mean"] * 1000  # Convert to ms
        metrics["inference_time_std_ms"] = timing_stats["std"] * 1000
        metrics["inference_time_min_ms"] = timing_stats["min"] * 1000
        metrics["inference_time_max_ms"] = timing_stats["max"] * 1000
        metrics["inference_time_mean_s"] = timing_stats["mean"]  # Keep in seconds too
        metrics["inference_time_std_s"] = timing_stats["std"]
    except Exception as e:
        errors["timing_error"] = str(e)
        metrics["inference_time_mean_ms"] = -1
        metrics["inference_time_std_ms"] = -1
        metrics["inference_time_min_ms"] = -1
        metrics["inference_time_max_ms"] = -1
        metrics["inference_time_mean_s"] = -1
        metrics["inference_time_std_s"] = -1

    return metrics, errors


def log_computational_metrics(metrics: dict[str, float], model_name: str) -> None:
    """
    Log computational metrics in a formatted way.

    Args:
        metrics: Dictionary of computed metrics
        model_name: Name of the model for logging
    """
    print(f"🔧 Computational Metrics for {model_name}:")
    print(
        f"   📊 Parameters: {metrics['total_params']:,} total ({metrics['total_params_M']:.2f}M), "
        f"{metrics['trainable_params']:,} trainable ({metrics['trainable_params_M']:.2f}M)"
    )

    if metrics["flops"] > 0:
        print(f"   ⚡ FLOPs: {metrics['flops']:,} ({metrics['gflops']:.3f}G)")
    else:
        print("   ⚡ FLOPs: Could not compute")

    print(f"   ⏱️  Inference Time: {metrics['inference_time_mean_ms']:.2f}±{metrics['inference_time_std_ms']:.2f}ms")
