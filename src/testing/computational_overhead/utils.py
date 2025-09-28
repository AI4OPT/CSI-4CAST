"""Utility functions for computational overhead analysis of neural network models.

This module provides comprehensive utilities for measuring and analyzing the computational
overhead of PyTorch and PyTorch Lightning models, specifically designed for CSI prediction
models but applicable to general neural network architectures.

Key Features:
- Parameter counting (total and trainable parameters)
- FLOP (Floating Point Operations) counting using DeepSpeed profiler
- Accurate runtime measurement with statistical analysis
- Device-agnostic timing with proper CUDA synchronization
- Model-specific input data generation for different architectures
- Memory-efficient benchmarking with proper cleanup

Functions:
- count_trainable_parameters(): Count trainable model parameters
- count_total_parameters(): Count all model parameters
- count_flops(): Measure computational cost in FLOPs
- measure_model_time(): Statistical runtime measurement with warmup
- get_input_data_for_model(): Generate appropriate synthetic inputs
- measure_inference_time_stats(): Quick timing measurements
- compute_all_metrics(): Comprehensive metric computation
- log_computational_metrics(): Formatted metric logging

The timing functions use proper statistical methods with warmup periods to ensure
accurate measurements, accounting for CUDA kernel initialization and cache effects.
"""

import time
from contextlib import nullcontext

import lightning.pytorch as pl
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from tqdm import tqdm

from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, TOT_ANTENNAS


def count_trainable_parameters(model: torch.nn.Module | pl.LightningModule) -> int:
    """Count the number of trainable parameters in the model.

    Iterates through all model parameters and counts only those that require
    gradients (trainable parameters). This metric is important for understanding
    the learning capacity and memory requirements during training.

    Args:
        model (torch.nn.Module | pl.LightningModule): The model to analyze

    Returns:
        int: Total number of trainable parameters

    Example:
        >>> model = MyModel()
        >>> trainable_params = count_trainable_parameters(model)
        >>> print(f"Trainable parameters: {trainable_params:,}")

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module | pl.LightningModule) -> int:
    """Count the total number of parameters in the model.

    Counts all parameters in the model, including both trainable and frozen
    parameters. This metric represents the total memory footprint of the model
    parameters and is useful for model size comparison.

    Args:
        model (torch.nn.Module | pl.LightningModule): The model to analyze

    Returns:
        int: Total number of parameters (trainable + frozen)

    Example:
        >>> model = MyModel()
        >>> total_params = count_total_parameters(model)
        >>> print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    """
    return sum(param.nelement() for param in model.parameters())


def count_flops(model: torch.nn.Module | pl.LightningModule, input_tensor: torch.Tensor) -> tuple[int, str | None]:
    """Count FLOPs (Floating Point Operations) for a single forward pass.

    Uses DeepSpeed's FLOP profiler to measure the computational cost of a model.
    FLOPs provide a hardware-independent measure of computational complexity,
    useful for comparing model efficiency across different architectures.

    Args:
        model (torch.nn.Module | pl.LightningModule): The model to profile
        input_tensor (torch.Tensor): Sample input tensor for the model

    Returns:
        tuple[int, str | None]: (flop_count, error_message)
            - flop_count: Number of FLOPs, or -1 if computation failed
            - error_message: None if successful, error string if failed

    Note:
        - Some models (like "NP") may return 0 FLOPs due to profiling limitations
        - Complex models or custom operations may not be fully supported
        - The profiler requires the model to be in a valid state for forward pass

    """
    # Special case: Neural Process models may not be supported by FLOP profiler
    if model.name == "NP":
        return 0, None

    def parse_flops(flops: str) -> int:
        """Parse FLOPs string from DeepSpeed profiler and return integer value.

        The profiler returns human-readable strings like "1.5G" or "250M".
        This function converts them to exact integer counts.

        Args:
            flops (str): FLOP count string (e.g., "1.5G", "250M", "10K")

        Returns:
            int: Exact FLOP count as integer

        Raises:
            ValueError: If the FLOP string format is not recognized

        """
        if "T" in flops:  # Teraflops (10^12)
            return int(float(flops.replace("T", "").strip()) * 1e12)
        elif "G" in flops:  # Gigaflops (10^9)
            return int(float(flops.replace("G", "").strip()) * 1e9)
        elif "M" in flops:  # Megaflops (10^6)
            return int(float(flops.replace("M", "").strip()) * 1e6)
        elif "K" in flops:  # Kiloflops (10^3)
            return int(float(flops.replace("K", "").strip()) * 1e3)
        raise ValueError(f"Invalid FLOPs string: {flops}")

    try:
        # Use DeepSpeed profiler to get FLOP count
        # Returns tuple: (flops_str, macs, params)
        flops, _, _ = get_model_profile(
            model=model,
            input_shape=tuple(input_tensor.shape),
            print_profile=False,  # Suppress detailed output
        )
        return parse_flops(flops), None
    except Exception as e:
        # Handle profiling failures gracefully
        print(f"⚠️ Could not compute FLOPs: {e}")
        return -1, str(e)  # Return error indicator and message


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
    """Benchmark forward-pass runtime for model with statistical accuracy.

    Performs precise timing measurements using proper warmup periods and
    device synchronization. The warmup phase allows CUDA kernels to initialize
    and caches to stabilize, ensuring accurate timing measurements.

    This function supports both inference and training mode timing:
    - Inference: Model in eval mode with torch.no_grad() context
    - Training: Model in train mode with gradient tracking enabled

    Args:
        model (torch.nn.Module | pl.LightningModule): The model to benchmark
        mode (str): Either "inference" or "training" mode
        device (torch.device): Device for computation (CPU/CUDA)
        input_data (torch.Tensor): Input tensor for model forward pass
        warmup_iterations (int, optional): Warmup iterations to stabilize timing.
                                         Defaults to 50.
        tot_iterations (int, optional): Total iterations including warmup.
                                      Defaults to 100.
        desc (str | None, optional): Description for progress bar. Defaults to None.

    Returns:
        list[float]: List of execution times per iteration in seconds (post warmup).
                    Length will be (tot_iterations - warmup_iterations).

    Raises:
        AssertionError: If mode is not "inference" or "training"

    Example:
        >>> times = measure_model_time(model, "inference", device, input_data)
        >>> avg_time = sum(times) / len(times)
        >>> print(f"Average inference time: {avg_time*1000:.2f}ms")

    Note:
        - Uses torch.cuda.synchronize() for accurate CUDA timing
        - Progress bar can be disabled by setting desc=None
        - Warmup iterations are essential for stable measurements

    """
    assert mode in {"inference", "training"}, "mode must be 'inference' or 'training'"

    # Configure model state and choose appropriate context manager
    if mode == "inference":
        model.eval()  # Set to evaluation mode (disables dropout, batch norm training)
        ctx = torch.no_grad()  # Disable gradient computation for memory efficiency
    else:
        model.train()  # Set to training mode (enables dropout, batch norm training)
        ctx = nullcontext()  # Keep gradients enabled for training mode timing

    # Initialize timing storage and progress bar description
    times: list[float] = []
    bar_desc = desc or f"{mode.title()} | {getattr(model, 'name', 'Model')} | timing"

    # Execute timing loop with proper context management
    with ctx:
        for i in tqdm(range(tot_iterations), desc=bar_desc, leave=False):
            # Warmup phase: execute model without timing to stabilize performance
            if i < warmup_iterations:
                _ = model(input_data)  # Forward pass without timing
                # Synchronize CUDA operations to ensure completion
                if device.type == "cuda":
                    torch.cuda.synchronize()
                continue  # Skip to next iteration without recording time

            # Timing phase: measure actual execution time
            start = time.perf_counter()  # High-precision timer start
            _ = model(input_data)  # Execute forward pass

            # Ensure all CUDA operations complete before stopping timer
            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()  # High-precision timer end
            times.append(end - start)  # Store timing result

    return times


def get_input_data_for_model(
    model: torch.nn.Module | pl.LightningModule, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Generate appropriate synthetic input data for the model based on its architecture.

    Different CSI prediction models expect different input formats:
    - Separate antenna models: Real tensors with flattened complex data
    - Gather antenna models: Complex tensors with explicit antenna dimension

    This function automatically detects the model type and generates compatible
    synthetic data for benchmarking purposes.

    Args:
        model (torch.nn.Module | pl.LightningModule): The model to generate input for
        batch_size (int): Number of samples in the batch
        device (torch.device): Device to create the tensor on (CPU/CUDA)

    Returns:
        torch.Tensor: Synthetic input tensor with appropriate shape and dtype:
            - For separate_antennas: [batch_size * num_antennas, hist_len, num_subcarriers*2] (real)
            - For gather_antennas: [batch_size, num_antennas, hist_len, num_subcarriers] (complex)

    Note:
        - Uses random normal distribution for synthetic data generation
        - Tensor shapes match the expected input format for CSI data
        - Complex tensors are created from separate real and imaginary components

    """
    if hasattr(model, "is_separate_antennas") and model.is_separate_antennas:
        # Separate antenna processing: each antenna treated as independent sample
        # Input format: [batch_size * num_antennas, hist_len, num_subcarriers*2] (real)
        # Complex data is flattened into real/imaginary pairs in the last dimension
        shape = (batch_size * TOT_ANTENNAS, HIST_LEN, NUM_SUBCARRIERS * 2)
        return torch.randn(*shape, device=device, dtype=torch.float32)
    else:
        # Gather antenna processing: all antennas processed together
        # Input format: [batch_size, num_antennas, hist_len, num_subcarriers] (complex)
        # Native complex tensor representation
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
    """Measure inference time statistics with quick measurement configuration.

    A simplified wrapper around measure_model_time() optimized for quick
    performance assessments. Uses fewer iterations than the full benchmarking
    function but still includes proper warmup for accuracy.

    This function is useful for:
    - Quick model comparison during development
    - Integration into larger analysis pipelines
    - Memory-constrained environments where full benchmarking isn't feasible

    Args:
        model (torch.nn.Module | pl.LightningModule): Model to benchmark
        input_data (torch.Tensor): Input tensor for the model forward pass
        device (torch.device): Device to run benchmarking on
        num_iterations (int, optional): Number of timing iterations. Defaults to 10.
        warmup_iterations (int, optional): Number of warmup iterations. Defaults to 5.

    Returns:
        dict[str, float]: Dictionary containing timing statistics in seconds:
            - "mean": Average inference time
            - "std": Standard deviation of inference times
            - "min": Minimum inference time observed
            - "max": Maximum inference time observed

    Example:
        >>> stats = measure_inference_time_stats(model, input_data, device)
        >>> print(f"Inference: {stats['mean']*1000:.2f}±{stats['std']*1000:.2f}ms")

    """
    # Use the main timing function with reduced iteration counts
    times = measure_model_time(
        model=model,
        mode="inference",
        device=device,
        input_data=input_data,
        warmup_iterations=warmup_iterations,
        tot_iterations=warmup_iterations + num_iterations,
        desc=None,  # Disable progress bar for quick measurements
    )

    # Compute statistical measures from timing data
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
    """Compute comprehensive computational metrics for a model.

    This is a high-level function that combines all metric computation utilities
    into a single call. It provides a complete computational profile including
    model complexity, computational cost, and runtime performance.

    The function handles errors gracefully, continuing computation even if some
    metrics fail (e.g., FLOP counting for unsupported models).

    Args:
        model (torch.nn.Module): Model to analyze
        device (torch.device): Device to run computations on
        batch_size (int, optional): Batch size for benchmarking. Defaults to 1.
        inference_iterations (int, optional): Timing iterations. Defaults to 10.
        warmup_iterations (int, optional): Warmup iterations. Defaults to 5.

    Returns:
        tuple[dict[str, float], dict[str, str]]:
            - metrics_dict: All computed metrics with standardized names
            - errors_dict: Any error messages encountered during computation

    Metrics included:
        - Parameter counts (total, trainable, in millions)
        - FLOP counts (raw, millions, billions)
        - Inference timing (mean, std, min, max in both ms and seconds)

    Example:
        >>> metrics, errors = compute_all_metrics(model, device)
        >>> if not errors:
        ...     print(f"Model: {metrics['total_params_M']:.1f}M params, {metrics['gflops']:.1f}G FLOPs")

    """
    # Initialize result dictionaries
    metrics = {}
    errors = {}

    # Generate appropriate input data for the model architecture
    sample_input = get_input_data_for_model(model, batch_size, device)

    # Compute parameter complexity metrics
    metrics["total_params"] = count_total_parameters(model)
    metrics["trainable_params"] = count_trainable_parameters(model)
    metrics["total_params_M"] = metrics["total_params"] / 1e6  # Convert to millions
    metrics["trainable_params_M"] = metrics["trainable_params"] / 1e6  # Convert to millions

    # Compute computational cost metrics (FLOPs)
    metrics["flops"], flops_error = count_flops(model, sample_input)
    if flops_error:
        errors["flops_error"] = flops_error

    # Convert FLOP counts to different scales for readability
    if metrics["flops"] > 0:
        metrics["mflops"] = metrics["flops"] / 1e6  # Millions of FLOPs
        metrics["gflops"] = metrics["flops"] / 1e9  # Billions of FLOPs
    else:
        # Use -1 to indicate failed computation
        metrics["mflops"] = -1
        metrics["gflops"] = -1

    # Measure runtime performance metrics
    try:
        timing_stats = measure_inference_time_stats(
            model, sample_input, device, num_iterations=inference_iterations, warmup_iterations=warmup_iterations
        )

        # Store timing metrics in multiple units for different use cases
        # Milliseconds for human readability
        metrics["inference_time_mean_ms"] = timing_stats["mean"] * 1000
        metrics["inference_time_std_ms"] = timing_stats["std"] * 1000
        metrics["inference_time_min_ms"] = timing_stats["min"] * 1000
        metrics["inference_time_max_ms"] = timing_stats["max"] * 1000

        # Seconds for precise calculations
        metrics["inference_time_mean_s"] = timing_stats["mean"]
        metrics["inference_time_std_s"] = timing_stats["std"]
    except Exception as e:
        # Handle timing failures gracefully
        errors["timing_error"] = str(e)
        # Set all timing metrics to -1 to indicate failure
        timing_keys = [
            "inference_time_mean_ms",
            "inference_time_std_ms",
            "inference_time_min_ms",
            "inference_time_max_ms",
            "inference_time_mean_s",
            "inference_time_std_s",
        ]
        for key in timing_keys:
            metrics[key] = -1

    return metrics, errors


def log_computational_metrics(metrics: dict[str, float], model_name: str) -> None:
    """Log computational metrics in a formatted, human-readable way.

    Provides a standardized format for displaying model performance metrics
    with appropriate units and emoji indicators for visual clarity. Handles
    missing or failed metrics gracefully.

    Args:
        metrics (dict[str, float]): Dictionary of computed metrics from compute_all_metrics()
        model_name (str): Name of the model for identification in logs

    Output Format:
        🔧 Computational Metrics for ModelName:
           📊 Parameters: X,XXX total (X.XXM), X,XXX trainable (X.XXM)
           ⚡ FLOPs: X,XXX,XXX (X.XXXG) or "Could not compute" if failed
           ⏱️  Inference Time: XX.XX±XX.XXms

    Example:
        >>> metrics = {"total_params": 1500000, "trainable_params": 1500000, ...}
        >>> log_computational_metrics(metrics, "MyModel")
        🔧 Computational Metrics for MyModel:
           📊 Parameters: 1,500,000 total (1.50M), 1,500,000 trainable (1.50M)
           ⚡ FLOPs: 2,500,000,000 (2.500G)
           ⏱️  Inference Time: 15.23±1.45ms

    """
    print(f"🔧 Computational Metrics for {model_name}:")

    # Display parameter information with comma separators and millions conversion
    print(
        f"   📊 Parameters: {metrics['total_params']:,} total ({metrics['total_params_M']:.2f}M), "
        f"{metrics['trainable_params']:,} trainable ({metrics['trainable_params_M']:.2f}M)"
    )

    # Display FLOP information with appropriate handling of computation failures
    if metrics["flops"] > 0:
        print(f"   ⚡ FLOPs: {metrics['flops']:,} ({metrics['gflops']:.3f}G)")
    else:
        print("   ⚡ FLOPs: Could not compute")

    # Display timing information with mean±std format for statistical significance
    print(f"   ⏱️  Inference Time: {metrics['inference_time_mean_ms']:.2f}±{metrics['inference_time_std_ms']:.2f}ms")
