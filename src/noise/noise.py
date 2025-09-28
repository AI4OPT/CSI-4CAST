"""Noise Generation Module for CSI Data

This module provides various noise generation functions for Channel State Information (CSI) data.
It supports different types of noise including vanilla Gaussian noise, phase noise,
package drop noise, and burst noise to simulate realistic wireless channel conditions.

Functions:
    gen_vanilla_noise_snr: Generate complex white Gaussian noise at specified SNR
    gen_phase_noise_nd: Generate phase noise with specified noise degree
    gen_packagedrop_noise_nd: Generate package drop noise (simulates data loss)
    gen_burst_noise_nd: Generate burst noise with bell-shaped temporal pattern

Author: CSI-4CAST Team
"""

import math

import torch


def gen_vanilla_noise_snr(H: torch.Tensor, SNR: float) -> torch.Tensor:
    """Generate complex white Gaussian noise for a tensor H at a given SNR.

    This function generates additive white Gaussian noise (AWGN) that matches the power
    characteristics of the input signal at the specified signal-to-noise ratio.
    The noise is complex-valued with independent real and imaginary components.

    Args:
        H (torch.Tensor): Complex-valued input tensor with shape
                         [batch_size, num_antennas, hist_len, num_subcarriers].
                         Must be dtype=torch.complex64 or torch.complex128.
        SNR (float): Desired signal-to-noise ratio in dB. Higher values mean less noise.

    Returns:
        torch.Tensor: Complex noise tensor of the same shape and dtype as H.
                     The noise power is calibrated to achieve the specified SNR.

    Raises:
        ValueError: If H is not a complex-valued tensor.

    Example:
        >>> data = torch.complex(torch.randn(4, 32, 16, 300), torch.randn(4, 32, 16, 300))
        >>> noise = gen_vanilla_noise_snr(data, SNR=20.0)  # 20 dB SNR
        >>> noisy_data = data + noise

    """
    if not H.is_complex():
        raise ValueError(f"H must be a complex tensor, got dtype={H.dtype}")

    # Step 1: Convert SNR from dB to linear scale
    # SNR_linear = 10^(SNR_dB/10), so noise variance factor = 1/SNR_linear
    sigma = 10 ** (-SNR / 10)  # Noise variance factor in linear scale

    # Step 2: Compute average signal power (E[|H|^2]) over all tensor elements
    # This gives us the reference power level for noise scaling
    power = torch.mean(torch.abs(H) ** 2)  # Real-valued scalar

    # Step 3: Calculate standard deviation for each real/imaginary component
    # For complex noise: total power = 2 * (std_real^2 + std_imag^2) = 2 * 2 * std^2
    # So std = sqrt(sigma * power / 2) for each component
    scale = math.sqrt(sigma / 2) * torch.sqrt(power)

    # Step 4: Generate independent Gaussian noise for real and imaginary parts
    # Each component follows N(0, scale^2) distribution
    real_part = torch.randn(H.shape, dtype=H.real.dtype, device=H.device)
    imag_part = torch.randn(H.shape, dtype=H.imag.dtype, device=H.device)

    # Step 5: Combine into complex noise tensor and apply scaling
    noise = torch.complex(real_part, imag_part) * scale

    return noise


def gen_phase_noise_nd(data: torch.Tensor, noise_degree: float = 0.01) -> torch.Tensor:
    """Generate phase noise for complex-valued torch tensor.

    Phase noise simulates oscillator instability in wireless communication systems.
    It adds random perturbations to the phase of the complex signal while preserving
    the magnitude. This type of noise is common in RF systems due to phase-locked
    loop (PLL) imperfections.

    Args:
        data (torch.Tensor): Complex-valued input tensor with shape
                           [batch_size, num_antennas, hist_len, num_subcarriers].
                           Must be dtype=torch.complex64 or torch.complex128.
        noise_degree (float, optional): Standard deviation of the phase noise in radians.
                                      Higher values create more phase distortion.
                                      Defaults to 0.01.

    Returns:
        torch.Tensor: Complex phase noise tensor of the same shape and dtype as data.
                     This represents the difference between the original and phase-noisy signal.

    Raises:
        ValueError: If data is not a complex-valued tensor.

    Example:
        >>> data = torch.complex(torch.randn(4, 32, 16, 300), torch.randn(4, 32, 16, 300))
        >>> phase_noise = gen_phase_noise_nd(data, noise_degree=0.05)
        >>> noisy_data = data + phase_noise

    """
    if not data.is_complex():
        raise ValueError(f"data must be a complex tensor, got dtype={data.dtype}")

    # Early return for zero noise degree to avoid unnecessary computation
    if noise_degree == 0:
        return torch.zeros_like(data)

    # Step 1: Extract magnitude and phase from complex data
    # Convert to polar representation: data = magnitude * exp(j * phase)
    magnitude = torch.abs(data)  # Magnitude (amplitude) of each complex sample
    phase = torch.angle(data)  # Phase angle in radians [-π, π]

    # Step 2: Generate random phase perturbations
    # Phase noise follows Gaussian distribution with specified standard deviation
    phase_noise = torch.randn_like(phase) * noise_degree
    new_phase = phase + phase_noise

    # Step 3: Compute the difference between original and phase-noisy signals
    # Convert back to rectangular form and compute the noise component
    real_delta = magnitude * (torch.cos(new_phase) - torch.cos(phase))
    imag_delta = magnitude * (torch.sin(new_phase) - torch.sin(phase))
    phase_noise = torch.complex(real_delta, imag_delta)

    return phase_noise.to(data.dtype)


def gen_packagedrop_noise_nd(data: torch.Tensor, noise_degree: float = 0.01) -> torch.Tensor:
    """Generate package drop noise for complex-valued torch tensor.

    Package drop noise simulates data packet loss in wireless communication systems.
    This occurs when entire time slots of data are lost due to network congestion,
    buffer overflows, or transmission errors. The function randomly sets entire
    time steps to zero based on the specified drop probability.

    Args:
        data (torch.Tensor): Complex-valued input tensor with shape
                           [batch_size, num_antennas, hist_len, num_subcarriers].
                           The hist_len dimension represents time steps.
        noise_degree (float, optional): Drop probability for each time step (0.0 to 1.0).
                                      Higher values mean more frequent packet drops.
                                      Defaults to 0.01 (1% drop rate).

    Returns:
        torch.Tensor: Noise tensor representing the difference between original and dropped data.
                     Same shape and dtype as input. Negative values indicate dropped samples.

    Example:
        >>> data = torch.complex(torch.randn(4, 32, 16, 300), torch.randn(4, 32, 16, 300))
        >>> drop_noise = gen_packagedrop_noise_nd(data, noise_degree=0.05)  # 5% drop rate
        >>> noisy_data = data + drop_noise

    """
    # Early return for zero drop rate to avoid unnecessary computation
    if noise_degree == 0:
        return torch.zeros_like(data)

    # Extract tensor dimensions: Batch, Antennas, Time, Subcarriers
    B, _, T, _ = data.shape

    # Step 1: Generate random drop mask for each batch and time step
    # Use Bernoulli distribution to randomly select time steps to drop
    drop_mask = torch.bernoulli(torch.full((B, T), noise_degree, device=data.device)).bool()

    # Step 2: Expand mask to match full tensor dimensions
    # Shape: [B, T] -> [B, 1, T, 1] -> [B, num_antennas, T, num_subcarriers]
    mask_expanded = drop_mask.unsqueeze(1).unsqueeze(-1).expand_as(data)

    # Step 3: Apply packet drops by setting selected time steps to zero
    noisy_data = data.clone()
    noisy_data[mask_expanded] = 0.0

    # Step 4: Compute noise as the difference (will be negative for dropped samples)
    noise = noisy_data - data

    return noise.to(data.dtype)


def gen_burst_noise_nd(data: torch.Tensor, noise_degree: float = 0.01) -> torch.Tensor:
    """Generate burst noise with bell-shaped temporal envelope.

    Burst noise simulates sudden interference events in wireless communication,
    such as microwave oven interference, radar pulses, or other transient
    electromagnetic disturbances. The noise appears as short-duration bursts
    with a bell-shaped (Gaussian) amplitude profile over time.

    Args:
        data (torch.Tensor): Complex-valued input tensor with shape
                           [batch_size, num_antennas, hist_len, num_subcarriers].
        noise_degree (float, optional): Controls both burst amplitude and occurrence probability.
                                      Higher values create stronger and more frequent bursts.
                                      Defaults to 0.01.

    Returns:
        torch.Tensor: Complex burst noise tensor with same shape and dtype as input.
                     Contains bell-shaped noise bursts at random time locations.

    Note:
        - Burst amplitude scales as 2 * noise_degree
        - Burst probability scales as 0.05 * noise_degree * 40
        - Each burst has a fixed length of 10 time steps
        - Bursts follow a geometric distribution for timing

    Example:
        >>> data = torch.complex(torch.randn(4, 32, 16, 300), torch.randn(4, 32, 16, 300))
        >>> burst_noise = gen_burst_noise_nd(data, noise_degree=0.02)
        >>> noisy_data = data + burst_noise

    """
    # Extract tensor dimensions
    B, N, T, K = data.shape

    # Configure burst parameters based on noise degree
    burst_amplitude = 2 * noise_degree  # Amplitude scaling factor
    burst_length = 10  # Fixed burst duration (time steps)
    burst_prob = 0.05 * noise_degree * 40  # Probability of burst occurrence

    # Initialize noise tensor with zeros
    noise = torch.zeros_like(data)

    # Early return for zero noise degree
    if noise_degree == 0:
        return noise

    # Step 1: Generate burst timing using geometric distribution
    # Geometric distribution models the "waiting time" until a burst occurs
    geo = torch.distributions.Geometric(burst_prob)
    burst_indicator = geo.sample((B,)) - 1  # Convert to 0-based indexing

    # Step 2: Generate random start positions for bursts
    burst_start = torch.randint(0, T, (B,), device=data.device)

    # Step 3: Generate bursts for each batch sample
    for b in range(B):
        # Check if a burst should occur within the time window
        if burst_indicator[b] < T:
            # Calculate burst time window
            s = burst_start[b].item()  # Start time
            e = min(s + burst_length, T)  # End time (clipped to tensor bounds)
            L = int(e - s)  # Actual burst length

            if L == 0:
                continue

            # Step 4: Create bell-shaped (Gaussian) envelope
            # Maps time indices to [-1, 1] range for symmetric bell shape
            lin = torch.linspace(-1, 1, L, device=data.device)
            bell = burst_amplitude * torch.exp(-0.5 * lin**2)
            bell = bell.view(1, 1, L, 1)  # Reshape for broadcasting: [1, 1, L, 1]

            # Step 5: Generate complex Gaussian noise modulated by bell envelope
            real = torch.randn((1, N, L, K), device=data.device) * bell
            imag = torch.randn((1, N, L, K), device=data.device) * bell
            burst = torch.complex(real, imag)

            # Step 6: Place burst in the noise tensor at calculated time window
            noise[b, :, s:e, :] = burst

    return noise.to(data.dtype)


if __name__ == "__main__":
    """
    Demo script to test noise generation functions.
    
    This section demonstrates the usage of all noise generation functions
    with sample CSI data tensors.
    """
    # Create sample complex CSI data tensor
    # Shape: [batch_size=5, num_antennas=32, hist_len=16, num_subcarriers=300]
    fake_input_real = torch.randn(5, 32, 16, 300)
    fake_input_imag = torch.randn(5, 32, 16, 300)
    fake_input = torch.complex(fake_input_real, fake_input_imag)

    # Test burst noise generation
    print("Testing burst noise generation...")
    noise = gen_burst_noise_nd(fake_input, 0.01)
    print(f"Burst noise shape: {noise.shape}")
    print(f"Burst noise dtype: {noise.dtype}")

    # Test other noise types
    print("\nTesting other noise types...")

    # Vanilla (AWGN) noise
    vanilla_noise = gen_vanilla_noise_snr(fake_input, SNR=20.0)
    print(f"Vanilla noise shape: {vanilla_noise.shape}")

    # Phase noise
    phase_noise = gen_phase_noise_nd(fake_input, noise_degree=0.05)
    print(f"Phase noise shape: {phase_noise.shape}")

    # Package drop noise
    drop_noise = gen_packagedrop_noise_nd(fake_input, noise_degree=0.02)
    print(f"Package drop noise shape: {drop_noise.shape}")

    print("\nAll noise generation functions tested successfully!")
