import math

import torch


def gen_vanilla_noise_snr(H: torch.Tensor, SNR: float) -> torch.Tensor:
    """Generate complex white Gaussian noise for a tensor H at a given SNR.

    Args:
        H (torch.Tensor): Complex-valued tensor (dtype=torch.complex64 or torch.complex128).
        SNR (float): Desired signal-to-noise ratio in dB.

    Returns:
        torch.Tensor: Complex noise tensor of the same shape and dtype as H.

    """
    if not H.is_complex():
        raise ValueError(f"H must be a complex tensor, got dtype={H.dtype}")

    # 1) Compute noise variance factor sigma
    sigma = 10 ** (-SNR / 10)  # linear scale

    # 2) Compute average signal power (E[|H|^2]) over all elements
    power = torch.mean(torch.abs(H) ** 2)  # real scalar

    # 3) Determine the standard deviation for each real/imag component:
    #    sqrt(sigma/2 * power)
    scale = math.sqrt(sigma / 2) * torch.sqrt(power)

    # 4) Generate real and imaginary Gaussian components N(0,1)
    real_part = torch.randn(H.shape, dtype=H.real.dtype, device=H.device)
    imag_part = torch.randn(H.shape, dtype=H.imag.dtype, device=H.device)

    # 5) Form a complex noise tensor and scale it
    noise = torch.complex(real_part, imag_part) * scale

    return noise


def gen_phase_noise_nd(data: torch.Tensor, noise_degree: float = 0.01) -> torch.Tensor:
    """Generate phase noise for complex-valued torch tensor.

    Args:
        data (torch.Tensor): Complex-valued tensor (dtype=torch.complex64 or torch.complex128).
        noise_degree (float): Standard deviation of the phase noise.

    Returns:
        torch.Tensor: Complex phase noise tensor of the same shape and dtype as data.

    """
    if not data.is_complex():
        raise ValueError(f"data must be a complex tensor, got dtype={data.dtype}")

    if noise_degree == 0:
        return torch.zeros_like(data)

    # [batch_size, num_antenna, hist_len, num_subcarriers]
    magnitude = torch.abs(data)
    phase = torch.angle(data)

    phase_noise = torch.randn_like(phase) * noise_degree
    new_phase = phase + phase_noise

    real_delta = magnitude * (torch.cos(new_phase) - torch.cos(phase))
    imag_delta = magnitude * (torch.sin(new_phase) - torch.sin(phase))
    phase_noise = torch.complex(real_delta, imag_delta)

    return phase_noise.to(data.dtype)


def gen_packagedrop_noise_nd(data: torch.Tensor, noise_degree: float = 0.01) -> torch.Tensor:
    """Generate package drop noise for complex-valued torch tensor.

    Args:
        data (torch.Tensor): Complex-valued tensor of shape [batch, antenna, hist_len, subcarriers].
        noise_degree (float): Drop ratio (probability of dropping each time step).

    Returns:
        torch.Tensor: Noise tensor representing the difference between original and dropped data.

    """
    if noise_degree == 0:
        return torch.zeros_like(data)

    B, _, T, _ = data.shape
    drop_mask = torch.bernoulli(torch.full((B, T), noise_degree, device=data.device)).bool()  # True = drop

    # expand mask to the full tensor shape
    mask_expanded = drop_mask.unsqueeze(1).unsqueeze(-1).expand_as(data)
    noisy_data = data.clone()
    noisy_data[mask_expanded] = 0.0

    noise = noisy_data - data

    return noise.to(data.dtype)


def gen_burst_noise_nd(data: torch.Tensor, noise_degree: float = 0.01) -> torch.Tensor:
    """PyTorch version of genBurstNoise_np: generates a bell-shaped burst of complex noise.

    Args:
        data (torch.Tensor): Complex input of shape [B batchsize, N num_antennas, T hist_len, K num_subcarriers]
        noise_degree (float): Controls burst amplitude and burst probability.

    Returns:
        torch.Tensor: Complex noise tensor, same shape & dtype as `data`.

    """
    B, N, T, K = data.shape
    burst_amplitude = 2 * noise_degree
    burst_length = 10
    burst_prob = 0.05 * noise_degree * 40

    # start with zeros of the same dtype & device
    noise = torch.zeros_like(data)
    if noise_degree == 0:
        return noise

    # geometric indicators for each batch
    geo = torch.distributions.Geometric(burst_prob)
    burst_indicator = geo.sample((B,)) - 1  # shape [B]
    burst_start = torch.randint(0, T, (B,), device=data.device)

    for b in range(B):
        if burst_indicator[b] < T:
            s = burst_start[b].item()
            e = min(s + burst_length, T)
            L = int(e - s)
            if L == 0:
                continue

            # bell-shaped envelope
            lin = torch.linspace(-1, 1, L, device=data.device)
            bell = burst_amplitude * torch.exp(-0.5 * lin**2)
            bell = bell.view(1, 1, L, 1)  # [1,1,L,1]

            # complex Gaussian noise
            real = torch.randn((1, N, L, K), device=data.device) * bell
            imag = torch.randn((1, N, L, K), device=data.device) * bell
            burst = torch.complex(real, imag)
            noise[b, :, s:e, :] = burst

    return noise.to(data.dtype)


if __name__ == "__main__":
    fake_input_real = torch.randn(5, 32, 16, 300)
    fake_input_imag = torch.randn(5, 32, 16, 300)
    fake_input = torch.complex(fake_input_real, fake_input_imag)

    noise = gen_burst_noise_nd(fake_input, 0.01)
    print(noise.shape)
    print(noise.dtype)
