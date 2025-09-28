"""Noise Testing Interface for CSI Data.

This module provides a unified interface for applying various types of noise
to Channel State Information (CSI) data during testing and evaluation.
It wraps the core noise generation functions with predefined SNR levels
and noise degrees for consistent testing across different scenarios.

Classes:
    Noise: Main interface class providing convenient methods for noise generation

"""

import json
from pathlib import Path
from typing import ClassVar

import torch

from src.noise.noise import (
    gen_burst_noise_nd,
    gen_packagedrop_noise_nd,
    gen_phase_noise_nd,
    gen_vanilla_noise_snr,
)


class Noise:
    """Unified noise generation interface for CSI data testing.

    This class provides convenient methods for applying different types of noise
    to CSI data with predefined SNR levels and noise degrees. It automatically
    maps SNR requirements to appropriate noise degrees using precomputed mappings.

    Attributes:
        list_vanilla_snr: Available SNR levels for vanilla (AWGN) noise in dB
        list_phase_snr: Available SNR levels for phase noise in dB
        list_burst_snr: Available SNR levels for burst noise in dB
        list_packagedrop_nd: Available noise degrees for package drop noise (drop rates)
        decide_nd: Mapping from SNR levels to noise degrees for phase/burst noise

    Methods:
        gen_phase_noise_snr: Generate phase noise at specified SNR level
        gen_burst_noise_snr: Generate burst noise at specified SNR level
        gen_packagedrop_noise_nd: Generate package drop noise at specified drop rate
        gen_vanilla_noise_snr: Generate vanilla AWGN at specified SNR level

    Aliases:
        vanilla: Alias for gen_vanilla_noise_snr
        phase: Alias for gen_phase_noise_snr
        burst: Alias for gen_burst_noise_snr
        packagedrop: Alias for gen_packagedrop_noise_nd

    """

    # Predefined SNR levels for different noise types (in dB)
    list_vanilla_snr: ClassVar[list[int]] = [0, 5, 10, 15, 20, 25]
    list_phase_snr: ClassVar[list[int]] = [10, 15, 20, 25]
    list_burst_snr: ClassVar[list[int]] = [10, 15, 20, 25]

    # Predefined noise degrees for package drop (drop probability)
    list_packagedrop_nd: ClassVar[list[float]] = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

    # Load precomputed SNR-to-noise-degree mappings
    path_decide_nd_json: ClassVar[Path] = Path("src/noise/results/decide_nd.json")
    with open(path_decide_nd_json) as f:
        decide_nd: ClassVar[dict[str, dict[str, dict[str, float]]]] = json.load(f)

    def gen_phase_noise_snr(self, data: torch.Tensor, snr: int) -> torch.Tensor:
        """Generate phase noise at specified SNR level.

        Args:
            data (torch.Tensor): Complex-valued CSI data tensor
            snr (int): Target SNR level in dB (must be in list_phase_snr)

        Returns:
            torch.Tensor: Phase noise tensor to be added to the data

        Raises:
            AssertionError: If SNR level is not supported

        """
        assert snr in self.list_phase_snr, f"snr must be in {self.list_phase_snr}"
        # Look up precomputed noise degree for target SNR
        nd = self.decide_nd["phase"][str(snr)]["noise_degree"]
        return gen_phase_noise_nd(data, nd)

    def gen_burst_noise_snr(self, data: torch.Tensor, snr: int) -> torch.Tensor:
        """Generate burst noise at specified SNR level.

        Args:
            data (torch.Tensor): Complex-valued CSI data tensor
            snr (int): Target SNR level in dB (must be in list_burst_snr)

        Returns:
            torch.Tensor: Burst noise tensor to be added to the data

        Raises:
            AssertionError: If SNR level is not supported

        """
        assert snr in self.list_burst_snr, f"snr must be in {self.list_burst_snr}"
        # Look up precomputed noise degree for target SNR
        nd = self.decide_nd["burst"][str(snr)]["noise_degree"]
        return gen_burst_noise_nd(data, nd)

    def gen_packagedrop_noise_nd(self, data: torch.Tensor, nd: float) -> torch.Tensor:
        """Generate package drop noise at specified drop rate.

        Args:
            data (torch.Tensor): Complex-valued CSI data tensor
            nd (float): Drop probability (0.0 to 1.0)

        Returns:
            torch.Tensor: Package drop noise tensor to be added to the data

        """
        return gen_packagedrop_noise_nd(data, nd)

    def gen_vanilla_noise_snr(self, data: torch.Tensor, snr: int) -> torch.Tensor:
        """Generate vanilla AWGN at specified SNR level.

        Args:
            data (torch.Tensor): Complex-valued CSI data tensor
            snr (int): Target SNR level in dB

        Returns:
            torch.Tensor: AWGN noise tensor to be added to the data

        """
        return gen_vanilla_noise_snr(data, snr)

    # Convenient aliases for shorter method names
    vanilla = gen_vanilla_noise_snr
    phase = gen_phase_noise_snr
    burst = gen_burst_noise_snr
    packagedrop = gen_packagedrop_noise_nd


if __name__ == "__main__":
    """
    Demo script to test the Noise interface class.

    This section demonstrates how to use the Noise class to generate
    different types of noise with various SNR levels and noise degrees.
    """
    # Initialize the noise generator
    noise = Noise()

    # Display available noise parameters
    print("Available noise parameters:")
    print(f"Phase noise SNR levels: {noise.list_phase_snr}")
    print(f"Burst noise SNR levels: {noise.list_burst_snr}")
    print(f"Package drop noise degrees: {noise.list_packagedrop_nd}")
    print(f"Vanilla noise SNR levels: {noise.list_vanilla_snr}")

    print(f"\nSNR-to-noise-degree mappings: {noise.decide_nd}")

    # Create sample CSI data tensor
    # Shape: [batch_size=4, num_antennas=32, hist_len=16, num_subcarriers=300]
    data = torch.complex(torch.randn(4, 32, 16, 300), torch.randn(4, 32, 16, 300))
    print(f"\nInput data shape: {data.shape}")

    # Test phase noise generation at different SNR levels
    print("\nTesting phase noise generation:")
    for snr in noise.list_phase_snr:
        noise_data = noise.phase(data, snr)
        print(f"Phase noise (SNR={snr}dB): {noise_data.shape}")
    print("--------------------------------")

    # Test burst noise generation at different SNR levels
    print("Testing burst noise generation:")
    for snr in noise.list_burst_snr:
        noise_data = noise.burst(data, snr)
        print(f"Burst noise (SNR={snr}dB): {noise_data.shape}")
    print("--------------------------------")

    # Test package drop noise generation at different drop rates
    print("Testing package drop noise generation:")
    for nd in noise.list_packagedrop_nd:
        noise_data = noise.packagedrop(data, nd)
        print(f"Package drop noise (drop_rate={nd}): {noise_data.shape}")
    print("--------------------------------")

    # Test vanilla AWGN generation at different SNR levels
    print("Testing vanilla AWGN generation:")
    for snr in noise.list_vanilla_snr:
        noise_data = noise.vanilla(data, snr)
        print(f"Vanilla noise (SNR={snr}dB): {noise_data.shape}")

    print("\nAll noise types tested successfully!")
