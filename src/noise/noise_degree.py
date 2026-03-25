"""Noise Degree Analysis and SNR Calibration.

This module performs comprehensive analysis of noise characteristics and calibrates
noise degrees to achieve target SNR levels. It generates mappings between noise
degrees and actual SNR values for different noise types, enabling consistent
noise application across different scenarios.

The main script performs:
1. Power computation for signal and noise
2. SNR analysis across different noise degrees
3. Reverse engineering to find optimal noise degrees for target SNRs
4. Generation of calibration mappings for scenarios

Functions:
    compute_power: Calculate average power of complex-valued signals
    compute_snr: Convert power ratio to SNR in decibels

"""

import torch


def compute_power(data: torch.Tensor) -> torch.Tensor:
    """Compute the average power of a complex-valued signal.

    Power is calculated as the mean of squared magnitudes across all tensor elements.

    Args:
        data (torch.Tensor): Complex-valued tensor with shape
                           [batch_size, num_antennas, hist_len, num_subcarriers].
                           Must be complex dtype.

    Returns:
        torch.Tensor: Real-valued scalar representing the average power.
                     Units are in power (squared magnitude).

    Example:
        >>> signal = torch.complex(torch.randn(4, 32, 16, 300), torch.randn(4, 32, 16, 300))
        >>> power = compute_power(signal)
        >>> print(f"Signal power: {power.item():.4f}")

    """
    return torch.mean(torch.abs(data) ** 2)


def compute_snr(power_signal: torch.Tensor, power_noise: torch.Tensor) -> torch.Tensor:
    """Convert signal and noise power to Signal-to-Noise Ratio in dB.

    SNR is computed as: SNR_dB = 10 * log10(P_signal / P_noise)
    where P_signal and P_noise are the average powers of signal and noise respectively.

    Args:
        power_signal (torch.Tensor): Average power of the signal (real scalar).
        power_noise (torch.Tensor): Average power of the noise (real scalar).
                                   Must be positive and non-zero.

    Returns:
        torch.Tensor: SNR value in decibels (dB). Higher values indicate better signal quality.

    Note:
        - Positive SNR means signal power > noise power
        - Zero SNR means signal power = noise power
        - Negative SNR means signal power < noise power

    Example:
        >>> signal_power = torch.tensor(1.0)
        >>> noise_power = torch.tensor(0.1)
        >>> snr_db = compute_snr(signal_power, noise_power)
        >>> print(f"SNR: {snr_db.item():.2f} dB")  # Should be ~10 dB

    """
    return 10 * torch.log10(power_signal / power_noise)


if __name__ == "__main__":
    """
    Main analysis script for noise degree calibration.

    This script performs comprehensive SNR analysis across different noise types
    and degrees, then generates optimal noise degree mappings for target SNR levels.

    Process:
    1. Load and normalize calibration datasets from different channel conditions
    2. Compute signal power for each calibration dataset
    3. Generate noise at various degrees and compute resulting SNR
    4. Perform reverse engineering to find optimal noise degrees for target SNRs
    5. Save results and mappings for use in testing

    Outputs:
    - snr.csv: Detailed SNR analysis results
    - decide_nd.json: Optimal noise degree mappings for target SNRs

    """
    from itertools import product
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from src.noise.noise import gen_burst_noise_nd, gen_phase_noise_nd
    from src.utils.data_utils import LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN, load_data
    from src.utils.dirs import DIR_DATA, DIR_OUTPUTS
    from src.utils.main_utils import make_logger
    from src.utils.norm_utils import normalize_input
    from src.utils.time_utils import get_current_time

    # Setup output directory with timestamp for reproducibility
    dir_output = Path(DIR_OUTPUTS) / "noise" / "noise_degree" / get_current_time()
    dir_output.mkdir(parents=True, exist_ok=True)
    logger = make_logger(dir_output)

    # Define output file paths
    path_df_snr = dir_output / "snr.csv"
    path_decide_nd_json = dir_output / "decide_nd.json"

    # Initialize DataFrame for storing SNR analysis results
    df_snr = pd.DataFrame(columns=["noise_type", "noise_degree", "snr_mean", "snr_std"])

    dir_data = Path(DIR_DATA)

    # ===== PHASE 1: Load and preprocess calibration datasets =====
    logger.info("Phase 1: Loading and preprocessing calibration datasets...")

    # Load calibration datasets from all combinations of channel conditions
    list_calibration_dataset = []
    list_signal_powers = []

    # Iterate through all combinations of channel model, delay spread, and minimum speed
    for cm, ds, ms in list(product(LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN)):
        logger.info(f"Loading calibration dataset: CM={cm}, DS={ds}, MS={ms}")

        # Load historical CSI data for calibration
        h_hist = load_data(
            dir_data=dir_data,
            list_cm=[cm],  # Channel model
            list_ds=[ds],  # Delay spread
            list_ms=[ms],  # Minimum speed (using training speeds)
            is_train=True,  # Use TRAINING data for calibration
            is_gen=False,  # Use real data (not generated)
            is_hist=True,  # Load historical data
            is_U2D=False,  # Use full CSI data (not U2D format)
        )

        # Normalize the CSI data to standard range
        h_hist, _ = normalize_input(h_hist, None, is_U2D=False)

        # Compute and store signal power for this dataset (done once for efficiency)
        power_signal = compute_power(h_hist)

        list_calibration_dataset.append(h_hist)
        list_signal_powers.append(power_signal)

    logger.info(f"Loaded {len(list_calibration_dataset)} calibration datasets")

    # ===== PHASE 2: Generate noise and compute SNR relationships =====
    logger.info("Phase 2: Analyzing SNR vs noise degree relationships...")

    # Analyze both burst and phase noise types
    for noise_type in ["burst", "phase"]:
        logger.info(f"Processing {noise_type} noise...")

        # Select appropriate noise generation function
        if noise_type == "burst":
            noise_func = gen_burst_noise_nd
        elif noise_type == "phase":
            noise_func = gen_phase_noise_nd
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")

        # Define noise degree sampling strategy
        # Dense sampling in low noise region (0.01-0.06) for better precision
        # Sparse sampling in high noise region (0.06-0.8) for coverage
        dense_range = np.arange(0.01, 0.06, 0.002)  # High precision range
        sparse_range = np.arange(0.06, 0.81, 0.01)  # Coverage range

        # Combine and clean up the noise degree list
        noise_degrees = np.concatenate([dense_range, sparse_range])
        noise_degrees = np.unique(noise_degrees)  # Remove duplicates and sort

        logger.info(f"Computing {len(noise_degrees)} noise degrees for {noise_type} noise")

        # Process each noise degree and compute SNR statistics
        for noise_degree in tqdm(noise_degrees, desc=f"Processing {noise_type} noise"):
            try:
                snr_list = []

                # Apply current noise degree to all calibration datasets
                for h_hist, power_signal in zip(list_calibration_dataset, list_signal_powers, strict=False):
                    try:
                        # Generate noise at current degree
                        noise = noise_func(h_hist, float(noise_degree))

                        # Compute power of generated noise (signal power pre-computed)
                        power_noise = compute_power(noise)

                        # Check for valid noise power to avoid division by zero
                        if power_noise > 1e-6:  # Small threshold for numerical stability
                            # Calculate SNR in dB
                            snr = compute_snr(power_signal, power_noise)
                            snr_list.append(snr.item())
                        else:
                            logger.warning(f"Near-zero noise power for {noise_type} with degree {noise_degree}")

                    except Exception as e:
                        logger.error(f"Error processing dataset for {noise_type} with degree {noise_degree}: {e}")
                        continue

                # Compute SNR statistics across all calibration datasets
                if snr_list:
                    snr_mean = np.mean(snr_list)  # Average SNR across calibration datasets
                    snr_std = np.std(snr_list)  # Standard deviation of SNR
                else:
                    snr_mean = np.nan
                    snr_std = np.nan
                    logger.warning(f"No valid SNR computed for {noise_type} with degree {noise_degree}")

            except Exception as e:
                logger.error(f"Error generating {noise_type} noise with degree {noise_degree}: {e}")
                snr_mean = np.nan
                snr_std = np.nan

            # Store results in DataFrame
            new_row = pd.DataFrame(
                {
                    "noise_type": [noise_type],
                    "noise_degree": [float(noise_degree)],
                    "snr_mean": [snr_mean],
                    "snr_std": [snr_std],
                }
            )
            df_snr = pd.concat([df_snr, new_row], ignore_index=True)

            # Save intermediate results for fault tolerance
            df_snr.to_csv(path_df_snr, index=False)

            # Log current progress
            logger.info(f"{noise_type} | degree {noise_degree:.3f} | SNR {snr_mean:.2f}±{snr_std:.2f} dB")

    logger.info(f"SNR analysis complete. Results saved to: {path_df_snr}")

    # ===== PHASE 3: Reverse engineering - find optimal noise degrees for target SNRs =====
    logger.info("Phase 3: Reverse engineering optimal noise degrees...")

    # Define target SNR levels for testing (in dB)
    # These are the SNR levels we want to achieve in our experiments
    target_snrs = {
        "phase": [10, 15, 20, 25],  # Phase noise target SNRs
        "burst": [10, 15, 20, 25],  # Burst noise target SNRs
    }

    # Initialize decision mapping dictionary
    decide_nd = {}

    # Process each noise type to find optimal noise degrees
    for noise_type in ["phase", "burst"]:
        logger.info(f"Finding optimal noise degrees for {noise_type} noise...")
        decide_nd[noise_type] = {}

        # Filter data for current noise type and remove invalid entries
        noise_data = df_snr[df_snr["noise_type"] == noise_type].copy()
        noise_data = noise_data.dropna()  # Remove NaN values

        if len(noise_data) == 0:
            logger.warning(f"No valid data available for {noise_type} noise")
            continue

        # Find optimal noise degree for each target SNR
        for target_snr in target_snrs[noise_type]:
            try:
                # Find the noise degree that produces SNR closest to target
                diff = (noise_data["snr_mean"] - target_snr).abs()
                closest_idx = diff.idxmin()

                # Extract the best match results
                noise_degree_row = noise_data.loc[closest_idx]
                best_noise_degree = noise_degree_row["noise_degree"]
                achieved_snr = noise_degree_row["snr_mean"]
                snr_error = abs(achieved_snr - target_snr)

                # Store the mapping
                decide_nd[noise_type][int(target_snr)] = {
                    "noise_degree": best_noise_degree,
                    "achieved_snr": achieved_snr,
                    "snr_diff": snr_error,
                }

                logger.info(
                    f"{noise_type}: Target {target_snr}dB → degree {best_noise_degree:.4f} "
                    f"(achieved {achieved_snr:.2f}dB, error {snr_error:.2f}dB)"
                )

            except Exception as e:
                logger.error(f"Error finding noise degree for {noise_type} SNR {target_snr}: {e}")
                # Store NaN values for failed cases
                decide_nd[noise_type][int(target_snr)] = {
                    "noise_degree": np.nan,
                    "achieved_snr": np.nan,
                    "snr_diff": np.nan,
                }

    # ===== PHASE 4: Save results and generate summary =====
    logger.info("Phase 4: Saving results and generating summary...")

    # Save decision mapping to JSON file for use in testing
    with open(path_decide_nd_json, "w") as f:
        json.dump(decide_nd, f, indent=2)

    logger.info(f"Reverse engineering complete. Decision mapping saved to: {path_decide_nd_json}")
    logger.info("Final summary of optimal noise degree mappings:")

    # Log the final mappings in a readable format
    for noise_type, snr_mappings in decide_nd.items():
        logger.info(f"\n{noise_type.upper()} NOISE MAPPINGS:")
        for target_snr, mapping in snr_mappings.items():
            if not np.isnan(mapping["noise_degree"]):
                logger.info(
                    f"  SNR {target_snr:2d}dB: degree {mapping['noise_degree']:.4f} "
                    f"(achieved {mapping['achieved_snr']:.2f}dB, error {mapping['snr_diff']:.2f}dB)"
                )
            else:
                logger.warning(f"  SNR {target_snr:2d}dB: FAILED TO FIND MAPPING")

    logger.info("\nAnalysis complete! Results available at:")
    logger.info(f"  - Detailed SNR data: {path_df_snr}")
    logger.info(f"  - Noise degree mappings: {path_decide_nd_json}")
