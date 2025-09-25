import torch


def compute_power(data: torch.Tensor) -> torch.Tensor:
    """
    compute the power of the signal or the noise
    - data is complex-valued tensor with shape [batch_size, num_antennas, hist_len, num_subcarriers]
    - return the power of the signal or the noise (real scalar)
    """
    return torch.mean(torch.abs(data) ** 2)


def compute_snr(power_signal: torch.Tensor, power_noise: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(power_signal / power_noise)


if __name__ == "__main__":
    """
    compute the relationship between the noise degree and the SNR
    - load the regular testing dataset under tdd scenario
    - compute the SNR for each noise degree
    - save the results in a csv file
    - do the reverse engineering to find the noise degree for each SNR
    - save the results in a csv file or json file
    - plot the results
    """

    import json
    from datetime import datetime
    from itertools import product
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from src.noise.noise import gen_burst_noise_nd, gen_phase_noise_nd
    from src.utils.main_utils import make_logger
    from src.utils.normalization import normalize_input
    from tqdm import tqdm

    from src.utils.data_utils import LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TEST, load_data
    from src.utils.dirs import DIR_DATA, DIR_OUTPUTS

    dir_output = Path(DIR_OUTPUTS) / "noise" / "noise_degree" / datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_output.mkdir(parents=True, exist_ok=True)
    logger = make_logger(dir_output)

    path_df_snr = dir_output / "snr.csv"
    # Initialize DataFrame for storing results
    df_snr = pd.DataFrame(columns=["noise_type", "noise_degree", "snr_mean", "snr_std"])

    path_decide_nd_json = dir_output / "decide_nd.json"

    dir_data = Path(DIR_DATA)

    # load the regular testing dataset and compute signal powers once
    list_testing_dataset = []
    list_signal_powers = []

    for cm, ds, ms in list(product(LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TEST)):
        h_hist = load_data(
            dir_data=dir_data,
            list_cm=[cm],
            list_ds=[ds],
            list_ms=[ms],
            is_train=False,
            is_gen=False,
            is_hist=True,
            is_U2D=False,
        )

        # normalize the data
        h_hist, _ = normalize_input(h_hist, None, is_U2D=False)
        # compute signal power once for this dataset
        power_signal = compute_power(h_hist)

        list_testing_dataset.append(h_hist)
        list_signal_powers.append(power_signal)

    # gen noise and compute snr
    for noise_type in ["burst", "phase"]:
        if noise_type == "burst":
            noise_func = gen_burst_noise_nd
        elif noise_type == "phase":
            noise_func = gen_phase_noise_nd
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")

        # Create a list of noise degrees with dense sampling in 0.01-0.06 range
        # Dense range: 0.01 to 0.06 with 0.002 increments
        dense_range = np.arange(0.01, 0.06, 0.002)
        # Sparse range: 0.06 to 0.8 with 0.01 increments
        sparse_range = np.arange(0.06, 0.81, 0.01)
        # Combine both ranges
        noise_degrees = np.concatenate([dense_range, sparse_range])
        # Remove potential duplicates and sort
        noise_degrees = np.unique(noise_degrees)

        for noise_degree in tqdm(noise_degrees, desc=f"Processing {noise_type} noise"):
            try:
                snr_list = []

                # Apply noise to each dataset and compute SNR
                for h_hist, power_signal in zip(list_testing_dataset, list_signal_powers, strict=False):
                    try:
                        # Generate noise
                        noise = noise_func(h_hist, float(noise_degree))

                        # Compute power of noise only (signal power already computed)
                        power_noise = compute_power(noise)

                        # Avoid division by zero
                        if power_noise > 1e-6:  # Very small threshold
                            snr = compute_snr(power_signal, power_noise)
                            snr_list.append(snr.item())
                        else:
                            logger.warning(f"Warning: Zero noise power for {noise_type} with degree {noise_degree}")

                    except Exception as e:
                        logger.error(f"Error processing dataset for {noise_type} with degree {noise_degree}: {e}")
                        continue

                # Compute statistics
                if snr_list:
                    snr_mean = np.mean(snr_list)
                    snr_std = np.std(snr_list)
                else:
                    snr_mean = np.nan
                    snr_std = np.nan
                    logger.warning(f"No valid SNR computed for {noise_type} with degree {noise_degree}")

            except Exception as e:
                logger.error(f"Error generating {noise_type} noise with degree {noise_degree}: {e}")
                snr_mean = np.nan
                snr_std = np.nan

            # Add to DataFrame
            new_row = pd.DataFrame(
                {
                    "noise_type": [noise_type],
                    "noise_degree": [float(noise_degree)],
                    "snr_mean": [snr_mean],
                    "snr_std": [snr_std],
                }
            )
            df_snr = pd.concat([df_snr, new_row], ignore_index=True)

            # Save DataFrame every iteration
            df_snr.to_csv(path_df_snr, index=False)

            # logging the current noise type, degree, and snr info
            logger.info(
                f"noise type {noise_type} | degree {noise_degree} | snr_mean {snr_mean:.2f} | snr_std {snr_std:.2f}"
            )

    logger.info(f"SNR analysis complete. Results saved to: {path_df_snr}")

    # Reverse engineering: find noise degrees for target SNRs
    logger.info("Starting reverse engineering...")

    # Load NoiseDegree target SNRs
    target_snrs = {
        "phase": [10, 15, 20, 25],
        "burst": [10, 15, 20, 25],
    }

    decide_nd = {}

    for noise_type in ["phase", "burst"]:
        decide_nd[noise_type] = {}
        noise_data = df_snr[df_snr["noise_type"] == noise_type].copy()
        noise_data = noise_data.dropna()  # Remove NaN values

        if len(noise_data) == 0:
            logger.warning(f"Warning: No valid data for {noise_type} noise")
            continue

        for target_snr in target_snrs[noise_type]:
            try:
                # Find closest SNR mean to target
                diff = (noise_data["snr_mean"] - target_snr).abs()
                closest_idx = diff.idxmin()

                # Get values as numpy scalars and convert to Python types
                noise_degree_row = noise_data.loc[closest_idx]
                best_noise_degree = noise_degree_row["noise_degree"]
                achieved_snr = noise_degree_row["snr_mean"]

                decide_nd[noise_type][int(target_snr)] = {
                    "noise_degree": best_noise_degree,
                    "achieved_snr": achieved_snr,
                    "snr_diff": abs(achieved_snr - target_snr),
                }

                logger.info(
                    f"{noise_type}: Target SNR {target_snr} -> Noise degree {best_noise_degree:.3f} (achieved SNR: {achieved_snr:.2f})"
                )

            except Exception as e:
                logger.error(f"Error finding noise degree for {noise_type} SNR {target_snr}: {e}")
                decide_nd[noise_type][int(target_snr)] = {
                    "noise_degree": np.nan,
                    "achieved_snr": np.nan,
                    "snr_diff": np.nan,
                }

    # Save decision mapping
    with open(path_decide_nd_json, "w") as f:
        json.dump(decide_nd, f, indent=2)

    logger.info(f"Reverse engineering complete. Decision mapping saved to: {path_decide_nd_json}")
    logger.info("Summary of noise degree mappings:")
    logger.info(json.dumps(decide_nd, indent=2))
