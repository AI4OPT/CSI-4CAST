import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


# constants
CSI_PERIODICITY = 5  # report CSI every 5 slots
NUM_OFDM_SYMBOLS = 14  # 14 OFDM symbols in one frame
CARRIER_FREQUENCY = 2.4e9  # 2.4 GHz
SUBCARRIER_SPACING = 30e3  # 30 kHz

# Batch size for the CSI generator - determines how many CSI samples
# (each sample is a length 20 CSI sequence) can be obtained in one generation.
# This value should be adjusted based on the system's CPU memory capacity.
# a sample is 20 slots * 32 antennas * 750 subcarriers
BATCH_SIZE = 10
BATCH_SIZE_DEBUG = 2  # 2 for debug

# Number of training samples to generate for each scenario
NUM_REPEAT_TRAIN = 1000
NUM_REPEAT_TRAIN_DEBUG = 2  # 2 for debug

# Number of testing samples to generate for each scenario
NUM_REPEAT_TEST = 100
NUM_REPEAT_TEST_DEBUG = 2  # 2 for debug

HIST_LEN = 16  # 16 slots historical csi for model input
PRED_LEN = 4  # 4 slots prediction for model output
TOT_LEN = HIST_LEN + PRED_LEN  # total length of time slots

NUM_SUBCARRIERS = 300  # 300 subcarriers for DL and UL
NUM_GAP_SUBCARRIERS = 150  # 150 subcarriers for gap
TOT_SUBCARRIERS = NUM_SUBCARRIERS * 2 + NUM_GAP_SUBCARRIERS  # total number of subcarriers in one slot

NUM_BS_ANT_ROW = 4  # number of BS antennas in row
NUM_BS_ANT_COL = 4  # number of BS antennas in column
NUM_DUPLEX = 2
TOT_ANTENNAS = NUM_BS_ANT_ROW * NUM_BS_ANT_COL * NUM_DUPLEX  # total number of BS antennas

NUM_UT_ANT = 1  # number of UT antennas


# regular dataset
# dataset for training, regular and robustness testing
LIST_CHANNEL_MODEL = ["A", "C", "D"]
LIST_DELAY_SPREAD = [30e-9, 100e-9, 300e-9]
LIST_MIN_SPEED_TRAIN = [1, 10, 30]
LIST_MIN_SPEED_TEST = [1, 10, 30]


# dataset for generalization testing
LIST_CHANNEL_MODEL_GEN = ["A", "B", "C", "D", "E"]
LIST_DELAY_SPREAD_GEN = [30e-9, 50e-9, 100e-9, 200e-9, 300e-9, 400e-9]
LIST_MIN_SPEED_TEST_GEN = sorted([*range(3, 46, 3), 1, 10])


def make_folder_name(cm: str, ds: float, ms: int, **kwargs) -> str:
    """Generate a standardized folder name based on channel model, delay spread, and minimum speed.

    Args:
        cm (str): Channel model identifier (e.g., 'A', 'B', 'C', 'D', 'E')
        ds (float): Delay spread in seconds (e.g., 30e-9, 100e-9, 300e-9)
        ms (int): Minimum speed in km/h (e.g., 1, 10, 30)
        **kwargs: Additional keyword arguments (unused)

    Returns:
        str: Formatted folder name in the format 'cm_{cm}_ds_{ds}_ms_{ms}'
             where ds is converted to nanoseconds and zero-padded to 3 digits,
             and ms is zero-padded to 3 digits

    Example:
        >>> make_folder_name('A', 30e-9, 10)
        'cm_A_ds_030_ms_010'

    """
    # the precision of the delay spread is int
    ds = round(ds * 1e9)
    ds_str = str(ds).zfill(3)

    # the precision of the min speed is .1
    ms_str = str(ms)
    ms_str = ms_str.zfill(3)

    # the file name
    return f"cm_{cm}_ds_{ds_str}_ms_{ms_str}"


def _load_data(
    dir_data: Path,
    cm: str,
    ds: float,
    ms: int,
    is_train: bool = True,
    is_gen: bool = False,
    is_hist: bool = True,
    is_U2D: bool = False,
    num_load: int = -1,
) -> torch.Tensor:
    """Load CSI dataset for a specific channel model, delay spread, and minimum speed configuration.

    Args:
        dir_data (Path): Root directory containing the dataset
        cm (str): Channel model identifier ('A', 'B', 'C', 'D', 'E')
        ds (float): Delay spread in seconds (30e-9 ~ 400e-9)
        ms (int): Minimum speed in m/s (1 ~ 45)
        is_train (bool, optional): If True, load training data. Defaults to True.
        is_gen (bool, optional): If True, load generalization test data. Defaults to False.
        is_hist (bool, optional): If True, load historical CSI data. If False, load prediction data. Defaults to True.
        is_U2D (bool, optional): If True, load prediction data for FDD scenario. Defaults to False (TDD scenario).
        num_load (int, optional): Number of samples to load. If -1, load all samples. Defaults to -1.

    Returns:
        torch.Tensor: Loaded CSI data tensor

    Raises:
        AssertionError: If both is_train and is_gen are True (generalization is only for test mode)

    Note:
        File naming convention:
        - Historical data: 'H_U_hist.pt'
        - TDD prediction: 'H_U_pred.pt'
        - FDD prediction: 'H_D_pred.pt'

    """
    assert not (is_train and is_gen), "generalization is only for test mode"

    file_name = "H_U_hist.pt" if is_hist else "H_D_pred.pt" if is_U2D else "H_U_pred.pt"
    folder_name = make_folder_name(cm, ds, ms)

    if is_train:
        dir_folder = dir_data / "train" / "regular" / folder_name
        path_file = dir_folder / file_name
    else:
        if is_gen:
            dir_folder = dir_data / "test" / "generalization" / folder_name
        else:
            dir_folder = dir_data / "test" / "regular" / folder_name

        path_file = dir_folder / file_name

    data = torch.load(path_file, weights_only=True)
    if num_load > 0:
        data = data[:num_load]
    return data


def load_data(
    dir_data: Path,
    list_cm: list[str],
    list_ds: list[float],
    list_ms: list[int],
    is_train: bool = True,
    is_gen: bool = False,
    is_hist: bool = True,
    is_U2D: bool = False,
    num_load: int = -1,
) -> torch.Tensor:
    """Load and concatenate CSI datasets across multiple channel configurations.

    Args:
        dir_data (Path): Root directory containing the dataset
        list_cm (list[str]): List of channel model identifiers
        list_ds (list[float]): List of delay spreads in seconds
        list_ms (list[int]): List of minimum speeds in m/s
        is_train (bool, optional): If True, load training data. Defaults to True.
        is_gen (bool, optional): If True, load generalization test data. Defaults to False.
        is_hist (bool, optional): If True, load historical CSI data. If False, load prediction data. Defaults to True.
        is_U2D (bool, optional): If True, load downlink prediction data when is_hist=False. Defaults to False.
        num_load (int, optional): Number of samples to load per configuration. If -1, load all samples. Defaults to -1.

    Returns:
        torch.Tensor: Concatenated CSI data tensor from all specified configurations

    Raises:
        AssertionError: If both is_train and is_gen are True (generalization is only for test mode)

    Note:
        This function iterates through all combinations of channel models, delay spreads,
        and minimum speeds, loading data for each configuration and concatenating them
        along the batch dimension.

    """
    assert not (is_train and is_gen), "generalization is only for test mode"

    data = []
    for cm in list_cm:
        for ds in list_ds:
            for ms in list_ms:
                data.append(
                    _load_data(
                        dir_data=dir_data,
                        cm=cm,
                        ds=ds,
                        ms=ms,
                        is_train=is_train,
                        is_gen=is_gen,
                        is_hist=is_hist,
                        is_U2D=is_U2D,
                        num_load=num_load,
                    )
                )

    # concatenate the data
    data = torch.cat(data, dim=0)
    return data


class CSIDataset(Dataset):
    """PyTorch Dataset class for CSI prediction tasks.

    This dataset handles pairs of historical and prediction CSI data for training
    and evaluation of CSI prediction models.

    Attributes:
        H_hist (torch.Tensor): Historical CSI data with shape
                              [batch_size, num_antennas, hist_len, num_subcarriers]
        H_pred (torch.Tensor): Prediction target CSI data with shape
                              [batch_size, num_antennas, pred_len, num_subcarriers]

    """

    def __init__(self, H_hist, H_pred):
        """Initialize the CSI dataset.

        Args:
            H_hist (torch.Tensor): Historical CSI data tensor
            H_pred (torch.Tensor): Prediction target CSI data tensor

        """
        super().__init__()
        self.H_hist = H_hist  # [batch_size, num_antennas, hist_len, num_subcarriers]
        self.H_pred = H_pred  # [batch_size, num_antennas, pred_len, num_subcarriers]

    def __len__(self):
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset

        """
        return len(self.H_hist)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (historical_csi, prediction_target_csi) pair

        """
        hist = self.H_hist[idx]
        pred = self.H_pred[idx]
        return hist, pred


def collect_fn_separate_antennas(batch):
    """Collate function for DataLoader that processes CSI data for separate antenna training.

    This function transforms batched CSI data from complex-valued antenna-grouped format
    to real-valued flattened format suitable for training models that process each
    antenna separately.

    Args:
        batch (list): List of (hist, pred) tuples from CSIDataset
        the original shape is [batch_size, num_antennas, hist_len, num_subcarriers] for hist
        and [batch_size, num_antennas, pred_len, num_subcarriers] for pred

    Returns:
        tuple: (hist_real, pred_real) where:
            - hist_real: Historical CSI data with shape
                        [batch_size * num_antennas, hist_len, num_subcarriers * 2]
            - pred_real: Prediction CSI data with shape
                        [batch_size * num_antennas, pred_len, num_subcarriers * 2]

    Transformation process:
        1. Stack batch samples: [batch_size, num_antennas, time_len, num_subcarriers] complex
        2. Flatten antennas: [batch_size * num_antennas, time_len, num_subcarriers] complex
        3. Convert to real: [batch_size * num_antennas, time_len, num_subcarriers * 2] real

    """
    list_hist, list_pred = zip(*batch, strict=False)
    hist = torch.stack(list_hist, dim=0)  # [batch_size, num_antennas, hist_len, num_subcarriers]
    pred = torch.stack(list_pred, dim=0)  # [batch_size, num_antennas, pred_len, num_subcarriers]

    hist = hist.view(-1, hist.shape[2], hist.shape[3])  # [batch_size * num_antennas, hist_len, num_subcarriers]
    pred = pred.view(-1, pred.shape[2], pred.shape[3])  # [batch_size * num_antennas, pred_len, num_subcarriers]

    # from [batch_size * num_antennas, hist_len, num_subcarriers] complex
    # -> [batch_size * num_antennas, hist_len, num_subcarriers*2] real
    hist = torch.view_as_real(hist)  # [batch_size * num_antennas, hist_len, num_subcarriers, 2] real
    pred = torch.view_as_real(pred)  # [batch_size * num_antennas, pred_len, num_subcarriers, 2] real
    hist = hist.view(hist.shape[0], hist.shape[1], -1)  # [batch_size * num_antennas, hist_len, num_subcarriers*2] real
    pred = pred.view(pred.shape[0], pred.shape[1], -1)  # [batch_size * num_antennas, pred_len, num_subcarriers*2] real

    return hist, pred  # return [batch_size * num_antennas, hist_len, num_subcarriers*2] real


def collect_fn_gather_antennas(batch):
    # from [batch_size, num_antennas, hist_len, num_subcarriers] complex
    list_hist, list_pred = zip(*batch, strict=False)
    hist = torch.stack(list_hist, dim=0)  # [batch_size, num_antennas, hist_len, num_subcarriers]
    pred = torch.stack(list_pred, dim=0)  # [batch_size, num_antennas, pred_len, num_subcarriers]

    return hist, pred  # return [batch_size, num_antennas, hist_len/pred_len, num_subcarriers] complex


# AWGN noise for training
SNR_RANGE_GAUSSIAN_NOISE_TRAIN = (0, 25)
