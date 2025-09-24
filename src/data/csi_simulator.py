import gc

import torch
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.ofdm import ResourceGrid

from src.utils.data_utils import (
    HIST_LEN,
    NUM_BS_ANT_COL,
    NUM_BS_ANT_ROW,
    NUM_GAP_SUBCARRIERS,
    NUM_SUBCARRIERS,
    NUM_UT_ANT,
    PRED_LEN,
    CSI_PERIODICITY,
    CARRIER_FREQUENCY,
    SUBCARRIER_SPACING,
    NUM_OFDM_SYMBOLS,
)


class CSI_Config:
    """
    Configuration class for CSI simulation parameters.
    
    This class defines all the necessary parameters for CSI generation including
    channel model settings, antenna configurations, and OFDM parameters.
    
    Class Attributes:
        csi_periodicity (int): CSI reporting periodicity in slots
        num_slots (int): Total number of time slots for simulation
        carrier_frequency (float): Carrier frequency in Hz
        fft_size (int): FFT size including gap subcarriers
        subcarrier_spacing (float): Subcarrier spacing in Hz
        num_ofdm_symbols (int): Number of OFDM symbols per frame
        num_bs_ant_row (int): Number of BS antenna rows
        num_bs_ant_col (int): Number of BS antenna columns
        num_ut_ant (int): Number of UT antennas
    """
    
    csi_periodicity = CSI_PERIODICITY
    num_slots = csi_periodicity * (HIST_LEN + PRED_LEN)

    carrier_frequency = CARRIER_FREQUENCY
    fft_size = NUM_GAP_SUBCARRIERS + 2 * NUM_SUBCARRIERS
    subcarrier_spacing = SUBCARRIER_SPACING
    num_ofdm_symbols = NUM_OFDM_SYMBOLS

    num_bs_ant_row = NUM_BS_ANT_ROW
    num_bs_ant_col = NUM_BS_ANT_COL

    num_ut_ant = NUM_UT_ANT

    def __init__(self, batch_size, cdl_model, delay_spread, min_speed, **kwargs):
        """
        Initialize CSI configuration with scenario-specific parameters.
        
        Args:
            batch_size (int): Number of CSI samples to generate per batch
            cdl_model (str): CDL channel model ('A', 'B', 'C', 'D', 'E')
            delay_spread (float): RMS delay spread in seconds (30e-9 ~ 400e-9)
            min_speed (float): Minimum speed in m/s (1 ~ 45)
            **kwargs: Additional keyword arguments (unused)
        """
        self.batch_size = batch_size
        self.cdl_model = cdl_model
        self.delay_spread = delay_spread
        self.min_speed = min_speed

    def getConfig(self):
        """
        Get the complete configuration dictionary for CSI simulation.
        
        Returns:
            dict: Dictionary containing all configuration parameters including:
                - batch_size: Number of samples per batch
                - num_slots: Total time slots
                - csi_periodicity: CSI reporting period
                - carrier_frequency: Carrier frequency in Hz
                - fft_size: FFT size
                - subcarrier_spacing: Subcarrier spacing in Hz
                - num_ofdm_symbols: OFDM symbols per slot
                - num_bs_ant_row/col: BS antenna array dimensions
                - num_ut_ant: Number of UT antennas
                - cdl_model: Channel model identifier
                - delay_spread: RMS delay spread in seconds
                - min_speed: Minimum speed in km/h
        """
        return {
            "batch_size": self.batch_size,
            "num_slots": self.num_slots,
            "csi_periodicity": self.csi_periodicity,
            "carrier_frequency": self.carrier_frequency,
            "fft_size": self.fft_size,
            "subcarrier_spacing": self.subcarrier_spacing,
            "num_ofdm_symbols": self.num_ofdm_symbols,
            "num_bs_ant_row": self.num_bs_ant_row,
            "num_bs_ant_col": self.num_bs_ant_col,
            "num_ut_ant": self.num_ut_ant,
            "cdl_model": self.cdl_model,
            "delay_spread": self.delay_spread,
            "min_speed": self.min_speed,
        }


class CSI_Simulator:
    """
    CSI (Channel State Information) simulator using 3GPP TR 38.901 channel models.
    
    This class generates realistic CSI data using Sionna's implementation of 3GPP
    clustered delay line (CDL) channel models. It simulates uplink transmission
    from user terminals (UT) to base stations (BS) with configurable antenna
    arrays and channel conditions.
    
    Attributes:
        config (dict): Configuration dictionary containing all simulation parameters
        rg (ResourceGrid): OFDM resource grid configuration
        ut_array (PanelArray): User terminal antenna array
        bs_array (PanelArray): Base station antenna array
        frequencies (array): Subcarrier frequencies
        cdl (CDL): 3GPP CDL channel model instance
    """
    
    def __init__(self, config):
        """
        Initialize the CSI simulator with the given configuration.
        
        Args:
            config (dict): Configuration dictionary from CSI_Config.getConfig()
                          containing all necessary simulation parameters
        """
        self.config = config

        self.fft_size = config["fft_size"]
        self.subcarrier_spacing = config["subcarrier_spacing"]
        self.num_ut_ant = config["num_ut_ant"]
        self.carrier_frequency = config["carrier_frequency"]
        self.num_bs_ant_row = config["num_bs_ant_row"]
        self.num_bs_ant_col = config["num_bs_ant_col"]
        self.cdl_model = config["cdl_model"]
        self.delay_spread = config["delay_spread"]
        self.min_speed = config["min_speed"]
        self.batch_size = config["batch_size"]
        self.num_slots = config["num_slots"]
        self.num_ofdm_symbols = config["num_ofdm_symbols"]
        self.csi_periodicity = config["csi_periodicity"]

        self.rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=self.num_ut_ant,
            num_streams_per_tx=1,
            cyclic_prefix_length=6,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
        )

        self.ut_array = PanelArray(
            num_rows_per_panel=self.num_ut_ant,
            num_cols_per_panel=self.num_ut_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency,
        )

        self.bs_array = PanelArray(
            num_rows_per_panel=self.num_bs_ant_row,
            num_cols_per_panel=self.num_bs_ant_col,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.carrier_frequency,
        )

        self.frequencies = subcarrier_frequencies(self.fft_size, self.subcarrier_spacing)

        self.cdl = CDL(
            model=self.cdl_model,
            delay_spread=self.delay_spread,
            carrier_frequency=self.carrier_frequency,
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction="uplink",
            min_speed=self.min_speed,
        )

    def __call__(self) -> torch.Tensor:
        """
        Generate a batch of CSI samples using the configured channel model.
        
        This method performs the complete CSI generation pipeline:
        1. Generate channel impulse response (CIR) using CDL model
        2. Convert CIR to OFDM channel response
        3. Sample at CSI reporting intervals
        4. Return as PyTorch tensor
        
        Returns:
            torch.Tensor: Generated CSI data with shape
                         [batch_size, num_antennas, hist_len + pred_len, fft_size]
                         Data type is torch.ComplexFloatTensor representing complex
                         channel coefficients for each subcarrier and antenna.
        
        Note:
            The output contains both historical (first HIST_LEN slots) and
            prediction target (last PRED_LEN slots) time periods.
            Subcarriers are organized as [UL_subcarriers, gap, DL_subcarriers].
        """
        # Generate channel impulse response using CDL model
        cir = self.cdl(
            self.batch_size,
            self.num_slots * self.num_ofdm_symbols,
            1 / self.rg.ofdm_symbol_duration,
        )

        # Convert CIR to OFDM channel response
        h_long = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        
        # Sample at CSI reporting intervals
        sampling_interval = self.num_ofdm_symbols * self.csi_periodicity
        h = h_long[:, :, :, :, :, 0:-1:sampling_interval, :]
        
        # Convert to PyTorch tensor and clean up memory
        h = torch.from_numpy(h.numpy())
        del h_long
        gc.collect()
        return h.squeeze()

    def __str__(self):
        """
        Return a formatted string representation of the CSI simulator configuration.
        
        Returns:
            str: Multi-line string showing all configuration parameters
        """
        info = " ---------------------------------- \n"
        info += "CSI Simulator with Configuration:\n"
        for key, value in self.config.items():
            info += f"- {key}: {value}\n"
        info += " ---------------------------------- \n"
        return info


if __name__ == "__main__":
    # local test for the simulator
    csi_config = CSI_Config(
        batch_size=2,
        cdl_model="A",
        delay_spread=30e-9,
        min_speed=10.0,
    )
    csi_config = csi_config.getConfig()
    csi_simulator = CSI_Simulator(csi_config)
    print(csi_simulator)

    for _ in range(10):
        h = csi_simulator()
        print(h.shape)
        print(torch.min(torch.abs(h)))
        print(torch.max(torch.abs(h)))
        print(torch.mean(torch.abs(h)))
        print(torch.std(torch.abs(h)))
        import pdb

        pdb.set_trace()
