"""CSI Simulator Module for 3GPP TR 38.901 Channel Models.

This module provides a comprehensive CSI (Channel State Information) simulation
framework based on 3GPP TR 38.901 clustered delay line (CDL) channel models.
It generates realistic channel responses for uplink transmission scenarios
between user terminals (UT) and base stations (BS).

Key Features:
    - 3GPP TR 38.901 compliant CDL channel models (A, B, C, D, E)
    - Configurable antenna arrays (single/dual polarization)
    - OFDM-based channel response generation
    - Batch processing for efficient dataset generation
    - Memory-optimized implementation with garbage collection

Dependencies:
    - Sionna: NVIDIA's open-source library for link-level simulations
    - PyTorch: For tensor operations and data handling
    
References:
    - 3GPP TR 38.901: "Study on channel model for frequencies from 0.5 to 100 GHz"
    - Sionna Documentation: https://nvlabs.github.io/sionna/
"""

import gc

import torch

# === Sionna Channel Modeling Components ===
# Sionna is NVIDIA's open-source library for link-level wireless communications simulation

from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
# - cir_to_ofdm_channel: Convert CIR to per-subcarrier frequency response
# - subcarrier_frequencies: Generate OFDM subcarrier frequency grid

from sionna.phy.channel.tr38901 import CDL, PanelArray
# - CDL: 3GPP TR 38.901 CDL channel models (A–E) with delay, Doppler, and spatial effects
# - PanelArray: Define BS/UT antenna array geometry and patterns for MIMO

from sionna.phy.ofdm import ResourceGrid
# - ResourceGrid: Specify OFDM lattice (symbols, subcarriers, CP) and pilot placement


# Import simulation constants and parameters
from src.utils.data_utils import (
    HIST_LEN,              # Number of historical time slots for model input
    NUM_BS_ANT_COL,        # Base station antenna array columns
    NUM_BS_ANT_ROW,        # Base station antenna array rows  
    NUM_GAP_SUBCARRIERS,   # Gap subcarriers between UL and DL bands
    NUM_SUBCARRIERS,       # Number of subcarriers per direction (UL/DL)
    NUM_UT_ANT,            # Number of user terminal antennas
    PRED_LEN,              # Number of prediction time slots for model output
    CSI_PERIODICITY,       # CSI reporting periodicity in OFDM symbols
    CARRIER_FREQUENCY,     # Carrier frequency in Hz
    SUBCARRIER_SPACING,    # Subcarrier spacing in Hz
    NUM_OFDM_SYMBOLS,      # Number of OFDM symbols per slot
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
        
    References:
        - https://nvlabs.github.io/sionna/phy/tutorials/OFDM_MIMO_Detection.html for the following configuration
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

        # Configure OFDM resource grid for channel estimation
        # Uses standard 5G NR parameters with Kronecker pilot pattern
        self.rg = ResourceGrid(
            num_ofdm_symbols=14,                    # Standard slot length (14 OFDM symbols)
            fft_size=self.fft_size,                 # Total subcarriers (UL + gap + DL)
            subcarrier_spacing=self.subcarrier_spacing,  # 15 kHz for sub-6 GHz
            num_tx=self.num_ut_ant,                 # Number of transmitting antennas (UT)
            num_streams_per_tx=1,                   # Single stream per antenna
            cyclic_prefix_length=6,                 # Normal cyclic prefix length
            pilot_pattern="kronecker",              # Kronecker pilot pattern for MIMO
            pilot_ofdm_symbol_indices=[2, 11],      # Pilot symbol positions in slot
        )

        # Configure user terminal (UT) antenna array
        # Single polarization omni-directional antennas for mobile devices
        self.ut_array = PanelArray(
            num_rows_per_panel=self.num_ut_ant,     # Square antenna array
            num_cols_per_panel=self.num_ut_ant,
            polarization="single",                  # Single polarization (vertical)
            polarization_type="V",                  # Vertical polarization
            antenna_pattern="omni",                 # Omni-directional pattern
            carrier_frequency=self.carrier_frequency,
        )

        # Configure base station (BS) antenna array  
        # Dual polarization with 3GPP 38.901 antenna pattern
        self.bs_array = PanelArray(
            num_rows_per_panel=self.num_bs_ant_row, # 4x4 antenna array (configurable)
            num_cols_per_panel=self.num_bs_ant_col,
            polarization="dual",                    # Dual polarization (+/-45°)
            polarization_type="cross",              # Cross-polarized antennas
            antenna_pattern="38.901",               # 3GPP standardized pattern
            carrier_frequency=self.carrier_frequency,
        )

        # Generate subcarrier frequencies for OFDM channel conversion
        self.frequencies = subcarrier_frequencies(self.fft_size, self.subcarrier_spacing)

        # Initialize 3GPP CDL channel model
        # Supports models A-E with different delay spread and mobility characteristics
        self.cdl = CDL(
            model=self.cdl_model,                   # CDL model type ('A', 'B', 'C', 'D', 'E')
            delay_spread=self.delay_spread,         # RMS delay spread in seconds
            carrier_frequency=self.carrier_frequency,
            ut_array=self.ut_array,                 # UT antenna configuration
            bs_array=self.bs_array,                 # BS antenna configuration
            direction="uplink",                     # Uplink transmission (UT → BS)
            min_speed=self.min_speed,               # Minimum speed for Doppler effects
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
        # Step 1: Generate channel impulse response (CIR) using 3GPP CDL model
        # The CDL model generates time-varying channel impulse responses based on
        # the specified propagation environment and mobility conditions
        cir = self.cdl(
            self.batch_size,                        # Number of independent channel realizations
            self.num_slots * self.num_ofdm_symbols, # Total number of time samples
            1 / self.rg.ofdm_symbol_duration,       # Sampling rate (samples per second)
        )

        # Step 2: Convert CIR to OFDM channel frequency response
        # Transform time-domain impulse response to frequency-domain coefficients
        # for each subcarrier using FFT-based conversion with normalization
        h_long = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        
        # Step 3: Sample at CSI reporting intervals
        # Extract channel samples at periodic intervals matching CSI reporting rate
        # This simulates realistic CSI acquisition where full channel knowledge
        # is only available at specific time instances
        sampling_interval = self.num_ofdm_symbols * self.csi_periodicity
        h = h_long[:, :, :, :, :, 0:-1:sampling_interval, :]
        
        # Step 4: Convert to PyTorch tensor and optimize memory usage
        # Convert from TensorFlow/NumPy format to PyTorch tensor
        # Clean up intermediate variables to prevent memory leaks
        h = torch.from_numpy(h.numpy())
        del h_long  # Free memory from large intermediate tensor
        gc.collect()  # Force garbage collection
        
        # Remove singleton dimensions and return final CSI tensor
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
