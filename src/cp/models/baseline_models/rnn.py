"""RNN-based CSI Prediction Model.

This module implements a Recurrent Neural Network (RNN) based approach for Channel State Information
(CSI) prediction. The model uses a sequence-to-sequence architecture with encoder-decoder structure
to predict future CSI values from historical observations.

Architecture:
- Encoder: Linear layer to map input features to RNN input size
- RNN Core: Multi-layer RNN for temporal modeling
- Decoder: Linear layer to map RNN output back to CSI feature space

The model supports:
- Multi-step ahead prediction
- Configurable hidden dimensions and number of layers
- Both training and inference modes
- PyTorch Lightning integration for easy training

Usage:
    from src.cp.models.baseline_models.rnn import RNN_pl

    model = RNN_pl(config)
    predictions = model(historical_csi)
"""

import torch
import torch.nn as nn

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel


class RNNUnit(nn.Module):
    """Basic RNN Unit with Encoder-Decoder Architecture.

    This unit implements a simple RNN cell with linear encoder and decoder layers.
    The encoder maps input features to RNN input space, the RNN processes temporal
    sequences, and the decoder maps back to output feature space.

    Args:
        features (int): Number of input/output features (CSI dimensions)
        input_size (int): Size of RNN input (after encoding)
        hidden_size (int): Size of RNN hidden state
        num_layers (int): Number of RNN layers. Default: 2

    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        """Initialize RNN unit with encoder-decoder structure.

        Args:
            features (int): Dimension of CSI features
            input_size (int): RNN input dimension
            hidden_size (int): RNN hidden state dimension
            num_layers (int): Number of stacked RNN layers

        """
        super().__init__()

        # Encoder: map CSI features to RNN input space
        self.encoder = nn.Sequential(nn.Linear(features, input_size))

        # RNN core: process temporal sequences
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)

        # Decoder: map RNN output back to CSI feature space
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))

    def forward(self, x, prev_hidden):
        """Forward pass through RNN unit.

        Args:
            x (torch.Tensor): Input tensor [seq_len, batch_size, features]
            prev_hidden (torch.Tensor): Previous hidden state [num_layers, batch_size, hidden_size]

        Returns:
            tuple: (output, current_hidden)
                - output: RNN output [seq_len, batch_size, features]
                - current_hidden: Updated hidden state [num_layers, batch_size, hidden_size]

        """
        L, B, _ = x.shape  # seq_len, batch_size, features

        # Flatten for linear layer processing
        output = x.reshape(L * B, -1)

        # Encode input features
        output = self.encoder(output)

        # Reshape back to sequence format for RNN
        output = output.reshape(L, B, -1)

        # Process through RNN
        output, cur_hidden = self.rnn(output, prev_hidden)

        # Flatten for decoder processing
        output = output.reshape(L * B, -1)

        # Decode back to CSI feature space
        output = self.decoder(output)

        # Reshape to final output format
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class RNN(nn.Module):
    """RNN-based CSI Prediction Model.

    This model implements a sequence-to-sequence RNN for CSI prediction using an autoregressive
    approach. It processes historical CSI observations and generates future predictions step by step.

    The model uses teacher forcing during training (feeding ground truth) and autoregressive
    generation during inference (feeding previous predictions).

    Args:
        dim_data (int): Dimension of CSI data features
        rnn_hidden_dim (int): Hidden dimension of RNN
        rnn_num_layers (int): Number of RNN layers. Default: 2
        pred_len (int): Number of prediction steps. Default: 4

    """

    def __init__(self, dim_data, rnn_hidden_dim, rnn_num_layers=2, pred_len=4, *args, **kwargs):
        """Initialize RNN prediction model.

        Args:
            dim_data (int): Input/output feature dimension
            rnn_hidden_dim (int): RNN hidden state dimension
            rnn_num_layers (int): Number of stacked RNN layers
            pred_len (int): Number of future time steps to predict

        """
        super().__init__()

        self.dim_data = dim_data
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.pred_len = pred_len

        # Initialize RNN unit with encoder-decoder architecture
        self.model = RNNUnit(dim_data, dim_data, rnn_hidden_dim, num_layers=rnn_num_layers)

    def train_pro(self, x):
        """Training procedure with autoregressive prediction.

        This method processes the input sequence and generates predictions autoregressively.
        During the historical phase, it uses ground truth inputs. During the prediction phase,
        it feeds its own outputs as inputs for the next step.

        Args:
            x (torch.Tensor): Input CSI sequence [batch_size, seq_len, features]

        Returns:
            torch.Tensor: Predicted CSI sequence [batch_size, pred_len, features]

        """
        BATCH_SIZE, seq_len, _ = x.shape

        # Initialize hidden state
        prev_hidden = torch.zeros(self.rnn_num_layers, BATCH_SIZE, self.rnn_hidden_dim).to(x.device)

        outputs = []

        # Process sequence: historical inputs + autoregressive predictions
        for idx in range(seq_len + self.pred_len - 1):
            if idx < seq_len:
                # Use ground truth input during historical phase
                # Reshape from [batch, seq, feat] to [seq, batch, feat] for RNN
                input_step = x[:, idx : idx + 1, ...].permute(1, 0, 2).contiguous()
                output, prev_hidden = self.model(input_step, prev_hidden)
            else:
                # Use previous output as input during prediction phase (autoregressive)
                output, prev_hidden = self.model(output, prev_hidden)

            # Collect outputs starting from the last historical step
            if idx >= seq_len - 1:
                outputs.append(output)

        # Concatenate outputs and reshape back to [batch, pred_len, features]
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x):
        """Forward pass of the RNN model.

        Args:
            x (torch.Tensor): Input CSI sequence [batch_size, seq_len, features]

        Returns:
            torch.Tensor: Predicted CSI sequence [batch_size, pred_len, features]

        """
        return self.train_pro(x)


class RNN_pl(BaseCSIModel):
    """PyTorch Lightning Wrapper for RNN-based CSI Prediction.

    This class wraps the RNN model with PyTorch Lightning functionality, providing
    automatic training, validation, and testing procedures. It inherits from BaseCSIModel
    which handles common functionality like loss computation, optimization, and logging.

    Features:
    - Automatic training and validation loops
    - Loss computation and logging
    - Optimizer and scheduler configuration
    - Model checkpointing and resumption
    - TensorBoard logging integration

    Args:
        config (ExperimentConfig): Complete experiment configuration containing
            model parameters, optimizer settings, scheduler config, and loss function

    """

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        """Initialize RNN Lightning module.

        Args:
            config (ExperimentConfig): Experiment configuration object containing:
                - model: Model architecture parameters
                - optimizer: Optimizer configuration
                - scheduler: Learning rate scheduler configuration
                - loss: Loss function configuration

        """
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )

        # Model identification and configuration
        self.name = "RNN"
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})

        # Create the RNN model with parameters from config
        self.model = RNN(**config.model.params)

    def __str__(self):
        """Return string representation of the model."""
        return self.name

    def forward(self, x):
        """Forward pass through the RNN model.

        Args:
            x (torch.Tensor): Historical CSI data [batch_size, seq_len, features]

        Returns:
            torch.Tensor: Predicted CSI data [batch_size, pred_len, features]

        """
        return self.model(x)
