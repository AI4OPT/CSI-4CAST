"""RNN-based CSI Prediction Model.

Sequence-to-sequence RNN with encoder-decoder architecture for CSI prediction.

Attribution:
    Modified from: [LLM4CP](https://github.com/liuboxun/LLM4CP/blob/master/models/model.py)

Usage:
    model = RNN_pl(config)
    predictions = model(historical_csi)
"""

import torch
import torch.nn as nn

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel


class RNNUnit(nn.Module):
    """RNN cell with linear encoder/decoder layers.

    Args:
        features (int): Input/output feature dimension
        input_size (int): RNN input dimension after encoding
        hidden_size (int): RNN hidden state dimension
        num_layers (int): Number of stacked RNN layers

    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        """Initialize the RNN unit."""
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))

    def forward(self, x, prev_hidden):
        """Forward pass through encoder-RNN-decoder.

        Args:
            x (torch.Tensor): Input [seq_len, batch_size, features]
            prev_hidden (torch.Tensor): Previous hidden state [num_layers, batch_size, hidden_size]

        Returns:
            tuple: (output, hidden) where output is [seq_len, batch_size, features]

        """
        L, B, _ = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.rnn(output, prev_hidden)
        output = output.reshape(L * B, -1)
        output = self.decoder(output)
        output = output.reshape(L, B, -1)
        return output, cur_hidden


class RNN(nn.Module):
    """Sequence-to-sequence RNN with autoregressive prediction.

    Args:
        dim_data (int): CSI feature dimension (e.g., num_subcarriers * 2 for complex data)
        rnn_hidden_dim (int): Hidden state dimension for RNN layers
        rnn_num_layers (int): Number of stacked RNN layers
        pred_len (int): Number of future time steps to predict

    """

    def __init__(self, dim_data, rnn_hidden_dim, rnn_num_layers=2, pred_len=4, *args, **kwargs):
        """Initialize the RNN model."""
        super().__init__()
        self.dim_data = dim_data
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.pred_len = pred_len
        self.model = RNNUnit(dim_data, dim_data, rnn_hidden_dim, num_layers=rnn_num_layers)

    def train_pro(self, x):
        """Autoregressive prediction with teacher forcing during historical phase.

        Process:
            1. Use ground truth inputs for historical time steps (teacher forcing)
            2. Use model's own outputs for prediction time steps (autoregressive)

        Args:
            x (torch.Tensor): Historical CSI data [batch_size, seq_len, features]

        Returns:
            torch.Tensor: Predicted CSI data [batch_size, pred_len, features]

        """
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.rnn_num_layers, BATCH_SIZE, self.rnn_hidden_dim).to(x.device)
        outputs = []

        for idx in range(seq_len + self.pred_len - 1):
            if idx < seq_len:
                # Use ground truth input during historical phase
                input_step = x[:, idx : idx + 1, ...].permute(1, 0, 2).contiguous()
                output, prev_hidden = self.model(input_step, prev_hidden)
            else:
                # Use previous output as input during prediction phase
                output, prev_hidden = self.model(output, prev_hidden)

            if idx >= seq_len - 1:
                outputs.append(output)

        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()
        return outputs

    def forward(self, x):
        """Forward pass through the RNN model.

        Args:
            x (torch.Tensor): Historical CSI data [batch_size, seq_len, features]

        Returns:
            torch.Tensor: Predicted CSI data [batch_size, pred_len, features]

        """
        return self.train_pro(x)


class RNN_pl(BaseCSIModel):
    """PyTorch Lightning wrapper for RNN model.

    Inherits training/validation loops, optimization, and logging from BaseCSIModel.
    Antenna processing mode controlled by config.model.is_separate_antennas.
    See src.utils.data_utils for details on separate vs gather antenna processing.

    Args:
        config (ExperimentConfig): Complete experiment configuration containing
            model params, optimizer, scheduler, and loss configurations

    """

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        """Initialize the RNN lightning module."""
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )
        self.name = "RNN"
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})
        self.model = RNN(**config.model.params)

    def __str__(self):
        """Return model name for identification."""
        return self.name

    def forward(self, x):
        """Forward pass through the wrapped RNN model.

        Args:
            x (torch.Tensor): Historical CSI data [batch_size, seq_len, features]

        Returns:
            torch.Tensor: Predicted CSI data [batch_size, pred_len, features]

        """
        return self.model(x)
