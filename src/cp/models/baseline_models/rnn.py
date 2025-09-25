import torch
import torch.nn as nn

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel


class RNNUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
        # print('x shape must be 3')

        L, B, _ = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.rnn(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class RNN(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, dim_data, rnn_hidden_dim, rnn_num_layers=2, pred_len=4, *args, **kwargs):
        super().__init__()

        self.dim_data = dim_data
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.pred_len = pred_len

        self.model = RNNUnit(dim_data, dim_data, rnn_hidden_dim, num_layers=rnn_num_layers)

    def train_pro(self, x):
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.rnn_num_layers, BATCH_SIZE, self.rnn_hidden_dim).to(x.device)
        outputs = []
        for idx in range(seq_len + self.pred_len - 1):
            if idx < seq_len:
                output, prev_hidden = self.model(x[:, idx : idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden)
            else:
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x):
        return self.train_pro(x)


class RNN_pl(BaseCSIModel):
    """
    PyTorch Lightning wrapper for RNN model
    """

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )

        self.name = "RNN"
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})

        # Create the RNN model with parameters from config
        self.model = RNN(**config.model.params)

    def __str__(self):
        return self.name

    def forward(self, x):
        """
        Forward pass using the RNN model
        """
        return self.model(x)
