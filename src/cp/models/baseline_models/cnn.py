import math

import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel
from src.utils.data_utils import HIST_LEN, PRED_LEN


class Autoencoder(nn.Module):
    def __init__(self, num_filters: int = 8, *args, **kwargs):
        super().__init__()

        self.postprocess = nn.Conv1d(HIST_LEN, PRED_LEN, 3, 1, 1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        list_n_filters = [2 ** (i + 1) for i in range(num_filters)]
        list_filter_sizes = [3 for _ in range(num_filters)]

        # Building the encoder
        for i in range(len(list_n_filters) - 1):
            self.encoder.append(
                nn.Conv2d(list_n_filters[i], list_n_filters[i + 1], list_filter_sizes[i], stride=1, padding=1)
            )
            # Initialize weights
            nn.init.uniform_(
                self.encoder[-1].weight,
                -1.0 / math.sqrt(list_n_filters[i]),
                1.0 / math.sqrt(list_n_filters[i]),
            )
            nn.init.constant_(self.encoder[-1].bias, 0)

        # Building the decoder
        list_n_filters.reverse()
        list_filter_sizes.reverse()
        for i in range(len(list_n_filters) - 1):
            self.decoder.append(
                nn.Conv2d(list_n_filters[i], list_n_filters[i + 1], list_filter_sizes[i], stride=1, padding=1)
            )
            # Initialize weights
            nn.init.uniform_(
                self.decoder[-1].weight,
                -1.0 / math.sqrt(list_n_filters[i]),
                1.0 / math.sqrt(list_n_filters[i]),
            )
            nn.init.constant_(self.decoder[-1].bias, 0)

    def forward(self, x):
        # x.shape is [batch_size = 512, hist_len = 16, num_subcarriers = 600 (300 * 2)] 2 means complex
        x = rearrange(x, "b l (s i) -> b i l s", i=2)  # [512, 2, 16, 300]
        # Encoder
        for layer in self.encoder:
            x = F.tanh(layer(x))

        # Decoder
        for layer in self.decoder:
            x = F.tanh(layer(x))
        # Postprocessor
        x = rearrange(x, "b i l s -> b l (s i)", i=2)  # back to [512, 16, 600] after this rearrange
        x = self.postprocess(x)

        return x


class CNN_pl(BaseCSIModel):
    """CNN model for CSI prediction."""

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )

        self.name = "CNN"
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})

        self.model = Autoencoder(**config.model.params)

    def __str__(self):
        return self.name

    def forward(self, x):
        """Forward pass of the CNN model."""
        # x.shape is [batch_size, hist_len, num_antennas*2]
        # where num_antennas*2 is the real and imaginary parts
        x = self.model(x)
        return x
