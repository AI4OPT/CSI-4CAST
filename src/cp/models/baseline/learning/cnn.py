"""CNN baseline models for CSI prediction.

This module implements a compact convolutional autoencoder used as a
learning-based baseline. It includes the core predictor and the
Lightning wrapper used by the training and evaluation pipelines.
"""

import math

from einops import rearrange
import torch.nn as nn
from torch.nn import functional as F

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel
from src.utils.data_utils import HIST_LEN, PRED_LEN


class Autoencoder(nn.Module):
    """Convolutional autoencoder used by the CNN baseline.

    The model converts the real-valued CSI history tensor into a
    two-channel image-like representation, applies a symmetric encoder
    and decoder stack, and projects the output to the prediction
    horizon.
    """

    def __init__(self, num_filters: int = 8, *args, **kwargs):
        """Initialize the convolutional encoder-decoder stack.

        Args:
            num_filters: Number of convolution stages used in the
                encoder and decoder.
            *args: Passed to the parent class.
            **kwargs: Passed to the parent class.

        """
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
        """Encode and decode a CSI history window.

        Args:
            x: Input CSI history with shape
                ``[batch, hist_len, num_subcarriers * 2]``.

        Returns:
            Predicted CSI sequence over the configured forecast horizon.

        """
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


class CNN_cp(BaseCSIModel):
    """Lightning wrapper for the CNN CSI predictor.

    This wrapper connects the convolutional baseline to the shared
    training loop by attaching optimizer, scheduler, and loss
    configuration from the experiment config.
    """

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        """Build the Lightning wrapper from an experiment config.

        Args:
            config: Experiment configuration containing model and
                training-related settings.
            *args: Passed to the parent class.
            **kwargs: Passed to the parent class.

        """
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
        """Return the model name used in logs and registries."""
        return self.name

    def forward(self, x):
        """Run the wrapped CNN predictor.

        Args:
            x: Input CSI tensor prepared by the data module.

        Returns:
            Model prediction for the next CSI steps.

        """
        # x.shape is [batch_size, hist_len, num_antennas*2]
        # where num_antennas*2 is the real and imaginary parts
        x = self.model(x)
        return x
