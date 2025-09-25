import torch.nn as nn

from src.utils.data_utils import PRED_LEN


class NOPREDICTMODEL(nn.Module):
    def __init__(self, pred_len=PRED_LEN, *args, **kwargs):
        super().__init__()

        self.is_separate_antennas = True
        self.name = "NP"

        self.pred_len = pred_len  # Number of time steps to repeat

    def __str__(self) -> str:
        return self.name

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            Repeated last step of x for `pred_len` timesteps.
        """
        last_step = x[:, [-1], :]  # Select the last time step (shape: batch_size, 1, feature_dim)
        return last_step.repeat(1, self.pred_len, 1)  # Repeat along the sequence length
