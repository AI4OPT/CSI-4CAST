"""Embedding layers shared by transformer-style CSI models.

This module provides reusable embedding blocks for sequence models that
consume CSI data. It includes positional, token, categorical time, and
continuous time-feature embeddings, plus a composite wrapper that
combines them.
"""

import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Standard sinusoidal positional embedding.

    This layer builds a fixed sinusoidal table and slices it to the
    sequence length requested at runtime.
    """

    def __init__(self, d_model, max_len=5000):
        """Precompute sinusoidal encodings up to ``max_len``.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum supported sequence length.

        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  # 5000,512
        pe.requires_grad_(False)

        position = torch.arange(0, max_len).float().unsqueeze(1)  # 1,5000
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # 512
        # position * div_term ： 5000 * 512
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 1,5000,512
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Return positional embeddings matching the input length."""
        # x : 64,t_in,16
        return self.pe[:, : x.size(1)]  # 1，t_in


class TokenEmbedding(nn.Module):
    """1D convolutional token embedding.

    The layer maps per-step feature vectors into the model dimension
    using a circular-padded convolution.
    """

    def __init__(self, c_in, d_model):
        """Initialize the token embedding layer.

        Args:
            c_in: Input feature dimension.
            d_model: Output embedding dimension.

        """
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode="circular"
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        """Embed tokens along the feature axis."""
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # 64,t_in,64
        return x


class FixedEmbedding(nn.Module):
    """Non-trainable sinusoidal categorical embedding.

    This layer creates a frozen embedding table for categorical values,
    using the same sinusoidal construction as transformer positional
    encodings.
    """

    def __init__(self, c_in, d_model):
        """Initialize the fixed categorical embedding.

        Args:
            c_in: Number of categorical values.
            d_model: Embedding dimension.

        """
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad_(False)

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """Embed categorical inputs with frozen weights."""
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Calendar-style embedding for temporal features.

    The layer embeds decomposed time fields such as hour, weekday, day,
    and month, and optionally minute information for finer-grained
    frequencies.
    """

    def __init__(self, d_model, embed_type="fixed", freq="h"):
        """Initialize temporal feature embeddings.

        Args:
            d_model: Embedding dimension.
            embed_type: Embedding strategy for categorical fields.
            freq: Frequency code controlling which fields are present.

        """
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """Embed decomposed temporal fields."""
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Linear embedding for continuous time features.

    This variant is used when time features are already represented as
    continuous numeric inputs instead of categorical indices.
    """

    def __init__(self, d_model, embed_type="timeF", freq="h"):
        """Initialize the linear time-feature projection.

        Args:
            d_model: Embedding dimension.
            embed_type: Kept for API compatibility with the embedding stack.
            freq: Frequency code used to determine input feature count.

        """
        super().__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """Project time features to the model dimension."""
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Combine value, position, and optional time embeddings.

    This wrapper is the main entry point used by transformer-style
    models. It merges value embeddings with positional encodings and, if
    provided, temporal features.
    """

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        """Initialize the composite embedding stack.

        Args:
            c_in: Input feature dimension.
            d_model: Embedding dimension.
            embed_type: Temporal embedding mode.
            freq: Frequency code for temporal features.
            dropout: Dropout probability applied after combining embeddings.

        """
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """Embed a CSI sequence with optional time markers."""
        if x_mark is None:
            # 2,25,512   1,25,512
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
