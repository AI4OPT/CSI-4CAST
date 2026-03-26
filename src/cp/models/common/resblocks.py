"""Residual convolution blocks shared by CSI models."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention block for convolutional feature maps."""

    def __init__(self, in_planes, ratio=4):
        """Initialize channel attention pooling and FC layers."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Compute channel attention weights."""
        # x has shape B, C, H, W
        # self.avg_pool(x) has shape B, C, 1, 1
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ResBlock(nn.Module):
    """Residual convolution block with channel attention."""

    def __init__(self, in_planes):
        """Initialize the residual convolution block."""
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Apply the residual block."""
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs
