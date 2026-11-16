"""Small causal TCN for frame-level possession classification."""

import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation)

    def forward(self, x):
        y = self.conv(x)
        return y[..., :x.shape[-1]]


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout(F.relu(self.norm1(self.conv1(x))))
        y = self.dropout(F.relu(self.norm2(self.conv2(y))))
        return F.relu(y + x)


class PossessionTCN(nn.Module):
    def __init__(self, in_channels, n_classes=4, channels=48,
                 levels=6, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.Sequential(*[
            TCNBlock(channels, kernel_size, 2 ** level, dropout)
            for level in range(levels)
        ])
        self.head = nn.Conv1d(channels, n_classes, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x)
