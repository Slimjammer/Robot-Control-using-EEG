"""
src/model/blocks.py
===================
Building blocks for EEGModel:
  FilterBank, SpatialBlock, MultiScaleBlock,
  PositionalEncoding, AttentionPooling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterBank(nn.Module):
    """
    Temporal convolution that learns per-frequency features.
    Kernel size ~100 ms window (25 samples at 250 Hz).
    """
    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=25, padding=12, bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.bn(self.conv(x)))


class SpatialBlock(nn.Module):
    """
    Depth-wise + point-wise convolution to mix spatial (channel) information.
    """
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.depthwise  = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.pointwise  = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn         = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.elu(self.bn(x))


class MultiScaleBlock(nn.Module):
    """
    Parallel convolutions with kernels 3/5/7 — captures patterns at
    multiple temporal scales. Adds a residual skip connection.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branch3  = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.branch5  = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.branch7  = nn.Conv1d(in_ch, out_ch, 7, padding=3)
        self.bn       = nn.BatchNorm1d(out_ch)
        self.dropout  = nn.Dropout(0.2)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch3(x) + self.branch5(x) + self.branch7(x)
        out = self.dropout(self.bn(out))
        return F.elu(self.residual(x) + 0.3 * out)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the Transformer encoder.
    """
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)].to(x.device)


class AttentionPooling(nn.Module):
    """
    Learns a weighted average over the time axis before the classifier.
    More informative than simple mean/last-step pooling.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (x * weights).sum(dim=1)                 # (B, dim)
