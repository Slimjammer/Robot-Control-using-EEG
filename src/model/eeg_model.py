"""
src/model/eeg_model.py
======================
EEGModel: FilterBank → SpatialBlock → MultiScale CNN
          → Learnable Positional Embedding → Transformer
          → AttentionPooling → Classifier
"""

import torch
import torch.nn as nn

from src.model.blocks import (
    FilterBank,
    SpatialBlock,
    MultiScaleBlock,
    AttentionPooling,
)


class EEGModel(nn.Module):
    """
    Hybrid CNN + Transformer model for EEG motor-imagery classification.

    Input shape : (batch, T, n_channels)   — (N, 1000, 10) during training
    Output shape: (batch, n_classes)

    Architecture:
      1. FilterBank      — temporal feature learning (like learnable bandpass)
      2. SpatialBlock    — mix spatial (electrode) information
      3. MultiScale CNN  — extract features at different time-scales
      4. Positional emb  — learnable token positions for the Transformer
      5. Transformer     — capture long-range temporal dependencies
      6. AttentionPool   — weighted temporal aggregation
      7. MLP head        — final classification
    """

    def __init__(self, n_channels: int, n_classes: int, max_len: int = 1000):
        super().__init__()

        # 1. Temporal frequency learning
        self.filterbank = FilterBank(n_channels, out_channels=32)

        # 2. Spatial mixing (depthwise + pointwise)
        self.spatial = SpatialBlock(in_channels=32, out_channels=64)

        # 3. Multi-scale temporal CNN
        self.cnn = nn.Sequential(
            MultiScaleBlock(64, 128),
            nn.MaxPool1d(2),          # T → T/2

            MultiScaleBlock(128, 128),
            nn.MaxPool1d(2),          # T/2 → T/4  (250 at 1000-sample input)
        )

        # 4. Learnable positional embedding (size = max_len // 4 after two pool layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, 128))

        # 5. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 6. Attention pooling over time
        self.attn_pool = AttentionPooling(128)

        # 7. Classification head
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C)
        x = x.permute(0, 2, 1)          # → (B, C, T)

        x = self.filterbank(x)           # → (B, 32, T)
        x = self.spatial(x)              # → (B, 64, T)
        x = self.cnn(x)                  # → (B, 128, T/4)

        x = x.permute(0, 2, 1)          # → (B, T/4, 128)

        # Add positional embedding (clip to actual sequence length)
        x = x + self.pos_embedding[:, :x.size(1)]

        x = self.transformer(x)          # → (B, T/4, 128)
        x = self.attn_pool(x)            # → (B, 128)

        return self.fc(x)                # → (B, n_classes)
