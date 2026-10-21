"""U2Fusion-Inspired Multimodal Medical Image Fusion Network (`backend/models/fusion.py`).

Clean-room PyTorch implementation inspired by U2Fusion (Xu et al., IEEE TPAMI 2020):
accepts a spatially aligned pair of 128x128 medical images `(img_a, img_b)` stacked
along the channel axis `(B, 2, 128, 128)`, extracts dense multi-layer complementary
features `(B, 64, 128, 128)`, allows OmniFM conditioning on the fused feature map,
and reconstructs a single fused output image `(B, 1, 128, 128)` in `[0, 1]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseFeatureBlock(nn.Module):
    """3-layer densely connected convolutional block for feature preservation."""

    def __init__(self, in_channels: int, growth_channels: int = 16, out_channels: int = 64) -> None:
        super().__init__()
        c1 = in_channels
        c2 = c1 + growth_channels
        c3 = c2 + growth_channels
        c4 = c3 + growth_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c3, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.compress = nn.Sequential(
            nn.Conv2d(c4, out_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv1(x)
        x1 = torch.cat([x, f1], dim=1)
        f2 = self.conv2(x1)
        x2 = torch.cat([x1, f2], dim=1)
        f3 = self.conv3(x2)
        x3 = torch.cat([x2, f3], dim=1)
        return self.compress(x3)


class U2FusionInspiredNet(nn.Module):
    """Dense encoder-decoder fusion network for paired `(B, 2, H, W)` images."""

    def __init__(self, in_channels: int = 2, feat_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dense_encoder = DenseFeatureBlock(
            in_channels=32,
            growth_channels=16,
            out_channels=feat_channels,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_channels + 32, 48, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, paired_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract `(deep_features, stem_skip)` from `(B, 2, H, W)` paired input."""
        stem_feat = self.stem(paired_input)
        deep_feat = self.dense_encoder(stem_feat)
        return deep_feat, stem_feat

    def decode(self, conditioned_features: torch.Tensor, stem_skip: torch.Tensor) -> torch.Tensor:
        """Reconstruct fused single-channel image `(B, 1, H, W)` in `[0, 1]`."""
        return self.decoder(torch.cat([conditioned_features, stem_skip], dim=1))

    def forward(self, paired_input: torch.Tensor) -> torch.Tensor:
        deep_feat, stem_skip = self.encode(paired_input)
        return self.decode(deep_feat, stem_skip)
