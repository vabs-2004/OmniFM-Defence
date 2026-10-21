"""Lightweight 2D U-Net for Brain MRI Tumor Segmentation (`backend/models/segmentation.py`).

Designed for 128x128 axial T1ce MRI slices on 6 GB VRAM while exposing an explicit
`encode()` / `decode()` split so `OmniFMSpatialConditioner` can condition the
`(B, 256, 16, 16)` bottleneck without disrupting encoder-decoder skip connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else (4 if out_channels % 4 == 0 else 1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightUNet(nn.Module):
    """Encoder-decoder U-Net with explicit bottleneck conditioning hook.

    Spatial progression for `(B, 1, 128, 128)` input:
    - `enc1`: `(B, 32, 128, 128)` (skip 1)
    - `enc2`: `(B, 64, 64, 64)`   (skip 2)
    - `enc3`: `(B, 128, 32, 32)`  (skip 3)
    - `bottleneck`: `(B, 256, 16, 16)` -> conditioned by OmniFM -> `(B, 256, 16, 16)`
    - `dec3` + skip 3 -> `(B, 128, 32, 32)`
    - `dec2` + skip 2 -> `(B, 64, 64, 64)`
    - `dec1` + skip 1 -> `(B, 32, 128, 128)`
    - `out_head` -> `(B, 1, 128, 128)` logits
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        cb = base_channels * 8

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = cb

        self.enc1 = DoubleConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConvBlock(c3, cb)

        self.up3 = nn.ConvTranspose2d(cb, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(c1 * 2, c1)

        self.out_head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Run encoder and return `(bottleneck_features, (skip1, skip2, skip3))`."""
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        b = self.bottleneck(self.pool3(s3))
        return b, (s1, s2, s3)

    def decode(
        self,
        bottleneck_features: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Decode conditioned bottleneck features with aligned skip connections."""
        s1, s2, s3 = skips

        u3 = self.up3(bottleneck_features)
        if u3.shape[-2:] != s3.shape[-2:]:
            u3 = F.interpolate(u3, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, s3], dim=1))

        u2 = self.up2(d3)
        if u2.shape[-2:] != s2.shape[-2:]:
            u2 = F.interpolate(u2, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != s1.shape[-2:]:
            u1 = F.interpolate(u1, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))

        return self.out_head(d1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, skips = self.encode(x)
        return self.decode(b, skips)
