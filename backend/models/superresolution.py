"""Lightweight Residual Super-Resolution Network (`backend/models/superresolution.py`).

Supports scale-selectable x2, x4, and x8 histopathology super-resolution from
64x64 low-resolution RGB patches (`(B, 3, 64, 64) -> (B, 3, 64*scale, 64*scale)`).
Exposes `extract_features()` and `upsample()` so `OmniFMSpatialConditioner`
can condition the deep feature map before sub-pixel (`PixelShuffle`) reconstruction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    """Residual 3x3 conv block without batch norm (standard EDSR-style block for SR)."""

    def __init__(self, channels: int, res_scale: float = 0.2) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.body(x)


def _build_pixelshuffle_upsampler(channels: int, scale: int) -> nn.Sequential:
    """Build cascade of x2 PixelShuffle blocks for scale in {2, 4, 8}."""
    if scale not in (2, 4, 8):
        raise ValueError(f"Unsupported SR scale {scale}: must be 2, 4, or 8.")
    num_steps = int(round(math.log2(scale)))
    layers: list[nn.Module] = []
    for _ in range(num_steps):
        layers.append(nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1))
        layers.append(nn.PixelShuffle(upscale_factor=2))
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class LightweightSRNet(nn.Module):
    """Multi-scale residual super-resolution network supporting x2, x4, and x8."""

    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 64,
        num_blocks: int = 4,
        default_scale: int = 2,
    ) -> None:
        super().__init__()
        if default_scale not in (2, 4, 8):
            raise ValueError(f"default_scale must be 2, 4, or 8, got {default_scale}.")

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.default_scale = default_scale

        self.head = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualConvBlock(feat_channels) for _ in range(num_blocks)],
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
        )

        self.upsamplers = nn.ModuleDict(
            {
                "2": _build_pixelshuffle_upsampler(feat_channels, 2),
                "4": _build_pixelshuffle_upsampler(feat_channels, 4),
                "8": _build_pixelshuffle_upsampler(feat_channels, 8),
            }
        )
        self.tail = nn.Conv2d(feat_channels, in_channels, kernel_size=3, padding=1)

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(deep_features, shallow_skip)` each of shape `(B, C_feat, H, W)`."""
        shallow = self.head(x)
        deep = self.body(shallow)
        return deep, shallow

    def upsample(
        self,
        conditioned_features: torch.Tensor,
        shallow_skip: torch.Tensor,
        scale: int | None = None,
    ) -> torch.Tensor:
        """Upsample conditioned features + shallow skip by `scale` in {2, 4, 8}."""
        s = int(scale if scale is not None else self.default_scale)
        key = str(s)
        if key not in self.upsamplers:
            raise ValueError(f"Unsupported super-resolution scale {s}. Allowed: 2, 4, 8.")
        fused = conditioned_features + shallow_skip
        up = self.upsamplers[key](fused)
        out = torch.sigmoid(self.tail(up))
        return out

    def forward(self, x: torch.Tensor, scale: int | None = None) -> torch.Tensor:
        deep, shallow = self.extract_features(x)
        return self.upsample(deep, shallow, scale=scale)
