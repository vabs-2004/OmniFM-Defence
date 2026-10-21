"""ResNet-18 Backbone Adapted for 28x28 Medical Images (`backend/models/classification.py`).

Uses a 3x3 stride-1 stem without initial max-pooling so 28x28 MedMNIST inputs
retain a `(B, 512, 4, 4)` spatial feature map at Stage 4 before OmniFM conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SmallInputResNet18Backbone(nn.Module):
    """ResNet-18 feature extractor tailored for small 28x28 RGB images.

    Spatial progression for `(B, 3, 28, 28)` input:
    - Stem (`3x3`, stride 1): `(B, 64, 28, 28)`
    - Layer 1 (stride 1): `(B, 64, 28, 28)`
    - Layer 2 (stride 2): `(B, 128, 14, 14)`
    - Layer 3 (stride 2): `(B, 256, 7, 7)`
    - Layer 4 (stride 2): `(B, 512, 4, 4)`
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        base = tv_models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = base.bn1
        self.relu = base.relu
        # Omit base.maxpool to preserve spatial resolution on 28x28 inputs
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Stage-4 spatial feature map of shape `(B, 512, H/7, W/7)` -> `(B, 512, 4, 4)` for 28x28."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
