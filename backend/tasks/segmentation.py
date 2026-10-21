"""Brain MRI Tumor Segmentation Task Adapter (`backend/tasks/segmentation.py`, Task 2.6 & 4.2).

Integrates `LightweightUNet` with `OmniFMSpatialConditioner` at the U-Net
bottleneck `(B, 256, 16, 16) <-> (B, 256, D)` while preserving all multi-scale
encoder-decoder skip connections at `32x32`, `64x64`, and `128x128`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from backend.models.segmentation import LightweightUNet
from backend.omnifm.composer import OmniFMSpatialConditioner
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.tasks.base import BaseTaskAdapter, TaskAdapterSpec, TaskForwardOutput


def soft_dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable Soft Dice loss over binary segmentation logits `(B, 1, H, W)`."""
    probs = torch.sigmoid(logits.float())
    tgt = target.float()
    if tgt.ndim == 3:
        tgt = tgt.unsqueeze(1)

    probs_flat = probs.flatten(start_dim=1)
    tgt_flat = tgt.flatten(start_dim=1)

    intersection = (probs_flat * tgt_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + tgt_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class SegmentationTaskAdapter(BaseTaskAdapter):
    """2D U-Net + OmniFM bottleneck-conditioned segmentation adapter."""

    _SPEC = TaskAdapterSpec(
        task_type="segmentation",
        input_shape_description="(B, 1, 128, 128) float32 normalized axial T1ce MRI slice",
        target_shape_description="(B, 1, 128, 128) float32/uint8 binary tumor mask in {0, 1}",
        preprocessing_description="Non-zero brain voxel z-score / min-max normalization to [0, 1], resized to 128x128",
        backbone_name="LightweightUNet (32 -> 64 -> 128 -> 256 bottleneck)",
        omnifm_insertion_stage="U-Net bottleneck feature map (B, 256, 16, 16)",
        feature_map_shape=(256, 16, 16),
        token_sequence_shape=(256, 128),
        skip_connection_policy="Encoder skip tensors (128x128, 64x64, 32x32) bypass token conversion directly into decoder stages; OmniFM conditions the 16x16 bottleneck in-place",
        head_or_decoder_description="3-stage transposed-conv U-Net decoder + 1x1 conv head emitting (B, 1, 128, 128) logits",
        task_loss_name="0.5 * BCEWithLogitsLoss + 0.5 * SoftDiceLoss",
        evaluation_metrics=("dice", "iou"),
        postprocessing_description="Sigmoid(logits) >= 0.5 -> binary tumor mask (B, 1, 128, 128)",
        share_task_head=True,
    )

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        embedding_dim: int = 128,
        num_heads: int = 4,
        top_k: int = 2,
        lambda_align: float = 0.1,
        enable_gskr: bool = True,
        enable_eca: bool = True,
        enable_psp: bool = True,
        enable_spalign: bool = True,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            lambda_align=lambda_align,
            enable_spalign=enable_spalign,
        )
        self.unet = LightweightUNet(
            in_channels=in_channels,
            out_channels=1,
            base_channels=base_channels,
        )
        self.omnifm = OmniFMSpatialConditioner(
            feature_channels=self.unet.bottleneck_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            top_k=top_k,
            max_token_grid=16,
            enable_gskr=enable_gskr,
            enable_eca=enable_eca,
            enable_psp=enable_psp,
        )

    @property
    def spec(self) -> TaskAdapterSpec:
        return self._SPEC

    def forward_features_and_predict(
        self,
        inputs: torch.Tensor,
        knowledge_bank: SpectralKnowledgeBank | None = None,
        **kwargs: Any,
    ) -> TaskForwardOutput:
        """Encode 2D MRI slice, condition bottleneck with OmniFM, and decode with skip connections."""
        bottleneck, skips = self.unet.encode(inputs)
        cond_out = self.omnifm(
            raw_image=inputs,
            feature_map=bottleneck,
            knowledge_bank=knowledge_bank,
            task_type=self.spec.task_type,
        )
        logits = self.unet.decode(cond_out.spatial_features, skips)
        return TaskForwardOutput(
            prediction=logits,
            conditioned=cond_out,
        )

    def compute_task_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        tgt = target.float()
        if tgt.ndim == 3:
            tgt = tgt.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(prediction.float(), tgt)
        dice = soft_dice_loss_from_logits(prediction, tgt)
        return 0.5 * bce + 0.5 * dice
