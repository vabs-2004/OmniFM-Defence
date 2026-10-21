"""Histopathology Super-Resolution Task Adapter (`backend/tasks/superresolution.py`, Task 2.6 & 4.3).

Integrates `LightweightSRNet` with `OmniFMSpatialConditioner` at the deep
residual feature stage `(B, 64, 64, 64)` (pooled to a `16x16 = 256` token grid
for cross-attention and PSP, then bilinearly projected back to `(B, 64, 64, 64)`
and combined with the shallow skip connection before PixelShuffle x2/x4/x8 upsampling).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from backend.models.superresolution import LightweightSRNet
from backend.omnifm.composer import OmniFMSpatialConditioner
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.tasks.base import BaseTaskAdapter, TaskAdapterSpec, TaskForwardOutput


class SuperResolutionTaskAdapter(BaseTaskAdapter):
    """Lightweight residual SR network + OmniFM spatial conditioning."""

    _SPEC = TaskAdapterSpec(
        task_type="superresolution",
        input_shape_description="(B, 3, 64, 64) float32 RGB low-resolution patch in [0, 1]",
        target_shape_description="(B, 3, 64*scale, 64*scale) float32 RGB high-resolution target patch in [0, 1]",
        preprocessing_description="Paired HR patch cropped on patient-split BreaKHis slide, bicubically downsampled by scale in {2, 4, 8} to 64x64 LR input",
        backbone_name="LightweightSRNet (64-ch 4-block EDSR-style body + x2/x4/x8 PixelShuffle)",
        omnifm_insertion_stage="Post-residual-body feature map (B, 64, 64, 64) pooled to 16x16 token grid (256 tokens)",
        feature_map_shape=(64, 64, 64),
        token_sequence_shape=(256, 128),
        skip_connection_policy="Global shallow feature map (B, 64, 64, 64) is preserved and added to the OmniFM-conditioned deep features prior to PixelShuffle upsampling",
        head_or_decoder_description="Scale-selectable PixelShuffle upsamplers for x2 (128x128), x4 (256x256), and x8 (512x512) + 3x3 RGB reconstruction",
        task_loss_name="L1Loss",
        evaluation_metrics=("psnr", "ssim"),
        postprocessing_description="Clamp output reconstruction to [0.0, 1.0]",
        share_task_head=True,
    )

    def __init__(
        self,
        feat_channels: int = 64,
        num_blocks: int = 4,
        default_scale: int = 2,
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
        self.default_scale = int(default_scale)
        self.sr_net = LightweightSRNet(
            in_channels=3,
            feat_channels=feat_channels,
            num_blocks=num_blocks,
            default_scale=default_scale,
        )
        self.omnifm = OmniFMSpatialConditioner(
            feature_channels=feat_channels,
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
        scale: int | None = None,
        **kwargs: Any,
    ) -> TaskForwardOutput:
        """Extract LR features, condition with OmniFM, and reconstruct HR image at `scale`."""
        active_scale = int(scale if scale is not None else self.default_scale)
        deep_feat, shallow_skip = self.sr_net.extract_features(inputs)
        cond_out = self.omnifm(
            raw_image=inputs,
            feature_map=deep_feat,
            knowledge_bank=knowledge_bank,
            task_type=self.spec.task_type,
        )
        sr_img = self.sr_net.upsample(
            conditioned_features=cond_out.spatial_features,
            shallow_skip=shallow_skip,
            scale=active_scale,
        )
        return TaskForwardOutput(
            prediction=sr_img,
            conditioned=cond_out,
            extra={"scale": active_scale},
        )

    def compute_task_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return F.l1_loss(prediction.float(), target.float())
