"""Heterogeneous Multimodal Medical Image Fusion Task Adapter (`backend/tasks/fusion.py`, Task 2.6 & 4.4).

Integrates `U2FusionInspiredNet` with `OmniFMSpatialConditioner` for paired
128x128 CT-MRI (`client_1`), PET-MRI (`client_2`), and SPECT-MRI (`client_3`)
images.

Documented Self-Supervised Fusion Training Objective & Evaluation Protocol:
---------------------------------------------------------------------------
Because multimodal image fusion has no ground-truth fused reference image in the
Harvard Medical Image Fusion dataset, training minimizes a U2Fusion-style
structure, intensity, and edge-gradient preservation objective against the
verified aligned input pair `(I_A, I_B)`:
    L_task = w_int * (0.5 * ||I_f - I_A||_1 + 0.5 * ||I_f - I_B||_1)
           + w_max * ||I_f - max(I_A, I_B)||_1
           + w_grad * ||grad(I_f) - max(|grad(I_A)|, |grad(I_B)|)||_1
           + w_ssim * (1 - 0.5 * (SSIM(I_f, I_A) + SSIM(I_f, I_B)))
Evaluation reports paired source-preservation metrics (`source_ssim`, `source_psnr`,
`source_vif`, `edge_l1`) explicitly documented as source-pair fusion metrics rather
than fabricated ground-truth reference scores.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from backend.models.fusion import U2FusionInspiredNet
from backend.omnifm.composer import OmniFMSpatialConditioner
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.tasks.base import BaseTaskAdapter, TaskAdapterSpec, TaskForwardOutput

CLIENT_TO_FUSION_MODALITY_PAIR: dict[str, str] = {
    "client_1": "CT-MRI",
    "client_2": "PET-MRI",
    "client_3": "SPECT-MRI",
}


def compute_sobel_gradients(img: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude `(B, 1, H, W)` for a single-channel image `(B, 1, H, W)`."""
    device = img.device
    dtype = torch.float32
    x = img.float()
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def _windowed_ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable local 7x7 mean/variance SSIM loss term `1 - SSIM(x, y)` in float32."""
    c1 = 0.01**2
    c2 = 0.03**2
    xf = x.float()
    yf = y.float()

    mu_x = F.avg_pool2d(xf, kernel_size=7, stride=1, padding=3)
    mu_y = F.avg_pool2d(yf, kernel_size=7, stride=1, padding=3)

    sigma_x = F.avg_pool2d(xf * xf, kernel_size=7, stride=1, padding=3) - mu_x.square()
    sigma_y = F.avg_pool2d(yf * yf, kernel_size=7, stride=1, padding=3) - mu_y.square()
    sigma_xy = F.avg_pool2d(xf * yf, kernel_size=7, stride=1, padding=3) - (mu_x * mu_y)

    sigma_x = torch.clamp(sigma_x, min=0.0)
    sigma_y = torch.clamp(sigma_y, min=0.0)

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / (den + 1e-8)
    return 1.0 - torch.clamp(ssim_map.mean(), min=-1.0, max=1.0)


class FusionTaskAdapter(BaseTaskAdapter):
    """U2Fusion-inspired network + OmniFM spatial conditioning for paired modality fusion."""

    _SPEC = TaskAdapterSpec(
        task_type="fusion",
        input_shape_description="(B, 2, 128, 128) float32 stacked aligned modality pair [I_A, I_B] in [0, 1]",
        target_shape_description="(B, 2, 128, 128) float32 source pair [I_A, I_B] used for self-supervised structure/intensity/gradient preservation",
        preprocessing_description="Paired CT-MRI, PET-MRI, or SPECT-MRI images converted to normalized float32 [0, 1] intensity at 128x128 and stacked along channel axis",
        backbone_name="U2FusionInspiredNet (32-ch stem + 3-layer DenseFeatureBlock -> 64-ch)",
        omnifm_insertion_stage="Post-DenseFeatureBlock representation (B, 64, 128, 128) pooled to 16x16 token grid (256 tokens)",
        feature_map_shape=(64, 128, 128),
        token_sequence_shape=(256, 128),
        skip_connection_policy="Full-resolution 128x128 stem features (32-ch) are concatenated with the OmniFM-conditioned 64-ch dense features in the decoder",
        head_or_decoder_description="3-layer conv decoder + Sigmoid producing (B, 1, 128, 128) fused image in [0, 1]",
        task_loss_name="U2Fusion Structure (SSIM) + Max-Intensity + Max-Sobel-Gradient L1 Loss",
        evaluation_metrics=("ssim", "vif", "l1"),
        postprocessing_description="Fused output in [0, 1] (optionally recolored with functional modality CbCr when source A is RGB PET/SPECT)",
        share_task_head=True,
    )

    def __init__(
        self,
        feat_channels: int = 64,
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
        self.fusion_net = U2FusionInspiredNet(in_channels=2, feat_channels=feat_channels)
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

    @staticmethod
    def normalize_pair_input(
        inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Accept either `(B, 2, H, W)` tensor or tuple `(img_a, img_b)` and return `(B, 2, H, W)`."""
        if isinstance(inputs, (tuple, list)):
            if len(inputs) != 2:
                raise ValueError(f"Expected pair of 2 tensors, got {len(inputs)}.")
            a, b = inputs[0], inputs[1]
            if a.ndim == 4 and a.shape[1] > 1:
                a = a.mean(dim=1, keepdim=True)
            if b.ndim == 4 and b.shape[1] > 1:
                b = b.mean(dim=1, keepdim=True)
            return torch.cat([a, b], dim=1)
        if inputs.ndim != 4 or inputs.shape[1] != 2:
            raise ValueError(
                f"Expected paired fusion input of shape (B, 2, H, W), got {tuple(inputs.shape)}."
            )
        return inputs

    def forward_features_and_predict(
        self,
        inputs: torch.Tensor,
        knowledge_bank: SpectralKnowledgeBank | None = None,
        **kwargs: Any,
    ) -> TaskForwardOutput:
        """Encode paired images, condition dense features with OmniFM, and decode fused image."""
        paired = self.normalize_pair_input(inputs)
        deep_feat, stem_skip = self.fusion_net.encode(paired)
        cond_out = self.omnifm(
            raw_image=paired,
            feature_map=deep_feat,
            knowledge_bank=knowledge_bank,
            task_type=self.spec.task_type,
        )
        fused_img = self.fusion_net.decode(cond_out.spatial_features, stem_skip)
        return TaskForwardOutput(
            prediction=fused_img,
            conditioned=cond_out,
        )

    def compute_task_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute U2Fusion-style structure + intensity + gradient preservation loss."""
        ref_pair = target if (target is not None and target.shape[1] == 2) else inputs
        if ref_pair is None or ref_pair.ndim != 4 or ref_pair.shape[1] != 2:
            if target is not None and target.shape == prediction.shape:
                return F.l1_loss(prediction.float(), target.float())
            raise ValueError(
                "Fusion compute_task_loss requires the source pair (B, 2, H, W) via target or inputs."
            )

        fused = prediction.float()
        img_a = ref_pair[:, 0:1, :, :].float()
        img_b = ref_pair[:, 1:2, :, :].float()

        # 1. Intensity preservation (mean + saliency max)
        l_int_mean = 0.5 * (F.l1_loss(fused, img_a) + F.l1_loss(fused, img_b))
        l_int_max = F.l1_loss(fused, torch.maximum(img_a, img_b))

        # 2. Sobel edge gradient preservation
        grad_f = compute_sobel_gradients(fused)
        grad_a = compute_sobel_gradients(img_a)
        grad_b = compute_sobel_gradients(img_b)
        l_grad = F.l1_loss(grad_f, torch.maximum(grad_a, grad_b))

        # 3. Structural similarity preservation against both modalities
        l_ssim = 0.5 * (_windowed_ssim_loss(fused, img_a) + _windowed_ssim_loss(fused, img_b))

        return 0.25 * l_int_mean + 0.25 * l_int_max + 0.25 * l_grad + 0.25 * l_ssim
