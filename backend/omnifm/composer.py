"""Reusable OmniFM Composition Layer (`backend/omnifm/composer.py`, Task 2.6).

Bridges spatial CNN feature maps `(B, C_feat, H_f, W_f)` and the token-based
OmniFM pipeline (`SpectralEncoder` -> `GSKR` -> `ECA` -> `PSP`), then restores
exact spatial feature dimensions `(B, C_feat, H_f, W_f)` via residual projection
so decoders and skip connections in segmentation, super-resolution, and fusion
remain spatially aligned.

Also supports clean component ablation flags (`enable_gskr`, `enable_eca`,
`enable_psp`) so disabled components are genuinely bypassed in forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.omnifm.eca import EmbeddingCrossAttention
from backend.omnifm.gskr import GSKRRetrievalResult, SpectralKnowledgeBank
from backend.omnifm.psp import PERSONALIZED_PROMPT_PARAM_NAME, PSPOutput, PrefixSuffixSpectralPrompting
from backend.omnifm.spectral import SpectralEncoder


@dataclass
class OmniFMConditionedOutput:
    """Outputs of `OmniFMSpatialConditioner` for a task backbone feature stage.

    Attributes:
        spatial_features: Conditioned feature map of shape `(B, C_feat, H_f, W_f)`,
            matching the input `feature_map` spatial dimensions 1-to-1.
        pooled_token: Global conditioned token of shape `(B, D)` suitable for
            classification or global context heads.
        local_spectral_token: L2-normalized spectral token `s` of shape `(B, D)`
            from `SpectralEncoder`.
        retrieved_prototypes: Retrieved global prototypes `S_g` of shape `(B, K_actual, D)`
            from `SpectralKnowledgeBank` (`K_actual == 0` when empty or GSKR disabled).
        prompted_sequence: Full token sequence `[Z_prefix || r_backbone || c_client]`
            of shape `(B, P_pre + L + P_suf, D)`.
        has_valid_prototypes: True when `retrieved_prototypes.shape[1] > 0`.
    """

    spatial_features: torch.Tensor
    pooled_token: torch.Tensor
    local_spectral_token: torch.Tensor
    retrieved_prototypes: torch.Tensor
    prompted_sequence: torch.Tensor
    has_valid_prototypes: bool


class OmniFMSpatialConditioner(nn.Module):
    """Composable OmniFM spectral-conditioning block for CNN feature maps.

    Spatial-to-Token and Token-to-Spatial Protocol:
    1. Input image `(B, C_img, H_img, W_img)` -> `SpectralEncoder` -> `s` `(B, D)`.
    2. `s` queries `SpectralKnowledgeBank` (if provided and `enable_gskr=True`) ->
       `S_g` `(B, K_actual, D)`.
    3. Input CNN feature map `(B, C_feat, H_f, W_f)`:
       - Optionally pooled to `(H_t, W_t)` where `H_t = min(H_f, max_token_grid)` and
         `W_t = min(W_f, max_token_grid)` to bound attention complexity on 6 GB VRAM.
       - Projected `C_feat -> D` via `1x1` convolution and flattened to tokens
         `r_backbone` of shape `(B, L, D)` with `L = H_t * W_t`.
    4. If `enable_eca=True`, `EmbeddingCrossAttention` conditions `r_backbone` on
       `[s, S_g]` -> `(r_eca, Z_prefix)`.
    5. If `enable_psp=True`, `PrefixSuffixSpectralPrompting` applies prefix and
       client-specific suffix prompts -> `spatial_tokens` `(B, L, D)` and
       `pooled_token` `(B, D)`.
    6. `spatial_tokens` `(B, L, D)` is reshaped to `(B, D, H_t, W_t)`, projected
       `D -> C_feat` via `1x1` convolution, bilinearly interpolated back to
       `(H_f, W_f)` if `H_t != H_f` or `W_t != W_f`, and added as a gated residual
       to `feature_map` `(B, C_feat, H_f, W_f)`.
    """

    def __init__(
        self,
        feature_channels: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        top_k: int = 2,
        low_pass_ratio: float = 0.25,
        max_token_grid: int = 16,
        enable_gskr: bool = True,
        enable_eca: bool = True,
        enable_psp: bool = True,
    ) -> None:
        super().__init__()
        if feature_channels < 1:
            raise ValueError(f"feature_channels must be >= 1, got {feature_channels}.")

        self.feature_channels = int(feature_channels)
        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.top_k = int(top_k)
        self.max_token_grid = int(max_token_grid)
        self.enable_gskr = bool(enable_gskr)
        self.enable_eca = bool(enable_eca)
        self.enable_psp = bool(enable_psp)

        self.spectral_encoder = SpectralEncoder(
            embedding_dim=self.embedding_dim,
            low_pass_ratio=low_pass_ratio,
        )
        self.eca = EmbeddingCrossAttention(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
        )
        self.psp = PrefixSuffixSpectralPrompting(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
        )

        # 1x1 projections between CNN feature channels C_feat and token dim D
        self.to_tokens_proj = nn.Conv2d(self.feature_channels, self.embedding_dim, kernel_size=1)
        self.from_tokens_proj = nn.Conv2d(self.embedding_dim, self.feature_channels, kernel_size=1)
        self.spatial_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.out_norm = nn.GroupNorm(
            num_groups=self._choose_num_groups(self.feature_channels),
            num_channels=self.feature_channels,
        )

    @staticmethod
    def _choose_num_groups(channels: int) -> int:
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def get_personalized_parameter_names(self, prefix: str = "") -> set[str]:
        """Return parameter names that must remain client-local (excluded from FedAvg)."""
        if not self.enable_psp:
            return set()
        psp_prefix = f"{prefix}.psp" if prefix else "psp"
        return self.psp.get_personalized_parameter_names(prefix=psp_prefix)

    def forward(
        self,
        raw_image: torch.Tensor,
        feature_map: torch.Tensor,
        knowledge_bank: SpectralKnowledgeBank | None = None,
        task_type: str | None = None,
        avoid_exact_self: bool = True,
    ) -> OmniFMConditionedOutput:
        """Condition `feature_map` `(B, C_feat, H_f, W_f)` using `raw_image` `(B, C_in, H_in, W_in)`."""
        if raw_image.ndim != 4:
            raise ValueError(f"Expected 4D raw_image (B, C, H, W), got {tuple(raw_image.shape)}.")
        if feature_map.ndim != 4:
            raise ValueError(
                f"Expected 4D feature_map (B, C_feat, H_f, W_f), got {tuple(feature_map.shape)}."
            )
        if raw_image.shape[0] != feature_map.shape[0]:
            raise ValueError(
                f"Batch size mismatch: raw_image {raw_image.shape[0]} vs feature_map {feature_map.shape[0]}."
            )
        if feature_map.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected feature_map with {self.feature_channels} channels, got {feature_map.shape[1]}."
            )

        bsz, _c_feat, h_f, w_f = feature_map.shape

        # 1. Extract local spectral token s: (B, D)
        local_spectral = self.spectral_encoder(raw_image)

        # 2. Retrieve global prototypes S_g via GSKR if enabled and bank provided
        if self.enable_gskr and knowledge_bank is not None:
            gskr_res: GSKRRetrievalResult = knowledge_bank.retrieve(
                query_tokens=local_spectral,
                top_k=self.top_k,
                task_type=task_type,
                avoid_exact_self=avoid_exact_self,
            )
            retrieved = gskr_res.prototypes.to(device=feature_map.device, dtype=feature_map.dtype)
            has_valid = gskr_res.has_valid
        else:
            retrieved = torch.empty(
                (bsz, 0, self.embedding_dim),
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
            has_valid = False

        # 3. Convert spatial feature map (B, C_feat, H_f, W_f) -> tokens (B, L, D)
        h_t = min(h_f, self.max_token_grid)
        w_t = min(w_f, self.max_token_grid)
        if (h_t, w_t) != (h_f, w_f):
            pooled_feat = F.adaptive_avg_pool2d(feature_map, (h_t, w_t))
        else:
            pooled_feat = feature_map

        proj_feat = self.to_tokens_proj(pooled_feat)  # (B, D, H_t, W_t)
        backbone_tokens = proj_feat.flatten(start_dim=2).transpose(1, 2).contiguous()  # (B, L, D)

        # 4. Apply Embedding-wise Cross-Attention (ECA)
        if self.enable_eca:
            eca_tokens, prefix_token = self.eca(
                backbone_tokens=backbone_tokens,
                retrieved_prototypes=retrieved,
                local_spectral_token=local_spectral.to(dtype=backbone_tokens.dtype),
            )
        else:
            eca_tokens = backbone_tokens
            prefix_token = local_spectral.to(dtype=backbone_tokens.dtype).unsqueeze(1)

        # 5. Apply Prefix-Suffix Spectral Prompting (PSP)
        if self.enable_psp:
            psp_out: PSPOutput = self.psp(
                backbone_tokens=eca_tokens,
                prefix_tokens=prefix_token,
            )
            cond_tokens = psp_out.spatial_tokens
            pooled_token = psp_out.pooled_token
            prompted_seq = psp_out.prompted_sequence
        else:
            cond_tokens = eca_tokens
            pooled_token = 0.5 * (
                eca_tokens.mean(dim=1) + local_spectral.to(dtype=backbone_tokens.dtype)
            )
            prompted_seq = eca_tokens

        # 6. Restore spatial layout: (B, L, D) -> (B, D, H_t, W_t) -> (B, C_feat, H_f, W_f)
        spatial_d = cond_tokens.transpose(1, 2).contiguous().view(bsz, self.embedding_dim, h_t, w_t)
        spatial_c = self.from_tokens_proj(spatial_d)
        if (h_t, w_t) != (h_f, w_f):
            spatial_c = F.interpolate(
                spatial_c,
                size=(h_f, w_f),
                mode="bilinear",
                align_corners=False,
            )

        gate = self.spatial_gate.to(dtype=feature_map.dtype, device=feature_map.device)
        conditioned_map = self.out_norm(feature_map + gate * spatial_c)

        return OmniFMConditionedOutput(
            spatial_features=conditioned_map,
            pooled_token=pooled_token,
            local_spectral_token=local_spectral,
            retrieved_prototypes=retrieved,
            prompted_sequence=prompted_seq,
            has_valid_prototypes=has_valid,
        )


__all__ = [
    "OmniFMConditionedOutput",
    "OmniFMSpatialConditioner",
    "PERSONALIZED_PROMPT_PARAM_NAME",
]
