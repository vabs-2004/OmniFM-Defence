"""Embedding-wise Cross-Attention (`backend/omnifm/eca.py`, Task 2.3).

Conditions tokenized backbone representations `Q` of shape `(B, L, D)` on
spectral prototypes `K, V` of shape `(B, M_ctx, D)` (combining local spectral
tokens and retrieved global spectral prototypes from GSKR):
    Z = Attention(Q, K, V)
with residual connections, LayerNorm, and a feed-forward projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EmbeddingCrossAttention(nn.Module):
    """Multi-head cross-attention conditioning backbone tokens on spectral prototypes.

    Inputs:
        backbone_tokens: Tensor of shape `(B, L, D)` acting as queries `Q`.
        retrieved_prototypes: Tensor of shape `(B, K, D)` from GSKR (`K >= 0`), or `None`.
        local_spectral_token: Optional Tensor of shape `(B, D)` or `(B, 1, D)` from
            `SpectralEncoder`. When provided, it is prepended to `retrieved_prototypes`
            so that keys/values have shape `(B, 1 + K, D)` and remain non-empty even
            when the knowledge bank is empty (`K == 0`).

    Outputs:
        Tuple `(fused_backbone_tokens, spectral_prefix_tokens)`:
        - `fused_backbone_tokens`: `(B, L, D)` conditioned backbone sequence.
        - `spectral_prefix_tokens`: `(B, 1, D)` pooled cross-attended spectral prefix
          token ready for Prefix-Suffix Spectral Prompting (PSP).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim < 8:
            raise ValueError(f"embedding_dim must be >= 8, got {embedding_dim}.")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embedding_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
        )
        self.prefix_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.prefix_norm = nn.LayerNorm(self.embedding_dim)

    def _build_spectral_context(
        self,
        backbone_tokens: torch.Tensor,
        retrieved_prototypes: torch.Tensor | None,
        local_spectral_token: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Assemble spectral key/value tensor `(B, M_ctx, D)` from local + retrieved tokens."""
        bsz, _seq_len, dim = backbone_tokens.shape
        parts: list[torch.Tensor] = []

        if local_spectral_token is not None:
            if local_spectral_token.ndim == 2:
                local_tok = local_spectral_token.unsqueeze(1)
            elif local_spectral_token.ndim == 3:
                local_tok = local_spectral_token
            else:
                raise ValueError(
                    f"local_spectral_token must be 2D (B, D) or 3D (B, M, D), got {tuple(local_spectral_token.shape)}."
                )
            if local_tok.shape[0] != bsz or local_tok.shape[2] != dim:
                raise ValueError(
                    f"local_spectral_token shape {tuple(local_tok.shape)} incompatible with (B={bsz}, D={dim})."
                )
            parts.append(local_tok.to(dtype=backbone_tokens.dtype, device=backbone_tokens.device))

        if retrieved_prototypes is not None:
            if retrieved_prototypes.ndim != 3:
                raise ValueError(
                    f"retrieved_prototypes must be 3D (B, K, D), got {tuple(retrieved_prototypes.shape)}."
                )
            if retrieved_prototypes.shape[0] != bsz or retrieved_prototypes.shape[2] != dim:
                raise ValueError(
                    f"retrieved_prototypes shape {tuple(retrieved_prototypes.shape)} incompatible with (B={bsz}, D={dim})."
                )
            if retrieved_prototypes.shape[1] > 0:
                parts.append(
                    retrieved_prototypes.to(
                        dtype=backbone_tokens.dtype, device=backbone_tokens.device
                    )
                )

        if not parts:
            return None
        return torch.cat(parts, dim=1)  # (B, M_ctx, D)

    def forward(
        self,
        backbone_tokens: torch.Tensor,
        retrieved_prototypes: torch.Tensor | None = None,
        local_spectral_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run cross-attention between `backbone_tokens` `(B, L, D)` and spectral context."""
        if backbone_tokens.ndim != 3 or backbone_tokens.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected backbone_tokens of shape (B, L, {self.embedding_dim}), "
                f"got {tuple(backbone_tokens.shape)}."
            )

        bsz, seq_len, dim = backbone_tokens.shape
        if bsz < 1 or seq_len < 1:
            raise ValueError(f"Invalid backbone_tokens shape {tuple(backbone_tokens.shape)}.")

        spectral_ctx = self._build_spectral_context(
            backbone_tokens=backbone_tokens,
            retrieved_prototypes=retrieved_prototypes,
            local_spectral_token=local_spectral_token,
        )

        # If neither local nor retrieved spectral tokens were supplied, fall back cleanly
        if spectral_ctx is None or spectral_ctx.shape[1] == 0:
            fused = self.norm2(backbone_tokens + self.ffn(self.norm1(backbone_tokens)))
            prefix = self.prefix_norm(self.prefix_proj(fused.mean(dim=1, keepdim=True)))
            return fused, prefix

        ctx_len = int(spectral_ctx.shape[1])

        # Multi-head projections:
        # Q: (B, L, D) -> (B, H, L, Dh)
        # K, V: (B, M_ctx, D) -> (B, H, M_ctx, Dh)
        q = (
            self.q_proj(backbone_tokens)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(spectral_ctx)
            .view(bsz, ctx_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(spectral_ctx)
            .view(bsz, ctx_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention in float32 for numerical stability
        attn_logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=-1).to(dtype=backbone_tokens.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, H, L, M_ctx) @ (B, H, M_ctx, Dh) -> (B, L, D)
        attn_out = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, dim)
        )
        attn_out = self.out_proj(attn_out)

        # Residual + LayerNorm + FFN
        x1 = self.norm1(backbone_tokens + attn_out)
        fused = self.norm2(x1 + self.ffn(x1))

        # Summary prefix token Z_prefix of shape (B, 1, D)
        prefix_raw = attn_out.mean(dim=1, keepdim=True) + spectral_ctx.mean(dim=1, keepdim=True)
        prefix_token = self.prefix_norm(self.prefix_proj(prefix_raw))

        return fused, prefix_token
