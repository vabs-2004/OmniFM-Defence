"""Prefix-Suffix Spectral Prompting (`backend/omnifm/psp.py`, Task 2.4).

Constructs the prompted token sequence:
    r' = [Z_prefix || r_backbone || c_client]
and projects the prompted sequence back into both:
  1. Conditioned spatial tokens `(B, L, D)` (preserving exact length `L = H_f * W_f`
     for spatial decoders in segmentation, super-resolution, and fusion), and
  2. A pooled representation `(B, D)` for classification heads.

Personalization Policy:
-----------------------
`client_suffix_prompt` (`c_client`) is explicitly marked as a client-local
personalized parameter via `PERSONALIZED_PARAM_KEYWORDS = ("client_suffix_prompt",)`
and `get_personalized_parameter_names()` so that federated aggregation (`FedAvg`)
never averages client-specific suffix prompts across hospitals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

PERSONALIZED_PROMPT_PARAM_NAME = "client_suffix_prompt"


@dataclass
class PSPOutput:
    """Outputs produced by `PrefixSuffixSpectralPrompting`.

    Attributes:
        prompted_sequence: Concatenated sequence `[Z_prefix || r_backbone || c_client]`
            of shape `(B, P_pre + L + P_suf, D)`.
        spatial_tokens: Prompt-conditioned backbone tokens of shape `(B, L, D)`,
            aligned 1-to-1 with the input spatial token grid.
        pooled_token: Global summary token of shape `(B, D)` combining prefix,
            backbone, and client-specific suffix tokens.
        prefix_tokens: Effective prefix tensor of shape `(B, P_pre, D)`.
        suffix_tokens: Expanded client-specific suffix tensor of shape `(B, P_suf, D)`.
    """

    prompted_sequence: torch.Tensor
    spatial_tokens: torch.Tensor
    pooled_token: torch.Tensor
    prefix_tokens: torch.Tensor
    suffix_tokens: torch.Tensor


class PrefixSuffixSpectralPrompting(nn.Module):
    """Prefix-Suffix Spectral Prompting (PSP) module.

    Args:
        embedding_dim: Token embedding dimension `D` (default: 128).
        num_prefix_tokens: Number of spectral prefix tokens `P_pre` (default: 1).
        num_suffix_tokens: Number of client-specific suffix prompt tokens `P_suf` (default: 1).
        num_heads: Attention heads used for lightweight prompt-to-backbone interaction (default: 4).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_prefix_tokens: int = 1,
        num_suffix_tokens: int = 1,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim < 8:
            raise ValueError(f"embedding_dim must be >= 8, got {embedding_dim}.")
        if num_prefix_tokens < 1 or num_suffix_tokens < 1:
            raise ValueError("num_prefix_tokens and num_suffix_tokens must be >= 1.")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embedding_dim = int(embedding_dim)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.num_suffix_tokens = int(num_suffix_tokens)
        self.num_heads = int(num_heads)

        # Shared prefix adapter mapping ECA prefix token (B, 1, D) -> (B, P_pre, D)
        self.shared_prefix_embed = nn.Parameter(
            torch.randn(1, self.num_prefix_tokens, self.embedding_dim) * 0.02
        )
        self.prefix_adapter = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Client-specific personalized suffix / CLS prompt (excluded from FedAvg)
        self.client_suffix_prompt = nn.Parameter(
            torch.randn(1, self.num_suffix_tokens, self.embedding_dim) * 0.02
        )

        # Prompt-aware gating / fusion layer over [Z_prefix || r_backbone || c_client]
        self.prompt_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.spatial_norm = nn.LayerNorm(self.embedding_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )

    def get_personalized_parameter_names(self, prefix: str = "") -> set[str]:
        """Return parameter names belonging to client-local personalized prompts."""
        base = f"{prefix}." if prefix else ""
        return {f"{base}{PERSONALIZED_PROMPT_PARAM_NAME}"}

    @staticmethod
    def is_personalized_parameter(param_name: str) -> bool:
        """Return True if `param_name` refers to a personalized PSP parameter."""
        return PERSONALIZED_PROMPT_PARAM_NAME in param_name.split(".")

    def forward(
        self,
        backbone_tokens: torch.Tensor,
        prefix_tokens: torch.Tensor | None = None,
    ) -> PSPOutput:
        """Apply Prefix-Suffix Spectral Prompting to `backbone_tokens` `(B, L, D)`."""
        if backbone_tokens.ndim != 3 or backbone_tokens.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected backbone_tokens of shape (B, L, {self.embedding_dim}), "
                f"got {tuple(backbone_tokens.shape)}."
            )

        bsz, seq_len, dim = backbone_tokens.shape
        if bsz < 1 or seq_len < 1:
            raise ValueError(f"Invalid backbone_tokens shape {tuple(backbone_tokens.shape)}.")

        # 1. Construct prefix tokens Z_prefix of shape (B, P_pre, D)
        base_prefix = self.shared_prefix_embed.expand(bsz, self.num_prefix_tokens, dim).to(
            dtype=backbone_tokens.dtype, device=backbone_tokens.device
        )
        if prefix_tokens is not None:
            if prefix_tokens.ndim == 2:
                prefix_tokens = prefix_tokens.unsqueeze(1)
            if (
                prefix_tokens.ndim != 3
                or prefix_tokens.shape[0] != bsz
                or prefix_tokens.shape[2] != dim
            ):
                raise ValueError(
                    f"Incompatible prefix_tokens shape {tuple(prefix_tokens.shape)} "
                    f"for (B={bsz}, D={dim})."
                )
            adapted_prefix = self.prefix_adapter(
                prefix_tokens.to(dtype=backbone_tokens.dtype, device=backbone_tokens.device)
            )
            if adapted_prefix.shape[1] == self.num_prefix_tokens:
                z_prefix = base_prefix + adapted_prefix
            else:
                z_prefix = base_prefix + adapted_prefix.mean(dim=1, keepdim=True)
        else:
            z_prefix = base_prefix

        # 2. Construct client-specific suffix tokens c_client of shape (B, P_suf, D)
        c_client = self.client_suffix_prompt.expand(bsz, self.num_suffix_tokens, dim).to(
            dtype=backbone_tokens.dtype, device=backbone_tokens.device
        )

        # 3. Concatenate sequence r' = [Z_prefix || r_backbone || c_client]
        prompted_seq = torch.cat([z_prefix, backbone_tokens, c_client], dim=1)

        # 4. Condition spatial tokens on joint (prefix + suffix) context while preserving length L
        prompt_summary = 0.5 * (
            z_prefix.mean(dim=1, keepdim=True) + c_client.mean(dim=1, keepdim=True)
        )
        prompt_expanded = prompt_summary.expand(bsz, seq_len, dim)
        delta = self.prompt_gate(torch.cat([backbone_tokens, prompt_expanded], dim=-1))
        spatial_tokens = self.spatial_norm(backbone_tokens + delta)

        # 5. Construct global pooled token (B, D) from [mean(Z_prefix), mean(spatial), mean(c_client)]
        pooled_concat = torch.cat(
            [
                z_prefix.mean(dim=1),
                spatial_tokens.mean(dim=1),
                c_client.mean(dim=1),
            ],
            dim=-1,
        )
        pooled_token = self.pool_proj(pooled_concat)

        return PSPOutput(
            prompted_sequence=prompted_seq,
            spatial_tokens=spatial_tokens,
            pooled_token=pooled_token,
            prefix_tokens=z_prefix,
            suffix_tokens=c_client,
        )
