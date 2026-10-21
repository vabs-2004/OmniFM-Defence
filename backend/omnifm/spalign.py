"""Spectral-Proximal Alignment (`backend/omnifm/spalign.py`, Task 2.5).

Computes the spectral alignment regularization loss between local spectral
tokens `s` `(B, D)` and the mean of retrieved global spectral prototypes `S_g`
`(B, K, D)`:
    L_align = mean(||s - mean(S_g)||^2)
    L_total = L_task + lambda_align * L_align

When no retrieved global prototypes exist (`S_g is None` or `K == 0`), `L_align`
is explicitly defined as a zero scalar tensor (`has_valid_prototypes=False`)
and `L_total = L_task`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class SPAlignLossBreakdown:
    """Structured loss output separating task loss, alignment loss, and total loss."""

    total_loss: torch.Tensor
    task_loss: torch.Tensor
    align_loss: torch.Tensor
    lambda_align: float
    has_valid_prototypes: bool

    def to_log_dict(self) -> dict[str, Any]:
        """Return detached scalar metrics for logging/persistence."""
        return {
            "total_loss": float(self.total_loss.detach().cpu().item()),
            "task_loss": float(self.task_loss.detach().cpu().item()),
            "align_loss": float(self.align_loss.detach().cpu().item()),
            "lambda_align": float(self.lambda_align),
            "has_valid_prototypes": bool(self.has_valid_prototypes),
        }


class SPAlignLoss(nn.Module):
    """Spectral-Proximal Alignment (SPAlign) regularizer and task-loss combiner."""

    def __init__(self, lambda_align: float = 0.1, enabled: bool = True) -> None:
        super().__init__()
        if lambda_align < 0.0:
            raise ValueError(f"lambda_align must be >= 0.0, got {lambda_align}.")
        self.lambda_align = float(lambda_align)
        self.enabled = bool(enabled)

    def compute_alignment_loss(
        self,
        local_spectral_tokens: torch.Tensor | None,
        retrieved_prototypes: torch.Tensor | None,
        reference_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """Compute `L_align = mean(||s - mean(S_g)||^2)` over batch `B`.

        Returns `(align_loss, has_valid_prototypes)`.
        """
        ref = local_spectral_tokens if local_spectral_tokens is not None else reference_tensor
        device = ref.device if ref is not None else torch.device("cpu")

        if (
            not self.enabled
            or self.lambda_align == 0.0
            or local_spectral_tokens is None
            or retrieved_prototypes is None
            or retrieved_prototypes.ndim != 3
            or retrieved_prototypes.shape[1] == 0
        ):
            return torch.zeros((), device=device, dtype=torch.float32), False

        if local_spectral_tokens.ndim != 2:
            raise ValueError(
                f"Expected local_spectral_tokens of shape (B, D), got {tuple(local_spectral_tokens.shape)}."
            )
        bsz, dim = local_spectral_tokens.shape
        if retrieved_prototypes.shape[0] != bsz or retrieved_prototypes.shape[2] != dim:
            raise ValueError(
                f"Shape mismatch between local_spectral_tokens {tuple(local_spectral_tokens.shape)} "
                f"and retrieved_prototypes {tuple(retrieved_prototypes.shape)}."
            )

        # Compute in float32 for numerical stability under AMP
        s_local = local_spectral_tokens.float()
        sg_mean = retrieved_prototypes.detach().float().mean(dim=1)  # (B, D)
        sq_l2 = (s_local - sg_mean).square().sum(dim=-1)  # (B,)
        align_loss = sq_l2.mean()
        return align_loss, True

    def forward(
        self,
        task_loss: torch.Tensor,
        local_spectral_tokens: torch.Tensor | None = None,
        retrieved_prototypes: torch.Tensor | None = None,
    ) -> SPAlignLossBreakdown:
        """Combine `task_loss` and `lambda_align * align_loss` into a structured breakdown."""
        align_loss, has_valid = self.compute_alignment_loss(
            local_spectral_tokens=local_spectral_tokens,
            retrieved_prototypes=retrieved_prototypes,
            reference_tensor=task_loss,
        )
        total_loss = task_loss.float() + (self.lambda_align * align_loss if has_valid else 0.0)
        return SPAlignLossBreakdown(
            total_loss=total_loss,
            task_loss=task_loss.float(),
            align_loss=align_loss,
            lambda_align=self.lambda_align,
            has_valid_prototypes=has_valid,
        )
