"""Base Task Adapter and Spatial-to-Token Architecture Contracts (`backend/tasks/base.py`, Task 2.6).

Every task adapter in OmniFM explicitly defines:
1. Input and target tensor formats and shapes.
2. Preprocessing and normalization rules.
3. Backbone feature representation and exact OmniFM insertion point `(B, C_feat, H_f, W_f)`.
4. How spatial CNN feature maps are tokenized into `(B, L, D)` and projected back
   to `(B, C_feat, H_f, W_f)` so skip connections and spatial decoders remain aligned.
5. Task head / decoder structure (and whether heads are dataset/client-specific).
6. Task loss combined with SPAlign (`L_total = L_task + lambda_align * L_align`).
7. Parameter-sharing policy (`shared_parameters` vs. `personalized_parameters`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from backend.omnifm.composer import OmniFMConditionedOutput
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.omnifm.psp import PrefixSuffixSpectralPrompting
from backend.omnifm.spalign import SPAlignLoss, SPAlignLossBreakdown


@dataclass(frozen=True)
class TaskAdapterSpec:
    """Explicit architectural and spatial-tokenization contract for a task adapter."""

    task_type: str
    input_shape_description: str
    target_shape_description: str
    preprocessing_description: str
    backbone_name: str
    omnifm_insertion_stage: str
    feature_map_shape: tuple[int, int, int]  # (C_feat, H_f, W_f) for default resolution
    token_sequence_shape: tuple[int, int]  # (L, D)
    skip_connection_policy: str
    head_or_decoder_description: str
    task_loss_name: str
    evaluation_metrics: tuple[str, ...]
    postprocessing_description: str
    share_task_head: bool


@dataclass
class TaskForwardOutput:
    """Standardized forward-pass output across all four task adapters."""

    prediction: torch.Tensor
    conditioned: OmniFMConditionedOutput
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTaskAdapter(nn.Module, ABC):
    """Abstract base class for task-specific models integrated with OmniFM."""

    def __init__(
        self,
        embedding_dim: int = 128,
        lambda_align: float = 0.1,
        enable_spalign: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.spalign = SPAlignLoss(lambda_align=lambda_align, enabled=enable_spalign)

    @property
    @abstractmethod
    def spec(self) -> TaskAdapterSpec:
        """Return the documented `TaskAdapterSpec` for this task adapter."""

    @abstractmethod
    def forward_features_and_predict(
        self,
        inputs: torch.Tensor,
        knowledge_bank: SpectralKnowledgeBank | None = None,
        **kwargs: Any,
    ) -> TaskForwardOutput:
        """Run backbone, OmniFM spatial conditioning, and task head/decoder."""

    @abstractmethod
    def compute_task_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the task-specific supervised or self-supervised objective `L_task`."""

    def compute_total_loss(
        self,
        forward_out: TaskForwardOutput,
        target: torch.Tensor,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SPAlignLossBreakdown:
        """Compute `L_task`, `L_align`, and `L_total = L_task + lambda_align * L_align`."""
        t_loss = self.compute_task_loss(
            prediction=forward_out.prediction,
            target=target,
            inputs=inputs,
            **kwargs,
        )
        return self.spalign(
            task_loss=t_loss,
            local_spectral_tokens=forward_out.conditioned.local_spectral_token,
            retrieved_prototypes=forward_out.conditioned.retrieved_prototypes,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        knowledge_bank: SpectralKnowledgeBank | None = None,
        **kwargs: Any,
    ) -> TaskForwardOutput:
        return self.forward_features_and_predict(
            inputs=inputs,
            knowledge_bank=knowledge_bank,
            **kwargs,
        )

    def get_personalized_parameter_names(self) -> set[str]:
        """Return parameter names that must stay local on the client (never aggregated in FedAvg).

        Always excludes:
        - `client_suffix_prompt` from `PrefixSuffixSpectralPrompting`.
        - Any task-specific incompatible heads (e.g. client-specific classification heads).
        """
        personalized: set[str] = set()
        for name, _param in self.named_parameters():
            if PrefixSuffixSpectralPrompting.is_personalized_parameter(name):
                personalized.add(name)
            elif not self.spec.share_task_head and name.startswith("heads."):
                personalized.add(name)
            elif not self.spec.share_task_head and name.startswith("classifier_head."):
                personalized.add(name)
        return personalized

    def get_shared_parameter_names(self) -> set[str]:
        """Return parameter names eligible for FedAvg aggregation across clients."""
        personalized = self.get_personalized_parameter_names()
        return {name for name, _ in self.named_parameters() if name not in personalized}

    def extract_shared_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a state_dict containing only compatible shared parameters."""
        shared_names = self.get_shared_parameter_names()
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k in shared_names or (not PrefixSuffixSpectralPrompting.is_personalized_parameter(k) and (self.spec.share_task_head or not (k.startswith("heads.") or k.startswith("classifier_head."))))
        }
