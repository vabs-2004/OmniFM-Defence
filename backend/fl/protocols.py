"""Federated Learning Protocols, Parameter-Sharing Policy Manifest, and Round State Machine (`backend/fl/protocols.py`, Task 5.1).

Defines:
1. `RoundState`: Explicit lifecycle states (`CREATED`, `WAITING_FOR_CLIENTS`,
   `TRAINING`, `UPDATES_VALIDATED`, `AGGREGATING`, `COMPLETED`, `FAILED`, `CANCELLED`).
2. `ParameterSharingPolicy`: Explicit manifest classifying every parameter in a
   task adapter into `shared`, `personalized`, or `task_head_excluded`.
3. `ClientUpdatePayload` and `RoundExecutionResult`: Typed data structures
   exchanged between logical hospital clients and the central FL server.

Transmitted Metadata & Privacy Notice:
--------------------------------------
Clients transmit only:
- Eligible shared model parameters (`dict[str, torch.Tensor]`)
- Deterministic SHA-256 hash of the serialized shared state (`update_hash`)
- Eligible training sample count (`num_samples`)
- Batch-averaged L2-normalized spectral prototype `(1, D)` (when OmniFM GSKR is active)
- Aggregate loss/evaluation metrics (`task_loss`, `align_loss`, `total_loss`, task metrics).
Raw images and patient/subject identifiers never leave the client. Note that
model updates and spectral prototypes are still treated as potentially sensitive
research artifacts rather than formally differentially private data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch

from backend.omnifm.psp import PrefixSuffixSpectralPrompting
from backend.tasks.base import BaseTaskAdapter

AlgorithmType = Literal["omnifm", "fedavg", "fedprox", "fedper"]


class RoundState(str, Enum):
    """Explicit round and job lifecycle states per PRD Section 7.5."""

    CREATED = "CREATED"
    WAITING_FOR_CLIENTS = "WAITING_FOR_CLIENTS"
    TRAINING = "TRAINING"
    UPDATES_VALIDATED = "UPDATES_VALIDATED"
    AGGREGATING = "AGGREGATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ParameterSharingPolicy:
    """Manifest declaring which model parameters are shared vs. client-local."""

    task_type: str
    algorithm: AlgorithmType
    shared_param_names: frozenset[str]
    personalized_param_names: frozenset[str]
    incompatible_head_param_names: frozenset[str]
    parameter_shapes: dict[str, tuple[int, ...]]

    @property
    def excluded_from_aggregation(self) -> frozenset[str]:
        return self.personalized_param_names | self.incompatible_head_param_names

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "algorithm": self.algorithm,
            "shared_count": len(self.shared_param_names),
            "personalized_count": len(self.personalized_param_names),
            "incompatible_head_count": len(self.incompatible_head_param_names),
            "shared_param_names": sorted(self.shared_param_names),
            "personalized_param_names": sorted(self.personalized_param_names),
            "incompatible_head_param_names": sorted(self.incompatible_head_param_names),
        }


def build_parameter_sharing_policy(
    adapter: BaseTaskAdapter,
    algorithm: AlgorithmType = "omnifm",
) -> ParameterSharingPolicy:
    """Build the explicit parameter-sharing policy manifest for a task adapter and FL algorithm.

    Rules:
    1. `client_suffix_prompt` (PSP) is ALWAYS personalized across all tasks/algorithms.
    2. For `classification`, dataset-specific `heads.*` have incompatible output dimensions
       (`7`, `9`, `8`) and distinct label semantics; they are ALWAYS excluded from FedAvg
       in `incompatible_head_param_names`.
    3. When `algorithm == "fedper"`, the task head/decoder layers (`heads.*`, `unet.out_head.*`,
       `unet.dec1.*`, `sr_net.tail.*`, `fusion_net.decoder.*`) are kept personalized on each client
       while the feature-extraction backbone is shared.
    """
    task_type = adapter.spec.task_type
    shared: set[str] = set()
    personalized: set[str] = set()
    incompatible_heads: set[str] = set()
    shapes: dict[str, tuple[int, ...]] = {}

    fedper_personalized_prefixes = (
        "heads.",
        "classifier_head.",
        "unet.out_head.",
        "unet.dec1.",
        "sr_net.tail.",
        "fusion_net.decoder.",
    )

    for name, param in adapter.named_parameters():
        shapes[name] = tuple(int(d) for d in param.shape)
        if PrefixSuffixSpectralPrompting.is_personalized_parameter(name):
            personalized.add(name)
        elif not adapter.spec.share_task_head and (
            name.startswith("heads.") or name.startswith("classifier_head.")
        ):
            incompatible_heads.add(name)
        elif algorithm == "fedper" and name.startswith(fedper_personalized_prefixes):
            personalized.add(name)
        else:
            shared.add(name)

    return ParameterSharingPolicy(
        task_type=task_type,
        algorithm=algorithm,
        shared_param_names=frozenset(shared),
        personalized_param_names=frozenset(personalized),
        incompatible_head_param_names=frozenset(incompatible_heads),
        parameter_shapes=shapes,
    )


@dataclass
class ClientUpdatePayload:
    """Validated update package returned by a logical hospital client after local training."""

    experiment_id: str
    round_num: int
    client_id: str
    task_type: str
    algorithm: AlgorithmType
    num_samples: int
    shared_state_dict: dict[str, torch.Tensor]
    update_hash: str
    spectral_prototype: torch.Tensor | None
    train_loss: float
    task_loss: float
    align_loss: float
    eval_metrics: dict[str, Any]
    duration_sec: float
    peak_vram_mb: float
    blockchain_tx_hash: str | None = None
    blockchain_status: str = "NOT_CONFIGURED"


@dataclass
class RoundExecutionResult:
    """Summary of a completed federated round."""

    experiment_id: str
    round_num: int
    task_type: str
    algorithm: AlgorithmType
    state: RoundState
    client_updates: list[ClientUpdatePayload]
    global_model_hash: str
    global_checkpoint_path: str
    global_metrics: dict[str, Any]
    knowledge_bank_size: int
    peak_vram_mb: float
    aggregation_tx_hash: str | None = None
    global_model_tx_hash: str | None = None
    audit_status: str = "NOT_CONFIGURED"
    extra: dict[str, Any] = field(default_factory=dict)
