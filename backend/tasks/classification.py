"""Medical Image Classification Task Adapter (`backend/tasks/classification.py`, Task 2.6 & 4.1).

Supports MedMNIST v2 datasets with distinct label spaces:
- `dermamnist` (client_1 default): 7 classes
- `pathmnist` (client_2 default): 9 classes
- `bloodmnist` (client_3 default): 8 classes

Critical Label-Space & Aggregation Rule:
----------------------------------------
Each dataset's classifier head in `self.heads` has a distinct semantic label space
and output dimension. `share_task_head=False` ensures that `self.heads` and
`client_suffix_prompt` are excluded from FedAvg aggregation; only the shared
ResNet-18 backbone and shared OmniFM conditioning layers are aggregated.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.models.classification import SmallInputResNet18Backbone
from backend.omnifm.composer import OmniFMSpatialConditioner
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.tasks.base import BaseTaskAdapter, TaskAdapterSpec, TaskForwardOutput

MEDMNIST_DATASET_NUM_CLASSES: dict[str, int] = {
    "dermamnist": 7,
    "pathmnist": 9,
    "bloodmnist": 8,
}

CLIENT_TO_MEDMNIST_DATASET: dict[str, str] = {
    "client_1": "dermamnist",
    "client_2": "pathmnist",
    "client_3": "bloodmnist",
}


class ClassificationTaskAdapter(BaseTaskAdapter):
    """ResNet-18 + OmniFM classification adapter with dataset-specific heads."""

    _SPEC = TaskAdapterSpec(
        task_type="classification",
        input_shape_description="(B, 3, 28, 28) float32 normalized RGB image",
        target_shape_description="(B,) int64 class index in dataset-specific label space",
        preprocessing_description="uint8 [0, 255] scaled to float32 [0, 1] and normalized with mean=0.5, std=0.5",
        backbone_name="SmallInputResNet18Backbone (3x3 stride-1 stem, no maxpool)",
        omnifm_insertion_stage="Post-Layer4 spatial feature map (B, 512, 4, 4)",
        feature_map_shape=(512, 4, 4),
        token_sequence_shape=(16, 128),
        skip_connection_policy="Residual stages inside ResNet-18; OmniFM conditions the (B, 512, 4, 4) bottleneck before global average pooling",
        head_or_decoder_description="Dataset-specific Linear(512 + D, num_classes) heads for dermamnist(7), pathmnist(9), bloodmnist(8)",
        task_loss_name="CrossEntropyLoss",
        evaluation_metrics=("accuracy", "macro_f1"),
        postprocessing_description="Softmax over dataset-specific logits -> probabilities, argmax class, and confidence",
        share_task_head=False,
    )

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        top_k: int = 2,
        lambda_align: float = 0.1,
        default_dataset: str = "dermamnist",
        dataset_num_classes: dict[str, int] | None = None,
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
        head_cfg = dataset_num_classes or MEDMNIST_DATASET_NUM_CLASSES
        if default_dataset not in head_cfg:
            raise ValueError(
                f"Unknown default_dataset '{default_dataset}'. Available: {list(head_cfg.keys())}."
            )
        self.default_dataset = default_dataset
        self.dataset_num_classes = dict(head_cfg)

        self.backbone = SmallInputResNet18Backbone(in_channels=3)
        self.omnifm = OmniFMSpatialConditioner(
            feature_channels=self.backbone.out_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            top_k=top_k,
            max_token_grid=4,
            enable_gskr=enable_gskr,
            enable_eca=enable_eca,
            enable_psp=enable_psp,
        )

        combined_dim = self.backbone.out_channels + self.embedding_dim
        self.heads = nn.ModuleDict(
            {
                ds_name: nn.Linear(combined_dim, int(n_cls))
                for ds_name, n_cls in self.dataset_num_classes.items()
            }
        )

    @property
    def spec(self) -> TaskAdapterSpec:
        return self._SPEC

    def resolve_dataset_key(self, dataset_name: str | None = None, client_id: str | None = None) -> str:
        """Resolve which dataset-specific classifier head to apply."""
        if dataset_name is not None:
            key = dataset_name.lower().strip()
            if key not in self.heads:
                raise ValueError(
                    f"Dataset head '{key}' not found in adapter heads {list(self.heads.keys())}."
                )
            return key
        if client_id is not None and client_id in CLIENT_TO_MEDMNIST_DATASET:
            mapped = CLIENT_TO_MEDMNIST_DATASET[client_id]
            if mapped in self.heads:
                return mapped
        return self.default_dataset

    def forward_features_and_predict(
        self,
        inputs: torch.Tensor,
        knowledge_bank: SpectralKnowledgeBank | None = None,
        dataset_name: str | None = None,
        client_id: str | None = None,
        **kwargs: Any,
    ) -> TaskForwardOutput:
        """Extract ResNet-18 features, condition with OmniFM, and run dataset-specific head."""
        ds_key = self.resolve_dataset_key(dataset_name=dataset_name, client_id=client_id)

        # 1. CNN feature map: (B, 512, 4, 4) for 28x28 input
        feat_map = self.backbone(inputs)

        # 2. OmniFM spatial conditioning: (B, 512, 4, 4) -> (B, 16, D) -> (B, 512, 4, 4)
        cond_out = self.omnifm(
            raw_image=inputs,
            feature_map=feat_map,
            knowledge_bank=knowledge_bank,
            task_type=self.spec.task_type,
        )

        # 3. Global average pool spatial features and concatenate with PSP pooled token
        pooled_cnn = F.adaptive_avg_pool2d(cond_out.spatial_features, (1, 1)).flatten(1)  # (B, 512)
        joint_repr = torch.cat(
            [pooled_cnn, cond_out.pooled_token.to(dtype=pooled_cnn.dtype)], dim=-1
        )  # (B, 512 + D)

        logits = self.heads[ds_key](joint_repr)  # (B, num_classes)
        return TaskForwardOutput(
            prediction=logits,
            conditioned=cond_out,
            extra={"dataset_name": ds_key, "num_classes": self.dataset_num_classes[ds_key]},
        )

    def compute_task_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if target.ndim > 1:
            target = target.view(-1)
        return F.cross_entropy(prediction.float(), target.long())
