"""Task Adapters and Factory (`backend/tasks`)."""

from __future__ import annotations

from typing import Any

from backend.tasks.base import BaseTaskAdapter, TaskAdapterSpec, TaskForwardOutput
from backend.tasks.classification import (
    CLIENT_TO_MEDMNIST_DATASET,
    MEDMNIST_DATASET_NUM_CLASSES,
    ClassificationTaskAdapter,
)
from backend.tasks.fusion import CLIENT_TO_FUSION_MODALITY_PAIR, FusionTaskAdapter
from backend.tasks.segmentation import SegmentationTaskAdapter
from backend.tasks.superresolution import SuperResolutionTaskAdapter

SUPPORTED_TASKS = ("classification", "segmentation", "superresolution", "fusion")


def create_task_adapter(task_type: str, **kwargs: Any) -> BaseTaskAdapter:
    """Instantiate the appropriate `BaseTaskAdapter` for `task_type`."""
    key = task_type.lower().strip()
    if key == "classification":
        return ClassificationTaskAdapter(**kwargs)
    if key == "segmentation":
        return SegmentationTaskAdapter(**kwargs)
    if key in ("superresolution", "sr"):
        return SuperResolutionTaskAdapter(**kwargs)
    if key == "fusion":
        return FusionTaskAdapter(**kwargs)
    raise ValueError(
        f"Unsupported task_type '{task_type}'. Must be one of {SUPPORTED_TASKS}."
    )


__all__ = [
    "BaseTaskAdapter",
    "CLIENT_TO_FUSION_MODALITY_PAIR",
    "CLIENT_TO_MEDMNIST_DATASET",
    "ClassificationTaskAdapter",
    "FusionTaskAdapter",
    "MEDMNIST_DATASET_NUM_CLASSES",
    " SUPPORTED_TASKS",
    "SegmentationTaskAdapter",
    "SuperResolutionTaskAdapter",
    "TaskAdapterSpec",
    "TaskForwardOutput",
    "create_task_adapter",
]
