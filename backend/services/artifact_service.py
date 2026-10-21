"""Artifact, Checkpoint Resolution, and Image I/O Service (`backend/services/artifact_service.py`, Tasks 6.4 & 6.5)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy import desc, select
import torch

from backend.config import settings
from backend.database.db import session_scope
from backend.database.models import ArtifactMetadata, GlobalModelVersion
from backend.tasks import BaseTaskAdapter, create_task_adapter
from backend.utils.hashing import hash_state_dict


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a requested model checkpoint or artifact does not exist."""


def resolve_and_load_adapter(
    task_type: str,
    experiment_id: str | None = None,
    round_num: int | None = None,
    client_id: str = "client_1",
    scale_factor: int = 4,
    allow_untrained_fallback: bool = True,
    database_url: str | None = None,
) -> tuple[BaseTaskAdapter, dict[str, Any]]:
    """Resolve and load a task adapter with a verified checkpoint if available.

    If `experiment_id` or `round_num` is explicitly provided and not found on disk/DB,
    raises `CheckpointNotFoundError`.
    If no checkpoint exists yet and `allow_untrained_fallback=True`, returns a
    deterministically initialized Round-0 adapter explicitly marked `is_untrained_baseline=True`.
    """
    task = task_type.lower().strip()
    adapter_kwargs: dict[str, Any] = {}
    if task == "superresolution":
        adapter_kwargs["default_scale"] = int(scale_factor)

    with session_scope(database_url) as session:
        stmt = select(GlobalModelVersion).where(GlobalModelVersion.task_type == task)
        if experiment_id is not None:
            stmt = stmt.where(GlobalModelVersion.experiment_id == experiment_id)
        if round_num is not None:
            stmt = stmt.where(GlobalModelVersion.round_num == int(round_num))
        stmt = stmt.order_by(desc(GlobalModelVersion.round_num), desc(GlobalModelVersion.id))
        record = session.execute(stmt).scalars().first()

        record_info = None
        if record is not None:
            record_info = {
                "experiment_id": record.experiment_id,
                "task_type": record.task_type,
                "algorithm": record.algorithm,
                "round_num": record.round_num,
                "model_hash": record.model_hash,
                "checkpoint_path": record.checkpoint_path,
                "tx_hash": record.tx_hash,
            }

    if record_info is None:
        if experiment_id is not None or round_num is not None or not allow_untrained_fallback:
            raise CheckpointNotFoundError(
                f"No checkpoint found for task_type='{task}', "
                f"experiment_id='{experiment_id}', round_num='{round_num}'."
            )
        adapter = create_task_adapter(task, **adapter_kwargs)
        adapter.eval()
        shared_hash = hash_state_dict(adapter.state_dict())
        return adapter, {
            "model_version": f"{task}:round_0_untrained",
            "experiment_id": None,
            "round_num": 0,
            "model_hash": shared_hash,
            "checkpoint_path": None,
            "is_untrained_baseline": True,
            "client_id": client_id,
        }

    ckpt_path = Path(record_info["checkpoint_path"])
    if not ckpt_path.is_file():
        raise CheckpointNotFoundError(
            f"Checkpoint metadata exists for ({record_info['experiment_id']}, round={record_info['round_num']}), "
            f"but file is missing at '{ckpt_path}'."
        )

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    shared_state: dict[str, torch.Tensor] = payload.get("shared_state_dict", {})
    recomputed = hash_state_dict(shared_state)
    expected_hash = payload.get("model_hash") or record_info["model_hash"]
    if expected_hash and recomputed != expected_hash:
        raise ValueError(
            f"Checkpoint hash integrity check failed for '{ckpt_path}': "
            f"expected {expected_hash}, recomputed {recomputed}."
        )

    ckpt_adapter_kwargs = dict(payload.get("adapter_kwargs") or {})
    ckpt_adapter_kwargs.update(adapter_kwargs)
    adapter = create_task_adapter(task, **ckpt_adapter_kwargs)
    current = adapter.state_dict()
    for k, v in shared_state.items():
        if k in current and current[k].shape == v.shape:
            current[k] = v.detach().cpu().clone()

    # Load client-specific personalized parameters (e.g. classification head / suffix prompt) if saved
    exp_dir = Path(settings.artifact_dir) / "experiments" / record_info["experiment_id"]
    p_path = (
        exp_dir
        / "clients"
        / client_id
        / f"{record_info['experiment_id']}_r{record_info['round_num']}_personalized.pt"
    )
    personalized_loaded = False
    if p_path.is_file():
        p_state = torch.load(p_path, map_location="cpu", weights_only=False)
        for k, v in p_state.items():
            if k in current and current[k].shape == v.shape:
                current[k] = v.detach().cpu().clone()
        personalized_loaded = True

    adapter.load_state_dict(current, strict=True)
    adapter.eval()

    return adapter, {
        "model_version": f"{record_info['experiment_id']}:r{record_info['round_num']}",
        "experiment_id": record_info["experiment_id"],
        "round_num": record_info["round_num"],
        "algorithm": record_info["algorithm"],
        "model_hash": recomputed,
        "checkpoint_path": str(ckpt_path),
        "tx_hash": record_info["tx_hash"],
        "is_untrained_baseline": False,
        "personalized_state_loaded": personalized_loaded,
        "client_id": client_id,
    }


def decode_image_bytes_to_tensor(
    raw_bytes: bytes,
    channels: int = 3,
    target_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode uploaded image bytes (PNG/JPEG/TIFF/NPY) into a `[1, C, H, W]` float32 tensor in `[0, 1]`."""
    if not raw_bytes:
        raise ValueError("Uploaded image file is empty.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            orig_size = img.size  # (W, H)
            orig_mode = img.mode
            if channels == 1:
                converted = img.convert("L")
            else:
                converted = img.convert("RGB")
            if target_size is not None:
                converted = converted.resize(
                    (target_size[1], target_size[0]), resample=Image.Resampling.BILINEAR
                )
            arr = np.asarray(converted, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Invalid or unsupported image payload: {exc}") from exc

    if channels == 1:
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)  # [1, 3, H, W]

    meta = {
        "original_width": orig_size[0],
        "original_height": orig_size[1],
        "original_mode": orig_mode,
        "processed_shape": list(tensor.shape),
        "intensity_range": [round(float(tensor.min().item()), 4), round(float(tensor.max().item()), 4)],
        "normalization": "min_max_0_1",
    }
    return tensor.contiguous(), meta


def encode_tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """Encode a `[1, C, H, W]` or `[C, H, W]` or `[H, W]` tensor in `[0, 1]` as a `data:image/png;base64,...` URI."""
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]
    t = torch.clamp(t, 0.0, 1.0)
    if t.ndim == 3:
        if t.shape[0] == 1:
            arr = (t[0].numpy() * 255.0).round().astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="L")
        else:
            arr = (t[:3].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="RGB")
    elif t.ndim == 2:
        arr = (t.numpy() * 255.0).round().astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="L")
    else:
        raise ValueError(f"Cannot encode tensor with shape {tuple(t.shape)} to PNG.")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def list_artifacts(
    experiment_id: str | None = None,
    task_type: str | None = None,
    round_num: int | None = None,
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Return persisted artifact metadata records from SQLite."""
    with session_scope(database_url) as session:
        stmt = select(ArtifactMetadata)
        if experiment_id:
            stmt = stmt.where(ArtifactMetadata.experiment_id == experiment_id)
        if task_type:
            stmt = stmt.where(ArtifactMetadata.task_type == task_type.lower().strip())
        if round_num is not None:
            stmt = stmt.where(ArtifactMetadata.round_num == int(round_num))
        stmt = stmt.order_by(desc(ArtifactMetadata.id))
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "artifact_id": r.artifact_id,
                "experiment_id": r.experiment_id,
                "task_type": r.task_type,
                "round_num": r.round_num,
                "client_id": r.client_id,
                "artifact_type": r.artifact_type,
                "file_path": r.file_path,
                "sha256_hash": r.sha256_hash,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
