"""Logging, deterministic seed management, run-config capture, and health check utilities.

Ensures that private keys and patient/subject identifiers are never written to
application logs or captured run metadata.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.config import Settings, settings

# Patterns that must never appear in logs (64-char hex keys, FeTS subject IDs, BreaKHis slide IDs)
PRIVATE_KEY_PATTERN = re.compile(r"\b(?:0x)?[a-fA-F0-9]{64}\b")
FETS_SUBJECT_PATTERN = re.compile(r"\bFeTS2022_\d{4,6}\b")
BREAKHIS_SLIDE_PATTERN = re.compile(r"\b(?:SOB|CNB)_[BM]_[A-Z]+-\d{2}-\d+[A-Z]*-\d+-\d+\b")


def sanitize_log_text(text: str) -> str:
    """Redact private keys and patient/subject identifiers from a log string."""
    text = PRIVATE_KEY_PATTERN.sub("[REDACTED_KEY]", text)
    text = FETS_SUBJECT_PATTERN.sub("[REDACTED_SUBJECT_ID]", text)
    text = BREAKHIS_SLIDE_PATTERN.sub("[REDACTED_SLIDE_ID]", text)
    return text


class PrivacyRedactingFilter(logging.Filter):
    """Logging filter that scrubs private keys and patient/subject IDs from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_log_text(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    sanitize_log_text(a) if isinstance(a, str) else a for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: (sanitize_log_text(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
        return True


def setup_logger(
    name: str = "omnifm",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a privacy-filtered logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(f, PrivacyRedactingFilter) for f in logger.filters):
        logger.addFilter(PrivacyRedactingFilter())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(PrivacyRedactingFilter())
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        existing_files = [
            getattr(h, "baseFilename", None)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        if resolved not in existing_files:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.addFilter(PrivacyRedactingFilter())
            logger.addHandler(file_handler)

    return logger


def set_global_seed(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def get_gpu_status() -> dict[str, Any]:
    """Return current CUDA and GPU memory status."""
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return {
            "cuda_available": False,
            "device_count": 0,
            "gpu_name": None,
            "total_vram_mb": 0.0,
            "allocated_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "max_allocated_vram_mb": 0.0,
        }

    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "total_vram_mb": round(props.total_memory / (1024**2), 2),
        "allocated_vram_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 4),
        "reserved_vram_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 4),
        "max_allocated_vram_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 4),
    }


def capture_run_metadata(
    cfg: Settings | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture effective configuration, seed, software versions, and GPU status safely."""
    active_cfg = cfg or settings
    effective_seed = seed if seed is not None else active_cfg.random_seed
    set_global_seed(effective_seed)

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": effective_seed,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": get_gpu_status(),
        },
        "config": active_cfg.safe_dict(),
    }
    if extra:
        sanitized_extra = json.loads(sanitize_log_text(json.dumps(extra)))
        metadata["extra"] = sanitized_extra
    return metadata


def save_run_metadata(
    output_path: str | Path,
    cfg: Settings | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture and persist sanitized run metadata to a JSON file."""
    metadata = capture_run_metadata(cfg=cfg, seed=seed, extra=extra)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def get_system_health(cfg: Settings | None = None) -> dict[str, Any]:
    """Perform a basic health check across runtime, GPU, storage paths, and config."""
    active_cfg = cfg or settings
    gpu_info = get_gpu_status()

    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "gpu": gpu_info,
        "storage": {
            "artifact_dir_exists": active_cfg.artifact_dir.exists(),
            "dataset_dir_exists": active_cfg.dataset_dir.exists(),
        },
        "database": {
            "configured": bool(active_cfg.database_url),
            "backend": active_cfg.database_url.split("://")[0],
        },
        "blockchain": {
            "audit_enabled": active_cfg.blockchain_audit_enabled,
            "ganache_url": active_cfg.ganache_url,
            "chain_id": active_cfg.ganache_chain_id,
            "contract_configured": bool(active_cfg.contract_address),
        },
    }
