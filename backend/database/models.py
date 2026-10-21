"""SQLAlchemy ORM Entities (`backend/database/models.py`, Tasks 5.5 & 6.1).

Defines the 8 core persistence tables required by PRD Section 9.3:
- `HospitalClient`
- `Experiment`
- `TrainingRound`
- `ClientUpdate`
- `GlobalModelVersion`
- `MetricRecord`
- `BlockchainTransaction`
- `ArtifactMetadata`

Large model checkpoints and images are stored on disk under `ARTIFACT_DIR`;
only file paths, SHA-256 hashes, and structured metadata are stored in SQLite.
Private keys are never stored in any table.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all OmniFM SQLite tables."""


class HospitalClient(Base):
    """Registered logical hospital client record (no private keys stored)."""

    __tablename__ = "hospital_clients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    hospital_name: Mapped[str] = mapped_column(String(128), nullable=False)
    modality: Mapped[str] = mapped_column(String(64), default="multi-modality", nullable=False)
    wallet_address: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    registered_on_chain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    registration_tx_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="READY", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class Experiment(Base):
    """Federated learning experiment job and configuration."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(32), default="omnifm", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="CREATED", index=True, nullable=False)
    current_round: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_rounds: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, default=42, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    logs_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    peak_vram_mb: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )


class TrainingRound(Base):
    """Per-round state machine record for an experiment."""

    __tablename__ = "training_rounds"
    __table_args__ = (
        UniqueConstraint("experiment_id", "round_num", name="uq_experiment_round"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("experiments.experiment_id"), index=True, nullable=False
    )
    round_num: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(32), default="CREATED", nullable=False)
    expected_clients: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    submitted_clients: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    aggregated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    global_model_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)
    global_checkpoint_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    knowledge_bank_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    peak_vram_mb: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    audit_status: Mapped[str] = mapped_column(String(32), default="NOT_CONFIGURED", nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ClientUpdate(Base):
    """Per-client update record within a communication round."""

    __tablename__ = "client_updates"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id", "round_num", "client_id", name="uq_exp_round_client_update"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    round_num: Mapped[int] = mapped_column(Integer, nullable=False)
    client_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    num_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    update_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    train_loss: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    task_loss: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    align_loss: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    eval_metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    peak_vram_mb: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    tx_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)
    blockchain_status: Mapped[str] = mapped_column(
        String(32), default="NOT_CONFIGURED", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class GlobalModelVersion(Base):
    """Versioned global checkpoint metadata per round (including round 0)."""

    __tablename__ = "global_model_versions"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id", "round_num", name="uq_exp_round_global_model"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(32), default="omnifm", nullable=False)
    round_num: Mapped[int] = mapped_column(Integer, nullable=False)
    model_hash: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(512), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    tx_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class MetricRecord(Base):
    """Per-round, per-client or global evaluation metric record."""

    __tablename__ = "metric_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(32), default="omnifm", nullable=False)
    round_num: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    scope: Mapped[str] = mapped_column(String(32), default="global", nullable=False)
    client_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class BlockchainTransaction(Base):
    """On-chain audit transaction reference and status."""

    __tablename__ = "blockchain_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    round_num: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    client_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    action_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    artifact_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)
    tx_hash: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    block_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wallet_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class ArtifactMetadata(Base):
    """Off-chain artifact file metadata linked to cryptographic SHA-256 hashes."""

    __tablename__ = "artifact_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_id: Mapped[str] = mapped_column(String(96), unique=True, index=True, nullable=False)
    experiment_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    round_num: Mapped[int] = mapped_column(Integer, nullable=False)
    client_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
