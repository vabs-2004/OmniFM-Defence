"""SQLite Database Engine, Session Management, and Persistence Helpers (`backend/database/db.py`, Tasks 5.5 & 6.1)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from backend.config import settings
from backend.database.models import (
    ArtifactMetadata,
    Base,
    BlockchainTransaction,
    ClientUpdate,
    Experiment,
    GlobalModelVersion,
    HospitalClient,
    MetricRecord,
    TrainingRound,
)

_ENGINE_CACHE: dict[str, Engine] = {}
_SESSION_FACTORY_CACHE: dict[str, sessionmaker[Session]] = {}


def get_engine(database_url: str | None = None) -> Engine:
    """Create or return cached SQLAlchemy engine for `database_url`."""
    url = database_url or settings.database_url
    if url not in _ENGINE_CACHE:
        is_sqlite = url.startswith("sqlite")
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        pool_kwargs = {"poolclass": NullPool} if is_sqlite else {}
        engine = create_engine(url, connect_args=connect_args, future=True, **pool_kwargs)
        Base.metadata.create_all(bind=engine)
        _ENGINE_CACHE[url] = engine
        _SESSION_FACTORY_CACHE[url] = sessionmaker(
            bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
        )
    return _ENGINE_CACHE[url]


def dispose_engines(database_url: str | None = None) -> None:
    """Dispose cached SQLAlchemy engine(s) and release underlying connections."""
    if database_url is not None:
        engine = _ENGINE_CACHE.pop(database_url, None)
        _SESSION_FACTORY_CACHE.pop(database_url, None)
        if engine is not None:
            engine.dispose()
        return
    for url, engine in list(_ENGINE_CACHE.items()):
        engine.dispose()
    _ENGINE_CACHE.clear()
    _SESSION_FACTORY_CACHE.clear()


def init_db(database_url: str | None = None) -> Engine:
    """Initialize all SQLAlchemy database tables."""
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return engine


@contextmanager
def session_scope(database_url: str | None = None) -> Generator[Session, None, None]:
    """Provide a transactional SQLAlchemy session scope."""
    url = database_url or settings.database_url
    get_engine(url)
    factory = _SESSION_FACTORY_CACHE[url]
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upsert_hospital_client(
    client_id: str,
    hospital_name: str,
    modality: str = "multi-modality",
    wallet_address: str = "",
    registered_on_chain: bool = False,
    registration_tx_hash: str | None = None,
    status: str = "READY",
    database_url: str | None = None,
) -> dict[str, Any]:
    """Create or update a registered logical hospital client in SQLite."""
    with session_scope(database_url) as session:
        stmt = select(HospitalClient).where(HospitalClient.client_id == client_id)
        existing = session.execute(stmt).scalar_one_or_none()
        if existing is None:
            record = HospitalClient(
                client_id=client_id,
                hospital_name=hospital_name,
                modality=modality,
                wallet_address=wallet_address,
                registered_on_chain=registered_on_chain,
                registration_tx_hash=registration_tx_hash,
                status=status,
            )
            session.add(record)
            session.flush()
        else:
            existing.hospital_name = hospital_name
            existing.modality = modality
            if wallet_address:
                existing.wallet_address = wallet_address
            if registered_on_chain:
                existing.registered_on_chain = True
            if registration_tx_hash:
                existing.registration_tx_hash = registration_tx_hash
            existing.status = status
            record = existing
            session.flush()

        return {
            "id": record.id,
            "client_id": record.client_id,
            "hospital_name": record.hospital_name,
            "modality": record.modality,
            "wallet_address": record.wallet_address,
            "registered_on_chain": record.registered_on_chain,
            "registration_tx_hash": record.registration_tx_hash,
            "status": record.status,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }


def create_or_update_experiment(
    experiment_id: str,
    task_type: str,
    algorithm: str = "omnifm",
    status: str = "CREATED",
    current_round: int = 0,
    total_rounds: int = 20,
    seed: int = 42,
    config_json: dict[str, Any] | None = None,
    log_line: str | None = None,
    error_message: str | None = None,
    peak_vram_mb: float | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Create or update an `Experiment` record in SQLite."""
    with session_scope(database_url) as session:
        stmt = select(Experiment).where(Experiment.experiment_id == experiment_id)
        exp = session.execute(stmt).scalar_one_or_none()
        if exp is None:
            logs = [log_line] if log_line else []
            exp = Experiment(
                experiment_id=experiment_id,
                task_type=task_type,
                algorithm=algorithm,
                status=status,
                current_round=current_round,
                total_rounds=total_rounds,
                seed=seed,
                config_json=config_json or {},
                logs_json=logs,
                error_message=error_message,
                peak_vram_mb=float(peak_vram_mb or 0.0),
            )
            session.add(exp)
        else:
            exp.status = status
            exp.current_round = int(current_round)
            exp.total_rounds = int(total_rounds)
            if config_json is not None:
                exp.config_json = config_json
            if log_line:
                current_logs = list(exp.logs_json or [])
                current_logs.append(log_line)
                exp.logs_json = current_logs[-500:]
            if error_message is not None:
                exp.error_message = error_message
            if peak_vram_mb is not None:
                exp.peak_vram_mb = max(float(exp.peak_vram_mb), float(peak_vram_mb))
            exp.updated_at = datetime.now(timezone.utc)
        session.flush()
        return {
            "experiment_id": exp.experiment_id,
            "task_type": exp.task_type,
            "algorithm": exp.algorithm,
            "status": exp.status,
            "current_round": exp.current_round,
            "total_rounds": exp.total_rounds,
            "seed": exp.seed,
            "config": exp.config_json,
            "error_message": exp.error_message,
            "peak_vram_mb": exp.peak_vram_mb,
        }


def append_experiment_log(
    experiment_id: str,
    message: str,
    database_url: str | None = None,
) -> None:
    """Append a timestamped log line to `Experiment.logs_json`."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {message}"
    with session_scope(database_url) as session:
        stmt = select(Experiment).where(Experiment.experiment_id == experiment_id)
        exp = session.execute(stmt).scalar_one_or_none()
        if exp is not None:
            logs = list(exp.logs_json or [])
            logs.append(line)
            exp.logs_json = logs[-500:]
            exp.updated_at = datetime.now(timezone.utc)


def record_blockchain_tx(
    action_type: str,
    status: str,
    experiment_id: str | None = None,
    task_type: str | None = None,
    round_num: int | None = None,
    client_id: str | None = None,
    artifact_hash: str | None = None,
    tx_hash: str | None = None,
    block_number: int | None = None,
    wallet_address: str | None = None,
    error_message: str | None = None,
    database_url: str | None = None,
) -> int:
    """Persist an on-chain or failed/pending blockchain transaction record."""
    with session_scope(database_url) as session:
        rec = BlockchainTransaction(
            experiment_id=experiment_id,
            task_type=task_type,
            round_num=round_num,
            client_id=client_id,
            action_type=action_type,
            artifact_hash=artifact_hash,
            tx_hash=tx_hash,
            block_number=block_number,
            wallet_address=wallet_address,
            status=status,
            error_message=error_message,
        )
        session.add(rec)
        session.flush()
        return int(rec.id)


def record_artifact_metadata(
    artifact_id: str,
    experiment_id: str,
    task_type: str,
    round_num: int,
    artifact_type: str,
    file_path: str,
    sha256_hash: str,
    client_id: str | None = None,
    database_url: str | None = None,
) -> None:
    """Persist or update off-chain artifact metadata."""
    with session_scope(database_url) as session:
        stmt = select(ArtifactMetadata).where(ArtifactMetadata.artifact_id == artifact_id)
        existing = session.execute(stmt).scalar_one_or_none()
        if existing is None:
            session.add(
                ArtifactMetadata(
                    artifact_id=artifact_id,
                    experiment_id=experiment_id,
                    task_type=task_type,
                    round_num=round_num,
                    client_id=client_id,
                    artifact_type=artifact_type,
                    file_path=file_path,
                    sha256_hash=sha256_hash,
                )
            )
        else:
            existing.file_path = file_path
            existing.sha256_hash = sha256_hash
