"""Experiment Lifecycle, Asynchronous Training Job Manager, and Metrics Service (`backend/services/experiment_service.py`, Tasks 6.3 & 6.5)."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import asc, desc, select
import torch
from torch.utils.data import DataLoader

from backend.blockchain.web3_connector import Web3Connector
from backend.config import Settings, settings
from backend.database.db import (
    append_experiment_log,
    create_or_update_experiment,
    init_db,
    record_blockchain_tx,
    session_scope,
    upsert_hospital_client,
)
from backend.database.models import (
    ClientUpdate,
    Experiment,
    GlobalModelVersion,
    HospitalClient,
    MetricRecord,
    TrainingRound,
)
from backend.fl.server import FederatedLearningServer
from backend.services.artifact_service import CheckpointNotFoundError
from backend.utils.logging import setup_logger
from backend.utils.preprocessing import build_client_dataloader

logger = setup_logger("omnifm.services.experiment")

VALID_TASK_TYPES = ("classification", "segmentation", "superresolution", "fusion")
VALID_ALGORITHMS = ("omnifm", "fedavg", "fedprox", "fedper")

DEFAULT_HOSPITALS: dict[str, dict[str, str]] = {
    "client_1": {
        "hospital_name": "Hospital A - Dermatology & MRI Center",
        "modality": "Dermatology / T1ce MRI / Histopathology 40x / CT-MRI",
    },
    "client_2": {
        "hospital_name": "Hospital B - Pathology & PET Center",
        "modality": "Colon Pathology / T1ce MRI / Histopathology 100x / PET-MRI",
    },
    "client_3": {
        "hospital_name": "Hospital C - Hematology & SPECT Center",
        "modality": "Hematology / T1ce MRI / Histopathology 200x-400x / SPECT-MRI",
    },
}


DEFAULT_DEV_WALLETS: dict[str, str] = {
    "client_1": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
    "client_2": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
    "client_3": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
}


def _resolve_client_wallet(client_id: str, cfg: Settings) -> str:
    """Resolve the public wallet address for `client_id` without exposing any private key."""
    key_map = {
        "client_1": cfg.client1_private_key,
        "client_2": cfg.client2_private_key,
        "client_3": cfg.client3_private_key,
    }
    secret = key_map.get(client_id)
    raw = secret.get_secret_value().strip() if secret is not None else ""
    if raw:
        try:
            from eth_account import Account

            pk = raw if raw.startswith("0x") else f"0x{raw}"
            return str(Account.from_key(pk).address)
        except Exception:
            pass
    return DEFAULT_DEV_WALLETS.get(client_id, "0x0000000000000000000000000000000000000000")


class ConcurrentTrainingJobError(RuntimeError):
    """Raised when starting or resuming a GPU training job while another job is active."""


class InvalidTrainingStateError(ValueError):
    """Raised when cancelling or resuming a job in an incompatible lifecycle state."""


def ensure_default_hospitals(
    cfg: Settings | None = None,
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Seed the 3 logical hospital clients if not yet present in SQLite."""
    active_cfg = cfg or settings
    init_db(database_url or active_cfg.database_url)
    results: list[dict[str, Any]] = []
    for cid, meta in DEFAULT_HOSPITALS.items():
        wallet = _resolve_client_wallet(cid, active_cfg)
        rec = upsert_hospital_client(
            client_id=cid,
            hospital_name=meta["hospital_name"],
            modality=meta["modality"],
            wallet_address=wallet,
            status="READY",
            database_url=database_url or active_cfg.database_url,
        )
        results.append(rec)
    return results


def register_hospital_client(
    client_id: str,
    hospital_name: str,
    modality: str,
    cfg: Settings | None = None,
    web3_connector: Web3Connector | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Register a logical hospital client and optionally record registration on-chain."""
    active_cfg = cfg or settings
    db_url = database_url or active_cfg.database_url
    cid = client_id.strip()
    if not cid or not hospital_name.strip():
        raise ValueError("Both 'client_id' and 'hospital_name' must be non-empty strings.")

    wallet = _resolve_client_wallet(cid, active_cfg)
    tx_hash: str | None = None
    registered_on_chain = False

    if (
        web3_connector is not None
        and wallet
        and web3_connector.is_connected()
        and web3_connector.is_contract_deployed()
    ):
        try:
            receipt = web3_connector.ensure_hospital_registered(
                client_id=cid,
                hospital_name=hospital_name.strip(),
                modality=modality.strip() or "multi-modality",
                wallet_address=wallet,
            )
            registered_on_chain = True
            rec_tx_hash = (
                receipt.get("tx_hash")
                if isinstance(receipt, dict)
                else getattr(receipt, "tx_hash", None)
            )
            rec_block = (
                receipt.get("block_number")
                if isinstance(receipt, dict)
                else getattr(receipt, "block_number", None)
            )
            if isinstance(rec_tx_hash, str) and rec_tx_hash:
                tx_hash = rec_tx_hash
                record_blockchain_tx(
                    action_type="HospitalRegistered",
                    status="CONFIRMED",
                    client_id=cid,
                    tx_hash=rec_tx_hash,
                    block_number=int(rec_block) if rec_block is not None else None,
                    wallet_address=wallet,
                    database_url=db_url,
                )
        except Exception as exc:
            record_blockchain_tx(
                action_type="HospitalRegistered",
                status="FAILED",
                client_id=cid,
                wallet_address=wallet,
                error_message=str(exc),
                database_url=db_url,
            )

    return upsert_hospital_client(
        client_id=cid,
        hospital_name=hospital_name.strip(),
        modality=modality.strip() or "multi-modality",
        wallet_address=wallet,
        registered_on_chain=registered_on_chain,
        registration_tx_hash=tx_hash,
        status="READY",
        database_url=db_url,
    )


def list_hospital_clients(
    cfg: Settings | None = None,
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Return all registered logical hospital clients (never exposing private keys)."""
    active_cfg = cfg or settings
    db_url = database_url or active_cfg.database_url
    with session_scope(db_url) as session:
        rows = (
            session.execute(select(HospitalClient).order_by(asc(HospitalClient.client_id)))
            .scalars()
            .all()
        )
        if not rows:
            ensure_default_hospitals(cfg=active_cfg, database_url=db_url)
            rows = (
                session.execute(select(HospitalClient).order_by(asc(HospitalClient.client_id)))
                .scalars()
                .all()
            )
        return [
            {
                "id": r.id,
                "client_id": r.client_id,
                "hospital_name": r.hospital_name,
                "modality": r.modality,
                "wallet_address": r.wallet_address,
                "registered_on_chain": r.registered_on_chain,
                "registration_tx_hash": r.registration_tx_hash,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def _build_default_dataloaders(
    task_type: str,
    client_ids: tuple[str, ...],
    batch_size: int,
    max_samples: int | None,
    scale_factor: int = 2,
    seed: int = 42,
) -> tuple[dict[str, DataLoader[Any]], dict[str, DataLoader[Any]]]:
    """Build train and eval DataLoaders for each logical hospital client."""
    train_loaders: dict[str, DataLoader[Any]] = {}
    eval_loaders: dict[str, DataLoader[Any]] = {}
    for cid in client_ids:
        tr_loader, _ = build_client_dataloader(
            task_type=task_type,
            client_id=cid,
            split="train",
            batch_size=batch_size,
            max_samples=max_samples,
            scale=scale_factor,
            seed=seed,
        )
        ev_loader, _ = build_client_dataloader(
            task_type=task_type,
            client_id=cid,
            split="val",
            batch_size=batch_size,
            max_samples=max_samples,
            scale=scale_factor,
            seed=seed,
        )
        train_loaders[cid] = tr_loader
        eval_loaders[cid] = ev_loader
    return train_loaders, eval_loaders


class TrainingJobManager:
    """Thread-safe manager for single-GPU federated training jobs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_servers: dict[str, FederatedLearningServer] = {}
        self._active_threads: dict[str, threading.Thread] = {}
        self._cancel_requested: set[str] = set()

    def start_training(
        self,
        task_type: str,
        num_rounds: int = 20,
        config: dict[str, Any] | None = None,
        training_id: str | None = None,
        cfg: Settings | None = None,
        web3_connector: Web3Connector | None = None,
        database_url: str | None = None,
        run_async: bool = True,
    ) -> dict[str, Any]:
        """Validate configuration and launch an asynchronous federated training job."""
        active_cfg = cfg or settings
        db_url = database_url or active_cfg.database_url
        init_db(db_url)
        ensure_default_hospitals(cfg=active_cfg, database_url=db_url)

        task = (task_type or "").lower().strip()
        if task not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{task_type}'. Must be one of {VALID_TASK_TYPES}."
            )
        if int(num_rounds) < 1 or int(num_rounds) > 100:
            raise ValueError(f"num_rounds must be between 1 and 100, got {num_rounds}.")

        job_cfg = dict(config or {})
        algorithm = str(job_cfg.get("algorithm", "omnifm")).lower().strip()
        if algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm '{algorithm}'. Must be one of {VALID_ALGORITHMS}."
            )

        scale_factor = int(job_cfg.get("scale_factor", 2))
        if task == "superresolution" and scale_factor not in (2, 4, 8):
            raise ValueError(f"Super-resolution scale_factor must be 2, 4, or 8; got {scale_factor}.")

        with self._lock:
            # Check for any running job in memory or DB
            for active_id, thread in list(self._active_threads.items()):
                if thread.is_alive():
                    raise ConcurrentTrainingJobError(
                        f"Another training job ('{active_id}') is currently active on the single GPU."
                    )
            with session_scope(db_url) as session:
                running_exp = (
                    session.execute(
                        select(Experiment).where(Experiment.status.in_(["RUNNING"]))
                    )
                    .scalars()
                    .first()
                )
                if running_exp is not None and running_exp.experiment_id in self._active_threads:
                    raise ConcurrentTrainingJobError(
                        f"Experiment '{running_exp.experiment_id}' is currently RUNNING."
                    )

            exp_id = training_id or f"exp_{task}_{algorithm}_{uuid.uuid4().hex[:8]}"
            seed = int(job_cfg.get("seed", active_cfg.random_seed))
            client_ids = tuple(job_cfg.get("client_ids", ["client_1", "client_2", "client_3"]))
            if not client_ids:
                raise ValueError("client_ids cannot be empty.")

            adapter_kwargs: dict[str, Any] = {}
            if task == "superresolution":
                adapter_kwargs["default_scale"] = scale_factor
            for flag in ("enable_gskr", "enable_eca", "enable_psp", "enable_spalign"):
                if flag in job_cfg:
                    adapter_kwargs[flag] = bool(job_cfg[flag])

            audit_required = bool(job_cfg.get("audit_required", False))
            server = FederatedLearningServer(
                experiment_id=exp_id,
                task_type=task,
                algorithm=algorithm,  # type: ignore[arg-type]
                client_ids=client_ids,
                cfg=active_cfg,
                database_url=db_url,
                web3_connector=web3_connector,
                audit_required=audit_required,
                adapter_kwargs=adapter_kwargs,
                seed=seed,
            )

            full_config = {
                "task_type": task,
                "algorithm": algorithm,
                "num_rounds": int(num_rounds),
                "client_ids": list(client_ids),
                "local_epochs": int(job_cfg.get("local_epochs", active_cfg.default_local_epochs)),
                "learning_rate": float(job_cfg.get("learning_rate", 1e-3)),
                "batch_size": int(job_cfg.get("batch_size", 4)),
                "max_batches": job_cfg.get("max_batches", 2),
                "max_samples": job_cfg.get("max_samples", 16),
                "scale_factor": scale_factor,
                "fedprox_mu": float(job_cfg.get("fedprox_mu", 0.01)),
                "spalign_weight": float(job_cfg.get("spalign_weight", active_cfg.spalign_lambda)),
                "device": str(
                    job_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
                ),
                "audit_required": audit_required,
                "simulate_failure": bool(job_cfg.get("simulate_failure", False)),
                "seed": seed,
                "adapter_kwargs": adapter_kwargs,
            }

            create_or_update_experiment(
                experiment_id=exp_id,
                task_type=task,
                algorithm=algorithm,
                status="CREATED",
                current_round=0,
                total_rounds=int(num_rounds),
                seed=seed,
                config_json=full_config,
                log_line=f"Job {exp_id} created for task={task}, algorithm={algorithm}, rounds={num_rounds}.",
                database_url=db_url,
            )
            self._active_servers[exp_id] = server
            self._cancel_requested.discard(exp_id)

            if run_async:
                worker = threading.Thread(
                    target=self._execute_job_worker,
                    kwargs={
                        "experiment_id": exp_id,
                        "server": server,
                        "start_round": 1,
                        "total_rounds": int(num_rounds),
                        "job_config": full_config,
                        "database_url": db_url,
                    },
                    name=f"omnifm-train-{exp_id}",
                    daemon=True,
                )
                self._active_threads[exp_id] = worker
                worker.start()
            else:
                self._execute_job_worker(
                    experiment_id=exp_id,
                    server=server,
                    start_round=1,
                    total_rounds=int(num_rounds),
                    job_config=full_config,
                    database_url=db_url,
                )

        return {
            "training_id": exp_id,
            "experiment_id": exp_id,
            "task_type": task,
            "algorithm": algorithm,
            "status": "CREATED" if run_async else self.get_training_status(exp_id, database_url=db_url)["status"],
            "total_rounds": int(num_rounds),
        }

    def _execute_job_worker(
        self,
        experiment_id: str,
        server: FederatedLearningServer,
        start_round: int,
        total_rounds: int,
        job_config: dict[str, Any],
        database_url: str,
    ) -> None:
        """Background worker executing sequential federated rounds."""
        try:
            if experiment_id in self._cancel_requested:
                create_or_update_experiment(
                    experiment_id=experiment_id,
                    task_type=server.task_type,
                    algorithm=server.algorithm,
                    status="CANCELLED",
                    current_round=max(0, start_round - 1),
                    total_rounds=total_rounds,
                    log_line=f"Experiment {experiment_id} cancelled before starting round {start_round}.",
                    database_url=database_url,
                )
                return

            if job_config.get("simulate_failure"):
                raise RuntimeError("Simulated training failure triggered via job configuration.")

            if start_round == 1 and not server.round_0_hash:
                server.initialize_round_zero(total_rounds=total_rounds)

            device = torch.device(job_config.get("device", "cpu"))
            max_samples = job_config.get("max_samples", 16)
            batch_size = int(job_config.get("batch_size", 4))
            scale_factor = int(job_config.get("scale_factor", 2))

            for r in range(start_round, total_rounds + 1):
                if experiment_id in self._cancel_requested:
                    create_or_update_experiment(
                        experiment_id=experiment_id,
                        task_type=server.task_type,
                        algorithm=server.algorithm,
                        status="CANCELLED",
                        current_round=r - 1,
                        total_rounds=total_rounds,
                        log_line=f"Experiment {experiment_id} cancelled safely before round {r}.",
                        database_url=database_url,
                    )
                    return

                server.run_round(
                    round_num=r,
                    local_epochs=int(job_config.get("local_epochs", 1)),
                    batch_size=batch_size,
                    max_train_samples=int(max_samples) if max_samples is not None else None,
                    max_val_samples=int(max_samples) if max_samples is not None else None,
                    lr=float(job_config.get("learning_rate", 1e-3)),
                    fedprox_mu=float(job_config.get("fedprox_mu", 0.01)),
                    scale=scale_factor,
                    device=device,
                )

            peak_vram = max(
                (res.peak_vram_mb for res in server.completed_rounds.values()),
                default=0.0,
            )
            create_or_update_experiment(
                experiment_id=experiment_id,
                task_type=server.task_type,
                algorithm=server.algorithm,
                status="COMPLETED",
                current_round=total_rounds,
                total_rounds=total_rounds,
                peak_vram_mb=peak_vram,
                log_line=f"Experiment {experiment_id} completed all {total_rounds} rounds.",
                database_url=database_url,
            )
        except Exception as exc:
            status = "CANCELLED" if "cancelled" in str(exc).lower() else "FAILED"
            logger.error("Job %s terminated with %s: %s", experiment_id, status, exc)
            with session_scope(database_url) as session:
                exp = session.execute(
                    select(Experiment).where(Experiment.experiment_id == experiment_id)
                ).scalar_one_or_none()
                cur_round = exp.current_round if exp else 0
            create_or_update_experiment(
                experiment_id=experiment_id,
                task_type=server.task_type,
                algorithm=server.algorithm,
                status=status,
                current_round=cur_round,
                total_rounds=total_rounds,
                error_message=str(exc) if status == "FAILED" else None,
                log_line=f"Experiment {experiment_id} ended with status={status}: {exc}",
                database_url=database_url,
            )
        finally:
            with self._lock:
                self._active_threads.pop(experiment_id, None)

    def cancel_training(
        self,
        training_id: str,
        database_url: str | None = None,
    ) -> dict[str, Any]:
        """Request cancellation at a safe client/round boundary."""
        db_url = database_url or settings.database_url
        with session_scope(db_url) as session:
            exp = session.execute(
                select(Experiment).where(Experiment.experiment_id == training_id)
            ).scalar_one_or_none()
            if exp is None:
                raise LookupError(f"Training job '{training_id}' not found.")
            current_status = exp.status
            task_type = exp.task_type
            algorithm = exp.algorithm
            cur_round = exp.current_round
            tot_rounds = exp.total_rounds

        if current_status in ("COMPLETED", "FAILED", "CANCELLED"):
            return {
                "training_id": training_id,
                "status": current_status,
                "message": f"Job '{training_id}' is already in terminal state '{current_status}'.",
            }

        with self._lock:
            self._cancel_requested.add(training_id)

        create_or_update_experiment(
            experiment_id=training_id,
            task_type=task_type,
            algorithm=algorithm,
            status="CANCELLED",
            current_round=cur_round,
            total_rounds=tot_rounds,
            log_line=f"Cancellation requested for job {training_id} at round {cur_round}.",
            database_url=db_url,
        )
        return {
            "training_id": training_id,
            "status": "CANCELLED",
            "current_round": cur_round,
            "total_rounds": tot_rounds,
            "message": "Cancellation requested at safe round/client boundary.",
        }

    def resume_training(
        self,
        training_id: str,
        cfg: Settings | None = None,
        web3_connector: Web3Connector | None = None,
        database_url: str | None = None,
        run_async: bool = True,
    ) -> dict[str, Any]:
        """Resume a cancelled/failed/interrupted experiment from its latest verified checkpoint."""
        active_cfg = cfg or settings
        db_url = database_url or active_cfg.database_url

        with session_scope(db_url) as session:
            exp = session.execute(
                select(Experiment).where(Experiment.experiment_id == training_id)
            ).scalar_one_or_none()
            if exp is None:
                raise LookupError(f"Training job '{training_id}' not found.")
            if exp.status == "COMPLETED" and exp.current_round >= exp.total_rounds:
                raise InvalidTrainingStateError(
                    f"Training job '{training_id}' is already COMPLETED ({exp.current_round}/{exp.total_rounds})."
                )
            exp_task = exp.task_type
            exp_alg = exp.algorithm
            exp_seed = exp.seed
            exp_total = exp.total_rounds
            exp_config = dict(exp.config_json or {})

            latest_gmv = (
                session.execute(
                    select(GlobalModelVersion)
                    .where(GlobalModelVersion.experiment_id == training_id)
                    .order_by(desc(GlobalModelVersion.round_num))
                )
                .scalars()
                .first()
            )
            latest_round = latest_gmv.round_num if latest_gmv is not None else None

        if latest_round is None:
            raise CheckpointNotFoundError(
                f"Cannot resume '{training_id}': no valid checkpoint recorded in GlobalModelVersion."
            )

        with self._lock:
            for active_id, thread in list(self._active_threads.items()):
                if thread.is_alive():
                    raise ConcurrentTrainingJobError(
                        f"Another training job ('{active_id}') is currently active on the single GPU."
                    )

            client_ids = tuple(exp_config.get("client_ids", ["client_1", "client_2", "client_3"]))
            adapter_kwargs = dict(exp_config.get("adapter_kwargs") or {})
            server = FederatedLearningServer(
                experiment_id=training_id,
                task_type=exp_task,
                algorithm=exp_alg,  # type: ignore[arg-type]
                client_ids=client_ids,
                cfg=active_cfg,
                database_url=db_url,
                web3_connector=web3_connector,
                audit_required=bool(exp_config.get("audit_required", False)),
                adapter_kwargs=adapter_kwargs,
                seed=exp_seed,
            )
            # Reload and verify checkpoint on disk (raises FileNotFoundError/ValueError if missing/tampered)
            try:
                server.reload_from_checkpoint(round_num=latest_round)
            except FileNotFoundError as exc:
                raise CheckpointNotFoundError(str(exc)) from exc

            next_round = int(latest_round) + 1
            if next_round > exp_total:
                create_or_update_experiment(
                    experiment_id=training_id,
                    task_type=exp_task,
                    algorithm=exp_alg,
                    status="COMPLETED",
                    current_round=exp_total,
                    total_rounds=exp_total,
                    log_line=f"Resumed job {training_id} already had checkpoint for final round {exp_total}.",
                    database_url=db_url,
                )
                return {
                    "training_id": training_id,
                    "status": "COMPLETED",
                    "resumed_from_round": latest_round,
                    "current_round": exp_total,
                    "total_rounds": exp_total,
                }

            # Clear simulated failure when resuming if needed
            exp_config["simulate_failure"] = False
            self._cancel_requested.discard(training_id)
            self._active_servers[training_id] = server

            create_or_update_experiment(
                experiment_id=training_id,
                task_type=exp_task,
                algorithm=exp_alg,
                status="RUNNING",
                current_round=int(latest_round),
                total_rounds=exp_total,
                error_message="",
                log_line=f"Resuming job {training_id} from verified round-{latest_round} checkpoint.",
                database_url=db_url,
            )

            if run_async:
                worker = threading.Thread(
                    target=self._execute_job_worker,
                    kwargs={
                        "experiment_id": training_id,
                        "server": server,
                        "start_round": next_round,
                        "total_rounds": exp_total,
                        "job_config": exp_config,
                        "database_url": db_url,
                    },
                    name=f"omnifm-resume-{training_id}",
                    daemon=True,
                )
                self._active_threads[training_id] = worker
                worker.start()
            else:
                self._execute_job_worker(
                    experiment_id=training_id,
                    server=server,
                    start_round=next_round,
                    total_rounds=exp_total,
                    job_config=exp_config,
                    database_url=db_url,
                )

        return {
            "training_id": training_id,
            "status": "RUNNING" if run_async else self.get_training_status(training_id, database_url=db_url)["status"],
            "resumed_from_round": int(latest_round),
            "next_round": next_round,
            "total_rounds": exp_total,
        }

    def get_training_status(
        self,
        training_id: str | None = None,
        database_url: str | None = None,
    ) -> dict[str, Any]:
        """Return detailed status for `training_id` (or the latest experiment if omitted)."""
        db_url = database_url or settings.database_url
        with session_scope(db_url) as session:
            if training_id:
                exp = session.execute(
                    select(Experiment).where(Experiment.experiment_id == training_id)
                ).scalar_one_or_none()
                if exp is None:
                    raise LookupError(f"Training job '{training_id}' not found.")
            else:
                exp = (
                    session.execute(select(Experiment).order_by(desc(Experiment.id)))
                    .scalars()
                    .first()
                )
                if exp is None:
                    return {
                        "training_id": None,
                        "status": "IDLE",
                        "current_round": 0,
                        "total_rounds": 0,
                        "clients": [],
                        "latest_metrics": {},
                        "error_message": None,
                    }

            exp_id = exp.experiment_id
            latest_round = (
                session.execute(
                    select(TrainingRound)
                    .where(TrainingRound.experiment_id == exp_id)
                    .order_by(desc(TrainingRound.round_num))
                )
                .scalars()
                .first()
            )
            latest_gmv = (
                session.execute(
                    select(GlobalModelVersion)
                    .where(GlobalModelVersion.experiment_id == exp_id)
                    .order_by(desc(GlobalModelVersion.round_num))
                )
                .scalars()
                .first()
            )
            client_updates = (
                session.execute(
                    select(ClientUpdate)
                    .where(ClientUpdate.experiment_id == exp_id)
                    .order_by(desc(ClientUpdate.round_num), asc(ClientUpdate.client_id))
                )
                .scalars()
                .all()
            )
            hospitals = (
                session.execute(select(HospitalClient).order_by(asc(HospitalClient.client_id)))
                .scalars()
                .all()
            )

            latest_by_client: dict[str, dict[str, Any]] = {}
            for cu in client_updates:
                if cu.client_id not in latest_by_client:
                    latest_by_client[cu.client_id] = {
                        "client_id": cu.client_id,
                        "round_num": cu.round_num,
                        "num_samples": cu.num_samples,
                        "train_loss": cu.train_loss,
                        "task_loss": cu.task_loss,
                        "align_loss": cu.align_loss,
                        "eval_metrics": cu.eval_metrics_json,
                        "update_hash": cu.update_hash,
                        "tx_hash": cu.tx_hash,
                        "blockchain_status": cu.blockchain_status,
                        "peak_vram_mb": cu.peak_vram_mb,
                    }

            client_status_list = []
            for h in hospitals:
                cu_info = latest_by_client.get(h.client_id)
                client_status_list.append(
                    {
                        "client_id": h.client_id,
                        "hospital_name": h.hospital_name,
                        "modality": h.modality,
                        "wallet_address": h.wallet_address,
                        "status": h.status,
                        "latest_update": cu_info,
                    }
                )

            return {
                "training_id": exp.experiment_id,
                "experiment_id": exp.experiment_id,
                "task_type": exp.task_type,
                "algorithm": exp.algorithm,
                "status": exp.status,
                "current_round": exp.current_round,
                "total_rounds": exp.total_rounds,
                "seed": exp.seed,
                "peak_vram_mb": exp.peak_vram_mb,
                "error_message": exp.error_message,
                "config": exp.config_json,
                "latest_metrics": latest_round.metrics_json if latest_round else {},
                "latest_checkpoint": (
                    {
                        "round_num": latest_gmv.round_num,
                        "model_hash": latest_gmv.model_hash,
                        "checkpoint_path": latest_gmv.checkpoint_path,
                        "tx_hash": latest_gmv.tx_hash,
                    }
                    if latest_gmv
                    else None
                ),
                "clients": client_status_list,
                "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
            }

    def get_training_logs(
        self,
        training_id: str,
        limit: int = 100,
        offset: int = 0,
        database_url: str | None = None,
    ) -> dict[str, Any]:
        """Return bounded/paginated logs for `training_id`."""
        db_url = database_url or settings.database_url
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        with session_scope(db_url) as session:
            exp = session.execute(
                select(Experiment).where(Experiment.experiment_id == training_id)
            ).scalar_one_or_none()
            if exp is None:
                raise LookupError(f"Training job '{training_id}' not found.")
            all_logs = list(exp.logs_json or [])
            sliced = all_logs[offset : offset + limit]
            return {
                "training_id": exp.experiment_id,
                "status": exp.status,
                "total_lines": len(all_logs),
                "limit": limit,
                "offset": offset,
                "logs": sliced,
            }


def list_experiments(
    task_type: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    database_url: str | None = None,
) -> dict[str, Any]:
    """List experiments persisted in SQLite."""
    db_url = database_url or settings.database_url
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    with session_scope(db_url) as session:
        stmt = select(Experiment)
        if task_type:
            stmt = stmt.where(Experiment.task_type == task_type.lower().strip())
        if status:
            stmt = stmt.where(Experiment.status == status.upper().strip())
        stmt = stmt.order_by(desc(Experiment.id)).offset(offset).limit(limit)
        rows = session.execute(stmt).scalars().all()
        items = [
            {
                "experiment_id": r.experiment_id,
                "training_id": r.experiment_id,
                "task_type": r.task_type,
                "algorithm": r.algorithm,
                "status": r.status,
                "current_round": r.current_round,
                "total_rounds": r.total_rounds,
                "seed": r.seed,
                "peak_vram_mb": r.peak_vram_mb,
                "error_message": r.error_message,
                "config": r.config_json,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in rows
        ]
        return {"total": len(items), "limit": limit, "offset": offset, "items": items}


def get_metrics_history(
    task_type: str | None = None,
    training_id: str | None = None,
    algorithm: str | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Return per-round global and per-client metrics history for charts and tables."""
    db_url = database_url or settings.database_url
    with session_scope(db_url) as session:
        stmt = select(MetricRecord)
        if task_type:
            stmt = stmt.where(MetricRecord.task_type == task_type.lower().strip())
        if training_id:
            stmt = stmt.where(MetricRecord.experiment_id == training_id)
        if algorithm:
            stmt = stmt.where(MetricRecord.algorithm == algorithm.lower().strip())
        stmt = stmt.order_by(asc(MetricRecord.round_num), asc(MetricRecord.id))
        rows = session.execute(stmt).scalars().all()

        global_metrics = []
        client_metrics = []
        algorithms_seen: set[str] = set()

        for r in rows:
            algorithms_seen.add(r.algorithm)
            entry = {
                "experiment_id": r.experiment_id,
                "task_type": r.task_type,
                "algorithm": r.algorithm,
                "round_num": r.round_num,
                "scope": r.scope,
                "client_id": r.client_id,
                "metrics": r.metrics_json,
                "timestamp": r.created_at.isoformat() if r.created_at else None,
            }
            if r.scope == "global":
                global_metrics.append(entry)
            else:
                client_metrics.append(entry)

        return {
            "task_type": task_type,
            "training_id": training_id,
            "baselines_and_algorithms": sorted(algorithms_seen),
            "global_history": global_metrics,
            "client_history": client_metrics,
            "is_measured": True,
        }


job_manager = TrainingJobManager()
