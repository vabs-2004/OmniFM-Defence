"""Federated Learning Server Orchestrator (`backend/fl/server.py`, Tasks 5.4 - 5.8).

Coordinates:
1. Global shared model initialization and Round-0 checkpoint/hash recording.
2. Sequential execution of logical hospital clients (`client_1`, `client_2`, `client_3`)
   on CPU or GPU while respecting the 6 GB VRAM budget.
3. Validation and single-execution weighted FedAvg via `RoundManager`.
4. Server-side `SpectralKnowledgeBank` updates and disk persistence.
5. SQLite persistence (`Experiment`, `TrainingRound`, `ClientUpdate`,
   `GlobalModelVersion`, `MetricRecord`, `ArtifactMetadata`, `BlockchainTransaction`).
6. Optional Ganache/Hardhat blockchain audit recording via `Web3Connector`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
import torch

from backend.blockchain.web3_connector import BlockchainError, Web3Connector
from backend.config import Settings, settings
from backend.database.db import (
    append_experiment_log,
    create_or_update_experiment,
    init_db,
    record_artifact_metadata,
    record_blockchain_tx,
    session_scope,
    upsert_hospital_client,
)
from backend.database.models import (
    ClientUpdate,
    GlobalModelVersion,
    MetricRecord,
    TrainingRound,
)
from backend.fl.aggregation import filter_shared_state_dict
from backend.fl.client import FederatedHospitalClient
from backend.fl.protocols import (
    AlgorithmType,
    ClientUpdatePayload,
    ParameterSharingPolicy,
    RoundExecutionResult,
    RoundState,
    build_parameter_sharing_policy,
)
from backend.fl.round_manager import RoundManager, RoundStateTransitionError
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.tasks import BaseTaskAdapter, create_task_adapter
from backend.utils.hashing import hash_state_dict
from backend.utils.logging import save_run_metadata, set_global_seed, setup_logger

logger = setup_logger("omnifm.fl.server")


class FederatedLearningServer:
    """Central federated orchestrator managing global state, knowledge bank, and audit records."""

    def __init__(
        self,
        experiment_id: str,
        task_type: str,
        algorithm: AlgorithmType = "omnifm",
        client_ids: Sequence[str] = ("client_1", "client_2", "client_3"),
        cfg: Settings | None = None,
        database_url: str | None = None,
        web3_connector: Web3Connector | None = None,
        audit_required: bool = False,
        knowledge_bank_path: str | Path | None = None,
        adapter_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.experiment_id = experiment_id
        self.task_type = task_type.lower().strip()
        self.algorithm: AlgorithmType = algorithm
        self.client_ids = tuple(client_ids)
        self.database_url = database_url or self.cfg.database_url
        self.web3_connector = web3_connector
        self.audit_required = bool(audit_required or self.cfg.blockchain_audit_enabled)
        self.adapter_kwargs = dict(adapter_kwargs or {})
        self.seed = int(seed if seed is not None else self.cfg.random_seed)

        set_global_seed(self.seed)
        init_db(self.database_url)

        self.exp_artifact_dir = Path(self.cfg.artifact_dir) / "experiments" / self.experiment_id
        self.exp_artifact_dir.mkdir(parents=True, exist_ok=True)

        self.knowledge_bank_path = Path(
            knowledge_bank_path or self.cfg.knowledge_bank_path
        )
        self.knowledge_bank = SpectralKnowledgeBank(
            embedding_dim=self.cfg.embedding_dim,
            max_size=self.cfg.max_knowledge_bank_size,
            top_k=self.cfg.gskr_top_k,
        )

        # Initialize template adapter to define parameter-sharing policy and Round-0 state
        template_adapter = self._create_adapter()
        self._worker_adapter: BaseTaskAdapter = template_adapter
        self.policy: ParameterSharingPolicy = build_parameter_sharing_policy(
            template_adapter, algorithm=self.algorithm
        )
        self.global_shared_state: dict[str, torch.Tensor] = filter_shared_state_dict(
            template_adapter.state_dict(), self.policy
        )
        self.round_0_hash: str = ""
        self.completed_rounds: dict[int, RoundExecutionResult] = {}

        # Instantiate sequential logical hospital clients
        self.clients: dict[str, FederatedHospitalClient] = {
            cid: FederatedHospitalClient(
                client_id=cid,
                task_type=self.task_type,
                algorithm=self.algorithm,
                cfg=self.cfg,
                local_checkpoint_dir=self.exp_artifact_dir / "clients" / cid,
                adapter_kwargs=self.adapter_kwargs,
            )
            for cid in self.client_ids
        }

        for cid in self.client_ids:
            upsert_hospital_client(
                client_id=cid,
                hospital_name=f"Hospital_{cid}",
                modality=self.task_type,
                database_url=self.database_url,
            )

    def _create_adapter(self) -> BaseTaskAdapter:
        is_omnifm = self.algorithm == "omnifm"
        kwargs: dict[str, Any] = {
            "embedding_dim": self.cfg.embedding_dim,
            "lambda_align": self.cfg.spalign_lambda if is_omnifm else 0.0,
            "enable_gskr": is_omnifm,
            "enable_eca": is_omnifm,
            "enable_psp": is_omnifm,
            "enable_spalign": is_omnifm,
        }
        kwargs.update(self.adapter_kwargs)
        return create_task_adapter(self.task_type, **kwargs)

    def initialize_round_zero(self, total_rounds: int = 1) -> str:
        """Save Round-0 initial global checkpoint, compute its SHA-256 hash, and persist metadata."""
        self.round_0_hash = hash_state_dict(self.global_shared_state)
        r0_ckpt_path = self.exp_artifact_dir / "global_round_0.pt"
        torch.save(
            {
                "experiment_id": self.experiment_id,
                "task_type": self.task_type,
                "algorithm": self.algorithm,
                "round_num": 0,
                "shared_state_dict": self.global_shared_state,
                "model_hash": self.round_0_hash,
                "policy": self.policy.to_manifest_dict(),
            },
            r0_ckpt_path,
        )

        save_run_metadata(
            self.exp_artifact_dir / "run_metadata.json",
            cfg=self.cfg,
            seed=self.seed,
            extra={
                "experiment_id": self.experiment_id,
                "task_type": self.task_type,
                "algorithm": self.algorithm,
                "client_ids": list(self.client_ids),
                "policy": self.policy.to_manifest_dict(),
            },
        )

        create_or_update_experiment(
            experiment_id=self.experiment_id,
            task_type=self.task_type,
            algorithm=self.algorithm,
            status=RoundState.CREATED.value,
            current_round=0,
            total_rounds=total_rounds,
            seed=self.seed,
            config_json={
                "task_type": self.task_type,
                "algorithm": self.algorithm,
                "client_ids": list(self.client_ids),
                "total_rounds": total_rounds,
                "seed": self.seed,
                "adapter_kwargs": self.adapter_kwargs,
                "policy": self.policy.to_manifest_dict(),
            },
            log_line=f"Initialized Round 0 global model (hash={self.round_0_hash})",
            database_url=self.database_url,
        )

        tx_hash: str | None = None
        if self.web3_connector is not None:
            try:
                if hasattr(self.web3_connector, "ensure_hospital_registered"):
                    for cid in self.client_ids:
                        reg_rec = self.web3_connector.ensure_hospital_registered(
                            client_id=cid,
                            hospital_name=f"Hospital_{cid}",
                            modality=self.task_type,
                        )
                        reg_tx_hash = (
                            reg_rec.get("tx_hash")
                            if isinstance(reg_rec, dict)
                            else getattr(reg_rec, "tx_hash", None)
                        )
                        reg_block = (
                            reg_rec.get("block_number")
                            if isinstance(reg_rec, dict)
                            else getattr(reg_rec, "block_number", None)
                        )
                        reg_sender = (
                            reg_rec.get("sender") or reg_rec.get("wallet")
                            if isinstance(reg_rec, dict)
                            else getattr(reg_rec, "sender", None)
                        )
                        if isinstance(reg_tx_hash, str) and reg_tx_hash:
                            record_blockchain_tx(
                                action_type="HospitalRegistered",
                                status="CONFIRMED",
                                experiment_id=self.experiment_id,
                                task_type=self.task_type,
                                round_num=0,
                                client_id=cid,
                                tx_hash=reg_tx_hash,
                                block_number=int(reg_block) if reg_block is not None else None,
                                wallet_address=str(reg_sender) if reg_sender else None,
                                database_url=self.database_url,
                            )

                already_r0 = False
                if isinstance(self.web3_connector, Web3Connector):
                    existing_r0 = self.web3_connector.get_global_model_hash(self.experiment_id, 0)
                    already_r0 = existing_r0 != ("0x" + "00" * 32)

                if not already_r0:
                    rec = self.web3_connector.store_global_model(
                        experiment_id=self.experiment_id,
                        round_num=0,
                        task_type=self.task_type,
                        global_model_hash=self.round_0_hash,
                    )
                    tx_hash = rec.tx_hash
                    record_blockchain_tx(
                        action_type="GlobalModelStored",
                        status="CONFIRMED",
                        experiment_id=self.experiment_id,
                        task_type=self.task_type,
                        round_num=0,
                        artifact_hash=self.round_0_hash,
                        tx_hash=rec.tx_hash,
                        block_number=rec.block_number,
                        wallet_address=rec.sender,
                        database_url=self.database_url,
                    )
            except BlockchainError as exc:
                record_blockchain_tx(
                    action_type="GlobalModelStored",
                    status="FAILED",
                    experiment_id=self.experiment_id,
                    task_type=self.task_type,
                    round_num=0,
                    artifact_hash=self.round_0_hash,
                    error_message=str(exc),
                    database_url=self.database_url,
                )
                if self.audit_required:
                    raise

        with session_scope(self.database_url) as session:
            stmt = select(GlobalModelVersion).where(
                GlobalModelVersion.experiment_id == self.experiment_id,
                GlobalModelVersion.round_num == 0,
            )
            if session.execute(stmt).scalar_one_or_none() is None:
                session.add(
                    GlobalModelVersion(
                        experiment_id=self.experiment_id,
                        task_type=self.task_type,
                        algorithm=self.algorithm,
                        round_num=0,
                        model_hash=self.round_0_hash,
                        checkpoint_path=str(r0_ckpt_path),
                        metrics_json={},
                        tx_hash=tx_hash,
                    )
                )

        record_artifact_metadata(
            artifact_id=f"{self.experiment_id}_global_r0",
            experiment_id=self.experiment_id,
            task_type=self.task_type,
            round_num=0,
            artifact_type="global_checkpoint",
            file_path=str(r0_ckpt_path),
            sha256_hash=self.round_0_hash,
            database_url=self.database_url,
        )
        return self.round_0_hash

    def run_round(
        self,
        round_num: int,
        local_epochs: int | None = None,
        batch_size: int | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        lr: float = 1e-3,
        fedprox_mu: float = 0.01,
        scale: int = 2,
        device: torch.device | str | None = None,
    ) -> RoundExecutionResult:
        """Execute a single communication round across all configured hospital clients."""
        if round_num in self.completed_rounds:
            raise RoundStateTransitionError(
                f"Duplicate round execution/aggregation blocked for "
                f"({self.experiment_id}, round={round_num})."
            )
        if not self.round_0_hash:
            self.initialize_round_zero(total_rounds=max(1, round_num))

        round_mgr = RoundManager(
            experiment_id=self.experiment_id,
            round_num=round_num,
            expected_client_ids=self.client_ids,
            policy=self.policy,
            require_all_clients=True,
        )
        round_mgr.transition_to(RoundState.WAITING_FOR_CLIENTS)
        round_mgr.transition_to(RoundState.TRAINING)

        create_or_update_experiment(
            experiment_id=self.experiment_id,
            task_type=self.task_type,
            algorithm=self.algorithm,
            status=RoundState.TRAINING.value,
            current_round=round_num,
            total_rounds=max(round_num, len(self.completed_rounds) + 1),
            seed=self.seed,
            log_line=f"Starting Round {round_num} across clients {list(self.client_ids)}",
            database_url=self.database_url,
        )

        # Dispatch clients sequentially with current global shared state and bank snapshot
        round_peak_vram = 0.0
        audit_status = "NOT_CONFIGURED"

        for cid in self.client_ids:
            client = self.clients[cid]
            upd = client.train_round(
                experiment_id=self.experiment_id,
                round_num=round_num,
                shared_state_dict=self.global_shared_state,
                knowledge_bank=self.knowledge_bank,
                policy=self.policy,
                local_epochs=local_epochs,
                batch_size=batch_size,
                max_train_samples=max_train_samples,
                max_val_samples=max_val_samples,
                lr=lr,
                fedprox_mu=fedprox_mu,
                scale=scale,
                device=device,
                seed=self.seed,
                reusable_adapter=self._worker_adapter,
            )
            round_peak_vram = max(round_peak_vram, upd.peak_vram_mb)

            # Save client shared update artifact metadata off-chain
            client_upd_path = (
                self.exp_artifact_dir / "clients" / cid / f"shared_update_r{round_num}.pt"
            )
            if round_num <= 1:
                client_upd_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(upd.shared_state_dict, client_upd_path)
            record_artifact_metadata(
                artifact_id=f"{self.experiment_id}_r{round_num}_{cid}_update",
                experiment_id=self.experiment_id,
                task_type=self.task_type,
                round_num=round_num,
                client_id=cid,
                artifact_type="client_update",
                file_path=str(client_upd_path),
                sha256_hash=upd.update_hash,
                database_url=self.database_url,
            )

            # Submit client update hash to blockchain if connector configured
            if self.web3_connector is not None:
                try:
                    if hasattr(self.web3_connector, "ensure_hospital_registered"):
                        reg_rec = self.web3_connector.ensure_hospital_registered(
                            client_id=cid,
                            hospital_name=f"Hospital_{cid}",
                            modality=self.task_type,
                        )
                        reg_tx_hash = (
                            reg_rec.get("tx_hash")
                            if isinstance(reg_rec, dict)
                            else getattr(reg_rec, "tx_hash", None)
                        )
                        reg_block = (
                            reg_rec.get("block_number")
                            if isinstance(reg_rec, dict)
                            else getattr(reg_rec, "block_number", None)
                        )
                        reg_sender = (
                            reg_rec.get("sender") or reg_rec.get("wallet")
                            if isinstance(reg_rec, dict)
                            else getattr(reg_rec, "sender", None)
                        )
                        if isinstance(reg_tx_hash, str) and reg_tx_hash:
                            record_blockchain_tx(
                                action_type="HospitalRegistered",
                                status="CONFIRMED",
                                experiment_id=self.experiment_id,
                                task_type=self.task_type,
                                round_num=round_num,
                                client_id=cid,
                                tx_hash=reg_tx_hash,
                                block_number=int(reg_block) if reg_block is not None else None,
                                wallet_address=str(reg_sender) if reg_sender else None,
                                database_url=self.database_url,
                            )
                    already_submitted = False
                    if isinstance(self.web3_connector, Web3Connector):
                        sub_state = self.web3_connector.get_client_submission(
                            self.experiment_id, round_num, cid
                        )
                        already_submitted = bool(sub_state.get("exists"))

                    if already_submitted:
                        upd.blockchain_status = "CONFIRMED"
                    else:
                        tx_rec = self.web3_connector.submit_model_update(
                            experiment_id=self.experiment_id,
                            round_num=round_num,
                            task_type=self.task_type,
                            model_hash=upd.update_hash,
                            client_id=cid,
                        )
                        upd.blockchain_tx_hash = tx_rec.tx_hash
                        upd.blockchain_status = "CONFIRMED"
                        record_blockchain_tx(
                            action_type="ModelUpdateSubmitted",
                            status="CONFIRMED",
                            experiment_id=self.experiment_id,
                            task_type=self.task_type,
                            round_num=round_num,
                            client_id=cid,
                            artifact_hash=upd.update_hash,
                            tx_hash=tx_rec.tx_hash,
                            block_number=tx_rec.block_number,
                            wallet_address=tx_rec.sender,
                            database_url=self.database_url,
                        )
                except BlockchainError as exc:
                    upd.blockchain_status = "FAILED"
                    record_blockchain_tx(
                        action_type="ModelUpdateSubmitted",
                        status="FAILED",
                        experiment_id=self.experiment_id,
                        task_type=self.task_type,
                        round_num=round_num,
                        client_id=cid,
                        artifact_hash=upd.update_hash,
                        error_message=str(exc),
                        database_url=self.database_url,
                    )
                    if self.audit_required:
                        round_mgr.transition_to(RoundState.FAILED)
                        raise

            round_mgr.accept_client_update(upd)
            self._persist_client_update(upd)

        # Validate submissions and run weighted FedAvg once
        self.global_shared_state = round_mgr.validate_and_aggregate()
        global_hash = hash_state_dict(self.global_shared_state)

        # Update and persist SpectralKnowledgeBank after aggregation
        for upd in round_mgr.submissions:
            if upd.spectral_prototype is not None:
                self.knowledge_bank.add_prototypes(
                    embeddings=upd.spectral_prototype,
                    task_type=self.task_type,
                    client_id=upd.client_id,
                    round_num=round_num,
                    sample_count=upd.num_samples,
                )
        self.knowledge_bank.save(self.knowledge_bank_path)

        # Save global checkpoint artifact
        ckpt_path = self.exp_artifact_dir / f"global_round_{round_num}.pt"
        torch.save(
            {
                "experiment_id": self.experiment_id,
                "task_type": self.task_type,
                "algorithm": self.algorithm,
                "round_num": round_num,
                "shared_state_dict": self.global_shared_state,
                "model_hash": global_hash,
                "knowledge_bank_size": len(self.knowledge_bank),
            },
            ckpt_path,
        )
        record_artifact_metadata(
            artifact_id=f"{self.experiment_id}_global_r{round_num}",
            experiment_id=self.experiment_id,
            task_type=self.task_type,
            round_num=round_num,
            artifact_type="global_checkpoint",
            file_path=str(ckpt_path),
            sha256_hash=global_hash,
            database_url=self.database_url,
        )

        # Record aggregation and global model hash on-chain if connector configured
        agg_tx_hash: str | None = None
        gm_tx_hash: str | None = None
        if self.web3_connector is not None:
            try:
                already_agg = False
                already_gm = False
                if isinstance(self.web3_connector, Web3Connector):
                    r_stat = self.web3_connector.get_round_status(
                        self.experiment_id, round_num
                    )
                    already_agg = bool(r_stat.get("aggregated"))
                    already_gm = bool(r_stat.get("global_model_stored"))

                if not already_agg:
                    agg_rec = self.web3_connector.trigger_aggregation(
                        experiment_id=self.experiment_id,
                        round_num=round_num,
                    )
                    agg_tx_hash = agg_rec.tx_hash
                    record_blockchain_tx(
                        action_type="AggregationTriggered",
                        status="CONFIRMED",
                        experiment_id=self.experiment_id,
                        task_type=self.task_type,
                        round_num=round_num,
                        tx_hash=agg_rec.tx_hash,
                        block_number=agg_rec.block_number,
                        wallet_address=agg_rec.sender,
                        database_url=self.database_url,
                    )

                if not already_gm:
                    gm_rec = self.web3_connector.store_global_model(
                        experiment_id=self.experiment_id,
                        round_num=round_num,
                        task_type=self.task_type,
                        global_model_hash=global_hash,
                    )
                    gm_tx_hash = gm_rec.tx_hash
                    record_blockchain_tx(
                        action_type="GlobalModelStored",
                        status="CONFIRMED",
                        experiment_id=self.experiment_id,
                        task_type=self.task_type,
                        round_num=round_num,
                        artifact_hash=global_hash,
                        tx_hash=gm_rec.tx_hash,
                        block_number=gm_rec.block_number,
                        wallet_address=gm_rec.sender,
                        database_url=self.database_url,
                    )
                audit_status = "CONFIRMED"
            except BlockchainError as exc:
                audit_status = "FAILED"
                record_blockchain_tx(
                    action_type="RoundAudit",
                    status="FAILED",
                    experiment_id=self.experiment_id,
                    task_type=self.task_type,
                    round_num=round_num,
                    artifact_hash=global_hash,
                    error_message=str(exc),
                    database_url=self.database_url,
                )
                if self.audit_required:
                    raise

        # Compute sample-weighted global aggregate metrics across participating clients
        global_metrics = self._aggregate_client_metrics(round_mgr.submissions)
        for upd in round_mgr.submissions:
            upd.shared_state_dict = {}

        result = RoundExecutionResult(
            experiment_id=self.experiment_id,
            round_num=round_num,
            task_type=self.task_type,
            algorithm=self.algorithm,
            state=round_mgr.state,
            client_updates=round_mgr.submissions,
            global_model_hash=global_hash,
            global_checkpoint_path=str(ckpt_path),
            global_metrics=global_metrics,
            knowledge_bank_size=len(self.knowledge_bank),
            peak_vram_mb=round_peak_vram,
            aggregation_tx_hash=agg_tx_hash,
            global_model_tx_hash=gm_tx_hash,
            audit_status=audit_status,
        )
        self.completed_rounds[round_num] = result
        self._persist_round_completion(result)
        import gc

        gc.collect()
        return result

    def run_training(
        self,
        num_rounds: int,
        start_round: int = 1,
        should_cancel: Callable[[], bool] | None = None,
        **round_kwargs: Any,
    ) -> list[RoundExecutionResult]:
        """Run multi-round federated training with cancellation and failure handling."""
        if start_round <= 1 and not self.round_0_hash:
            self.initialize_round_zero(total_rounds=num_rounds)

        results: list[RoundExecutionResult] = []
        try:
            for r in range(start_round, num_rounds + 1):
                if should_cancel is not None and should_cancel():
                    create_or_update_experiment(
                        experiment_id=self.experiment_id,
                        task_type=self.task_type,
                        algorithm=self.algorithm,
                        status=RoundState.CANCELLED.value,
                        current_round=r - 1,
                        total_rounds=num_rounds,
                        seed=self.seed,
                        log_line=f"Cancelled cleanly before Round {r}.",
                        database_url=self.database_url,
                    )
                    return results

                res = self.run_round(round_num=r, **round_kwargs)
                results.append(res)

            create_or_update_experiment(
                experiment_id=self.experiment_id,
                task_type=self.task_type,
                algorithm=self.algorithm,
                status=RoundState.COMPLETED.value,
                current_round=num_rounds,
                total_rounds=num_rounds,
                seed=self.seed,
                log_line=f"Completed all {num_rounds} rounds successfully.",
                peak_vram_mb=max((r.peak_vram_mb for r in results), default=0.0),
                database_url=self.database_url,
            )
            return results
        except Exception as exc:
            create_or_update_experiment(
                experiment_id=self.experiment_id,
                task_type=self.task_type,
                algorithm=self.algorithm,
                status=RoundState.FAILED.value,
                current_round=len(self.completed_rounds),
                total_rounds=num_rounds,
                seed=self.seed,
                error_message=str(exc),
                log_line=f"Training failed: {exc}",
                database_url=self.database_url,
            )
            raise

    def reload_from_checkpoint(self, round_num: int) -> dict[str, Any]:
        """Reload global shared model state and knowledge bank from a saved checkpoint."""
        ckpt_path = self.exp_artifact_dir / f"global_round_{round_num}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Global checkpoint not found for reload: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        loaded_state: dict[str, torch.Tensor] = payload["shared_state_dict"]
        recomputed_hash = hash_state_dict(loaded_state)
        if recomputed_hash != payload["model_hash"]:
            raise ValueError(
                f"Checkpoint hash mismatch on reload of {ckpt_path}: "
                f"expected {payload['model_hash']}, got {recomputed_hash}."
            )

        self.global_shared_state = {k: v.detach().cpu().clone() for k, v in loaded_state.items()}
        if self.knowledge_bank_path.is_file():
            self.knowledge_bank = SpectralKnowledgeBank.load(self.knowledge_bank_path)

        for cid, client in self.clients.items():
            p_file = (
                self.exp_artifact_dir
                / "clients"
                / cid
                / f"{self.experiment_id}_r{round_num}_personalized.pt"
            )
            if p_file.is_file():
                client._personalized_state = torch.load(
                    p_file, map_location="cpu", weights_only=False
                )

        return payload

    @staticmethod
    def _aggregate_client_metrics(updates: Sequence[ClientUpdatePayload]) -> dict[str, Any]:
        total_n = float(sum(u.num_samples for u in updates))
        avg_train_loss = sum(u.train_loss * u.num_samples for u in updates) / total_n
        avg_task_loss = sum(u.task_loss * u.num_samples for u in updates) / total_n
        avg_align_loss = sum(u.align_loss * u.num_samples for u in updates) / total_n

        numeric_keys: set[str] = set()
        for u in updates:
            for k, v in u.eval_metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_keys.add(k)

        agg: dict[str, Any] = {
            "train_loss": round(avg_train_loss, 6),
            "task_loss": round(avg_task_loss, 6),
            "align_loss": round(avg_align_loss, 6),
        }
        for k in sorted(numeric_keys):
            vals = [float(u.eval_metrics[k]) for u in updates if k in u.eval_metrics]
            if vals:
                agg[k] = round(float(np.mean(vals)), 6)
        return agg

    def _persist_client_update(self, upd: ClientUpdatePayload) -> None:
        with session_scope(self.database_url) as session:
            existing = session.execute(
                select(ClientUpdate).where(
                    ClientUpdate.experiment_id == upd.experiment_id,
                    ClientUpdate.round_num == upd.round_num,
                    ClientUpdate.client_id == upd.client_id,
                )
            ).scalar_one_or_none()
            if existing is not None:
                existing.task_type = upd.task_type
                existing.num_samples = upd.num_samples
                existing.update_hash = upd.update_hash
                existing.train_loss = upd.train_loss
                existing.task_loss = upd.task_loss
                existing.align_loss = upd.align_loss
                existing.eval_metrics_json = upd.eval_metrics
                existing.duration_sec = upd.duration_sec
                existing.peak_vram_mb = upd.peak_vram_mb
                existing.tx_hash = upd.blockchain_tx_hash or existing.tx_hash
                existing.blockchain_status = upd.blockchain_status
            else:
                session.add(
                    ClientUpdate(
                        experiment_id=upd.experiment_id,
                        round_num=upd.round_num,
                        client_id=upd.client_id,
                        task_type=upd.task_type,
                        num_samples=upd.num_samples,
                        update_hash=upd.update_hash,
                        train_loss=upd.train_loss,
                        task_loss=upd.task_loss,
                        align_loss=upd.align_loss,
                        eval_metrics_json=upd.eval_metrics,
                        duration_sec=upd.duration_sec,
                        peak_vram_mb=upd.peak_vram_mb,
                        tx_hash=upd.blockchain_tx_hash,
                        blockchain_status=upd.blockchain_status,
                    )
                )
            session.add(
                MetricRecord(
                    experiment_id=upd.experiment_id,
                    task_type=upd.task_type,
                    algorithm=upd.algorithm,
                    round_num=upd.round_num,
                    scope="client",
                    client_id=upd.client_id,
                    metrics_json={
                        "train_loss": upd.train_loss,
                        "task_loss": upd.task_loss,
                        "align_loss": upd.align_loss,
                        **upd.eval_metrics,
                    },
                )
            )

    def _persist_round_completion(self, res: RoundExecutionResult) -> None:
        with session_scope(self.database_url) as session:
            existing_tr = session.execute(
                select(TrainingRound).where(
                    TrainingRound.experiment_id == res.experiment_id,
                    TrainingRound.round_num == res.round_num,
                )
            ).scalar_one_or_none()
            if existing_tr is not None:
                existing_tr.state = res.state.value
                existing_tr.expected_clients = len(self.client_ids)
                existing_tr.submitted_clients = len(res.client_updates)
                existing_tr.aggregated = True
                existing_tr.global_model_hash = res.global_model_hash
                existing_tr.global_checkpoint_path = res.global_checkpoint_path
                existing_tr.knowledge_bank_size = res.knowledge_bank_size
                existing_tr.peak_vram_mb = res.peak_vram_mb
                existing_tr.audit_status = res.audit_status
                existing_tr.metrics_json = res.global_metrics
                existing_tr.completed_at = datetime.now(timezone.utc)
            else:
                session.add(
                    TrainingRound(
                        experiment_id=res.experiment_id,
                        round_num=res.round_num,
                        state=res.state.value,
                        expected_clients=len(self.client_ids),
                        submitted_clients=len(res.client_updates),
                        aggregated=True,
                        global_model_hash=res.global_model_hash,
                        global_checkpoint_path=res.global_checkpoint_path,
                        knowledge_bank_size=res.knowledge_bank_size,
                        peak_vram_mb=res.peak_vram_mb,
                        audit_status=res.audit_status,
                        metrics_json=res.global_metrics,
                        completed_at=datetime.now(timezone.utc),
                    )
                )

            existing_gmv = session.execute(
                select(GlobalModelVersion).where(
                    GlobalModelVersion.experiment_id == res.experiment_id,
                    GlobalModelVersion.round_num == res.round_num,
                )
            ).scalar_one_or_none()
            if existing_gmv is not None:
                existing_gmv.model_hash = res.global_model_hash
                existing_gmv.checkpoint_path = res.global_checkpoint_path
                existing_gmv.metrics_json = res.global_metrics
                existing_gmv.tx_hash = res.global_model_tx_hash or existing_gmv.tx_hash
            else:
                session.add(
                    GlobalModelVersion(
                        experiment_id=res.experiment_id,
                        task_type=res.task_type,
                        algorithm=res.algorithm,
                        round_num=res.round_num,
                        model_hash=res.global_model_hash,
                        checkpoint_path=res.global_checkpoint_path,
                        metrics_json=res.global_metrics,
                        tx_hash=res.global_model_tx_hash,
                    )
                )

            session.add(
                MetricRecord(
                    experiment_id=res.experiment_id,
                    task_type=res.task_type,
                    algorithm=res.algorithm,
                    round_num=res.round_num,
                    scope="global",
                    client_id=None,
                    metrics_json=res.global_metrics,
                )
            )
        append_experiment_log(
            experiment_id=self.experiment_id,
            message=(
                f"Completed Round {res.round_num}: global_hash={res.global_model_hash[:14]}... "
                f"bank_size={res.knowledge_bank_size} peak_vram={res.peak_vram_mb} MB "
                f"metrics={res.global_metrics}"
            ),
            database_url=self.database_url,
        )
