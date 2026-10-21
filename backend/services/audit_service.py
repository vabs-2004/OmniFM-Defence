"""Blockchain Audit Log and Artifact Provenance Verification Service (`backend/services/audit_service.py`, Task 6.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import desc, func, select
import torch

from backend.blockchain.web3_connector import Web3Connector
from backend.database.db import session_scope
from backend.database.models import (
    ArtifactMetadata,
    BlockchainTransaction,
    ClientUpdate,
    GlobalModelVersion,
    TrainingRound,
)
from backend.utils.hashing import hash_file, hash_state_dict, normalize_bytes32_hex

AUDIT_DISCLAIMER = (
    "On-chain transaction inclusion and SHA-256 hash matching verify artifact provenance and "
    "immutability after recording; they do NOT constitute cryptographic proof of honest client "
    "training or raw dataset quality."
)


def get_blockchain_audit_log(
    experiment_id: str | None = None,
    task_type: str | None = None,
    round_num: int | None = None,
    client_id: str | None = None,
    action_type: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Query paginated blockchain audit records and summary statistics."""
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    with session_scope(database_url) as session:
        stmt = select(BlockchainTransaction)
        count_stmt = select(func.count(BlockchainTransaction.id))

        filters = []
        if experiment_id:
            filters.append(BlockchainTransaction.experiment_id == experiment_id)
        if task_type:
            filters.append(BlockchainTransaction.task_type == task_type.lower().strip())
        if round_num is not None:
            filters.append(BlockchainTransaction.round_num == int(round_num))
        if client_id:
            filters.append(BlockchainTransaction.client_id == client_id)
        if action_type:
            filters.append(BlockchainTransaction.action_type == action_type)
        if status:
            filters.append(BlockchainTransaction.status == status.upper())

        for cond in filters:
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        total = int(session.execute(count_stmt).scalar() or 0)
        stmt = stmt.order_by(desc(BlockchainTransaction.id)).offset(offset).limit(limit)
        rows = session.execute(stmt).scalars().all()

        confirmed_stmt = select(func.count(BlockchainTransaction.id)).where(
            BlockchainTransaction.status == "CONFIRMED"
        )
        for cond in filters:
            confirmed_stmt = confirmed_stmt.where(cond)
        confirmed_count = int(session.execute(confirmed_stmt).scalar() or 0)

        round_stmt = select(func.count(TrainingRound.id)).where(TrainingRound.aggregated.is_(True))
        if experiment_id:
            round_stmt = round_stmt.where(TrainingRound.experiment_id == experiment_id)
        completed_rounds = int(session.execute(round_stmt).scalar() or 0)

        items = [
            {
                "id": r.id,
                "experiment_id": r.experiment_id,
                "task_type": r.task_type,
                "round_num": r.round_num,
                "client_id": r.client_id,
                "action_type": r.action_type,
                "artifact_hash": r.artifact_hash,
                "tx_hash": r.tx_hash,
                "block_number": r.block_number,
                "wallet_address": r.wallet_address,
                "status": r.status,
                "error_message": r.error_message,
                "timestamp": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "confirmed_transactions": confirmed_count,
        "completed_rounds": completed_rounds,
        "audit_disclaimer": AUDIT_DISCLAIMER,
        "items": items,
    }


def verify_artifact_provenance(
    artifact_id: str | None = None,
    experiment_id: str | None = None,
    round_num: int | None = None,
    client_id: str | None = None,
    provided_hash: str | None = None,
    file_path: str | None = None,
    web3_connector: Web3Connector | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Recompute or compare an artifact's canonical SHA-256 hash against off-chain and on-chain records."""
    recorded_hash: str | None = None
    resolved_path: str | None = file_path
    record_meta: dict[str, Any] = {}
    tx_record: dict[str, Any] | None = None

    with session_scope(database_url) as session:
        if artifact_id:
            art = session.execute(
                select(ArtifactMetadata).where(ArtifactMetadata.artifact_id == artifact_id)
            ).scalar_one_or_none()
            if art is not None:
                recorded_hash = art.sha256_hash
                resolved_path = resolved_path or art.file_path
                experiment_id = experiment_id or art.experiment_id
                round_num = round_num if round_num is not None else art.round_num
                client_id = client_id or art.client_id
                record_meta = {
                    "source": "ArtifactMetadata",
                    "artifact_id": art.artifact_id,
                    "artifact_type": art.artifact_type,
                    "experiment_id": art.experiment_id,
                    "task_type": art.task_type,
                    "round_num": art.round_num,
                    "client_id": art.client_id,
                    "file_path": art.file_path,
                    "recorded_hash": art.sha256_hash,
                }

        if recorded_hash is None and experiment_id is not None and round_num is not None:
            if client_id:
                cu = session.execute(
                    select(ClientUpdate).where(
                        ClientUpdate.experiment_id == experiment_id,
                        ClientUpdate.round_num == int(round_num),
                        ClientUpdate.client_id == client_id,
                    )
                ).scalar_one_or_none()
                if cu is not None:
                    recorded_hash = cu.update_hash
                    record_meta = {
                        "source": "ClientUpdate",
                        "experiment_id": cu.experiment_id,
                        "task_type": cu.task_type,
                        "round_num": cu.round_num,
                        "client_id": cu.client_id,
                        "recorded_hash": cu.update_hash,
                        "tx_hash": cu.tx_hash,
                        "blockchain_status": cu.blockchain_status,
                    }
            else:
                gmv = session.execute(
                    select(GlobalModelVersion).where(
                        GlobalModelVersion.experiment_id == experiment_id,
                        GlobalModelVersion.round_num == int(round_num),
                    )
                ).scalar_one_or_none()
                if gmv is not None:
                    recorded_hash = gmv.model_hash
                    resolved_path = resolved_path or gmv.checkpoint_path
                    record_meta = {
                        "source": "GlobalModelVersion",
                        "experiment_id": gmv.experiment_id,
                        "task_type": gmv.task_type,
                        "round_num": gmv.round_num,
                        "file_path": gmv.checkpoint_path,
                        "recorded_hash": gmv.model_hash,
                        "tx_hash": gmv.tx_hash,
                    }

        if experiment_id is not None:
            tx_stmt = select(BlockchainTransaction).where(
                BlockchainTransaction.experiment_id == experiment_id
            )
            if round_num is not None:
                tx_stmt = tx_stmt.where(BlockchainTransaction.round_num == int(round_num))
            if client_id is not None:
                tx_stmt = tx_stmt.where(BlockchainTransaction.client_id == client_id)
            tx_stmt = tx_stmt.order_by(desc(BlockchainTransaction.id))
            tx = session.execute(tx_stmt).scalars().first()
            if tx is not None:
                tx_record = {
                    "tx_hash": tx.tx_hash,
                    "block_number": tx.block_number,
                    "action_type": tx.action_type,
                    "status": tx.status,
                    "artifact_hash": tx.artifact_hash,
                }
                if recorded_hash is None:
                    recorded_hash = tx.artifact_hash

    if recorded_hash is None:
        raise LookupError(
            "No matching artifact or blockchain record found for verification context "
            f"(artifact_id={artifact_id}, experiment_id={experiment_id}, round_num={round_num}, client_id={client_id})."
        )

    recomputed_hash: str | None = None
    verification_method = "none"

    if resolved_path and Path(resolved_path).is_file():
        p = Path(resolved_path)
        if p.suffix == ".pt":
            payload = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "shared_state_dict" in payload:
                recomputed_hash = hash_state_dict(payload["shared_state_dict"])
                verification_method = "canonical_state_dict_sha256"
            elif isinstance(payload, dict):
                recomputed_hash = hash_state_dict(payload)
                verification_method = "canonical_state_dict_sha256"
            else:
                recomputed_hash = hash_file(p)
                verification_method = "file_sha256"
        else:
            recomputed_hash = hash_file(p)
            verification_method = "file_sha256"
    elif provided_hash:
        recomputed_hash = normalize_bytes32_hex(provided_hash)
        verification_method = "provided_hash_comparison"

    norm_recorded = normalize_bytes32_hex(recorded_hash)
    norm_recomputed = normalize_bytes32_hex(recomputed_hash) if recomputed_hash else None
    off_chain_hash_match = bool(norm_recomputed is not None and norm_recomputed == norm_recorded)

    on_chain_verified = False
    on_chain_status = "UNAVAILABLE"
    on_chain_hash: str | None = None
    on_chain_error: str | None = None

    if (
        web3_connector is not None
        and experiment_id is not None
        and round_num is not None
    ):
        try:
            if web3_connector.is_connected() and web3_connector.is_contract_deployed():
                ver = web3_connector.verify_model_hash(
                    experiment_id=experiment_id,
                    round_num=int(round_num),
                    expected_hash=norm_recorded,
                    client_id=client_id,
                )
                on_chain_hash = ver.get("on_chain_hash")
                if not ver.get("exists"):
                    on_chain_status = "NOT_ON_CHAIN"
                    on_chain_verified = False
                elif ver.get("match"):
                    on_chain_status = "VERIFIED"
                    on_chain_verified = True
                else:
                    on_chain_status = "MISMATCH"
                    on_chain_verified = False
            else:
                on_chain_status = "UNAVAILABLE"
                on_chain_error = "Blockchain RPC or deployed contract is unavailable."
        except Exception as exc:
            on_chain_status = "UNAVAILABLE"
            on_chain_verified = False
            on_chain_error = str(exc)

    tx_included = bool(tx_record and tx_record.get("status") == "CONFIRMED" and tx_record.get("tx_hash"))
    overall_match = bool(off_chain_hash_match and on_chain_status != "MISMATCH")

    return {
        "match": overall_match,
        "hash_match": off_chain_hash_match,
        "off_chain_hash_match": off_chain_hash_match,
        "recorded_hash": norm_recorded,
        "recomputed_hash": norm_recomputed,
        "on_chain_hash": on_chain_hash,
        "verification_method": verification_method,
        "transaction_included_on_chain": tx_included,
        "on_chain_contract_verified": on_chain_verified,
        "on_chain_verification_status": on_chain_status,
        "on_chain_error": on_chain_error,
        "record": record_meta,
        "transaction": tx_record,
        "audit_disclaimer": AUDIT_DISCLAIMER,
    }

