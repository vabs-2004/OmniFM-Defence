"""Blockchain Audit, Artifact Hash Verification, and Metrics History Endpoints (`backend/api/endpoints/audit.py`, Task 6.5)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.blockchain.web3_connector import Web3Connector
from backend.config import settings
from backend.services.artifact_service import list_artifacts
from backend.services.audit_service import (
    get_blockchain_audit_log,
    verify_artifact_provenance,
)
from backend.services.experiment_service import get_metrics_history

router = APIRouter(tags=["Audit & Metrics"])


class BlockchainVerifyRequest(BaseModel):
    """Request schema for `POST /api/blockchain/verify`."""

    artifact_id: str | None = Field(default=None, description="ArtifactMetadata ID, e.g., exp_1_r1_global")
    experiment_id: str | None = Field(default=None, description="Experiment ID")
    round_num: int | None = Field(default=None, description="Federated round number")
    client_id: str | None = Field(default=None, description="Optional client_id for client update verification")
    provided_hash: str | None = Field(default=None, description="Optional 0x-prefixed SHA-256 hex digest to compare")
    file_path: str | None = Field(default=None, description="Optional local checkpoint path to recompute hash")


def _get_db_url(request: Request) -> str:
    return getattr(request.app.state, "database_url", None) or settings.database_url


def _get_web3(request: Request) -> Web3Connector | None:
    if hasattr(request.app.state, "web3_connector"):
        return request.app.state.web3_connector
    try:
        return Web3Connector(cfg=settings)
    except Exception:
        return None


@router.get("/api/blockchain/log")
def get_blockchain_log_endpoint(
    request: Request,
    experiment_id: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    round_num: int | None = Query(default=None),
    client_id: str | None = Query(default=None),
    action_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return paginated blockchain audit events, transaction hashes, and confirmation status."""
    return get_blockchain_audit_log(
        experiment_id=experiment_id,
        task_type=task_type,
        round_num=round_num,
        client_id=client_id,
        action_type=action_type,
        status=status,
        limit=limit,
        offset=offset,
        database_url=_get_db_url(request),
    )


@router.post("/api/blockchain/verify")
def verify_blockchain_artifact_endpoint(
    payload: BlockchainVerifyRequest,
    request: Request,
) -> dict[str, Any]:
    """Recompute or compare an artifact's SHA-256 hash against recorded off-chain and on-chain hashes."""
    if not payload.artifact_id and not payload.experiment_id:
        raise HTTPException(
            status_code=400,
            detail="Either 'artifact_id' or 'experiment_id' must be provided for verification.",
        )
    try:
        return verify_artifact_provenance(
            artifact_id=payload.artifact_id,
            experiment_id=payload.experiment_id,
            round_num=payload.round_num,
            client_id=payload.client_id,
            provided_hash=payload.provided_hash,
            file_path=payload.file_path,
            web3_connector=_get_web3(request),
            database_url=_get_db_url(request),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/metrics/history")
def get_metrics_history_endpoint(
    request: Request,
    task_type: str | None = Query(default=None),
    training_id: str | None = Query(default=None),
    algorithm: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return per-round global and per-client metrics history and baseline identifiers."""
    return get_metrics_history(
        task_type=task_type,
        training_id=training_id,
        algorithm=algorithm,
        database_url=_get_db_url(request),
    )


@router.get("/api/artifacts")
def get_artifacts_endpoint(
    request: Request,
    experiment_id: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    round_num: int | None = Query(default=None),
) -> dict[str, Any]:
    """List persisted off-chain artifact metadata records."""
    items = list_artifacts(
        experiment_id=experiment_id,
        task_type=task_type,
        round_num=round_num,
        database_url=_get_db_url(request),
    )
    return {"total": len(items), "items": items}
