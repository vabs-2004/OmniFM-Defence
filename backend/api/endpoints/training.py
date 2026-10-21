"""Asynchronous Federated Training Control and Experiment Endpoints (`backend/api/endpoints/training.py`, Tasks 6.3 & 6.5)."""

from __future__ import annotations

import base64
import hmac
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.blockchain.web3_connector import Web3Connector
from backend.config import Settings, settings
from backend.fl.aggregation import AggregationValidationError
from backend.fl.protocols import ClientUpdatePayload
from backend.fl.round_manager import RoundStateTransitionError
from backend.services.artifact_service import CheckpointNotFoundError
from backend.services.experiment_service import (
    ConcurrentTrainingJobError,
    InvalidTrainingStateError,
    TrainingJobManager,
    job_manager,
    list_experiments,
)
from backend.utils.hashing import deserialize_state_dict_canonical, hash_state_dict

router = APIRouter(tags=["Experiments & Training"])


class TrainingStartRequest(BaseModel):
    """Request schema for `POST /api/training/start`."""

    task_type: str = Field(..., description="classification | segmentation | superresolution | fusion")
    num_rounds: int = Field(default=20, ge=1, le=100, description="Number of federated rounds")
    execution_mode: str = Field(
        default="single_laptop",
        description="single_laptop (default failsafe) | distributed_two_laptop",
    )
    config: dict[str, Any] = Field(default_factory=dict, description="Algorithm, batch size, ablation flags")
    training_id: str | None = Field(default=None, description="Optional explicit experiment identifier")
    run_async: bool = Field(default=True, description="Run training asynchronously in background thread")


class TrainingResumeRequest(BaseModel):
    """Optional request body for `POST /api/training/{training_id}/resume`."""

    run_async: bool = Field(default=True, description="Resume asynchronously in background thread")


class RemoteClientSubmitRequest(BaseModel):
    """Request schema for `POST /api/fl/rounds/{experiment_id}/{round_num}/submit`."""

    client_id: str = Field(..., min_length=1, max_length=64, description="Logical hospital client ID")
    num_samples: int = Field(..., gt=0, le=1_000_000, description="Local training sample count N_k")
    shared_state_b64: str = Field(
        ...,
        min_length=16,
        max_length=64_000_000,
        description="Base64-encoded OMNIFM_STATE_V1 canonical shared parameter payload",
    )
    spectral_prototype_b64: str | None = Field(
        default=None,
        max_length=4_000_000,
        description="Optional Base64-encoded OMNIFM_STATE_V1 spectral prototype tensor",
    )
    update_hash: str = Field(
        ...,
        min_length=64,
        max_length=66,
        description="SHA-256 hex digest of shared_state_dict",
    )
    train_loss: float = Field(default=0.0, description="Local training loss")
    task_loss: float = Field(default=0.0, description="Local task loss")
    align_loss: float = Field(default=0.0, description="Local spectral alignment loss")
    eval_metrics: dict[str, float] = Field(default_factory=dict, description="Local validation metrics")
    peak_vram_mb: float = Field(default=0.0, ge=0.0, description="Peak memory usage in MB")


def _get_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or settings


def _get_db_url(request: Request) -> str:
    return getattr(request.app.state, "database_url", None) or _get_settings(request).database_url


def _get_web3(request: Request) -> Web3Connector | None:
    if hasattr(request.app.state, "web3_connector"):
        return request.app.state.web3_connector
    try:
        return Web3Connector(cfg=_get_settings(request))
    except Exception:
        return None


def _get_manager(request: Request) -> TrainingJobManager:
    return getattr(request.app.state, "job_manager", None) or job_manager


def _verify_remote_client_token(request: Request) -> None:
    """Authenticate remote hospital worker using constant-time token comparison."""
    cfg = _get_settings(request)
    expected_token = cfg.fl_client_auth_token.get_secret_value().strip()
    if not expected_token:
        raise HTTPException(
            status_code=403,
            detail="Distributed remote client endpoints are disabled: FL_CLIENT_AUTH_TOKEN is not configured.",
        )
    provided_token = (request.headers.get("X-OmniFM-Client-Token") or "").strip()
    if not provided_token or not hmac.compare_digest(
        provided_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-OmniFM-Client-Token header.",
        )


@router.post("/api/training/start")
def start_training_endpoint(payload: TrainingStartRequest, request: Request) -> dict[str, Any]:
    """Validate configuration and create an asynchronous federated training job."""
    mgr = _get_manager(request)
    try:
        return mgr.start_training(
            task_type=payload.task_type,
            num_rounds=payload.num_rounds,
            execution_mode=payload.execution_mode,
            config=payload.config,
            training_id=payload.training_id,
            cfg=_get_settings(request),
            web3_connector=_get_web3(request),
            database_url=_get_db_url(request),
            run_async=payload.run_async,
        )
    except ConcurrentTrainingJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/training/status")
def get_training_status_endpoint(
    request: Request,
    training_id: str | None = Query(default=None, description="Training/Experiment ID"),
) -> dict[str, Any]:
    """Return status, current round, per-client status, latest metrics, and error details."""
    mgr = _get_manager(request)
    try:
        return mgr.get_training_status(training_id=training_id, database_url=_get_db_url(request))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/training/{training_id}/cancel")
def cancel_training_endpoint(training_id: str, request: Request) -> dict[str, Any]:
    """Request safe cancellation at a client/round boundary."""
    mgr = _get_manager(request)
    try:
        return mgr.cancel_training(training_id=training_id, database_url=_get_db_url(request))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/training/{training_id}/resume")
def resume_training_endpoint(
    training_id: str,
    request: Request,
    payload: TrainingResumeRequest | None = None,
) -> dict[str, Any]:
    """Resume a recoverable job from its latest verified checkpoint."""
    mgr = _get_manager(request)
    run_async = payload.run_async if payload is not None else True
    try:
        return mgr.resume_training(
            training_id=training_id,
            cfg=_get_settings(request),
            web3_connector=_get_web3(request),
            database_url=_get_db_url(request),
            run_async=run_async,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConcurrentTrainingJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (InvalidTrainingStateError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/training/{training_id}/logs")
def get_training_logs_endpoint(
    training_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return bounded/paginated logs for a training job."""
    mgr = _get_manager(request)
    try:
        return mgr.get_training_logs(
            training_id=training_id,
            limit=limit,
            offset=offset,
            database_url=_get_db_url(request),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/experiments")
def list_experiments_endpoint(
    request: Request,
    task_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List experiments with task, configuration, status, and timestamps."""
    return list_experiments(
        task_type=task_type,
        status=status,
        limit=limit,
        offset=offset,
        database_url=_get_db_url(request),
    )


@router.get("/api/fl/rounds/current")
def get_current_fl_round_task_endpoint(
    request: Request,
    client_id: str = Query(..., min_length=1, description="Logical hospital client ID (e.g. client_2)"),
    experiment_id: str | None = Query(default=None, description="Optional experiment ID filter"),
) -> dict[str, Any]:
    """Return the active round assignment and global shared state for a remote hospital worker."""
    _verify_remote_client_token(request)
    mgr = _get_manager(request)
    try:
        return mgr.get_remote_client_task(
            client_id=client_id,
            experiment_id=experiment_id,
            database_url=_get_db_url(request),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/fl/rounds/{experiment_id}/{round_num}/submit")
def submit_remote_fl_round_update_endpoint(
    experiment_id: str,
    round_num: int,
    payload: RemoteClientSubmitRequest,
    request: Request,
) -> dict[str, Any]:
    """Validate and register a remote hospital client's shared parameter update for `round_num`."""
    _verify_remote_client_token(request)
    mgr = _get_manager(request)

    try:
        raw_shared = base64.b64decode(payload.shared_state_b64, validate=True)
        shared_state_dict = deserialize_state_dict_canonical(raw_shared)
        del raw_shared
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid shared_state_b64 canonical payload: {exc}",
        ) from exc

    recomputed_hash = hash_state_dict(shared_state_dict)
    recomp_hex = (
        recomputed_hash.lower()[2:]
        if recomputed_hash.lower().startswith("0x")
        else recomputed_hash.lower()
    )
    provided_hex = payload.update_hash.strip().lower()
    if provided_hex.startswith("0x"):
        provided_hex = provided_hex[2:]
    if recomp_hex != provided_hex:
        raise HTTPException(
            status_code=400,
            detail=(
                f"update_hash mismatch for client '{payload.client_id}': "
                f"expected {recomputed_hash}, got {payload.update_hash}."
            ),
        )

    spectral_proto = None
    if payload.spectral_prototype_b64:
        try:
            raw_proto = base64.b64decode(
                payload.spectral_prototype_b64.encode("ascii"), validate=True
            )
            proto_dict = deserialize_state_dict_canonical(raw_proto)
            spectral_proto = proto_dict.get("spectral_prototype")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid spectral_prototype_b64 canonical payload: {exc}",
            ) from exc

    eval_metrics_dict: dict[str, Any] = {}
    for k, v in (payload.eval_metrics or {}).items():
        try:
            eval_metrics_dict[str(k)] = float(v)
        except (TypeError, ValueError):
            eval_metrics_dict[str(k)] = str(v)

    update_obj = ClientUpdatePayload(
        experiment_id=experiment_id,
        round_num=int(round_num),
        client_id=payload.client_id.strip(),
        task_type="classification",
        algorithm="omnifm",
        num_samples=int(payload.num_samples),
        shared_state_dict=shared_state_dict,
        update_hash=recomputed_hash,
        spectral_prototype=spectral_proto,
        train_loss=float(payload.train_loss),
        task_loss=float(payload.task_loss),
        align_loss=float(payload.align_loss),
        eval_metrics=eval_metrics_dict,
        duration_sec=0.0,
        peak_vram_mb=float(payload.peak_vram_mb),
    )

    try:
        return mgr.submit_remote_client_update(
            experiment_id=experiment_id,
            round_num=int(round_num),
            update=update_obj,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RoundStateTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AggregationValidationError as exc:
        msg = str(exc).lower()
        if "duplicate" in msg or "round" in msg or "state" in msg:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

