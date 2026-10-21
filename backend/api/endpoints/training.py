"""Asynchronous Federated Training Control and Experiment Endpoints (`backend/api/endpoints/training.py`, Tasks 6.3 & 6.5)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.blockchain.web3_connector import Web3Connector
from backend.config import settings
from backend.services.artifact_service import CheckpointNotFoundError
from backend.services.experiment_service import (
    ConcurrentTrainingJobError,
    InvalidTrainingStateError,
    TrainingJobManager,
    job_manager,
    list_experiments,
)

router = APIRouter(tags=["Experiments & Training"])


class TrainingStartRequest(BaseModel):
    """Request schema for `POST /api/training/start`."""

    task_type: str = Field(..., description="classification | segmentation | superresolution | fusion")
    num_rounds: int = Field(default=20, ge=1, le=100, description="Number of federated rounds")
    config: dict[str, Any] = Field(default_factory=dict, description="Algorithm, batch size, ablation flags")
    training_id: str | None = Field(default=None, description="Optional explicit experiment identifier")
    run_async: bool = Field(default=True, description="Run training asynchronously in background thread")


class TrainingResumeRequest(BaseModel):
    """Optional request body for `POST /api/training/{training_id}/resume`."""

    run_async: bool = Field(default=True, description="Resume asynchronously in background thread")


def _get_db_url(request: Request) -> str:
    return getattr(request.app.state, "database_url", None) or settings.database_url


def _get_web3(request: Request) -> Web3Connector | None:
    if hasattr(request.app.state, "web3_connector"):
        return request.app.state.web3_connector
    try:
        return Web3Connector(cfg=settings)
    except Exception:
        return None


def _get_manager(request: Request) -> TrainingJobManager:
    return getattr(request.app.state, "job_manager", None) or job_manager


@router.post("/api/training/start")
def start_training_endpoint(payload: TrainingStartRequest, request: Request) -> dict[str, Any]:
    """Validate configuration and create an asynchronous federated training job."""
    mgr = _get_manager(request)
    try:
        return mgr.start_training(
            task_type=payload.task_type,
            num_rounds=payload.num_rounds,
            config=payload.config,
            training_id=payload.training_id,
            cfg=settings,
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
            cfg=settings,
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
