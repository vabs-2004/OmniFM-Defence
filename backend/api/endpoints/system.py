"""System Health and Hospital Client Endpoints (`backend/api/endpoints/system.py`, Task 6.2)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from backend.blockchain.web3_connector import Web3Connector
from backend.config import settings
from backend.database.db import session_scope
from backend.services.experiment_service import (
    list_hospital_clients,
    register_hospital_client,
)
from backend.utils.logging import get_system_health, sanitize_log_text

router = APIRouter(tags=["System & Clients"])


class HospitalRegisterRequest(BaseModel):
    """Request schema for `POST /api/hospital/register`."""

    model_config = ConfigDict(extra="allow")

    client_id: str = Field(..., min_length=1, description="Logical client identifier, e.g., client_1")
    hospital_name: str = Field(..., min_length=1, description="Human-readable hospital name")
    modality: str = Field(default="multi-modality", description="Supported imaging modalities")


def _get_db_url(request: Request) -> str:
    return getattr(request.app.state, "database_url", None) or settings.database_url


def _get_web3_with_diagnostic(request: Request) -> tuple[Web3Connector | None, str | None]:
    if hasattr(request.app.state, "web3_connector"):
        return request.app.state.web3_connector, None
    try:
        return Web3Connector(cfg=settings), None
    except Exception as exc:
        return None, sanitize_log_text(str(exc))


@router.get("/api/health")
def health_check(request: Request) -> dict[str, Any]:
    """Return API, SQLite database, GPU/VRAM, and Ganache blockchain status without exposing secrets."""
    db_url = _get_db_url(request)
    db_ok = False
    db_error: str | None = None
    try:
        with session_scope(db_url) as session:
            session.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        db_error = sanitize_log_text(str(exc))

    w3, chain_error = _get_web3_with_diagnostic(request)
    chain_connected = False
    contract_deployed = False
    observed_chain_id: int | None = None
    contract_owner: str | None = None
    server_wallet_is_owner: bool | None = None

    if w3 is not None:
        try:
            chain_connected = bool(w3.is_connected())
        except Exception as exc:
            chain_connected = False
            chain_error = sanitize_log_text(str(exc))

        if chain_connected:
            try:
                if isinstance(w3, Web3Connector):
                    info = w3.verify_chain_and_contract()
                    observed_chain_id = int(info["chain_id"])
                    contract_deployed = int(info.get("bytecode_len", 0)) > 0
                    if contract_deployed:
                        contract_owner = w3.get_contract_owner()
                        srv_addr = w3.get_configured_server_address()
                        if srv_addr and contract_owner:
                            server_wallet_is_owner = (
                                srv_addr.lower() == contract_owner.lower()
                            )
                elif hasattr(w3, "is_contract_deployed"):
                    contract_deployed = bool(w3.is_contract_deployed())
            except Exception as exc:
                contract_deployed = False
                chain_error = sanitize_log_text(str(exc))

    sys_health = get_system_health(cfg=settings)
    return {
        "status": "ok" if db_ok else "degraded",
        "api": "online",
        "database": {
            "connected": db_ok,
            "engine": "sqlite",
            "error": db_error,
        },
        "gpu": sys_health.get("gpu", {}),
        "blockchain": {
            "rpc_url": settings.ganache_url,
            "expected_chain_id": settings.ganache_chain_id,
            "chain_id": observed_chain_id,
            "connected": chain_connected,
            "contract_deployed": contract_deployed,
            "contract_address": settings.contract_address or None,
            "contract_owner": contract_owner,
            "server_wallet_is_owner": server_wallet_is_owner,
            "audit_enabled": settings.blockchain_audit_enabled,
            "error": chain_error,
        },
        "datasets": sys_health.get("datasets", {}),
    }


@router.post("/api/hospital/register")
def register_hospital(payload: HospitalRegisterRequest, request: Request) -> dict[str, Any]:
    """Register a logical hospital client and its configured wallet address.

    Never accepts private keys in the request body.
    """
    extra = payload.model_extra or {}
    forbidden_keys = {"private_key", "privateKey", "secret_key", "mnemonic"}
    if any(k in extra for k in forbidden_keys):
        raise HTTPException(
            status_code=400,
            detail="Security violation: Private keys must never be sent in the request body.",
        )

    w3, _ = _get_web3_with_diagnostic(request)
    try:
        rec = register_hospital_client(
            client_id=payload.client_id,
            hospital_name=payload.hospital_name,
            modality=payload.modality,
            cfg=settings,
            web3_connector=w3,
            database_url=_get_db_url(request),
        )
        return rec
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/hospital")
def get_hospitals(request: Request) -> dict[str, Any]:
    """List registered logical hospital clients without exposing private keys."""
    clients = list_hospital_clients(cfg=settings, database_url=_get_db_url(request))
    return {"total": len(clients), "clients": clients}
