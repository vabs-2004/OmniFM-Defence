"""OmniFM Blockchain Federated Learning FastAPI Application (`backend/main.py`, Task 6.2)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.endpoints import (
    audit_router,
    predict_router,
    system_router,
    training_router,
)
from backend.blockchain.web3_connector import Web3Connector
from backend.config import settings
from backend.database.db import init_db
from backend.services.experiment_service import (
    TrainingJobManager,
    ensure_default_hospitals,
    job_manager,
)


def create_app(
    database_url: str | None = None,
    web3_connector: Web3Connector | None = None,
    job_manager_instance: TrainingJobManager | None = None,
    job_manager: TrainingJobManager | None = None,
    settings: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    from backend.config import settings as default_settings

    active_settings = settings or default_settings
    db_url = database_url or active_settings.database_url

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        init_db(app_instance.state.database_url)
        ensure_default_hospitals(
            cfg=getattr(app_instance.state, "settings", default_settings),
            database_url=app_instance.state.database_url,
        )
        yield

    application = FastAPI(
        title="OmniFM Blockchain Federated Medical AI API",
        description=(
            "Local research prototype for frequency-aware federated medical foundation models "
            "(OmniFM: SpectralEncoder, GSKR, ECA, PSP, SPAlign) with Ethereum/Ganache audit trails."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    application.state.database_url = db_url
    application.state.settings = active_settings
    application.state.job_manager = job_manager_instance or job_manager
    if web3_connector is not None:
        application.state.web3_connector = web3_connector

    raw_origins = active_settings.cors_origins
    if isinstance(raw_origins, str):
        cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    else:
        cors_origins = list(raw_origins or [])
    if not cors_origins:
        cors_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(system_router)
    application.include_router(training_router)
    application.include_router(predict_router)
    application.include_router(audit_router)

    init_db(db_url)
    ensure_default_hospitals(cfg=settings, database_url=db_url)

    return application


app = create_app()
