"""API Endpoint Routers Package (`backend/api/endpoints/__init__.py`)."""

from backend.api.endpoints.audit import router as audit_router
from backend.api.endpoints.predict import router as predict_router
from backend.api.endpoints.system import router as system_router
from backend.api.endpoints.training import router as training_router

__all__ = [
    "audit_router",
    "predict_router",
    "system_router",
    "training_router",
]
