"""Federated Learning Orchestration Package (`backend/fl`)."""

from backend.fl.aggregation import (
    AggregationValidationError,
    filter_shared_state_dict,
    validate_client_updates_for_aggregation,
    weighted_fedavg,
)
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
from backend.fl.server import FederatedLearningServer

__all__ = [
    "AggregationValidationError",
    "AlgorithmType",
    "ClientUpdatePayload",
    "FederatedHospitalClient",
    "FederatedLearningServer",
    "ParameterSharingPolicy",
    "RoundExecutionResult",
    "RoundManager",
    "RoundState",
    "RoundStateTransitionError",
    "build_parameter_sharing_policy",
    "filter_shared_state_dict",
    "validate_client_updates_for_aggregation",
    "weighted_fedavg",
]
