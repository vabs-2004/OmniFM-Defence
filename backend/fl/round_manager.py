"""Server Round State Machine (`backend/fl/round_manager.py`, Task 5.4).

Manages the per-round state transitions:
    CREATED -> WAITING_FOR_CLIENTS -> TRAINING -> UPDATES_VALIDATED -> AGGREGATING -> COMPLETED
(with transitions to `FAILED` or `CANCELLED` on errors or cancellation requests),
and prevents duplicate client submissions or duplicate aggregation within any
`(experiment_id, round_num)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from backend.fl.aggregation import (
    AggregationValidationError,
    validate_client_updates_for_aggregation,
    weighted_fedavg,
)
from backend.fl.protocols import (
    ClientUpdatePayload,
    ParameterSharingPolicy,
    RoundState,
)


class RoundStateTransitionError(RuntimeError):
    """Raised on illegal round state transitions or duplicate aggregation attempts."""


_ALLOWED_TRANSITIONS: dict[RoundState, set[RoundState]] = {
    RoundState.CREATED: {
        RoundState.WAITING_FOR_CLIENTS,
        RoundState.TRAINING,
        RoundState.FAILED,
        RoundState.CANCELLED,
    },
    RoundState.WAITING_FOR_CLIENTS: {
        RoundState.TRAINING,
        RoundState.UPDATES_VALIDATED,
        RoundState.FAILED,
        RoundState.CANCELLED,
    },
    RoundState.TRAINING: {
        RoundState.UPDATES_VALIDATED,
        RoundState.FAILED,
        RoundState.CANCELLED,
    },
    RoundState.UPDATES_VALIDATED: {
        RoundState.AGGREGATING,
        RoundState.FAILED,
        RoundState.CANCELLED,
    },
    RoundState.AGGREGATING: {
        RoundState.COMPLETED,
        RoundState.FAILED,
        RoundState.CANCELLED,
    },
    RoundState.COMPLETED: set(),
    RoundState.FAILED: set(),
    RoundState.CANCELLED: set(),
}


class RoundManager:
    """Enforces round lifecycle transitions, client quorum, and single-aggregation rules."""

    def __init__(
        self,
        experiment_id: str,
        round_num: int,
        expected_client_ids: Sequence[str],
        policy: ParameterSharingPolicy,
        require_all_clients: bool = True,
        min_quorum_clients: int | None = None,
    ) -> None:
        if round_num < 1:
            raise ValueError(f"round_num must be >= 1, got {round_num}.")
        if not expected_client_ids:
            raise ValueError("expected_client_ids cannot be empty.")

        self.experiment_id = experiment_id
        self.round_num = int(round_num)
        self.expected_client_ids = tuple(expected_client_ids)
        self.policy = policy
        self.require_all_clients = bool(require_all_clients)
        self.min_quorum = (
            len(self.expected_client_ids)
            if self.require_all_clients
            else max(1, int(min_quorum_clients or 1))
        )

        self.state: RoundState = RoundState.CREATED
        self.aggregated: bool = False
        self._submissions: dict[str, ClientUpdatePayload] = {}

    @property
    def submissions(self) -> list[ClientUpdatePayload]:
        """Return ordered list of accepted client updates."""
        return [
            self._submissions[cid]
            for cid in self.expected_client_ids
            if cid in self._submissions
        ]

    def transition_to(self, next_state: RoundState) -> RoundState:
        """Transition to `next_state` if permitted by the state machine."""
        if next_state == self.state:
            return self.state
        allowed = _ALLOWED_TRANSITIONS.get(self.state, set())
        if next_state not in allowed:
            raise RoundStateTransitionError(
                f"Illegal round state transition for ({self.experiment_id}, round={self.round_num}): "
                f"{self.state.value} -> {next_state.value}."
            )
        self.state = next_state
        return self.state

    def accept_client_update(self, update: ClientUpdatePayload) -> None:
        """Validate and record a client update for this round."""
        if self.aggregated or self.state in (
            RoundState.AGGREGATING,
            RoundState.COMPLETED,
            RoundState.FAILED,
            RoundState.CANCELLED,
        ):
            raise RoundStateTransitionError(
                f"Cannot accept client update for ({self.experiment_id}, round={self.round_num}) "
                f"in state {self.state.value} (aggregated={self.aggregated})."
            )
        if self.state == RoundState.CREATED:
            self.transition_to(RoundState.TRAINING)
        if update.client_id not in self.expected_client_ids:
            raise AggregationValidationError(
                f"Unexpected client_id='{update.client_id}'. Expected one of {self.expected_client_ids}."
            )
        if update.client_id in self._submissions:
            raise AggregationValidationError(
                f"Duplicate update submission from '{update.client_id}' for "
                f"({self.experiment_id}, round={self.round_num})."
            )

        validate_client_updates_for_aggregation(
            updates=[update],
            policy=self.policy,
            expected_experiment_id=self.experiment_id,
            expected_round_num=self.round_num,
        )
        self._submissions[update.client_id] = update

    def check_quorum_met(self) -> bool:
        """Return True if the required client submission count has been reached."""
        if self.require_all_clients:
            return all(cid in self._submissions for cid in self.expected_client_ids)
        return len(self._submissions) >= self.min_quorum

    def validate_and_aggregate(self) -> dict[str, torch.Tensor]:
        """Validate all collected submissions and execute weighted FedAvg exactly once."""
        if self.aggregated:
            raise RoundStateTransitionError(
                f"Duplicate aggregation blocked for ({self.experiment_id}, round={self.round_num})."
            )
        if not self.check_quorum_met():
            raise RoundStateTransitionError(
                f"Incomplete client submissions for ({self.experiment_id}, round={self.round_num}): "
                f"received {len(self._submissions)}/{len(self.expected_client_ids)} "
                f"(require_all_clients={self.require_all_clients}, min_quorum={self.min_quorum})."
            )

        ordered_updates = self.submissions
        validate_client_updates_for_aggregation(
            updates=ordered_updates,
            policy=self.policy,
            expected_experiment_id=self.experiment_id,
            expected_round_num=self.round_num,
        )
        self.transition_to(RoundState.UPDATES_VALIDATED)
        self.transition_to(RoundState.AGGREGATING)

        aggregated_state = weighted_fedavg(
            client_states=[u.shared_state_dict for u in ordered_updates],
            sample_counts=[u.num_samples for u in ordered_updates],
            policy=self.policy,
        )
        self.aggregated = True
        self.transition_to(RoundState.COMPLETED)
        return aggregated_state
