"""Weighted FedAvg Aggregation for Compatible Shared Parameters (`backend/fl/aggregation.py`, Task 5.3).

Computes sample-weighted federated averaging:
    global_param[k] = sum_i (n_i * client_param_i[k]) / sum_i n_i

Enforces strict parameter-sharing validation:
1. Requires at least one valid client submission with `num_samples > 0`.
2. Rejects any tensor containing NaN or Inf values.
3. Excludes personalized parameters (`client_suffix_prompt`, FedPer local heads)
   and incompatible classification heads (`heads.*`) declared in `ParameterSharingPolicy`.
4. Verifies that every shared parameter has identical shape across all participating clients.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from backend.fl.protocols import ClientUpdatePayload, ParameterSharingPolicy
from backend.omnifm.psp import PrefixSuffixSpectralPrompting


class AggregationValidationError(ValueError):
    """Raised when client updates fail pre-aggregation validation."""


def filter_shared_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    policy: ParameterSharingPolicy,
) -> dict[str, torch.Tensor]:
    """Filter a full or partial `state_dict` to retain only policy-approved shared parameters/buffers."""
    filtered: dict[str, torch.Tensor] = {}
    excluded = policy.excluded_from_aggregation

    for name, tensor in state_dict.items():
        if name in excluded:
            continue
        if PrefixSuffixSpectralPrompting.is_personalized_parameter(name):
            continue
        if any(name.startswith(ex.rsplit(".", 1)[0] + ".") for ex in policy.incompatible_head_param_names):
            continue
        if name in policy.shared_param_names or not (
            name in policy.personalized_param_names or name in policy.incompatible_head_param_names
        ):
            filtered[name] = tensor.detach().cpu().clone()
    return filtered


def validate_client_updates_for_aggregation(
    updates: Sequence[ClientUpdatePayload],
    policy: ParameterSharingPolicy,
    expected_experiment_id: str | None = None,
    expected_round_num: int | None = None,
) -> None:
    """Validate client identities, metadata, sample counts, parameter shapes, and finiteness."""
    if not updates:
        raise AggregationValidationError("Cannot aggregate an empty list of client updates.")

    seen_clients: set[str] = set()
    for upd in updates:
        if upd.client_id in seen_clients:
            raise AggregationValidationError(
                f"Duplicate client submission detected for client_id='{upd.client_id}'."
            )
        seen_clients.add(upd.client_id)

        if expected_experiment_id is not None and upd.experiment_id != expected_experiment_id:
            raise AggregationValidationError(
                f"Client '{upd.client_id}' experiment_id='{upd.experiment_id}' "
                f"does not match expected '{expected_experiment_id}'."
            )
        if expected_round_num is not None and int(upd.round_num) != int(expected_round_num):
            raise AggregationValidationError(
                f"Client '{upd.client_id}' round_num={upd.round_num} "
                f"does not match expected round {expected_round_num}."
            )
        if upd.task_type != policy.task_type:
            raise AggregationValidationError(
                f"Client '{upd.client_id}' task_type='{upd.task_type}' "
                f"does not match policy task_type='{policy.task_type}'."
            )
        if int(upd.num_samples) <= 0:
            raise AggregationValidationError(
                f"Client '{upd.client_id}' reported invalid sample count {upd.num_samples}; must be > 0."
            )
        if not upd.shared_state_dict:
            raise AggregationValidationError(
                f"Client '{upd.client_id}' submitted an empty shared_state_dict."
            )

        for key, tensor in upd.shared_state_dict.items():
            if key in policy.excluded_from_aggregation or PrefixSuffixSpectralPrompting.is_personalized_parameter(key):
                raise AggregationValidationError(
                    f"Client '{upd.client_id}' included excluded personalized/incompatible parameter '{key}' in shared_state_dict."
                )
            if not torch.isfinite(tensor).all():
                raise AggregationValidationError(
                    f"Client '{upd.client_id}' parameter '{key}' contains non-finite (NaN/Inf) values."
                )
            if key in policy.parameter_shapes:
                expected_shape = policy.parameter_shapes[key]
                if tuple(int(d) for d in tensor.shape) != expected_shape:
                    raise AggregationValidationError(
                        f"Client '{upd.client_id}' parameter '{key}' shape {tuple(tensor.shape)} "
                        f"does not match expected shape {expected_shape}."
                    )


def weighted_fedavg(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    sample_counts: Sequence[int],
    policy: ParameterSharingPolicy | None = None,
) -> dict[str, torch.Tensor]:
    """Compute sample-weighted FedAvg across compatible client state_dicts.

    Args:
        client_states: Sequence of state_dicts from participating clients.
        sample_counts: Positive integer sample counts `n_i` corresponding to each client.
        policy: Optional `ParameterSharingPolicy` to enforce exclusion of personalized
            and incompatible head parameters.

    Returns:
        Aggregated shared state_dict on CPU.
    """
    if len(client_states) == 0 or len(client_states) != len(sample_counts):
        raise AggregationValidationError(
            "client_states and sample_counts must be non-empty and have equal length."
        )

    weights = [int(n) for n in sample_counts]
    if any(n <= 0 for n in weights):
        raise AggregationValidationError(
            f"All client sample_counts must be > 0, got {weights}."
        )
    total_samples = float(sum(weights))

    # Determine eligible keys from the first client state without cloning all tensors
    def _is_eligible_key(name: str) -> bool:
        if PrefixSuffixSpectralPrompting.is_personalized_parameter(name):
            return False
        if policy is None:
            return True
        if name in policy.excluded_from_aggregation:
            return False
        if any(
            name.startswith(ex.rsplit(".", 1)[0] + ".")
            for ex in policy.incompatible_head_param_names
        ):
            return False
        return name in policy.shared_param_names or not (
            name in policy.personalized_param_names
            or name in policy.incompatible_head_param_names
        )

    eligible_keys = [k for k in client_states[0].keys() if _is_eligible_key(k)]
    if not eligible_keys:
        raise AggregationValidationError("No eligible shared parameters found to aggregate.")

    expected_key_set = set(eligible_keys)
    for idx, st in enumerate(client_states):
        st_keys = {k for k in st.keys() if _is_eligible_key(k)}
        if st_keys != expected_key_set:
            missing = expected_key_set - st_keys
            extra = st_keys - expected_key_set
            raise AggregationValidationError(
                f"Client index {idx} shared parameter keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}."
            )
        for k in eligible_keys:
            tensor = st[k]
            ref_shape = client_states[0][k].shape
            if tensor.shape != ref_shape:
                raise AggregationValidationError(
                    f"Shape mismatch on parameter '{k}' at client index {idx}: "
                    f"{tuple(tensor.shape)} vs {tuple(ref_shape)}."
                )
            if not bool(torch.isfinite(tensor).all().item()):
                raise AggregationValidationError(
                    f"Non-finite (NaN/Inf) values in parameter '{k}' at client index {idx}."
                )

    aggregated: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key in eligible_keys:
            ref_tensor = client_states[0][key]
            if ref_tensor.is_floating_point():
                acc = torch.zeros(ref_tensor.shape, dtype=torch.float32, device="cpu")
                for st, n_i in zip(client_states, weights):
                    w = float(n_i) / total_samples
                    t_cpu = st[key].detach().to(device="cpu", dtype=torch.float32)
                    acc.add_(t_cpu, alpha=w)
                aggregated[key] = acc.to(dtype=ref_tensor.dtype).contiguous()
            else:
                # Integer buffers (e.g. BatchNorm num_batches_tracked): take max across clients
                max_buf = client_states[0][key].detach().to(device="cpu").clone()
                for st in client_states[1:]:
                    torch.maximum(
                        max_buf,
                        st[key].detach().to(device="cpu"),
                        out=max_buf,
                    )
                aggregated[key] = max_buf.to(dtype=ref_tensor.dtype).contiguous()

    return aggregated

