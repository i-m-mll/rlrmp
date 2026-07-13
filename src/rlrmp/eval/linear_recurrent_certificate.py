"""Evaluation-side provider for linear-recurrent augmented certificates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


LINEAR_RECURRENT_AUGMENTED_PROVIDER = "rlrmp.eval.linear_recurrent_augmented"


def linear_recurrent_augmented_component_kwargs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize numeric ``augmented_linear`` component inputs.

    Feedbax records node states after each graph step. Thus a same-index cached
    mechanics/hidden pair is ``z_t = [x_t - x*_t; h_{t-1}]``: the post-step mechanics
    state and the hidden state that produced it. The next action first updates the
    hidden state from this pair. The candidate controller is the zero-affine
    identity-activation ``LeakyRNNCell`` used by the canonical training base.

    Reference action, transition, value, and Bellman inputs are deliberately required.
    They cannot be inferred honestly from the trained controller or rollout cache. The
    grouped materializer verifies ``evaluation_manifest_id`` against its dependency;
    ``training_manifest_id`` and ``reference_id`` remain explicit caller assertions until
    those identities are represented as structured grouped dependencies.
    """

    cached = _require_mapping(inputs, "cached_evaluation")
    controller = _require_mapping(inputs, "trained_controller")
    dynamics = _require_mapping(inputs, "dynamics")
    reference = _require_mapping(inputs, "reference")
    provenance = _require_mapping(inputs, "provenance")
    source_ids = {
        key: _require_identifier(provenance, key)
        for key in ("evaluation_manifest_id", "training_manifest_id", "reference_id")
    }

    coupled = _require_array(cached, "controller_visible_coupled_states", ndim=3)
    target = _require_array(cached, "target_coupled_states")
    hidden = _require_array(cached, "hidden_states", ndim=3)
    if coupled.shape[:2] != hidden.shape[:2]:
        raise ValueError(
            "linear-recurrent certificate coupled and hidden states must share batch/time axes"
        )
    target = _broadcast_target(target, coupled.shape)
    augmented_states = np.concatenate((coupled - target, hidden), axis=-1)

    if controller.get("use_bias") is not False:
        raise ValueError("linear-recurrent certificate requires use_bias=false")
    if controller.get("readout_use_bias") is not False:
        raise ValueError("linear-recurrent certificate requires readout_use_bias=false")
    if controller.get("use_noise", False) is not False:
        raise ValueError("linear-recurrent certificate requires use_noise=false")
    if controller.get("architecture") != "linear_recurrence":
        raise ValueError("linear-recurrent certificate requires linear_recurrence architecture")
    if controller.get("component_type") != "VanillaRNN":
        raise ValueError("linear-recurrent certificate requires registered VanillaRNN component")
    if controller.get("activation") != "identity":
        raise ValueError("linear-recurrent certificate requires identity activation")

    input_weight = _require_array(controller, "input_weight", ndim=2)
    recurrent_weight = _require_array(controller, "recurrent_weight", ndim=2)
    readout_weight = _require_array(controller, "readout_weight", ndim=2)
    hidden_dim, observation_dim = input_weight.shape
    if recurrent_weight.shape != (hidden_dim, hidden_dim):
        raise ValueError("recurrent_weight must have shape (hidden, hidden)")
    if readout_weight.shape[1] != hidden_dim:
        raise ValueError("readout_weight must have shape (action, hidden)")
    if hidden.shape[-1] != hidden_dim:
        raise ValueError("cached hidden-state width conflicts with trained controller")

    dt = _require_positive_scalar(controller, "dt")
    tau = _require_positive_scalar(controller, "tau")
    alpha = dt / tau
    if not 0.0 < alpha <= 1.0:
        raise ValueError("linear-recurrent certificate requires 0 < dt/tau <= 1")
    effective_input = alpha * input_weight
    effective_recurrence = (1.0 - alpha) * np.eye(
        hidden_dim, dtype=recurrent_weight.dtype
    ) + alpha * recurrent_weight

    state_dim = coupled.shape[-1]
    state_transition = _time_matrix(dynamics, "state_transition", rows=state_dim, cols=state_dim)
    action_input = _time_matrix(
        dynamics, "action_input", rows=state_dim, cols=readout_weight.shape[0]
    )
    observation_map = _time_matrix(
        dynamics,
        "controller_observation_map",
        rows=observation_dim,
        cols=state_dim,
    )
    horizon = coupled.shape[1] - 1
    if horizon < 1:
        raise ValueError("linear-recurrent certificate requires at least two state samples")
    state_transition = _broadcast_time(state_transition, horizon, "state_transition")
    action_input = _broadcast_time(action_input, horizon, "action_input")
    observation_map = _broadcast_time(observation_map, horizon, "controller_observation_map")

    action_dim = readout_weight.shape[0]
    augmented_dim = state_dim + hidden_dim
    effective_input_map = np.einsum("ij,tjk->tik", effective_input, observation_map)
    action_from_state = np.einsum("ij,tjk->tik", readout_weight, effective_input_map)
    action_from_hidden = readout_weight @ effective_recurrence
    candidate_action = np.zeros((horizon, action_dim, augmented_dim), dtype=float)
    candidate_action[:, :, :state_dim] = action_from_state
    candidate_action[:, :, state_dim:] = action_from_hidden
    candidate_transition = np.zeros((horizon, augmented_dim, augmented_dim), dtype=float)
    candidate_transition[:, :state_dim, :state_dim] = state_transition + np.einsum(
        "tij,tjk->tik", action_input, action_from_state
    )
    candidate_transition[:, :state_dim, state_dim:] = np.einsum(
        "tij,jk->tik", action_input, action_from_hidden
    )
    candidate_transition[:, state_dim:, :state_dim] = effective_input_map
    candidate_transition[:, state_dim:, state_dim:] = effective_recurrence

    reference_action = _require_array(
        reference, "reference_augmented_action_sensitivity", ndim=(2, 3)
    )
    reference_transition = _require_array(reference, "reference_transition", ndim=(2, 3))
    candidate_values = _require_array(reference, "candidate_value_matrices", ndim=(2, 3))
    reference_values = _require_array(reference, "reference_value_matrices", ndim=(2, 3))
    bellman_hessian = _require_array(reference, "bellman_hessian", ndim=(2, 3))
    _require_matrix_tail(reference_action, (action_dim, augmented_dim), "reference action")
    _require_matrix_tail(
        reference_transition, (augmented_dim, augmented_dim), "reference transition"
    )
    _require_matrix_tail(candidate_values, (augmented_dim, augmented_dim), "candidate value")
    _require_matrix_tail(reference_values, (augmented_dim, augmented_dim), "reference value")
    _require_matrix_tail(bellman_hessian, (action_dim, action_dim), "Bellman Hessian")

    result: dict[str, Any] = {
        "augmented_states": augmented_states,
        "candidate_augmented_action_sensitivity": candidate_action,
        "reference_augmented_action_sensitivity": reference_action,
        "candidate_transition": candidate_transition,
        "reference_transition": reference_transition,
        "candidate_value_matrices": candidate_values,
        "reference_value_matrices": reference_values,
        "bellman_hessian": bellman_hessian,
        "recurrence_diagnostics": {
            "linear_recurrence": True,
            "basis": (
                "controller_visible_target_relative_post_step_coupled_state;"
                "previous_step_hidden_state"
            ),
            "state_timing": "feedbax_post_step_history_pair",
            "zero_affine": True,
            "identity_activation": True,
            "alpha": alpha,
            "recurrent_spectral_radius": float(
                np.max(np.abs(np.linalg.eigvals(effective_recurrence)))
            ),
            "hidden_dim": hidden_dim,
            "observation_dim": observation_dim,
            "verified_source_ids": {
                "evaluation_manifest_id": source_ids["evaluation_manifest_id"],
            },
            "caller_asserted_source_ids": {
                "training_manifest_id": source_ids["training_manifest_id"],
                "reference_id": source_ids["reference_id"],
            },
        },
        "state_label": "target_relative_post_step_coupled_state_and_previous_hidden",
    }
    for key in ("candidate_actions", "reference_actions", "action_weight"):
        if key in cached:
            result[key] = np.asarray(cached[key])
        elif key in reference:
            result[key] = np.asarray(reference[key])
    return result


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"linear-recurrent certificate requires mapping {key!r}")
    return value


def _require_array(
    parent: Mapping[str, Any], key: str, *, ndim: int | tuple[int, ...] | None = None
) -> np.ndarray:
    if key not in parent:
        raise ValueError(f"linear-recurrent certificate requires numeric input {key!r}")
    value = np.asarray(parent[key])
    allowed = (ndim,) if isinstance(ndim, int) else ndim
    if allowed is not None and value.ndim not in allowed:
        raise ValueError(f"{key} must have ndim in {allowed}, got shape {value.shape}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{key} must be numeric")
    return value


def _require_positive_scalar(parent: Mapping[str, Any], key: str) -> float:
    if key not in parent:
        raise ValueError(f"linear-recurrent certificate requires scalar {key!r}")
    value = float(parent[key])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _require_identifier(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"linear-recurrent certificate requires identifier {key!r}")
    return value


def _broadcast_target(target: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if target.ndim == 1:
        target = target[None, None, :]
    elif target.ndim == 2:
        target = target[None, :, :]
    try:
        return np.broadcast_to(target, shape)
    except ValueError as exc:
        raise ValueError(
            "target_coupled_states must broadcast to controller-visible coupled states"
        ) from exc


def _time_matrix(parent: Mapping[str, Any], key: str, *, rows: int, cols: int) -> np.ndarray:
    value = _require_array(parent, key, ndim=(2, 3))
    _require_matrix_tail(value, (rows, cols), key)
    return value[None, ...] if value.ndim == 2 else value


def _broadcast_time(value: np.ndarray, horizon: int, name: str) -> np.ndarray:
    if value.shape[0] not in (1, horizon):
        raise ValueError(f"{name} time axis must be 1 or {horizon}, got {value.shape[0]}")
    return np.broadcast_to(value, (horizon, *value.shape[1:]))


def _require_matrix_tail(value: np.ndarray, shape: tuple[int, int], name: str) -> None:
    if value.shape[-2:] != shape:
        raise ValueError(f"{name} trailing shape must be {shape}, got {value.shape[-2:]}")


__all__ = [
    "LINEAR_RECURRENT_AUGMENTED_PROVIDER",
    "linear_recurrent_augmented_component_kwargs",
]
