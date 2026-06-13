"""Movement-window GRU/extLQG reach diagnostics.

This module operates on already-materialized rollout arrays. It deliberately
does not load checkpoints or run models; callers supply GRU and extLQG position,
velocity, command, and optional full C&S state arrays from whatever materializer
owns those bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from rlrmp.analysis.cs_game_card import build_canonical_game


SCHEMA_VERSION = "rlrmp.gru_movement_window_diagnostic.v1"
FULL_QRF_TERMS = (
    "total",
    "running_state",
    "terminal_state",
    "command_control",
    "force_filter_state",
    "disturbance_integrator_state",
)
RolloutRole = Literal["gru", "extlqg", "reference", "other"]


@dataclass(frozen=True)
class MovementWindowSpec:
    """Window over rollout arrays and the canonical C&S cost schedule."""

    start_step: int = 0
    n_steps: int = 60
    schedule_start_step: int = 0
    dt: float = 0.01
    label: str = "canonical_movement_window"

    @classmethod
    def canonical(cls) -> "MovementWindowSpec":
        """Return the canonical 60-step C&S movement window."""

        plant, schedule = build_canonical_game()
        return cls(
            start_step=0,
            n_steps=int(schedule.T),
            schedule_start_step=0,
            dt=float(plant.dt),
        )

    @property
    def stop_step(self) -> int:
        """Exclusive stop index in supplied rollout arrays."""

        return self.start_step + self.n_steps

    @property
    def schedule_stop_step(self) -> int:
        """Exclusive stop index in the canonical cost schedule."""

        return self.schedule_start_step + self.n_steps

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable window metadata."""

        return {
            "label": self.label,
            "start_step": int(self.start_step),
            "stop_step": int(self.stop_step),
            "n_steps": int(self.n_steps),
            "schedule_start_step": int(self.schedule_start_step),
            "schedule_stop_step": int(self.schedule_stop_step),
            "dt": float(self.dt),
            "schedule_source": "rlrmp.analysis.cs_game_card.build_canonical_game",
        }


@dataclass(frozen=True)
class ReachRollout:
    """Precomputed rollout arrays for one controller/reference row.

    Arrays use shape ``(..., T, dim)``. Leading axes can represent replicates,
    trials, sampled initial conditions, or any other sample dimensions.
    """

    label: str
    role: RolloutRole
    velocity: Any
    position: Any | None = None
    command: Any | None = None
    states: Any | None = None
    initial_states: Any | None = None
    target_position: Any | None = None
    initial_position: Any | None = None
    reach_direction: Any | None = None
    direction_labels: Any | None = None
    state_basis: Literal["absolute_workspace", "target_centered"] = "absolute_workspace"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_movement_window_diagnostic(
    rollouts: Sequence[ReachRollout | Mapping[str, Any]],
    *,
    window: MovementWindowSpec | None = None,
    reference_role: str = "extlqg",
) -> dict[str, Any]:
    """Summarize movement-window velocity and full-Q/R/Q_f cost diagnostics.

    Args:
        rollouts: Controller/reference rollout records.
        window: Movement-window definition. Defaults to the canonical 60-step
            C&S movement window.
        reference_role: Role used as the default extLQG reference for
            candidate-vs-reference comparisons.

    Returns:
        JSON-compatible diagnostic payload. The diagnostic is audit-only and
        does not imply checkpoint selection.
    """

    if not rollouts:
        raise ValueError("at least one rollout is required")
    window = window or MovementWindowSpec.canonical()
    normalized = [_coerce_rollout(row) for row in rollouts]
    rows = [_summarize_rollout(row, window=window) for row in normalized]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "direction_conditioned_velocity_and_movement_window_full_qrf",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "movement_window": window.to_json(),
        "rows": rows,
        "comparisons": _compare_to_reference(rows, reference_role=reference_role),
    }


def summarize_direction_conditioned_velocity(
    *,
    velocity: Any,
    reach_direction: Any | None = None,
    target_position: Any | None = None,
    initial_position: Any | None = None,
    direction_labels: Any | None = None,
    window: MovementWindowSpec | None = None,
) -> dict[str, Any]:
    """Return reach-direction-conditioned velocity profiles and peak summaries."""

    window = window or MovementWindowSpec.canonical()
    velocity_array = _slice_time(np.asarray(velocity, dtype=np.float64), window=window)
    if velocity_array.ndim < 2 or velocity_array.shape[-1] != 2:
        raise ValueError(
            "velocity must have shape (..., T, 2); "
            f"got {velocity_array.shape}"
        )
    leading_shape = velocity_array.shape[:-2]
    directions = _resolve_directions(
        leading_shape=leading_shape,
        reach_direction=reach_direction,
        target_position=target_position,
        initial_position=initial_position,
    )
    labels = _resolve_direction_labels(
        directions,
        direction_labels=direction_labels,
        leading_shape=leading_shape,
    )
    unit = _unit_vectors(directions)
    lateral_unit = np.stack([-unit[..., 1], unit[..., 0]], axis=-1)
    forward = np.sum(velocity_array * unit[..., None, :], axis=-1)
    lateral = np.sum(velocity_array * lateral_unit[..., None, :], axis=-1)
    flat_forward = forward.reshape(-1, window.n_steps)
    flat_lateral = lateral.reshape(-1, window.n_steps)
    flat_labels = labels.reshape(-1)
    time_s = np.arange(window.n_steps, dtype=np.float64) * float(window.dt)
    by_direction = {}
    for label in sorted({str(label) for label in flat_labels}):
        mask = flat_labels == label
        direction_forward = flat_forward[mask]
        direction_lateral = flat_lateral[mask]
        mean_forward = np.mean(direction_forward, axis=0)
        mean_lateral = np.mean(direction_lateral, axis=0)
        peak_idx = int(np.argmax(mean_forward))
        by_direction[label] = {
            "n_samples": int(direction_forward.shape[0]),
            "time_s": time_s.tolist(),
            "mean_forward_velocity_m_s": mean_forward.tolist(),
            "std_forward_velocity_m_s": np.std(direction_forward, axis=0).tolist(),
            "mean_lateral_velocity_m_s": mean_lateral.tolist(),
            "std_lateral_velocity_m_s": np.std(direction_lateral, axis=0).tolist(),
            "peak_forward_velocity_m_s": float(mean_forward[peak_idx]),
            "time_of_peak_forward_velocity_s": float(time_s[peak_idx]),
            "mean_abs_lateral_velocity_m_s": float(np.mean(np.abs(direction_lateral))),
        }
    return {
        "status": "available",
        "basis": "reach_direction_parallel_lateral",
        "direction_source": _direction_source(
            reach_direction=reach_direction,
            target_position=target_position,
            initial_position=initial_position,
        ),
        "window": window.to_json(),
        "by_direction": by_direction,
    }


def score_movement_window_full_qrf_cost(
    *,
    states: Any,
    commands: Any,
    initial_states: Any,
    target_position: Any | None = None,
    state_basis: Literal["absolute_workspace", "target_centered"] = "absolute_workspace",
    window: MovementWindowSpec | None = None,
) -> dict[str, Any]:
    """Score full-Q/R/Q_f cost terms on the declared movement window."""

    if state_basis not in {"absolute_workspace", "target_centered"}:
        raise ValueError(
            "state_basis must be 'absolute_workspace' or 'target_centered', "
            f"got {state_basis!r}"
        )
    window = window or MovementWindowSpec.canonical()
    _plant, schedule = build_canonical_game()
    if window.schedule_stop_step > int(schedule.T):
        raise ValueError(
            f"window schedule stop {window.schedule_stop_step} exceeds schedule.T={schedule.T}"
        )
    state_array = np.asarray(states, dtype=np.float64)
    command_array = np.asarray(commands, dtype=np.float64)
    initial_array = np.asarray(initial_states, dtype=np.float64)
    if state_array.ndim < 2:
        raise ValueError(f"states must have shape (..., T, state_dim); got {state_array.shape}")
    if command_array.shape[:-2] != state_array.shape[:-2]:
        raise ValueError(
            "commands leading axes must match states; "
            f"got {command_array.shape[:-2]} vs {state_array.shape[:-2]}"
        )
    if state_array.shape[-1] != schedule.Q.shape[-1]:
        raise ValueError(
            f"expected state dim {schedule.Q.shape[-1]}, got {state_array.shape[-1]}"
        )
    if command_array.shape[-1] != schedule.R.shape[-1]:
        raise ValueError(
            f"expected command dim {schedule.R.shape[-1]}, got {command_array.shape[-1]}"
        )
    if state_array.shape[-2] < window.stop_step:
        raise ValueError(
            f"states have {state_array.shape[-2]} time steps, need {window.stop_step}"
        )
    if command_array.shape[-2] < window.stop_step:
        raise ValueError(
            f"commands have {command_array.shape[-2]} time steps, need {window.stop_step}"
        )

    leading_shape = state_array.shape[:-2]
    initial_array = np.broadcast_to(initial_array, (*leading_shape, state_array.shape[-1]))
    pre_state_full = np.concatenate(
        [initial_array[..., None, :], state_array[..., :-1, :]],
        axis=-2,
    )
    x_pre = pre_state_full[..., window.start_step : window.stop_step, :]
    x_terminal = state_array[..., window.stop_step - 1, :]
    if state_basis == "absolute_workspace":
        target = _target_for_leading_shape(target_position, leading_shape=leading_shape)
        x_pre = _goal_centered_vectors(x_pre, target_position=target)
        x_terminal = _goal_centered_vectors(x_terminal, target_position=target)
        state_transform = "subtract target_position from each 8D physical block x/y"
    else:
        state_transform = "none; states are already target-centered"

    q = np.asarray(
        schedule.Q[window.schedule_start_step : window.schedule_stop_step],
        dtype=np.float64,
    )
    r = np.asarray(
        schedule.R[window.schedule_start_step : window.schedule_stop_step],
        dtype=np.float64,
    )
    q_f = np.asarray(schedule.Q_f, dtype=np.float64)
    groups = _state_term_groups(state_array.shape[-1])
    running_state = _state_quadratic_group(x_pre, q, groups["running_state"])
    force_filter = _state_quadratic_group(x_pre, q, groups["force_filter_state"])
    disturbance_integrator = _state_quadratic_group(
        x_pre,
        q,
        groups["disturbance_integrator_state"],
    )
    terminal_state = _terminal_quadratic_group(x_terminal, q_f, groups["running_state"])
    terminal_force = _terminal_quadratic_group(x_terminal, q_f, groups["force_filter_state"])
    terminal_integrator = _terminal_quadratic_group(
        x_terminal,
        q_f,
        groups["disturbance_integrator_state"],
    )
    command_control = np.sum(
        np.einsum(
            "...ti,tij,...tj->...t",
            command_array[..., window.start_step : window.stop_step, :],
            r,
            command_array[..., window.start_step : window.stop_step, :],
        ),
        axis=-1,
    )
    force_filter = force_filter + terminal_force
    disturbance_integrator = disturbance_integrator + terminal_integrator
    total = (
        running_state
        + terminal_state
        + command_control
        + force_filter
        + disturbance_integrator
    )
    term_sum = (
        running_state
        + terminal_state
        + command_control
        + force_filter
        + disturbance_integrator
    )
    return {
        "status": "available",
        "lens": "movement_window_realized_full_qrf",
        "basis": {
            "state_key": "states.mechanics.vector",
            "command_key": "states.net.output or extLQG u_command",
            "state_basis": state_basis,
            "state_transform": state_transform,
            "schedule_source": "rlrmp.analysis.cs_game_card.build_canonical_game",
            "term_split": "coordinate masks over each 8D delay block",
        },
        "window": window.to_json(),
        "total": _summary_with_values(total),
        "running_state": _summary_with_values(running_state),
        "terminal_state": _summary_with_values(terminal_state),
        "command_control": _summary_with_values(command_control),
        "force_filter_state": _summary_with_values(force_filter),
        "disturbance_integrator_state": _summary_with_values(disturbance_integrator),
        "term_sum_delta": _summary_stats(total - term_sum),
    }


def _summarize_rollout(row: ReachRollout, *, window: MovementWindowSpec) -> dict[str, Any]:
    velocity = summarize_direction_conditioned_velocity(
        velocity=row.velocity,
        reach_direction=row.reach_direction,
        target_position=row.target_position,
        initial_position=(
            row.initial_position
            if row.initial_position is not None
            else _initial_position_from_position(row.position)
        ),
        direction_labels=row.direction_labels,
        window=window,
    )
    cost = (
        score_movement_window_full_qrf_cost(
            states=row.states,
            commands=row.command,
            initial_states=row.initial_states,
            target_position=row.target_position,
            state_basis=row.state_basis,
            window=window,
        )
        if row.states is not None and row.command is not None and row.initial_states is not None
        else {
            "status": "not_available",
            "reason": "states, command, and initial_states are required for full-Q/R/Q_f scoring",
        }
    )
    return {
        "label": row.label,
        "role": row.role,
        "metadata": dict(row.metadata),
        "velocity": velocity,
        "full_qrf_cost": cost,
    }


def _compare_to_reference(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_role: str,
) -> list[dict[str, Any]]:
    references = [row for row in rows if str(row.get("role")) == reference_role]
    if not references:
        return []
    reference = references[0]
    comparisons = []
    for row in rows:
        if row is reference:
            continue
        comparisons.append(
            {
                "candidate_label": row["label"],
                "candidate_role": row["role"],
                "reference_label": reference["label"],
                "reference_role": reference["role"],
                "selection_role": "audit_only_not_used_for_checkpoint_selection",
                "velocity_by_direction": _compare_velocity_by_direction(row, reference),
                "full_qrf_cost": _compare_cost_terms(row, reference),
            }
        )
    return comparisons


def _compare_velocity_by_direction(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_velocity = candidate.get("velocity", {})
    reference_velocity = reference.get("velocity", {})
    if (
        not isinstance(candidate_velocity, Mapping)
        or not isinstance(reference_velocity, Mapping)
        or candidate_velocity.get("status") != "available"
        or reference_velocity.get("status") != "available"
    ):
        return {"status": "not_available", "reason": "velocity summaries unavailable"}
    candidate_dirs = candidate_velocity.get("by_direction", {})
    reference_dirs = reference_velocity.get("by_direction", {})
    shared = sorted(set(candidate_dirs).intersection(reference_dirs))
    result: dict[str, Any] = {"status": "available", "directions": {}}
    for label in shared:
        cand = candidate_dirs[label]
        ref = reference_dirs[label]
        cand_profile = np.asarray(cand["mean_forward_velocity_m_s"], dtype=np.float64)
        ref_profile = np.asarray(ref["mean_forward_velocity_m_s"], dtype=np.float64)
        if cand_profile.shape != ref_profile.shape:
            result["directions"][label] = {
                "status": "not_available",
                "reason": "profile shapes differ",
                "candidate_shape": list(cand_profile.shape),
                "reference_shape": list(ref_profile.shape),
            }
            continue
        peak_delta = float(cand["peak_forward_velocity_m_s"] - ref["peak_forward_velocity_m_s"])
        result["directions"][label] = {
            "status": "available",
            "peak_forward_velocity_delta_m_s": peak_delta,
            "peak_forward_velocity_ratio": _safe_ratio(
                cand["peak_forward_velocity_m_s"],
                ref["peak_forward_velocity_m_s"],
            ),
            "forward_velocity_profile_rmse_m_s": float(
                np.sqrt(np.mean((cand_profile - ref_profile) ** 2))
            ),
            "mean_abs_lateral_velocity_delta_m_s": float(
                cand["mean_abs_lateral_velocity_m_s"] - ref["mean_abs_lateral_velocity_m_s"]
            ),
        }
    return result


def _compare_cost_terms(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_cost = candidate.get("full_qrf_cost", {})
    reference_cost = reference.get("full_qrf_cost", {})
    if (
        not isinstance(candidate_cost, Mapping)
        or not isinstance(reference_cost, Mapping)
        or candidate_cost.get("status") != "available"
        or reference_cost.get("status") != "available"
    ):
        return {"status": "not_available", "reason": "full-Q/R/Q_f summaries unavailable"}
    terms = {}
    for term in FULL_QRF_TERMS:
        cand_mean = float(candidate_cost[term]["mean"])
        ref_mean = float(reference_cost[term]["mean"])
        terms[term] = {
            "candidate_mean": cand_mean,
            "reference_mean": ref_mean,
            "delta_mean": cand_mean - ref_mean,
            "ratio_to_reference": _safe_ratio(cand_mean, ref_mean),
        }
    return {"status": "available", "terms": terms}


def _coerce_rollout(row: ReachRollout | Mapping[str, Any]) -> ReachRollout:
    if isinstance(row, ReachRollout):
        return row
    return ReachRollout(**dict(row))


def _slice_time(array: np.ndarray, *, window: MovementWindowSpec) -> np.ndarray:
    if array.shape[-2] < window.stop_step:
        raise ValueError(f"array has {array.shape[-2]} time steps, need {window.stop_step}")
    return array[..., window.start_step : window.stop_step, :]


def _initial_position_from_position(position: Any | None) -> np.ndarray | None:
    if position is None:
        return None
    array = np.asarray(position, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 2:
        return None
    return array[..., 0, :]


def _resolve_directions(
    *,
    leading_shape: tuple[int, ...],
    reach_direction: Any | None,
    target_position: Any | None,
    initial_position: Any | None,
) -> np.ndarray:
    if reach_direction is not None:
        direction = np.asarray(reach_direction, dtype=np.float64)
    elif target_position is not None and initial_position is not None:
        direction = (
            _broadcast_vector(target_position, leading_shape=leading_shape)
            - _broadcast_vector(initial_position, leading_shape=leading_shape)
        )
    else:
        raise ValueError(
            "reach_direction is required unless target_position and initial_position "
            "can define the reach direction"
        )
    return _broadcast_vector(direction, leading_shape=leading_shape)


def _resolve_direction_labels(
    directions: np.ndarray,
    *,
    direction_labels: Any | None,
    leading_shape: tuple[int, ...],
) -> np.ndarray:
    if direction_labels is None:
        flat = directions.reshape(-1, 2)
        labels = np.asarray([_direction_label(vec) for vec in flat], dtype=object)
        return labels.reshape(leading_shape)
    labels = np.asarray(direction_labels, dtype=object)
    return np.broadcast_to(labels, leading_shape)


def _direction_label(vector: np.ndarray) -> str:
    unit = _unit_vectors(np.asarray(vector, dtype=np.float64))
    angle = float(np.degrees(np.arctan2(unit[..., 1], unit[..., 0])))
    if abs(angle) < 1e-6:
        return "+x"
    if abs(angle - 90.0) < 1e-6:
        return "+y"
    if abs(abs(angle) - 180.0) < 1e-6:
        return "-x"
    if abs(angle + 90.0) < 1e-6:
        return "-y"
    return f"{angle:.1f}deg"


def _direction_source(
    *,
    reach_direction: Any | None,
    target_position: Any | None,
    initial_position: Any | None,
) -> str:
    if reach_direction is not None:
        return "explicit_reach_direction"
    if target_position is not None and initial_position is not None:
        return "target_position_minus_initial_position"
    return "unavailable"


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("reach directions must have nonzero norm")
    return array / norm


def _broadcast_vector(value: Any, *, leading_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (2,):
        return np.broadcast_to(array, (*leading_shape, 2))
    return np.broadcast_to(array, (*leading_shape, 2))


def _target_for_leading_shape(value: Any | None, *, leading_shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.zeros((*leading_shape, 2), dtype=np.float64)
    return _broadcast_vector(value, leading_shape=leading_shape)


def _goal_centered_vectors(values: np.ndarray, *, target_position: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.shape[-1] % 8 != 0:
        raise ValueError(f"state dimension {result.shape[-1]} is not divisible by 8")
    target = np.asarray(target_position, dtype=np.float64)
    if target.shape[-1] != 2:
        raise ValueError(f"target_position trailing dimension must be 2, got {target.shape}")
    while target.ndim < result.ndim - 1:
        target = target[..., None, :]
    for start in range(0, result.shape[-1], 8):
        result[..., start : start + 2] -= target
    return result


def _state_term_groups(state_dim: int) -> dict[str, list[int]]:
    if state_dim % 8 != 0:
        raise ValueError(f"state dimension {state_dim} is not divisible by 8")
    groups = {
        "running_state": [],
        "force_filter_state": [],
        "disturbance_integrator_state": [],
    }
    for start in range(0, state_dim, 8):
        groups["running_state"].extend(range(start, start + 4))
        groups["force_filter_state"].extend(range(start + 4, start + 6))
        groups["disturbance_integrator_state"].extend(range(start + 6, start + 8))
    return groups


def _state_quadratic_group(values: np.ndarray, matrices: np.ndarray, indices: Sequence[int]) -> Any:
    idx = np.asarray(indices, dtype=np.int64)
    selected = values[..., idx]
    selected_matrices = matrices[:, idx[:, None], idx]
    terms = np.einsum("...ti,tij,...tj->...t", selected, selected_matrices, selected)
    return np.sum(terms, axis=-1)


def _terminal_quadratic_group(
    values: np.ndarray,
    matrix: np.ndarray,
    indices: Sequence[int],
) -> Any:
    idx = np.asarray(indices, dtype=np.int64)
    selected = values[..., idx]
    selected_matrix = matrix[idx[:, None], idx]
    return np.einsum("...i,ij,...j->...", selected, selected_matrix, selected)


def _summary_with_values(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        **_summary_stats(array),
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _summary_stats(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    flat = array.reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(denominator) or abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


__all__ = [
    "FULL_QRF_TERMS",
    "SCHEMA_VERSION",
    "MovementWindowSpec",
    "ReachRollout",
    "build_movement_window_diagnostic",
    "score_movement_window_full_qrf_cost",
    "summarize_direction_conditioned_velocity",
]
