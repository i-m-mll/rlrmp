"""Science reducers for cached C&S GRU evaluation states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np

from rlrmp.analysis.math.summary_stats import summary_stats
from rlrmp.eval.checkpoint_selection import ReplicateCheckpointSelection
from rlrmp.model.feedback_descriptors import (
    DESCRIPTOR_PAYLOAD_KEY,
    resolve_controller_feedback_view_from_gru_input,
)
from rlrmp.runtime.spec_migrations import GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION

SCHEMA_VERSION = GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION
DEFAULT_JACOBIAN_TIMEPOINTS = ("first", "peak_forward_velocity", "terminal")
CONTROLLER_FEEDBACK_SCALE_STATISTIC = "p95_norm"
GATE_SATURATION_LOW = 0.05
GATE_SATURATION_HIGH = 0.95
SIGN_CHANGE_EPS = 1e-6
_summary_stats = partial(summary_stats, quantiles=(0.05, 0.5, 0.95))


@dataclass(frozen=True)
class RolloutEvaluation:
    """Host- or device-backed cached rollout arrays for one GRU run."""

    position: Any
    velocity: Any
    command: Any
    hidden: Any
    gru_input: Any
    initial_position: Any
    initial_velocity: Any
    target_position: Any
    dt: float
    checkpoint_selection: tuple[ReplicateCheckpointSelection, ...] = ()


def summarize_rollout_behavior(evaluation: RolloutEvaluation) -> dict[str, Any]:
    """Return JSON-compatible rollout behavior metrics for one evaluation."""

    command = jnp.asarray(evaluation.command, dtype=jnp.float64)
    position = jnp.asarray(evaluation.position, dtype=jnp.float64)
    velocity = jnp.asarray(evaluation.velocity, dtype=jnp.float64)
    hidden = jnp.asarray(evaluation.hidden, dtype=jnp.float64)
    initial_position = jnp.asarray(evaluation.initial_position, dtype=jnp.float64)
    initial_velocity = jnp.asarray(evaluation.initial_velocity, dtype=jnp.float64)
    target_position = jnp.asarray(evaluation.target_position, dtype=jnp.float64)

    command_norm = jnp.linalg.norm(command, axis=-1)
    first_five = command_norm[..., : min(5, command_norm.shape[-1])]
    command_jerk = jnp.diff(command, n=2, axis=2)
    command_jerk_norm = jnp.linalg.norm(command_jerk, axis=-1)
    terminal_error = position[:, :, -1, :] - target_position[None, :, -1, :]
    endpoint_error = jnp.linalg.norm(terminal_error, axis=-1)
    terminal_speed = jnp.linalg.norm(velocity[:, :, -1, :], axis=-1)
    hidden_norm = jnp.linalg.norm(hidden, axis=-1)

    full_position = _prepend_initial(initial_position, position)
    full_velocity = _prepend_initial(initial_velocity, velocity)
    overshoot = _overshoot_along_reach(
        position=full_position,
        initial_position=initial_position,
        target_position=target_position[:, -1, :],
    )
    forward_velocity = _forward_velocity_along_reach(
        velocity=full_velocity,
        initial_position=initial_position,
        target_position=target_position[:, -1, :],
    )
    sign_changes = _post_peak_sign_changes(forward_velocity)
    velocity_summary = _velocity_profile_summary(forward_velocity, evaluation.dt)

    return {
        "command_norm": _summary_stats(command_norm),
        "first_five_step_command_norm": _summary_stats(first_five),
        "command_jerk_norm": _summary_stats(command_jerk_norm),
        "endpoint_error_m": _summary_stats(endpoint_error),
        "terminal_speed_m_s": _summary_stats(terminal_speed),
        "overshoot_m": _summary_stats(overshoot),
        "post_peak_forward_velocity_sign_changes": _summary_stats(sign_changes),
        "velocity_profile": velocity_summary,
        "hidden_state_norm": _summary_stats(hidden_norm),
        "per_replicate": _per_replicate_behavior(
            command_norm=command_norm,
            first_five=first_five,
            command_jerk_norm=command_jerk_norm,
            endpoint_error=endpoint_error,
            terminal_speed=terminal_speed,
            overshoot=overshoot,
            sign_changes=sign_changes,
            hidden_norm=hidden_norm,
            forward_velocity=forward_velocity,
            dt=evaluation.dt,
        ),
    }


def summarize_gru_gates(gru_cell: Any, evaluation: RolloutEvaluation) -> dict[str, Any]:
    """Summarize reset/update/candidate distributions from Equinox GRU equations."""

    gates = compute_gru_gate_arrays(gru_cell, evaluation.gru_input, evaluation.hidden)
    return {
        "equations": "Equinox GRUCell reset/update/candidate gates evaluated on rollout states",
        "reset_gate": _gate_summary(gates["reset"], bounded=True),
        "update_gate": _gate_summary(gates["update"], bounded=True),
        "candidate_activation": _gate_summary(gates["candidate"], bounded=False),
        "per_replicate": [
            {
                "replicate": int(rep_idx),
                "reset_gate": _gate_summary(gates["reset"][rep_idx], bounded=True),
                "update_gate": _gate_summary(gates["update"][rep_idx], bounded=True),
                "candidate_activation": _gate_summary(
                    gates["candidate"][rep_idx],
                    bounded=False,
                ),
            }
            for rep_idx in range(gates["reset"].shape[0])
        ],
    }


def summarize_controller_feedback_scales(
    evaluation: RolloutEvaluation,
    *,
    run_id: str | None = None,
    checkpoint_policy: str | None = None,
    statistic: str = CONTROLLER_FEEDBACK_SCALE_STATISTIC,
    feedback_descriptor_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize rollout-derived scales for controller-visible feedback channels.

    Descriptor resolution owns the feedback-vector basis and any legacy adoption
    needed for historical GRU inputs that did not serialize descriptor records.
    """

    if statistic != CONTROLLER_FEEDBACK_SCALE_STATISTIC:
        raise ValueError(f"unsupported feedback scale statistic {statistic!r}")

    gru_input = jnp.asarray(evaluation.gru_input, dtype=jnp.float64)
    if gru_input.ndim < 4:
        raise ValueError("evaluation.gru_input must have shape replicate/trial/time/feature")
    input_dim = int(gru_input.shape[-1])
    try:
        feedback_view = resolve_controller_feedback_view_from_gru_input(
            gru_input,
            payload=feedback_descriptor_payload,
            source="gru_evaluation_diagnostics.controller_feedback_scales",
        )
    except ValueError as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "run_id": run_id,
            "checkpoint_policy": checkpoint_policy,
            "gru_input_dim": input_dim,
            "feedback_dim": input_dim,
            "components": {},
        }

    components: dict[str, Any] = {}
    for component in feedback_view.iter_components():
        values = component.values
        if values is None:
            raise ValueError("resolved controller feedback view did not bind values")
        norm = jnp.linalg.norm(values, axis=-1)
        abs_values = jnp.abs(values)
        components[component.component_id] = {
            "descriptor_id": component.descriptor_id,
            "label": component.label,
            "units": component.units,
            "feedback_basis_indices": list(
                range(component.slice.start, component.slice.stop, component.slice.step)
            ),
            "gru_input_indices": list(component.absolute_indices),
            "rms_norm": float(jnp.sqrt(jnp.mean(jnp.square(norm)))),
            "p95_norm": float(jnp.quantile(norm.reshape(-1), 0.95)),
            "per_component_rms": [
                float(jnp.sqrt(jnp.mean(jnp.square(abs_values[..., idx]))))
                for idx in range(abs_values.shape[-1])
            ],
            "per_component_p95_abs": [
                float(jnp.quantile(abs_values[..., idx].reshape(-1), 0.95))
                for idx in range(abs_values.shape[-1])
            ],
        }
        components[component.component_id]["reference_scale"] = float(
            components[component.component_id][statistic]
        )
        components[component.component_id]["reference_scale_statistic"] = statistic

    return {
        "status": "available",
        "schema_version": "rlrmp.controller_feedback_scales.v1",
        DESCRIPTOR_PAYLOAD_KEY: dict(feedback_view.payload),
        "descriptor_basis_hash": feedback_view.descriptor_basis_hash,
        "run_id": run_id,
        "checkpoint_policy": checkpoint_policy,
        "source": "nominal_selected_checkpoint_rollouts.states.net.input.controller_feedback",
        "feedback_basis": feedback_view.basis_id,
        "gru_input_dim": input_dim,
        "feedback_dim": feedback_view.feedback_dim,
        "feedback_start_index": feedback_view.start_index,
        "statistic": statistic,
        "scale_rule": "amplitude = component.reference_scale * level_fraction_of_reach",
        "components": components,
    }


def compute_gru_gate_arrays(
    gru_cell: Any,
    gru_input: Any,
    hidden: Any,
) -> dict[str, Any]:
    """Return GRU gate arrays with shape ``(replicate, trial, time, hidden)``."""

    gru_input = jnp.asarray(gru_input, dtype=jnp.float64)
    hidden = jnp.asarray(hidden, dtype=jnp.float64)
    n_replicates = int(hidden.shape[0])
    h_prev = _previous_hidden(hidden)
    gate_rows = []
    for rep_idx in range(n_replicates):
        cell = _select_replicate_tree(gru_cell, rep_idx, n_replicates)
        gate_rows.append(_compute_single_replicate_gates(cell, gru_input[rep_idx], h_prev[rep_idx]))
    return {
        key: jnp.stack([row[key] for row in gate_rows], axis=0)
        for key in ("reset", "update", "candidate")
    }


def summarize_gru_jacobians(
    gru_cell: Any,
    evaluation: RolloutEvaluation,
    *,
    timepoint_policy: Sequence[str] = DEFAULT_JACOBIAN_TIMEPOINTS,
) -> dict[str, Any]:
    """Summarize sampled hidden-to-hidden recurrent Jacobians for a GRU rollout."""

    n_replicates = int(evaluation.hidden.shape[0])
    h_prev = _previous_hidden(evaluation.hidden)
    full_velocity = _prepend_initial(evaluation.initial_velocity, evaluation.velocity)
    forward_velocity = _forward_velocity_along_reach(
        velocity=full_velocity,
        initial_position=evaluation.initial_position,
        target_position=evaluation.target_position[:, -1, :],
    )[:, :, 1:]
    samples: list[dict[str, Any]] = []
    for rep_idx in range(n_replicates):
        cell = _select_replicate_tree(gru_cell, rep_idx, n_replicates)
        sample_times = _jacobian_sample_times(
            forward_velocity[rep_idx, 0],
            evaluation.hidden.shape[2],
            timepoint_policy=timepoint_policy,
        )
        for label, time_idx in sample_times:
            x_t = jnp.asarray(evaluation.gru_input[rep_idx, 0, time_idx])
            h_t = jnp.asarray(h_prev[rep_idx, 0, time_idx])
            jacobian = np.asarray(jax.jacfwd(lambda h: cell(x_t, h))(h_t), dtype=np.float64)
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            eigenvalues = np.linalg.eigvals(jacobian)
            samples.append(
                {
                    "replicate": int(rep_idx),
                    "trial": 0,
                    "time_index": int(time_idx),
                    "timepoint": label,
                    "frobenius_norm": float(np.linalg.norm(jacobian, ord="fro")),
                    "spectral_norm": float(singular_values[0]) if singular_values.size else 0.0,
                    "spectral_radius": float(np.max(np.abs(eigenvalues)))
                    if eigenvalues.size
                    else 0.0,
                }
            )

    return {
        "status": "sampled",
        "sample_policy": (
            "hidden-to-hidden Jacobian d h_t / d h_{t-1}; trial 0 only; "
            f"timepoints={tuple(timepoint_policy)}"
        ),
        "n_samples": len(samples),
        "frobenius_norm": _summary_stats([sample["frobenius_norm"] for sample in samples]),
        "spectral_norm": _summary_stats([sample["spectral_norm"] for sample in samples]),
        "spectral_radius": _summary_stats([sample["spectral_radius"] for sample in samples]),
        "samples": samples,
    }


def diagnostic_definitions() -> dict[str, str]:
    """Return schema-level metric definitions."""

    return {
        "command": "controller output before efferent/motor-channel noise, from state.net.output",
        "command_norm": "Euclidean norm of command at every evaluated time step",
        "first_five_step_command_norm": (
            "command_norm restricted to the first min(5, n_time_steps) evaluated samples"
        ),
        "command_jerk_norm": (
            "Euclidean norm of the second finite difference of command along time"
        ),
        "endpoint_error_m": "Euclidean distance between final effector position and final target",
        "terminal_speed_m_s": "Euclidean norm of final effector velocity",
        "overshoot_m": (
            "maximum positive projection past the final target along the initial-to-target "
            "reach axis; zero when the trajectory never crosses beyond the target"
        ),
        "post_peak_forward_velocity_sign_changes": (
            "number of sign flips in reach-axis velocity after its per-trial peak, "
            f"ignoring samples with absolute value <= {SIGN_CHANGE_EPS}"
        ),
        "hidden_state_norm": "Euclidean norm of GRU hidden state at every evaluated time step",
        "controller_feedback_scales": (
            "RMS and p95-norm scales computed from nominal rollout "
            "states.net.input trailing feedback channels. The reference scale is "
            f"{CONTROLLER_FEEDBACK_SCALE_STATISTIC}."
        ),
        "gru_gate_saturation": (
            f"bounded gate saturation uses value < {GATE_SATURATION_LOW} and "
            f"value > {GATE_SATURATION_HIGH}; candidate saturation uses "
            f"abs(value) > {GATE_SATURATION_HIGH}"
        ),
        "local_recurrent_jacobian": (
            "sampled hidden-to-hidden Jacobian of the GRU cell at rollout states; "
            "not a standard-certificate transition metric"
        ),
    }


def _compute_single_replicate_gates(
    cell: Any,
    x: Any,
    h_prev: Any,
) -> dict[str, Any]:
    weight_ih = jnp.asarray(cell.weight_ih, dtype=jnp.float64)
    weight_hh = jnp.asarray(cell.weight_hh, dtype=jnp.float64)
    bias = jnp.asarray(cell.bias if cell.use_bias else 0.0, dtype=jnp.float64)
    bias_n = jnp.asarray(cell.bias_n if cell.use_bias else 0.0, dtype=jnp.float64)
    flat_x = x.reshape((-1, x.shape[-1]))
    flat_h = h_prev.reshape((-1, h_prev.shape[-1]))
    igates = jnp.split(flat_x @ weight_ih.T + bias, 3, axis=-1)
    hgates = jnp.split(flat_h @ weight_hh.T, 3, axis=-1)
    reset = _sigmoid(igates[0] + hgates[0])
    update = _sigmoid(igates[1] + hgates[1])
    candidate = jnp.tanh(igates[2] + reset * (hgates[2] + bias_n))
    out_shape = x.shape[:-1] + (flat_h.shape[-1],)
    return {
        "reset": reset.reshape(out_shape),
        "update": update.reshape(out_shape),
        "candidate": candidate.reshape(out_shape),
    }


def _per_replicate_behavior(
    *,
    command_norm: np.ndarray,
    first_five: np.ndarray,
    command_jerk_norm: np.ndarray,
    endpoint_error: np.ndarray,
    terminal_speed: np.ndarray,
    overshoot: np.ndarray,
    sign_changes: np.ndarray,
    hidden_norm: np.ndarray,
    forward_velocity: np.ndarray,
    dt: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep_idx in range(command_norm.shape[0]):
        rows.append(
            {
                "replicate": int(rep_idx),
                "command_norm": _summary_stats(command_norm[rep_idx]),
                "first_five_step_command_norm": _summary_stats(first_five[rep_idx]),
                "command_jerk_norm": _summary_stats(command_jerk_norm[rep_idx]),
                "endpoint_error_m": _summary_stats(endpoint_error[rep_idx]),
                "terminal_speed_m_s": _summary_stats(terminal_speed[rep_idx]),
                "overshoot_m": _summary_stats(overshoot[rep_idx]),
                "post_peak_forward_velocity_sign_changes": _summary_stats(sign_changes[rep_idx]),
                "hidden_state_norm": _summary_stats(hidden_norm[rep_idx]),
                "velocity_profile": _velocity_profile_summary(forward_velocity[rep_idx], dt),
            }
        )
    return rows


def _velocity_profile_summary(forward_velocity: Any, dt: float) -> dict[str, Any]:
    forward_velocity = jnp.asarray(forward_velocity, dtype=jnp.float64)
    per_trial_peak = jnp.max(forward_velocity, axis=-1)
    per_trial_peak_index = jnp.argmax(forward_velocity, axis=-1)
    mean_profile = jnp.mean(forward_velocity.reshape((-1, forward_velocity.shape[-1])), axis=0)
    peak_idx = int(jnp.argmax(mean_profile))
    return {
        "peak_forward_velocity_m_s": _summary_stats(per_trial_peak),
        "time_to_peak_forward_velocity_s": _summary_stats(per_trial_peak_index * dt),
        "mean_profile_peak_forward_velocity_m_s": float(mean_profile[peak_idx]),
        "mean_profile_time_to_peak_forward_velocity_s": float(peak_idx * dt),
        "mean_profile_terminal_forward_velocity_m_s": float(mean_profile[-1]),
    }


def _gate_summary(values: Any, *, bounded: bool) -> dict[str, Any]:
    values = jnp.asarray(values, dtype=jnp.float64)
    summary = _summary_stats(values)
    if bounded:
        summary.update(
            {
                "low_saturation_fraction": float(jnp.mean(values < GATE_SATURATION_LOW)),
                "high_saturation_fraction": float(jnp.mean(values > GATE_SATURATION_HIGH)),
            }
        )
    else:
        summary["abs_high_saturation_fraction"] = float(
            jnp.mean(jnp.abs(values) > GATE_SATURATION_HIGH)
        )
    return summary


def _previous_hidden(hidden: Any) -> Any:
    hidden = jnp.asarray(hidden)
    zeros = jnp.zeros_like(hidden[:, :, :1, :])
    return jnp.concatenate([zeros, hidden[:, :, :-1, :]], axis=2)


def _prepend_initial(initial: Any, values: Any) -> Any:
    initial = jnp.asarray(initial, dtype=jnp.float64)
    values = jnp.asarray(values, dtype=jnp.float64)
    return jnp.concatenate(
        [
            jnp.broadcast_to(
                initial[None, :, None, :],
                values.shape[:1] + initial[:, None, :].shape,
            ),
            values,
        ],
        axis=2,
    )


def _forward_velocity_along_reach(
    *,
    velocity: Any,
    initial_position: Any,
    target_position: Any,
) -> Any:
    direction, _distance = _reach_direction(initial_position, target_position)
    return jnp.sum(velocity * direction[None, :, None, :], axis=-1)


def _overshoot_along_reach(
    *,
    position: Any,
    initial_position: Any,
    target_position: Any,
) -> Any:
    direction, distance = _reach_direction(initial_position, target_position)
    displacement = position - initial_position[None, :, None, :]
    projection = jnp.sum(displacement * direction[None, :, None, :], axis=-1)
    return jnp.maximum(jnp.max(projection - distance[None, :, None], axis=-1), 0.0)


def _reach_direction(
    initial_position: Any,
    target_position: Any,
) -> tuple[Any, Any]:
    initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
    target_position = jnp.asarray(target_position, dtype=jnp.float64)
    delta = target_position - initial_position
    distance = jnp.linalg.norm(delta, axis=-1)
    safe_distance = jnp.where(distance > 0.0, distance, 1.0)
    return delta / safe_distance[:, None], distance


def _post_peak_sign_changes(forward_velocity: Any) -> np.ndarray:
    forward_velocity = np.asarray(forward_velocity, dtype=np.float64)
    out = np.zeros(forward_velocity.shape[:2], dtype=np.float64)
    for index in np.ndindex(forward_velocity.shape[:2]):
        values = forward_velocity[index]
        peak_idx = int(np.argmax(values))
        post_peak = values[peak_idx + 1 :]
        signs = np.sign(post_peak[np.abs(post_peak) > SIGN_CHANGE_EPS])
        out[index] = float(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0.0
    return out


def _jacobian_sample_times(
    forward_velocity: np.ndarray,
    n_time: int,
    *,
    timepoint_policy: Sequence[str],
) -> list[tuple[str, int]]:
    times: list[tuple[str, int]] = []
    for label in timepoint_policy:
        if label == "first":
            time_idx = 0
        elif label == "peak_forward_velocity":
            time_idx = int(np.argmax(forward_velocity))
        elif label == "terminal":
            time_idx = n_time - 1
        else:
            raise ValueError(f"Unknown Jacobian timepoint policy {label!r}")
        entry = (label, max(0, min(int(time_idx), n_time - 1)))
        if entry not in times:
            times.append(entry)
    return times


def _select_replicate_tree(tree: Any, replicate: int, n_replicates: int) -> Any:
    return jt.map(
        lambda leaf: leaf[replicate] if _is_replicate_array(leaf, n_replicates) else leaf,
        tree,
    )


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


def _sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + jnp.exp(-x))




__all__ = [
    "DEFAULT_JACOBIAN_TIMEPOINTS",
    "SCHEMA_VERSION",
    "RolloutEvaluation",
    "compute_gru_gate_arrays",
    "diagnostic_definitions",
    "summarize_controller_feedback_scales",
    "summarize_gru_gates",
    "summarize_gru_jacobians",
    "summarize_rollout_behavior",
]

