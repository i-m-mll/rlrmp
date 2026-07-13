"""Guided policy-distillation losses over external feedback/action histories.

The helpers in this module deliberately operate on the controller-visible
history contract instead of recurrent hidden state. A student or teacher policy
is any callable

``policy(feedback_history, action_history) -> action_history``

where the histories may include batch axes. This keeps local-map supervision
usable for GRUs, analytical output-feedback teachers, and future recurrent
controllers without treating hidden coordinates as certificate evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

PolicyMap = Callable[[jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True)
class DistillationLossWeights:
    """Weights for composable guided-distillation loss components."""

    clean_action: float = 0.0
    perturbation_response: float = 0.0
    input_output_jvp: float = 0.0
    student_forced_rollout_anchor: float = 0.0

    def summary(self) -> dict[str, float]:
        """Return a JSON-serializable weight summary."""

        return asdict(self)


@dataclass(frozen=True)
class CSH0DistillationConfig:
    """C&S h0 GRU distillation contract for the first 6D teacher row."""

    feedback_basis: str = "target_relative_delayed_feedback_plus_force_filter"
    action_basis: str = "controller_command_history"
    hidden_state_supervision: bool = False
    n_jvp_directions: int = 16
    jvp_direction_basis: str = "controller_visible_feedback_and_action_history"
    jvp_direction_sampler: str = "banked_probe_directions"
    weights: DistillationLossWeights = field(
        default_factory=lambda: DistillationLossWeights(
            clean_action=1.0,
            perturbation_response=1.0,
            input_output_jvp=0.25,
            student_forced_rollout_anchor=0.25,
        )
    )

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable config summary for run specs."""

        payload = asdict(self)
        payload["weights"] = self.weights.summary()
        return payload


class DistillationLossResult(NamedTuple):
    """Total guided loss and unweighted component losses."""

    total: jax.Array
    components: dict[str, jax.Array]


def cs_h0_distillation_config(
    *,
    weights: DistillationLossWeights | None = None,
    n_jvp_directions: int = 16,
) -> CSH0DistillationConfig:
    """Return the first 6D H-infinity h0 GRU distillation config.

    Args:
        weights: Optional component weights. Defaults enable clean action,
            perturbation-response, input-output JVP, and rollout-anchor hooks.
        n_jvp_directions: Number of banked directional probes used per local-map
            batch. Directions are vectorized in the JVP loss.

    Returns:
        A frozen config suitable for embedding in a no-launch run spec.
    """

    return CSH0DistillationConfig(
        weights=weights
        or DistillationLossWeights(
            clean_action=1.0,
            perturbation_response=1.0,
            input_output_jvp=0.25,
            student_forced_rollout_anchor=0.25,
        ),
        n_jvp_directions=int(n_jvp_directions),
    )


def mean_squared_error(
    diff: jax.Array,
    *,
    weight: jax.Array | float | None = None,
) -> jax.Array:
    """Return a broadcast-weighted mean-square error."""

    squared = jnp.square(diff)
    if weight is None:
        return jnp.mean(squared)
    weight_array = jnp.asarray(weight, dtype=squared.dtype)
    weighted = squared * weight_array
    denominator = jnp.sum(jnp.ones_like(squared) * weight_array)
    return jnp.sum(weighted) / jnp.maximum(denominator, jnp.asarray(1, squared.dtype))


def clean_action_imitation_loss(
    student_actions: jax.Array,
    teacher_actions: jax.Array,
    *,
    weight: jax.Array | float | None = None,
) -> jax.Array:
    """Match clean student and teacher action histories."""

    return mean_squared_error(student_actions - teacher_actions, weight=weight)


def perturbation_response_imitation_loss(
    *,
    student_base_actions: jax.Array,
    teacher_base_actions: jax.Array,
    student_perturbed_actions: jax.Array,
    teacher_perturbed_actions: jax.Array,
    weight: jax.Array | float | None = None,
) -> jax.Array:
    """Match the action response induced by externally meaningful perturbations."""

    student_response = student_perturbed_actions - student_base_actions
    teacher_response = teacher_perturbed_actions - teacher_base_actions
    return mean_squared_error(student_response - teacher_response, weight=weight)


def student_forced_rollout_anchor_loss(
    student_rollout: jax.Array,
    anchor_rollout: jax.Array,
    *,
    weight: jax.Array | float | None = None,
) -> jax.Array:
    """Anchor student-forced rollout summaries to a teacher or base-run target."""

    return mean_squared_error(student_rollout - anchor_rollout, weight=weight)


def _direction_batch_shape(direction: jax.Array, primal: jax.Array, name: str) -> tuple[int, ...]:
    primal_ndim = primal.ndim
    if direction.ndim < primal_ndim:
        raise ValueError(
            f"{name} direction must have at least {primal_ndim} dimensions, got {direction.ndim}."
        )
    if direction.shape[-primal_ndim:] != primal.shape:
        raise ValueError(
            f"{name} direction trailing shape {direction.shape[-primal_ndim:]} "
            f"does not match primal shape {primal.shape}."
        )
    return tuple(direction.shape[: direction.ndim - primal_ndim])


def _flatten_direction_batch(direction: jax.Array, primal: jax.Array) -> jax.Array:
    primal_ndim = primal.ndim
    direction_shape = direction.shape[: direction.ndim - primal_ndim]
    flat_count = 1
    for size in direction_shape:
        flat_count *= int(size)
    return jnp.reshape(direction, (flat_count, *primal.shape))


def batched_directional_jvps(
    policy: PolicyMap,
    feedback_history: jax.Array,
    action_history: jax.Array,
    feedback_directions: jax.Array,
    action_directions: jax.Array,
) -> jax.Array:
    """Return policy JVPs for a batch of history-space directions.

    Args:
        policy: Callable mapping ``(feedback_history, action_history)`` to an
            action history. The output may include the same batch axes as the
            inputs.
        feedback_history: Controller-visible feedback or observation history,
            shape ``(..., time, feedback_dim)`` by convention.
        action_history: Externally meaningful action history/context, shape
            ``(..., time, action_dim)`` by convention.
        feedback_directions: Direction batch with shape
            ``D + feedback_history.shape``.
        action_directions: Direction batch with shape
            ``D + action_history.shape``.

    Returns:
        A JVP tensor with shape ``D + policy(feedback_history, action_history).shape``.

    Notes:
        The implementation uses ``jax.linearize`` once per policy and a single
        ``jax.vmap`` over the flattened direction batch. It does not materialize
        dense Jacobians in the training path.
    """

    feedback_directions = jnp.asarray(feedback_directions)
    action_directions = jnp.asarray(action_directions)
    direction_shape = _direction_batch_shape(
        feedback_directions,
        feedback_history,
        "feedback",
    )
    action_direction_shape = _direction_batch_shape(
        action_directions,
        action_history,
        "action",
    )
    if action_direction_shape != direction_shape:
        raise ValueError(
            "Feedback and action directions must have the same leading direction "
            f"shape, got {direction_shape} and {action_direction_shape}."
        )
    flat_feedback = _flatten_direction_batch(feedback_directions, feedback_history)
    flat_actions = _flatten_direction_batch(action_directions, action_history)
    output, linearized = jax.linearize(policy, feedback_history, action_history)
    flat_jvps = jax.vmap(lambda df, da: linearized(df, da))(flat_feedback, flat_actions)
    return jnp.reshape(flat_jvps, (*direction_shape, *output.shape))


def input_output_jvp_matching_loss(
    *,
    student_policy: PolicyMap,
    teacher_policy: PolicyMap,
    feedback_history: jax.Array,
    action_history: jax.Array,
    feedback_directions: jax.Array,
    action_directions: jax.Array,
    weight: jax.Array | float | None = None,
) -> jax.Array:
    """Match student and teacher local action-history maps by directional JVPs."""

    student_jvps = batched_directional_jvps(
        student_policy,
        feedback_history,
        action_history,
        feedback_directions,
        action_directions,
    )
    teacher_jvps = batched_directional_jvps(
        teacher_policy,
        feedback_history,
        action_history,
        feedback_directions,
        action_directions,
    )
    return mean_squared_error(student_jvps - teacher_jvps, weight=weight)


def guided_distillation_loss(
    *,
    student_policy: PolicyMap,
    teacher_policy: PolicyMap,
    feedback_history: jax.Array,
    action_history: jax.Array,
    config: CSH0DistillationConfig | None = None,
    action_weight: jax.Array | float | None = None,
    perturbation_feedback_history: jax.Array | None = None,
    perturbation_action_history: jax.Array | None = None,
    perturbation_weight: jax.Array | float | None = None,
    feedback_directions: jax.Array | None = None,
    action_directions: jax.Array | None = None,
    jvp_weight: jax.Array | float | None = None,
    student_forced_rollout: jax.Array | None = None,
    rollout_anchor: jax.Array | None = None,
    rollout_anchor_weight: jax.Array | float | None = None,
) -> DistillationLossResult:
    """Compose guided distillation losses for a student/teacher policy pair."""

    config = config or cs_h0_distillation_config()
    weights = config.weights
    components: dict[str, jax.Array] = {}
    total = jnp.asarray(0.0)

    student_actions = student_policy(feedback_history, action_history)
    teacher_actions = teacher_policy(feedback_history, action_history)

    if weights.clean_action:
        clean = clean_action_imitation_loss(
            student_actions,
            teacher_actions,
            weight=action_weight,
        )
        components["clean_action"] = clean
        total = total + weights.clean_action * clean

    if weights.perturbation_response:
        if perturbation_feedback_history is None or perturbation_action_history is None:
            raise ValueError(
                "Perturbation histories are required when perturbation_response weight is nonzero."
            )
        student_perturbed = student_policy(
            perturbation_feedback_history, perturbation_action_history
        )
        teacher_perturbed = teacher_policy(
            perturbation_feedback_history, perturbation_action_history
        )
        response = perturbation_response_imitation_loss(
            student_base_actions=student_actions,
            teacher_base_actions=teacher_actions,
            student_perturbed_actions=student_perturbed,
            teacher_perturbed_actions=teacher_perturbed,
            weight=perturbation_weight,
        )
        components["perturbation_response"] = response
        total = total + weights.perturbation_response * response

    if weights.input_output_jvp:
        if feedback_directions is None or action_directions is None:
            raise ValueError(
                "Directional probes are required when input_output_jvp weight is nonzero."
            )
        jvp_loss = input_output_jvp_matching_loss(
            student_policy=student_policy,
            teacher_policy=teacher_policy,
            feedback_history=feedback_history,
            action_history=action_history,
            feedback_directions=feedback_directions,
            action_directions=action_directions,
            weight=jvp_weight,
        )
        components["input_output_jvp"] = jvp_loss
        total = total + weights.input_output_jvp * jvp_loss

    if weights.student_forced_rollout_anchor:
        if student_forced_rollout is None or rollout_anchor is None:
            raise ValueError(
                "Student-forced rollout and anchor arrays are required when "
                "student_forced_rollout_anchor weight is nonzero."
            )
        anchor = student_forced_rollout_anchor_loss(
            student_forced_rollout,
            rollout_anchor,
            weight=rollout_anchor_weight,
        )
        components["student_forced_rollout_anchor"] = anchor
        total = total + weights.student_forced_rollout_anchor * anchor

    return DistillationLossResult(total=total, components=components)


__all__ = [
    "CSH0DistillationConfig",
    "DistillationLossResult",
    "DistillationLossWeights",
    "PolicyMap",
    "batched_directional_jvps",
    "clean_action_imitation_loss",
    "cs_h0_distillation_config",
    "guided_distillation_loss",
    "input_output_jvp_matching_loss",
    "mean_squared_error",
    "perturbation_response_imitation_loss",
    "student_forced_rollout_anchor_loss",
]
