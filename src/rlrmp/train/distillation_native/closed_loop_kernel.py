"""Preflight surface for closed-loop extLQG distillation into the h0 GRU.

This module owns the issue a378b34 run/spec contract. It deliberately does not
reuse the older guided teacher-feedback-bank trainer: full training must happen
through a Feedbax closed-loop rollout where student actions update the plant
state that the student later observes.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.objectives.loss import AbstractLoss, TermTree
from rlrmp.model.trainable import staged_network_trainable_parts

TEACHER_ISSUE_ID = "376d023"
DEFAULT_TEACHER_GAINS_KEY = "extlqg_controller_gains"
@dataclass(frozen=True)
class ClosedLoopLossWeights:
    """Weights for pure closed-loop extLQG distillation components."""

    kinematics_trajectory: float = 1.0
    velocity: float = 1.0
    endpoint: float = 0.0
    settling: float = 0.0
    action_force_trajectory: float = 1.0
    perturbation_response_trajectory: float = 1.0
    directional_input_output_jvp: float = 0.25
    task_qr_rollout: float = 0.0

    def summary(self) -> dict[str, float]:
        """Return a JSON-serializable weight summary."""

        return asdict(self)


class ExtLQGClosedLoopReference(eqx.Module):
    """Analytical closed-loop extLQG reference over shared observable channels."""

    plant_a: jax.Array
    plant_b: jax.Array
    controller_gains: jax.Array
    observation_matrix: jax.Array
    feedback_gains: jax.Array
    state_dim: int = eqx.field(static=True)

    @classmethod
    def from_package(
        cls,
        path: str | Path,
        *,
        teacher_gains_key: str = DEFAULT_TEACHER_GAINS_KEY,
    ) -> "ExtLQGClosedLoopReference":
        """Load the extLQG package produced by the analytical-teacher lane."""

        package_path = Path(path)
        if not package_path.is_file():
            raise FileNotFoundError(
                f"Teacher package not found at {package_path}. Sync or materialize "
                f"[issue:{TEACHER_ISSUE_ID}] before training a378b34."
            )
        arrays = np.load(package_path)
        required = ("plant_A", "plant_B", "observation_matrix", teacher_gains_key)
        missing = [key for key in required if key not in arrays.files]
        if missing:
            raise ValueError(
                f"Teacher package {package_path} is missing required keys: {', '.join(missing)}."
            )
        plant_a = jnp.asarray(arrays["plant_A"], dtype=jnp.float32)
        plant_b = jnp.asarray(arrays["plant_B"], dtype=jnp.float32)
        controller_gains = jnp.asarray(arrays[teacher_gains_key], dtype=jnp.float32)
        observation_matrix = jnp.asarray(arrays["observation_matrix"], dtype=jnp.float32)
        if plant_a.ndim != 2 or plant_a.shape[0] != plant_a.shape[1]:
            raise ValueError("teacher plant_A must be square.")
        if plant_b.shape != (plant_a.shape[0], 2):
            raise ValueError("teacher plant_B must have shape (state_dim, 2).")
        if controller_gains.shape[1:] != (2, plant_a.shape[0]):
            raise ValueError(f"{teacher_gains_key} must have shape (time, 2, {plant_a.shape[0]}).")
        feedback_pinv = jnp.linalg.pinv(observation_matrix)
        feedback_gains = -jnp.einsum("tus,sf->tuf", controller_gains, feedback_pinv)
        return cls(
            plant_a=plant_a,
            plant_b=plant_b,
            controller_gains=controller_gains,
            observation_matrix=observation_matrix,
            feedback_gains=feedback_gains,
            state_dim=int(plant_a.shape[0]),
        )

    def rollout(
        self,
        *,
        initial_vector: jax.Array,
        target_pos: jax.Array,
        n_steps: int,
    ) -> dict[str, jax.Array]:
        """Roll the analytical controller from trial starts in target-centered coordinates."""

        x0 = self._initial_teacher_state(initial_vector, target_pos)
        indices = jnp.clip(jnp.arange(int(n_steps)), 0, self.controller_gains.shape[0] - 1)
        # Keep the scan carry contract stable even when an earlier test or an
        # analysis process enabled JAX x64 before constructing this module.
        # Training state is intentionally float32; teacher packages or direct
        # constructors may otherwise supply float64 matrices.
        gains = jnp.asarray(self.controller_gains[indices], dtype=x0.dtype)
        plant_a = jnp.asarray(self.plant_a, dtype=x0.dtype)
        plant_b = jnp.asarray(self.plant_b, dtype=x0.dtype)

        def step(state: jax.Array, gain: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
            action = -jnp.einsum("us,...s->...u", gain, state)
            next_state = state @ plant_a.T + action @ plant_b.T
            return next_state, (next_state, action)

        _, (relative_states_t, actions_t) = jax.lax.scan(step, x0, gains)
        relative_states = jnp.moveaxis(relative_states_t, 0, -2)
        actions = jnp.moveaxis(actions_t, 0, -2)
        target = jnp.expand_dims(jnp.asarray(target_pos, dtype=relative_states.dtype), axis=-2)
        return {
            "position": relative_states[..., 0:2] + target,
            "velocity": relative_states[..., 2:4],
            "force_filter": relative_states[..., 4:6],
            "action": actions,
        }

    def feedback_policy(self, feedback_history: jax.Array) -> jax.Array:
        """Approximate local 6D feedback-to-action teacher map for JVP matching."""

        n_steps = int(feedback_history.shape[-2])
        indices = jnp.clip(jnp.arange(n_steps), 0, self.feedback_gains.shape[0] - 1)
        gains = self.feedback_gains[indices]
        return jnp.einsum("tuf,...tf->...tu", gains, feedback_history)

    def _initial_teacher_state(self, initial_vector: jax.Array, target_pos: jax.Array) -> jax.Array:
        initial_vector = jnp.asarray(initial_vector, dtype=jnp.float32)
        target_pos = jnp.asarray(target_pos, dtype=jnp.float32)
        batch_shape = jnp.broadcast_shapes(initial_vector.shape[:-1], target_pos.shape[:-1])
        initial_vector = jnp.broadcast_to(initial_vector, (*batch_shape, initial_vector.shape[-1]))
        target_pos = jnp.broadcast_to(target_pos, (*batch_shape, 2))
        teacher_state = jnp.zeros((*batch_shape, self.state_dim), dtype=initial_vector.dtype)
        shared = min(6, self.state_dim, initial_vector.shape[-1])
        teacher_state = teacher_state.at[..., :shared].set(initial_vector[..., :shared])
        if self.state_dim >= 2:
            teacher_state = teacher_state.at[..., 0:2].set(initial_vector[..., 0:2] - target_pos)
        return teacher_state


class ClosedLoopDistillationLoss(AbstractLoss):
    """Feedbax loss for pure closed-loop extLQG distillation.

    The loss is called by Feedbax after the normal closed-loop
    rollout. The analytical reference is rolled from the same batched trial
    initial states and targets; matching happens on shared observable channels
    rather than pretending the 36D teacher and 48D student state bases are the
    same object.
    """

    reference: ExtLQGClosedLoopReference
    weights: ClosedLoopLossWeights = eqx.field(static=True)
    label: str = eqx.field(default="closed_loop_extlqg_distillation", static=True)

    @jax.named_scope("rlrmp.ClosedLoopDistillationLoss")
    def __call__(self, states: Any, trial_specs: Any, model: Any) -> TermTree:
        components = closed_loop_distillation_components(
            states,
            trial_specs,
            model,
            reference=self.reference,
        )
        weights = self.weights.summary()
        leaves = {
            name: TermTree.leaf(name, value).with_weight(weights[name])
            for name, value in components.items()
            if weights.get(name, 0.0) > 0.0
        }
        if self.weights.task_qr_rollout > 0.0:
            raise ValueError(
                "task_qr_rollout is intentionally not mixed into the first pure "
                "closed-loop distillation row."
            )
        return TermTree.branch(self.label, leaves, originator=self)

    def skeleton(self, batch_dims: tuple[int, ...]) -> TermTree:
        weights = self.weights.summary()
        leaves = {
            name: TermTree.leaf(name, jnp.empty(batch_dims)).with_weight(weight)
            for name, weight in weights.items()
            if weight > 0.0 and name != "task_qr_rollout"
        }
        return TermTree.branch(self.label, leaves, originator=self)


def _base_run_spec(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_serialized_hps(hps: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(hps)
    if normalized.get("hidden_type") == "equinox.nn._rnn.GRUCell":
        normalized["hidden_type"] = None
    if "intervention_scaleup_batches" not in normalized:
        normalized["intervention_scaleup_batches"] = [0, 0]
    pgd = normalized.get("broad_epsilon_pgd_training")
    if isinstance(pgd, dict) and not pgd.get("enabled"):
        budget_contract = pgd.get("budget_contract")
        if isinstance(budget_contract, dict) and budget_contract.get("effective_l2_radius_15cm"):
            budget_contract.setdefault(
                "budget_source",
                {
                    "key": "disabled_closed_loop_distillation_no_pgd",
                    "note": (
                        "Closed-loop distillation disables PGD; retained radius is provenance only."
                    ),
                },
            )
    return normalized


def _standard_hps_from_spec(
    spec: dict[str, Any],
    *,
    n_replicates: int,
    hidden_size: int,
    batch_size: int,
    n_batches: int,
    controller_lr: float,
    lr_warmup_batches: int,
    lr_warmup_init_fraction: float,
    lr_cosine_alpha: float,
    gradient_clip_norm: float,
    trainable_dtype: str,
) -> TreeNamespace:
    hps = spec.get("hps")
    if hps is None:
        base_run_spec = spec.get("base_contract", {}).get("run_spec")
        if not base_run_spec:
            raise ValueError(
                "closed-loop distillation specs without inline hps require "
                "base_contract.run_spec"
            )
        hps = _normalize_serialized_hps(
            _base_run_spec(base_run_spec)["hps"]
        )
    else:
        hps = _normalize_serialized_hps(hps)
    hps["batch_size"] = int(batch_size)
    hps["n_batches_condition"] = int(n_batches)
    hps["learning_rate_0"] = float(controller_lr)
    hps["constant_lr_iterations"] = int(lr_warmup_batches)
    hps["warmup_init_fraction"] = float(lr_warmup_init_fraction)
    hps["cosine_annealing_alpha"] = float(lr_cosine_alpha)
    hps["gradient_clip_norm"] = float(gradient_clip_norm)
    model = hps.setdefault("model", {})
    model["n_replicates"] = int(n_replicates)
    model["hidden_size"] = int(hidden_size)
    population = model.setdefault("population_structure", {})
    population["n_input_only"] = 0
    population["n_readout_only"] = 0
    population["n_recurrent_only"] = 0
    population["n_input_readout"] = int(hidden_size)
    model["trainable_dtype"] = str(trainable_dtype)
    return dict_to_namespace(hps, to_type=TreeNamespace)


def _where_train_fn(model: Any) -> tuple[Any, ...]:
    return staged_network_trainable_parts(model.nodes["net"])


def _target_position(trial_specs: Any, states: Any) -> jax.Array:
    target_spec = trial_specs.targets.get("mechanics.effector.pos", None)
    if target_spec is None or not hasattr(target_spec, "value"):
        pos = jnp.asarray(states.mechanics.effector.pos)
        return jnp.zeros((*pos.shape[:-2], 2), dtype=pos.dtype)
    target_value = jnp.asarray(target_spec.value, dtype=jnp.float32)
    return target_value[..., -1, :]


def _initial_vector(trial_specs: Any, states: Any) -> jax.Array:
    vector = jnp.asarray(states.mechanics.vector, dtype=jnp.float32)
    if "mechanics.vector" in trial_specs.inits:
        initial = jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=vector.dtype)
        return jnp.broadcast_to(initial, (*vector.shape[:-2], vector.shape[-1]))
    return jnp.zeros((*vector.shape[:-2], vector.shape[-1]), dtype=vector.dtype)


def _per_trial_mse(diff: jax.Array) -> jax.Array:
    diff = jnp.asarray(diff, dtype=jnp.float32)
    if diff.ndim <= 1:
        return jnp.mean(jnp.square(diff))
    return jnp.mean(jnp.square(diff), axis=tuple(range(1, diff.ndim)))


def _last_window(values: jax.Array, width: int = 10) -> jax.Array:
    n_time = int(values.shape[-2])
    start = max(0, n_time - int(width))
    return values[..., start:, :]


def _coordinate_feedback_directions(feedback_history: jax.Array) -> jax.Array:
    """Return full coordinate-basis directions for the local 6D feedback input."""

    feedback_history = jnp.asarray(feedback_history, dtype=jnp.float32)
    feedback_dim = int(feedback_history.shape[-1])
    basis = jnp.eye(feedback_dim, dtype=feedback_history.dtype)
    direction_shape = (feedback_dim, *feedback_history.shape)
    return jnp.broadcast_to(
        basis.reshape((feedback_dim, *([1] * (feedback_history.ndim - 1)), feedback_dim)),
        direction_shape,
    )


def _model_feedback_policy(model: Any, feedback_history: jax.Array) -> jax.Array:
    """Run the standard h0 controller on a controller-visible feedback history."""

    net_node = model.nodes["net"]

    def single(feedback: jax.Array) -> jax.Array:
        hidden = net_node.h0_encoder(feedback[0])

        def step(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_hidden = net_node.net.hidden(value, carry)
            action = net_node.net.readout(next_hidden)
            return next_hidden, action

        _, actions = jax.lax.scan(step, hidden, feedback)
        return actions

    return jax.vmap(single)(feedback_history)


def _model_local_feedback_jvps(model: Any, feedback_history: jax.Array) -> jax.Array:
    """Return full local feedback-to-action Jacobian columns by basis JVPs.

    The local map is the per-step controller update ``feedback_t -> action_t``
    with the recurrent carry entering that step held fixed. The returned tensor
    has shape ``(feedback_dim, batch, time, action_dim)``.
    """

    net_node = model.nodes["net"]
    feedback_history = jnp.asarray(feedback_history, dtype=jnp.float32)
    feedback_dim = int(feedback_history.shape[-1])
    basis = jnp.eye(feedback_dim, dtype=feedback_history.dtype)

    def sequence_jvps(feedback: jax.Array) -> jax.Array:
        hidden0 = net_node.h0_encoder(feedback[0])

        def collect_pre_hidden(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_hidden = net_node.net.hidden(value, carry)
            return next_hidden, carry

        _, hidden_before = jax.lax.scan(collect_pre_hidden, hidden0, feedback)

        def step_jvps(hidden: jax.Array, value: jax.Array) -> jax.Array:
            def action_for_feedback(local_feedback: jax.Array) -> jax.Array:
                return net_node.net.readout(net_node.net.hidden(local_feedback, hidden))

            return jax.vmap(
                lambda direction: jax.jvp(action_for_feedback, (value,), (direction,))[1]
            )(basis)

        time_major = jax.vmap(step_jvps)(hidden_before, feedback)
        return jnp.moveaxis(time_major, 0, 1)

    return jax.vmap(sequence_jvps)(feedback_history).transpose(1, 0, 2, 3)


def _teacher_local_feedback_jvps(
    reference: ExtLQGClosedLoopReference,
    feedback_history: jax.Array,
) -> jax.Array:
    """Return full local teacher feedback-to-action Jacobian columns."""

    n_steps = int(feedback_history.shape[-2])
    feedback_dim = int(feedback_history.shape[-1])
    indices = jnp.clip(jnp.arange(n_steps), 0, reference.feedback_gains.shape[0] - 1)
    gains = reference.feedback_gains[indices]
    columns = jnp.moveaxis(gains[:, :, :feedback_dim], -1, 0)
    return jnp.broadcast_to(
        columns[:, None, :, :], (feedback_dim, feedback_history.shape[0], n_steps, 2)
    )


def _full_local_jacobian_component(
    *,
    model: Any,
    reference: ExtLQGClosedLoopReference,
    feedback_history: jax.Array,
) -> jax.Array:
    if feedback_history.shape[-1] != reference.observation_matrix.shape[0]:
        return jnp.zeros(feedback_history.shape[0], dtype=jnp.float32)
    student_jvps = _model_local_feedback_jvps(model, feedback_history)
    teacher_jvps = _teacher_local_feedback_jvps(reference, feedback_history)
    return jnp.mean(jnp.square(student_jvps - teacher_jvps), axis=(0, 2, 3))


def closed_loop_distillation_components(
    states: Any,
    trial_specs: Any,
    model: Any,
    *,
    reference: ExtLQGClosedLoopReference,
) -> dict[str, jax.Array]:
    """Compute unweighted per-trial closed-loop distillation components."""

    pos = jnp.asarray(states.mechanics.effector.pos, dtype=jnp.float32)
    vel = jnp.asarray(states.mechanics.effector.vel, dtype=jnp.float32)
    vector = jnp.asarray(states.mechanics.vector, dtype=jnp.float32)
    command = jnp.asarray(states.net.output, dtype=jnp.float32)
    target_pos = _target_position(trial_specs, states)
    initial_vector = _initial_vector(trial_specs, states)
    teacher = reference.rollout(
        initial_vector=initial_vector,
        target_pos=target_pos,
        n_steps=int(pos.shape[-2]),
    )
    force_filter = vector[..., 4:6]
    feedback_history = jnp.asarray(states.net.input, dtype=jnp.float32)
    return {
        "kinematics_trajectory": _per_trial_mse(pos - teacher["position"]),
        "velocity": _per_trial_mse(vel - teacher["velocity"]),
        "endpoint": _per_trial_mse(pos[..., -1, :] - teacher["position"][..., -1, :]),
        "settling": _per_trial_mse(_last_window(pos) - _last_window(teacher["position"]))
        + _per_trial_mse(_last_window(vel) - _last_window(teacher["velocity"])),
        "action_force_trajectory": _per_trial_mse(command - teacher["action"])
        + _per_trial_mse(force_filter - teacher["force_filter"]),
        "perturbation_response_trajectory": _per_trial_mse(
            (pos - pos[..., :1, :]) - (teacher["position"] - teacher["position"][..., :1, :])
        ),
        "directional_input_output_jvp": _full_local_jacobian_component(
            model=model,
            reference=reference,
            feedback_history=feedback_history,
        ),
    }


def build_closed_loop_loss(
    spec: dict[str, Any],
    *,
    reference: ExtLQGClosedLoopReference | None = None,
) -> ClosedLoopDistillationLoss:
    """Build the custom Feedbax loss from the run-spec weights."""

    weights = spec["loss_surface"]["weights"]
    teacher = spec["teacher_contract"]
    reference = reference or ExtLQGClosedLoopReference.from_package(
        teacher["teacher_package"],
        teacher_gains_key=teacher["teacher_gains_key"],
    )
    return ClosedLoopDistillationLoss(
        reference=reference,
        weights=ClosedLoopLossWeights(
            kinematics_trajectory=float(weights["kinematics_trajectory"]),
            velocity=float(weights["velocity"]),
            endpoint=float(weights["endpoint"]),
            settling=float(weights["settling"]),
            action_force_trajectory=float(weights["action_force_trajectory"]),
            perturbation_response_trajectory=float(weights["perturbation_response_trajectory"]),
            directional_input_output_jvp=float(weights["directional_input_output_jvp"]),
            task_qr_rollout=float(weights["task_qr_rollout"]),
        ),
    )


def _training_hps_from_spec(
    spec: dict[str, Any],
    *,
    n_batches: int | None = None,
    batch_size: int | None = None,
    n_replicates: int | None = None,
    hidden_size: int | None = None,
) -> TreeNamespace:
    """Materialize the task/model parameters required by the native kernel."""

    student = spec["student_contract"]
    return _standard_hps_from_spec(
        spec,
        n_replicates=int(n_replicates or student["n_replicates"]),
        hidden_size=int(hidden_size or student["hidden_size"]),
        batch_size=int(batch_size or student["batch_size"]),
        n_batches=int(n_batches or student["n_train_batches"]),
        controller_lr=float(student["controller_lr"]),
        lr_warmup_batches=int(student["lr_warmup_batches"]),
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=float(student["lr_cosine_alpha"]),
        gradient_clip_norm=float(student["gradient_clip_norm"]),
        trainable_dtype=str(student["trainable_dtype"]),
    )
