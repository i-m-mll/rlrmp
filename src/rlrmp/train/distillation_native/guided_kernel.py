"""CLI support for guided C&S GRU distillation run specs.

The reusable loss/JVP implementation and native executor live in this
capability package. The public entry module only delegates CLI calls here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax

from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from jax_cookbook.tree import filter_spec_leaves
from rlrmp.model.trainable import staged_network_trainable_parts
from rlrmp.train.distillation_native.losses import (
    CSH0DistillationConfig,
    guided_distillation_loss,
)

DEFAULT_TEACHER_GAINS_KEY = "extlqg_controller_gains"
DEFAULT_TRAINABLE_DTYPE = "float32"
REQUIRED_TEACHER_KEYS = (
    "plant_A",
    "plant_B",
    "x0",
    "observation_matrix",
)


def _normalize_serialized_hps(hps: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(hps)
    if normalized.get("hidden_type") == "equinox.nn._rnn.GRUCell":
        normalized["hidden_type"] = None
    pgd = normalized.get("broad_epsilon_pgd_training")
    if isinstance(pgd, dict) and not pgd.get("enabled", False):
        budget_contract = pgd.get("budget_contract")
        if isinstance(budget_contract, dict) and budget_contract.get("effective_l2_radius_15cm"):
            budget_contract.setdefault(
                "budget_source",
                {
                    "key": "disabled_guided_distillation_no_pgd",
                    "note": "Guided distillation disables PGD; retained radius is provenance only.",
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
    trainable_dtype: str = DEFAULT_TRAINABLE_DTYPE,
    population_mask_mode: str | None = None,
) -> TreeNamespace:
    if population_mask_mode is None:
        population_mask_mode = spec.get("model_contract", {}).get("population_mask_mode")
    hps = spec.get("hps")
    if hps is None:
        raise ValueError(
            "native guided-distillation specs must embed hps; "
            "legacy base-run reconstruction is retired"
        )
    hps = _normalize_serialized_hps(hps)
    hps["batch_size"] = int(batch_size)
    hps["n_batches_condition"] = int(n_batches)
    hps["learning_rate_0"] = float(controller_lr)
    hps["constant_lr_iterations"] = int(lr_warmup_batches)
    hps["warmup_init_fraction"] = float(lr_warmup_init_fraction)
    hps["cosine_annealing_alpha"] = float(lr_cosine_alpha)
    hps["gradient_clip_norm"] = float(gradient_clip_norm)
    hps["model"]["n_replicates"] = int(n_replicates)
    hps["model"]["hidden_size"] = int(hidden_size)
    population = hps["model"].setdefault("population_structure", {})
    population["n_input_only"] = 0
    population["n_readout_only"] = 0
    population["n_recurrent_only"] = 0
    population["n_input_readout"] = int(hidden_size)
    hps["model"]["trainable_dtype"] = str(trainable_dtype)
    if population_mask_mode is not None:
        hps["model"]["population_mask_mode"] = str(population_mask_mode)
    return dict_to_namespace(hps, to_type=TreeNamespace)


class BankedAffineTeacher(eqx.Module):
    """Local affine analytical teacher over a sampled external-feedback bank."""

    base_feedback: jax.Array
    base_actions: jax.Array
    feedback_gains: jax.Array

    def __call__(self, feedback_history: jax.Array, action_history: jax.Array) -> jax.Array:
        del action_history
        feedback_delta = feedback_history - self.base_feedback
        return self.base_actions + jnp.einsum(
            "...tuf,...tf->...tu",
            self.feedback_gains,
            feedback_delta,
        )


@dataclass(frozen=True)
class StandardControllerParts:
    """Controller submodules for old SimpleStagedNetwork and native Graph forms."""

    h0_encoder: Any
    hidden_cell: Any
    readout: Any


def _require_teacher_package(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Teacher package not found at {path}. Generate or sync the "
            "376d023 analytical teacher package before running --full-train."
        )
    package = np.load(path)
    missing = [key for key in REQUIRED_TEACHER_KEYS if key not in package.files]
    if missing:
        raise ValueError(f"Teacher package {path} is missing required keys: {', '.join(missing)}.")
    return {key: package[key] for key in package.files}


def load_teacher_package(
    path: str | Path,
    *,
    teacher_gains_key: str = DEFAULT_TEACHER_GAINS_KEY,
) -> dict[str, jax.Array]:
    """Load and validate the analytical teacher package."""

    arrays = _require_teacher_package(Path(path))
    if teacher_gains_key not in arrays:
        raise ValueError(
            f"Teacher package {path} is missing selected gains key: {teacher_gains_key}."
        )
    plant_a = arrays["plant_A"]
    plant_b = arrays["plant_B"]
    controller_gains = arrays[teacher_gains_key]
    observation_matrix = arrays["observation_matrix"]
    if plant_a.ndim != 2 or plant_a.shape[0] != plant_a.shape[1]:
        raise ValueError("teacher plant_A must be square")
    if plant_b.ndim != 2 or plant_b.shape[0] != plant_a.shape[0]:
        raise ValueError("teacher plant_B must have shape (state_dim, action_dim)")
    if controller_gains.ndim != 3 or controller_gains.shape[1:] != (
        plant_b.shape[1],
        plant_a.shape[0],
    ):
        raise ValueError(f"{teacher_gains_key} must have shape (time, action_dim, state_dim)")
    if observation_matrix.ndim != 2 or observation_matrix.shape[1] != plant_a.shape[0]:
        raise ValueError("observation_matrix must have shape (feedback_dim, state_dim)")
    package = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in arrays.items()}
    package["controller_gains"] = jnp.asarray(controller_gains, dtype=jnp.float32)
    return package


def forcing_fraction_for_batch(spec: dict[str, Any], batch_index: int) -> float:
    """Return the student-forcing fraction encoded by the staged run spec."""

    for phase in spec["training_schedule"]["phases"]:
        if int(phase["start_batch"]) <= batch_index < int(phase["end_batch"]):
            return float(phase["student_forcing_fraction"])
    return float(spec["training_schedule"]["phases"][-1]["student_forcing_fraction"])


def _teacher_rollout(
    package: dict[str, jax.Array],
    initial_states: jax.Array,
    *,
    horizon: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    plant_a = package["plant_A"]
    plant_b = package["plant_B"]
    observation_matrix = package["observation_matrix"]
    gains = package["controller_gains"][:horizon]
    feedback_pinv = jnp.linalg.pinv(observation_matrix)
    feedback_gains = -jnp.einsum("tus,sf->tuf", gains, feedback_pinv)

    def step(states: jax.Array, gain: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        feedback = states @ observation_matrix.T
        actions = -jnp.einsum("us,bs->bu", gain, states)
        next_states = states @ plant_a.T + actions @ plant_b.T
        return next_states, (states, feedback, actions)

    _, (states, feedback, actions) = jax.lax.scan(step, initial_states, gains)
    gains_by_batch = jnp.broadcast_to(
        feedback_gains[None, ...],
        (initial_states.shape[0], *feedback_gains.shape),
    )
    return states.swapaxes(0, 1), feedback.swapaxes(0, 1), actions.swapaxes(0, 1), gains_by_batch


def materialize_teacher_batch(
    package: dict[str, jax.Array],
    *,
    key: jax.Array,
    batch_size: int,
    horizon: int,
    n_jvp_directions: int,
    initial_state_std: float = 0.02,
    observation_perturbation_std: float = 0.05,
) -> dict[str, jax.Array]:
    """Materialize one deterministic analytical rollout/probe batch."""

    state_key, perturb_key, direction_key = jr.split(key, 3)
    x0 = package["x0"]
    initial_states = x0 + initial_state_std * jr.normal(
        state_key,
        (batch_size, x0.shape[0]),
        dtype=x0.dtype,
    )
    states, feedback, teacher_actions, feedback_gains = _teacher_rollout(
        package,
        initial_states,
        horizon=horizon,
    )
    del states
    perturb_feedback = feedback + observation_perturbation_std * jr.normal(
        perturb_key,
        feedback.shape,
        dtype=feedback.dtype,
    )
    direction_keys = jr.split(direction_key, 2)
    feedback_directions = 0.01 * jr.normal(
        direction_keys[0],
        (n_jvp_directions, *feedback.shape),
        dtype=feedback.dtype,
    )
    action_directions = 0.01 * jr.normal(
        direction_keys[1],
        (n_jvp_directions, *teacher_actions.shape),
        dtype=teacher_actions.dtype,
    )
    return {
        "feedback_history": feedback,
        "teacher_actions": teacher_actions,
        "perturbation_feedback_history": perturb_feedback,
        "feedback_directions": feedback_directions,
        "action_directions": action_directions,
        "feedback_gains": feedback_gains,
    }


def _make_optimizer(
    *,
    learning_rate: float,
    n_batches: int,
    warmup_batches: int,
    warmup_init_fraction: float,
    cosine_alpha: float,
    gradient_clip_norm: float,
) -> optax.GradientTransformation:
    effective_warmup = max(0, min(warmup_batches, n_batches - 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * warmup_init_fraction,
        peak_value=learning_rate,
        warmup_steps=effective_warmup,
        decay_steps=max(1, n_batches),
        end_value=learning_rate * cosine_alpha,
    )
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.adamw(schedule, weight_decay=0.0),
    )


def _standard_model_actions(model: Any, feedback_history: jax.Array) -> jax.Array:
    parts = standard_controller_parts(model)

    def single(feedback: jax.Array) -> jax.Array:
        hidden = parts.h0_encoder(feedback[0])

        def step(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_hidden = parts.hidden_cell(value, carry)
            action = parts.readout(next_hidden)
            return next_hidden, action

        _, actions = jax.lax.scan(step, hidden, feedback)
        return actions

    if feedback_history.ndim == 2:
        return single(feedback_history)
    return jax.vmap(single)(feedback_history)


def standard_controller_parts(model: Any) -> StandardControllerParts:
    """Return h0 encoder, GRU cell, and readout for supported standard graph shapes."""

    net_node = model.nodes["net"]
    if hasattr(net_node, "net"):
        return StandardControllerParts(
            h0_encoder=net_node.h0_encoder,
            hidden_cell=net_node.net.hidden,
            readout=net_node.net.readout,
        )
    graph_nodes = getattr(net_node, "nodes", None)
    if isinstance(graph_nodes, dict):
        return StandardControllerParts(
            h0_encoder=graph_nodes["h0_encoder"].layer,
            hidden_cell=graph_nodes["cell"].cell,
            readout=graph_nodes["readout"].layer,
        )
    raise TypeError(f"Unsupported standard distillation controller type: {type(net_node)!r}")


def standard_controller_feedback_dim(model: Any) -> int:
    """Return the controller-visible feedback dimension from the GRU input weights."""

    return int(standard_controller_parts(model).hidden_cell.weight_ih.shape[-1])


def standard_controller_action_dim(model: Any) -> int:
    """Return the controller command dimension from the readout weights."""

    return int(_linear_weight(standard_controller_parts(model).readout).shape[-2])


def _linear_weight(layer: Any) -> jax.Array:
    """Return the underlying linear weight for plain or masked linear modules."""

    return layer.linear.weight if hasattr(layer, "linear") else layer.weight


def _linear_bias(layer: Any) -> jax.Array | None:
    """Return the underlying linear bias for plain or masked linear modules."""

    return layer.linear.bias if hasattr(layer, "linear") else layer.bias


def _where_train_fn(model: Any) -> tuple[Any, ...]:
    return staged_network_trainable_parts(model.nodes["net"])


def _where_train_spec(model: Any) -> Any:
    return filter_spec_leaves(model, _where_train_fn)


def _dtype_from_name(name: str) -> jnp.dtype:
    dtype = jnp.dtype(name)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError(
            f"Unsupported guided-distillation trainable dtype {name!r}; "
            "expected 'float32' or 'float64'."
        )
    return dtype


def _trainable_dtype_name(spec: dict[str, Any], args: Any) -> str:
    """Resolve the deliberate trainable dtype request for distillation training."""

    requested = []
    model_contract = spec.get("model_contract", {})
    if model_contract.get("trainable_dtype") is not None:
        requested.append(str(model_contract["trainable_dtype"]))
    hps_model = spec.get("hps", {}).get("model", {})
    if hps_model.get("trainable_dtype") is not None:
        requested.append(str(hps_model["trainable_dtype"]))
    if not requested:
        requested.append(str(getattr(args, "trainable_dtype", DEFAULT_TRAINABLE_DTYPE)))
    unique = set(requested)
    if len(unique) != 1:
        raise ValueError(f"Conflicting trainable dtype requests in run spec: {sorted(unique)}.")
    return requested[0]


def _trainable_float_leaves(model: Any, where_train_spec: Any) -> list[jax.Array]:
    trainable, _frozen = eqx.partition(model, where_train_spec)
    return [
        leaf
        for leaf in jt.leaves(trainable)
        if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def _cast_trainable_float_leaves(
    model: Any,
    where_train_spec: Any,
    dtype: jnp.dtype,
) -> Any:
    trainable, frozen = eqx.partition(model, where_train_spec)

    def cast_leaf(leaf: Any) -> Any:
        if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(dtype)
        return leaf

    return eqx.combine(jt.map(cast_leaf, trainable), frozen)


def _assert_trainable_float_dtype(
    model: Any,
    where_train_spec: Any,
    dtype: jnp.dtype,
    *,
    context: str,
) -> None:
    leaves = _trainable_float_leaves(model, where_train_spec)
    if not leaves:
        raise ValueError(f"{context} has no floating trainable leaves.")
    observed = sorted({str(leaf.dtype) for leaf in leaves})
    if any(jnp.dtype(leaf.dtype) != dtype for leaf in leaves):
        raise TypeError(
            f"{context} trainable leaves must be {dtype.name}; observed {observed}. "
            "Pass --trainable-dtype float64 only for a deliberate float64 run."
        )


def _enforce_trainable_float_dtype(
    model: Any,
    where_train_spec: Any,
    dtype: jnp.dtype,
    *,
    context: str,
) -> Any:
    model = _cast_trainable_float_leaves(model, where_train_spec, dtype)
    _assert_trainable_float_dtype(model, where_train_spec, dtype, context=context)
    return model


def _loss_for_batch(
    model: Any,
    batch: dict[str, jax.Array],
    config: CSH0DistillationConfig,
    *,
    student_forcing_fraction: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    teacher_actions = batch["teacher_actions"]
    teacher_context = jnp.zeros_like(teacher_actions)
    teacher_context = teacher_context.at[:, 1:, :].set(teacher_actions[:, :-1, :])
    teacher_context = jax.lax.stop_gradient(teacher_context)
    teacher_forced_student = jax.lax.stop_gradient(
        _standard_model_actions(model, batch["feedback_history"])
    )
    student_context = jnp.zeros_like(teacher_actions)
    student_context = student_context.at[:, 1:, :].set(teacher_forced_student[:, :-1, :])
    action_context = (
        1.0 - student_forcing_fraction
    ) * teacher_context + student_forcing_fraction * student_context
    perturbation_context = action_context
    teacher = BankedAffineTeacher(
        base_feedback=batch["feedback_history"],
        base_actions=teacher_actions,
        feedback_gains=batch["feedback_gains"],
    )

    def student_policy(feedback_history: jax.Array, action_history: jax.Array) -> jax.Array:
        del action_history
        return _standard_model_actions(model, feedback_history)

    result = guided_distillation_loss(
        student_policy=student_policy,
        teacher_policy=teacher,
        feedback_history=batch["feedback_history"],
        action_history=action_context,
        config=config,
        perturbation_feedback_history=batch["perturbation_feedback_history"],
        perturbation_action_history=perturbation_context,
        feedback_directions=batch["feedback_directions"],
        action_directions=batch["action_directions"],
        student_forced_rollout=_standard_model_actions(model, batch["feedback_history"]),
        rollout_anchor=teacher_actions,
    )
    return result.total, result.components


@eqx.filter_jit
def _train_step(
    trainable_model: Any,
    frozen_model: Any,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: dict[str, jax.Array],
    config: CSH0DistillationConfig,
    student_forcing_fraction: float,
) -> tuple[Any, optax.OptState, jax.Array, dict[str, jax.Array]]:
    def loss_for_trainable(trainable: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        return _loss_for_batch(
            eqx.combine(trainable, frozen_model),
            batch,
            config,
            student_forcing_fraction=student_forcing_fraction,
        )

    (loss, components), grads = eqx.filter_value_and_grad(
        loss_for_trainable,
        has_aux=True,
    )(trainable_model)
    updates, optimizer_state = optimizer.update(
        grads,
        optimizer_state,
        trainable_model,
    )
    return eqx.apply_updates(trainable_model, updates), optimizer_state, loss, components


def _init_standard_model_ensemble(
    *,
    hps: TreeNamespace,
    key: jax.Array,
) -> Any:
    import rlrmp.analysis  # noqa: F401
    from rlrmp.train.task_model import setup_task_model_pair

    return setup_task_model_pair(hps, key=key).model


def _init_optimizer_state(
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    where_train_spec: Any,
) -> optax.OptState:
    trainable, _frozen = eqx.partition(model, where_train_spec)
    return eqx.filter_vmap(optimizer.init)(trainable)


def _replicate_keys(root_key: jax.Array, *, offset: int, n_replicates: int) -> jax.Array:
    replicate_indices = jnp.arange(n_replicates, dtype=jnp.uint32)
    return jax.vmap(lambda index: jr.fold_in(root_key, index + offset))(replicate_indices)


def _materialize_replicate_batches(
    package: dict[str, jax.Array],
    *,
    keys: jax.Array,
    batch_size: int,
    horizon: int,
    n_jvp_directions: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    split_keys = jax.vmap(lambda key: jr.split(key, 2))(keys)
    next_keys = split_keys[:, 0]
    materialize_keys = split_keys[:, 1]
    batches = eqx.filter_vmap(
        lambda key: materialize_teacher_batch(
            package,
            key=key,
            batch_size=batch_size,
            horizon=horizon,
            n_jvp_directions=n_jvp_directions,
        )
    )(materialize_keys)
    return next_keys, batches


@eqx.filter_jit
def _batched_train_step(
    models: Any,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    where_train_spec: Any,
    batches: dict[str, jax.Array],
    config: CSH0DistillationConfig,
    student_forcing_fraction: float,
) -> tuple[Any, optax.OptState, jax.Array, dict[str, jax.Array]]:
    trainable_models, frozen_models = eqx.partition(models, where_train_spec)
    trainable_models, optimizer_state, losses, components = eqx.filter_vmap(
        lambda trainable_model, frozen_model, state, batch: _train_step(
            trainable_model,
            frozen_model,
            state,
            optimizer,
            batch,
            config,
            student_forcing_fraction,
        )
    )(trainable_models, frozen_models, optimizer_state, batches)
    return eqx.combine(trainable_models, frozen_models), optimizer_state, losses, components
