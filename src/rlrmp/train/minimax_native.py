"""Native-executor kernels for the RLRMP minimax training method."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
import optax
from feedbax import prepare_trial
from feedbax.contracts.training import ObjectiveSlotSpec
from feedbax.objectives.loss import AbstractLoss
from feedbax.objectives.service import LossService, LoweredObjective
from feedbax.objectives.spec import ObjectiveExecutionRequirements
from feedbax.objectives.streaming import make_streaming_loss_fn
from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.iteration import run_component
from feedbax.training.train import TaskTrainer, make_delayed_cosine_schedule, train_pair

from rlrmp.intervention_compat import LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET
from rlrmp.train.adversarial_training import (
    _inject_adversary_delta_A,
    _inject_adversary_forces,
)
from rlrmp.train.adversary import GaussianBumpAdversary, LinearDynamicsAdversary
from rlrmp.train.executor.adapters import ChunkKernelAdapter, UpdateKernel
from rlrmp.train.executor.guards import make_stop_after_batches_predicate
from rlrmp.train.executor.initial_slots import RlrmpRuntime, split_initial_keys
from rlrmp.train.executor.slots import (
    ADVERSARY_LOSS,
    ADVERSARY_OPTIMIZER,
    ADVERSARY_POPULATION,
    CONTROLLER,
    CONTROLLER_LOSS,
    CONTROLLER_OPTIMIZER,
    OBJECTIVE,
    RNG,
    TRIAL_BATCH,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.model.trainable import staged_network_trainable_parts

WARMUP_KERNEL_REF = "rlrmp.minimax.warmup_controller_descent"
INNER_ASCENT_KERNEL_REF = "rlrmp.minimax.inner_adversary_ascent"
PROJECTION_KERNEL_REF = "rlrmp.minimax.frobenius_ball_projection"
OUTER_DESCENT_KERNEL_REF = "rlrmp.minimax.outer_controller_descent"
NO_ADVERSARIAL_BATCHES_REF = "rlrmp.minimax.no_adversarial_batches"
ADVERSARIAL_COMPLETE_REF = "rlrmp.minimax.adversarial_complete"


@dataclass(frozen=True)
class MinimaxPreparedBatch:
    """One adversarial batch shared by the inner and outer minimax kernels."""

    trial_specs: Any
    trial_keys: Any
    batch_index: int
    active_member_index: int


@dataclass(frozen=True)
class MinimaxControllerLayout:
    """Static model layout used to keep executor controller slots array-only."""

    treedef_single: Any
    treedef_ensembled: Any
    is_per_replicate: tuple[bool, ...]
    shared_leaves: tuple[Any, ...]


class MinimaxControllerState(NamedTuple):
    """Pickleable controller slot: only dynamic per-replicate array leaves."""

    per_replicate_leaves: tuple[Any, ...]


@dataclass(frozen=True)
class MinimaxNativeRuntime:
    """Runtime-only minimax objects passed through Feedbax kernel context."""

    hps: Any
    args: Any
    pair: Any
    controller_layout: MinimaxControllerLayout
    key_warmup: Any
    adversary_optimizer: Any
    controller_optimizer: Any
    adv_batch_size: int

    @property
    def n_replicates(self) -> int:
        return int(self.hps.model.n_replicates)

    @property
    def use_linear_dynamics(self) -> bool:
        return self.args.adversary_type == "linear_dynamics"


class MinimaxExternalObjectiveLoss(AbstractLoss):
    """Placeholder lowered loss for minimax's runtime-owned external objective."""

    label: str = "rlrmp_minimax_external_objective"

    def term(self, states: Any, trial_specs: Any, model: Any) -> Any:
        del states, trial_specs, model
        return jnp.asarray(0.0)


class MinimaxExternalObjectiveLossService(LossService):
    """Lower the governed minimax external objective for native execution."""

    def lower_objective_slot(
        self,
        slot: ObjectiveSlotSpec,
        *,
        graph: Any = None,
        trial_axis: str = "batch",
        path: str = "/objective",
    ) -> LoweredObjective:
        if slot.kind == "external" and slot.schema_id == "rlrmp.minimax_objective":
            del graph, trial_axis, path
            return LoweredObjective(
                loss=MinimaxExternalObjectiveLoss(),
                requirements=ObjectiveExecutionRequirements(),
                source_kind="objective_spec",
            )
        return super().lower_objective_slot(
            slot,
            graph=graph,
            trial_axis=trial_axis,
            path=path,
        )


def build_minimax_native_initial_slots(
    *,
    run_spec: Mapping[str, Any],
    hps: Any,
    args: Any,
    key: Any,
) -> tuple[dict[str, Any], RlrmpRuntime]:
    """Return native minimax slots plus runtime context for one run."""

    del run_spec
    key_init, key_warmup, key_adv = split_initial_keys(key)
    pair = setup_task_model_pair(hps, key=key_init)
    controller_layout = _controller_layout(pair.model, int(hps.model.n_replicates))
    adv_batch_size = int(args.adv_batch_size or hps.batch_size)
    adversary_optimizer = optax.adam(
        args.linear_dynamics_lr if args.adversary_type == "linear_dynamics" else args.adversary_lr
    )
    controller_optimizer = optax.adamw(args.controller_lr, weight_decay=0.0)
    adversaries = _make_adversary_population(args=args, hps=hps)
    adv_opt_states = [
        _init_vmapped_adversary_opt_state(
            adversary,
            adversary_optimizer=adversary_optimizer,
            n_replicates=int(hps.model.n_replicates),
        )
        for adversary in adversaries
    ]
    runtime = MinimaxNativeRuntime(
        hps=hps,
        args=args,
        pair=pair,
        controller_layout=controller_layout,
        key_warmup=key_warmup,
        adversary_optimizer=adversary_optimizer,
        controller_optimizer=controller_optimizer,
        adv_batch_size=adv_batch_size,
    )
    return (
        {
            CONTROLLER: _controller_state_from_model(pair.model, controller_layout),
            CONTROLLER_OPTIMIZER: _init_controller_opt_state(pair.model, runtime),
            ADVERSARY_POPULATION: adversaries,
            ADVERSARY_OPTIMIZER: adv_opt_states,
            RNG: key_adv,
            TRIAL_BATCH: None,
            OBJECTIVE: None,
            CONTROLLER_LOSS: 0.0,
            ADVERSARY_LOSS: 0.0,
        },
        RlrmpRuntime(components={"minimax": runtime}),
    )


def minimax_update_kernels(payload: Any) -> Mapping[str, UpdateKernel]:
    """Return Feedbax update kernels for the minimax method payload."""

    return {
        WARMUP_KERNEL_REF: ChunkKernelAdapter(
            chunk_fn=_warmup_controller_descent,
            reads=(CONTROLLER, CONTROLLER_OPTIMIZER, RNG),
            writes=(CONTROLLER, CONTROLLER_OPTIMIZER, RNG, CONTROLLER_LOSS),
            metric_slots=(CONTROLLER_LOSS,),
            name="minimax warmup controller descent",
        ).to_kernel(payload),
        INNER_ASCENT_KERNEL_REF: ChunkKernelAdapter(
            chunk_fn=_inner_adversary_ascent,
            reads=(CONTROLLER, ADVERSARY_POPULATION, ADVERSARY_OPTIMIZER, RNG),
            writes=(ADVERSARY_POPULATION, ADVERSARY_OPTIMIZER, TRIAL_BATCH, RNG, ADVERSARY_LOSS),
            metric_slots=(ADVERSARY_LOSS,),
            prng_slot=RNG,
            name="minimax inner adversary ascent",
        ).to_kernel(payload),
        PROJECTION_KERNEL_REF: ChunkKernelAdapter(
            chunk_fn=_frobenius_ball_projection,
            reads=(ADVERSARY_POPULATION, TRIAL_BATCH),
            writes=(ADVERSARY_POPULATION,),
            name="minimax adversary projection",
        ).to_kernel(payload),
        OUTER_DESCENT_KERNEL_REF: ChunkKernelAdapter(
            chunk_fn=_outer_controller_descent,
            reads=(CONTROLLER, CONTROLLER_OPTIMIZER, ADVERSARY_POPULATION, TRIAL_BATCH, RNG),
            writes=(CONTROLLER, CONTROLLER_OPTIMIZER, RNG, CONTROLLER_LOSS),
            metric_slots=(CONTROLLER_LOSS,),
            name="minimax outer controller descent",
        ).to_kernel(payload),
    }


def minimax_guard_predicates(payload: Any) -> Mapping[str, UpdateKernel]:
    """Return minimax phase-transition predicates."""

    config = payload.config if payload is not None else None
    n_adversary_batches = int(getattr(config, "n_adversary_batches", 0))
    stop_after = make_stop_after_batches_predicate(completed_slot="_unused_minimax_completed")

    def no_adversarial_batches(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> bool:
        del slots, coordinate, context
        return n_adversary_batches == 0

    def adversarial_complete(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> bool:
        del slots
        runtime_stop = _minimax_runtime_stop_after_batches(stop_after, coordinate, context)
        completed = max(0, int(coordinate.global_step) - 1)
        return completed >= n_adversary_batches or runtime_stop

    return {
        NO_ADVERSARIAL_BATCHES_REF: no_adversarial_batches,
        ADVERSARIAL_COMPLETE_REF: adversarial_complete,
    }


def _warmup_controller_descent(
    runtime: RlrmpRuntime,
    payload: Any,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del payload, coordinate
    minimax = _runtime(runtime)
    args = minimax.args
    controller = _model_from_controller_state(
        chunk_slots[CONTROLLER],
        minimax.controller_layout,
        ensembled=True,
    )
    history = None
    if int(args.n_warmup_batches) > 0:
        pair = minimax.pair._replace(model=controller)
        warmup_schedule = make_delayed_cosine_schedule(
            args.controller_lr,
            constant_steps=0,
            total_steps=args.n_warmup_batches,
        )
        warmup_optimizer = optax.inject_hyperparams(
            lambda learning_rate: optax.adamw(learning_rate, weight_decay=0.0)
        )(learning_rate=warmup_schedule)
        trainer = TaskTrainer(optimizer=warmup_optimizer, checkpointing=False)
        controller, history = train_pair(
            trainer,
            pair,
            n_batches=args.n_warmup_batches,
            key=minimax.key_warmup,
            ensembled=True,
            loss_func=minimax.pair.task.loss_func,
            where_train=_make_where_train(sisu_gating=args.sisu_gating),
            batch_size=minimax.hps.batch_size,
            log_step=max(1, args.n_warmup_batches),
        )
    ctrl_opt_state = _init_controller_opt_state(controller, minimax)
    controller_loss = _last_history_loss(history)
    return {
        CONTROLLER: _controller_state_from_model(controller, minimax.controller_layout),
        CONTROLLER_OPTIMIZER: ctrl_opt_state,
        RNG: chunk_slots[RNG],
        CONTROLLER_LOSS: controller_loss,
    }


def _inner_adversary_ascent(
    runtime: RlrmpRuntime,
    payload: Any,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del payload
    minimax = _runtime(runtime)
    batch_index = max(0, int(coordinate.global_step) - 1)
    if batch_index >= int(minimax.args.n_adversary_batches):
        return {
            ADVERSARY_POPULATION: chunk_slots[ADVERSARY_POPULATION],
            ADVERSARY_OPTIMIZER: chunk_slots[ADVERSARY_OPTIMIZER],
            TRIAL_BATCH: None,
            ADVERSARY_LOSS: float(chunk_slots.get(ADVERSARY_LOSS, 0.0)),
        }
    prepared = _prepare_adversarial_batch(minimax, chunk_slots[RNG], batch_index)
    adversaries = list(chunk_slots[ADVERSARY_POPULATION])
    adv_opt_states = list(chunk_slots[ADVERSARY_OPTIMIZER])
    active_adv = adversaries[prepared.active_member_index]
    active_opt = adv_opt_states[prepared.active_member_index]
    if minimax.use_linear_dynamics:
        active_adv, active_opt, adv_loss_vals = _vmapped_linear_adversary_ascent(
            minimax,
            chunk_slots[CONTROLLER],
            active_adv,
            active_opt,
            prepared.trial_specs,
            prepared.trial_keys,
        )
    else:
        active_adv, active_opt, adv_loss_vals = _vmapped_gaussian_adversary_ascent(
            minimax,
            chunk_slots[CONTROLLER],
            active_adv,
            active_opt,
            prepared.trial_specs,
            prepared.trial_keys,
        )
    adversaries[prepared.active_member_index] = active_adv
    adv_opt_states[prepared.active_member_index] = active_opt
    return {
        ADVERSARY_POPULATION: adversaries,
        ADVERSARY_OPTIMIZER: adv_opt_states,
        TRIAL_BATCH: prepared,
        ADVERSARY_LOSS: _metric_mean(adv_loss_vals),
    }


def _frobenius_ball_projection(
    runtime: RlrmpRuntime,
    payload: Any,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del runtime, payload, coordinate
    prepared = chunk_slots.get(TRIAL_BATCH)
    adversaries = list(chunk_slots[ADVERSARY_POPULATION])
    if prepared is not None:
        active_adv = adversaries[prepared.active_member_index]
        if hasattr(active_adv, "project"):
            adversaries[prepared.active_member_index] = eqx.filter_vmap(lambda adv: adv.project())(
                active_adv
            )
    return {ADVERSARY_POPULATION: adversaries}


def _outer_controller_descent(
    runtime: RlrmpRuntime,
    payload: Any,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del payload, coordinate
    minimax = _runtime(runtime)
    prepared = chunk_slots.get(TRIAL_BATCH)
    if prepared is None:
        return {
            CONTROLLER: chunk_slots[CONTROLLER],
            CONTROLLER_OPTIMIZER: chunk_slots[CONTROLLER_OPTIMIZER],
            RNG: chunk_slots[RNG],
            CONTROLLER_LOSS: float(chunk_slots.get(CONTROLLER_LOSS, 0.0)),
        }
    adversary = chunk_slots[ADVERSARY_POPULATION][prepared.active_member_index]
    controller, ctrl_opt_state, ctrl_loss_vals = _vmapped_controller_descent(
        minimax,
        chunk_slots[CONTROLLER],
        chunk_slots[CONTROLLER_OPTIMIZER],
        adversary,
        prepared.trial_specs,
        prepared.trial_keys,
    )
    return {
        CONTROLLER: controller,
        CONTROLLER_OPTIMIZER: ctrl_opt_state,
        RNG: chunk_slots[RNG],
        CONTROLLER_LOSS: _metric_mean(ctrl_loss_vals),
    }


def _make_adversary_population(*, args: Any, hps: Any) -> list[Any]:
    n_timesteps = hps.task.n_steps - 1
    n_replicates = int(hps.model.n_replicates)
    adversaries: list[Any] = []
    for index in range(int(args.n_adversaries)):
        rep_keys = jr.split(jr.PRNGKey(7 + index), n_replicates)
        if args.adversary_type == "gaussian_bump":
            adversary = eqx.filter_vmap(
                lambda key: GaussianBumpAdversary(
                    n_bumps=args.n_bumps,
                    n_timesteps=n_timesteps,
                    n_force_dims=2,
                    force_max=args.force_max,
                    dt=hps.dt,
                    key=key,
                )
            )(rep_keys)
        elif args.adversary_type == "linear_dynamics":
            adversary = eqx.filter_vmap(
                lambda key: LinearDynamicsAdversary(
                    n_state=4,
                    n_dim=2,
                    eta_max=args.linear_dynamics_eta_max,
                    n_inner_steps=args.n_adversary_steps,
                    learning_rate=args.linear_dynamics_lr,
                    key=key,
                )
            )(rep_keys)
        else:
            raise ValueError(f"Unknown adversary_type: {args.adversary_type!r}")
        adversaries.append(adversary)
    return adversaries


def _prepare_adversarial_batch(
    runtime: MinimaxNativeRuntime,
    key: Any,
    batch_index: int,
) -> MinimaxPreparedBatch:
    trial_keys = jr.split(key, runtime.adv_batch_size)
    batch_info = BatchInfo(
        size=jnp.int32(runtime.adv_batch_size),
        current=jnp.int32(batch_index),
        total=jnp.int32(runtime.args.n_adversary_batches),
    )
    trial_specs = jax.vmap(
        lambda trial_key: runtime.pair.task.get_train_trial_with_intervenor_params(
            trial_key,
            batch_info,
        )
    )(trial_keys)
    trial_specs = eqx.tree_at(
        lambda specs: specs.timeline.n_steps,
        trial_specs,
        int(runtime.hps.task.n_steps),
    )
    return MinimaxPreparedBatch(
        trial_specs=trial_specs,
        trial_keys=trial_keys,
        batch_index=batch_index,
        active_member_index=batch_index % int(runtime.args.n_adversaries),
    )


def _vmapped_gaussian_adversary_ascent(
    runtime: MinimaxNativeRuntime,
    controller: Any,
    adversary: Any,
    adv_opt_state: Any,
    trial_specs: Any,
    trial_keys: Any,
) -> tuple[Any, Any, Any]:
    loss_vals = jnp.zeros(runtime.n_replicates)
    for _ in range(int(runtime.args.n_adversary_steps)):
        loss_vals, d_forces = eqx.filter_vmap(
            lambda model, adv: _single_rep_loss_and_force_grad(
                runtime,
                model,
                adv,
                trial_specs,
                trial_keys,
            )
        )(controller.per_replicate_leaves, adversary)
        adversary, adv_opt_state = eqx.filter_vmap(
            lambda adv, grad, opt_state: _adversary_update(
                runtime.adversary_optimizer,
                adv,
                grad,
                opt_state,
            )
        )(adversary, d_forces, adv_opt_state)
    return adversary, adv_opt_state, loss_vals


def _vmapped_linear_adversary_ascent(
    runtime: MinimaxNativeRuntime,
    controller: Any,
    adversary: Any,
    adv_opt_state: Any,
    trial_specs: Any,
    trial_keys: Any,
) -> tuple[Any, Any, Any]:
    models = controller.per_replicate_leaves
    loss_vals = jnp.zeros(runtime.n_replicates)
    for _ in range(int(runtime.args.n_adversary_steps)):
        adversary, adv_opt_state, loss_vals = eqx.filter_vmap(
            lambda model, adv, opt_state: _single_rep_linear_adversary_step(
                runtime,
                model,
                adv,
                opt_state,
                trial_specs,
                trial_keys,
            )
        )(models, adversary, adv_opt_state)
    return adversary, adv_opt_state, loss_vals


def _vmapped_controller_descent(
    runtime: MinimaxNativeRuntime,
    controller: Any,
    ctrl_opt_state: Any,
    adversary: Any,
    trial_specs: Any,
    trial_keys: Any,
) -> tuple[Any, Any, Any]:
    models = controller.per_replicate_leaves
    updated_models, updated_opt_state, loss_vals = eqx.filter_vmap(
        lambda model, opt_state, adv: _single_rep_controller_step(
            runtime,
            model,
            opt_state,
            adv,
            trial_specs,
            trial_keys,
        )
    )(models, ctrl_opt_state, adversary)
    return MinimaxControllerState(tuple(updated_models)), updated_opt_state, loss_vals


def _single_rep_loss_and_force_grad(
    runtime: MinimaxNativeRuntime,
    per_rep_leaves: Any,
    adversary: Any,
    trial_specs: Any,
    trial_keys: Any,
) -> tuple[Any, Any]:
    model = _single_model_from_leaves(per_rep_leaves, runtime.controller_layout)
    model_sg = _stop_gradient_arrays(model)
    force_profile = adversary()
    forces = jnp.broadcast_to(force_profile, (runtime.adv_batch_size, *force_profile.shape))

    def loss_fn(force_batch: Any) -> Any:
        return _compute_loss(
            runtime,
            model_sg,
            _inject_adversary_forces(trial_specs, force_batch),
            trial_keys,
        )

    return jax.value_and_grad(loss_fn)(forces)


def _single_rep_linear_adversary_step(
    runtime: MinimaxNativeRuntime,
    per_rep_leaves: Any,
    adversary: Any,
    adv_opt_state: Any,
    trial_specs: Any,
    trial_keys: Any,
) -> tuple[Any, Any, Any]:
    model = _single_model_from_leaves(per_rep_leaves, runtime.controller_layout)
    model_sg = _stop_gradient_arrays(model)

    def loss_fn(adv: Any) -> Any:
        return _compute_loss(
            runtime,
            model_sg,
            _inject_adversary_delta_A(trial_specs, adv.delta_A, runtime.adv_batch_size),
            trial_keys,
        )

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(adversary)
    neg_grads = jt.map(lambda grad: -grad if eqx.is_array(grad) else grad, grads)
    updates, new_opt_state = runtime.adversary_optimizer.update(
        eqx.filter(neg_grads, eqx.is_array),
        adv_opt_state,
        eqx.filter(adversary, eqx.is_array),
    )
    new_adversary = eqx.apply_updates(adversary, updates).project()
    return new_adversary, new_opt_state, loss_val


def _single_rep_controller_step(
    runtime: MinimaxNativeRuntime,
    per_rep_leaves: Any,
    ctrl_opt_state: Any,
    adversary: Any,
    trial_specs: Any,
    trial_keys: Any,
) -> tuple[Any, Any, Any]:
    model = _single_model_from_leaves(per_rep_leaves, runtime.controller_layout)
    if runtime.use_linear_dynamics:
        adv_trial_specs = _inject_adversary_delta_A(
            trial_specs,
            adversary.delta_A,
            runtime.adv_batch_size,
        )
    else:
        force_profile = adversary()
        forces = jnp.broadcast_to(force_profile, (runtime.adv_batch_size, *force_profile.shape))
        adv_trial_specs = _inject_adversary_forces(trial_specs, forces)

    def ctrl_loss(model_candidate: Any) -> Any:
        return _compute_loss(runtime, model_candidate, adv_trial_specs, trial_keys)

    loss_val, grads = eqx.filter_value_and_grad(ctrl_loss)(model)
    trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
    updates, new_opt_state = runtime.controller_optimizer.update(
        trainable_grads,
        ctrl_opt_state,
        eqx.filter(_get_trainable(model), eqx.is_array),
    )
    updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
    new_model = eqx.tree_at(_trainable_where(model), model, updated_trainable)
    return tuple(_per_rep_leaves_from_single_model(new_model, runtime.controller_layout)), (
        new_opt_state
    ), loss_val


def _adversary_update(
    adversary_optimizer: Any,
    adversary: Any,
    d_loss_d_forces: Any,
    adv_opt_state: Any,
) -> tuple[Any, Any]:
    def forces_fn(candidate: Any) -> Any:
        force_profile = candidate()
        return jnp.broadcast_to(force_profile, d_loss_d_forces.shape)

    _, vjp_fn = jax.vjp(lambda candidate: eqx.filter(forces_fn(candidate), eqx.is_array), adversary)
    neg_d_forces = jt.map(lambda grad: -grad, d_loss_d_forces)
    (param_grads,) = vjp_fn(neg_d_forces)
    updates, new_opt_state = adversary_optimizer.update(
        eqx.filter(param_grads, eqx.is_array),
        adv_opt_state,
        eqx.filter(adversary, eqx.is_array),
    )
    return eqx.apply_updates(adversary, updates), new_opt_state


def _compute_loss(
    runtime: MinimaxNativeRuntime,
    model: Any,
    trial_specs: Any,
    trial_keys: Any,
) -> Any:
    if runtime.args.streaming_loss:
        return _eval_trials_streaming(
            runtime.pair.task,
            model,
            trial_specs,
            trial_keys,
            runtime.pair.task.loss_func,
        )
    states = _eval_trials_with_declarative_component_parameter_inputs(
        runtime.pair.task,
        model,
        trial_specs,
        trial_keys,
    )
    return runtime.pair.task.loss_func(states, trial_specs, model).total.mean()


def _eval_trials_with_declarative_component_parameter_inputs(
    task: Any,
    model: Any,
    trial_specs: Any,
    keys: Any,
) -> Any:
    del task

    def eval_single(trial_spec: Any, key: Any) -> Any:
        key_run = jr.split(key, 2)[1]
        prepared = prepare_trial(model, trial_spec)
        prepared_inputs = _with_declarative_component_parameter_inputs(
            model,
            prepared.inputs,
        )
        _outputs, _final_state, state_history = run_component(
            model,
            prepared_inputs,
            prepared.init_state,
            key=key_run,
            n_steps=prepared.n_steps,
        )
        return jt.map(lambda x: x[1:] if x is not None else x, state_history)

    return eqx.filter_vmap(eval_single)(trial_specs, keys)


def _eval_trials_streaming(
    task: Any,
    model: Any,
    trial_specs: Any,
    keys: Any,
    loss_func: Any,
) -> Any:
    def eval_single(trial_spec: Any, key: Any) -> Any:
        key_run = jr.split(key, 2)[1]
        prepared = prepare_trial(model, trial_spec)
        prepared_inputs = _with_declarative_component_parameter_inputs(
            model,
            prepared.inputs,
        )
        streaming_fn = make_streaming_loss_fn(loss_func, trial_spec, model, prepared.n_steps)
        _outputs, _final_state, total_loss = run_component(
            model,
            prepared_inputs,
            prepared.init_state,
            key=key_run,
            n_steps=prepared.n_steps,
            streaming_loss_fn=streaming_fn,
        )
        return total_loss

    return eqx.filter_vmap(eval_single)(trial_specs, keys).mean()


def _with_declarative_component_parameter_inputs(model: Any, inputs: Any) -> Any:
    target = LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET
    legacy_key = f"intervene:{target['task_parameter_label']}"
    declared_key = (
        f"task:{target['source_data_id']}->{target['target_node_id']}.{target['target_port']}"
    )
    if (
        isinstance(inputs, dict)
        and legacy_key in inputs
        and declared_key in getattr(model, "input_ports", ())
        and declared_key not in inputs
    ):
        primary_inputs = {
            key: value
            for key, value in inputs.items()
            if key != legacy_key and not key.startswith("task:")
        }
        return {"input": primary_inputs, declared_key: inputs[legacy_key]}
    return inputs


def _get_trainable(model: Any) -> Any:
    net = model.get_node("net")
    cls_name = type(net).__name__
    if cls_name == "AffineFeedbackController":
        if getattr(net, "feedforward", None) is not None:
            return model.get_node_attrs("net", "gain", "feedforward")
        return model.get_node_attrs("net", "gain")
    if cls_name == "LinearController":
        return model.get_node_attrs("net", "K")
    if cls_name == "LinearTrackerController":
        return model.get_node_attrs("net", "K", "u_ff")
    return staged_network_trainable_parts(net)


def _trainable_where(model: Any) -> Any:
    net = model.get_node("net")
    cls_name = type(net).__name__
    if cls_name == "AffineFeedbackController":
        if getattr(net, "feedforward", None) is not None:
            return lambda candidate: candidate.get_node_attrs("net", "gain", "feedforward")
        return lambda candidate: candidate.get_node_attrs("net", "gain")
    if cls_name == "LinearController":
        return lambda candidate: candidate.get_node_attrs("net", "K")
    if cls_name == "LinearTrackerController":
        return lambda candidate: candidate.get_node_attrs("net", "K", "u_ff")
    return lambda candidate: staged_network_trainable_parts(candidate.get_node("net"))


def _make_where_train(sisu_gating: str = "additive") -> dict[int, Any]:
    del sisu_gating
    return {0: _trainable_where}


def _init_vmapped_adversary_opt_state(
    adversary: Any,
    *,
    adversary_optimizer: Any,
    n_replicates: int,
) -> Any:
    single_adv = jt.map(
        lambda value: value[0] if (eqx.is_array(value) and value.ndim > 0) else value,
        adversary,
        is_leaf=eqx.is_array,
    )
    single_state = adversary_optimizer.init(eqx.filter(single_adv, eqx.is_array))
    return jt.map(
        lambda value: jnp.stack([value] * n_replicates) if eqx.is_array(value) else value,
        single_state,
        is_leaf=eqx.is_array,
    )


def _init_controller_opt_state(controller: Any, runtime: MinimaxNativeRuntime) -> Any:
    single_model = _single_replicate_model(
        controller,
        runtime.n_replicates,
        runtime.controller_layout,
    )
    single_state = runtime.controller_optimizer.init(
        eqx.filter(_get_trainable(single_model), eqx.is_array)
    )
    return jt.map(
        lambda value: jnp.stack([value] * runtime.n_replicates)
        if eqx.is_array(value)
        else value,
        single_state,
        is_leaf=eqx.is_array,
    )


def _controller_layout(model: Any, n_replicates: int) -> MinimaxControllerLayout:
    flat_ensembled, treedef_ensembled = jtu.tree_flatten(model)
    is_per_replicate = tuple(_has_rep_axis(leaf, n_replicates) for leaf in flat_ensembled)
    shared_leaves = tuple(
        leaf for leaf, per_rep in zip(flat_ensembled, is_per_replicate, strict=True) if not per_rep
    )
    single_model = _single_replicate_model(model, n_replicates, None)
    _, treedef_single = jtu.tree_flatten(single_model)
    return MinimaxControllerLayout(
        treedef_single=treedef_single,
        treedef_ensembled=treedef_ensembled,
        is_per_replicate=is_per_replicate,
        shared_leaves=shared_leaves,
    )


def _controller_state_from_model(
    model: Any,
    layout: MinimaxControllerLayout,
) -> MinimaxControllerState:
    flat_leaves = jtu.tree_flatten(model)[0]
    return MinimaxControllerState(
        tuple(
            leaf
            for leaf, per_rep in zip(flat_leaves, layout.is_per_replicate, strict=True)
            if per_rep
        )
    )


def _model_from_controller_state(
    state: MinimaxControllerState,
    layout: MinimaxControllerLayout,
    *,
    ensembled: bool,
) -> Any:
    if not ensembled:
        return _single_model_from_leaves(state.per_replicate_leaves, layout)
    flat_leaves: list[Any] = []
    per_rep_index = 0
    shared_index = 0
    for is_per_rep in layout.is_per_replicate:
        if is_per_rep:
            flat_leaves.append(state.per_replicate_leaves[per_rep_index])
            per_rep_index += 1
        else:
            flat_leaves.append(layout.shared_leaves[shared_index])
            shared_index += 1
    return jtu.tree_unflatten(layout.treedef_ensembled, flat_leaves)


def _single_model_from_leaves(
    per_replicate_leaves: Any,
    layout: MinimaxControllerLayout,
) -> Any:
    flat_leaves: list[Any] = []
    per_rep_index = 0
    shared_index = 0
    for is_per_rep in layout.is_per_replicate:
        if is_per_rep:
            flat_leaves.append(per_replicate_leaves[per_rep_index])
            per_rep_index += 1
        else:
            flat_leaves.append(layout.shared_leaves[shared_index])
            shared_index += 1
    return jtu.tree_unflatten(layout.treedef_single, flat_leaves)


def _per_rep_leaves_from_single_model(
    model: Any,
    layout: MinimaxControllerLayout,
) -> tuple[Any, ...]:
    flat_leaves = jtu.tree_flatten(model)[0]
    return tuple(
        leaf
        for leaf, per_rep in zip(flat_leaves, layout.is_per_replicate, strict=True)
        if per_rep
    )


def _single_replicate_model(
    controller: Any,
    n_replicates: int,
    layout: MinimaxControllerLayout | None,
) -> Any:
    if isinstance(controller, MinimaxControllerState):
        if layout is None:
            raise ValueError("layout is required for MinimaxControllerState reconstruction")
        return _single_model_from_leaves(
            tuple(leaf[0] for leaf in controller.per_replicate_leaves),
            layout,
        )
    return jt.map(
        lambda value: value[0] if _has_rep_axis(value, n_replicates) else value,
        controller,
        is_leaf=eqx.is_array,
    )


def _has_rep_axis(value: Any, n_replicates: int) -> bool:
    return bool(eqx.is_array(value) and value.ndim > 0 and value.shape[0] == n_replicates)


def _stop_gradient_arrays(value: Any) -> Any:
    return jt.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        value,
        is_leaf=eqx.is_array,
    )


def _metric_mean(values: Any) -> float:
    return float(jax.device_get(jnp.mean(values)))


def _last_history_loss(history: Any) -> float:
    if history is None:
        return 0.0
    loss = getattr(history, "loss", None)
    if loss is None:
        losses = getattr(history, "losses", None)
        if losses is None:
            return 0.0
        loss = losses
    array = jnp.asarray(loss)
    if array.size == 0:
        return 0.0
    return float(jax.device_get(array.reshape(-1)[-1]))


def _runtime(runtime: RlrmpRuntime) -> MinimaxNativeRuntime:
    minimax = runtime.component("minimax")
    if not isinstance(minimax, MinimaxNativeRuntime):
        raise ValueError("minimax native kernels require runtime component 'minimax'")
    return minimax


def _minimax_runtime_stop_after_batches(
    stop_after: UpdateKernel,
    coordinate: Any,
    context: Mapping[str, Any],
) -> bool:
    runtime = context.get("rlrmp_runtime")
    stop_after_batches = getattr(runtime, "stop_after_batches", None)
    if stop_after_batches is None:
        return False
    completed = max(0, int(coordinate.global_step) - 1)
    return completed >= int(stop_after_batches)
