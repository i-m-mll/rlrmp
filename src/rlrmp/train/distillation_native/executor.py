"""Native-executor kernels for RLRMP distillation training methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from feedbax.contracts.training import ObjectiveSlotSpec, TrainingRunSpec
from feedbax.objectives.loss import AbstractLoss
from feedbax.objectives.service import LossService, LoweredObjective
from feedbax.objectives.spec import ObjectiveExecutionRequirements
from feedbax.training.executor import execute_training_run_spec

from rlrmp.runtime.training_run_specs import FEEDBAX_TRAINING_RUN_SPEC_KEY
from rlrmp.train.distillation_native import closed_loop_kernel
from rlrmp.train.distillation_native import guided_kernel
from rlrmp.train.distillation_native.losses import (
    DistillationLossWeights,
    cs_h0_distillation_config,
)
from rlrmp.train.cs_nominal_gru import (
    _initial_training_state,
    _run_cs_supervised_training_chunk,
    make_delayed_cosine_schedule,
)
from rlrmp.train.executor.adapters import ChunkKernelAdapter, UpdateKernel
from rlrmp.train.executor.initial_slots import RlrmpRuntime
from rlrmp.train.minimax_native import (
    _controller_layout,
    _controller_state_from_model,
    _model_from_controller_state,
)

MODEL = "model"
OPTIMIZER = "optimizer"
PRNG = "prng"
COMPLETED_BATCHES = "completed_batches"
OBJECTIVE = "objective"
TRAIN_LOSS = "train_loss"
TEACHER_REFERENCE = "teacher_reference"
CLOSED_LOOP_ROLLOUT = "closed_loop_rollout"
TEACHER_BANK = "teacher_bank"
FORCING_SCHEDULE = "forcing_schedule"
JVP_PROBES = "jvp_probes"

CLOSED_LOOP_KERNEL_REF = "rlrmp.train.distillation_native.closed_loop_gradient_update"
GUIDED_KERNEL_REF = "rlrmp.train.distillation_native.guided_gradient_update"


@dataclass(frozen=True)
class ClosedLoopNativeRuntime:
    """Runtime-only closed-loop distillation objects."""

    source_run_spec: Mapping[str, Any]
    pair: Any
    model_layout: Any
    hps: Any
    optimizer: Any
    loss_func: Any
    batch_size: int
    n_batches: int


@dataclass(frozen=True)
class GuidedNativeRuntime:
    """Runtime-only guided distillation objects."""

    source_run_spec: Mapping[str, Any]
    model_layout: Any
    package: Mapping[str, Any]
    config: Any
    optimizer: Any
    where_train_spec: Any
    batch_size: int
    horizon: int
    n_jvp_directions: int
    n_batches: int


class DistillationExternalObjectiveLoss(AbstractLoss):
    """Placeholder lowered loss for runtime-owned distillation objectives."""

    label: str = "rlrmp_distillation_external_objective"

    def term(self, states: Any, trial_specs: Any, model: Any) -> Any:
        del states, trial_specs, model
        return jnp.asarray(0.0)


class DistillationExternalObjectiveLossService(LossService):
    """Lower RLRMP distillation external objectives for native execution."""

    def lower_objective_slot(
        self,
        slot: ObjectiveSlotSpec,
        *,
        graph: Any = None,
        trial_axis: str = "batch",
        path: str = "/objective",
    ) -> LoweredObjective:
        if slot.kind == "external" and slot.schema_id in {
            "rlrmp.closed_loop_distillation.objective",
            "rlrmp.guided_distillation.objective",
        }:
            del graph, trial_axis, path
            return LoweredObjective(
                loss=DistillationExternalObjectiveLoss(),
                requirements=ObjectiveExecutionRequirements(),
                source_kind="objective_spec",
            )
        return super().lower_objective_slot(
            slot,
            graph=graph,
            trial_axis=trial_axis,
            path=path,
        )


def build_distillation_native_initial_slots(
    *,
    source_run_spec: Mapping[str, Any],
    method: str,
    key: Any,
) -> tuple[dict[str, Any], RlrmpRuntime]:
    """Build native-executor initial slots and runtime context for distillation."""

    if method == "closed_loop_distillation":
        return _build_closed_loop_initial_slots(source_run_spec=source_run_spec, key=key)
    if method == "guided_distillation":
        return _build_guided_initial_slots(source_run_spec=source_run_spec, key=key)
    raise ValueError(f"unknown distillation method {method!r}")


def execute_distillation_training_run_spec_native(
    source_run_spec: Mapping[str, Any] | TrainingRunSpec,
    *,
    method: str | None = None,
    run_id: str | None = None,
    key: Any | None = None,
    manifest_root: Path | str | None = None,
    checkpoint_root: Path | str | None = None,
    resume: bool = False,
    stop_after_barrier: str | None = None,
    **kwargs: Any,
):
    """Execute a distillation ``TrainingRunSpec`` through Feedbax's native executor."""

    from rlrmp.runtime.training_run_specs import register_rlrmp_distillation_methods

    register_rlrmp_distillation_methods()
    if isinstance(source_run_spec, TrainingRunSpec):
        training_spec = source_run_spec
        if method is None:
            method = _method_from_training_spec(training_spec)
        source_payload: Mapping[str, Any] = training_spec.method_payload.payload
    else:
        source_payload = source_run_spec
        training_spec = TrainingRunSpec.model_validate(
            source_payload[FEEDBAX_TRAINING_RUN_SPEC_KEY]
        )
        if method is None:
            method = _method_from_source_spec(source_payload)
    seed = int(source_payload.get("seed", 0)) if key is None else None
    initial_slots, runtime = build_distillation_native_initial_slots(
        source_run_spec=source_payload,
        method=method,
        key=jr.PRNGKey(seed) if key is None else key,
    )
    return execute_training_run_spec(
        training_spec,
        run_id=run_id,
        initial_slots=initial_slots,
        kernel_context={"rlrmp_runtime": runtime},
        manifest_root=manifest_root,
        checkpoint_root=checkpoint_root,
        loss_service=kwargs.pop("loss_service", DistillationExternalObjectiveLossService()),
        resume=resume,
        stop_after_barrier=stop_after_barrier,
        **kwargs,
    )


def native_distillation_model_from_slot(
    model_slot: Any,
    *,
    source_run_spec: Mapping[str, Any],
    method: str,
    key: Any,
) -> Any:
    """Reconstruct a full model from a pickleable native distillation model slot."""

    initial_slots, runtime = build_distillation_native_initial_slots(
        source_run_spec=source_run_spec,
        method=method,
        key=key,
    )
    del initial_slots
    if method == "closed_loop_distillation":
        layout = _closed_loop_runtime(runtime).model_layout
    elif method == "guided_distillation":
        layout = _guided_runtime(runtime).model_layout
    else:
        raise ValueError(f"unknown distillation method {method!r}")
    return _model_from_controller_state(model_slot, layout, ensembled=True)


def distillation_update_kernels(method: str, payload: Any = None) -> Mapping[str, UpdateKernel]:
    """Return native update kernels for one distillation method."""

    del payload
    if method == "closed_loop_distillation":
        return {
            CLOSED_LOOP_KERNEL_REF: ChunkKernelAdapter(
                chunk_fn=_closed_loop_training_chunk,
                reads=(MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES),
                writes=(MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, TRAIN_LOSS),
                metric_slots=(TRAIN_LOSS,),
                name="closed-loop distillation native training",
            ).to_kernel(None)
        }
    if method == "guided_distillation":
        return {
            GUIDED_KERNEL_REF: ChunkKernelAdapter(
                chunk_fn=_guided_training_chunk,
                reads=(MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES),
                writes=(MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, TRAIN_LOSS),
                metric_slots=(TRAIN_LOSS,),
                name="guided distillation native training",
            ).to_kernel(None)
        }
    raise ValueError(f"unknown distillation method {method!r}")


def _build_closed_loop_initial_slots(
    *,
    source_run_spec: Mapping[str, Any],
    key: Any,
) -> tuple[dict[str, Any], RlrmpRuntime]:
    import rlrmp.analysis  # noqa: F401
    from rlrmp.train.task_model import setup_task_model_pair

    key_init, key_train = jr.split(key, 2)
    hps = closed_loop_kernel._training_hps_from_spec(source_run_spec)
    pair = setup_task_model_pair(hps, key=key_init)
    model_layout = _controller_layout(
        pair.model, int(source_run_spec["student_contract"]["n_replicates"])
    )
    optimizer = _closed_loop_optimizer(source_run_spec)
    initial = _initial_training_state(
        model=pair.model,
        trainer=optimizer,
        where_train=closed_loop_kernel._where_train_fn,
        key=key_train,
    )
    runtime = ClosedLoopNativeRuntime(
        source_run_spec=source_run_spec,
        pair=pair,
        model_layout=model_layout,
        hps=hps,
        optimizer=optimizer,
        loss_func=closed_loop_kernel.build_closed_loop_loss(source_run_spec),
        batch_size=int(source_run_spec["student_contract"]["batch_size"]),
        n_batches=int(source_run_spec["student_contract"]["n_train_batches"]),
    )
    return (
        {
            MODEL: _controller_state_from_model(pair.model, model_layout),
            OPTIMIZER: initial.optimizer_state,
            PRNG: key_train,
            COMPLETED_BATCHES: jnp.asarray(0, dtype=jnp.int32),
            OBJECTIVE: None,
            TEACHER_REFERENCE: None,
            CLOSED_LOOP_ROLLOUT: None,
            TRAIN_LOSS: 0.0,
        },
        RlrmpRuntime(components={"closed_loop_distillation": runtime}),
    )


def _closed_loop_optimizer(source_run_spec: Mapping[str, Any]) -> optax.GradientTransformation:
    student = source_run_spec["student_contract"]
    schedule = make_delayed_cosine_schedule(
        float(student["controller_lr"]),
        constant_steps=0,
        total_steps=max(1, int(student["n_train_batches"])),
        alpha=float(student["lr_cosine_alpha"]),
    )
    return optax.chain(
        optax.clip_by_global_norm(float(student["gradient_clip_norm"])),
        optax.inject_hyperparams(optax.adamw)(learning_rate=schedule, weight_decay=0.0),
    )


def _build_guided_initial_slots(
    *,
    source_run_spec: Mapping[str, Any],
    key: Any,
) -> tuple[dict[str, Any], RlrmpRuntime]:
    model_hps = source_run_spec.get("hps", {}).get("model", {})
    n_batches = int(
        source_run_spec["n_train_batches"]
        if "n_train_batches" in source_run_spec
        else source_run_spec["training_config"]["n_batches"]
    )
    batch_size = int(
        source_run_spec["batch_size"]
        if "batch_size" in source_run_spec
        else source_run_spec["training_config"]["batch_size"]
    )
    n_replicates = int(
        model_hps.get("n_replicates", source_run_spec["model_contract"]["n_replicates"])
    )
    hidden_size = int(
        model_hps.get("hidden_size", source_run_spec["model_contract"]["hidden_size"])
    )
    horizon = int(source_run_spec["teacher_bank"]["horizon"])
    n_jvp_directions = int(source_run_spec["distillation_surface"]["config"]["n_jvp_directions"])
    trainable_dtype = guided_kernel._dtype_from_name(
        guided_kernel._trainable_dtype_name(
            source_run_spec, _namespace_from_guided_spec(source_run_spec)
        )
    )
    population_mask_mode = str(
        model_hps.get(
            "population_mask_mode",
            source_run_spec["model_contract"].get("population_mask_mode", "plain_all_ones"),
        )
    )
    key_init, _unused = jr.split(key, 2)
    package = guided_kernel.load_teacher_package(
        source_run_spec["teacher_contract"]["teacher_package"],
        teacher_gains_key=str(source_run_spec["teacher_bank"]["teacher_gains_key"]),
    )
    hps = guided_kernel._standard_hps_from_spec(
        source_run_spec,
        n_replicates=n_replicates,
        hidden_size=hidden_size,
        batch_size=batch_size,
        n_batches=n_batches,
        controller_lr=float(
            source_run_spec.get("controller_lr", source_run_spec["optimizer"]["controller_lr"])
        ),
        lr_warmup_batches=int(source_run_spec["optimizer"]["lr_warmup_batches"]),
        lr_warmup_init_fraction=float(source_run_spec["optimizer"]["lr_warmup_init_fraction"]),
        lr_cosine_alpha=float(source_run_spec["optimizer"]["lr_cosine_alpha"]),
        gradient_clip_norm=float(source_run_spec["optimizer"]["gradient_clip_norm"]),
        trainable_dtype=trainable_dtype.name,
        population_mask_mode=population_mask_mode,
    )
    model = guided_kernel._init_standard_model_ensemble(hps=hps, key=key_init)
    where_train_spec = guided_kernel._where_train_spec(model)
    model = guided_kernel._enforce_trainable_float_dtype(
        model,
        where_train_spec,
        trainable_dtype,
        context="native guided-distillation model",
    )
    optimizer = guided_kernel._make_optimizer(
        learning_rate=float(source_run_spec["optimizer"]["controller_lr"]),
        n_batches=n_batches,
        warmup_batches=int(source_run_spec["optimizer"]["lr_warmup_batches"]),
        warmup_init_fraction=float(source_run_spec["optimizer"]["lr_warmup_init_fraction"]),
        cosine_alpha=float(source_run_spec["optimizer"]["lr_cosine_alpha"]),
        gradient_clip_norm=float(source_run_spec["optimizer"]["gradient_clip_norm"]),
    )
    optimizer_state = guided_kernel._init_optimizer_state(
        model=model,
        optimizer=optimizer,
        where_train_spec=where_train_spec,
    )
    model_layout = _controller_layout(model, n_replicates)
    runtime = GuidedNativeRuntime(
        source_run_spec=source_run_spec,
        model_layout=model_layout,
        package=package,
        config=_guided_loss_config(source_run_spec),
        optimizer=optimizer,
        where_train_spec=where_train_spec,
        batch_size=batch_size,
        horizon=horizon,
        n_jvp_directions=n_jvp_directions,
        n_batches=n_batches,
    )
    return (
        {
            MODEL: _controller_state_from_model(model, model_layout),
            OPTIMIZER: optimizer_state,
            PRNG: guided_kernel._replicate_keys(
                key,
                offset=10_000,
                n_replicates=n_replicates,
            ),
            COMPLETED_BATCHES: jnp.asarray(0, dtype=jnp.int32),
            OBJECTIVE: None,
            TEACHER_BANK: None,
            FORCING_SCHEDULE: None,
            JVP_PROBES: None,
            TRAIN_LOSS: 0.0,
        },
        RlrmpRuntime(components={"guided_distillation": runtime}),
    )


def _closed_loop_training_chunk(
    runtime: RlrmpRuntime,
    payload: Any,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del payload, coordinate
    native = _closed_loop_runtime(runtime)
    completed = int(chunk_slots[COMPLETED_BATCHES])
    remaining = max(0, native.n_batches - completed)
    if remaining == 0:
        return {
            MODEL: chunk_slots[MODEL],
            OPTIMIZER: chunk_slots[OPTIMIZER],
            PRNG: chunk_slots[PRNG],
            COMPLETED_BATCHES: jnp.asarray(completed, dtype=jnp.int32),
            TRAIN_LOSS: float(chunk_slots.get(TRAIN_LOSS, 0.0)),
        }
    model = _model_from_controller_state(
        chunk_slots[MODEL],
        native.model_layout,
        ensembled=True,
    )
    trained_model, history, optimizer_state = _run_cs_supervised_training_chunk(
        optimizer=native.optimizer,
        task=native.pair.task,
        loss_func=native.loss_func,
        model=model,
        optimizer_state=chunk_slots[OPTIMIZER],
        hps=native.hps,
        where_train=closed_loop_kernel._where_train_fn,
        key=chunk_slots[PRNG],
        start_batch=completed,
        chunk_batches=remaining,
        log_progress=False,
        log_every=max(1, remaining),
        pre_step_fn=None,
    )
    return {
        MODEL: _controller_state_from_model(trained_model, native.model_layout),
        OPTIMIZER: optimizer_state,
        PRNG: chunk_slots[PRNG],
        COMPLETED_BATCHES: jnp.asarray(native.n_batches, dtype=jnp.int32),
        TRAIN_LOSS: _last_history_loss(history),
    }


def _guided_training_chunk(
    runtime: RlrmpRuntime,
    payload: Any,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del payload, coordinate
    native = _guided_runtime(runtime)
    model = _model_from_controller_state(
        chunk_slots[MODEL],
        native.model_layout,
        ensembled=True,
    )
    optimizer_state = chunk_slots[OPTIMIZER]
    batch_keys = chunk_slots[PRNG]
    completed = int(chunk_slots[COMPLETED_BATCHES])
    last_loss = float(chunk_slots.get(TRAIN_LOSS, 0.0))
    for batch_index in range(completed, native.n_batches):
        batch_keys, batches = guided_kernel._materialize_replicate_batches(
            native.package,
            keys=batch_keys,
            batch_size=native.batch_size,
            horizon=native.horizon,
            n_jvp_directions=native.n_jvp_directions,
        )
        model, optimizer_state, losses, _components = guided_kernel._batched_train_step(
            model,
            optimizer_state,
            native.optimizer,
            native.where_train_spec,
            batches,
            native.config,
            guided_kernel.forcing_fraction_for_batch(native.source_run_spec, batch_index),
        )
        last_loss = float(np.mean(np.asarray(jax.device_get(losses))))
    return {
        MODEL: _controller_state_from_model(model, native.model_layout),
        OPTIMIZER: optimizer_state,
        PRNG: batch_keys,
        COMPLETED_BATCHES: jnp.asarray(native.n_batches, dtype=jnp.int32),
        TRAIN_LOSS: last_loss,
    }


def _guided_loss_config(source_run_spec: Mapping[str, Any]) -> Any:
    weights = source_run_spec["distillation_surface"]["config"]["weights"]
    return cs_h0_distillation_config(
        weights=DistillationLossWeights(
            clean_action=float(weights["clean_action"]),
            perturbation_response=float(weights["perturbation_response"]),
            input_output_jvp=float(weights["input_output_jvp"]),
            student_forced_rollout_anchor=float(weights["student_forced_rollout_anchor"]),
        ),
        n_jvp_directions=int(source_run_spec["distillation_surface"]["config"]["n_jvp_directions"]),
    )


def _namespace_from_guided_spec(source_run_spec: Mapping[str, Any]) -> Any:
    class _Namespace:
        pass

    namespace = _Namespace()
    namespace.trainable_dtype = source_run_spec["model_contract"].get("trainable_dtype", "float32")
    namespace.population_mask_mode = source_run_spec["model_contract"].get(
        "population_mask_mode",
        "plain_all_ones",
    )
    return namespace


def _last_history_loss(history: Any) -> float:
    if history is None:
        return 0.0
    loss = getattr(history, "loss", None)
    if loss is None:
        losses = getattr(history, "losses", None)
        if losses is None:
            return 0.0
        loss = losses
    term_scalar = _term_tree_scalar(loss)
    if term_scalar is not None:
        return term_scalar
    array = jnp.asarray(loss)
    if array.size == 0:
        return 0.0
    return float(jax.device_get(array.reshape(-1)[-1]))


def _term_tree_scalar(value: Any) -> float | None:
    term_value = getattr(value, "value", None)
    if term_value is not None:
        try:
            return float(jax.device_get(jnp.mean(jnp.asarray(term_value))))
        except (TypeError, ValueError):
            return None
    children = getattr(value, "children", None)
    if not children:
        return None
    total = 0.0
    found = False
    for child in children:
        child_value = _term_tree_scalar(child)
        if child_value is not None:
            total += float(getattr(child, "weight", 1.0)) * child_value
            found = True
    if not found:
        return None
    return total * float(getattr(value, "weight", 1.0))


def _closed_loop_runtime(runtime: RlrmpRuntime) -> ClosedLoopNativeRuntime:
    native = runtime.component("closed_loop_distillation")
    if not isinstance(native, ClosedLoopNativeRuntime):
        raise ValueError("closed-loop distillation kernels require runtime component")
    return native


def _guided_runtime(runtime: RlrmpRuntime) -> GuidedNativeRuntime:
    native = runtime.component("guided_distillation")
    if not isinstance(native, GuidedNativeRuntime):
        raise ValueError("guided distillation kernels require runtime component")
    return native


def _method_from_source_spec(source_run_spec: Mapping[str, Any]) -> str:
    if {"student_contract", "closed_loop_semantics", "loss_surface"} <= set(source_run_spec):
        return "closed_loop_distillation"
    if {"model_contract", "teacher_bank", "training_schedule", "distillation_surface"} <= set(
        source_run_spec
    ):
        return "guided_distillation"
    raise ValueError("source run spec does not carry a recognized distillation payload")


def _method_from_training_spec(training_spec: TrainingRunSpec) -> str:
    if training_spec.method_ref.key == "rlrmp/closed_loop_distillation/v1":
        return "closed_loop_distillation"
    if training_spec.method_ref.key == "rlrmp/guided_distillation/v1":
        return "guided_distillation"
    raise ValueError(f"unknown distillation method_ref {training_spec.method_ref.key!r}")
