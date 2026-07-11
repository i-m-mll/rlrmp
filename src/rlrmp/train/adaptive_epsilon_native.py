"""Native-executor contract and kernels for adaptive-epsilon GRU training."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
import optax
from feedbax.contracts.checkpoints import (
    BatchIndexedCheckpointLeafSpec,
    CheckpointContinuationRequest,
)
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    ArtifactPolicySpec,
    CheckpointProgressPolicySpec,
    ExecutionPolicySpec,
    LrScheduleSpec,
    GraphTopologySourceSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    RiskAggregationSpec,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingRunSpec,
    WorkerExecutionSpec,
)
from feedbax.contracts.worker import (
    AxisSpec,
    CheckpointBarrierSpec,
    EffectivePhaseSpec,
    MetricGuardSpec,
    MethodContractSpec,
    OptimizerTargetBinding,
    PhaseProgramSpec,
    PhaseSpec,
    PhaseTransitionSpec,
    ResumeCoordinateSpec,
    StateSlotSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
    derive_consistency_predicate,
)
from feedbax.objectives.loss import AbstractLoss
from feedbax.objectives.service import LossService, LoweredObjective
from feedbax.objectives.spec import ObjectiveExecutionRequirements
from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.parameter_constraints import project_component_parameters
from feedbax.training.optimizers import (
    build_optimizer as build_feedbax_optimizer,
    learning_rate_schedule as feedbax_learning_rate_schedule,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlrmp.model.feedback_descriptors import DESCRIPTOR_PAYLOAD_KEY
from rlrmp.model.feedbax_graph import graph_spec_payload
from rlrmp.paths import portable_repo_path
from rlrmp.train.executor.adapters import ChunkKernelAdapter, RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.guards import make_stop_predicate
from rlrmp.train.executor.initial_slots import RlrmpRuntime, split_initial_keys
from rlrmp.train.executor.slots import (
    ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES,
    ADAPTIVE_EPSILON_SCHEMA,
    ADAPTIVE_EPSILON_STATE,
    COMPLETED_BATCHES,
    DAMAGE_METRIC,
    EPSILON_SCALE,
    HISTORY_CHUNK_BYTES,
    MODEL,
    OBJECTIVE,
    OPTIMIZER,
    PRNG,
    TRAIN_LOSS,
    ZERO_ADVERSARY_GUARD,
    artifact_sink_specs,
    checkpoint_slot_specs,
    supervised_state_slots,
)

ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_ID = (
    "rlrmp.spec.training_method.adaptive_epsilon_curriculum_payload"
)
ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_VERSION = (
    "rlrmp.spec.training_method.adaptive_epsilon_curriculum_payload.v2"
)
TRAIN_CHUNK_KERNEL_REF = "rlrmp.adaptive_epsilon_curriculum.train_chunk"
STOP_PREDICATE_REF = "rlrmp.adaptive_epsilon_curriculum.stop"
TRAIN_CHUNK_BARRIER = "after_adaptive_epsilon_train_chunk"
LR_CONTINUATION_RESTART = "restart"
LR_CONTINUATION_CONTINUE = "continue"
LRContinuationMode = Literal["restart", "continue"]

# These paths are the six diagnostics whose final axis is the full training
# horizon in the adaptive controller optimizer.  They deliberately name the
# raw optimizer topology used by the nominal-to-adaptive fork: Feedbax extends
# those leaves before the target/post transform serializes the adaptive slot.
ADAPTIVE_EPSILON_BATCH_INDEXED_CHECKPOINT_LEAVES = (
    BatchIndexedCheckpointLeafSpec(slot=OPTIMIZER, tree_path="/1"),
    BatchIndexedCheckpointLeafSpec(slot=OPTIMIZER, tree_path="/2"),
    BatchIndexedCheckpointLeafSpec(slot=OPTIMIZER, tree_path="/3"),
    BatchIndexedCheckpointLeafSpec(slot=OPTIMIZER, tree_path="/30"),
    BatchIndexedCheckpointLeafSpec(slot=OPTIMIZER, tree_path="/31"),
    BatchIndexedCheckpointLeafSpec(slot=OPTIMIZER, tree_path="/32"),
)


class AdaptiveEpsilonMethodPayload(BaseModel):
    """Governed payload for adaptive-epsilon curriculum native training."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    controller_optimizer: OptimizerSpec | None = None
    n_train_batches: int = Field(gt=0)
    chunk_batches: int = Field(gt=0)
    controller_training_mode: str
    damage_schedule: dict[str, Any]
    lambda_update: dict[str, Any]
    outer_adversarial_weight: dict[str, Any]
    pgd_inner_maximizer: dict[str, Any]
    checkpointing: dict[str, Any]
    lr_continuation_mode: LRContinuationMode | None = None
    rlrmp_extension_payload: str = "rlrmp_run_spec"

    @model_validator(mode="after")
    def _validate_payload(self) -> "AdaptiveEpsilonMethodPayload":
        if not bool(self.config.get("adaptive_epsilon_curriculum", False)):
            raise ValueError("adaptive-epsilon native payload requires enabled curriculum")
        if self.damage_schedule.get("setpoint_basis") != "damage_to_clean_loss_ratio":
            raise ValueError(
                "adaptive-epsilon native payload requires damage_schedule.setpoint_basis="
                "'damage_to_clean_loss_ratio'; older absolute damage setpoints are "
                "archival-only and cannot be used for new launches"
            )
        if self.chunk_batches > self.n_train_batches:
            raise ValueError("chunk_batches cannot exceed n_train_batches")
        if self.controller_training_mode not in {
            "loss_blend",
            "epsilon_scaled_outer_training",
        }:
            raise ValueError(f"unknown adaptive-epsilon mode {self.controller_training_mode!r}")
        if (
            self.controller_optimizer is not None
            and self.controller_optimizer.lr_schedule is None
        ):
            raise ValueError("adaptive-epsilon controller_optimizer requires lr_schedule")
        return self


class SerializedPyTreeSlot:
    """Pickleable byte slot with a stable structural ABI repr."""

    def __init__(self, payload: bytes) -> None:
        self.payload = bytes(payload)

    def __repr__(self) -> str:
        return "SerializedPyTreeSlot(payload=<bytes>)"


@dataclass
class AdaptiveEpsilonNativeRuntime:
    """Runtime-only objects used by adaptive-epsilon native kernels."""

    hps: Any
    args: Any
    task: Any
    trainer: Any
    where_train: Any
    model_template: Any
    optimizer_template: Any
    run_spec: Mapping[str, Any] | TrainingRunSpec
    lr_continuation_mode: LRContinuationMode | None = None
    trainer_resume_context: "LRResumeContext | None" = None
    optimizer_hyperparams_aligned: bool = False
    history: Any | None = None
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LRResumeContext:
    """Schedule context used to align injected Optax counts with run steps."""

    mode: LRContinuationMode
    schedule_origin_step: int
    current_step: int
    optimizer_count_at_current_step: int


@dataclass(frozen=True)
class AdaptiveEpsilonControllerOptimizerBuild:
    """Shared optimizer-build result for execution and LR reporting."""

    optimizer: optax.GradientTransformation
    learning_rate_schedule: optax.Schedule


class AdaptiveEpsilonExternalObjectiveLoss(AbstractLoss):
    """Placeholder lowered loss for runtime-owned adaptive-epsilon training."""

    label: str = "rlrmp_adaptive_epsilon_external_objective"

    def term(self, states: Any, trial_specs: Any, model: Any) -> Any:
        del states, trial_specs, model
        return jnp.asarray(0.0)


class AdaptiveEpsilonExternalObjectiveLossService(LossService):
    """Lower the governed C&S GRU external objective for native execution."""

    def lower_objective_slot(
        self,
        slot: ObjectiveSlotSpec,
        *,
        graph: Any = None,
        trial_axis: str = "batch",
        path: str = "/objective",
    ) -> LoweredObjective:
        if slot.kind == "external" and slot.schema_id == "rlrmp.cs_gru_objective":
            del graph, trial_axis, path
            return LoweredObjective(
                loss=AdaptiveEpsilonExternalObjectiveLoss(),
                requirements=ObjectiveExecutionRequirements(),
                source_kind="objective_spec",
            )
        return super().lower_objective_slot(
            slot,
            graph=graph,
            trial_axis=trial_axis,
            path=path,
        )


def adaptive_epsilon_method_ref() -> MethodRefSpec:
    """Return the RLRMP method ref for adaptive-epsilon curriculum training."""

    return MethodRefSpec(package="rlrmp", name="adaptive_epsilon_curriculum", version="v1")


def ensure_adaptive_epsilon_training_method_registered() -> None:
    """Install the adaptive-epsilon native method in Feedbax's registry."""

    if ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF in (
        DEFAULT_TRAINING_METHOD_REGISTRY.available_keys()
    ):
        return
    DEFAULT_TRAINING_METHOD_REGISTRY.register(
        TrainingMethodRegistration(
            method_ref=ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
            payload_schema_id=ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=AdaptiveEpsilonMethodPayload,
            contract_factory=adaptive_epsilon_method_contract,
            update_kernels_factory=adaptive_epsilon_update_kernels,
            guard_predicates_factory=adaptive_epsilon_guard_predicates,
            rejected_payload_versions=(
                "rlrmp.spec.training_method.adaptive_epsilon_curriculum_payload.v0",
                "rlrmp.spec.training_method.adaptive_epsilon_curriculum_payload.v1",
            ),
            owner="rlrmp.train.adaptive_epsilon_native",
            package="rlrmp",
        )
    )


def adaptive_epsilon_method_payload(run_spec: Mapping[str, Any]) -> MethodPayloadEnvelope:
    """Return the governed method payload for an adaptive-epsilon C&S run."""

    from rlrmp.train.cs_nominal_gru import _args_values_from_run_spec

    hps = _mapping(run_spec, "hps")
    adaptive = _mapping(hps, "adaptive_epsilon_curriculum")
    config = _args_values_from_run_spec(dict(run_spec))
    for key in ("output_dir", "spec_dir"):
        if config.get(key) is not None:
            config[key] = portable_repo_path(str(config[key]))
    checkpointing = _mapping(run_spec, "checkpointing")
    for key in ("checkpoint_dir", "latest_checkpoint"):
        if checkpointing.get(key) is not None:
            checkpointing[key] = portable_repo_path(str(checkpointing[key]))
    payload = AdaptiveEpsilonMethodPayload(
        config=config,
        controller_optimizer=adaptive_epsilon_controller_optimizer_spec(run_spec),
        n_train_batches=int(run_spec.get("n_train_batches", 1)),
        chunk_batches=max(
            1,
            int(_mapping(run_spec, "checkpointing").get("interval_batches", 1)),
        ),
        controller_training_mode=str(adaptive.get("controller_training_mode", "loss_blend")),
        damage_schedule=_mapping(adaptive, "damage_schedule"),
        lambda_update=_mapping(adaptive, "lambda_update"),
        outer_adversarial_weight=_mapping(adaptive, "outer_adversarial_weight"),
        pgd_inner_maximizer=_mapping(
            _mapping(hps, "broad_epsilon_pgd_training"),
            "inner_maximizer",
        ),
        checkpointing=checkpointing,
        lr_continuation_mode=_lr_continuation_mode_from_run_spec(run_spec),
    )
    return MethodPayloadEnvelope(
        schema_id=ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=payload.model_dump(mode="json", exclude_none=True),
    )


def adaptive_epsilon_controller_optimizer_spec(
    run_spec: Mapping[str, Any],
) -> OptimizerSpec:
    """Return the Feedbax optimizer contract for an adaptive-epsilon controller."""

    hps = _mapping(run_spec, "hps")
    optimizer = _mapping(run_spec, "optimizer")
    learning_rate = _float_from_sources(
        optimizer,
        hps,
        keys=("learning_rate_0", "controller_lr", "learning_rate"),
        default=1e-2,
    )
    schedule_name = str(
        optimizer.get(
            "lr_schedule",
            hps.get("lr_schedule", "delayed_cosine"),
        )
    )
    total_steps = int(
        _float_from_sources(
            optimizer,
            hps,
            keys=("total_steps", "n_batches_condition", "n_train_batches"),
            default=float(run_spec.get("n_train_batches", 1)),
        )
    )
    constant_lr_iterations = int(
        _float_from_sources(
            optimizer,
            hps,
            keys=("constant_lr_iterations", "lr_warmup_batches"),
            default=0.0,
        )
    )
    schedule_kwargs: dict[str, Any] = {
        "kind": schedule_name,
        "learning_rate_0": learning_rate,
        "constant_lr_iterations": constant_lr_iterations,
        "warmup_init_fraction": _float_from_sources(
            optimizer,
            hps,
            keys=("warmup_init_fraction", "lr_warmup_init_fraction"),
            default=0.0,
        ),
        "cosine_annealing_alpha": _float_from_sources(
            optimizer,
            hps,
            keys=("cosine_annealing_alpha", "lr_cosine_alpha"),
            default=0.0,
        ),
    }
    if schedule_name != "constant":
        schedule_kwargs["total_steps"] = total_steps
    optimizer_type = str(optimizer.get("type", optimizer.get("name", "adamw")))
    params = {
        "weight_decay": _float_from_sources(
            optimizer,
            hps,
            keys=("weight_decay",),
            default=0.0,
        )
    }
    return OptimizerSpec(
        type=optimizer_type,
        params=params,
        lr_schedule=LrScheduleSpec(**schedule_kwargs),
    )


def build_adaptive_epsilon_training_run_spec(
    run_spec: Mapping[str, Any],
    *,
    graph_spec: Any,
    output_dir: Path,
    spec_dir: Path,
    training_config: TrainingConfig,
    objective: ObjectiveSlotSpec,
    task: TaskSpec,
    risk_aggregation: RiskAggregationSpec,
    method_extensions: Mapping[str, Any],
    execution: ExecutionPolicySpec,
    artifacts: ArtifactPolicySpec,
    checkpoint_progress: CheckpointProgressPolicySpec,
    metadata: Mapping[str, Any],
) -> TrainingRunSpec:
    """Build an adaptive-epsilon native ``TrainingRunSpec`` from the C&S payload."""

    del output_dir, spec_dir
    ensure_adaptive_epsilon_training_method_registered()
    method_payload = adaptive_epsilon_method_payload(run_spec)
    contract = adaptive_epsilon_method_contract()
    effective_phase = adaptive_epsilon_effective_phase_spec(contract)
    return TrainingRunSpec(
        graph=GraphTopologySourceSpec(
            inline=graph_spec_payload(graph_spec),
            schema_id=getattr(graph_spec, "schema_id", None),
            schema_version=getattr(graph_spec, "schema_version", None),
            metadata={
                "source": "materialized_runtime_graph",
                "native_method": ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
                DESCRIPTOR_PAYLOAD_KEY: metadata.get(DESCRIPTOR_PAYLOAD_KEY),
                "descriptor_basis_hash": metadata.get("descriptor_basis_hash"),
            },
        ),
        task=task,
        training_config=training_config,
        objective=objective,
        risk_aggregation=risk_aggregation,
        method_ref=adaptive_epsilon_method_ref(),
        method_payload=method_payload,
        method_extensions=dict(method_extensions),
        worker_execution=WorkerExecutionSpec(
            method_contract=contract,
            effective_phase=effective_phase,
            metadata={
                "native_executor": "feedbax.training.executor.execute_training_run_spec",
                "kernel_owner": "rlrmp.train.adaptive_epsilon_native",
            },
        ),
        execution=execution,
        artifacts=artifacts,
        checkpoint_progress=checkpoint_progress,
        metadata={
            **dict(metadata),
            "native_method": ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
        },
    )


def adaptive_epsilon_method_contract() -> MethodContractSpec:
    """Return the adaptive-epsilon native phase program."""

    reads = [
        MODEL,
        OPTIMIZER,
        PRNG,
        COMPLETED_BATCHES,
        ADAPTIVE_EPSILON_STATE,
        ZERO_ADVERSARY_GUARD,
        OBJECTIVE,
    ]
    writes = [
        MODEL,
        OPTIMIZER,
        PRNG,
        COMPLETED_BATCHES,
        ADAPTIVE_EPSILON_STATE,
        ZERO_ADVERSARY_GUARD,
        TRAIN_LOSS,
        DAMAGE_METRIC,
        EPSILON_SCALE,
        HISTORY_CHUNK_BYTES,
        ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES,
    ]
    program = PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="adaptive_epsilon_train_chunk",
                kind="outer_loop",
                reads=reads,
                writes=writes,
                update_steps=["adaptive_epsilon_train_chunk"],
                legal_next=["done", "adaptive_epsilon_train_chunk"],
                checkpoint_barrier=TRAIN_CHUNK_BARRIER,
                loop_axis="batch",
                metadata={
                    "native_kernel_granularity": "checkpoint_sized_adaptive_epsilon_chunk",
                    "lambda_update": "after_damage_eval_measurement",
                },
            ),
            PhaseSpec(
                name="done",
                kind="evaluation",
                reads=[
                    MODEL,
                    OPTIMIZER,
                    PRNG,
                    COMPLETED_BATCHES,
                    ADAPTIVE_EPSILON_STATE,
                    ZERO_ADVERSARY_GUARD,
                    TRAIN_LOSS,
                    DAMAGE_METRIC,
                    EPSILON_SCALE,
                ],
            ),
        ],
        initial_phase="adaptive_epsilon_train_chunk",
        update_steps=[
            UpdateStepSpec(
                name="adaptive_epsilon_train_chunk",
                kind="gradient",
                kernel=UpdateKernelSpec(kernel_ref=TRAIN_CHUNK_KERNEL_REF),
                reads=reads,
                writes=writes,
                axes=["batch", "replicate", "adversary_inner_step"],
                optimizer_binding="optimizer_to_model",
                metadata={
                    "controller_direction": "minimize",
                    "damage_eval": "fixed_nominal_batch_lambda_control",
                    "math_owner": (
                        "rlrmp.train.cs_nominal_gru._run_adaptive_epsilon_training_chunk"
                    ),
                },
            )
        ],
        transitions=[
            PhaseTransitionSpec(
                source="adaptive_epsilon_train_chunk",
                target="done",
                barrier=TRAIN_CHUNK_BARRIER,
                guard=MetricGuardSpec(
                    predicate_ref=STOP_PREDICATE_REF,
                    metric_slots=[TRAIN_LOSS, DAMAGE_METRIC, EPSILON_SCALE],
                    bookkeeping_slots=[COMPLETED_BATCHES, ZERO_ADVERSARY_GUARD],
                ),
            ),
            PhaseTransitionSpec(
                source="adaptive_epsilon_train_chunk",
                target="adaptive_epsilon_train_chunk",
                barrier=TRAIN_CHUNK_BARRIER,
            ),
        ],
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="optimizer_to_model",
                optimizer_slot=OPTIMIZER,
                target_slot=MODEL,
                direction="minimize",
                projection="after_step",
                phase_scope=["adaptive_epsilon_train_chunk"],
                objective_reads=[OBJECTIVE],
            )
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name=TRAIN_CHUNK_BARRIER,
                phase="adaptive_epsilon_train_chunk",
                slots=checkpoint_slot_specs(ADAPTIVE_EPSILON_SCHEMA),
                artifact_sinks=artifact_sink_specs(ADAPTIVE_EPSILON_SCHEMA),
                resume_coordinate=ResumeCoordinateSpec(
                    phase="adaptive_epsilon_train_chunk",
                    completed_barrier=TRAIN_CHUNK_BARRIER,
                ),
            )
        ],
        metadata={
            "phase_program_identity": "rlrmp.adaptive_epsilon_curriculum.chunked.v1",
            "checkpoint_barrier_policy": "after_each_adaptive_epsilon_chunk",
        },
    )
    return MethodContractSpec(
        method_ref=ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
        method_payload_schema_version=ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_VERSION,
        axes=[
            AxisSpec(name="batch", role="batch"),
            AxisSpec(name="replicate", role="replicate"),
            AxisSpec(name="adversary_inner_step", role="epoch"),
        ],
        state_slots=[
            *supervised_state_slots(ADAPTIVE_EPSILON_SCHEMA),
            StateSlotSpec(
                name=ADAPTIVE_EPSILON_STATE,
                role="auxiliary",
                metadata={"owner": "adaptive_epsilon_lambda_controller"},
            ),
            StateSlotSpec(
                name=ZERO_ADVERSARY_GUARD,
                role="auxiliary",
                metadata={"owner": "adaptive_epsilon_zero_adversary_guard"},
            ),
        ],
        phase_program=program,
        metadata={
            "runtime_context": "rlrmp_runtime.components.adaptive_epsilon",
            "supported_modes": ["loss_blend", "epsilon_scaled_outer_training"],
        },
    )


def adaptive_epsilon_effective_phase_spec(
    contract: MethodContractSpec | None = None,
) -> EffectivePhaseSpec:
    """Return the EffectivePhaseSpec for the adaptive-epsilon contract."""

    active_contract = contract or adaptive_epsilon_method_contract()
    return EffectivePhaseSpec(
        method_ref=active_contract.method_ref,
        axes=active_contract.axes,
        state_slots=active_contract.state_slots,
        phase_program=active_contract.phase_program,
        consistency_predicate=derive_consistency_predicate(active_contract.phase_program),
    )


def adaptive_epsilon_update_kernels(
    payload: BaseModel | None = None,
) -> Mapping[str, Any]:
    """Return Feedbax update kernels for adaptive-epsilon training."""

    return {
        TRAIN_CHUNK_KERNEL_REF: ChunkKernelAdapter(
            chunk_fn=_adaptive_epsilon_train_chunk,
            reads=(
                MODEL,
                OPTIMIZER,
                PRNG,
                COMPLETED_BATCHES,
                ADAPTIVE_EPSILON_STATE,
                ZERO_ADVERSARY_GUARD,
            ),
            writes=(
                MODEL,
                OPTIMIZER,
                PRNG,
                COMPLETED_BATCHES,
                ADAPTIVE_EPSILON_STATE,
                ZERO_ADVERSARY_GUARD,
                TRAIN_LOSS,
                DAMAGE_METRIC,
                EPSILON_SCALE,
                HISTORY_CHUNK_BYTES,
                ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES,
            ),
            metric_slots=(TRAIN_LOSS, DAMAGE_METRIC, EPSILON_SCALE),
            prng_slot=PRNG,
            name="adaptive-epsilon curriculum train chunk",
        ).to_kernel(payload)
    }


def adaptive_epsilon_guard_predicates(payload: BaseModel | None = None) -> Mapping[str, Any]:
    """Return adaptive-epsilon native phase-transition predicates."""

    return {STOP_PREDICATE_REF: make_stop_predicate(payload)}


@eqx.filter_jit
def _adaptive_epsilon_train_step(
    task: Any,
    loss_func: Any,
    batch_info: BatchInfo,
    flat_model: Any,
    treedef_model: Any,
    flat_opt_state: Any,
    treedef_opt_state: Any,
    where_train_spec: Any,
    optimizer: optax.GradientTransformation,
    pgd_config: Any,
    key: Any,
    energy_lambda: Any,
    outer_weight: Any,
    force_filter_feedback: bool,
    controller_training_mode: str,
) -> tuple[Any, Any, Any, Any, Any, dict[str, jnp.ndarray]]:
    from rlrmp.train.cs_nominal_gru import (
        ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
        ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
        _apply_trial_spec_initial_state,
        _eval_trial_specs_for_training,
        _sample_adaptive_epsilon_training_batch,
        _scale_direct_epsilon_trial_specs,
        _weighted_loss_tree,
        _with_default_intervention_inputs,
    )
    from rlrmp.train.cs_perturbation_training import (
        add_zero_graph_channel_inputs,
        run_broad_epsilon_pgd_inner_maximizer,
    )

    key_trials, key_init, key_model = jr.split(key, 3)
    keys_trials = jr.split(key_trials, batch_info.size)
    keys_init = jr.split(key_init, batch_info.size)
    keys_model = jr.split(key_model, batch_info.size)
    trial_specs = _sample_adaptive_epsilon_training_batch(
        task,
        batch_info=batch_info,
        keys_trials=keys_trials,
    )
    model = jtu.tree_unflatten(treedef_model, flat_model)
    trial_specs = add_zero_graph_channel_inputs(
        trial_specs,
        force_filter_feedback=force_filter_feedback,
    )
    trial_specs = _with_default_intervention_inputs(model, trial_specs)
    adv_specs, inner_diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        task,
        model,
        trial_specs,
        loss_func,
        keys_model,
        config=pgd_config,
        soft_energy_lambda_override=energy_lambda,
        return_diagnostics=True,
    )
    adv_specs = jt.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        adv_specs,
    )
    init_states = eqx.filter_vmap(lambda _: init_state_from_component(model))(keys_init)
    init_states = eqx.filter_vmap(
        lambda state, trial_spec: _apply_trial_spec_initial_state(model, state, trial_spec)
    )(init_states, trial_specs)
    diff_model, static_model = eqx.partition(model, where_train_spec)
    opt_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)

    def paired_loss(current_diff_model):
        current_model = eqx.combine(current_diff_model, static_model)
        clean_states = _eval_trial_specs_for_training(
            current_model,
            trial_specs,
            init_states,
            keys_model,
        )
        adv_states = _eval_trial_specs_for_training(
            current_model,
            adv_specs,
            init_states,
            keys_model,
        )
        clean_losses = loss_func(clean_states, trial_specs, current_model)
        adv_losses = loss_func(adv_states, adv_specs, current_model)
        weighted_losses = _weighted_loss_tree(clean_losses, adv_losses, outer_weight)
        diagnostics = {
            "training_batch_damage_raw": jnp.asarray(adv_losses.total - clean_losses.total),
            "training_batch_full_strength_damage_raw": jnp.asarray(
                adv_losses.total - clean_losses.total
            ),
            "training_batch_applied_scaled_damage_raw": jnp.asarray(
                adv_losses.total - clean_losses.total
            ),
            "training_batch_clean_loss_total": jnp.asarray(clean_losses.total),
            "training_batch_adversarial_loss_total": jnp.asarray(adv_losses.total),
            "training_batch_full_strength_adversarial_loss_total": jnp.asarray(
                adv_losses.total
            ),
            "training_batch_applied_scaled_loss_total": jnp.asarray(adv_losses.total),
            "training_batch_weighted_loss_total": jnp.asarray(weighted_losses.total),
            "energy_lambda_used": jnp.asarray(energy_lambda, dtype=jnp.float32),
            "outer_weight_used": jnp.asarray(outer_weight, dtype=jnp.float32),
            "epsilon_scale_used": jnp.asarray(outer_weight, dtype=jnp.float32),
            "controller_training_mode_is_epsilon_scaled_outer": jnp.asarray(False),
        }
        diagnostics.update({f"inner_{name}": value for name, value in inner_diagnostics.items()})
        return weighted_losses.total, (weighted_losses, diagnostics)

    def epsilon_scaled_outer_loss(current_diff_model):
        current_model = eqx.combine(current_diff_model, static_model)
        applied_specs = _scale_direct_epsilon_trial_specs(
            clean_specs=trial_specs,
            adv_specs=adv_specs,
            epsilon_scale=outer_weight,
        )
        applied_states = _eval_trial_specs_for_training(
            current_model,
            applied_specs,
            init_states,
            keys_model,
        )
        applied_losses = loss_func(applied_states, applied_specs, current_model)
        clean_states = _eval_trial_specs_for_training(
            current_model,
            trial_specs,
            init_states,
            keys_model,
        )
        full_adv_states = _eval_trial_specs_for_training(
            current_model,
            adv_specs,
            init_states,
            keys_model,
        )
        clean_losses = loss_func(clean_states, trial_specs, current_model)
        full_adv_losses = loss_func(full_adv_states, adv_specs, current_model)
        diagnostics = {
            "training_batch_damage_raw": jnp.asarray(
                full_adv_losses.total - clean_losses.total
            ),
            "training_batch_full_strength_damage_raw": jnp.asarray(
                full_adv_losses.total - clean_losses.total
            ),
            "training_batch_applied_scaled_damage_raw": jnp.asarray(
                applied_losses.total - clean_losses.total
            ),
            "training_batch_clean_loss_total": jnp.asarray(clean_losses.total),
            "training_batch_adversarial_loss_total": jnp.asarray(full_adv_losses.total),
            "training_batch_full_strength_adversarial_loss_total": jnp.asarray(
                full_adv_losses.total
            ),
            "training_batch_applied_scaled_loss_total": jnp.asarray(applied_losses.total),
            "training_batch_weighted_loss_total": jnp.asarray(applied_losses.total),
            "energy_lambda_used": jnp.asarray(energy_lambda, dtype=jnp.float32),
            "outer_weight_used": jnp.asarray(outer_weight, dtype=jnp.float32),
            "epsilon_scale_used": jnp.asarray(outer_weight, dtype=jnp.float32),
            "controller_training_mode_is_epsilon_scaled_outer": jnp.asarray(True),
        }
        diagnostics.update({f"inner_{name}": value for name, value in inner_diagnostics.items()})
        return applied_losses.total, (applied_losses, diagnostics)

    if controller_training_mode == ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND:
        loss_fn = paired_loss
    elif controller_training_mode == ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER:
        loss_fn = epsilon_scaled_outer_loss
    else:
        raise ValueError(
            f"Unknown adaptive epsilon controller training mode: {controller_training_mode}"
        )
    (_, (losses, diagnostics)), grads = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=True,
    )(diff_model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model = project_component_parameters(model)
    flat_model = jtu.tree_leaves(model)
    flat_opt_state = jtu.tree_leaves(opt_state)
    return losses, adv_specs, flat_model, flat_opt_state, grads, diagnostics


def build_adaptive_epsilon_native_initial_slots(
    *,
    run_spec: Mapping[str, Any] | TrainingRunSpec,
    hps: Any,
    args: Any,
    key: Any,
    lr_continuation_mode: str | None = None,
) -> tuple[dict[str, Any], RlrmpRuntime]:
    """Return adaptive-epsilon native initial slots plus runtime context."""

    from rlrmp.runtime.checkpoint_custody import serialize_pytree_slot
    from rlrmp.train.cs_nominal_gru import (
        _initial_adaptive_epsilon_state,
        _initial_adaptive_epsilon_zero_guard,
        _initial_training_state,
        _where_train,
        setup_task_model_pair,
    )

    if not bool(getattr(getattr(hps, "adaptive_epsilon_curriculum", None), "enabled", False)):
        raise ValueError("adaptive-epsilon native initial slots require enabled curriculum")
    key_init, key_train, _key_adversary = split_initial_keys(key)
    pair = setup_task_model_pair(hps, key=key_init)
    trainer = build_adaptive_epsilon_controller_optimizer(run_spec, hps)
    where_train = _where_train()[0]
    state = _initial_training_state(
        model=pair.model,
        trainer=trainer,
        where_train=where_train,
        key=key_train,
    )
    adaptive_state = _initial_adaptive_epsilon_state(hps)
    zero_guard = _initial_adaptive_epsilon_zero_guard(enabled=True)
    runtime = AdaptiveEpsilonNativeRuntime(
        hps=hps,
        args=args,
        task=pair.task,
        trainer=trainer,
        where_train=where_train,
        model_template=state.model,
        optimizer_template=state.optimizer_state,
        run_spec=run_spec,
        lr_continuation_mode=_resolve_lr_continuation_mode(
            lr_continuation_mode,
            _lr_continuation_mode_from_run_spec(run_spec),
        ),
    )
    return (
        {
            MODEL: SerializedPyTreeSlot(serialize_pytree_slot(state.model)),
            OPTIMIZER: SerializedPyTreeSlot(serialize_pytree_slot(state.optimizer_state)),
            PRNG: state.key,
            COMPLETED_BATCHES: jnp.asarray(0, dtype=jnp.int32),
            ADAPTIVE_EPSILON_STATE: _adaptive_state_slot(adaptive_state),
            ZERO_ADVERSARY_GUARD: _json_slot(zero_guard),
            OBJECTIVE: None,
            TRAIN_LOSS: 0.0,
            DAMAGE_METRIC: 0.0,
            EPSILON_SCALE: 0.0,
            HISTORY_CHUNK_BYTES: b"",
            ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES: b"",
        },
        RlrmpRuntime(
            components={"adaptive_epsilon": runtime},
            stop_after_batches=getattr(args, "stop_after_batches", None),
        ),
    )


def build_adaptive_epsilon_controller_optimizer(
    run_spec: Mapping[str, Any] | TrainingRunSpec,
    hps: Any,
    *,
    schedule_origin_step: int = 0,
    current_step: int = 0,
    optimizer_count_at_current_step: int = 0,
) -> optax.GradientTransformation:
    """Build the adaptive-epsilon controller optimizer from the Feedbax spec."""

    return _build_adaptive_epsilon_controller_optimizer_parts(
        run_spec,
        hps,
        schedule_origin_step=schedule_origin_step,
        current_step=current_step,
        optimizer_count_at_current_step=optimizer_count_at_current_step,
    ).optimizer


def adaptive_epsilon_controller_lr_points(
    run_spec: Mapping[str, Any] | TrainingRunSpec,
    hps: Any,
    *,
    schedule_origin_step: int,
    current_step: int,
    optimizer_count_at_current_step: int,
    schedule_steps: Sequence[int],
) -> list[dict[str, Any]]:
    """Evaluate controller LRs through the same optimizer-build path as execution."""

    built = _build_adaptive_epsilon_controller_optimizer_parts(
        run_spec,
        hps,
        schedule_origin_step=schedule_origin_step,
        current_step=current_step,
        optimizer_count_at_current_step=optimizer_count_at_current_step,
    )
    schedule_position = int(current_step) - int(schedule_origin_step)
    rows: list[dict[str, Any]] = []
    for step in schedule_steps:
        count = int(optimizer_count_at_current_step) + (int(step) - schedule_position)
        lr = float(jax.device_get(built.learning_rate_schedule(jnp.asarray(count))))
        rows.append(
            {
                "step": int(step),
                "program_step": int(schedule_origin_step) + int(step),
                "optimizer_count": int(count),
                "lr": lr,
            }
        )
    return rows


def lr_resume_context_for_mode(
    *,
    mode: str | None,
    completed_batches: int,
    optimizer_count_at_current_step: int,
) -> LRResumeContext:
    """Return the runtime LR schedule context for a resume/fork mode."""

    normalized = _resolve_lr_continuation_mode(mode, None) or LR_CONTINUATION_CONTINUE
    completed = int(completed_batches)
    if normalized == LR_CONTINUATION_RESTART:
        return LRResumeContext(
            mode=normalized,
            schedule_origin_step=completed,
            current_step=completed,
            optimizer_count_at_current_step=int(optimizer_count_at_current_step),
        )
    return LRResumeContext(
        mode=normalized,
        schedule_origin_step=0,
        current_step=completed,
        optimizer_count_at_current_step=int(optimizer_count_at_current_step),
    )


def lr_report_schedule_steps(schedule_spec: LrScheduleSpec, *, start_position: int) -> list[int]:
    """Return discriminating origin-relative LR positions for gate/reporting."""

    start = max(0, int(start_position))
    steps = {start}
    warmup_end = max(0, int(schedule_spec.constant_lr_iterations or 0))
    total_steps = int(schedule_spec.total_steps or max(start, warmup_end, 1))
    if start <= warmup_end:
        steps.add((start + warmup_end) // 2)
        steps.add(warmup_end)
    elif warmup_end > 0:
        steps.add(warmup_end)
    steps.add(total_steps)
    if total_steps > 0:
        steps.add(max(start, total_steps - 1))
    return sorted(steps)


def _build_adaptive_epsilon_controller_optimizer_parts(
    run_spec: Mapping[str, Any] | TrainingRunSpec,
    hps: Any,
    *,
    schedule_origin_step: int = 0,
    current_step: int = 0,
    optimizer_count_at_current_step: int = 0,
) -> AdaptiveEpsilonControllerOptimizerBuild:
    """Build the adaptive controller optimizer and expose its realized schedule."""

    from rlrmp.train.cs_nominal_gru import (
        _gradient_diagnostics_transform,
        _update_diagnostics_transform,
    )

    optimizer_spec = _controller_optimizer_spec_from_run(run_spec, hps)
    schedule = _shifted_schedule(
        optimizer_spec.lr_schedule,
        schedule_origin_step=schedule_origin_step,
        current_step=current_step,
        optimizer_count_at_current_step=optimizer_count_at_current_step,
    )
    transforms: list[optax.GradientTransformation] = []
    if bool(getattr(hps, "training_diagnostics", True)):
        transforms.append(
            _gradient_diagnostics_transform(
                schedule=schedule,
                n_batches=int(hps.n_batches_condition),
                gradient_clip_norm=getattr(hps, "gradient_clip_norm", None),
            )
        )
    transforms.append(
        build_feedbax_optimizer(
            optimizer_spec,
            schedule_origin_step=schedule_origin_step,
            current_step=current_step,
            optimizer_count_at_current_step=optimizer_count_at_current_step,
            gradient_clip=getattr(hps, "gradient_clip_norm", None),
        )
    )
    if bool(getattr(hps, "training_diagnostics", True)):
        transforms.append(_update_diagnostics_transform(n_batches=int(hps.n_batches_condition)))
    return AdaptiveEpsilonControllerOptimizerBuild(
        optimizer=optax.chain(*transforms),
        learning_rate_schedule=schedule,
    )


def execute_adaptive_epsilon_training_run_spec_native(
    spec: TrainingRunSpec | Mapping[str, Any],
    *,
    run_id: str | None = None,
    key: Any | None = None,
    manifest_root: Path | str | None = None,
    checkpoint_root: Path | str | None = None,
    resume: bool = False,
    stop_after_barrier: str | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an adaptive-epsilon ``TrainingRunSpec`` through Feedbax."""

    from feedbax.training.executor import execute_training_run_spec
    from rlrmp.train.cs_nominal_gru import _config_namespace, build_hps

    ensure_adaptive_epsilon_training_method_registered()
    training_spec = (
        spec if isinstance(spec, TrainingRunSpec) else TrainingRunSpec.model_validate(spec)
    )
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        training_spec.method_ref,
        training_spec.method_payload,
        path="/method_payload",
    )
    if not isinstance(payload, AdaptiveEpsilonMethodPayload):
        raise TypeError("TrainingRunSpec does not carry an adaptive-epsilon method payload")
    args = _config_namespace(payload.config)
    hps = build_hps(args)
    explicit_lr_continuation_mode = kwargs.pop("lr_continuation_mode", None)
    initial_slots, runtime = build_adaptive_epsilon_native_initial_slots(
        run_spec=training_spec,
        hps=hps,
        args=args,
        key=key if key is not None else jr.PRNGKey(args.seed),
        lr_continuation_mode=explicit_lr_continuation_mode,
    )
    resume_slot_transform = kwargs.pop("resume_slot_transform", None)
    return execute_training_run_spec(
        training_spec,
        run_id=run_id,
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        manifest_root=manifest_root,
        checkpoint_root=checkpoint_root,
        loss_service=kwargs.pop("loss_service", AdaptiveEpsilonExternalObjectiveLossService()),
        resume=resume,
        resume_slot_transform=_resume_slot_transform(resume_slot_transform),
        stop_after_barrier=stop_after_barrier,
        **kwargs,
    )


def _adaptive_epsilon_train_chunk(
    runtime: RlrmpRuntime,
    payload: AdaptiveEpsilonMethodPayload,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del coordinate
    from rlrmp.runtime.checkpoint_custody import serialize_pytree_slot
    from rlrmp.train.cs_nominal_gru import (
        _append_history,
        _adaptive_epsilon_outer_weight,
        _adaptive_epsilon_schedule_batch,
        _adaptive_epsilon_zero_guard_from_state,
        _latest_loss_scalars,
        _run_adaptive_epsilon_training_chunk,
        _update_adaptive_epsilon_zero_guard,
    )

    native = _runtime(runtime)
    completed = int(chunk_slots[COMPLETED_BATCHES])
    remaining = int(payload.n_train_batches) - completed
    chunk_batches = min(int(payload.chunk_batches), remaining)
    if runtime.stop_after_batches is not None:
        chunk_batches = min(chunk_batches, int(runtime.stop_after_batches) - completed)
    current_state = _adaptive_state_from_slot(chunk_slots[ADAPTIVE_EPSILON_STATE])
    current_guard = _guard_from_slot(chunk_slots[ZERO_ADVERSARY_GUARD])
    if current_state is not None and not current_guard:
        current_guard = _adaptive_epsilon_zero_guard_from_state(current_state, enabled=True)
    if chunk_batches < 1:
        return {
            MODEL: chunk_slots[MODEL],
            OPTIMIZER: chunk_slots[OPTIMIZER],
            COMPLETED_BATCHES: chunk_slots[COMPLETED_BATCHES],
            ADAPTIVE_EPSILON_STATE: chunk_slots[ADAPTIVE_EPSILON_STATE],
            ZERO_ADVERSARY_GUARD: chunk_slots[ZERO_ADVERSARY_GUARD],
            TRAIN_LOSS: float(chunk_slots.get(TRAIN_LOSS, 0.0)),
            DAMAGE_METRIC: float(chunk_slots.get(DAMAGE_METRIC, 0.0)),
            EPSILON_SCALE: float(chunk_slots.get(EPSILON_SCALE, 0.0)),
            HISTORY_CHUNK_BYTES: b"",
            ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES: b"",
        }
    model = _deserialize_pytree_slot_value(
        chunk_slots[MODEL],
        native.model_template,
        slot=MODEL,
    )
    optimizer_state = _deserialize_optimizer_slot_value(
        chunk_slots[OPTIMIZER],
        native.optimizer_template,
    )
    trainer = _trainer_for_chunk(
        native,
        optimizer_state=optimizer_state,
        completed_batches=completed,
    )
    if not native.optimizer_hyperparams_aligned:
        optimizer_state = _align_optimizer_learning_rate(native, optimizer_state)
        native.optimizer_hyperparams_aligned = True
    (
        model,
        history_chunk,
        optimizer_state,
        adaptive_state,
        diagnostics,
    ) = _run_adaptive_epsilon_training_chunk(
        trainer=trainer,
        task=native.task,
        model=model,
        optimizer_state=optimizer_state,
        adaptive_state=current_state,
        hps=native.hps,
        where_train=native.where_train,
        key=chunk_slots[PRNG],
        start_batch=completed,
        chunk_batches=chunk_batches,
        log_progress=not bool(getattr(native.args, "disable_progress", False))
        and not bool(getattr(native.args, "quiet_progress", False)),
    )
    diagnostic_arrays = dict(diagnostics)
    next_completed = completed + chunk_batches
    updated_guard = _update_adaptive_epsilon_zero_guard(current_guard, diagnostic_arrays)
    if adaptive_state is not None:
        from dataclasses import replace

        adaptive_state = replace(adaptive_state, zero_adversary_guard=updated_guard)
    loss_scalars = _latest_loss_scalars(history_chunk, chunk_batches=chunk_batches)
    native.history = _append_history(native.history, history_chunk)
    native.records.append(
        {
            "completed_batches": next_completed,
            "chunk_batches": chunk_batches,
            "history_chunk": history_chunk,
            "diagnostics": diagnostic_arrays,
        }
    )
    damage_metric = _diagnostic_scalar(diagnostic_arrays, "adaptive_epsilon_measured_damage")
    epsilon_scale = _diagnostic_scalar(diagnostic_arrays, "adaptive_epsilon_epsilon_scale_used")
    if epsilon_scale == 0.0 and adaptive_state is not None:
        schedule_batch = _adaptive_epsilon_schedule_batch(adaptive_state, next_completed - 1)
        epsilon_scale = float(
            _adaptive_epsilon_outer_weight(
                native.hps.adaptive_epsilon_curriculum,
                schedule_batch,
            )
        )
    return {
        MODEL: SerializedPyTreeSlot(serialize_pytree_slot(model)),
        OPTIMIZER: SerializedPyTreeSlot(serialize_pytree_slot(optimizer_state)),
        COMPLETED_BATCHES: jnp.asarray(next_completed, dtype=jnp.int32),
        ADAPTIVE_EPSILON_STATE: _adaptive_state_slot(adaptive_state),
        ZERO_ADVERSARY_GUARD: _json_slot(updated_guard),
        TRAIN_LOSS: float(loss_scalars.get("total", 0.0)),
        DAMAGE_METRIC: damage_metric,
        EPSILON_SCALE: epsilon_scale,
        HISTORY_CHUNK_BYTES: _summary_bytes({"completed_batches": next_completed}),
        ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES: _summary_bytes(
            {
                "completed_batches": next_completed,
                "diagnostic_keys": sorted(diagnostic_arrays),
                "zero_adversary_guard": updated_guard,
            }
        ),
    }


def _runtime(runtime: RlrmpRuntime) -> AdaptiveEpsilonNativeRuntime:
    value = runtime.component("adaptive_epsilon")
    if not isinstance(value, AdaptiveEpsilonNativeRuntime):
        raise TypeError("missing adaptive_epsilon native runtime component")
    return value


def _trainer_for_chunk(
    native: AdaptiveEpsilonNativeRuntime,
    *,
    optimizer_state: Any,
    completed_batches: int,
) -> optax.GradientTransformation:
    """Finalize and cache the LR-shifted trainer from the actual restored slot."""

    if native.trainer_resume_context is not None:
        return native.trainer
    optimizer_count = optimizer_count_at_current_step(optimizer_state)
    context = lr_resume_context_for_mode(
        mode=native.lr_continuation_mode,
        completed_batches=completed_batches,
        optimizer_count_at_current_step=optimizer_count,
    )
    native.trainer = build_adaptive_epsilon_controller_optimizer(
        native.run_spec,
        native.hps,
        schedule_origin_step=context.schedule_origin_step,
        current_step=context.current_step,
        optimizer_count_at_current_step=context.optimizer_count_at_current_step,
    )
    native.trainer_resume_context = context
    return native.trainer


def _align_optimizer_learning_rate(
    native: AdaptiveEpsilonNativeRuntime,
    optimizer_state: Any,
) -> Any:
    context = native.trainer_resume_context
    if context is None:
        return optimizer_state
    schedule_position = context.current_step - context.schedule_origin_step
    point = adaptive_epsilon_controller_lr_points(
        native.run_spec,
        native.hps,
        schedule_origin_step=context.schedule_origin_step,
        current_step=context.current_step,
        optimizer_count_at_current_step=context.optimizer_count_at_current_step,
        schedule_steps=[schedule_position],
    )[0]
    lr = jnp.asarray(point["lr"], dtype=jnp.float32)

    def replace_lr(leaf: Any) -> Any:
        if not _is_injected_hyperparams_state(leaf):
            return leaf
        hyperparams = dict(leaf.hyperparams)
        current = jnp.asarray(hyperparams["learning_rate"])
        hyperparams["learning_rate"] = jnp.full_like(current, lr)
        return leaf._replace(hyperparams=hyperparams)

    return jt.map(replace_lr, optimizer_state, is_leaf=_is_injected_hyperparams_state)


def _deserialize_pytree_slot_value(value: Any, template: Any, *, slot: str) -> Any:
    from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot

    payload = value.payload if isinstance(value, SerializedPyTreeSlot) else value
    return deserialize_pytree_slot(payload, template, slot=slot)


def _deserialize_optimizer_slot_value(
    value: Any,
    template: Any,
) -> Any:
    from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot

    payload = value.payload if isinstance(value, SerializedPyTreeSlot) else value
    return deserialize_pytree_slot(payload, template, slot=OPTIMIZER)


def _adaptive_state_slot(state: Any) -> bytes:
    return _json_slot(None if state is None else state.to_json())


def _adaptive_state_from_slot(value: Any) -> Any:
    from rlrmp.train.cs_nominal_gru import AdaptiveEpsilonState

    payload = _json_from_slot(value)
    if not isinstance(payload, Mapping):
        return None
    return AdaptiveEpsilonState(
        lambda_value=float(payload["lambda_value"]),
        damage_ema=(
            None if payload.get("damage_ema") is None else float(payload["damage_ema"])
        ),
        clean_loss_ema=(
            None
            if payload.get("clean_loss_ema") is None
            else float(payload["clean_loss_ema"])
        ),
        last_update_batch=payload.get("last_update_batch"),
        update_count=int(payload.get("update_count", 0)),
        schedule_start_batch=int(payload.get("schedule_start_batch", 0)),
        zero_adversary_guard=(
            dict(payload["zero_adversary_guard"])
            if isinstance(payload.get("zero_adversary_guard"), Mapping)
            else None
        ),
        gain_estimate=(
            None if payload.get("gain_estimate") is None else float(payload["gain_estimate"])
        ),
        gain_samples=int(payload.get("gain_samples", 0)),
        pending_lambda_log_step=(
            None
            if payload.get("pending_lambda_log_step") is None
            else float(payload["pending_lambda_log_step"])
        ),
        pending_log_damage_ema=(
            None
            if payload.get("pending_log_damage_ema") is None
            else float(payload["pending_log_damage_ema"])
        ),
        last_log_damage_ema=(
            None
            if payload.get("last_log_damage_ema") is None
            else float(payload["last_log_damage_ema"])
        ),
        ema_noise_floor=(
            None if payload.get("ema_noise_floor") is None else float(payload["ema_noise_floor"])
        ),
        last_lambda_step_sign=int(payload.get("last_lambda_step_sign", 0)),
        lambda_step_count=int(payload.get("lambda_step_count", 0)),
        lambda_step_alternations=int(payload.get("lambda_step_alternations", 0)),
    )


def _guard_from_slot(value: Any) -> dict[str, Any]:
    from rlrmp.train.cs_nominal_gru import _normalize_adaptive_epsilon_zero_guard

    payload = _json_from_slot(value)
    return _normalize_adaptive_epsilon_zero_guard(payload, enabled=True)


def _resume_slot_transform(transform: Any | None) -> Any:
    """Normalize resumed adaptive slots without changing batch-horizon leaves.

    Feedbax applies ``CheckpointContinuationRequest`` before any target/post
    adapter transform.  Resizing serialized optimizer payloads here would hide
    the source structure and bypass its declared-leaf validation.
    """

    def normalize(slots: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(transform(slots) if transform is not None else slots)
        payload[TRAIN_LOSS] = 0.0
        payload[DAMAGE_METRIC] = 0.0
        payload[EPSILON_SCALE] = 0.0
        return payload

    return normalize


def attach_adaptive_epsilon_checkpoint_continuation(
    training_spec: TrainingRunSpec,
    *,
    source_completed_batches: int,
    target_total_batches: int,
) -> TrainingRunSpec:
    """Attach the explicit adaptive optimizer-horizon continuation contract.

    The declared leaves refer to the raw optimizer checkpoint topology.  The
    Stage-2 fork gate extends it under Feedbax custody, then serializes the
    resulting target optimizer in its documented target/post adapter.
    """

    request = CheckpointContinuationRequest(
        source_completed_batches=source_completed_batches,
        target_total_batches=target_total_batches,
        batch_indexed_leaves=list(ADAPTIVE_EPSILON_BATCH_INDEXED_CHECKPOINT_LEAVES),
    )
    checkpoint_progress = training_spec.checkpoint_progress.model_copy(
        update={"continuation": request}
    )
    return training_spec.model_copy(update={"checkpoint_progress": checkpoint_progress})


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return dict(item) if isinstance(item, Mapping) else {}


def _lr_continuation_mode_from_run_spec(
    run_spec: Mapping[str, Any] | TrainingRunSpec,
) -> LRContinuationMode | None:
    if isinstance(run_spec, TrainingRunSpec):
        payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
            run_spec.method_ref,
            run_spec.method_payload,
            path="/method_payload",
        )
        if isinstance(payload, AdaptiveEpsilonMethodPayload):
            return payload.lr_continuation_mode
        return None
    raw_payload = _mapping(_mapping(run_spec, "method_payload"), "payload")
    for source in (raw_payload, run_spec, _mapping(run_spec, "optimizer")):
        mode = _resolve_lr_continuation_mode(
            source.get("lr_continuation_mode"),
            source.get("learning_rate_continuation_mode"),
        )
        if mode is not None:
            return mode
    lr_continuation = run_spec.get("lr_continuation")
    if isinstance(lr_continuation, Mapping):
        return _resolve_lr_continuation_mode(lr_continuation.get("mode"), None)
    return None


def _resolve_lr_continuation_mode(
    primary: Any,
    fallback: Any,
) -> LRContinuationMode | None:
    value = primary if primary is not None else fallback
    if value is None:
        return None
    mode = str(value).lower().strip()
    if mode in {LR_CONTINUATION_RESTART, LR_CONTINUATION_CONTINUE}:
        return mode  # type: ignore[return-value]
    raise ValueError(
        "lr_continuation_mode must be 'restart' or 'continue'; "
        f"got {value!r}"
    )


def _is_injected_hyperparams_state(value: Any) -> bool:
    fields = getattr(value, "_fields", ())
    return {"count", "hyperparams", "inner_state"}.issubset(set(fields))


def _injected_learning_rate_schedule_count(value: Any) -> Any:
    hyperparams_states = getattr(value, "hyperparams_states", None)
    if isinstance(hyperparams_states, Mapping):
        learning_rate_state = hyperparams_states.get("learning_rate")
        count = getattr(learning_rate_state, "count", None)
        if count is not None:
            return count
    return value.count


def optimizer_count_at_current_step(optimizer_state: Any) -> int:
    """Return the injected Optax schedule count stored in an optimizer state."""

    for leaf in jt.leaves(optimizer_state, is_leaf=_is_injected_hyperparams_state):
        if not _is_injected_hyperparams_state(leaf):
            continue
        counts = jnp.asarray(_injected_learning_rate_schedule_count(leaf)).reshape(-1)
        if counts.size == 0:
            continue
        first = int(jax.device_get(counts[0]))
        if bool(jax.device_get(jnp.any(counts != counts[0]))):
            raise ValueError(
                "adaptive-epsilon optimizer count differs across replicated states; "
                "cannot derive one LR resume context"
            )
        return first
    raise ValueError(
        "adaptive-epsilon optimizer state lacks injected-hyperparameter count; "
        "cannot derive LR resume context"
    )


def _float_from_sources(
    *sources: Mapping[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return float(value)
    return float(default)


def _controller_optimizer_spec_from_run(
    run_spec: Mapping[str, Any] | TrainingRunSpec,
    hps: Any,
) -> OptimizerSpec:
    if isinstance(run_spec, TrainingRunSpec):
        payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
            run_spec.method_ref,
            run_spec.method_payload,
            path="/method_payload",
        )
        if (
            isinstance(payload, AdaptiveEpsilonMethodPayload)
            and payload.controller_optimizer is not None
        ):
            return payload.controller_optimizer
    elif isinstance(run_spec, Mapping):
        raw_payload = _mapping(_mapping(run_spec, "method_payload"), "payload")
        raw_optimizer = raw_payload.get("controller_optimizer")
        if isinstance(raw_optimizer, Mapping):
            return OptimizerSpec.model_validate(raw_optimizer)
        if "hps" in run_spec:
            return adaptive_epsilon_controller_optimizer_spec(run_spec)
    hps_mapping = {
        "learning_rate_0": getattr(hps, "learning_rate_0", 1e-2),
        "lr_schedule": getattr(hps, "lr_schedule", "delayed_cosine"),
        "constant_lr_iterations": getattr(hps, "constant_lr_iterations", 0),
        "n_batches_condition": getattr(hps, "n_batches_condition", 1),
        "warmup_init_fraction": getattr(hps, "warmup_init_fraction", 0.0),
        "cosine_annealing_alpha": getattr(hps, "cosine_annealing_alpha", 0.0),
        "weight_decay": getattr(hps, "weight_decay", 0.0),
    }
    return adaptive_epsilon_controller_optimizer_spec({"hps": hps_mapping})


def _shifted_schedule(
    schedule_spec: LrScheduleSpec | None,
    *,
    schedule_origin_step: int,
    current_step: int,
    optimizer_count_at_current_step: int,
) -> optax.Schedule:
    if schedule_spec is None:
        raise ValueError("adaptive-epsilon controller optimizer requires lr_schedule")
    base_schedule = feedbax_learning_rate_schedule(schedule_spec)
    schedule_position = int(current_step) - int(schedule_origin_step)
    if schedule_position < 0:
        raise ValueError(
            "current_step must be greater than or equal to schedule_origin_step; "
            f"got current_step={current_step}, schedule_origin_step={schedule_origin_step}"
        )
    count_origin = int(optimizer_count_at_current_step)

    def schedule(count: Any) -> Any:
        return base_schedule(
            (jnp.asarray(count, dtype=jnp.int32) - count_origin) + schedule_position
        )

    return schedule


def _diagnostic_scalar(diagnostics: Mapping[str, Any], key: str) -> float:
    value = diagnostics.get(key)
    if value is None:
        return 0.0
    array = jnp.asarray(value)
    return float(array.reshape(-1)[0])


def _summary_bytes(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    buffer.write(json.dumps(dict(payload), sort_keys=True).encode("utf-8"))
    buffer.write(b"\n")
    return buffer.getvalue()


def _json_slot(payload: Any) -> bytes:
    return SerializedPyTreeSlot(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _json_from_slot(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, SerializedPyTreeSlot):
        value = value.payload
    if isinstance(value, bytes):
        if not value:
            return None
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value
