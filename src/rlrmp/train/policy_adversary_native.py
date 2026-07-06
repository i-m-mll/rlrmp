"""Native-executor contract and kernels for policy-adversary GRU training."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    ArtifactPolicySpec,
    CheckpointProgressPolicySpec,
    ExecutionPolicySpec,
    GraphTopologySourceSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
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
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlrmp.model.feedback_descriptors import DESCRIPTOR_PAYLOAD_KEY
from rlrmp.model.feedbax_graph import graph_spec_payload
from rlrmp.train.cs_perturbation_training import (
    config_from_policy_adversary_hps,
    make_policy_adversary,
)
from rlrmp.train.executor.adapters import ChunkKernelAdapter, RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.guards import make_stop_predicate
from rlrmp.train.executor.initial_slots import RlrmpRuntime, split_initial_keys
from rlrmp.train.executor.slots import (
    ADVERSARY_LOSS,
    ADVERSARY_OPTIMIZER,
    ADVERSARY_POLICY,
    COMPLETED_BATCHES,
    HISTORY_CHUNK_BYTES,
    MODEL,
    OBJECTIVE,
    OPTIMIZER,
    POLICY_ADVERSARY_DIAGNOSTICS_BYTES,
    POLICY_ADVERSARY_SCHEMA,
    POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
    PRNG,
    TRAIN_LOSS,
    artifact_sink_specs,
    checkpoint_slot_specs,
    supervised_state_slots,
)

POLICY_ADVERSARY_METHOD_PAYLOAD_SCHEMA_ID = (
    "rlrmp.spec.training_method.policy_adversary_supervised_payload"
)
POLICY_ADVERSARY_METHOD_PAYLOAD_SCHEMA_VERSION = (
    "rlrmp.spec.training_method.policy_adversary_supervised_payload.v1"
)
TRAIN_CHUNK_KERNEL_REF = "rlrmp.policy_adversary_supervised.train_chunk"
STOP_PREDICATE_REF = "rlrmp.policy_adversary_supervised.stop"
TRAIN_CHUNK_BARRIER = "after_policy_adversary_train_chunk"


class PolicyAdversaryMethodPayload(BaseModel):
    """Governed payload for the policy-adversary supervised native method."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    n_train_batches: int = Field(gt=0)
    chunk_batches: int = Field(gt=0)
    policy: dict[str, Any]
    inner_optimizer: dict[str, Any]
    objective: dict[str, Any]
    checkpointing: dict[str, Any]
    rlrmp_extension_payload: str = "rlrmp_run_spec"

    @model_validator(mode="after")
    def _validate_payload(self) -> "PolicyAdversaryMethodPayload":
        if not bool(self.config.get("policy_adversary_training", False)):
            raise ValueError("policy-adversary native payload requires enabled training")
        if self.chunk_batches > self.n_train_batches:
            raise ValueError("chunk_batches cannot exceed n_train_batches")
        return self


class SerializedPyTreeSlot:
    """Pickleable PyTree-byte slot with a stable structural ABI repr."""

    def __init__(self, payload: bytes) -> None:
        self.payload = bytes(payload)

    def __repr__(self) -> str:
        return "SerializedPyTreeSlot(payload=<bytes>)"


@dataclass
class PolicyAdversaryNativeRuntime:
    """Runtime-only objects used by policy-adversary native kernels."""

    hps: Any
    args: Any
    task: Any
    trainer: Any
    where_train: Any
    adversary_optimizer: optax.GradientTransformation
    model_template: Any
    optimizer_template: Any
    adversary_policy_template: Any
    adversary_optimizer_template: Any
    history: Any | None = None
    records: list[dict[str, Any]] = field(default_factory=list)


class PolicyAdversaryExternalObjectiveLoss(AbstractLoss):
    """Placeholder lowered loss for runtime-owned policy-adversary training."""

    label: str = "rlrmp_policy_adversary_external_objective"

    def term(self, states: Any, trial_specs: Any, model: Any) -> Any:
        del states, trial_specs, model
        return jnp.asarray(0.0)


class PolicyAdversaryExternalObjectiveLossService(LossService):
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
                loss=PolicyAdversaryExternalObjectiveLoss(),
                requirements=ObjectiveExecutionRequirements(),
                source_kind="objective_spec",
            )
        return super().lower_objective_slot(
            slot,
            graph=graph,
            trial_axis=trial_axis,
            path=path,
        )


def policy_adversary_method_ref() -> MethodRefSpec:
    """Return the RLRMP method ref for policy-adversary supervised training."""

    return MethodRefSpec(package="rlrmp", name="policy_adversary_supervised", version="v1")


def ensure_policy_adversary_training_method_registered() -> None:
    """Install the policy-adversary native method in Feedbax's registry."""

    if POLICY_ADVERSARY_SUPERVISED_METHOD_REF in DEFAULT_TRAINING_METHOD_REGISTRY.available_keys():
        return
    DEFAULT_TRAINING_METHOD_REGISTRY.register(
        TrainingMethodRegistration(
            method_ref=POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
            payload_schema_id=POLICY_ADVERSARY_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=POLICY_ADVERSARY_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=PolicyAdversaryMethodPayload,
            contract_factory=policy_adversary_method_contract,
            update_kernels_factory=policy_adversary_update_kernels,
            guard_predicates_factory=policy_adversary_guard_predicates,
            rejected_payload_versions=(
                "rlrmp.spec.training_method.policy_adversary_supervised_payload.v0",
            ),
            owner="rlrmp.train.policy_adversary_native",
            package="rlrmp",
        )
    )


def policy_adversary_method_payload(run_spec: Mapping[str, Any]) -> MethodPayloadEnvelope:
    """Return the governed method payload for a C&S policy-adversary run."""

    from rlrmp.train.cs_nominal_gru import _args_values_from_run_spec

    hps = _mapping(run_spec, "hps")
    policy_adversary = _mapping(hps, "policy_adversary_training")
    payload = PolicyAdversaryMethodPayload(
        config=_args_values_from_run_spec(dict(run_spec)),
        n_train_batches=int(run_spec.get("n_train_batches", 1)),
        chunk_batches=max(
            1,
            int(_mapping(run_spec, "checkpointing").get("interval_batches", 1)),
        ),
        policy=_mapping(policy_adversary, "policy"),
        inner_optimizer=_mapping(policy_adversary, "inner_optimizer"),
        objective=_mapping(policy_adversary, "objective"),
        checkpointing=_mapping(run_spec, "checkpointing"),
    )
    return MethodPayloadEnvelope(
        schema_id=POLICY_ADVERSARY_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=POLICY_ADVERSARY_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=payload.model_dump(mode="json", exclude_none=True),
    )


def build_policy_adversary_training_run_spec(
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
    """Build a policy-adversary native ``TrainingRunSpec`` from the C&S payload."""

    ensure_policy_adversary_training_method_registered()
    method_payload = policy_adversary_method_payload(run_spec)
    contract = policy_adversary_method_contract()
    effective_phase = policy_adversary_effective_phase_spec(contract)
    return TrainingRunSpec(
        graph=GraphTopologySourceSpec(
            inline=graph_spec_payload(graph_spec),
            schema_id=getattr(graph_spec, "schema_id", None),
            schema_version=getattr(graph_spec, "schema_version", None),
            metadata={
                "source": "materialized_runtime_graph",
                "native_method": POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
                DESCRIPTOR_PAYLOAD_KEY: metadata.get(DESCRIPTOR_PAYLOAD_KEY),
                "descriptor_basis_hash": metadata.get("descriptor_basis_hash"),
            },
        ),
        task=task,
        training_config=training_config,
        objective=objective,
        risk_aggregation=risk_aggregation,
        method_ref=policy_adversary_method_ref(),
        method_payload=method_payload,
        method_extensions=dict(method_extensions),
        worker_execution=WorkerExecutionSpec(
            method_contract=contract,
            effective_phase=effective_phase,
            metadata={
                "native_executor": "feedbax.training.executor.execute_training_run_spec",
                "kernel_owner": "rlrmp.train.policy_adversary_native",
            },
        ),
        execution=execution,
        artifacts=artifacts,
        checkpoint_progress=checkpoint_progress,
        metadata={
            **dict(metadata),
            "native_method": POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
        },
    )


def policy_adversary_method_contract() -> MethodContractSpec:
    """Return the policy-adversary supervised native phase program."""

    reads = [
        MODEL,
        OPTIMIZER,
        PRNG,
        COMPLETED_BATCHES,
        ADVERSARY_POLICY,
        ADVERSARY_OPTIMIZER,
        OBJECTIVE,
    ]
    writes = [
        MODEL,
        OPTIMIZER,
        PRNG,
        COMPLETED_BATCHES,
        ADVERSARY_POLICY,
        ADVERSARY_OPTIMIZER,
        TRAIN_LOSS,
        ADVERSARY_LOSS,
        HISTORY_CHUNK_BYTES,
        POLICY_ADVERSARY_DIAGNOSTICS_BYTES,
    ]
    program = PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="policy_adversary_train_chunk",
                kind="outer_loop",
                reads=reads,
                writes=writes,
                update_steps=["policy_adversary_train_chunk"],
                legal_next=["done", "policy_adversary_train_chunk"],
                checkpoint_barrier=TRAIN_CHUNK_BARRIER,
                loop_axis="batch",
                metadata={
                    "native_kernel_granularity": "checkpoint_sized_policy_adversary_chunk",
                    "adversary_update": "persistent_policy_ascent_before_controller_step",
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
                    ADVERSARY_POLICY,
                    ADVERSARY_OPTIMIZER,
                    TRAIN_LOSS,
                    ADVERSARY_LOSS,
                ],
            ),
        ],
        initial_phase="policy_adversary_train_chunk",
        update_steps=[
            UpdateStepSpec(
                name="policy_adversary_train_chunk",
                kind="gradient",
                kernel=UpdateKernelSpec(kernel_ref=TRAIN_CHUNK_KERNEL_REF),
                reads=reads,
                writes=writes,
                axes=["batch", "replicate", "adversary_inner_step"],
                optimizer_binding="controller_and_adversary_optimizers",
                metadata={
                    "controller_direction": "minimize",
                    "adversary_direction": "maximize",
                    "math_owner": (
                        "rlrmp.train.cs_nominal_gru._run_policy_adversary_training_chunk"
                    ),
                },
            )
        ],
        transitions=[
            PhaseTransitionSpec(
                source="policy_adversary_train_chunk",
                target="done",
                barrier=TRAIN_CHUNK_BARRIER,
                guard=MetricGuardSpec(
                    predicate_ref=STOP_PREDICATE_REF,
                    metric_slots=[TRAIN_LOSS, ADVERSARY_LOSS],
                    bookkeeping_slots=[COMPLETED_BATCHES],
                ),
            ),
            PhaseTransitionSpec(
                source="policy_adversary_train_chunk",
                target="policy_adversary_train_chunk",
                barrier=TRAIN_CHUNK_BARRIER,
            ),
        ],
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="controller_and_adversary_optimizers",
                optimizer_slot=OPTIMIZER,
                target_slot=MODEL,
                direction="minimize",
                projection="after_step",
                phase_scope=["policy_adversary_train_chunk"],
                objective_reads=[OBJECTIVE],
                metadata={
                    "adversary_optimizer_slot": ADVERSARY_OPTIMIZER,
                    "adversary_target_slot": ADVERSARY_POLICY,
                    "adversary_direction": "maximize",
                },
            )
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name=TRAIN_CHUNK_BARRIER,
                phase="policy_adversary_train_chunk",
                slots=checkpoint_slot_specs(POLICY_ADVERSARY_SCHEMA),
                artifact_sinks=artifact_sink_specs(POLICY_ADVERSARY_SCHEMA),
                resume_coordinate=ResumeCoordinateSpec(
                    phase="policy_adversary_train_chunk",
                    completed_barrier=TRAIN_CHUNK_BARRIER,
                ),
            )
        ],
        metadata={
            "phase_program_identity": "rlrmp.policy_adversary_supervised.chunked.v1",
            "checkpoint_barrier_policy": "after_each_policy_adversary_chunk",
        },
    )
    return MethodContractSpec(
        method_ref=POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
        method_payload_schema_version=POLICY_ADVERSARY_METHOD_PAYLOAD_SCHEMA_VERSION,
        axes=[
            AxisSpec(name="batch", role="batch"),
            AxisSpec(name="replicate", role="replicate"),
            AxisSpec(name="adversary_inner_step", role="epoch"),
        ],
        state_slots=[
            *supervised_state_slots(POLICY_ADVERSARY_SCHEMA),
            StateSlotSpec(
                name=ADVERSARY_POLICY,
                role="model",
                metadata={"owner": "policy_adversary"},
            ),
            StateSlotSpec(
                name=ADVERSARY_OPTIMIZER,
                role="optimizer",
                metadata={"target_slot": ADVERSARY_POLICY},
            ),
        ],
        phase_program=program,
        metadata={
            "adversary_policy_slot": ADVERSARY_POLICY,
            "adversary_optimizer_slot": ADVERSARY_OPTIMIZER,
        },
    )


def policy_adversary_effective_phase_spec(
    contract: MethodContractSpec | None = None,
) -> EffectivePhaseSpec:
    """Return the EffectivePhaseSpec for the policy-adversary contract."""

    active_contract = contract or policy_adversary_method_contract()
    return EffectivePhaseSpec(
        method_ref=active_contract.method_ref,
        axes=active_contract.axes,
        state_slots=active_contract.state_slots,
        phase_program=active_contract.phase_program,
        consistency_predicate=derive_consistency_predicate(active_contract.phase_program),
    )


def policy_adversary_update_kernels(
    payload: BaseModel | None = None,
) -> Mapping[str, Any]:
    """Return Feedbax update kernels for policy-adversary supervised training."""

    return {
        TRAIN_CHUNK_KERNEL_REF: ChunkKernelAdapter(
            chunk_fn=_policy_adversary_train_chunk,
            reads=(
                MODEL,
                OPTIMIZER,
                PRNG,
                COMPLETED_BATCHES,
                ADVERSARY_POLICY,
                ADVERSARY_OPTIMIZER,
            ),
            writes=(
                MODEL,
                OPTIMIZER,
                PRNG,
                COMPLETED_BATCHES,
                ADVERSARY_POLICY,
                ADVERSARY_OPTIMIZER,
                TRAIN_LOSS,
                ADVERSARY_LOSS,
                HISTORY_CHUNK_BYTES,
                POLICY_ADVERSARY_DIAGNOSTICS_BYTES,
            ),
            metric_slots=(TRAIN_LOSS, ADVERSARY_LOSS),
            prng_slot=PRNG,
            name="policy-adversary supervised train chunk",
        ).to_kernel(payload)
    }


def policy_adversary_guard_predicates(payload: BaseModel | None = None) -> Mapping[str, Any]:
    """Return policy-adversary native phase-transition predicates."""

    return {STOP_PREDICATE_REF: make_stop_predicate(payload)}


@eqx.filter_jit
def _advance_policy_adversary_compiled(
    policy: Any,
    optimizer_state: Any,
    optimizer: optax.GradientTransformation,
    task: Any,
    model: Any,
    hps: Any,
    key: Any,
    batch_index: Any,
) -> tuple[Any, Any, dict[str, jnp.ndarray]]:
    """Run the persistent policy-adversary ascent steps in one compiled update."""

    from rlrmp.train.cs_nominal_gru import _policy_adversary_batch_objective

    cfg = config_from_policy_adversary_hps(hps.policy_adversary_training)

    def loss_for_policy(candidate_policy):
        objective, diagnostics = _policy_adversary_batch_objective(
            candidate_policy,
            task,
            model,
            hps,
            key=key,
            batch_index=batch_index,
        )
        return -objective, diagnostics

    diagnostics = {}
    for _ in range(int(cfg.n_steps)):
        (_loss, diagnostics), grads = eqx.filter_value_and_grad(
            loss_for_policy,
            has_aux=True,
        )(policy)
        updates, optimizer_state = optimizer.update(
            grads,
            optimizer_state,
            eqx.filter(policy, eqx.is_array),
        )
        policy = eqx.apply_updates(policy, updates)
    diagnostics = {
        **diagnostics,
        "n_ascent_steps": jnp.asarray(cfg.n_steps, dtype=jnp.float32),
        "learning_rate": jnp.asarray(cfg.learning_rate, dtype=jnp.float32),
    }
    return policy, optimizer_state, diagnostics


def build_policy_adversary_native_initial_slots(
    *,
    run_spec: Mapping[str, Any] | TrainingRunSpec,
    hps: Any,
    args: Any,
    key: Any,
) -> tuple[dict[str, Any], RlrmpRuntime]:
    """Return policy-adversary native initial slots plus runtime context."""

    del run_spec
    from rlrmp.train.cs_nominal_gru import (
        _build_trainer,
        _initial_training_state,
        _where_train,
        setup_task_model_pair,
    )
    from rlrmp.runtime.checkpoint_custody import serialize_pytree_slot

    if not bool(getattr(getattr(hps, "policy_adversary_training", None), "enabled", False)):
        raise ValueError("policy-adversary native initial slots require enabled training")
    key_init, key_train, key_adversary = split_initial_keys(key)
    pair = setup_task_model_pair(hps, key=key_init)
    trainer = _build_trainer(hps)
    where_train = _where_train()[0]
    state = _initial_training_state(
        model=pair.model,
        trainer=trainer,
        where_train=where_train,
        key=key_train,
    )
    adversary_cfg = config_from_policy_adversary_hps(hps.policy_adversary_training)
    adversary_policy = make_policy_adversary(
        adversary_cfg,
        key=key_adversary,
        horizon=max(1, int(hps.task.n_steps) - 1),
    )
    adversary_optimizer = optax.adam(float(adversary_cfg.learning_rate))
    adversary_optimizer_state = adversary_optimizer.init(
        eqx.filter(adversary_policy, eqx.is_array)
    )
    runtime = PolicyAdversaryNativeRuntime(
        hps=hps,
        args=args,
        task=pair.task,
        trainer=trainer,
        where_train=where_train,
        adversary_optimizer=adversary_optimizer,
        model_template=state.model,
        optimizer_template=state.optimizer_state,
        adversary_policy_template=adversary_policy,
        adversary_optimizer_template=adversary_optimizer_state,
    )
    return (
        {
            MODEL: SerializedPyTreeSlot(serialize_pytree_slot(state.model)),
            OPTIMIZER: SerializedPyTreeSlot(serialize_pytree_slot(state.optimizer_state)),
            PRNG: state.key,
            COMPLETED_BATCHES: jnp.asarray(0, dtype=jnp.int32),
            ADVERSARY_POLICY: SerializedPyTreeSlot(serialize_pytree_slot(adversary_policy)),
            ADVERSARY_OPTIMIZER: SerializedPyTreeSlot(
                serialize_pytree_slot(adversary_optimizer_state)
            ),
            OBJECTIVE: None,
            TRAIN_LOSS: 0.0,
            ADVERSARY_LOSS: 0.0,
            HISTORY_CHUNK_BYTES: b"",
            POLICY_ADVERSARY_DIAGNOSTICS_BYTES: b"",
        },
        RlrmpRuntime(
            components={"policy_adversary": runtime},
            stop_after_batches=getattr(args, "stop_after_batches", None),
        ),
    )


def execute_policy_adversary_training_run_spec_native(
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
    """Execute a policy-adversary ``TrainingRunSpec`` through Feedbax."""

    from feedbax.training.executor import execute_training_run_spec
    from rlrmp.train.cs_nominal_gru import _config_namespace, build_hps

    ensure_policy_adversary_training_method_registered()
    training_spec = (
        spec if isinstance(spec, TrainingRunSpec) else TrainingRunSpec.model_validate(spec)
    )
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        training_spec.method_ref,
        training_spec.method_payload,
        path="/method_payload",
    )
    if not isinstance(payload, PolicyAdversaryMethodPayload):
        raise TypeError("TrainingRunSpec does not carry a policy-adversary method payload")
    args = _config_namespace(payload.config)
    hps = build_hps(args)
    initial_slots, runtime = build_policy_adversary_native_initial_slots(
        run_spec=training_spec,
        hps=hps,
        args=args,
        key=key if key is not None else jr.PRNGKey(args.seed),
    )
    resume_slot_transform = kwargs.pop("resume_slot_transform", None)
    return execute_training_run_spec(
        training_spec,
        run_id=run_id,
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        manifest_root=manifest_root,
        checkpoint_root=checkpoint_root,
        loss_service=kwargs.pop("loss_service", PolicyAdversaryExternalObjectiveLossService()),
        resume=resume,
        resume_slot_transform=_resume_slot_transform(resume_slot_transform),
        stop_after_barrier=stop_after_barrier,
        **kwargs,
    )


def _policy_adversary_train_chunk(
    runtime: RlrmpRuntime,
    payload: PolicyAdversaryMethodPayload,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del coordinate
    from rlrmp.train.cs_nominal_gru import (
        _append_history,
        _latest_loss_scalars,
        _policy_adversary_diagnostics_arrays,
        _run_policy_adversary_training_chunk,
    )
    from rlrmp.runtime.checkpoint_custody import serialize_pytree_slot

    native = _runtime(runtime)
    completed = int(chunk_slots[COMPLETED_BATCHES])
    remaining = int(payload.n_train_batches) - completed
    chunk_batches = min(int(payload.chunk_batches), remaining)
    if runtime.stop_after_batches is not None:
        chunk_batches = min(chunk_batches, int(runtime.stop_after_batches) - completed)
    if chunk_batches < 1:
        return {
            MODEL: chunk_slots[MODEL],
            OPTIMIZER: chunk_slots[OPTIMIZER],
            COMPLETED_BATCHES: chunk_slots[COMPLETED_BATCHES],
            ADVERSARY_POLICY: chunk_slots[ADVERSARY_POLICY],
            ADVERSARY_OPTIMIZER: chunk_slots[ADVERSARY_OPTIMIZER],
            TRAIN_LOSS: float(chunk_slots.get(TRAIN_LOSS, 0.0)),
            ADVERSARY_LOSS: float(chunk_slots.get(ADVERSARY_LOSS, 0.0)),
            HISTORY_CHUNK_BYTES: b"",
            POLICY_ADVERSARY_DIAGNOSTICS_BYTES: b"",
        }
    model = _deserialize_pytree_slot_value(
        chunk_slots[MODEL],
        native.model_template,
        slot=MODEL,
    )
    optimizer_state = _deserialize_pytree_slot_value(
        chunk_slots[OPTIMIZER],
        native.optimizer_template,
        slot=OPTIMIZER,
    )
    adversary_policy = _deserialize_pytree_slot_value(
        chunk_slots[ADVERSARY_POLICY],
        native.adversary_policy_template,
        slot=ADVERSARY_POLICY,
    )
    adversary_optimizer_state = _deserialize_pytree_slot_value(
        chunk_slots[ADVERSARY_OPTIMIZER],
        native.adversary_optimizer_template,
        slot=ADVERSARY_OPTIMIZER,
    )
    (
        model,
        history_chunk,
        optimizer_state,
        adversary_policy,
        adversary_optimizer_state,
        diagnostics,
    ) = _run_policy_adversary_training_chunk(
        trainer=native.trainer,
        task=native.task,
        model=model,
        optimizer_state=optimizer_state,
        adversary_policy=adversary_policy,
        adversary_optimizer_state=adversary_optimizer_state,
        adversary_optimizer=native.adversary_optimizer,
        hps=native.hps,
        where_train=native.where_train,
        key=chunk_slots[PRNG],
        start_batch=completed,
        chunk_batches=chunk_batches,
        log_progress=False,
    )
    next_completed = completed + chunk_batches
    loss_scalars = _latest_loss_scalars(history_chunk, chunk_batches=chunk_batches)
    adversary_loss = _diagnostic_scalar(diagnostics, "adversary_objective")
    diagnostic_arrays = _policy_adversary_diagnostics_arrays(
        diagnostics,
        batch_index=next_completed - 1,
        chunk_batches=chunk_batches,
    )
    native.history = _append_history(native.history, history_chunk)
    native.records.append(
        {
            "completed_batches": next_completed,
            "chunk_batches": chunk_batches,
            "history_chunk": history_chunk,
            "diagnostics": diagnostic_arrays,
        }
    )
    return {
        MODEL: SerializedPyTreeSlot(serialize_pytree_slot(model)),
        OPTIMIZER: SerializedPyTreeSlot(serialize_pytree_slot(optimizer_state)),
        COMPLETED_BATCHES: jnp.asarray(next_completed, dtype=jnp.int32),
        ADVERSARY_POLICY: SerializedPyTreeSlot(serialize_pytree_slot(adversary_policy)),
        ADVERSARY_OPTIMIZER: SerializedPyTreeSlot(
            serialize_pytree_slot(adversary_optimizer_state)
        ),
        TRAIN_LOSS: float(loss_scalars.get("total", 0.0)),
        ADVERSARY_LOSS: adversary_loss,
        HISTORY_CHUNK_BYTES: _summary_bytes({"completed_batches": next_completed}),
        POLICY_ADVERSARY_DIAGNOSTICS_BYTES: _summary_bytes(
            {
                "completed_batches": next_completed,
                "diagnostic_keys": sorted(diagnostic_arrays),
            }
        ),
    }


def _runtime(runtime: RlrmpRuntime) -> PolicyAdversaryNativeRuntime:
    value = runtime.component("policy_adversary")
    if not isinstance(value, PolicyAdversaryNativeRuntime):
        raise TypeError("missing policy_adversary native runtime component")
    return value


def _deserialize_pytree_slot_value(value: Any, template: Any, *, slot: str) -> Any:
    from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot

    payload = value.payload if isinstance(value, SerializedPyTreeSlot) else value
    return deserialize_pytree_slot(payload, template, slot=slot)


def _resume_slot_transform(transform: Any | None) -> Any:
    def normalize(slots: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(transform(slots) if transform is not None else slots)
        payload[TRAIN_LOSS] = 0.0
        payload[ADVERSARY_LOSS] = 0.0
        return payload

    return normalize


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return dict(item) if isinstance(item, Mapping) else {}


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
