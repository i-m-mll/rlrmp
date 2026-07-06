"""Tests for rlrmp native-executor scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import pytest
from feedbax.contracts.training import (
    ArtifactPolicySpec,
    GraphTopologySourceSpec,
    LossTermSpec,
    ObjectiveSlotSpec,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    StandardSupervisedMethodPayload,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingMethodRegistry,
    WorkerExecutionSpec,
    TrainingRunSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
    standard_supervised_update_kernels,
)
from feedbax.contracts.worker import (
    BarrierArtifactSinkSpec,
    MetricGuardSpec,
    PhaseTransitionSpec,
    ProgressCoordinate,
    StateSlotSpec,
)
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)
from feedbax.training.executor import execute_training_run_spec

from rlrmp.train.executor.adapters import (
    ChunkKernelAdapter,
    ChunkKernelAdapterError,
    RLRMP_RUNTIME_CONTEXT_KEY,
)
from rlrmp.train.executor.equivalence import (
    TOY_BARRIER,
    _toy_executor,
    _toy_initial_slots,
    _toy_program,
    run_toy_paired_equivalence,
    run_toy_resume_equivalence,
)
from rlrmp.train.executor.guards import make_stop_predicate
from rlrmp.train.executor.initial_slots import RlrmpRuntime
from rlrmp.train.executor.slots import (
    COMPLETED_BATCHES,
    CS_SUPERVISED_SCHEMA,
    HISTORY_CHUNK_BYTES,
    MODEL,
    OPTIMIZER,
    TRAIN_LOSS,
    artifact_sink_specs,
    checkpoint_slot_specs,
)


@dataclass(frozen=True)
class _Payload:
    n_train_batches: int


def test_chunk_kernel_adapter_runs_guarded_self_loop_through_phase_executor() -> None:
    payload = _Payload(n_train_batches=3)
    initial_slots = _toy_initial_slots(seed=1)
    result = _toy_executor(payload).run(
        initial_slots,
        run_id="adapter-self-loop",
        context={RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime(components={"chunk_batches": 1})},
    )

    assert result.slots[COMPLETED_BATCHES] == 3
    assert result.coordinate.phase == "done"
    assert len(result.checkpoint_visits) == 3
    train_losses = [
        coordinate.metrics[TRAIN_LOSS]
        for coordinate in result.progress
        if coordinate.phase == "train_chunk"
    ]
    assert train_losses == [
        pytest.approx(1.5),
        pytest.approx(2.5),
        pytest.approx(3.5),
    ]


def test_chunk_kernel_adapter_rejects_bad_writes_and_metric_arrays() -> None:
    def bad_chunk(runtime, payload, slots, coordinate):
        del runtime, payload, slots, coordinate
        return {MODEL: jnp.asarray([1.0]), TRAIN_LOSS: jnp.asarray(1.0), "extra": 1}

    adapter = ChunkKernelAdapter(
        chunk_fn=bad_chunk,
        reads=(MODEL,),
        writes=(MODEL, TRAIN_LOSS),
        metric_slots=(TRAIN_LOSS,),
    )
    kernel = adapter.to_kernel(_Payload(n_train_batches=1))

    with pytest.raises(ChunkKernelAdapterError, match="undeclared writes"):
        kernel(
            {MODEL: jnp.asarray([0.0])},
            ProgressCoordinate(run_id="bad", phase="train_chunk"),
            {RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime()},
        )

    def array_metric_chunk(runtime, payload, slots, coordinate):
        del runtime, payload, slots, coordinate
        return {MODEL: jnp.asarray([1.0]), TRAIN_LOSS: jnp.asarray(1.0)}

    metric_adapter = ChunkKernelAdapter(
        chunk_fn=array_metric_chunk,
        reads=(MODEL,),
        writes=(MODEL, TRAIN_LOSS),
        metric_slots=(TRAIN_LOSS,),
    )
    with pytest.raises(ChunkKernelAdapterError, match="Python float"):
        metric_adapter.to_kernel(_Payload(n_train_batches=1))(
            {MODEL: jnp.asarray([0.0])},
            ProgressCoordinate(run_id="metric", phase="train_chunk"),
            {RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime()},
        )


def test_guard_predicate_or_semantics() -> None:
    predicate = make_stop_predicate(_Payload(n_train_batches=5))
    coordinate = ProgressCoordinate(run_id="guards", phase="train_chunk")

    assert not predicate(
        {COMPLETED_BATCHES: 2}, coordinate, {RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime()}
    )
    assert predicate(
        {COMPLETED_BATCHES: 5}, coordinate, {RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime()}
    )
    assert predicate(
        {COMPLETED_BATCHES: 3},
        coordinate,
        {RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime(stop_after_batches=3)},
    )
    assert predicate(
        {COMPLETED_BATCHES: 1, "zero_adversary_guard": {"should_stop": True}},
        coordinate,
        {RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime()},
    )


def test_slot_schema_keeps_sink_slots_out_of_checkpoint_custody() -> None:
    checkpoint_slots = checkpoint_slot_specs(CS_SUPERVISED_SCHEMA)
    sink_specs = artifact_sink_specs(CS_SUPERVISED_SCHEMA)

    assert HISTORY_CHUNK_BYTES not in {slot.slot for slot in checkpoint_slots}
    assert [sink.slot for sink in sink_specs] == [HISTORY_CHUNK_BYTES]


def test_execute_training_run_spec_excludes_sink_only_slots_from_checkpoint_custody(
    tmp_path: Path,
) -> None:
    registry, program = _chunked_registry_with_history_sink()
    result = execute_training_run_spec(
        _standard_supervised_run_spec(),
        run_id="rlrmp-sink-only",
        initial_slots=_standard_initial_slots(),
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime()},
        manifest_root=tmp_path / "runs",
        checkpoint_root=tmp_path / "checkpoints",
        registry=registry,
    )

    history_artifacts = [
        artifact
        for artifact in result.manifest.artifacts
        if artifact.role == "rlrmp_training_history_chunk"
    ]
    assert len(history_artifacts) == 2
    assert all(
        Path(artifact.uri).read_bytes().startswith(b"history-") for artifact in history_artifacts
    )
    assert all(
        HISTORY_CHUNK_BYTES not in {slot.slot for slot in write.manifest.slots}
        for write in result.checkpoint_writes
    )

    loaded = load_latest_checkpoint(
        tmp_path / "checkpoints",
        expected_run_spec=_standard_supervised_run_spec(),
        expected_phase_program=program,
        expected_slots=_standard_initial_slots(),
    )
    assert HISTORY_CHUNK_BYTES not in loaded.slots


def test_toy_equivalence_harness_reports_fixed_seed_and_resume_equivalence() -> None:
    paired = run_toy_paired_equivalence(n_chunks=3, seed=42)
    resumed = run_toy_resume_equivalence(n_chunks=3, kill_after_chunk=1, seed=42)

    assert paired.passed
    assert paired.completed_batches == (3, 3)
    assert paired.max_abs_diff == 0.0
    assert resumed.passed
    assert resumed.completed_batches == (3, 3)
    assert resumed.loss_series[0] == resumed.loss_series[1]


def test_resume_structural_abi_rejects_resized_diagnostics_buffer(tmp_path: Path) -> None:
    run_spec = _training_run_spec(tmp_path)
    program = _toy_program()
    coordinate = ProgressCoordinate(
        run_id="resize-caveat",
        phase="train_chunk",
        global_step=1,
        completed_barrier=TOY_BARRIER,
    )
    slots = _toy_initial_slots(seed=0)
    write_checkpoint_transaction(
        tmp_path / "checkpoints",
        run_spec=run_spec,
        phase_program=program,
        barrier_name=TOY_BARRIER,
        coordinate=coordinate,
        slots=slots,
    )
    resized_slots = dict(slots)
    resized_slots[OPTIMIZER] = {
        "step": slots[OPTIMIZER]["step"],
        "diagnostics": jnp.zeros((4,), dtype=jnp.float32),
    }

    with pytest.raises(CheckpointCompatibilityError, match="structural ABI mismatch"):
        load_latest_checkpoint(
            tmp_path / "checkpoints",
            expected_run_spec=run_spec,
            expected_phase_program=program,
            expected_slots=resized_slots,
        )


def _training_run_spec(tmp_path: Path) -> TrainingRunSpec:
    contract = standard_supervised_method_contract()
    return TrainingRunSpec(
        graph=GraphTopologySourceSpec(ref="toy"),
        task=TaskSpec(type="toy"),
        training_config=TrainingConfig(n_batches=1, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="toy_loss", label="toy_loss"),
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=contract,
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
        artifacts=ArtifactPolicySpec(manifest_root=str(tmp_path / "manifests")),
    )


def _minimal_graph() -> dict[str, object]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _standard_supervised_run_spec() -> TrainingRunSpec:
    return TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=1, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state",
                label="target",
                selector="port:gain.output",
                target_value=[0.0],
            )
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    )


def _standard_initial_slots() -> dict[str, object]:
    return {
        MODEL: jnp.array([0.0]),
        OPTIMIZER: {"count": jnp.array([1.0])},
        "prng": jnp.array([0, 1], dtype=jnp.uint32),
    }


def _chunked_registry_with_history_sink() -> tuple[TrainingMethodRegistry, object]:
    contract = standard_supervised_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    phase = program.phases[0].model_copy(
        update={
            "legal_next": ["train_batch"],
            "writes": [*program.phases[0].writes, HISTORY_CHUNK_BYTES],
        }
    )
    update_step = program.update_steps[0].model_copy(
        update={"writes": [*program.update_steps[0].writes, HISTORY_CHUNK_BYTES]}
    )
    barrier = program.checkpoint_barriers[0].model_copy(
        update={
            "artifact_sinks": [
                BarrierArtifactSinkSpec(
                    slot=HISTORY_CHUNK_BYTES,
                    role="rlrmp_training_history_chunk",
                    logical_name="history_chunk.eqx",
                    media_type="application/octet-stream",
                    encoding="raw",
                )
            ]
        }
    )
    transition = PhaseTransitionSpec(
        source="train_batch",
        target="train_batch",
        barrier="after_train_batch",
        guard=MetricGuardSpec(predicate_ref="rlrmp.executor.keep_training", metric_slots=[]),
    )
    program = program.model_copy(
        update={
            "phases": [phase],
            "transitions": [transition],
            "update_steps": [update_step],
            "checkpoint_barriers": [barrier],
        }
    )
    contract = contract.model_copy(
        update={
            "phase_program": program,
            "state_slots": [
                *contract.state_slots,
                StateSlotSpec(name=HISTORY_CHUNK_BYTES, role="auxiliary", required=False),
            ],
        }
    )
    base_kernel = standard_supervised_update_kernels()[
        "feedbax.training.standard_supervised.gradient_update"
    ]

    def gradient_update(slots, coordinate, context):
        assert RLRMP_RUNTIME_CONTEXT_KEY in context
        updates = dict(base_kernel(slots, coordinate, context))
        updates[HISTORY_CHUNK_BYTES] = f"history-{coordinate.global_step}".encode()
        return updates

    def keep_training(slots, coordinate, context):
        del slots
        assert RLRMP_RUNTIME_CONTEXT_KEY in context
        return coordinate.global_step < 2

    registry = TrainingMethodRegistry()
    registry.register(
        TrainingMethodRegistration(
            method_ref="feedbax/standard_supervised/v1",
            payload_schema_id=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=lambda: contract,
            update_kernels_factory=lambda _payload: {
                "feedbax.training.standard_supervised.gradient_update": gradient_update
            },
            guard_predicates_factory=lambda _payload: {
                "rlrmp.executor.keep_training": keep_training
            },
            owner="rlrmp.tests",
            package="rlrmp",
        )
    )
    return registry, program
