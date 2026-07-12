"""Tiny RLRMP-owned orchestration fixture used by lifecycle contract tests.

This module is deliberately private test support.  It exercises the stock
training executor and orchestration driver without constructing a scientific
model or performing accelerator-heavy work.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    LossTermSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    StandardSupervisedMethodPayload,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_method_contract,
    standard_supervised_update_kernels,
)
from feedbax.contracts.worker import (
    EffectivePhaseSpec,
    MetricGuardSpec,
    PhaseTransitionSpec,
    derive_consistency_predicate,
)
from feedbax.orchestration.events import RunEventEmitter
from feedbax.training.executor import execute_training_run_spec
from feedbax.training.interruption import CancellationDecision


FIXTURE_METHOD_REF = "rlrmp/orchestration_fixture/v1"
FIXTURE_PAYLOAD_SCHEMA_ID = "rlrmp.spec.training_method.orchestration_fixture_payload"
FIXTURE_PAYLOAD_SCHEMA_VERSION = "rlrmp.spec.training_method.orchestration_fixture_payload.v1"


def fixture_method_ref() -> MethodRefSpec:
    """Return the RLRMP-owned fixture method reference."""
    return MethodRefSpec(package="rlrmp", name="orchestration_fixture", version="v1")


def fixture_method_payload(*, scheduled: bool = False) -> MethodPayloadEnvelope:
    """Return a payload whose optimizer is on Feedbax's stock discovery path."""
    optimizer = (
        OptimizerSpec(
            type="adamw",
            params={"weight_decay": 0.0},
            lr_schedule={
                "origin": {"kind": "run_start"},
                "kind": "constant",
                "learning_rate_0": 0.01,
            },
        )
        if scheduled
        else OptimizerSpec(
            type="adamw",
            params={"learning_rate": 0.01, "weight_decay": 0.0},
        )
    )
    payload = StandardSupervisedMethodPayload(optimizer=optimizer)
    return MethodPayloadEnvelope(
        schema_id=FIXTURE_PAYLOAD_SCHEMA_ID,
        schema_version=FIXTURE_PAYLOAD_SCHEMA_VERSION,
        payload=payload.model_dump(mode="json"),
    )


def fixture_method_contract() -> Any:
    """Return a two-step phase program with one durable barrier per batch."""
    base = standard_supervised_method_contract()
    program = base.phase_program.model_copy(deep=True)
    phase = program.phases[0].model_copy(update={"legal_next": ["train_batch"]})
    transition = PhaseTransitionSpec(
        source="train_batch",
        target="train_batch",
        barrier="after_train_batch",
        guard=MetricGuardSpec(
            predicate_ref="rlrmp.orchestration_fixture.continue_training",
            metric_slots=[],
        ),
    )
    program = program.model_copy(update={"phases": [phase], "transitions": [transition]})
    return base.model_copy(
        update={
            "method_ref": FIXTURE_METHOD_REF,
            "method_payload_schema_version": FIXTURE_PAYLOAD_SCHEMA_VERSION,
            "phase_program": program,
        }
    )


def fixture_effective_phase_spec() -> EffectivePhaseSpec:
    """Return the effective phase declaration paired with the fixture contract."""
    contract = fixture_method_contract()
    return EffectivePhaseSpec(
        method_ref=contract.method_ref,
        axes=contract.axes,
        state_slots=contract.state_slots,
        phase_program=contract.phase_program,
        consistency_predicate=derive_consistency_predicate(contract.phase_program),
    )


def register_fixture_method() -> None:
    """Register the fixture method idempotently in this process."""
    if FIXTURE_METHOD_REF in DEFAULT_TRAINING_METHOD_REGISTRY.available_keys():
        return

    base_kernel = standard_supervised_update_kernels()[
        "feedbax.training.standard_supervised.gradient_update"
    ]

    def continue_training(
        slots: Mapping[str, Any], coordinate: Any, context: Mapping[str, Any]
    ) -> bool:
        del slots, context
        return int(coordinate.program_step) < 2

    DEFAULT_TRAINING_METHOD_REGISTRY.register(
        TrainingMethodRegistration(
            method_ref=FIXTURE_METHOD_REF,
            payload_schema_id=FIXTURE_PAYLOAD_SCHEMA_ID,
            payload_schema_version=FIXTURE_PAYLOAD_SCHEMA_VERSION,
            payload_model=StandardSupervisedMethodPayload,
            contract_factory=fixture_method_contract,
            update_kernels_factory=lambda _payload: {
                "feedbax.training.standard_supervised.gradient_update": base_kernel
            },
            guard_predicates_factory=lambda _payload: {
                "rlrmp.orchestration_fixture.continue_training": continue_training
            },
            owner="rlrmp.tests",
            package="rlrmp",
        )
    )


def fixture_training_run_spec(*, seed: int = 17, scheduled: bool = False) -> TrainingRunSpec:
    """Return the tiny training spec used by lifecycle tests."""
    graph = {
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
    return TrainingRunSpec(
        graph={"inline": graph},
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=2, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state",
                label="target",
                selector="port:gain.output",
                target_value=[0.0],
            )
        ),
        method_ref=fixture_method_ref(),
        method_payload=fixture_method_payload(scheduled=scheduled),
        worker_execution=WorkerExecutionSpec(
            method_contract=fixture_method_contract(),
            effective_phase=fixture_effective_phase_spec(),
        ),
        checkpoint_progress={"checkpoint_interval": 1},
        metadata={
            "seeds": {"fixture": seed},
            **(
                {
                    "resume_context": {
                        "schedule_origin_step": 0,
                        "current_step": 0,
                        "optimizer_count_at_current_step": 0,
                    },
                    "optimizer_build_context": {
                        "schedule_origin_step": 0,
                        "current_step": 0,
                        "optimizer_count_at_current_step": 0,
                    },
                }
                if scheduled
                else {}
            ),
        },
    )


def execute_fixture_packet(packet_path: Path) -> None:
    """Execute one fixture row and emit the artifacts consumed by CERTIFY."""
    register_fixture_method()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    run_spec = TrainingRunSpec.model_validate(packet["payload"])
    row_dir = Path(os.environ["FEEDBAX_ROW_DIR"])
    stop_after_batches = packet.get("stop_after_batches")
    initial_slots = {
        "model": 0,
        "optimizer": {"count": 1},
        "prng": [0, 1],
        "batch_counter": 0,
    }

    def cancellation_probe(coordinate: Any) -> CancellationDecision | None:
        if stop_after_batches is None:
            return None
        if int(coordinate.program_step) < int(stop_after_batches):
            return None
        return CancellationDecision(
            action="stop",
            source="test",
            requested_at_unix_seconds=time.time(),
        )

    with RunEventEmitter.from_env(heartbeat_seconds=None) as emitter:
        result = execute_training_run_spec(
            run_spec,
            run_id=packet["row_id"],
            initial_slots=initial_slots,
            manifest_root=row_dir / "feedbax-manifests",
            checkpoint_root=row_dir / "checkpoints",
            run_event_emitter=emitter,
            cancellation_probe=(cancellation_probe if stop_after_batches is not None else None),
        )

    envelope = packet["envelope"]
    environment_fingerprint = os.environ["FEEDBAX_ENV_FINGERPRINT"]
    seeds = run_spec.metadata["seeds"]
    manifest = result.manifest.model_copy(
        update={
            "run_set_id": packet["run_set_id"],
            "intent_hash": envelope["authored_intent"]["intent_hash"],
            "resolved_semantics_root_hash": envelope["resolved_snapshot"]["root_hash"],
            "execution_hash": envelope["execution_capsule"]["execution_hash"],
            "input_data_identities": envelope["immutable_inputs"],
            "metadata": {
                **result.manifest.metadata,
                "environment_fingerprint": environment_fingerprint,
                "seeds": seeds,
            },
        }
    )
    completed = int(result.final_slots.get("batch_counter", 0))
    checkpoint_coordinates = [
        int(write.manifest.completed_coordinate.program_step) for write in result.checkpoint_writes
    ]
    diagnostics = {
        "completed_batches": completed,
        "checkpoint_coordinates": checkpoint_coordinates,
        "lr_trace": {"0": 0.01, "1": 0.01, "2": 0.01},
        "optimizer_build_context": {
            "schedule_origin_step": 0,
            "current_step": 0,
            "optimizer_count_at_current_step": 0,
        },
        "resume_context": {
            "schedule_origin_step": 0,
            "current_step": 0,
            "optimizer_count_at_current_step": 0,
        },
        "seeds": seeds,
        "terminal_status": result.status,
    }
    _write_json(row_dir / "manifest.json", manifest.model_dump(mode="json", exclude_none=True))
    _write_json(row_dir / "training-diagnostics.json", diagnostics)
    _write_json(
        row_dir / "training_summary.json",
        {"status": result.status, "completed_batches": completed},
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    packet_path = args.packet
    if not packet_path.is_absolute():
        packet_path = Path(os.environ["FEEDBAX_ROW_DIR"]) / packet_path
    execute_fixture_packet(packet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIXTURE_METHOD_REF",
    "execute_fixture_packet",
    "fixture_effective_phase_spec",
    "fixture_method_contract",
    "fixture_method_payload",
    "fixture_method_ref",
    "fixture_training_run_spec",
    "register_fixture_method",
]
