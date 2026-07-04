"""TrainingRunSpec contract tests for RLRMP distillation methods."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from feedbax.contracts.training import TrainingConfig, TrainingRunSpec
from pydantic import ValidationError

from rlrmp.runtime.training_run_specs import (
    CLOSED_LOOP_DISTILLATION_METHOD_REF,
    CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    GUIDED_DISTILLATION_METHOD_REF,
    GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
    MissingTrainingRunSpecFieldError,
    build_distillation_training_run_spec,
)
from rlrmp.train import closed_loop_distillation, guided_distillation


def _tracked_closed_loop_spec_with_horizon() -> dict:
    payload = json.loads(
        Path("results/a378b34/runs/h0_extlqg_6d_closed_loop_distillation.json").read_text(
            encoding="utf-8"
        )
    )
    payload["teacher_contract"]["horizon"] = 60
    return payload


def _legacy_distillation_training_config(run_spec: dict, *, method: str) -> TrainingConfig:
    if method == "closed_loop_distillation":
        student = run_spec.get("student_contract", {})
        return TrainingConfig(
            n_batches=int(student.get("n_train_batches", 1)),
            batch_size=int(student.get("batch_size", 1)),
            learning_rate=float(student.get("controller_lr", 1e-3)),
            grad_clip=float(student.get("gradient_clip_norm", 1.0)),
            hidden_dim=int(student.get("hidden_size", 0)),
            network_type="gru",
            n_reach_steps=int(run_spec.get("teacher_contract", {}).get("horizon", 60)),
            effort_weight=float(
                run_spec.get("loss_surface", {})
                .get("weights", {})
                .get("action_force_trajectory", 1.0)
            ),
            snapshot_interval=int(run_spec.get("checkpointing", {}).get("interval_batches", 1)),
        )
    model = run_spec.get("model_contract", {})
    optimizer = run_spec.get("optimizer", {})
    teacher_bank = run_spec.get("teacher_bank", {})
    return TrainingConfig(
        n_batches=int(
            run_spec.get(
                "n_train_batches",
                run_spec.get("training_schedule", {}).get("total_batches", 1),
            )
        ),
        batch_size=int(run_spec.get("batch_size", model.get("batch_size", 1))),
        learning_rate=float(run_spec.get("controller_lr", optimizer.get("controller_lr", 1e-3))),
        grad_clip=float(optimizer.get("gradient_clip_norm", 1.0)),
        hidden_dim=int(model.get("hidden_size", 0)),
        network_type="gru",
        n_reach_steps=int(teacher_bank.get("horizon", 60)),
        effort_weight=float(
            run_spec.get("distillation_surface", {})
            .get("components", {})
            .get("clean_action", {})
            .get("weight", 1.0)
        ),
        snapshot_interval=int(run_spec.get("checkpointing", {}).get("interval_batches", 1)),
    )


def _assert_complete_distillation_spec_matches_legacy_config(
    run_spec: dict,
    *,
    method: str,
    output_dir: Path,
    spec_path: Path,
) -> TrainingRunSpec:
    actual = build_distillation_training_run_spec(
        run_spec,
        method=method,
        output_dir=output_dir,
        spec_path=spec_path,
    )
    legacy = actual.model_copy(
        update={
            "training_config": _legacy_distillation_training_config(run_spec, method=method),
        }
    )
    assert actual.model_dump(mode="json") == legacy.model_dump(mode="json")
    return actual


def test_closed_loop_distillation_training_run_spec_round_trips() -> None:
    run_spec = _tracked_closed_loop_spec_with_horizon()
    training_spec = _assert_complete_distillation_spec_matches_legacy_config(
        run_spec,
        method="closed_loop_distillation",
        output_dir=Path(run_spec["artifact_output_dir"]),
        spec_path=Path("results/a378b34/runs/h0_extlqg_6d_closed_loop_distillation.json"),
    )
    round_tripped = TrainingRunSpec.model_validate(training_spec.model_dump(mode="json"))

    assert training_spec.method_ref.key == CLOSED_LOOP_DISTILLATION_METHOD_REF
    assert training_spec.method_payload.schema_version == (
        CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION
    )
    assert training_spec.method_payload.payload["loss_surface"]["weights"]["velocity"] == 1.0
    assert [axis.name for axis in training_spec.worker_execution.method_contract.axes] == [
        "batch",
        "replicate",
        "rollout",
    ]
    assert "teacher_reference" in {
        slot.name for slot in training_spec.worker_execution.method_contract.state_slots
    }
    assert round_tripped == training_spec


def test_closed_loop_distillation_builder_fails_closed_without_horizon() -> None:
    args = closed_loop_distillation._build_parser().parse_args([])

    with pytest.raises(MissingTrainingRunSpecFieldError, match="teacher_contract.horizon"):
        closed_loop_distillation.build_closed_loop_distillation_spec(args)


def test_guided_distillation_training_run_spec_round_trips() -> None:
    args = guided_distillation.build_parser().parse_args(
        [
            "--n-batches",
            "19",
            "--batch-size",
            "5",
            "--hidden-size",
            "13",
            "--n-jvp-directions",
            "4",
            "--rollout-anchor-weight",
            "0.1",
        ]
    )

    run_spec = guided_distillation.build_distillation_spec(args)
    training_spec = _assert_complete_distillation_spec_matches_legacy_config(
        run_spec,
        method="guided_distillation",
        output_dir=Path(run_spec["artifact_output_dir"]),
        spec_path=Path(run_spec["training_entry"]["run_spec_path"]),
    )
    assert training_spec == TrainingRunSpec.model_validate(run_spec[FEEDBAX_TRAINING_RUN_SPEC_KEY])
    round_tripped = TrainingRunSpec.model_validate(training_spec.model_dump(mode="json"))

    assert training_spec.method_ref.key == GUIDED_DISTILLATION_METHOD_REF
    assert training_spec.method_payload.schema_version == GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION
    assert (
        training_spec.method_payload.payload["distillation_surface"]["config"]["n_jvp_directions"]
        == 4
    )
    assert [phase.name for phase in training_spec.worker_execution.method_contract.phase_program.phases] == [
        "teacher_forced_warm_start",
        "mixed_teacher_student_forcing",
        "mostly_student_forced",
    ]
    assert "teacher_bank" in {
        slot.name for slot in training_spec.worker_execution.method_contract.state_slots
    }
    assert round_tripped == training_spec


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("unknown_method_ref", "unknown method_ref 'rlrmp/not_registered/v1'"),
        ("unsupported_payload_version", "unsupported method payload schema version"),
        ("missing_worker_axes", "axes must declare worker axes"),
    ],
)
def test_distillation_training_run_spec_fails_before_launch(
    mutation: str,
    match: str,
) -> None:
    run_spec = guided_distillation.build_distillation_spec(
        guided_distillation.build_parser().parse_args([])
    )
    payload = copy.deepcopy(run_spec[FEEDBAX_TRAINING_RUN_SPEC_KEY])

    if mutation == "unknown_method_ref":
        payload["method_ref"] = {"package": "rlrmp", "name": "not_registered", "version": "v1"}
    elif mutation == "unsupported_payload_version":
        payload["method_payload"]["schema_version"] = (
            "rlrmp.spec.training_method.guided_distillation_payload.v0"
        )
    elif mutation == "missing_worker_axes":
        payload["worker_execution"]["method_contract"]["axes"] = []
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(mutation)

    with pytest.raises(ValidationError, match=match):
        TrainingRunSpec.model_validate(payload)
