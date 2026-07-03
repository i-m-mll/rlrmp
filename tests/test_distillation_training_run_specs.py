"""TrainingRunSpec contract tests for RLRMP distillation methods."""

from __future__ import annotations

import copy

import pytest
from feedbax.contracts.training import TrainingRunSpec
from pydantic import ValidationError

from rlrmp.runtime.training_run_specs import (
    CLOSED_LOOP_DISTILLATION_METHOD_REF,
    CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    GUIDED_DISTILLATION_METHOD_REF,
    GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
)
from rlrmp.train import closed_loop_distillation, guided_distillation


def test_closed_loop_distillation_training_run_spec_round_trips() -> None:
    args = closed_loop_distillation._build_parser().parse_args(
        [
            "--n-batches",
            "17",
            "--batch-size",
            "3",
            "--hidden-size",
            "11",
            "--velocity-weight",
            "0.5",
        ]
    )

    run_spec = closed_loop_distillation.build_closed_loop_distillation_spec(args)
    training_spec = TrainingRunSpec.model_validate(run_spec[FEEDBAX_TRAINING_RUN_SPEC_KEY])
    round_tripped = TrainingRunSpec.model_validate(training_spec.model_dump(mode="json"))

    assert training_spec.method_ref.key == CLOSED_LOOP_DISTILLATION_METHOD_REF
    assert training_spec.method_payload.schema_version == (
        CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION
    )
    assert training_spec.method_payload.payload["loss_surface"]["weights"]["velocity"] == 0.5
    assert [axis.name for axis in training_spec.worker_execution.method_contract.axes] == [
        "batch",
        "replicate",
        "rollout",
    ]
    assert "teacher_reference" in {
        slot.name for slot in training_spec.worker_execution.method_contract.state_slots
    }
    assert round_tripped == training_spec


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
    training_spec = TrainingRunSpec.model_validate(run_spec[FEEDBAX_TRAINING_RUN_SPEC_KEY])
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
