"""TrainingRunSpec contract tests for RLRMP distillation methods."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest
from feedbax.contracts.training import TrainingConfig, TrainingRunSpec
from pydantic import BaseModel, ValidationError

from rlrmp.runtime.training_run_specs import (
    CLOSED_LOOP_DISTILLATION_METHOD_REF,
    CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    GUIDED_DISTILLATION_METHOD_REF,
    GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
    ClosedLoopDistillationMethodPayload,
    GuidedDistillationMethodPayload,
    MissingTrainingRunSpecFieldError,
    _closed_loop_distillation_payload_model,
    _guided_distillation_payload_model,
    build_distillation_training_run_spec,
    closed_loop_distillation_method_payload,
    guided_distillation_method_payload,
)
from rlrmp.train import closed_loop_distillation, guided_distillation
from rlrmp.train.distillation_native import (
    execute_distillation_training_run_spec_native,
    native_distillation_model_from_slot,
)
from rlrmp.train.executor.equivalence import (
    FAMILY_TOLERANCE,
    assert_paired_equivalent,
    run_paired_equivalence,
)


DISTILLATION_PAYLOAD_SPECS = (
    (
        Path("results/9727d79/runs/h0_extlqg_6d_standard_graph_distillation.json"),
        "guided_distillation",
    ),
    (
        Path("results/9727d79/runs/h0_hinf_6d_guided_distillation.json"),
        "guided_distillation",
    ),
    (
        Path("results/9727d79/runs/h0_hinf_6d_standard_graph_distillation.json"),
        "guided_distillation",
    ),
    (
        Path("results/a378b34/runs/h0_extlqg_6d_closed_loop_distillation.json"),
        "closed_loop_distillation",
    ),
)


def _tracked_closed_loop_spec_with_horizon() -> dict:
    payload = json.loads(
        Path("results/a378b34/runs/h0_extlqg_6d_closed_loop_distillation.json").read_text(
            encoding="utf-8"
        )
    )
    payload["teacher_contract"]["horizon"] = 60
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _collect_model_instances(model: BaseModel) -> list[BaseModel]:
    instances = [model]
    for value in model.__dict__.values():
        if isinstance(value, BaseModel):
            instances.extend(_collect_model_instances(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, BaseModel):
                    instances.extend(_collect_model_instances(item))
    return instances


def _array_tree(value: object) -> object:
    return eqx.filter(value, eqx.is_array)


def _assert_array_trees_close(left: object, right: object) -> None:
    left_tree = _array_tree(left)
    right_tree = _array_tree(right)
    if jt.structure(left_tree) != jt.structure(right_tree):
        left_leaves = jt.leaves(left_tree)
        right_leaves = jt.leaves(right_tree)
        assert [leaf.shape for leaf in left_leaves] == [leaf.shape for leaf in right_leaves]
        assert max(
            (
                float(jnp.max(jnp.abs(left_leaf - right_leaf)))
                for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
            ),
            default=0.0,
        ) <= FAMILY_TOLERANCE.atol
        return
    report = run_paired_equivalence(
        "distillation.array_tree",
        lambda: left_tree,
        lambda: right_tree,
        left_label="left",
        right_label="right",
    )
    assert_paired_equivalent(report)


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


@pytest.mark.parametrize(("path", "method"), DISTILLATION_PAYLOAD_SPECS)
def test_tracked_distillation_method_payload_corpus_validates(path: Path, method: str) -> None:
    run_spec = json.loads(path.read_text(encoding="utf-8"))

    if method == "closed_loop_distillation":
        model = _closed_loop_distillation_payload_model(run_spec, spec_path=path)
        assert isinstance(model, ClosedLoopDistillationMethodPayload)
    else:
        model = _guided_distillation_payload_model(run_spec, spec_path=path)
        assert isinstance(model, GuidedDistillationMethodPayload)

    assert model.model_dump(mode="json", exclude_none=True)


def test_distillation_method_payload_matches_pre_refactor_golden_fixture() -> None:
    fixtures = json.loads(
        Path("tests/fixtures/distillation_method_payload_golden.json").read_text(
            encoding="utf-8"
        )
    )
    closed_loop_spec = _tracked_closed_loop_spec_with_horizon()
    guided_spec = guided_distillation.build_distillation_spec(
        guided_distillation.build_parser().parse_args([])
    )

    cases = {
        "closed_loop_tracked_with_horizon": closed_loop_distillation_method_payload(
            closed_loop_spec
        ),
        "guided_generated_default": guided_distillation_method_payload(guided_spec),
    }

    for name, envelope in cases.items():
        assert _canonical_json(envelope.model_dump(mode="json", exclude_none=True)) == (
            fixtures["cases"][name]["method_payload_envelope_json"]
        )


def test_distillation_payload_submodels_reject_extra_keys() -> None:
    closed_loop_model = _closed_loop_distillation_payload_model(
        _tracked_closed_loop_spec_with_horizon()
    )
    guided_model = _guided_distillation_payload_model(
        guided_distillation.build_distillation_spec(
            guided_distillation.build_parser().parse_args([])
        )
    )
    instances_by_type = {
        type(instance): instance
        for model in (closed_loop_model, guided_model)
        for instance in _collect_model_instances(model)
    }

    for model_type, instance in instances_by_type.items():
        payload = instance.model_dump(mode="json", exclude_none=True)
        payload["unexpected_extra_key"] = "rejected"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            model_type.model_validate(payload)


def test_closed_loop_distillation_builder_fails_closed_without_horizon() -> None:
    args = closed_loop_distillation._build_parser().parse_args([])
    run_spec = closed_loop_distillation.build_closed_loop_distillation_spec(args)
    del run_spec["teacher_contract"]["horizon"]

    with pytest.raises(MissingTrainingRunSpecFieldError, match="teacher_contract.horizon"):
        build_distillation_training_run_spec(
            run_spec,
            method="closed_loop_distillation",
            output_dir=Path(run_spec["artifact_output_dir"]),
            spec_path=Path(run_spec["training_entry"]["run_spec_path"]),
        )


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
        "done",
    ]
    assert "teacher_bank" in {
        slot.name for slot in training_spec.worker_execution.method_contract.state_slots
    }
    assert round_tripped == training_spec


def test_closed_loop_distillation_native_executor_runs_fixed_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        closed_loop_distillation,
        "DEFAULT_OUTPUT_DIR",
        str(tmp_path / "legacy-closed-loop"),
    )
    args = closed_loop_distillation._build_parser().parse_args(
        [
            "--n-batches",
            "1",
            "--batch-size",
            "1",
            "--n-replicates",
            "1",
            "--hidden-size",
            "6",
            "--output-dir",
            str(tmp_path / "legacy-closed-loop"),
        ]
    )
    source_spec = closed_loop_distillation.build_closed_loop_distillation_spec(args)
    with pytest.raises(RuntimeError, match="Legacy Feedbax train_pair support has been removed"):
        closed_loop_distillation.run_closed_loop_distillation_training(
            spec=source_spec,
            key=jr.PRNGKey(0),
            n_batches=1,
            batch_size=1,
            n_replicates=1,
            hidden_size=6,
            confirm_full_train=True,
        )
    native = execute_distillation_training_run_spec_native(
        source_spec,
        method="closed_loop_distillation",
        run_id="native-closed-loop-fixed-seed",
        key=jr.PRNGKey(0),
        manifest_root=tmp_path / "manifests" / "closed-loop-native",
        checkpoint_root=tmp_path / "checkpoints" / "closed-loop-native",
        manifest_conflict_policy="reuse-identical",
    )
    native_model = native_distillation_model_from_slot(
        native.final_slots["model"],
        source_run_spec=source_spec,
        method="closed_loop_distillation",
        key=jr.PRNGKey(0),
    )

    assert native_model is not None
    assert int(native.final_slots["completed_batches"]) == 1
    assert native.final_slots["train_loss"] != 0.0


def test_guided_distillation_native_executor_matches_fixed_seed_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = guided_distillation.build_parser().parse_args(
        [
            "--n-batches",
            "1",
            "--batch-size",
            "1",
            "--n-replicates",
            "1",
            "--hidden-size",
            "6",
            "--n-jvp-directions",
            "1",
            "--output-dir",
            str(tmp_path / "legacy-guided"),
            "--no-checkpoint",
        ]
    )
    source_spec = guided_distillation.build_distillation_spec(args)
    cli_result = guided_distillation.run_guided_distillation_training(args)
    native = execute_distillation_training_run_spec_native(
        source_spec,
        method="guided_distillation",
        run_id="native-guided-fixed-seed",
        key=jr.PRNGKey(0),
        manifest_root=tmp_path / "manifests" / "guided-native",
        checkpoint_root=tmp_path / "checkpoints" / "guided-native",
        manifest_conflict_policy="reuse-identical",
    )
    native_model = native_distillation_model_from_slot(
        native.final_slots["model"],
        source_run_spec=source_spec,
        method="guided_distillation",
        key=jr.PRNGKey(0),
    )

    assert cli_result["completed_batches"] == 1
    assert Path(cli_result["training_manifest_path"]).is_file()
    assert guided_distillation.standard_controller_parts(native_model).hidden_cell.weight_ih.shape == (
        1,
        18,
        6,
    )
    assert int(native.final_slots["completed_batches"]) == 1
    assert native.final_slots["train_loss"] != 0.0


@pytest.mark.parametrize(
    ("method", "barrier"),
    [
        ("closed_loop_distillation", "after_closed_loop_rollout_distillation"),
        ("guided_distillation", "after_teacher_forced_warm_start"),
    ],
)
def test_distillation_native_executor_same_length_resume_matches_uninterrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    barrier: str,
) -> None:
    if method == "closed_loop_distillation":
        monkeypatch.setattr(
            closed_loop_distillation,
            "DEFAULT_OUTPUT_DIR",
            str(tmp_path / "closed-loop"),
        )
        args = closed_loop_distillation._build_parser().parse_args(
            [
                "--n-batches",
                "1",
                "--batch-size",
                "1",
                "--n-replicates",
                "1",
                "--hidden-size",
                "6",
                "--output-dir",
                str(tmp_path / "closed-loop"),
            ]
        )
        source_spec = closed_loop_distillation.build_closed_loop_distillation_spec(args)
    else:
        args = guided_distillation.build_parser().parse_args(
            [
                "--n-batches",
                "1",
                "--batch-size",
                "1",
                "--n-replicates",
                "1",
                "--hidden-size",
                "6",
                "--n-jvp-directions",
                "1",
                "--output-dir",
                str(tmp_path / "guided"),
                "--no-checkpoint",
            ]
        )
        source_spec = guided_distillation.build_distillation_spec(args)

    full = execute_distillation_training_run_spec_native(
        source_spec,
        method=method,
        run_id=f"native-{method}-full",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "full",
        checkpoint_root=tmp_path / "checkpoints" / "full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "resume"
    stopped = execute_distillation_training_run_spec_native(
        source_spec,
        method=method,
        run_id=f"native-{method}-resume",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "stopped",
        checkpoint_root=checkpoint_root,
        stop_after_barrier=barrier,
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_distillation_training_run_spec_native(
        source_spec,
        method=method,
        run_id=f"native-{method}-resume",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "resumed",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )

    _assert_array_trees_close(full.final_slots["model"], resumed.final_slots["model"])
    assert stopped.final_coordinate.completed_barrier == barrier
    assert resumed.final_coordinate.phase == "done"


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
