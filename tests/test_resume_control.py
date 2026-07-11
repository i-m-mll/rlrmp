"""Tests for fail-closed training resume launch controls."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.training.checkpoint_custody import (
    CheckpointIntegrityError,
    fork_checkpoint_transaction,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)

from rlrmp.train.cs_nominal_gru import build_parser
from rlrmp.train.config_cli import parse_config
from rlrmp.train.executor.cs_supervised import (
    _cs_supervised_execution_registry,
    build_cs_supervised_native_initial_slots,
    build_run_spec_execution_context,
)
from rlrmp.train.executor.slots import OPTIMIZER
from rlrmp.train.resume_control import (
    CS_SUPERVISED_BATCH_INDEXED_CHECKPOINT_LEAVES,
    LAUNCH_CONTINUATION_PREFIX,
    LaunchContinuation,
    attach_cs_supervised_checkpoint_continuation,
    completed_batches_from_latest,
    declare_cs_supervised_checkpoint_continuation,
    emit_launch_continuation,
    resolve_launch_continuation,
)
from rlrmp.runtime.training_run_specs import feedbax_training_run_spec_from_payload
from rlrmp.runtime.checkpoint_custody import cs_custody_training_spec
from rlrmp.train.training_configs import MinimaxConfig


def _baseline_recipe_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )


@pytest.fixture(scope="module")
def custody_checkpoint_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize one valid custody source for typed-document canaries."""

    recipe_path = _baseline_recipe_path()
    parser = build_parser()
    context = build_run_spec_execution_context(
        parser.parse_args(["--run-spec", str(recipe_path)]),
        parser=parser,
    )
    source_spec = feedbax_training_run_spec_from_payload(context.run_spec)
    source_slots, _runtime = build_cs_supervised_native_initial_slots(
        run_spec=context.run_spec,
        hps=context.hps,
        args=context.args,
        key=jax.random.PRNGKey(int(context.args.seed)),
    )
    root = tmp_path_factory.mktemp("typed-custody") / "source"
    write_checkpoint_transaction(
        root,
        run_spec=source_spec,
        phase_program=source_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_train_chunk",
        coordinate=ProgressCoordinate(
            run_id="typed-custody-source",
            phase="train_chunk",
            program_step=24,
            completed_barrier="after_train_chunk",
        ),
        slots=source_slots,
        completed_training_batches=12_000,
    )
    return root


def _copy_custody_documents(source_root: Path, target_root: Path) -> tuple[Path, Path]:
    """Copy just the published pointer and manifest for reader-only mutation tests."""

    pointer_path = source_root / "latest.json"
    pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_relative_path = Path(pointer_payload["manifest_relative_path"])
    target_pointer = target_root / "latest.json"
    target_manifest = target_root / manifest_relative_path
    target_manifest.parent.mkdir(parents=True)
    shutil.copyfile(source_root / manifest_relative_path, target_manifest)
    target_pointer.parent.mkdir(parents=True, exist_ok=True)
    target_pointer.write_text(json.dumps(pointer_payload), encoding="utf-8")
    return target_pointer, target_manifest


def test_resume_without_latest_json_is_hard_error(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "row" / "checkpoints"

    with pytest.raises(FileNotFoundError, match=r"checkpoints/latest\.json"):
        resolve_launch_continuation(
            checkpoint_root=checkpoint_root,
            resume_requested=True,
            allow_fresh_start=False,
            stop_target_batches=12_500,
        )


def test_allow_fresh_start_override_emits_fresh_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    continuation = resolve_launch_continuation(
        checkpoint_root=tmp_path / "checkpoints",
        resume_requested=True,
        allow_fresh_start=True,
        stop_target_batches=500,
    )

    with caplog.at_level(logging.INFO, logger="rlrmp.tests.resume"):
        emit_launch_continuation(continuation, logger=logging.getLogger("rlrmp.tests.resume"))

    line = capsys.readouterr().out.strip()
    assert continuation.resume is False
    assert line.startswith(f"{LAUNCH_CONTINUATION_PREFIX} ")
    assert "resume_source=fresh-start-override" in line
    assert "completed_batches=0" in line
    assert "stop_target_batches=500" in line
    assert "continuation_batches=500" in line
    assert caplog.records[-1].message == line


def test_resume_uses_manifest_total_not_custody_order_coordinate(
    custody_checkpoint_root: Path,
) -> None:
    continuation = resolve_launch_continuation(
        checkpoint_root=custody_checkpoint_root,
        resume_requested=True,
        allow_fresh_start=False,
        stop_target_batches=12_200,
    )

    assert continuation.completed_batches == 12_000
    assert continuation.stop_target_batches == 12_200
    assert continuation.continuation_batches == 200


def test_resume_rejects_manifest_pointer_escaping_checkpoint_root(
    tmp_path: Path,
    custody_checkpoint_root: Path,
) -> None:
    latest_path, _manifest_path = _copy_custody_documents(
        custody_checkpoint_root,
        tmp_path / "checkpoints",
    )
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    pointer["manifest_relative_path"] = "../outside/manifest.json"
    latest_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="escapes custody root"):
        completed_batches_from_latest(latest_path)


@pytest.mark.parametrize(
    ("document", "field", "value"),
    [
        ("latest", "schema_id", "rlrmp.invalid.latest"),
        ("latest", "schema_version", "rlrmp.invalid.latest.v1"),
        ("manifest", "schema_id", "rlrmp.invalid.manifest"),
        ("manifest", "schema_version", "rlrmp.invalid.manifest.v1"),
    ],
)
def test_resume_rejects_invalid_typed_custody_schema_identity_or_version(
    tmp_path: Path,
    custody_checkpoint_root: Path,
    document: str,
    field: str,
    value: str,
) -> None:
    latest_path, manifest_path = _copy_custody_documents(
        custody_checkpoint_root,
        tmp_path / "checkpoints",
    )
    path = latest_path if document == "latest" else manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="invalid|unsupported"):
        completed_batches_from_latest(latest_path)


def test_resume_rejects_pointer_manifest_batch_total_disagreement(
    tmp_path: Path,
    custody_checkpoint_root: Path,
) -> None:
    latest_path, _manifest_path = _copy_custody_documents(
        custody_checkpoint_root,
        tmp_path / "checkpoints",
    )
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    pointer["completed_training_batches"] = 24
    latest_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ValueError, match="completed_training_batches disagrees"):
        completed_batches_from_latest(latest_path)


def test_resume_uses_feedbax_legacy_global_step_migration(
    tmp_path: Path,
    custody_checkpoint_root: Path,
) -> None:
    latest_path, _manifest_path = _copy_custody_documents(
        custody_checkpoint_root,
        tmp_path / "checkpoints",
    )
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    pointer["schema_version"] = "feedbax.manifest.training_checkpoint_latest_pointer.v2"
    coordinate = pointer["completed_coordinate"]
    coordinate["global_step"] = coordinate.pop("program_step")
    latest_path.write_text(json.dumps(pointer), encoding="utf-8")

    assert completed_batches_from_latest(latest_path) == 12_000


def test_non_positive_continuation_is_hard_error(
    custody_checkpoint_root: Path,
) -> None:

    with pytest.raises(ValueError, match="non-positive launch continuation"):
        resolve_launch_continuation(
            checkpoint_root=custody_checkpoint_root,
            resume_requested=True,
            allow_fresh_start=False,
            stop_target_batches=12_000,
        )


def test_cs_supervised_continuation_declares_actual_baseline_diagnostic_paths() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )
    training_spec = feedbax_training_run_spec_from_payload(
        json.loads(recipe_path.read_text(encoding="utf-8"))
    )
    continuation = LaunchContinuation(
        resume=True,
        resume_source="/tmp/checkpoints/latest.json",
        completed_batches=12_000,
        stop_target_batches=12_200,
        continuation_batches=200,
    )

    attached = attach_cs_supervised_checkpoint_continuation(training_spec, continuation)

    request = attached.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.target_total_batches == 12_200
    assert request.additional_batches is None
    assert [(leaf.slot, leaf.tree_path) for leaf in request.batch_indexed_leaves] == [
        (leaf.slot, leaf.tree_path)
        for leaf in CS_SUPERVISED_BATCH_INDEXED_CHECKPOINT_LEAVES
    ]
    assert [(leaf.slot, leaf.tree_path) for leaf in request.batch_indexed_leaves] == [
        ("optimizer", "/1"),
        ("optimizer", "/2"),
        ("optimizer", "/3"),
        ("optimizer", "/30"),
        ("optimizer", "/31"),
        ("optimizer", "/32"),
    ]


def test_cs_supervised_continuation_does_not_change_exact_parity_spec() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )
    training_spec = feedbax_training_run_spec_from_payload(
        json.loads(recipe_path.read_text(encoding="utf-8"))
    )
    fresh = LaunchContinuation(
        resume=False,
        resume_source="fresh-start",
        completed_batches=0,
        stop_target_batches=12_000,
        continuation_batches=12_000,
    )

    assert attach_cs_supervised_checkpoint_continuation(training_spec, fresh) is training_spec


def test_target_bound_launch_fork_still_attaches_runnable_continuation() -> None:
    training_spec = feedbax_training_run_spec_from_payload(
        json.loads(_baseline_recipe_path().read_text(encoding="utf-8"))
    )
    launch_fork = LaunchContinuation(
        resume=True,
        resume_source="/tmp/launch-fork/latest.json",
        completed_batches=12_000,
        stop_target_batches=12_200,
        continuation_batches=200,
        source_target_batches=12_200,
    )

    attached = attach_cs_supervised_checkpoint_continuation(training_spec, launch_fork)

    request = attached.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.target_total_batches == 12_200


def test_cs_supervised_resume_registry_uses_attached_custody_contract() -> None:
    recipe_path = Path(__file__).resolve().parents[1] / "results/cb3685a/runs/seam_probe.json"
    parser = build_parser()
    context = build_run_spec_execution_context(
        parser.parse_args(["--run-spec", str(recipe_path)]),
        parser=parser,
    )
    continuation = LaunchContinuation(
        resume=True,
        resume_source="/tmp/checkpoints/latest.json",
        completed_batches=12_000,
        stop_target_batches=12_200,
        continuation_batches=200,
    )
    training_spec = attach_cs_supervised_checkpoint_continuation(
        cs_custody_training_spec(context.run_spec),
        continuation,
    )

    registry = _cs_supervised_execution_registry(training_spec)
    execution_contract = registry.resolve(training_spec.method_ref, path="/method_ref").contract_factory()

    assert execution_contract == training_spec.worker_execution.method_contract
    request = training_spec.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.target_total_batches == 12_200
    barrier = execution_contract.phase_program.checkpoint_barriers[0]
    assert {slot.slot for slot in barrier.slots} == {
        "model",
        "optimizer",
        "prng",
        "completed_batches",
        "history",
        "adversary_policy",
        "adversary_optimizer",
        "adaptive_epsilon_state",
        "checkpoint_metadata",
    }


def test_stage2_authoring_declares_12000_to_16500_total_horizon() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )
    training_spec = feedbax_training_run_spec_from_payload(
        json.loads(recipe_path.read_text(encoding="utf-8"))
    )

    attached = declare_cs_supervised_checkpoint_continuation(
        training_spec,
        source_completed_batches=12_000,
        target_total_batches=16_500,
    )

    request = attached.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.target_total == 16_500
    assert request.additional_batches is None


def test_nominal_fork_extends_the_real_six_diagnostic_optimizer_leaves(
    tmp_path: Path,
) -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )
    parser = build_parser()
    context = build_run_spec_execution_context(
        parser.parse_args(["--run-spec", str(recipe_path)]),
        parser=parser,
    )
    source_spec = feedbax_training_run_spec_from_payload(context.run_spec)
    source_slots, _runtime = build_cs_supervised_native_initial_slots(
        run_spec=context.run_spec,
        hps=context.hps,
        args=context.args,
        key=jax.random.PRNGKey(int(context.args.seed)),
    )
    expected_slots = dict(source_slots)
    target_optimizer = list(source_slots[OPTIMIZER])
    for index in (1, 2, 3, 30, 31, 32):
        values = jnp.asarray(target_optimizer[index])
        target_optimizer[index] = jnp.pad(values, ((0, 0), (0, 200)))
    expected_slots[OPTIMIZER] = tuple(target_optimizer)
    target_spec = declare_cs_supervised_checkpoint_continuation(
        source_spec,
        source_completed_batches=12_000,
        target_total_batches=12_200,
    )
    program = source_spec.worker_execution.method_contract.phase_program
    coordinate = ProgressCoordinate(
        run_id="nominal-seam",
        phase="train_chunk",
        global_step=12_000,
        completed_barrier="after_train_chunk",
    )
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    write_checkpoint_transaction(
        source_root,
        run_spec=source_spec,
        phase_program=program,
        barrier_name="after_train_chunk",
        coordinate=coordinate,
        slots=source_slots,
        completed_training_batches=12_000,
    )

    fork_checkpoint_transaction(
        source_root,
        target_root,
        target_run_spec=target_spec,
        target_phase_program=program,
        expected_slots=expected_slots,
        continuation_request=target_spec.checkpoint_progress.continuation,
    )
    resumed = load_latest_checkpoint(
        target_root,
        expected_run_spec=target_spec,
        expected_phase_program=program,
        expected_slots=expected_slots,
        continuation_request=target_spec.checkpoint_progress.continuation,
    )
    optimizer = resumed.slots[OPTIMIZER]
    for index in (1, 2, 3, 30, 31, 32):
        assert optimizer[index].shape == (5, 12_200)
        prefix = optimizer[index][..., :12_000]
        source = source_slots[OPTIMIZER][index]
        same = jnp.equal(prefix, source)
        if jnp.issubdtype(prefix.dtype, jnp.floating):
            same = jnp.logical_or(same, jnp.logical_and(jnp.isnan(prefix), jnp.isnan(source)))
        assert bool(jnp.all(same))


def test_cli_flags_document_resume_override_and_global_stop_target() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    normalized_help = " ".join(help_text.split())

    assert parser.parse_args(["--allow-fresh-start"]).allow_fresh_start is True
    assert "--allow-fresh-start" in help_text
    assert "Global completed-batch index" in normalized_help
    assert "not a relative count" in normalized_help

    minimax_config = parse_config(
        MinimaxConfig,
        ["--allow-fresh-start"],
        description="test minimax config",
    )
    assert minimax_config.allow_fresh_start is True
