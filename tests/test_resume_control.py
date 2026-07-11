"""Tests for fail-closed training resume launch controls."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.training.checkpoint_custody import (
    fork_checkpoint_transaction,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)

from rlrmp.train.cs_nominal_gru import build_parser
from rlrmp.train.config_cli import parse_config
from rlrmp.train.executor.cs_supervised import (
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
from rlrmp.train.training_configs import MinimaxConfig


def _write_latest(path: Path, *, global_step: int) -> Path:
    latest_path = path / "latest.json"
    path.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {
                "completed_coordinate": {
                    "phase": "train",
                    "global_step": global_step,
                    "completed_barrier": "after_train_chunk",
                }
            }
        ),
        encoding="utf-8",
    )
    return latest_path


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


def test_resume_summary_uses_latest_json_completed_batches(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    latest_path = _write_latest(checkpoint_root, global_step=12_000)

    continuation = resolve_launch_continuation(
        checkpoint_root=checkpoint_root,
        resume_requested=True,
        allow_fresh_start=False,
        stop_target_batches=12_500,
    )

    assert continuation.resume is True
    assert continuation.resume_source == str(latest_path)
    assert continuation.completed_batches == 12_000
    assert continuation.stop_target_batches == 12_500
    assert continuation.continuation_batches == 500
    assert "continuation_batches=500" in continuation.format_line()


def test_resume_uses_manifest_total_not_custody_order_coordinate(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    manifest_path = checkpoint_root / "transactions/tx-baseline/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"completed_training_batches": 12_000}),
        encoding="utf-8",
    )
    (checkpoint_root / "latest.json").write_text(
        json.dumps(
            {
                "manifest_relative_path": "transactions/tx-baseline/manifest.json",
                "completed_coordinate": {"global_step": 24},
                "completed_training_batches": 12_000,
            }
        ),
        encoding="utf-8",
    )

    continuation = resolve_launch_continuation(
        checkpoint_root=checkpoint_root,
        resume_requested=True,
        allow_fresh_start=False,
        stop_target_batches=12_200,
    )

    assert continuation.completed_batches == 12_000
    assert continuation.stop_target_batches == 12_200
    assert continuation.continuation_batches == 200


def test_resume_rejects_manifest_pointer_escaping_checkpoint_root(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "latest.json").write_text(
        json.dumps({"manifest_relative_path": "../outside/manifest.json"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes checkpoint root"):
        completed_batches_from_latest(checkpoint_root / "latest.json")


def test_non_positive_continuation_is_hard_error(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _write_latest(checkpoint_root, global_step=12_500)

    with pytest.raises(ValueError, match="non-positive launch continuation"):
        resolve_launch_continuation(
            checkpoint_root=checkpoint_root,
            resume_requested=True,
            allow_fresh_start=False,
            stop_target_batches=12_500,
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
