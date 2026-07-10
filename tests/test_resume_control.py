"""Tests for fail-closed training resume launch controls."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from rlrmp.train.cs_nominal_gru import build_parser
from rlrmp.train.config_cli import parse_config
from rlrmp.train.resume_control import (
    LAUNCH_CONTINUATION_PREFIX,
    emit_launch_continuation,
    resolve_launch_continuation,
)
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
