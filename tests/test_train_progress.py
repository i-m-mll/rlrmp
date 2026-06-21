"""Tests for the grep-friendly batch-progress logging helper.

Issue a22627a (W5): remote training monitors need a stable per-batch progress
line they can grep for "last batch seen". These tests pin the line format and
cadence, and verify the feedbax ``batch_callbacks`` factory emits lines without
reading any JAX array (no per-step device->host sync is introduced).
"""

from __future__ import annotations

import importlib.util
import logging
import re
from pathlib import Path

import pytest

from rlrmp.train.progress import (
    BATCH_LINE_TOKEN,
    DEFAULT_LOG_EVERY,
    SMOKE_TOTAL_THRESHOLD,
    batch_log_every,
    format_batch_line,
    make_batch_log_callbacks,
    should_log_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, file_path: Path):
    """Import a script-style module by file path without executing as __main__."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBatchLogEvery:
    def test_smoke_size_logs_every_batch(self) -> None:
        assert batch_log_every(SMOKE_TOTAL_THRESHOLD) == 1
        assert batch_log_every(10) == 1

    def test_large_run_uses_default_interval(self) -> None:
        assert batch_log_every(1000) == DEFAULT_LOG_EVERY

    def test_custom_default(self) -> None:
        assert batch_log_every(1000, default=25) == 25


class TestShouldLogBatch:
    def test_logs_on_cadence(self) -> None:
        assert should_log_batch(0, 1000, every=10)
        assert should_log_batch(10, 1000, every=10)
        assert not should_log_batch(5, 1000, every=10)

    def test_always_logs_final_batch(self) -> None:
        # 999 is not a multiple of 10 but is the last index.
        assert should_log_batch(999, 1000, every=10)


class TestFormatBatchLine:
    def test_starts_with_token(self) -> None:
        line = format_batch_line("warmup", 42, 1000)
        assert line.startswith(BATCH_LINE_TOKEN + " ")

    def test_phase_and_batch_fields(self) -> None:
        line = format_batch_line("warmup", 42, 1000)
        assert "phase=warmup" in line
        assert "batch=42/1000" in line

    def test_optional_loss_and_elapsed(self) -> None:
        line = format_batch_line("adversarial", 5, 20, loss=3.21, elapsed=12.34)
        assert "loss=3.21" in line
        assert "elapsed=12.3s" in line

    def test_loss_omitted_when_none(self) -> None:
        line = format_batch_line("warmup", 1, 10)
        assert "loss=" not in line
        assert "elapsed=" not in line

    def test_extra_fields_appended(self) -> None:
        line = format_batch_line("adversarial", 1, 10, adv_loss=0.5)
        assert "adv_loss=0.5" in line

    def test_phase_rejects_spaces(self) -> None:
        with pytest.raises(ValueError):
            format_batch_line("bad phase", 0, 1)

    def test_matches_documented_contract(self) -> None:
        """The example line in the module docstring must parse with this regex.

        poll_run.sh greps for ``BATCH`` then extracts ``batch=<i>/<n>``.
        """
        line = format_batch_line("warmup", 42, 1000, loss=3.21, elapsed=12.3)
        match = re.search(r"\bBATCH\b.*\bbatch=(\d+)/(\d+)\b", line)
        assert match is not None
        assert match.group(1) == "42"
        assert match.group(2) == "1000"


class TestMakeBatchLogCallbacks:
    def test_keys_follow_cadence_and_include_final(self) -> None:
        callbacks = make_batch_log_callbacks("warmup", 1000, every=100)
        keys = sorted(callbacks)
        assert keys[0] == 0
        assert 100 in keys
        # Final batch index (999) is always logged.
        assert 999 in keys

    def test_smoke_size_logs_every_batch(self) -> None:
        callbacks = make_batch_log_callbacks("warmup", 5)
        assert sorted(callbacks) == [0, 1, 2, 3, 4]

    def test_respects_start_batch(self) -> None:
        callbacks = make_batch_log_callbacks("warmup", 10, start_batch=3)
        assert min(callbacks) >= 3

    def test_callbacks_are_zero_arg_and_emit_lines(self, caplog) -> None:
        log = logging.getLogger("test_progress")
        clock_values = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        callbacks = make_batch_log_callbacks(
            "warmup", 3, logger=log, clock=lambda: next(clock_values)
        )
        with caplog.at_level(logging.INFO, logger="test_progress"):
            for funcs in callbacks.values():
                for func in funcs:
                    func()  # zero-arg invocation, as feedbax calls them
        lines = [r.message for r in caplog.records]
        assert all(m.startswith(BATCH_LINE_TOKEN) for m in lines)
        assert any("phase=warmup" in m for m in lines)
        assert any("elapsed=" in m for m in lines)

    def test_train_phase_callbacks_emit_lines(self, caplog) -> None:
        """The ``train`` phase used by train_part2_5.py emits BATCH lines."""
        log = logging.getLogger("test_progress_train")
        callbacks = make_batch_log_callbacks("train", 5, logger=log)
        with caplog.at_level(logging.INFO, logger="test_progress_train"):
            for funcs in callbacks.values():
                for func in funcs:
                    func()
        lines = [r.message for r in caplog.records]
        assert lines
        assert all(m.startswith(BATCH_LINE_TOKEN) for m in lines)
        assert any("phase=train batch=" in m for m in lines)


class TestTrainScriptWiring:
    """Both training entry-points wire the zero-arg BATCH-progress callbacks.

    FIX D / W5: ``train_part2_5.py`` must wire ``make_batch_log_callbacks`` into
    its ``train_pair`` call, the same host-side pattern ``train_minimax.py``
    already uses for its warmup phase — so neither path goes dark between the
    JIT message and the completion sentinel, and no new per-step device->host
    sync is introduced.
    """

    def test_minimax_imports_progress_helper(self) -> None:
        module = _load_script_module(
            "train_minimax_progress_wiring",
            REPO_ROOT / "scripts" / "train_minimax.py",
        )
        assert hasattr(module, "make_batch_log_callbacks")

    def test_part2_5_imports_progress_helper(self) -> None:
        module = _load_script_module(
            "train_part2_5_progress_wiring",
            REPO_ROOT / "scripts" / "train_part2_5.py",
        )
        assert hasattr(module, "make_batch_log_callbacks")

    def test_part2_5_run_training_wires_batch_callbacks(self) -> None:
        """``run_training`` references the BATCH-progress helper + callbacks kwarg."""
        source = (REPO_ROOT / "scripts" / "train_part2_5.py").read_text(
            encoding="utf-8"
        )
        assert "make_batch_log_callbacks(" in source
        assert "batch_callbacks" in source
