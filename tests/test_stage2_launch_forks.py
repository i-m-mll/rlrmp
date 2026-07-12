from types import SimpleNamespace
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import scripts.prepare_stage2_launch_forks as launch_forks
from scripts.prepare_stage2_launch_forks import (
    extend_optimizer_histories,
    validate_launch_fork,
)
from rlrmp.train.executor.cs_supervised import _run_spec_payload_schema_version


def _optimizer(horizon: int) -> tuple:
    values = [jnp.zeros((1,), dtype=jnp.float32) for _ in range(33)]
    for index in HISTORY_INDICES:
        values[index] = jnp.arange(horizon, dtype=jnp.float32)[None, :]
    return tuple(values)


def test_extend_optimizer_histories_preserves_prefix_and_target_horizon() -> None:
    source = _optimizer(SOURCE_COMPLETED_BATCHES)
    target = _optimizer(TARGET_TOTAL_BATCHES)
    extended = extend_optimizer_histories(
        source,
        target,
        source_completed_batches=SOURCE_COMPLETED_BATCHES,
        target_total_batches=TARGET_TOTAL_BATCHES,
    )
    for index in HISTORY_INDICES:
        assert extended[index].shape[-1] == TARGET_TOTAL_BATCHES
        assert jnp.array_equal(extended[index][..., :SOURCE_COMPLETED_BATCHES], source[index])


def test_extend_optimizer_histories_rejects_non_growing_target_horizon() -> None:
    with pytest.raises(ValueError, match="no compatible batch-history leaves"):
        extend_optimizer_histories(
            _optimizer(SOURCE_COMPLETED_BATCHES),
            _optimizer(SOURCE_COMPLETED_BATCHES),
            source_completed_batches=SOURCE_COMPLETED_BATCHES,
            target_total_batches=TARGET_TOTAL_BATCHES,
        )


def test_validate_launch_fork_requires_row_bound_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        launch_forks,
        "_adaptive_state_from_slot",
        lambda _slot: SimpleNamespace(schedule_start_batch=SOURCE_COMPLETED_BATCHES),
    )
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            completed_training_batches=TARGET_TOTAL_BATCHES,
            metadata={"rlrmp_stage2_launch_fork": {
                "matrix_row_id": "other", "source_completed_batches": SOURCE_COMPLETED_BATCHES,
                "target_total_batches": TARGET_TOTAL_BATCHES,
            }},
        ),
        slots={"completed_batches": SOURCE_COMPLETED_BATCHES,
               "adaptive_epsilon_state": object(),
               "optimizer": SimpleNamespace(payload=b"serialized")},
    )
    with pytest.raises(ValueError, match="wrong matrix row"):
        validate_launch_fork(
            loaded,
            row_id="flat_3e-5-epsilon-ramp",
            source_completed_batches=SOURCE_COMPLETED_BATCHES,
            target_total_batches=TARGET_TOTAL_BATCHES,
        )


def test_stage2_launch_manifest_executes_full_training() -> None:
    manifest = json.loads(
        Path("results/c6c5997/deploy/stage2_rows_manifest.json").read_text()
    )
    for row in manifest["rows"]:
        command = row["command"]
        assert "_run_full_training_from_context" in command
        assert '"--resume"' in command
        assert "scripts/train_cs_nominal_gru.py --run-spec" not in command


def test_manifest_payload_version_tracks_inline_legacy_spec() -> None:
    run_spec = json.loads(
        Path("results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text()
    )
    assert _run_spec_payload_schema_version(run_spec) == "rlrmp.cs_stochastic_gru.v1"
SOURCE_COMPLETED_BATCHES = 12_000
TARGET_TOTAL_BATCHES = 16_500
HISTORY_INDICES = (1, 2, 3, 30, 31, 32)
