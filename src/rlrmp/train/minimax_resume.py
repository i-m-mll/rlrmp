"""Strict checkpoint-resume verification for the native minimax trainer."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from feedbax.contracts.training import TrainingRunSpec
from feedbax.training.checkpoint_custody import load_latest_checkpoint

from rlrmp.runtime.checkpoint_custody import (
    MINIMAX_ADVERSARIAL_BARRIER,
    MINIMAX_WARMUP_BARRIER,
)
from rlrmp.train.minimax_native.method import (
    build_hps,
    build_minimax_native_initial_slots,
    minimax_training_run_spec_to_config,
)
from rlrmp.train.resume_control import emit_launch_continuation, resolve_launch_continuation
from rlrmp.train.training_configs import MinimaxConfig


logger = logging.getLogger(__name__)


def verify_minimax_checkpoint_resume(
    spec: TrainingRunSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Load and strictly validate a minimax checkpoint without training."""

    import jax.random as jr

    training_spec = (
        spec if isinstance(spec, TrainingRunSpec) else TrainingRunSpec.model_validate(spec)
    )
    config = MinimaxConfig.model_validate(minimax_training_run_spec_to_config(training_spec))
    checkpoint_root = Path(config.output_dir) / "checkpoints_adversarial"
    continuation = resolve_launch_continuation(
        checkpoint_root=checkpoint_root,
        resume_requested=True,
        allow_fresh_start=False,
        stop_target_batches=config.n_warmup_batches + config.n_adversary_batches,
        completed_batches_from_latest=lambda path: _completed_minimax_batches(path, config),
    )
    emit_launch_continuation(continuation, logger=logger)
    initial_slots, _runtime = build_minimax_native_initial_slots(
        run_spec=training_spec,
        hps=build_hps(config),
        args=config,
        key=jr.PRNGKey(config.seed),
    )
    loaded = load_latest_checkpoint(
        checkpoint_root,
        expected_run_spec=training_spec,
        expected_phase_program=(
            training_spec.worker_execution.method_contract.phase_program
        ),
        expected_slots=initial_slots,
    )
    return {
        "verified_resume": True,
        "checkpoint_root": str(checkpoint_root),
        "transaction_id": loaded.manifest.transaction_id,
        "completed_batches": continuation.completed_batches,
        "continuation_batches": continuation.continuation_batches,
    }


def _completed_minimax_batches(path: Path, config: MinimaxConfig) -> int:
    """Translate the latest minimax custody coordinate into completed batches."""

    coordinate = json.loads(path.read_text(encoding="utf-8")).get(
        "completed_coordinate", {}
    )
    total = config.n_warmup_batches + config.n_adversary_batches
    if coordinate.get("phase") == "done":
        return total
    if coordinate.get("completed_barrier") == MINIMAX_WARMUP_BARRIER:
        return config.n_warmup_batches
    if coordinate.get("completed_barrier") == MINIMAX_ADVERSARIAL_BARRIER:
        return min(total, config.n_warmup_batches + int(coordinate.get("global_step", 0)))
    raise ValueError(f"unsupported minimax checkpoint coordinate in {path}: {coordinate!r}")
