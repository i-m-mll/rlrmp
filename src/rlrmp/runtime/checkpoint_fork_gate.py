"""RLRMP adapter for Feedbax training-run matrix checkpoint forks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
)
from feedbax.training.run_matrix import (
    fork_matrix_checkpoints,
    materialize_run_matrix,
)

from rlrmp.runtime.lr_continuation import RlrmpLrContinuationReporter
from rlrmp.runtime.training_run_specs import (
    register_rlrmp_cs_supervised_method,
    register_rlrmp_distillation_methods,
)
from rlrmp.train.minimax_native import (
    ensure_minimax_training_method_registered,
)


@dataclass(frozen=True)
class ForkTarget:
    """One target row checkpoint root for a matrix fork."""

    row_id: str
    checkpoint_root: Path


def register_rlrmp_training_methods() -> None:
    """Register RLRMP Feedbax training methods in this process."""

    ensure_minimax_training_method_registered()
    register_rlrmp_cs_supervised_method()
    register_rlrmp_distillation_methods()


def parse_target(value: str) -> ForkTarget:
    """Parse ``ROW=CHECKPOINT_ROOT`` into a fork target."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("target must be ROW=CHECKPOINT_ROOT")
    row_id, checkpoint_root = value.split("=", 1)
    if not row_id:
        raise argparse.ArgumentTypeError("target row id must not be empty")
    if not checkpoint_root:
        raise argparse.ArgumentTypeError("target checkpoint root must not be empty")
    return ForkTarget(row_id=row_id, checkpoint_root=Path(checkpoint_root))


def load_matrix(path: Path) -> TrainingRunMatrixSpec:
    """Load and validate one ``TrainingRunMatrixSpec`` document."""

    return TrainingRunMatrixSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))


def fork_checkpoints_with_parity(
    *,
    matrix_path: Path,
    source_checkpoint_root: Path,
    targets: Sequence[ForkTarget],
    parity_output_path: Path,
    repo_root: Path | None = None,
    skip_fork: bool = False,
) -> dict[str, Any]:
    """Materialize a matrix, fork row checkpoints, and write Feedbax parity JSON."""

    if not targets:
        raise ValueError("at least one fork target is required")
    register_rlrmp_training_methods()
    matrix = load_matrix(matrix_path)
    materialized = materialize_run_matrix(
        matrix,
        repo_root=Path.cwd() if repo_root is None else repo_root,
        method_registry=DEFAULT_TRAINING_METHOD_REGISTRY,
    )
    target_roots = {target.row_id: target.checkpoint_root for target in targets}
    reporter = RlrmpLrContinuationReporter(source_checkpoint_root=source_checkpoint_root)
    return fork_matrix_checkpoints(
        matrix,
        materialized,
        source_checkpoint_root=source_checkpoint_root,
        target_checkpoint_roots=target_roots,
        parity_output_path=parity_output_path,
        skip_fork=skip_fork,
        lr_reporter=reporter,
        tool_version="rlrmp.checkpoint_fork_gate.v2",
    )
