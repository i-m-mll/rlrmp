"""Shared routing for compact RLRMP matrix authoring intent."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.training.manifest_preflight import validate_training_run_spec
from feedbax.training.run_matrix import (
    MaterializedRunMatrix,
    TrainingRowLowerer,
    materialize_adapted_run_matrix,
    materialize_run_matrix,
    resolve_base_payload_with_attribution,
)

from rlrmp.train.matrix_lowering import (
    is_rlrmp_training_authoring_intent,
    lower_rlrmp_training_row,
)


def training_matrix_spec(
    value: TrainingRunMatrixSpec | Mapping[str, Any],
) -> TrainingRunMatrixSpec:
    """Return the current strict matrix contract for a typed or authored value."""

    if isinstance(value, TrainingRunMatrixSpec):
        return value
    migrated = default_spec_registry.migrate("TrainingRunMatrixSpec", value)
    return TrainingRunMatrixSpec.model_validate(migrated.payload)


def rlrmp_training_row_lowerer(
    matrix: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
) -> TrainingRowLowerer | None:
    """Select compact lowering without changing legacy resolved-row identity."""

    resolved, _attribution = resolve_base_payload_with_attribution(
        training_matrix_spec(matrix),
        repo_root=repo_root,
    )
    if is_rlrmp_training_authoring_intent(resolved):
        return lower_rlrmp_training_row
    return None


def materialize_rlrmp_training_matrix(
    matrix: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
) -> MaterializedRunMatrix:
    """Materialize compact intent through lowering and preserve legacy matrices."""

    spec = training_matrix_spec(matrix)
    lowerer = rlrmp_training_row_lowerer(spec, repo_root=repo_root)
    if lowerer is None:
        return materialize_run_matrix(spec, repo_root=repo_root)
    return materialize_adapted_run_matrix(
        spec,
        repo_root=repo_root,
        row_validator=validate_rlrmp_training_payload,
        row_lowerer=lowerer,
    )


def validate_rlrmp_training_payload(payload: dict[str, Any], _row_id: str) -> Any:
    """Validate one lowered execution payload without changing its bytes."""

    return validate_training_run_spec(payload)


__all__ = [
    "materialize_rlrmp_training_matrix",
    "rlrmp_training_row_lowerer",
    "training_matrix_spec",
    "validate_rlrmp_training_payload",
]
