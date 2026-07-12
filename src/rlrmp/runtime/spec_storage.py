"""RLRMP entry points for Feedbax three-layer training-spec storage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.spec_storage import (
    build_resolved_semantics_snapshot,
    store_canonical_json_artifact,
)
from feedbax.training.spec_storage import (
    TrainingSpecStorageResult,
    emit_training_run_spec_storage,
)

from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods


def emit_rlrmp_training_run_spec_storage(
    authored: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
    authored_path: Path,
    custody_root: Path,
    materializer_commit: str,
    dependency_lock_path: Path,
    input_data_identities: list[dict[str, Any]] | None = None,
    environment_digest: str | None = None,
) -> TrainingSpecStorageResult:
    """Emit an RLRMP matrix as authored intent plus immutable custody records.

    RLRMP training methods are registered before Feedbax resolves the matrix, so
    project-specific method payloads receive the same validation used by the
    checkpoint-fork launch path.
    """

    register_rlrmp_training_methods()
    return emit_training_run_spec_storage(
        authored,
        repo_root=repo_root,
        authored_path=authored_path,
        custody_root=custody_root,
        materializer_commit=materializer_commit,
        dependency_lock_path=dependency_lock_path,
        input_data_identities=input_data_identities,
        environment_digest=environment_digest,
    )


def migrate_inline_training_run_matrix(
    authored: Mapping[str, Any],
    *,
    repo_root: Path,
    authored_path: Path,
    custody_root: Path,
    materializer_commit: str,
    dependency_lock_path: Path,
) -> TrainingSpecStorageResult:
    """Preserve an inline base exactly, then replace it with its custody ref.

    The snapshot is stored before the authored file is rewritten. This ordering
    is intentional: a failed emission cannot remove the only copy of historical
    resolved semantics.
    """

    legacy_document = dict(authored)
    base = legacy_document.get("base")
    if not isinstance(base, Mapping) or set(base) != {"inline"}:
        raise ValueError("migration requires one legacy /base/inline payload")
    inline = base["inline"]
    if not isinstance(inline, Mapping):
        raise ValueError("legacy /base/inline must be an object")
    base_snapshot = build_resolved_semantics_snapshot(inline)
    base_artifact = store_canonical_json_artifact(
        base_snapshot,
        root=custody_root,
        role="training_run_resolved_base",
        logical_name=f"{authored_path.stem}.historical-base.resolved.json",
    )
    base_path = custody_root / str(base_artifact.metadata["relative_path"])
    document = default_spec_registry.migrate("TrainingRunMatrixSpec", legacy_document).payload
    document["base"] = {
        "kind": "resolved_output",
        "ref": str(base_path.relative_to(repo_root)),
        "resolved_root_hash": base_snapshot["root_hash"],
        "symbolic_name": f"{authored_path.stem}.historical-base",
    }
    input_identities = inline.get("consumed_data_identities", [])
    if not isinstance(input_identities, list):
        raise ValueError("legacy consumed_data_identities must be a list")
    return emit_rlrmp_training_run_spec_storage(
        document,
        repo_root=repo_root,
        authored_path=authored_path,
        custody_root=custody_root,
        materializer_commit=materializer_commit,
        dependency_lock_path=dependency_lock_path,
        input_data_identities=input_identities,
    )
