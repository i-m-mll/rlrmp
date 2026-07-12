"""Authored training-matrix validation and execution.

This module deliberately keeps document loading independent of runtime training
imports.  In particular, importing it and calling
``load_authored_training_intent`` does not import JAX.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from feedbax.contracts.run_matrix import TrainingRunMatrixSpec


@dataclass(frozen=True)
class LaunchRuntimeControls:
    """Operational controls which never alter authored scientific semantics."""

    resume: bool = False
    allow_fresh_start: bool = False
    stop_after_batches: int | None = None
    disable_progress: bool = False
    quiet_progress: bool = False
    log_step: int = 1
    manifest_root: Path | None = None
    checkpoint_root: Path | None = None

    def __post_init__(self) -> None:
        if self.stop_after_batches is not None and self.stop_after_batches < 1:
            raise ValueError("stop_after_batches must be positive")
        if self.log_step < 1:
            raise ValueError("log_step must be positive")
        if self.allow_fresh_start and not self.resume:
            raise ValueError("allow_fresh_start requires resume")


@dataclass(frozen=True)
class AuthoredLaunch:
    """Validated authored document and the root used to resolve its references."""

    document: TrainingRunMatrixSpec
    path: Path
    repo_root: Path


@dataclass(frozen=True)
class LaunchRow:
    """One compiled row handed to an execution backend."""

    row_id: str
    planned_run_id: str
    run_spec: Any


@dataclass(frozen=True)
class _PreparedExecution:
    """Runtime-only values shared by execution and strict resume verification."""

    initial_slots: Any
    kernel_context: Any
    loss_service: Any
    resume_slot_transform: Any


class LaunchBackend(Protocol):
    """Backend boundary replaced by ASSEMBLE in issue 158b580."""

    def execute(self, row: LaunchRow, controls: LaunchRuntimeControls) -> Any:
        """Execute or submit one already-compiled row."""


def load_authored_training_intent(
    path: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> AuthoredLaunch:
    """Load and strictly validate an authored ``TrainingRunMatrixSpec``.

    This accepts only the governed matrix document.  Flat family configs,
    nested ``TrainingRunSpec`` objects, and historical outer recipes therefore
    fail strict matrix validation rather than being guessed from their shape.
    """

    resolved_path = Path(path).resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    # Feedbax currently initializes its public package (including JAX) before
    # contract submodules. Keep this import at the validation call boundary so
    # importing the frontend remains lightweight; the feedbax lane will make
    # the contract import itself lightweight before integration.
    from feedbax.contracts.migrations import migrate_structured_spec_payload
    from feedbax.contracts.run_matrix import TrainingRunMatrixSpec

    migrated = migrate_structured_spec_payload("TrainingRunMatrixSpec", payload)
    document = TrainingRunMatrixSpec.model_validate(migrated.payload)
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    return AuthoredLaunch(document=document, path=resolved_path, repo_root=root)


def accepted_authored_document(
    path: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> bool:
    """Return whether the frontend accepts a document, independent of backend."""

    try:
        load_authored_training_intent(path, repo_root=repo_root)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def compile_authored_training_intent(launch: AuthoredLaunch) -> tuple[LaunchRow, ...]:
    """Compile an authored matrix through Feedbax's registered compiler."""

    _register_runtime()
    from feedbax.training.run_matrix import materialize_run_matrix

    materialized = materialize_run_matrix(launch.document, repo_root=launch.repo_root)
    unresolved = [row.row_id for row in materialized.rows if row.spec is None]
    if unresolved:
        raise ValueError(f"matrix compiler did not resolve registered methods for rows: {unresolved}")
    return tuple(
        LaunchRow(
            row_id=row.row_id,
            planned_run_id=row.planned_run_id,
            run_spec=row.spec,
        )
        for row in materialized.rows
    )


def select_launch_rows(rows: Sequence[LaunchRow], selector: str | None) -> tuple[LaunchRow, ...]:
    """Select one named row, or all rows when no selector is supplied."""

    if selector is None:
        return tuple(rows)
    selected = tuple(row for row in rows if row.row_id == selector)
    if not selected:
        available = ", ".join(row.row_id for row in rows)
        raise ValueError(f"unknown row selector {selector!r}; available rows: {available}")
    return selected


def execute_authored_training_intent(
    launch: AuthoredLaunch,
    *,
    row: str | None = None,
    controls: LaunchRuntimeControls | None = None,
    backend: LaunchBackend | None = None,
) -> tuple[Any, ...]:
    """Compile and execute selected rows through the transitional boundary."""

    compiled = compile_authored_training_intent(launch)
    selected = select_launch_rows(compiled, row)
    active_backend = backend or TransitionalFeedbaxBackend()
    active_controls = controls or LaunchRuntimeControls()
    return tuple(active_backend.execute(item, active_controls) for item in selected)


def launch_evidence(
    rows: Sequence[LaunchRow], controls: LaunchRuntimeControls
) -> dict[str, Any]:
    """Return the operational evidence attached to a planned transitional launch."""

    return {
        "backend": "transitional_feedbax_direct",
        "rows": [{"row_id": item.row_id, "run_id": item.planned_run_id} for item in rows],
        "runtime_controls": {
            "resume": controls.resume,
            "allow_fresh_start": controls.allow_fresh_start,
            "resume_policy": (
                "resume_if_checkpoint_exists_else_fresh"
                if controls.resume and controls.allow_fresh_start
                else ("require_checkpoint" if controls.resume else "fresh")
            ),
            "stop_after_batches": controls.stop_after_batches,
            "disable_progress": controls.disable_progress,
            "quiet_progress": controls.quiet_progress,
            "log_step": controls.log_step,
            "manifest_root": (
                None if controls.manifest_root is None else str(controls.manifest_root)
            ),
            "checkpoint_root": (
                None if controls.checkpoint_root is None else str(controls.checkpoint_root)
            ),
        },
    }


def verify_resume_authored_training_intent(
    launch: AuthoredLaunch,
    *,
    row: str | None = None,
    checkpoint_root: Path | None = None,
    backend: TransitionalFeedbaxBackend | None = None,
) -> tuple[dict[str, Any], ...]:
    """Strictly verify selected rows can resume without running training steps."""

    selected = select_launch_rows(compile_authored_training_intent(launch), row)
    active_backend = backend or TransitionalFeedbaxBackend()
    controls = LaunchRuntimeControls(
        resume=True,
        disable_progress=True,
        checkpoint_root=checkpoint_root,
    )
    return tuple(active_backend.verify_resume(item, controls) for item in selected)


class TransitionalFeedbaxBackend:
    """Temporary direct Feedbax executor backend, deleted at ASSEMBLE cutover."""

    def execute(self, row: LaunchRow, controls: LaunchRuntimeControls) -> Any:
        """Execute one compiled row without minting a second run-set identity."""

        from feedbax.training.executor import execute_training_run_spec

        resume = controls.resume
        if resume and controls.allow_fresh_start:
            root = controls.checkpoint_root or _checkpoint_root(row.run_spec)
            resume = (root / "latest.json").is_file()
        prepared = self._prepare(row, resume=resume)

        progress_callback = _progress_callback(controls)
        cancellation_probe = _batch_limit_probe(controls.stop_after_batches)
        return execute_training_run_spec(
            row.run_spec,
            run_id=row.planned_run_id,
            initial_slots=prepared.initial_slots,
            kernel_context=prepared.kernel_context,
            loss_service=prepared.loss_service,
            manifest_root=controls.manifest_root,
            checkpoint_root=controls.checkpoint_root,
            resume=resume,
            resume_slot_transform=prepared.resume_slot_transform,
            progress_callback=progress_callback,
            cancellation_probe=cancellation_probe,
        )

    def verify_resume(
        self,
        row: LaunchRow,
        controls: LaunchRuntimeControls,
    ) -> dict[str, Any]:
        """Prepare executor inputs and strictly load the configured checkpoint."""

        from feedbax.training.checkpoint_custody import load_latest_checkpoint

        prepared = self._prepare(row, resume=True)
        if prepared.initial_slots is None:
            raise ValueError(
                "strict resume verification requires an execution-preparation "
                f"provider for method_ref {row.run_spec.method_ref.key!r}"
            )
        root = controls.checkpoint_root or _checkpoint_root(row.run_spec)
        continuation = row.run_spec.checkpoint_progress.continuation
        loaded = load_latest_checkpoint(
            root,
            expected_run_spec=row.run_spec,
            expected_phase_program=(
                row.run_spec.worker_execution.method_contract.phase_program
            ),
            expected_slots=prepared.initial_slots,
            resume_slot_transform=prepared.resume_slot_transform,
            continuation_request=continuation,
            allow_new_lineage_override=continuation is not None,
        )
        return {
            "row_id": row.row_id,
            "checkpoint_root": str(root),
            "status": "valid",
            "transaction_id": loaded.manifest.transaction_id,
        }

    def _prepare(self, row: LaunchRow, *, resume: bool) -> _PreparedExecution:
        """Build the method-owned runtime context used by execute and verify."""

        _enforce_x64_precondition(row.run_spec)
        from feedbax.training import (
            DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY,
            ExecutionPreparationRequest,
        )

        if (
            DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.get(
                row.run_spec.method_ref.key
            )
            is None
        ):
            return _PreparedExecution(
                initial_slots=None,
                kernel_context=None,
                loss_service=None,
                resume_slot_transform=None,
            )
        result = DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
            ExecutionPreparationRequest(
                run_spec=row.run_spec,
                run_id=row.planned_run_id,
                resume=resume,
            )
        )
        return _PreparedExecution(
            initial_slots=result.initial_slots,
            kernel_context=result.kernel_context,
            loss_service=result.loss_service,
            resume_slot_transform=result.resume_slot_transform,
        )


def _register_runtime() -> None:
    from feedbax.training import DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY

    from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods
    from rlrmp.train.execution_preparation import register_execution_preparations

    register_rlrmp_training_methods()
    register_execution_preparations(DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY)


def _enforce_x64_precondition(run_spec: Any) -> None:
    """Fail closed when execution inherited an x64-enabled JAX process."""

    import jax

    if bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("training execution requires JAX x64 to be disabled")
    payload = getattr(run_spec.method_payload, "payload", run_spec.method_payload)
    config = getattr(payload, "config", None)
    allow_x64 = getattr(config, "allow_x64", None)
    if allow_x64 is False and bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("authored launch requires x64-disabled execution")


def _progress_callback(controls: LaunchRuntimeControls) -> Callable[[Any], None] | None:
    if controls.disable_progress:
        return None

    count = 0

    def report(coordinate: Any) -> None:
        nonlocal count
        count += 1
        if controls.quiet_progress or count % controls.log_step:
            return
        print(f"BATCH coordinate={coordinate}")

    return report


def _batch_limit_probe(limit: int | None) -> Callable[[Any], str | None] | None:
    if limit is None:
        return None
    seen = 0

    def probe(_coordinate: Any) -> str | None:
        nonlocal seen
        seen += 1
        return "stop" if seen >= limit else None

    return probe


def _checkpoint_root(run_spec: Any) -> Path:
    metadata = run_spec.checkpoint_progress.metadata
    configured = metadata.get("checkpoint_dir")
    if isinstance(configured, str) and configured:
        return Path(configured)
    return Path(run_spec.artifacts.artifact_root) / "checkpoints"


__all__ = [
    "AuthoredLaunch",
    "LaunchBackend",
    "LaunchRow",
    "LaunchRuntimeControls",
    "TransitionalFeedbaxBackend",
    "accepted_authored_document",
    "compile_authored_training_intent",
    "execute_authored_training_intent",
    "launch_evidence",
    "load_authored_training_intent",
    "select_launch_rows",
    "verify_resume_authored_training_intent",
]
