"""Authored training-matrix validation and execution.

This module deliberately keeps document loading independent of runtime training
imports.  In particular, importing it and calling
``load_authored_training_intent`` does not import JAX.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

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


class RowSelection(BaseModel):
    """Operational row subset persisted in assembly-request metadata."""

    model_config = ConfigDict(extra="forbid")
    row_ids: list[str] = Field(default_factory=list)


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
    from rlrmp.train.matrix_materialization import materialize_rlrmp_training_matrix

    materialized = materialize_rlrmp_training_matrix(
        launch.document,
        repo_root=launch.repo_root,
    )
    unresolved = [row.row_id for row in materialized.rows if row.spec is None]
    if unresolved:
        raise ValueError(
            f"matrix compiler did not resolve registered methods for rows: {unresolved}"
        )
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
    driver: str = "local",
    runpod_profile: Path | None = None,
) -> Any:
    """Execute through Feedbax's persisted ASSEMBLE-to-REGISTER lifecycle."""
    from feedbax.orchestration import (
        AuthorizedBatchStop,
        RowConformanceRuntimeInputs,
        StageEngine,
        build_core_check_registry,
    )

    active_controls = controls or LaunchRuntimeControls()
    _validate_execute_controls(launch, active_controls)
    request, context, registry = build_orchestration_request(launch, row=row, driver=driver)
    fork_record = _run_fork_gate(launch, row=row)

    def driver_factory(bundle: Any) -> Any:
        return orchestration_driver_for_bundle(
            bundle,
            driver=driver,
            repo_root=launch.repo_root,
            controls=active_controls,
            fork_record=fork_record,
            runpod_profile=runpod_profile,
        )

    engine = StageEngine.from_request(
        request,
        context=context,
        registry=registry,
        driver_factory=driver_factory,
        conformance_registry=build_core_check_registry(),
        row_conformance_inputs=(
            {
                row_id: RowConformanceRuntimeInputs(
                    authorized_batch_stop=AuthorizedBatchStop(
                        stop_after_batches=active_controls.stop_after_batches
                    )
                )
                for row_id in _selected_authored_row_ids(launch, row=row)
            }
            if active_controls.stop_after_batches is not None
            else {}
        ),
    )
    return engine.run()


def _selected_authored_row_ids(launch: AuthoredLaunch, *, row: str | None) -> tuple[str, ...]:
    """Return row IDs selected by the authored request without recompiling it."""

    available = tuple(item.row_id for item in launch.document.rows)
    if row is None:
        return available
    if row not in available:
        raise ValueError(f"unknown row selector {row!r}; available rows: {', '.join(available)}")
    return (row,)


def orchestration_driver_for_bundle(
    bundle: Any,
    *,
    driver: str,
    repo_root: Path,
    controls: LaunchRuntimeControls,
    fork_record: tuple[Path, str] | None,
    runpod_profile: Path | None,
) -> Any:
    """Construct the selected operational driver from explicit local inputs."""
    from rlrmp.train.orchestration_drivers import (
        RlrmpRunPodDriver,
        local_driver_for_bundle,
    )

    if driver == "local":
        if runpod_profile is not None:
            raise ValueError("local driver does not accept a RunPod operational profile")
        return local_driver_for_bundle(
            bundle,
            resume=controls.resume,
            fork_record_path=(fork_record[0] if fork_record else None),
            fork_record_sha256=(fork_record[1] if fork_record else None),
            stop_after_batches=controls.stop_after_batches,
        )
    if driver == "runpod":
        from rlrmp.train.runpod_profiles import load_runpod_profile

        if runpod_profile is None:
            raise ValueError("runpod driver requires --runpod-profile")
        return RlrmpRunPodDriver(
            config=load_runpod_profile(runpod_profile, repo_root=repo_root),
            resume=controls.resume,
            fork_record_path=(fork_record[0] if fork_record else None),
            fork_record_sha256=(fork_record[1] if fork_record else None),
            stop_after_batches=controls.stop_after_batches,
        )
    raise ValueError(f"unknown orchestration driver {driver!r}")


def launch_evidence(rows: Sequence[LaunchRow], controls: LaunchRuntimeControls) -> dict[str, Any]:
    """Return the operational evidence attached to a planned transitional launch."""

    return {
        "backend": "feedbax_orchestration",
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
) -> tuple[dict[str, Any], ...]:
    """Strictly verify selected rows can resume without running training steps."""

    selected = select_launch_rows(compile_authored_training_intent(launch), row)
    controls = LaunchRuntimeControls(
        resume=True,
        disable_progress=True,
        checkpoint_root=checkpoint_root,
    )
    evidence = []
    for item in selected:
        from feedbax.training.checkpoint_custody import load_latest_checkpoint

        prepared = _prepare_execution(item, resume=True)
        if prepared.initial_slots is None:
            raise ValueError(
                "strict resume verification requires an execution-preparation "
                f"provider for method_ref {item.run_spec.method_ref.key!r}"
            )
        root = controls.checkpoint_root or _checkpoint_root(item.run_spec)
        continuation = item.run_spec.checkpoint_progress.continuation
        loaded = load_latest_checkpoint(
            root,
            expected_run_spec=item.run_spec,
            expected_phase_program=(item.run_spec.worker_execution.method_contract.phase_program),
            expected_slots=prepared.initial_slots,
            resume_slot_transform=prepared.resume_slot_transform,
            continuation_request=continuation,
            allow_new_lineage_override=continuation is not None,
        )
        evidence.append(
            {
                "row_id": item.row_id,
                "checkpoint_root": str(root),
                "status": "valid",
                "transaction_id": loaded.manifest.transaction_id,
            }
        )
    return tuple(evidence)


def prepare_authored_training_rows(
    launch: AuthoredLaunch, *, row: str | None = None, resume: bool = True
) -> tuple[dict[str, Any], ...]:
    """Construct real execution preparations without taking a training step."""
    selected = select_launch_rows(compile_authored_training_intent(launch), row)
    evidence = []
    for item in selected:
        prepared = _prepare_execution(item, resume=resume)
        if prepared.initial_slots is None:
            raise ValueError(f"row {item.row_id!r} has no execution preparation provider")
        evidence.append(
            {
                "row_id": item.row_id,
                "slot_names": sorted(prepared.initial_slots),
                "has_kernel_context": prepared.kernel_context is not None,
                "has_loss_service": prepared.loss_service is not None,
            }
        )
    return tuple(evidence)


def build_orchestration_request(
    launch: AuthoredLaunch,
    *,
    row: str | None,
    driver: str,
) -> tuple[Any, Any, Any]:
    """Build the durable request and RLRMP assembly dependencies."""
    from feedbax.orchestration import (
        AssemblyCompilerRegistry,
        AssemblyContext,
        AssemblyInputDeclaration,
        BudgetPolicy,
        CompilerIdentity,
        LaunchPolicy,
        SchemaArtifactRef,
    )

    from rlrmp.train.orchestration_compiler import (
        COMPILER_ID,
        COMPILER_VERSION,
        register_orchestrated_training_compiler,
    )
    from rlrmp.train.orchestration_inputs import (
        CheckpointTransactionInputResolver,
        checkpoint_transaction_locator,
    )

    if getattr(launch.document.base, "kind", None) == "inline":
        raise ValueError(
            "orchestrated execute requires an emitted authored document; run "
            "scripts/emit_training_run_matrix.py first"
        )
    sidecar = launch.path.with_suffix(launch.path.suffix + ".artifact.json")
    if not sidecar.is_file():
        raise ValueError(
            "authored matrix has no SchemaArtifactRef; run "
            "scripts/emit_training_run_matrix.py first (execute never emits specs)"
        )
    authored_ref = SchemaArtifactRef.model_validate_json(sidecar.read_text(encoding="utf-8"))
    metadata = launch.document.metadata
    source_root = metadata.get("source_checkpoint_root")
    transaction_id = metadata.get("source_checkpoint_transaction_id")
    inputs: list[Any] = []
    if launch.document.fork is None:
        if "source_checkpoint_root" in metadata or "source_checkpoint_transaction_id" in metadata:
            raise ValueError("fresh execute cannot declare source checkpoint metadata")
    else:
        if not isinstance(source_root, str) or not isinstance(transaction_id, str):
            raise ValueError("execute requires one common source checkpoint transaction")
        root = (launch.repo_root / source_root).resolve()
        manifest = root / "transactions" / transaction_id / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"source checkpoint transaction is missing: {transaction_id}")
        manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        inputs.append(
            AssemblyInputDeclaration(
                role="source_checkpoint",
                kind="checkpoint_transaction",
                locator=checkpoint_transaction_locator(
                    root,
                    transaction_id=transaction_id,
                    manifest_sha256=manifest_digest,
                ),
            )
        )
    selection = RowSelection(row_ids=[] if row is None else [row])
    available = {item.row_id for item in launch.document.rows}
    unknown = set(selection.row_ids) - available
    if unknown:
        raise ValueError(f"unknown row ids in selection: {sorted(unknown)}")
    request = _build_assembly_request(
        authored=authored_ref,
        compiler=CompilerIdentity(compiler_id=COMPILER_ID, compiler_version=COMPILER_VERSION),
        inputs=inputs,
        driver=driver,
        repo_root=launch.repo_root,
        launch_policy=LaunchPolicy(
            max_parallel_rows=max(1, len(selection.row_ids) or len(available)),
            warm_first=True,
        ),
        budget=BudgetPolicy(max_wall_clock_seconds=7 * 24 * 3600),
        orchestration_root=str(launch.repo_root / "_artifacts" / "orchestration"),
        metadata={"row_selection": selection.model_dump(mode="json")},
    )
    registry = AssemblyCompilerRegistry()
    register_orchestrated_training_compiler(registry)
    context = AssemblyContext(
        custody_root=launch.repo_root / "_artifacts" / "orchestration-custody",
        repo_root=launch.repo_root,
        input_resolver=CheckpointTransactionInputResolver(),
        authored_ref=authored_ref,
    )
    return request, context, registry


def _build_assembly_request(
    *,
    authored: Any,
    compiler: Any,
    inputs: list[Any],
    driver: str,
    repo_root: Path,
    launch_policy: Any,
    budget: Any,
    orchestration_root: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Build a production request with an honest environment declaration."""
    from feedbax.orchestration import EnvironmentDeclaration, RunAssemblyRequest

    lockfile = repo_root / "uv.lock"
    lockfile_hashes = (
        {"uv.lock": hashlib.sha256(lockfile.read_bytes()).hexdigest()} if lockfile.is_file() else {}
    )
    return RunAssemblyRequest(
        authored=authored,
        compiler=compiler,
        inputs=inputs,
        driver=driver,
        environment=EnvironmentDeclaration(
            python_version=sys.version.split()[0],
            lockfile_hashes=lockfile_hashes,
        ),
        launch_policy=launch_policy,
        budget=budget,
        orchestration_root=orchestration_root,
        metadata=dict(metadata or {}),
    )


def _validate_execute_controls(launch: AuthoredLaunch, controls: LaunchRuntimeControls) -> None:
    continuation = launch.document.fork is not None
    if controls.resume != continuation:
        raise ValueError("--resume must match the authored continuation envelope")
    source = launch.document.metadata.get("source_checkpoint_root")
    if controls.checkpoint_root is not None and isinstance(source, str):
        expected = (launch.repo_root / source).resolve()
        if controls.checkpoint_root.resolve() != expected:
            raise ValueError("--checkpoint-root must match the authored immutable input")
    if controls.allow_fresh_start:
        raise ValueError("--allow-fresh-start cannot override an immutable continuation")


def _run_fork_gate(launch: AuthoredLaunch, *, row: str | None) -> tuple[Path, str] | None:
    """Fork and verify derived row targets before ASSEMBLE."""
    if launch.document.fork is None:
        return None
    from rlrmp.runtime.checkpoint_fork_gate import ForkTarget, fork_checkpoints_with_parity

    selected = select_launch_rows(compile_authored_training_intent(launch), row)
    source = launch.document.metadata.get("source_checkpoint_root")
    if not isinstance(source, str):
        raise ValueError("forked execute requires metadata.source_checkpoint_root")
    targets = [ForkTarget(item.row_id, _checkpoint_root(item.run_spec)) for item in selected]
    output = launch.repo_root / "_artifacts" / "orchestration" / "fork-parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    table = fork_checkpoints_with_parity(
        matrix_path=launch.path,
        source_checkpoint_root=(launch.repo_root / source).resolve(),
        targets=targets,
        parity_output_path=output,
        repo_root=launch.repo_root,
    )
    source_transaction = launch.document.metadata.get("source_checkpoint_transaction_id")
    source_manifest = (
        (launch.repo_root / source).resolve()
        / "transactions"
        / str(source_transaction)
        / "manifest.json"
    )
    targets_payload = []
    transaction_by_row = {
        str(item["row_id"]): str(item["transaction_id"])
        for item in table.get("rows", [])
        if item.get("transaction_id")
    }
    for target in targets:
        transaction_id = transaction_by_row.get(target.row_id)
        if transaction_id is None:
            raise ValueError(f"fork gate emitted no target transaction for row {target.row_id!r}")
        target_manifest = target.checkpoint_root / "transactions" / transaction_id / "manifest.json"
        targets_payload.append(
            {
                "row_id": target.row_id,
                "checkpoint_root": str(target.checkpoint_root.resolve()),
                "transaction_id": transaction_id,
                "manifest_sha256": hashlib.sha256(target_manifest.read_bytes()).hexdigest(),
            }
        )
    binding_record = {
        "schema_version": "rlrmp.fork_gate_binding.v1",
        "parity": table,
        "source_input": {
            "transaction_id": source_transaction,
            "manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        },
        "targets": targets_payload,
    }
    encoded = json.dumps(binding_record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    output.write_bytes(encoded)
    return output, hashlib.sha256(encoded).hexdigest()


def _prepare_execution(row: LaunchRow, *, resume: bool) -> _PreparedExecution:
    _enforce_x64_precondition(row.run_spec)
    from feedbax.training import (
        DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY,
        ExecutionPreparationRequest,
    )

    if DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.get(row.run_spec.method_ref.key) is None:
        return _PreparedExecution(None, None, None, None)
    result = DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
        ExecutionPreparationRequest(run_spec=row.run_spec, run_id=row.planned_run_id, resume=resume)
    )
    return _PreparedExecution(
        result.initial_slots,
        result.kernel_context,
        result.loss_service,
        result.resume_slot_transform,
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
    "LaunchRow",
    "LaunchRuntimeControls",
    "RowSelection",
    "accepted_authored_document",
    "build_orchestration_request",
    "compile_authored_training_intent",
    "execute_authored_training_intent",
    "launch_evidence",
    "load_authored_training_intent",
    "orchestration_driver_for_bundle",
    "prepare_authored_training_rows",
    "select_launch_rows",
    "verify_resume_authored_training_intent",
]
