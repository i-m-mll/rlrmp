"""RLRMP execution binding for Feedbax training-matrix assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from feedbax.contracts.run_matrix import TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID
from feedbax.orchestration import CompiledRunSet, RowLaunchSpec
from feedbax.training.diagnostics import (
    NativeTrainingDiagnosticsInput,
    ScheduleContextDiagnostic,
)
from feedbax.training.spec_storage import (
    TrainingRunIdentityAdapter,
    compile_training_run_matrix,
)

from rlrmp.train.matrix_materialization import (
    rlrmp_training_row_lowerer,
    validate_rlrmp_training_payload,
)


COMPILER_ID = "rlrmp.orchestrated_launch"
COMPILER_VERSION = "v1"


class RlrmpOrchestratedLaunchCompiler:
    """Lower matrix semantics with Feedbax, then bind the RLRMP row executor."""

    def compile(
        self,
        request: Any,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: Any,
    ) -> CompiledRunSet:
        lowerer = rlrmp_training_row_lowerer(authored, repo_root=context.repo_root)
        compiled = compile_training_run_matrix(
            authored,
            run_set_id=run_set_id,
            context=context,
            allow_inline_base=False,
            row_validator=(validate_rlrmp_training_payload if lowerer is not None else None),
            row_lowerer=lowerer,
        )
        selection = request.metadata.get("row_selection", {})
        selected_ids = (
            set(selection.get("row_ids", [])) if isinstance(selection, Mapping) else set()
        )
        available = {item.row_id for item in compiled.rows}
        unknown = selected_ids - available
        if unknown:
            raise ValueError(f"unknown row ids in assembly selection: {sorted(unknown)}")
        rows = []
        for row in compiled.rows:
            if selected_ids and row.row_id not in selected_ids:
                continue
            rows.append(
                row.model_copy(
                    update={
                        "launch": RowLaunchSpec(
                            command=[
                                "uv",
                                "run",
                                "--no-sync",
                                "python",
                                "-m",
                                "rlrmp.train.orchestrated_row",
                                "--packet",
                                "{packet_path}",
                            ],
                            collect=[
                                "manifest.json",
                                "training-diagnostics.json",
                                "training_summary.json",
                            ],
                            payload_routing={"kind": "rlrmp-row-launch-packet-v1"},
                            metadata={
                                "native_training_diagnostics": _native_training_diagnostics(
                                    authored,
                                    row.payload,
                                    run_set_id=run_set_id,
                                    row_id=row.row_id,
                                    seed=row.provenance.seed
                                    if row.provenance is not None
                                    else None,
                                ).model_dump(mode="json", exclude_none=True)
                            },
                        ),
                    }
                )
            )
        return CompiledRunSet(rows=rows)


def register_orchestrated_training_compiler(registry: Any) -> None:
    """Register the exact RLRMP compiler dispatch triple."""
    registry.register(
        schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        compiler_id=COMPILER_ID,
        compiler_version=COMPILER_VERSION,
        compiler=RlrmpOrchestratedLaunchCompiler(),
        identity_adapter=TrainingRunIdentityAdapter(),
    )


def _native_training_diagnostics(
    authored: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    run_set_id: str,
    row_id: str,
    seed: int | None,
) -> NativeTrainingDiagnosticsInput:
    """Build typed diagnostic inputs without mutating scientific payloads."""

    continuation_context = _continuation_context(authored, payload)
    resume_context = (
        ScheduleContextDiagnostic.model_validate(continuation_context)
        if continuation_context is not None
        else None
    )
    optimizer_context = (
        ScheduleContextDiagnostic.model_validate(
            _optimizer_build_context(payload, continuation_context)
        )
        if continuation_context is not None
        else None
    )
    return NativeTrainingDiagnosticsInput(
        seeds=[] if seed is None else [seed],
        resume_context=resume_context,
        optimizer_build_context=optimizer_context,
        metadata={"run_set_id": run_set_id, "row_id": row_id},
    )


def _continuation_context(
    authored: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, int] | None:
    fork = authored.get("fork")
    metadata = authored.get("metadata")
    if not isinstance(fork, Mapping) or not isinstance(metadata, Mapping):
        return None
    completed = metadata.get("source_completed_training_batches")
    if not isinstance(completed, int):
        continuation = _path(payload, "checkpoint_progress", "continuation")
        completed = (
            continuation.get("source_completed_batches")
            if isinstance(continuation, Mapping)
            else None
        )
    if not isinstance(completed, int):
        return None
    origin = completed if fork.get("lr_continuation") == "restart" else 0
    return {
        "schedule_origin_step": origin,
        "current_step": completed,
        "optimizer_count_at_current_step": 0,
    }


def _optimizer_build_context(
    payload: Mapping[str, Any], resume_context: Mapping[str, int]
) -> dict[str, int]:
    """Derive context through the executor's public optimizer builder."""
    optimizer = _path(payload, "method_payload", "payload", "controller_optimizer")
    if not isinstance(optimizer, Mapping):
        optimizer = _path(payload, "method_payload", "payload", "optimizer")
    if isinstance(optimizer, Mapping):
        from feedbax.contracts.training import OptimizerSpec
        from feedbax.training.optimizers import build_optimizer

        build_optimizer(OptimizerSpec.model_validate(optimizer), **resume_context)
    return dict(resume_context)


def _path(value: Mapping[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


__all__ = [
    "COMPILER_ID",
    "COMPILER_VERSION",
    "RlrmpOrchestratedLaunchCompiler",
    "register_orchestrated_training_compiler",
]
