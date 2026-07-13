"""RLRMP execution binding for Feedbax training-matrix assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from feedbax.contracts.run_matrix import TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID
from feedbax.contracts.training import OptimizerSpec
from feedbax.orchestration import CompiledRunSet, RowLaunchSpec
from feedbax.orchestration.schedule_eval import (
    ScheduleEvalContext,
    evaluate_schedule_samples,
    schedule_sample_steps,
)
from feedbax.training.diagnostics import (
    LearningRateDiagnostic,
    NativeTrainingDiagnosticsInput,
    ScheduleContextDiagnostic,
)
from feedbax.training.optimizers import build_optimizer
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

    optimizer_spec = _optimizer_spec(payload)
    continuation_context = _continuation_context(authored, payload)
    schedule_context = (
        continuation_context
        if continuation_context is not None
        else (_fresh_schedule_context() if optimizer_spec is not None else None)
    )
    resume_context = (
        ScheduleContextDiagnostic.model_validate(schedule_context)
        if schedule_context is not None
        else None
    )
    optimizer_context = (
        ScheduleContextDiagnostic.model_validate(
            _optimizer_build_context(optimizer_spec, schedule_context)
            if optimizer_spec is not None
            else schedule_context
        )
        if schedule_context is not None
        else None
    )
    lr_trace = (
        _optimizer_lr_trace(optimizer_spec, schedule_context)
        if optimizer_spec is not None and schedule_context is not None
        else []
    )
    return NativeTrainingDiagnosticsInput(
        seeds=[] if seed is None else [seed],
        lr_trace=lr_trace,
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


def _fresh_schedule_context() -> dict[str, int]:
    """Return the explicit optimizer clock for a fresh training row."""

    return {
        "schedule_origin_step": 0,
        "current_step": 0,
        "optimizer_count_at_current_step": 0,
    }


def _optimizer_spec(payload: Mapping[str, Any]) -> OptimizerSpec | None:
    """Return the governed controller optimizer when the row declares one."""

    optimizer = _path(payload, "method_payload", "payload", "controller_optimizer")
    if not isinstance(optimizer, Mapping):
        optimizer = _path(payload, "method_payload", "payload", "optimizer")
    return OptimizerSpec.model_validate(optimizer) if isinstance(optimizer, Mapping) else None


def _optimizer_build_context(
    optimizer_spec: OptimizerSpec, schedule_context: Mapping[str, int]
) -> dict[str, int]:
    """Derive context through the executor's public optimizer builder."""

    build_optimizer(optimizer_spec, **schedule_context)
    return dict(schedule_context)


def _optimizer_lr_trace(
    optimizer_spec: OptimizerSpec, schedule_context: Mapping[str, int]
) -> list[LearningRateDiagnostic]:
    """Sample the optimizer realized by Feedbax's public construction path."""

    context = ScheduleEvalContext(**schedule_context)
    steps = schedule_sample_steps(optimizer_spec, context)
    samples = evaluate_schedule_samples(optimizer_spec, context, steps)
    return [LearningRateDiagnostic(step=step, learning_rate=samples[step]) for step in steps]


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
