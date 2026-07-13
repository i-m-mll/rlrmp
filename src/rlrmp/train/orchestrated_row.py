"""Execute one ASSEMBLE-bound RLRMP training row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from feedbax.contracts.training import TrainingRunSpec
from feedbax.orchestration import ExecutionIdentityEnvelope
from feedbax.orchestration.events import RunEventEmitter
from feedbax.training.interruption import CancellationDecision
from feedbax.training.diagnostics import (
    NativeExecutionProducerContext,
    NativeTrainingDiagnosticsInput,
)

from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.initial_slots import RlrmpRuntime


class RowLaunchPacket(BaseModel):
    """Canonical, transport-safe inputs for one row process."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "rlrmp.orchestrated_row_packet.v2"
    run_set_id: str
    row_id: str
    envelope: ExecutionIdentityEnvelope
    payload: dict[str, Any]
    row_dir: str
    staged_checkpoint_root: str | None = None
    fork_record_path: str | None = None
    fork_record_sha256: str | None = None
    resume: bool = False
    stop_after_batches: int | None = None
    native_training_diagnostics: NativeTrainingDiagnosticsInput = Field(
        default_factory=NativeTrainingDiagnosticsInput
    )


def load_packet(path: Path) -> RowLaunchPacket:
    """Load and verify packet payload bytes against its execution envelope."""
    packet = RowLaunchPacket.model_validate_json(path.read_text(encoding="utf-8"))
    payload_bytes = _canonical_bytes(packet.payload)
    if hashlib.sha256(payload_bytes).hexdigest() != packet.envelope.payload.sha256:
        raise ValueError("row packet payload digest does not match execution envelope")
    return packet


def execute_packet(packet: RowLaunchPacket) -> Path:
    """Run the registered executor and project its outputs for CERTIFY/COLLECT."""
    from feedbax.training import (
        DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY,
        ExecutionPreparationRequest,
    )
    from feedbax.training.executor import execute_training_run_spec

    from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods
    from rlrmp.train.execution_preparation import register_execution_preparations

    register_rlrmp_training_methods()
    register_execution_preparations(DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY)
    run_spec = TrainingRunSpec.model_validate(packet.payload)
    execution_context = _native_execution_context(packet)
    provenance = execution_context.execution.row_provenance
    assert provenance is not None
    planned_run_id = provenance.planned_run_id
    if packet.resume:
        _verify_staged_checkpoint(packet)
    preparation = DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
        ExecutionPreparationRequest(
            run_spec=run_spec,
            run_id=planned_run_id,
            resume=packet.resume,
        )
    )
    row_dir = Path(packet.row_dir)
    row_dir.mkdir(parents=True, exist_ok=True)
    emitter = RunEventEmitter.from_env()
    result = execute_training_run_spec(
        run_spec,
        run_id=planned_run_id,
        initial_slots=preparation.initial_slots,
        kernel_context=preparation.kernel_context,
        loss_service=preparation.loss_service,
        manifest_root=row_dir / "feedbax-manifests",
        checkpoint_root=packet.staged_checkpoint_root,
        resume=packet.resume,
        resume_slot_transform=preparation.resume_slot_transform,
        run_event_emitter=emitter,
        cancellation_probe=_batch_limit_probe(
            packet.stop_after_batches,
            kernel_context=preparation.kernel_context,
        ),
        execution_context=execution_context,
    )
    _write_json(
        row_dir / "training_summary.json",
        {
            "run_set_id": packet.run_set_id,
            "row_id": packet.row_id,
            "status": result.status,
            "completed_batches": result.diagnostics.completed_batches,
            "metrics": dict(result.manifest.summary_metrics),
        },
    )
    if emitter is not None:
        emitter.close()
    return result.manifest_path


def _native_execution_context(packet: RowLaunchPacket) -> NativeExecutionProducerContext:
    """Bind the assembly envelope to the exact native row producer."""

    if packet.envelope.row_provenance is None:
        raise ValueError("orchestrated RLRMP row requires TrainingRowProvenance")
    return NativeExecutionProducerContext(
        execution=packet.envelope,
        environment_fingerprint=os.environ.get("FEEDBAX_ENV_FINGERPRINT", "unknown"),
        collection_root=packet.row_dir,
        diagnostics=packet.native_training_diagnostics,
    )


def _verify_staged_checkpoint(packet: RowLaunchPacket) -> None:
    if packet.staged_checkpoint_root is None:
        raise ValueError("resume=True requires a staged checkpoint root")
    if len(packet.envelope.immutable_inputs) != 1:
        raise ValueError("resume requires exactly one immutable checkpoint input")
    expected = packet.envelope.immutable_inputs[0]
    if packet.fork_record_path is None or packet.fork_record_sha256 is None:
        raise ValueError("resume packet requires a digest-pinned fork gate record")
    record_path = Path(packet.fork_record_path)
    record_bytes = record_path.read_bytes()
    if hashlib.sha256(record_bytes).hexdigest() != packet.fork_record_sha256:
        raise ValueError("fork gate record digest mismatch")
    record = json.loads(record_bytes)
    source = record.get("source_input", {})
    source_transaction = expected.identifier.rsplit(":", 1)[-1]
    if source.get("transaction_id") != source_transaction:
        raise ValueError("fork target source transaction does not match envelope input")
    if source.get("manifest_sha256") != expected.digest.value:
        raise ValueError("fork target source digest does not match envelope input")
    binding = next(
        (item for item in record.get("targets", []) if item.get("row_id") == packet.row_id),
        None,
    )
    if binding is None:
        raise ValueError(f"fork gate record has no target for row {packet.row_id!r}")
    transaction = str(binding.get("transaction_id", ""))
    manifest = Path(packet.staged_checkpoint_root) / "transactions" / transaction / "manifest.json"
    if not manifest.is_file():
        raise ValueError("staged fork target transaction is missing")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != binding.get("manifest_sha256"):
        raise ValueError("staged fork target manifest digest mismatch")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    lineage = payload.get("segment_lineage") or {}
    parent = lineage.get("parent_transaction_id") or payload.get("metadata", {}).get(
        "forked_from_transaction_id"
    )
    if parent != source_transaction:
        raise ValueError("staged fork target provenance does not chain to envelope source")


def _batch_limit_probe(
    limit: int | None,
    *,
    kernel_context: Mapping[str, Any],
) -> Callable[[Any], CancellationDecision | None] | None:
    if limit is None:
        return None
    runtime = kernel_context.get(RLRMP_RUNTIME_CONTEXT_KEY)
    if not isinstance(runtime, RlrmpRuntime):
        raise RuntimeError(
            "stop-after-batches requires an RLRMP runtime with authoritative "
            "completed-batch progress"
        )
    if runtime.completed_batches_reader is None:
        raise RuntimeError(
            "stop-after-batches requires authoritative completed-batch progress; "
            "the prepared RLRMP runtime does not expose it"
        )

    def probe(_coordinate: Any) -> CancellationDecision | None:
        completed_batches = runtime.read_completed_batches()
        if completed_batches < limit:
            return None
        return CancellationDecision(
            action="stop",
            source="non_interactive",
            requested_at_unix_seconds=time.time(),
        )

    return probe


def _canonical_bytes(value: Any) -> bytes:
    from feedbax.contracts.spec_storage import training_spec_canonical_bytes

    return training_spec_canonical_bytes(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    supplied = parser.parse_args().packet
    if str(supplied) == "{packet_path}":
        row_dir = os.environ.get("FEEDBAX_RUN_ROW_DIR") or os.environ.get("FEEDBAX_ROW_DIR")
        if row_dir is None:
            raise ValueError("packet placeholder requires FEEDBAX_RUN_ROW_DIR")
        supplied = Path(row_dir) / "launch-packet.json"
    packet = load_packet(supplied)
    execute_packet(packet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
