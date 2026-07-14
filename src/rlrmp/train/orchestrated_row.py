"""Execute one ASSEMBLE-bound RLRMP training row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY, TrainingRunSpec
from feedbax.orchestration import ExecutionIdentityEnvelope
from feedbax.orchestration.events import RunEventEmitter
from feedbax.training.interruption import CancellationDecision
from feedbax.training.diagnostics import (
    NativeExecutionProducerContext,
    NativeTrainingDiagnosticsInput,
)

from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.initial_slots import RlrmpRuntime
from rlrmp.train.native_manifest import (
    RLRMP_NATIVE_MANIFEST_COMPANION_KEY,
    RlrmpNativeManifestCompanion,
)
from rlrmp.train.resume_control import target_training_batches
from rlrmp.runtime.training_run_specs import (
    feedbax_training_run_spec_from_rlrmp_record,
)


class SameRowResumeBinding(BaseModel):
    """Transaction-pinned custody authorization for operational row recovery."""

    model_config = ConfigDict(extra="forbid")
    checkpoint_root: str = Field(min_length=1)
    transaction_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_batches: int = Field(ge=0)


class RowLaunchPacket(BaseModel):
    """Canonical, transport-safe inputs for one row process."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["rlrmp.orchestrated_row_packet.v3"] = "rlrmp.orchestrated_row_packet.v3"
    run_set_id: str
    row_id: str
    envelope: ExecutionIdentityEnvelope
    payload: dict[str, Any]
    row_dir: str
    staged_checkpoint_root: str | None = None
    fork_record_path: str | None = None
    fork_record_sha256: str | None = None
    same_row_resume_binding: SameRowResumeBinding | None = None
    resume: bool = False
    stop_after_batches: int | None = None
    native_training_diagnostics: NativeTrainingDiagnosticsInput = Field(
        default_factory=NativeTrainingDiagnosticsInput
    )
    native_manifest_companion: RlrmpNativeManifestCompanion | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_v2_packet(cls, data: Any) -> Any:
        if (
            isinstance(data, Mapping)
            and data.get("schema_version") == "rlrmp.orchestrated_row_packet.v2"
        ):
            raise ValueError(
                "legacy orchestrated row packet v2 lacks the required native-manifest "
                "companion wire contract and must be regenerated"
            )
        return data

    @model_validator(mode="after")
    def _validate_native_manifest_companion(self) -> "RowLaunchPacket":
        metadata = self.payload.get("metadata")
        embedded = (
            metadata.get(RLRMP_NATIVE_MANIFEST_COMPANION_KEY)
            if isinstance(metadata, Mapping)
            else None
        )
        if embedded is None:
            if self.native_manifest_companion is not None:
                raise ValueError(
                    "row packet native-manifest companion is absent from execution payload"
                )
            if self.payload.get("schema_id") == "feedbax.spec.training_run":
                raise ValueError(
                    "RLRMP Feedbax training row requires a digest-bound native-manifest "
                    "companion; generic legacy manifest emission is forbidden"
                )
            return self
        validated = RlrmpNativeManifestCompanion.model_validate(embedded)
        if self.native_manifest_companion is None:
            raise ValueError(
                "row packet execution payload companion was not carried into the packet"
            )
        if validated != self.native_manifest_companion:
            raise ValueError(
                "row packet native-manifest companion does not match execution payload"
            )
        execution_payload = dict(self.payload)
        execution_metadata = dict(metadata)
        execution_metadata.pop(RLRMP_NATIVE_MANIFEST_COMPANION_KEY)
        execution_payload["metadata"] = execution_metadata
        actual_generic = TrainingRunSpec.model_validate(execution_payload).model_dump(
            mode="json", exclude_none=True
        )
        nested_generic = feedbax_training_run_spec_from_rlrmp_record(
            validated.training_spec_payload
        ).model_dump(mode="json", exclude_none=True)
        if actual_generic != nested_generic:
            raise ValueError(
                "/payload generic TrainingRunSpec does not match "
                "/native_manifest_companion/training_spec_payload/"
                "feedbax_training_run_spec"
            )
        return self


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

    from rlrmp import register_feedbax_training_methods
    from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods
    from rlrmp.train.execution_preparation import register_execution_preparations

    register_rlrmp_training_methods()
    register_feedbax_training_methods(DEFAULT_TRAINING_METHOD_REGISTRY)
    register_execution_preparations(DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY)
    run_spec = TrainingRunSpec.model_validate(packet.payload)
    execution_context = _native_execution_context(packet)
    provenance = execution_context.execution.row_provenance
    assert provenance is not None
    planned_run_id = provenance.planned_run_id
    preparation = DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
        ExecutionPreparationRequest(
            run_spec=run_spec,
            run_id=planned_run_id,
            resume=packet.resume,
        )
    )
    if packet.resume:
        if packet.same_row_resume_binding is not None:
            _verify_same_row_checkpoint(
                packet,
                run_spec=run_spec,
                preparation=preparation,
                planned_run_id=planned_run_id,
            )
        else:
            _verify_staged_checkpoint(packet)
    row_dir = Path(packet.row_dir)
    row_dir.mkdir(parents=True, exist_ok=True)
    emitter = RunEventEmitter.from_env()
    external_payload_kwargs = (
        packet.native_manifest_companion.external_training_payload_kwargs()
        if packet.native_manifest_companion is not None
        else {}
    )
    manifest_metadata_projection = (
        packet.native_manifest_companion.manifest_metadata_projection()
        if packet.native_manifest_companion is not None
        else None
    )
    result = execute_training_run_spec(
        run_spec,
        run_id=planned_run_id,
        initial_slots=preparation.initial_slots,
        kernel_context=preparation.kernel_context,
        loss_service=preparation.loss_service,
        manifest_root=row_dir / "feedbax-manifests",
        checkpoint_root=packet.staged_checkpoint_root,
        registry=DEFAULT_TRAINING_METHOD_REGISTRY,
        resume=packet.resume,
        resume_slot_transform=preparation.resume_slot_transform,
        run_event_emitter=emitter,
        cancellation_probe=_batch_limit_probe(
            packet.stop_after_batches,
            kernel_context=preparation.kernel_context,
        ),
        execution_context=execution_context,
        manifest_metadata_projection=manifest_metadata_projection,
        **external_payload_kwargs,
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


def _verify_same_row_checkpoint(
    packet: RowLaunchPacket,
    *,
    run_spec: TrainingRunSpec,
    preparation: Any,
    planned_run_id: str,
) -> None:
    """Revalidate a transaction-pinned operational resume before execution."""

    if packet.staged_checkpoint_root is None:
        raise ValueError("same-row resume requires a staged checkpoint root")
    if packet.envelope.immutable_inputs:
        raise ValueError("same-row resume cannot carry authored immutable checkpoint inputs")
    if packet.fork_record_path is not None or packet.fork_record_sha256 is not None:
        raise ValueError("same-row resume cannot carry a fork gate record")
    binding = packet.same_row_resume_binding
    assert binding is not None
    root = Path(packet.staged_checkpoint_root).resolve()
    if Path(binding.checkpoint_root).resolve() != root:
        raise ValueError("same-row resume binding checkpoint root does not match staged custody")

    from feedbax.training import load_checkpoint_custody_documents
    from feedbax.training.checkpoint_custody import load_latest_checkpoint

    loaded = load_latest_checkpoint(
        root,
        expected_run_spec=run_spec,
        expected_phase_program=run_spec.worker_execution.method_contract.phase_program,
        expected_slots=preparation.initial_slots,
        resume_slot_transform=preparation.resume_slot_transform,
        continuation_request=None,
        allow_new_lineage_override=False,
    )
    documents = load_checkpoint_custody_documents(root)
    latest = documents.latest_pointer.document
    if loaded.manifest.transaction_id != binding.transaction_id:
        raise ValueError("same-row resume transaction changed after launch authorization")
    if latest.manifest_sha256 != binding.manifest_sha256:
        raise ValueError("same-row resume manifest changed after launch authorization")
    if documents.manifest.document.transaction_id != loaded.manifest.transaction_id:
        raise ValueError("same-row resume custody changed during typed validation")
    if loaded.manifest.run_id != planned_run_id or latest.run_id != planned_run_id:
        raise ValueError("same-row resume checkpoint run identity does not match row provenance")
    completed = loaded.manifest.completed_training_batches
    target = target_training_batches(run_spec)
    if completed is None or completed < 0:
        raise ValueError("same-row resume checkpoint lacks valid completed-training progress")
    if completed != binding.completed_batches:
        raise ValueError(
            "same-row resume completed-batch binding changed after launch authorization"
        )
    if completed >= target:
        raise ValueError(
            "same-row resume checkpoint is already complete: "
            f"completed_batches={completed} target_batches={target}"
        )


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
