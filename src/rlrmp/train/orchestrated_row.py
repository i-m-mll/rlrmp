"""Execute one ASSEMBLE-bound RLRMP training row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from feedbax.contracts.training import TrainingRunSpec
from feedbax.orchestration.bundle import ExecutionIdentityEnvelope
from feedbax.orchestration.events import RunEventEmitter


class RowLaunchPacket(BaseModel):
    """Canonical, transport-safe inputs for one row process."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "rlrmp.orchestrated_row_packet.v1"
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
    if packet.resume:
        _verify_staged_checkpoint(packet)
    preparation = DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
        ExecutionPreparationRequest(
            run_spec=run_spec,
            run_id=packet.row_id,
            resume=packet.resume,
        )
    )
    row_dir = Path(packet.row_dir)
    row_dir.mkdir(parents=True, exist_ok=True)
    emitter = RunEventEmitter.from_env()
    result = execute_training_run_spec(
        run_spec,
        run_id=packet.row_id,
        initial_slots=preparation.initial_slots,
        kernel_context=preparation.kernel_context,
        loss_service=preparation.loss_service,
        manifest_root=row_dir / "feedbax-manifests",
        checkpoint_root=packet.staged_checkpoint_root,
        resume=packet.resume,
        resume_slot_transform=preparation.resume_slot_transform,
        run_event_emitter=emitter,
        cancellation_probe=_batch_limit_probe(packet.stop_after_batches),
    )
    environment_fingerprint = os.environ.get("FEEDBAX_ENV_FINGERPRINT", "unknown")
    manifest = result.manifest.model_copy(
        update={
            "run_set_id": packet.run_set_id,
            "intent_hash": packet.envelope.authored_intent.intent_hash,
            "resolved_semantics_root_hash": packet.envelope.resolved_snapshot.root_hash,
            "execution_hash": packet.envelope.execution_capsule.execution_hash,
            "input_data_identities": [
                item.model_dump(mode="json", exclude_none=True)
                for item in packet.envelope.immutable_inputs
            ],
            "metadata": {
                **result.manifest.metadata,
                "environment_fingerprint": environment_fingerprint,
            },
        }
    )
    manifest_path = row_dir / "manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json", exclude_none=True))
    diagnostics = _training_diagnostics(packet, result)
    _write_json(row_dir / "training-diagnostics.json", diagnostics)
    _write_json(
        row_dir / "training_summary.json",
        {
            "run_set_id": packet.run_set_id,
            "row_id": packet.row_id,
            "status": result.status,
            "completed_batches": diagnostics["completed_batches"],
            "metrics": dict(manifest.summary_metrics),
        },
    )
    if emitter is not None:
        emitter.close()
    return manifest_path


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


def _training_diagnostics(packet: RowLaunchPacket, result: Any) -> dict[str, Any]:
    payload = packet.payload
    metadata = dict(payload.get("metadata") or {})
    continuation = payload.get("checkpoint_progress", {}).get("continuation") or {}
    segment_completed = int(
        continuation.get(
            "additional_batches",
            payload.get(
                "n_batches",
                payload.get("training_config", {}).get(
                    "n_batches", payload.get("training", {}).get("n_batches", 0)
                ),
            ),
        )
    )
    absolute_completed = int(payload.get("training_config", {}).get("n_batches", segment_completed))
    checkpoint_interval = int(
        payload.get(
            "checkpoint_interval",
            payload.get("checkpoint_progress", {}).get(
                "checkpoint_interval",
                payload.get("training", {}).get("checkpoint_interval", 0),
            ),
        )
        or segment_completed
        or 1
    )
    source = int(metadata.get("resume_context", {}).get("current_step", 0))
    checkpoints = list(range(checkpoint_interval, segment_completed + 1, checkpoint_interval))
    segment_program_steps = list(range(1, len(result.checkpoint_writes) + 1))
    custody_program_steps = _custody_checkpoint_program_steps(result.checkpoint_writes)
    diagnostics = {
        "completed_batches": absolute_completed,
        "segment_completed_batches": segment_completed,
        "checkpoint_coordinates": checkpoints,
        "segment_checkpoint_program_steps": segment_program_steps,
        "absolute_completed_batches": [source + item for item in checkpoints],
        "resume_context": metadata.get("resume_context", {}),
        "optimizer_build_context": metadata.get("optimizer_build_context", {}),
        "lr_trace": _lr_trace(payload, metadata),
        "seeds": _diagnostic_seeds(payload, metadata),
    }
    if custody_program_steps is not None:
        diagnostics["custody_checkpoint_program_steps"] = custody_program_steps
    return diagnostics


def _custody_checkpoint_program_steps(writes: Any) -> list[int] | None:
    """Read authoritative ordinals from checkpoint transaction manifests."""
    steps = []
    for write in writes:
        manifest = getattr(write, "manifest", None)
        coordinate = getattr(manifest, "completed_coordinate", None)
        step = getattr(coordinate, "program_step", None)
        if isinstance(step, bool) or not isinstance(step, int):
            return None
        steps.append(step)
    return steps


def _diagnostic_seeds(payload: dict[str, Any], metadata: dict[str, Any]) -> Any:
    """Preserve the authored seed shape consumed by Feedbax conformance."""
    if "seeds" in payload:
        return payload["seeds"]
    if "seeds" in metadata:
        return metadata["seeds"]
    return metadata.get("seed", payload.get("seed", 0))


def _lr_trace(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, float]:
    optimizer = payload.get("method_payload", {}).get("payload", {}).get("controller_optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("lr_schedule") is None:
        return {}
    from feedbax.contracts.training import OptimizerSpec
    from feedbax.orchestration.schedule_eval import (
        evaluate_schedule_samples,
        require_schedule_context,
        schedule_sample_steps,
    )

    context = require_schedule_context(
        metadata.get("optimizer_build_context", {}), label="optimizer_build_context"
    )
    spec = OptimizerSpec.model_validate(optimizer)
    return {
        str(step): value
        for step, value in evaluate_schedule_samples(
            spec, context, schedule_sample_steps(spec, context)
        ).items()
    }


def _batch_limit_probe(limit: int | None) -> Any:
    if limit is None:
        return None
    seen = 0

    def probe(_coordinate: Any) -> str | None:
        nonlocal seen
        seen += 1
        return "stop" if seen >= limit else None

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
