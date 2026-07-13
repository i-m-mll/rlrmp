"""RLRMP driver bindings and RunPod launch-packet transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from feedbax.orchestration import (
    LocalOrchestrationDriver,
    RunBundle,
    RunRowSpec,
    RunSetState,
)
from feedbax.orchestration.drivers.runpod import (
    RunPodDriverConfig,
    RunPodOrchestrationDriver,
    RunPodTransport,
)
from feedbax.training.diagnostics import (
    NativeTrainingDiagnosticsInput,
    ScheduleContextDiagnostic,
)

from rlrmp.train.orchestrated_row import RowLaunchPacket


def local_driver_for_bundle(
    bundle: RunBundle,
    *,
    resume: bool = False,
    fork_record_path: Path | None = None,
    fork_record_sha256: str | None = None,
    same_row_resume_bindings: Mapping[str, Mapping[str, str | int]] | None = None,
    stop_after_batches: int | None = None,
) -> LocalOrchestrationDriver:
    """Materialize local row packets and return the stock local driver."""
    for row in bundle.rows:
        row_dir = bundle.run_set_dir / "rows" / row.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        _write_packet(
            row_dir / "launch-packet.json",
            _packet_for_row(
                bundle,
                row,
                row_dir=row_dir,
                resume=resume,
                checkpoint_root=_resume_checkpoint_root(
                    fork_record_path,
                    same_row_resume_bindings,
                    row.row_id,
                ),
                fork_record_path=fork_record_path,
                fork_record_sha256=fork_record_sha256,
                same_row_resume_binding=_same_row_binding(
                    same_row_resume_bindings,
                    row.row_id,
                ),
                stop_after_batches=stop_after_batches,
            ),
        )
    return LocalOrchestrationDriver(cwd=Path.cwd())


class RlrmpRunPodDriver:
    """Compose stock RunPod lifecycle machinery with canonical row packets."""

    def __init__(
        self,
        *,
        config: RunPodDriverConfig | None = None,
        transport: RunPodTransport | None = None,
        resume: bool = False,
        fork_record_path: Path | None = None,
        fork_record_sha256: str | None = None,
        same_row_resume_bindings: Mapping[str, Mapping[str, str | int]] | None = None,
        stop_after_batches: int | None = None,
    ) -> None:
        self.stock = RunPodOrchestrationDriver(config=config, transport=transport)
        self.config = self.stock.config
        self.resume = resume
        self.fork_record_path = fork_record_path
        self.fork_record_sha256 = fork_record_sha256
        self.same_row_resume_bindings = same_row_resume_bindings
        self.stop_after_batches = stop_after_batches
        self.poll_interval_seconds = self.stock.poll_interval_seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stock, name)

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        stock = self.stock.stage_inputs(bundle, state)
        staged: list[dict[str, str]] = []
        remote_run = self.stock._remote_run_dir(bundle)
        for row in bundle.rows:
            local_dir = bundle.run_set_dir / "packets" / row.row_id
            local_dir.mkdir(parents=True, exist_ok=True)
            remote_row = f"{remote_run}/rows/{row.row_id}"
            remote_checkpoint = None
            same_row_binding = _same_row_binding(self.same_row_resume_bindings, row.row_id)
            if self.resume:
                binding = _target_binding(self.fork_record_path, row.row_id)
                if binding is None:
                    binding = same_row_binding
                target_root = Path(str(binding["checkpoint_root"])) if binding else None
                if target_root is None:
                    raise ValueError("RunPod resume requires a fork target binding")
                remote_checkpoint = f"{remote_run}/inputs/{row.row_id}/checkpoint"
                self.stock._ssh(f"mkdir -p '{remote_checkpoint}'")
                self.stock.transport.rsync(
                    f"{target_root}/", f"{remote_checkpoint}/", delete=True
                ).check(f"stage checkpoint {row.row_id}")
                self.stock._ssh(_remote_target_verification_command(remote_checkpoint, binding))
            remote_fork_record = None
            if self.fork_record_path is not None:
                remote_fork_record = f"{remote_run}/inputs/fork-gate.json"
                self.stock._ssh(f"mkdir -p '{remote_run}/inputs'")
                self.stock.transport.rsync(str(self.fork_record_path), remote_fork_record).check(
                    "stage fork gate record"
                )
            packet = _packet_for_row(
                bundle,
                row,
                row_dir=Path(remote_row),
                resume=self.resume,
                checkpoint_root=Path(remote_checkpoint) if remote_checkpoint else None,
                fork_record_path=(Path(remote_fork_record) if remote_fork_record else None),
                fork_record_sha256=self.fork_record_sha256,
                same_row_resume_binding=_remote_same_row_binding(
                    same_row_binding,
                    remote_checkpoint=remote_checkpoint,
                ),
                stop_after_batches=self.stop_after_batches,
            )
            rewritten = _stage_identity_artifacts(self.stock, packet, local_dir, remote_row)
            packet_path = local_dir / "launch-packet.json"
            _write_packet(packet_path, rewritten)
            self.stock._ssh(f"mkdir -p '{remote_row}'")
            self.stock.transport.rsync(str(packet_path), f"{remote_row}/launch-packet.json").check(
                f"stage row packet {row.row_id}"
            )
            staged.append(
                {"row_id": row.row_id, "remote_packet": f"{remote_row}/launch-packet.json"}
            )
        return {**dict(stock), "rlrmp_packets": staged}

    def collect(self, bundle: RunBundle, row: RunRowSpec, state: RunSetState) -> Mapping[str, str]:
        del state
        remote_run = self.stock._remote_run_dir(bundle)
        destination = bundle.run_set_dir / "collected" / row.row_id
        destination.mkdir(parents=True, exist_ok=True)
        collected: dict[str, str] = {}
        for name in ("manifest.json", "training-diagnostics.json", "training_summary.json"):
            local_output = destination / name
            self.stock.transport.rsync(
                f"{remote_run}/rows/{row.row_id}/{name}", str(local_output), delete=False
            ).check(f"collect {row.row_id}:{name}")
            collected[name] = str(local_output)
        remote = f"{remote_run}/events/{row.row_id}.events.jsonl"
        local = bundle.run_set_dir / "events" / f"{row.row_id}.events.jsonl"
        local.parent.mkdir(parents=True, exist_ok=True)
        self.stock.transport.rsync(remote, str(local), delete=False).check(
            f"collect canonical events {row.row_id}"
        )
        collected[local.name] = str(local)
        return collected


def _packet_for_row(
    bundle: RunBundle,
    row: RunRowSpec,
    *,
    row_dir: Path,
    resume: bool,
    checkpoint_root: Path | None,
    fork_record_path: Path | None,
    fork_record_sha256: str | None,
    stop_after_batches: int | None,
    same_row_resume_binding: Mapping[str, str | int] | None = None,
) -> RowLaunchPacket:
    payload_ref = row.execution.payload
    if payload_ref.uri is None:
        raise ValueError(f"row {row.row_id!r} payload is not materialized")
    payload = json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))
    native_diagnostics = NativeTrainingDiagnosticsInput.model_validate(
        row.launch.metadata.get("native_training_diagnostics", {})
    )
    if same_row_resume_binding is not None:
        completed_batches = same_row_resume_binding.get("completed_batches")
        if isinstance(completed_batches, bool) or not isinstance(completed_batches, int):
            raise ValueError("same-row resume binding lacks typed completed-batch progress")
        schedule_context = ScheduleContextDiagnostic(
            schedule_origin_step=0,
            current_step=completed_batches,
            optimizer_count_at_current_step=completed_batches,
        )
        native_diagnostics = native_diagnostics.model_copy(
            update={
                "resume_context": schedule_context,
                "optimizer_build_context": schedule_context,
            }
        )
    return RowLaunchPacket(
        run_set_id=bundle.run_set_id,
        row_id=row.row_id,
        envelope=row.execution,
        payload=payload,
        row_dir=str(row_dir),
        staged_checkpoint_root=str(checkpoint_root) if checkpoint_root is not None else None,
        fork_record_path=str(fork_record_path) if fork_record_path is not None else None,
        fork_record_sha256=fork_record_sha256,
        same_row_resume_binding=same_row_resume_binding,
        resume=resume,
        stop_after_batches=stop_after_batches,
        native_training_diagnostics=native_diagnostics,
    )


def _stage_identity_artifacts(
    stock: RunPodOrchestrationDriver,
    packet: RowLaunchPacket,
    local_dir: Path,
    remote_row: str,
) -> RowLaunchPacket:
    envelope = packet.envelope
    updates: dict[str, Any] = {}
    for field in ("payload", "authored_intent", "resolved_snapshot", "execution_capsule"):
        ref = getattr(envelope, field)
        if ref.uri is None:
            raise ValueError(f"identity artifact {field} is not materialized")
        source = Path(ref.uri)
        target = f"{remote_row}/identity/{field}.json"
        stock._ssh(f"mkdir -p '{remote_row}/identity'")
        stock.transport.rsync(str(source), target).check(f"stage identity artifact {field}")
        updates[field] = ref.model_copy(update={"uri": target})
    return packet.model_copy(update={"envelope": envelope.model_copy(update=updates)})


def _write_packet(path: Path, packet: RowLaunchPacket) -> None:
    path.write_text(packet.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")


def _target_checkpoint_root(record_path: Path | None, row_id: str) -> Path | None:
    binding = _target_binding(record_path, row_id)
    return None if binding is None else Path(str(binding["checkpoint_root"]))


def _resume_checkpoint_root(
    record_path: Path | None,
    same_row_bindings: Mapping[str, Mapping[str, str | int]] | None,
    row_id: str,
) -> Path | None:
    target = _target_checkpoint_root(record_path, row_id)
    if target is not None:
        return target
    binding = _same_row_binding(same_row_bindings, row_id)
    return None if binding is None else Path(str(binding["checkpoint_root"]))


def _same_row_binding(
    bindings: Mapping[str, Mapping[str, str | int]] | None,
    row_id: str,
) -> Mapping[str, str | int] | None:
    if not bindings:
        return None
    binding = bindings.get(row_id)
    if binding is None:
        raise ValueError(f"same-row resume binding has no target for row {row_id!r}")
    return binding


def _remote_same_row_binding(
    binding: Mapping[str, str | int] | None,
    *,
    remote_checkpoint: str | None,
) -> Mapping[str, str | int] | None:
    if binding is None:
        return None
    if remote_checkpoint is None:
        raise ValueError("remote same-row resume requires staged checkpoint custody")
    return {**binding, "checkpoint_root": remote_checkpoint}


def _target_binding(record_path: Path | None, row_id: str) -> Mapping[str, Any] | None:
    if record_path is None:
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    binding = next(
        (item for item in record.get("targets", []) if item.get("row_id") == row_id), None
    )
    if binding is None:
        raise ValueError(f"fork gate record has no target for row {row_id!r}")
    return binding


def _remote_target_verification_command(remote_root: str, binding: Mapping[str, Any]) -> str:
    """Return an exact remote transaction-manifest digest probe."""
    manifest = f"{remote_root}/transactions/{binding['transaction_id']}/manifest.json"
    return f"printf '%s  %s\n' '{binding['manifest_sha256']}' '{manifest}' | sha256sum -c -"


__all__ = ["RlrmpRunPodDriver", "local_driver_for_bundle"]
