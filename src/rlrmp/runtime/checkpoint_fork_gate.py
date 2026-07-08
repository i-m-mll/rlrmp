"""Pre-launch checkpoint-fork parity gate for multi-row training launches."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import optax
from feedbax.config.namespace import dict_to_namespace
from feedbax.contracts.training import TrainingRunSpec
from feedbax.training.checkpoint_custody import fork_checkpoint_transaction

from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    register_rlrmp_cs_supervised_method,
    register_rlrmp_distillation_methods,
)
from rlrmp.train.cs_nominal_gru import _learning_rate_schedule, make_delayed_cosine_schedule
from rlrmp.train.minimax import (
    MINIMAX_METHOD_REF,
    ensure_minimax_training_method_registered,
    minimax_training_run_spec_to_config,
)


PARITY_SCHEMA_VERSION = "rlrmp.checkpoint_fork_parity.v1"
LR_CONTINUATION_PREFIX = "LR_CONTINUATION"
_PLAN_DECLARATION_PATTERN = re.compile(
    r"(?:lr|learning[-_ ]rate)[-_ ]continuation"
    r"(?:[-_ ](?:schedule|mode|semantics))?\s*(?::|=|\|)\s*"
    r"(?P<mode>restart|continue)\b",
    re.IGNORECASE,
)


class ForkParityError(RuntimeError):
    """Raised when a forked checkpoint does not match the source by manifest digest."""


class LRContinuationDeclarationError(ValueError):
    """Raised when the run plan does not declare restart/continue LR semantics."""


@dataclass(frozen=True)
class ForkTarget:
    """One target row to fork into and verify."""

    row_id: str
    spec_path: Path
    checkpoint_root: Path


def register_rlrmp_training_methods() -> None:
    """Register RLRMP Feedbax training methods in this process."""

    ensure_minimax_training_method_registered()
    register_rlrmp_cs_supervised_method()
    register_rlrmp_distillation_methods()


def parse_target(value: str) -> ForkTarget:
    """Parse ``ROW=SPEC_JSON:CHECKPOINT_ROOT`` into a fork target."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "target must be ROW=SPEC_JSON:CHECKPOINT_ROOT"
        )
    row_id, rest = value.split("=", 1)
    if not row_id:
        raise argparse.ArgumentTypeError("target row id must not be empty")
    if ":" not in rest:
        raise argparse.ArgumentTypeError(
            "target must be ROW=SPEC_JSON:CHECKPOINT_ROOT"
        )
    spec_path, checkpoint_root = rest.split(":", 1)
    if not spec_path or not checkpoint_root:
        raise argparse.ArgumentTypeError(
            "target spec path and checkpoint root must not be empty"
        )
    return ForkTarget(
        row_id=row_id,
        spec_path=Path(spec_path),
        checkpoint_root=Path(checkpoint_root),
    )


def load_row_payload(path: Path) -> dict[str, Any]:
    """Load one row recipe JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"row spec must contain a JSON object: {path}")
    return payload


def training_spec_from_row_payload(payload: Mapping[str, Any]) -> TrainingRunSpec:
    """Extract and validate the Feedbax TrainingRunSpec before wrapper validation."""

    register_rlrmp_training_methods()
    if FEEDBAX_TRAINING_RUN_SPEC_KEY in payload:
        raw_spec = payload[FEEDBAX_TRAINING_RUN_SPEC_KEY]
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"{FEEDBAX_TRAINING_RUN_SPEC_KEY} must be a JSON object")
        return TrainingRunSpec.model_validate(dict(raw_spec))
    return TrainingRunSpec.model_validate(dict(payload))


def declared_lr_continuation_mode(run_plan_path: Path) -> str:
    """Return the declared LR continuation mode from a spec-lock/RUN_PLAN file."""

    text = run_plan_path.read_text(encoding="utf-8")
    match = _PLAN_DECLARATION_PATTERN.search(text)
    if match is None:
        raise LRContinuationDeclarationError(
            "RUN_PLAN/spec-lock must declare LR continuation schedule as "
            "'LR continuation schedule: restart' or 'LR continuation schedule: continue'"
        )
    return match.group("mode").lower()


def fork_checkpoints_with_parity(
    *,
    source_checkpoint_root: Path,
    targets: Sequence[ForkTarget],
    run_plan_path: Path,
    parity_output_path: Path,
    skip_fork: bool = False,
    tool_version: str = "rlrmp.checkpoint_fork_gate.v1",
) -> dict[str, Any]:
    """Fork a source checkpoint into target rows and write a parity table."""

    if not targets:
        raise ValueError("at least one fork target is required")
    mode = declared_lr_continuation_mode(run_plan_path)
    row_payloads = {target.row_id: load_row_payload(target.spec_path) for target in targets}
    row_specs = {
        row_id: training_spec_from_row_payload(payload)
        for row_id, payload in row_payloads.items()
    }

    fork_results: dict[str, Mapping[str, Any]] = {}
    if not skip_fork:
        for target in targets:
            spec = row_specs[target.row_id]
            result = fork_checkpoint_transaction(
                source_checkpoint_root,
                target.checkpoint_root,
                target_run_spec=spec,
                target_phase_program=spec.worker_execution.method_contract.phase_program,
                tool_version=tool_version,
                metadata={"rlrmp_row_id": target.row_id},
            )
            fork_results[target.row_id] = {
                "transaction_id": result.manifest.transaction_id,
                "manifest_path": str(result.manifest_path),
                "latest_pointer_path": str(result.latest_pointer_path),
                "slot_transfer_modes": dict(result.slot_transfer_modes),
            }

    source_manifest_path, source_manifest = _latest_manifest(source_checkpoint_root)
    lr_continuation = _lr_continuation_report(
        source_manifest,
        row_payloads[targets[0].row_id],
        row_specs[targets[0].row_id],
        declared_mode=mode,
    )
    print(
        f"{LR_CONTINUATION_PREFIX} step={lr_continuation['step']} "
        f"lr={lr_continuation['lr']:.12g}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for target in targets:
        target_manifest_path, target_manifest = _latest_manifest(target.checkpoint_root)
        row = _row_parity(
            row_id=target.row_id,
            spec_path=target.spec_path,
            checkpoint_root=target.checkpoint_root,
            source_manifest=source_manifest,
            target_manifest=target_manifest,
            target_manifest_path=target_manifest_path,
            fork_summary=fork_results.get(target.row_id, {}),
        )
        rows.append(row)
        for slot, slot_row in row["slots"].items():
            if not slot_row["match"]:
                mismatches.append(
                    "fork parity mismatch "
                    f"row={target.row_id} slot={slot} "
                    f"source={slot_row.get('source_digest')} "
                    f"target={slot_row.get('target_digest')}"
                )

    table = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "source": {
            "checkpoint_root": str(source_checkpoint_root),
            "manifest_path": str(source_manifest_path),
            "transaction_id": source_manifest.get("transaction_id"),
            "completed_coordinate": source_manifest.get("completed_coordinate"),
            "slot_digests": _slot_digest_map(source_manifest),
        },
        "lr_continuation": lr_continuation,
        "targets": rows,
        "ok": not mismatches,
    }
    parity_output_path.parent.mkdir(parents=True, exist_ok=True)
    parity_output_path.write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mismatches:
        raise ForkParityError("; ".join(mismatches))
    return table


def _latest_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    latest_path = root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    manifest_rel = latest.get("manifest_relative_path")
    if not isinstance(manifest_rel, str) or not manifest_rel:
        raise ValueError(f"latest pointer lacks manifest_relative_path: {latest_path}")
    manifest_path = root / manifest_rel
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"checkpoint manifest must contain a JSON object: {manifest_path}")
    return manifest_path, manifest


def _row_parity(
    *,
    row_id: str,
    spec_path: Path,
    checkpoint_root: Path,
    source_manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    target_manifest_path: Path,
    fork_summary: Mapping[str, Any],
) -> dict[str, Any]:
    source_slots = _slot_digest_map(source_manifest)
    target_slots = _slot_digest_map(target_manifest)
    fork_source = (target_manifest.get("fork_provenance") or {}).get("source") or {}
    source_transaction_id = source_manifest.get("transaction_id")
    if fork_source.get("transaction_id") != source_transaction_id:
        raise ForkParityError(
            "target fork provenance does not point at source "
            f"row={row_id} source={source_transaction_id} "
            f"target_source={fork_source.get('transaction_id')}"
        )

    slots: dict[str, dict[str, Any]] = {}
    for slot in sorted(set(source_slots) | set(target_slots)):
        source_digest = source_slots.get(slot)
        target_digest = target_slots.get(slot)
        slots[slot] = {
            "source_digest": source_digest,
            "target_digest": target_digest,
            "match": source_digest == target_digest,
        }
    return {
        "row_id": row_id,
        "spec_path": str(spec_path),
        "checkpoint_root": str(checkpoint_root),
        "manifest_path": str(target_manifest_path),
        "transaction_id": target_manifest.get("transaction_id"),
        "completed_coordinate": target_manifest.get("completed_coordinate"),
        "slot_transfer_modes": dict(fork_summary.get("slot_transfer_modes", {})),
        "slots": slots,
        "ok": all(slot["match"] for slot in slots.values()),
    }


def _slot_digest_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    digests: dict[str, str] = {}
    slots = manifest.get("slots")
    if not isinstance(slots, list):
        raise ValueError("checkpoint manifest lacks slots list")
    for slot_payload in slots:
        if not isinstance(slot_payload, Mapping):
            raise ValueError("checkpoint manifest slot entry is not an object")
        slot = str(slot_payload.get("slot"))
        content = slot_payload.get("content_digest")
        digest = None
        if isinstance(content, Mapping):
            digest = content.get("slot_root_sha256")
        if not digest:
            digest = slot_payload.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"checkpoint manifest slot {slot!r} lacks a digest")
        digests[slot] = digest
    return digests


def _lr_continuation_report(
    source_manifest: Mapping[str, Any],
    row_payload: Mapping[str, Any],
    row_spec: TrainingRunSpec,
    *,
    declared_mode: str,
) -> dict[str, Any]:
    completed_batches = _completed_batches(source_manifest, row_spec)
    step = 0 if declared_mode == "restart" else completed_batches
    lr = _learning_rate_at_step(row_payload, row_spec, step)
    return {
        "declared_mode": declared_mode,
        "completed_batches": completed_batches,
        "step": step,
        "lr": lr,
    }


def _completed_batches(
    source_manifest: Mapping[str, Any],
    row_spec: TrainingRunSpec,
) -> int:
    coordinate = source_manifest.get("completed_coordinate")
    if not isinstance(coordinate, Mapping):
        raise ValueError("source checkpoint manifest lacks completed_coordinate")
    global_step = int(coordinate.get("global_step", 0))
    if _method_ref_string(row_spec) != MINIMAX_METHOD_REF:
        return global_step
    config = minimax_training_run_spec_to_config(row_spec)
    barrier = coordinate.get("completed_barrier")
    phase = coordinate.get("phase")
    if barrier == "after_warmup" or phase == "warmup":
        return int(config.get("n_warmup_batches", 0))
    if barrier == "after_adversarial" or phase == "adversarial":
        return int(config.get("n_warmup_batches", 0)) + global_step
    return global_step


def _learning_rate_at_step(
    row_payload: Mapping[str, Any],
    row_spec: TrainingRunSpec,
    step: int,
) -> float:
    hps = row_payload.get("hps")
    if isinstance(hps, Mapping) and "learning_rate_0" in hps:
        schedule = _learning_rate_schedule(dict_to_namespace(dict(hps)))
        return float(jax.device_get(schedule(int(step))))
    if _method_ref_string(row_spec) == MINIMAX_METHOD_REF:
        config = minimax_training_run_spec_to_config(row_spec)
        if int(step) < int(config.get("n_warmup_batches", 0)):
            schedule = make_delayed_cosine_schedule(
                float(config["controller_lr"]),
                constant_steps=0,
                total_steps=max(1, int(config.get("n_warmup_batches", 1))),
            )
            return float(jax.device_get(schedule(int(step))))
        return float(config["controller_lr"])
    learning_rate = _constant_learning_rate(row_payload)
    if learning_rate is not None:
        return learning_rate
    schedule = optax.constant_schedule(0.0)
    return float(jax.device_get(schedule(int(step))))


def _constant_learning_rate(row_payload: Mapping[str, Any]) -> float | None:
    optimizer = row_payload.get("optimizer")
    if isinstance(optimizer, Mapping):
        for key in ("learning_rate", "controller_lr"):
            if key in optimizer:
                return float(optimizer[key])
        controller = optimizer.get("controller")
        if isinstance(controller, Mapping) and "learning_rate" in controller:
            return float(controller["learning_rate"])
    summary = row_payload.get("training_summary")
    if isinstance(summary, Mapping) and "controller_lr" in summary:
        return float(summary["controller_lr"])
    return None


def _method_ref_string(spec: TrainingRunSpec) -> str:
    method_ref = spec.method_ref
    return f"{method_ref.package}/{method_ref.name}/{method_ref.version}"
