"""Checkpoint slot/barrier plumbing for C&S native training."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
import equinox as eqx
import jax.numpy as jnp
from jax_cookbook import save as fbx_save
from feedbax.training.checkpoint_custody import CheckpointCompatibilityError
from feedbax.contracts.worker import CheckpointBarrierSpec, CheckpointSlotSpec
from rlrmp.io import write_compact_json
from rlrmp.runtime.checkpoint_custody import (
    has_custody_checkpoint,
    has_feedbax_training_spec,
    load_cs_checkpoint_transaction,
    deserialize_pytree_slot,
    serialize_pytree_slot,
    write_cs_checkpoint_transaction,
)
from rlrmp.runtime.training_run_specs import feedbax_training_run_spec_from_payload

SCHEMA_VERSION = "rlrmp.cs_stochastic_gru.v1"

ADAPTIVE_EPSILON_ZERO_ADVERSARY_GAIN_TOLERANCE = 1e-8

ADAPTIVE_EPSILON_ZERO_ADVERSARY_STOP_REASON = (
    "adaptive_epsilon_zero_adversary_two_consecutive_checkpoints"
)


@dataclass(frozen=True)
class AdaptiveEpsilonState:
    """Host-side adaptive-lambda state for soft-energy direct-epsilon training."""

    lambda_value: float
    damage_ema: float | None = None
    clean_loss_ema: float | None = None
    last_update_batch: int | None = None
    update_count: int = 0
    schedule_start_batch: int = 0
    zero_adversary_guard: dict[str, Any] | None = None
    gain_estimate: float | None = None
    gain_samples: int = 0
    pending_lambda_log_step: float | None = None
    pending_log_damage_ema: float | None = None
    last_log_damage_ema: float | None = None
    ema_noise_floor: float | None = None
    last_lambda_step_sign: int = 0
    lambda_step_count: int = 0
    lambda_step_alternations: int = 0

    def to_json(self) -> dict[str, Any]:
        payload = {
            "lambda_value": float(self.lambda_value),
            "damage_ema": None if self.damage_ema is None else float(self.damage_ema),
            "clean_loss_ema": (None if self.clean_loss_ema is None else float(self.clean_loss_ema)),
            "last_update_batch": self.last_update_batch,
            "update_count": int(self.update_count),
            "schedule_start_batch": int(self.schedule_start_batch),
            "gain_estimate": (None if self.gain_estimate is None else float(self.gain_estimate)),
            "gain_samples": int(self.gain_samples),
            "pending_lambda_log_step": (
                None
                if self.pending_lambda_log_step is None
                else float(self.pending_lambda_log_step)
            ),
            "pending_log_damage_ema": (
                None if self.pending_log_damage_ema is None else float(self.pending_log_damage_ema)
            ),
            "last_log_damage_ema": (
                None if self.last_log_damage_ema is None else float(self.last_log_damage_ema)
            ),
            "ema_noise_floor": (
                None if self.ema_noise_floor is None else float(self.ema_noise_floor)
            ),
            "last_lambda_step_sign": int(self.last_lambda_step_sign),
            "lambda_step_count": int(self.lambda_step_count),
            "lambda_step_alternations": int(self.lambda_step_alternations),
        }
        if self.zero_adversary_guard is not None:
            payload["zero_adversary_guard"] = dict(self.zero_adversary_guard)
        return payload


@dataclass(frozen=True)
class TrainingState:
    """Serializable state needed to resume the chunked C&S GRU training loop."""

    model: Any
    optimizer_state: Any
    completed_batches: int
    key: Any
    history: Any | None
    adversary_policy: Any | None = None
    adversary_optimizer_state: Any | None = None
    adaptive_epsilon_state: AdaptiveEpsilonState | None = None


def _dump_json_metadata_bytes(file: Any, hyperparameters: dict[str, Any] | None) -> None:
    file.write(json.dumps(hyperparameters, sort_keys=True).encode("utf-8") + b"\n")


def _save_pytree(path: Path, tree: Any, *, hyperparameters: dict[str, Any] | None = None) -> None:
    fbx_save(
        path,
        tree,
        hyperparameters=hyperparameters,
        dump_fn=_dump_json_metadata_bytes,
    )


def save_training_checkpoint(
    checkpoint_root: Path,
    state: TrainingState,
    *,
    args: argparse.Namespace,
    run_spec: dict[str, Any],
    write_custody: bool = True,
    custody_result: Any | None = None,
) -> Path:
    """Write a Feedbax custody checkpoint and compatibility materialization."""

    metadata = _training_checkpoint_metadata(args, state, run_spec)
    if write_custody and has_feedbax_training_spec(run_spec):
        slots = _cs_checkpoint_slots(state, metadata)
        _validate_checkpoint_slots(
            _checkpoint_barrier_from_run_spec(run_spec),
            slots,
        )
        custody_result = write_cs_checkpoint_transaction(
            checkpoint_root,
            run_spec=run_spec,
            completed_batches=state.completed_batches,
            slots=slots,
            status=_checkpoint_transaction_status(args, state),
        )
    return _save_training_checkpoint_materialization(
        checkpoint_root,
        state,
        metadata=metadata,
        custody_result=custody_result,
    )


def _save_training_checkpoint_materialization(
    checkpoint_root: Path,
    state: TrainingState,
    *,
    metadata: dict[str, Any],
    custody_result: Any | None = None,
) -> Path:
    """Write the historical numbered checkpoint directory for compatibility."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _remove_stale_legacy_materializations(checkpoint_root, issue=str(metadata.get("issue", "")))
    checkpoint_name = f"checkpoint_{state.completed_batches:07d}"
    target = checkpoint_root / checkpoint_name
    tmp = checkpoint_root / f".{checkpoint_name}.tmp"
    if tmp.exists():
        _remove_tree(tmp)
    tmp.mkdir(parents=True)

    eqx.tree_serialise_leaves(tmp / "model.eqx", state.model)
    eqx.tree_serialise_leaves(tmp / "optimizer_state.eqx", state.optimizer_state)
    if state.adversary_policy is not None:
        eqx.tree_serialise_leaves(tmp / "adversary_policy.eqx", state.adversary_policy)
    if state.adversary_optimizer_state is not None:
        eqx.tree_serialise_leaves(
            tmp / "adversary_optimizer_state.eqx",
            state.adversary_optimizer_state,
        )
    if state.history is not None:
        _save_pytree(tmp / "history.eqx", state.history)
    _atomic_write_json(tmp / "metadata.json", metadata)
    _atomic_write_json(
        tmp / "provenance.json",
        _legacy_checkpoint_provenance(
            metadata=metadata,
            custody_result=custody_result,
        ),
    )
    if target.exists():
        _remove_tree(target)
    os.replace(tmp, target)
    _atomic_latest_link(checkpoint_root, checkpoint_name)
    _atomic_write_json(
        checkpoint_root / "checkpoint_index.json",
        {
            "latest": checkpoint_name,
            "latest_path": str(latest_checkpoint_path(checkpoint_root)),
            "completed_batches": state.completed_batches,
        },
    )
    return target


def _checkpoint_transaction_status(
    args: argparse.Namespace,
    state: TrainingState,
) -> Literal["partial", "final"]:
    try:
        n_train_batches = int(args.n_train_batches)
    except (TypeError, ValueError):
        return "partial"
    return "final" if int(state.completed_batches) >= n_train_batches else "partial"


def _legacy_checkpoint_provenance(
    *,
    metadata: Mapping[str, Any],
    custody_result: Any | None,
) -> dict[str, Any]:
    return {
        "schema_version": "rlrmp.legacy_checkpoint_provenance.v1",
        "issue": str(metadata.get("issue", "")),
        "source_transaction_id": _custody_transaction_id(custody_result),
        "materialized_at": datetime.now(UTC).isoformat(),
        "writer": "rlrmp.train.cs_nominal_gru._save_training_checkpoint_materialization",
        "authoritative": False,
    }


def _custody_transaction_id(custody_result: Any | None) -> str | None:
    if custody_result is None:
        return None
    manifest = getattr(custody_result, "manifest", None)
    transaction_id = getattr(manifest, "transaction_id", None)
    if transaction_id:
        return str(transaction_id)
    latest_pointer = getattr(custody_result, "latest_pointer", None)
    transaction_id = getattr(latest_pointer, "transaction_id", None)
    return None if transaction_id is None else str(transaction_id)


def _remove_stale_legacy_materializations(checkpoint_root: Path, *, issue: str) -> None:
    """Remove legacy checkpoint dirs whose metadata belongs to another issue."""

    for path in checkpoint_root.glob("checkpoint_[0-9]*"):
        if not path.is_dir() or path.is_symlink():
            continue
        legacy_issue = _legacy_materialization_issue(path)
        if legacy_issue != issue:
            _remove_tree(path)

    latest = checkpoint_root / "checkpoint_latest"
    if not latest.exists() and not latest.is_symlink():
        return
    try:
        latest_issue = _legacy_materialization_issue(latest.resolve())
    except FileNotFoundError:
        latest_issue = None
    if latest_issue != issue:
        if latest.is_dir() and not latest.is_symlink():
            _remove_tree(latest)
        else:
            latest.unlink()


def _legacy_materialization_issue(path: Path) -> str | None:
    provenance_path = path / "provenance.json"
    provenance = _read_json_if_present(provenance_path)
    if isinstance(provenance, dict) and provenance.get("issue"):
        return str(provenance["issue"])

    metadata = _read_json_if_present(path / "metadata.json")
    if isinstance(metadata, dict) and metadata.get("issue"):
        return str(metadata["issue"])
    return None


def _read_json_if_present(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_latest_checkpoint(
    checkpoint_root: Path,
    *,
    model_template: Any,
    optimizer_state_template: Any,
    history_template: Any | None = None,
    adversary_policy_template: Any | None = None,
    adversary_optimizer_state_template: Any | None = None,
    run_spec: dict[str, Any] | None = None,
) -> TrainingState:
    """Load the latest checkpoint using Feedbax custody, with legacy fallback."""

    if (
        run_spec is not None
        and has_feedbax_training_spec(run_spec)
        and has_custody_checkpoint(checkpoint_root)
    ):
        expected_slots = _cs_expected_slots(
            model_template=model_template,
            optimizer_state_template=optimizer_state_template,
            adversary_policy_template=adversary_policy_template,
            adversary_optimizer_state_template=adversary_optimizer_state_template,
        )
        _validate_checkpoint_slots(
            _checkpoint_barrier_from_run_spec(run_spec),
            expected_slots,
        )
        loaded = load_cs_checkpoint_transaction(
            checkpoint_root,
            run_spec=run_spec,
            expected_slots=expected_slots,
        )
        return _training_state_from_cs_slots(
            loaded.slots,
            model_template=model_template,
            optimizer_state_template=optimizer_state_template,
            adversary_policy_template=adversary_policy_template,
            adversary_optimizer_state_template=adversary_optimizer_state_template,
        )

    return _load_latest_checkpoint_materialization(
        checkpoint_root,
        model_template=model_template,
        optimizer_state_template=optimizer_state_template,
        history_template=history_template,
        adversary_policy_template=adversary_policy_template,
        adversary_optimizer_state_template=adversary_optimizer_state_template,
    )


def _load_latest_checkpoint_materialization(
    checkpoint_root: Path,
    *,
    model_template: Any,
    optimizer_state_template: Any,
    history_template: Any | None = None,
    adversary_policy_template: Any | None = None,
    adversary_optimizer_state_template: Any | None = None,
) -> TrainingState:
    """Load historical ``checkpoint_latest`` using explicit PyTree templates."""

    checkpoint_path = latest_checkpoint_path(checkpoint_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_latest found under {checkpoint_root}")
    metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
    model = eqx.tree_deserialise_leaves(checkpoint_path / "model.eqx", model_template)
    optimizer_state = eqx.tree_deserialise_leaves(
        checkpoint_path / "optimizer_state.eqx",
        optimizer_state_template,
    )
    history_path = checkpoint_path / "history.eqx"
    history = (
        eqx.tree_deserialise_leaves(history_path, history_template)
        if history_template is not None and history_path.exists()
        else None
    )
    adversary_policy_path = checkpoint_path / "adversary_policy.eqx"
    adversary_policy = (
        eqx.tree_deserialise_leaves(adversary_policy_path, adversary_policy_template)
        if adversary_policy_template is not None and adversary_policy_path.exists()
        else None
    )
    adversary_optimizer_state_path = checkpoint_path / "adversary_optimizer_state.eqx"
    adversary_optimizer_state = (
        eqx.tree_deserialise_leaves(
            adversary_optimizer_state_path,
            adversary_optimizer_state_template,
        )
        if (
            adversary_optimizer_state_template is not None
            and adversary_optimizer_state_path.exists()
        )
        else None
    )
    adaptive_payload = metadata.get("adaptive_epsilon_state")
    adaptive_epsilon_state = (
        AdaptiveEpsilonState(
            lambda_value=float(adaptive_payload["lambda_value"]),
            damage_ema=(
                None
                if adaptive_payload.get("damage_ema") is None
                else float(adaptive_payload["damage_ema"])
            ),
            clean_loss_ema=(
                None
                if adaptive_payload.get("clean_loss_ema") is None
                else float(adaptive_payload["clean_loss_ema"])
            ),
            last_update_batch=adaptive_payload.get("last_update_batch"),
            update_count=int(adaptive_payload.get("update_count", 0)),
            schedule_start_batch=int(adaptive_payload.get("schedule_start_batch", 0)),
            zero_adversary_guard=(
                _normalize_adaptive_epsilon_zero_guard(
                    adaptive_payload.get("zero_adversary_guard"),
                    enabled=True,
                )
                if isinstance(adaptive_payload.get("zero_adversary_guard"), dict)
                else None
            ),
            gain_estimate=(
                None
                if adaptive_payload.get("gain_estimate") is None
                else float(adaptive_payload["gain_estimate"])
            ),
            gain_samples=int(adaptive_payload.get("gain_samples", 0)),
            pending_lambda_log_step=(
                None
                if adaptive_payload.get("pending_lambda_log_step") is None
                else float(adaptive_payload["pending_lambda_log_step"])
            ),
            pending_log_damage_ema=(
                None
                if adaptive_payload.get("pending_log_damage_ema") is None
                else float(adaptive_payload["pending_log_damage_ema"])
            ),
            last_log_damage_ema=(
                None
                if adaptive_payload.get("last_log_damage_ema") is None
                else float(adaptive_payload["last_log_damage_ema"])
            ),
            ema_noise_floor=(
                None
                if adaptive_payload.get("ema_noise_floor") is None
                else float(adaptive_payload["ema_noise_floor"])
            ),
            last_lambda_step_sign=int(adaptive_payload.get("last_lambda_step_sign", 0)),
            lambda_step_count=int(adaptive_payload.get("lambda_step_count", 0)),
            lambda_step_alternations=int(adaptive_payload.get("lambda_step_alternations", 0)),
        )
        if isinstance(adaptive_payload, dict)
        else None
    )
    return TrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=int(metadata["completed_batches"]),
        key=jnp.asarray(metadata["next_prng_key"], dtype=jnp.uint32),
        history=history,
        adversary_policy=adversary_policy,
        adversary_optimizer_state=adversary_optimizer_state,
        adaptive_epsilon_state=adaptive_epsilon_state,
    )


def _training_checkpoint_metadata(
    args: argparse.Namespace,
    state: TrainingState,
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "schema_version": f"{SCHEMA_VERSION}.checkpoint.v1",
        "issue": str(args.issue),
        "completed_batches": state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "seed": int(args.seed),
        "next_prng_key": _plain(state.key),
        "stochastic_preset": str(args.stochastic_preset),
        "run_spec": run_spec,
    }
    if state.adaptive_epsilon_state is not None:
        metadata["adaptive_epsilon_state"] = state.adaptive_epsilon_state.to_json()
    return metadata


def _checkpoint_barrier_from_run_spec(
    run_spec: dict[str, Any],
) -> CheckpointBarrierSpec:
    """Return the canonical C&S checkpoint barrier from the composed run spec."""

    training_spec = feedbax_training_run_spec_from_payload(run_spec)
    barriers = training_spec.worker_execution.method_contract.phase_program.checkpoint_barriers
    if len(barriers) != 1:
        raise CheckpointCompatibilityError(
            f"C&S training requires exactly one declared checkpoint barrier; found {len(barriers)}"
        )
    return barriers[0]


def _validate_checkpoint_slots(
    barrier: CheckpointBarrierSpec,
    slots: Mapping[str, Any],
) -> None:
    """Check required custody slots against the barrier's CheckpointSlotSpecs."""

    missing = [
        slot.slot
        for slot in barrier.slots
        if isinstance(slot, CheckpointSlotSpec) and slot.required and slot.slot not in slots
    ]
    if missing:
        raise CheckpointCompatibilityError(
            f"checkpoint barrier {barrier.name!r} is missing required slots: {missing}"
        )


def _cs_checkpoint_slots(
    state: TrainingState,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    slots: dict[str, Any] = {
        "model": serialize_pytree_slot(state.model),
        "optimizer": serialize_pytree_slot(state.optimizer_state),
        "prng": state.key,
        "completed_batches": jnp.asarray(state.completed_batches, dtype=jnp.int32),
        "checkpoint_metadata": metadata,
    }
    if state.history is not None:
        slots["history"] = state.history
    if state.adversary_policy is not None:
        slots["adversary_policy"] = serialize_pytree_slot(state.adversary_policy)
    if state.adversary_optimizer_state is not None:
        slots["adversary_optimizer"] = serialize_pytree_slot(state.adversary_optimizer_state)
    if state.adaptive_epsilon_state is not None:
        slots["adaptive_epsilon_state"] = state.adaptive_epsilon_state
        zero_guard = getattr(state.adaptive_epsilon_state, "zero_adversary_guard", None)
        if isinstance(zero_guard, dict):
            slots["zero_adversary_guard"] = dict(zero_guard)
    return slots


def _cs_expected_slots(
    *,
    model_template: Any,
    optimizer_state_template: Any,
    adversary_policy_template: Any | None,
    adversary_optimizer_state_template: Any | None,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "model": serialize_pytree_slot(model_template),
        "optimizer": serialize_pytree_slot(optimizer_state_template),
        "prng": jnp.asarray([0, 0], dtype=jnp.uint32),
        "completed_batches": jnp.asarray(0, dtype=jnp.int32),
    }
    if adversary_policy_template is not None:
        expected["adversary_policy"] = serialize_pytree_slot(adversary_policy_template)
    if adversary_optimizer_state_template is not None:
        expected["adversary_optimizer"] = serialize_pytree_slot(adversary_optimizer_state_template)
    return expected


def _training_state_from_cs_slots(
    slots: Mapping[str, Any],
    *,
    model_template: Any,
    optimizer_state_template: Any,
    adversary_policy_template: Any | None,
    adversary_optimizer_state_template: Any | None,
) -> TrainingState:
    try:
        model = deserialize_pytree_slot(slots["model"], model_template, slot="model")
        optimizer_state = deserialize_pytree_slot(
            slots["optimizer"],
            optimizer_state_template,
            slot="optimizer",
        )
        adversary_policy = (
            deserialize_pytree_slot(
                slots["adversary_policy"],
                adversary_policy_template,
                slot="adversary_policy",
            )
            if adversary_policy_template is not None and "adversary_policy" in slots
            else None
        )
        adversary_optimizer_state = (
            deserialize_pytree_slot(
                slots["adversary_optimizer"],
                adversary_optimizer_state_template,
                slot="adversary_optimizer",
            )
            if (adversary_optimizer_state_template is not None and "adversary_optimizer" in slots)
            else None
        )
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "checkpoint PyTree slot could not be deserialized with the resume template"
        ) from exc
    return TrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=int(slots["completed_batches"]),
        key=jnp.asarray(slots["prng"], dtype=jnp.uint32),
        history=slots.get("history"),
        adversary_policy=adversary_policy,
        adversary_optimizer_state=adversary_optimizer_state,
        adaptive_epsilon_state=slots.get("adaptive_epsilon_state"),
    )


def latest_checkpoint_path(checkpoint_root: Path) -> Path:
    """Return the path used by the durable latest-checkpoint contract."""

    return checkpoint_root / "checkpoint_latest"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    write_compact_json(path, payload, atomic=True)


def _initial_adaptive_epsilon_zero_guard(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "stop_reason": ADAPTIVE_EPSILON_ZERO_ADVERSARY_STOP_REASON,
        "gain_tolerance": ADAPTIVE_EPSILON_ZERO_ADVERSARY_GAIN_TOLERANCE,
        "checkpoints_seen": 0,
        "consecutive_active_zero_adversary_checkpoints": 0,
        "should_stop": False,
        "last_checkpoint": None,
    }


def _normalize_adaptive_epsilon_zero_guard(
    payload: Any,
    *,
    enabled: bool,
) -> dict[str, Any]:
    guard = _initial_adaptive_epsilon_zero_guard(enabled=enabled)
    if not isinstance(payload, dict):
        return guard
    guard["enabled"] = bool(payload.get("enabled", enabled))
    guard["stop_reason"] = str(
        payload.get("stop_reason", ADAPTIVE_EPSILON_ZERO_ADVERSARY_STOP_REASON)
    )
    guard["gain_tolerance"] = float(
        payload.get(
            "gain_tolerance",
            ADAPTIVE_EPSILON_ZERO_ADVERSARY_GAIN_TOLERANCE,
        )
    )
    guard["checkpoints_seen"] = int(payload.get("checkpoints_seen", 0))
    guard["consecutive_active_zero_adversary_checkpoints"] = int(
        payload.get("consecutive_active_zero_adversary_checkpoints", 0)
    )
    guard["should_stop"] = bool(payload.get("should_stop", False))
    last_checkpoint = payload.get("last_checkpoint")
    guard["last_checkpoint"] = last_checkpoint if isinstance(last_checkpoint, dict) else None
    return guard


def _atomic_latest_link(checkpoint_root: Path, checkpoint_name: str) -> None:
    latest = checkpoint_root / "checkpoint_latest"
    tmp_link = checkpoint_root / ".checkpoint_latest.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(checkpoint_name, tmp_link)
    os.replace(tmp_link, latest)


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def _plain(value: Any) -> Any:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


__all__ = [
    "ADAPTIVE_EPSILON_ZERO_ADVERSARY_GAIN_TOLERANCE",
    "ADAPTIVE_EPSILON_ZERO_ADVERSARY_STOP_REASON",
    "AdaptiveEpsilonState",
    "SCHEMA_VERSION",
    "TrainingState",
    "_atomic_latest_link",
    "_atomic_write_json",
    "_checkpoint_barrier_from_run_spec",
    "_checkpoint_transaction_status",
    "_cs_checkpoint_slots",
    "_cs_expected_slots",
    "_custody_transaction_id",
    "_dump_json_metadata_bytes",
    "_initial_adaptive_epsilon_zero_guard",
    "_legacy_checkpoint_provenance",
    "_legacy_materialization_issue",
    "_load_latest_checkpoint_materialization",
    "_normalize_adaptive_epsilon_zero_guard",
    "_plain",
    "_read_json_if_present",
    "_remove_stale_legacy_materializations",
    "_remove_tree",
    "_save_pytree",
    "_save_training_checkpoint_materialization",
    "_training_checkpoint_metadata",
    "_training_state_from_cs_slots",
    "_validate_checkpoint_slots",
    "latest_checkpoint_path",
    "load_latest_checkpoint",
    "save_training_checkpoint",
]
