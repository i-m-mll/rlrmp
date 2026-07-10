"""RLRMP-specific learning-rate continuation reporting for checkpoint forks."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.random as jr
from feedbax.config.namespace import dict_to_namespace
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    OptimizerSpec,
    TrainingRunSpec,
)
from feedbax.training.optimizers import learning_rate_at_step
from feedbax.training.run_matrix import ForkParityError

from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot
from rlrmp.train.adaptive_epsilon_native import (
    ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    AdaptiveEpsilonMethodPayload,
    adaptive_epsilon_controller_lr_points,
    build_adaptive_epsilon_native_initial_slots,
    lr_report_schedule_steps,
    lr_resume_context_for_mode,
    optimizer_count_at_current_step,
)
from rlrmp.train.cs_nominal_gru import (
    _config_namespace,
    _learning_rate_schedule,
    build_hps,
    make_delayed_cosine_schedule,
)
from rlrmp.train.executor.slots import OPTIMIZER
from rlrmp.train.minimax import (
    MINIMAX_METHOD_REF,
    minimax_training_run_spec_to_config,
)


class RlrmpLrContinuationReporter:
    """Report RLRMP method-specific LR continuation points for matrix forks."""

    def __init__(self, *, source_checkpoint_root: Path) -> None:
        self.source_checkpoint_root = source_checkpoint_root

    def points(
        self,
        *,
        source_manifest: Mapping[str, Any],
        row_payload: Mapping[str, Any],
        row_spec: TrainingRunSpec,
        declared_mode: str,
    ) -> list[dict[str, Any]]:
        """Return LR points with RLRMP method-specific resume semantics."""
        completed_batches = _completed_batches(source_manifest, row_spec)
        if _method_ref_string(row_spec) == ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF:
            source_manifest_path, latest_manifest = _latest_manifest(self.source_checkpoint_root)
            return _adaptive_epsilon_lr_continuation_points(
                source_manifest_path,
                latest_manifest,
                row_spec,
                declared_mode=declared_mode,
                completed_batches=completed_batches,
            )
        step = 0 if declared_mode == "restart" else completed_batches
        lr = _learning_rate_at_step(row_payload, row_spec, step)
        return [
            {
                "step": step,
                "global_step": step,
                "optimizer_count": step,
                "lr": lr,
                "mode": declared_mode,
                "completed_batches": completed_batches,
            }
        ]


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


def _adaptive_epsilon_lr_continuation_points(
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    row_spec: TrainingRunSpec,
    *,
    declared_mode: str,
    completed_batches: int,
) -> list[dict[str, Any]]:
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        row_spec.method_ref,
        row_spec.method_payload,
        path="/method_payload",
    )
    if not isinstance(payload, AdaptiveEpsilonMethodPayload):
        raise TypeError("adaptive-epsilon row did not validate to AdaptiveEpsilonMethodPayload")
    if payload.controller_optimizer is None or payload.controller_optimizer.lr_schedule is None:
        raise ValueError("adaptive-epsilon row lacks a declared controller LR schedule")
    args = _config_namespace(payload.config)
    hps = build_hps(args)
    _initial_slots, runtime = build_adaptive_epsilon_native_initial_slots(
        run_spec=row_spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(int(getattr(args, "seed", 0))),
        lr_continuation_mode=declared_mode,
    )
    native = runtime.component("adaptive_epsilon")
    raw_optimizer_slot = _load_manifest_slot(
        source_manifest_path,
        source_manifest,
        OPTIMIZER,
    )
    optimizer_state = (
        deserialize_pytree_slot(
            raw_optimizer_slot.payload,
            native.optimizer_template,
            slot=OPTIMIZER,
        )
        if hasattr(raw_optimizer_slot, "payload")
        else raw_optimizer_slot
    )
    restored_count = optimizer_count_at_current_step(optimizer_state)
    context = lr_resume_context_for_mode(
        mode=declared_mode,
        completed_batches=completed_batches,
        optimizer_count_at_current_step=restored_count,
    )
    start_position = context.current_step - context.schedule_origin_step
    schedule_steps = lr_report_schedule_steps(
        payload.controller_optimizer.lr_schedule,
        start_position=start_position,
    )
    points = adaptive_epsilon_controller_lr_points(
        row_spec,
        hps,
        schedule_origin_step=context.schedule_origin_step,
        current_step=context.current_step,
        optimizer_count_at_current_step=context.optimizer_count_at_current_step,
        schedule_steps=schedule_steps,
    )
    _validate_lr_points(payload.controller_optimizer.lr_schedule, points)
    return [
        {
            **point,
            "mode": declared_mode,
            "completed_batches": completed_batches,
            "schedule_origin_step": context.schedule_origin_step,
            "current_step": context.current_step,
            "optimizer_count_at_current_step": context.optimizer_count_at_current_step,
        }
        for point in points
    ]


def _load_manifest_slot(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    slot_name: str,
) -> Any:
    for slot in manifest.get("slots", []):
        if not isinstance(slot, Mapping) or slot.get("slot") != slot_name:
            continue
        relative_path = slot.get("relative_path")
        if not isinstance(relative_path, str):
            break
        return pickle.loads((manifest_path.parent / relative_path).read_bytes())
    raise ValueError(f"checkpoint manifest lacks slot {slot_name!r}")


def _validate_lr_points(
    schedule_spec: Any,
    points: Sequence[Mapping[str, Any]],
) -> None:
    for point in points:
        step = int(point["step"])
        expected = float(
            jax.device_get(
                learning_rate_at_step(
                    schedule_spec,
                    current_step=step,
                    schedule_origin_step=0,
                )
            )
        )
        observed = float(point["lr"])
        tolerance = max(1e-12, abs(expected) * 1e-5)
        if abs(observed - expected) > tolerance:
            raise ForkParityError(
                "LR continuation report does not match declared schedule "
                f"at step={step}: observed={observed:.12g} expected={expected:.12g}"
            )


def _completed_batches(
    source_manifest: Mapping[str, Any],
    row_spec: TrainingRunSpec,
) -> int:
    coordinate = source_manifest.get("completed_coordinate")
    if isinstance(coordinate, Mapping):
        global_step = int(coordinate.get("global_step", 0))
    else:
        global_step = int(source_manifest.get("completed_training_batches", 0))
    if _method_ref_string(row_spec) != MINIMAX_METHOD_REF:
        return global_step
    config = minimax_training_run_spec_to_config(row_spec)
    barrier = coordinate.get("completed_barrier") if isinstance(coordinate, Mapping) else None
    phase = coordinate.get("phase") if isinstance(coordinate, Mapping) else None
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
    optimizer_spec = _declared_optimizer_spec(row_spec)
    if optimizer_spec is not None and optimizer_spec.lr_schedule is not None:
        return float(
            jax.device_get(
                learning_rate_at_step(
                    optimizer_spec.lr_schedule,
                    current_step=int(step),
                    schedule_origin_step=0,
                )
            )
        )
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
    training_config = row_payload.get("training_config")
    if isinstance(training_config, Mapping) and "learning_rate" in training_config:
        return float(training_config["learning_rate"])
    return 0.0


def _declared_optimizer_spec(row_spec: TrainingRunSpec) -> OptimizerSpec | None:
    payload = row_spec.method_payload.payload
    for key in ("controller_optimizer", "optimizer"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return OptimizerSpec.model_validate(candidate)
    return None


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
