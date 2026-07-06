"""Adaptive-epsilon curriculum native-executor equivalence tests."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY, TrainingRunSpec

from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot, serialize_pytree_slot
from rlrmp.train.adaptive_epsilon_native import (
    AdaptiveEpsilonMethodPayload,
    AdaptiveEpsilonNativeRuntime,
    SerializedPyTreeSlot,
    _adaptive_state_from_slot,
    _adaptive_state_slot,
    _guard_from_slot,
    _json_slot,
    build_adaptive_epsilon_native_initial_slots,
    ensure_adaptive_epsilon_training_method_registered,
    execute_adaptive_epsilon_training_run_spec_native,
)
from rlrmp.train.cs_nominal_gru import (
    ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
    ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    _adaptive_epsilon_zero_guard_from_state,
    _config_namespace,
    _latest_loss_scalars,
    _run_adaptive_epsilon_training_chunk,
    _update_adaptive_epsilon_zero_guard,
    build_hps,
    build_parser,
    write_run_spec,
)
from rlrmp.train.executor.equivalence import assert_paired_equivalent, run_paired_equivalence
from rlrmp.train.executor.slots import (
    ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    ADAPTIVE_EPSILON_STATE,
    COMPLETED_BATCHES,
    DAMAGE_METRIC,
    EPSILON_SCALE,
    MODEL,
    OPTIMIZER,
    PRNG,
    TRAIN_LOSS,
    ZERO_ADVERSARY_GUARD,
)


@pytest.mark.parametrize(
    "controller_mode",
    [
        ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
        ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
    ],
)
def test_adaptive_epsilon_run_spec_uses_native_method(
    tmp_path: Path,
    controller_mode: str,
) -> None:
    spec = _adaptive_epsilon_training_spec(tmp_path, controller_mode=controller_mode)

    ensure_adaptive_epsilon_training_method_registered()

    assert spec.method_ref.key == ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF
    assert ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF in (
        DEFAULT_TRAINING_METHOD_REGISTRY.available_keys()
    )
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, AdaptiveEpsilonMethodPayload)
    assert payload.controller_training_mode == controller_mode
    assert payload.n_train_batches == 2
    assert payload.chunk_batches == 1
    slot_names = {slot.name for slot in spec.worker_execution.method_contract.state_slots}
    assert {ADAPTIVE_EPSILON_STATE, ZERO_ADVERSARY_GUARD} <= slot_names


@pytest.mark.parametrize(
    "controller_mode",
    [
        ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
        ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
    ],
)
def test_adaptive_epsilon_native_executor_matches_driver_chunk_loop(
    tmp_path: Path,
    controller_mode: str,
) -> None:
    spec = _adaptive_epsilon_training_spec(tmp_path, controller_mode=controller_mode)
    key = jr.PRNGKey(0)
    legacy_slots, legacy_runtime = _legacy_adaptive_epsilon_chunk_loop(spec, key=key)

    result = execute_adaptive_epsilon_training_run_spec_native(
        spec,
        run_id=f"adaptive-epsilon-native-fixed-seed-{controller_mode}",
        key=key,
        manifest_root=tmp_path / "manifests" / "executor" / controller_mode,
        checkpoint_root=tmp_path / "checkpoints" / "executor" / controller_mode,
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        f"adaptive_epsilon.{controller_mode}.driver",
        lambda: legacy_slots,
        lambda: result.final_slots,
        comparable=lambda slots: _comparable_slots(slots, legacy_runtime),
        left_label="driver_chunk_loop",
        right_label="native_executor",
    )
    assert_paired_equivalent(report)
    assert int(result.final_slots[COMPLETED_BATCHES]) == 2
    assert result.final_coordinate.phase == "done"
    assert result.final_slots[TRAIN_LOSS] != 0.0


def test_adaptive_epsilon_native_executor_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
    )
    key = jr.PRNGKey(1)
    _, template_runtime = _legacy_adaptive_epsilon_chunk_loop(spec, key=key)
    full = execute_adaptive_epsilon_training_run_spec_native(
        spec,
        run_id="adaptive-epsilon-native-full",
        key=key,
        manifest_root=tmp_path / "manifests" / "full",
        checkpoint_root=tmp_path / "checkpoints" / "full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "resume"
    partial = execute_adaptive_epsilon_training_run_spec_native(
        spec,
        run_id="adaptive-epsilon-native-resume",
        key=key,
        manifest_root=tmp_path / "manifests" / "resume-partial",
        checkpoint_root=checkpoint_root,
        stop_after_barrier="after_adaptive_epsilon_train_chunk",
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_adaptive_epsilon_training_run_spec_native(
        spec,
        run_id="adaptive-epsilon-native-resume",
        key=key,
        manifest_root=tmp_path / "manifests" / "resume-final",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        "adaptive_epsilon.resume",
        lambda: full.final_slots,
        lambda: resumed.final_slots,
        comparable=lambda slots: _comparable_slots(slots, template_runtime),
        left_label="uninterrupted",
        right_label="resumed",
    )
    assert partial.final_coordinate.completed_barrier == "after_adaptive_epsilon_train_chunk"
    assert_paired_equivalent(report)


def _adaptive_epsilon_training_spec(
    tmp_path: Path,
    *,
    controller_mode: str,
) -> TrainingRunSpec:
    args = _adaptive_epsilon_args(
        output_dir=str(tmp_path / controller_mode / "bulk"),
        spec_dir=str(tmp_path / controller_mode / "spec"),
        adaptive_epsilon_controller_training_mode=controller_mode,
    )
    payload = write_run_spec(args)["run_spec"]
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def _adaptive_epsilon_args(**overrides: Any) -> argparse.Namespace:
    args = build_parser().parse_args([])
    defaults = {
        "n_train_batches": 2,
        "batch_size": 1,
        "n_replicates": 1,
        "hidden_size": 4,
        "dry_run": True,
        "full_train": True,
        "resume": True,
        "checkpoint_interval_batches": 1,
        "controller_lr": 1e-3,
        "gradient_clip_norm": 5.0,
        "lr_warmup_batches": 1,
        "lr_warmup_init_fraction": 0.1,
        "lr_cosine_alpha": 0.01,
        "log_step": 1,
        "disable_progress": True,
        "quiet_progress": True,
        "target_relative_multitarget": True,
        "force_filter_feedback": True,
        "broad_epsilon_pgd_training": True,
        "broad_epsilon_pgd_steps": 1,
        "broad_epsilon_pgd_step_size_fraction": 0.5,
        "broad_epsilon_pgd_objective": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        "broad_epsilon_pgd_energy_lambda": 1.0,
        "adaptive_epsilon_curriculum": True,
        "adaptive_epsilon_update_interval_batches": 1,
        "adaptive_epsilon_outer_weight_start": 0.25,
        "adaptive_epsilon_outer_weight_final": 0.25,
        "adaptive_epsilon_outer_weight_ramp_batches": 0,
        "loss_objective": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    }
    for key, value in {**defaults, **overrides}.items():
        setattr(args, key, value)
    return args


def _legacy_adaptive_epsilon_chunk_loop(
    spec: TrainingRunSpec,
    *,
    key: Any,
) -> tuple[dict[str, Any], AdaptiveEpsilonNativeRuntime]:
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, AdaptiveEpsilonMethodPayload)
    args = _config_namespace(payload.config)
    hps = build_hps(args)
    slots, runtime = build_adaptive_epsilon_native_initial_slots(
        run_spec=spec,
        hps=hps,
        args=args,
        key=key,
    )
    native = runtime.component("adaptive_epsilon")
    assert isinstance(native, AdaptiveEpsilonNativeRuntime)
    while int(slots[COMPLETED_BATCHES]) < payload.n_train_batches:
        completed = int(slots[COMPLETED_BATCHES])
        chunk_batches = min(payload.chunk_batches, payload.n_train_batches - completed)
        key_chunk, key_next = jr.split(slots[PRNG], 2)
        model = _deserialize_slot(slots[MODEL], native.model_template, slot=MODEL)
        optimizer_state = deserialize_pytree_slot(
            _slot_payload(slots[OPTIMIZER]),
            native.optimizer_template,
            slot=OPTIMIZER,
        )
        adaptive_state = _adaptive_state_from_slot(slots[ADAPTIVE_EPSILON_STATE])
        guard = _guard_from_slot(slots[ZERO_ADVERSARY_GUARD])
        if adaptive_state is not None and not guard:
            guard = _adaptive_epsilon_zero_guard_from_state(adaptive_state, enabled=True)
        (
            model,
            history_chunk,
            optimizer_state,
            adaptive_state,
            diagnostics,
        ) = _run_adaptive_epsilon_training_chunk(
            trainer=native.trainer,
            task=native.task,
            model=model,
            optimizer_state=optimizer_state,
            adaptive_state=adaptive_state,
            hps=native.hps,
            where_train=native.where_train,
            key=key_chunk,
            start_batch=completed,
            chunk_batches=chunk_batches,
            log_progress=False,
        )
        guard = _update_adaptive_epsilon_zero_guard(guard, diagnostics)
        adaptive_state = replace(adaptive_state, zero_adversary_guard=guard)
        slots.update(
            {
                MODEL: SerializedPyTreeSlot(serialize_pytree_slot(model)),
                OPTIMIZER: SerializedPyTreeSlot(serialize_pytree_slot(optimizer_state)),
                PRNG: key_next,
                COMPLETED_BATCHES: jnp.asarray(completed + chunk_batches, dtype=jnp.int32),
                ADAPTIVE_EPSILON_STATE: _adaptive_state_slot(adaptive_state),
                ZERO_ADVERSARY_GUARD: _json_slot(guard),
                TRAIN_LOSS: float(_latest_loss_scalars(history_chunk, chunk_batches=chunk_batches)[
                    "total"
                ]),
                DAMAGE_METRIC: _diagnostic_scalar(
                    diagnostics,
                    "adaptive_epsilon_measured_damage",
                ),
                EPSILON_SCALE: _diagnostic_scalar(
                    diagnostics,
                    "adaptive_epsilon_epsilon_scale_used",
                ),
            }
        )
    return slots, native


def _comparable_slots(
    slots: Mapping[str, Any],
    runtime: AdaptiveEpsilonNativeRuntime,
) -> dict[str, Any]:
    state = _adaptive_state_from_slot(slots[ADAPTIVE_EPSILON_STATE])
    return {
        MODEL: _array_leaves(_deserialize_slot(slots[MODEL], runtime.model_template, slot=MODEL)),
        OPTIMIZER: _array_leaves(
            deserialize_pytree_slot(
                _slot_payload(slots[OPTIMIZER]),
                runtime.optimizer_template,
                slot=OPTIMIZER,
            )
        ),
        PRNG: slots[PRNG],
        COMPLETED_BATCHES: slots[COMPLETED_BATCHES],
        ADAPTIVE_EPSILON_STATE: _numeric_state_payload(state),
        ZERO_ADVERSARY_GUARD: _numeric_guard_payload(
            _guard_from_slot(slots[ZERO_ADVERSARY_GUARD])
        ),
        TRAIN_LOSS: slots[TRAIN_LOSS],
        DAMAGE_METRIC: slots[DAMAGE_METRIC],
        EPSILON_SCALE: slots[EPSILON_SCALE],
    }


def _deserialize_slot(value: Any, template: Any, *, slot: str) -> Any:
    return deserialize_pytree_slot(_slot_payload(value), template, slot=slot)


def _slot_payload(value: Any) -> bytes:
    return value.payload if isinstance(value, SerializedPyTreeSlot) else value


def _array_leaves(value: Any) -> Any:
    return jt.map(_cast_bool_array, eqx.filter(value, eqx.is_array))


def _cast_bool_array(value: Any) -> Any:
    if getattr(getattr(value, "dtype", None), "kind", None) == "b":
        return value.astype(jnp.int8)
    return value


def _numeric_state_payload(state: Any) -> dict[str, Any]:
    if state is None:
        return {}
    payload = state.to_json()
    return {
        "lambda_value": payload["lambda_value"],
        "damage_ema": -1.0 if payload["damage_ema"] is None else payload["damage_ema"],
        "last_update_batch": (
            -1 if payload["last_update_batch"] is None else payload["last_update_batch"]
        ),
        "update_count": payload["update_count"],
        "schedule_start_batch": payload["schedule_start_batch"],
        "zero_adversary_guard": _numeric_guard_payload(
            payload.get("zero_adversary_guard", {})
        ),
    }


def _numeric_guard_payload(guard: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(guard.get("enabled", False)),
        "consecutive_active_zero_checkpoints": int(
            guard.get("consecutive_active_zero_checkpoints", 0)
        ),
        "should_stop": bool(guard.get("should_stop", False)),
    }


def _diagnostic_scalar(diagnostics: Mapping[str, Any], key: str) -> float:
    value = diagnostics.get(key)
    if value is None:
        return 0.0
    array = jnp.asarray(value)
    return float(array.reshape(-1)[0])
