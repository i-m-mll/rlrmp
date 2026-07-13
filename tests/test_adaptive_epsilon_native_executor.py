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
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    LrScheduleSpec,
    MethodPayloadEnvelope,
    OptimizerSpec,
    TrainingRunSpec,
)
from feedbax.training import ExecutionPreparationRequest

from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot, serialize_pytree_slot
from rlrmp.train.adaptive_epsilon_native import (
    ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_VERSION,
    AdaptiveEpsilonMethodPayload,
    AdaptiveEpsilonNativeRuntime,
    SerializedPyTreeSlot,
    _adaptive_state_from_slot,
    _adaptive_epsilon_train_chunk,
    _adaptive_state_slot,
    _deserialize_optimizer_slot_value,
    _guard_from_slot,
    _json_slot,
    attach_adaptive_epsilon_checkpoint_continuation,
    adaptive_epsilon_controller_lr_points,
    adaptive_epsilon_controller_optimizer_spec,
    build_adaptive_epsilon_native_initial_slots,
    ensure_adaptive_epsilon_training_method_registered,
    execute_adaptive_epsilon_training_run_spec_native,
    lr_resume_context_for_mode,
    optimizer_count_at_current_step,
)
from rlrmp.train.cs_nominal_gru import (
    AdaptiveEpsilonState,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    GradientDiagnosticsState,
    UpdateDiagnosticsState,
    _adaptive_epsilon_zero_guard_from_state,
    _config_namespace,
    _latest_loss_scalars,
    _run_adaptive_epsilon_training_chunk,
    _update_adaptive_epsilon_state,
    _update_adaptive_epsilon_zero_guard,
    build_hps,
    CsNominalGruConfig,
    write_run_spec,
)
from rlrmp.train.science_vocabulary import AdaptiveEpsilonControllerMode
from rlrmp.train.executor.equivalence import assert_paired_equivalent, run_paired_equivalence
from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.execution_preparation import prepare_adaptive_epsilon
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
        AdaptiveEpsilonControllerMode.LOSS_BLEND,
        AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER,
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
    assert payload.controller_optimizer is not None
    assert payload.controller_optimizer.lr_schedule is not None
    assert payload.controller_optimizer.lr_schedule.kind == "warmup_cosine"
    assert payload.controller_optimizer.lr_schedule.learning_rate_0 == pytest.approx(1e-3)
    assert payload.controller_optimizer.lr_schedule.constant_lr_iterations == 1
    assert "learning_rate" not in payload.controller_optimizer.params
    assert spec.method_payload.schema_version == ADAPTIVE_EPSILON_METHOD_PAYLOAD_SCHEMA_VERSION
    assert payload.damage_schedule["setpoint_basis"] == "damage_to_clean_loss_ratio"
    assert payload.n_train_batches == 2
    assert payload.chunk_batches == 1
    assert "resume_context" not in spec.metadata
    assert "optimizer_build_context" not in spec.metadata
    slot_names = {slot.name for slot in spec.worker_execution.method_contract.state_slots}
    assert {ADAPTIVE_EPSILON_STATE, ZERO_ADVERSARY_GUARD} <= slot_names


def test_adaptive_epsilon_execution_preparation_builds_runtime_inputs(tmp_path: Path) -> None:
    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
    )

    prepared = prepare_adaptive_epsilon(ExecutionPreparationRequest(run_spec=spec, resume=True))

    assert isinstance(prepared.initial_slots[MODEL], SerializedPyTreeSlot)
    assert RLRMP_RUNTIME_CONTEXT_KEY in prepared.kernel_context
    assert prepared.loss_service is not None
    assert callable(prepared.resume_slot_transform)


@pytest.mark.parametrize("additional_batches", [4_500, 1])
def test_adaptive_epsilon_execution_preparation_uses_segment_local_history_template(
    tmp_path: Path,
    additional_batches: int,
) -> None:
    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
        n_train_batches=16_500,
        n_replicates=5,
    )
    spec = attach_adaptive_epsilon_checkpoint_continuation(
        spec,
        source_completed_batches=12_000,
        target_total_batches=12_000 + additional_batches,
    )

    prepared = prepare_adaptive_epsilon(
        ExecutionPreparationRequest(run_spec=spec, resume=True)
    )
    runtime = prepared.kernel_context[RLRMP_RUNTIME_CONTEXT_KEY]
    native = runtime.component("adaptive_epsilon")
    optimizer_leaves = tuple(jt.leaves(native.optimizer_template))
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )

    assert isinstance(payload, AdaptiveEpsilonMethodPayload)
    assert payload.n_train_batches == 16_500
    assert native.args.n_train_batches == additional_batches
    assert optimizer_leaves[1].shape == (5, additional_batches)
    assert all(
        getattr(leaf, "shape", None) != (5, 16_500) for leaf in optimizer_leaves
    )


def test_adaptive_epsilon_rejects_old_absolute_setpoint_payload(
    tmp_path: Path,
) -> None:
    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
    )
    method_payload = spec.method_payload.model_dump(mode="json", exclude_none=True)
    method_payload["schema_version"] = (
        "rlrmp.spec.training_method.adaptive_epsilon_curriculum_payload.v1"
    )
    legacy_payload = MethodPayloadEnvelope.model_validate(method_payload)

    ensure_adaptive_epsilon_training_method_registered()

    with pytest.raises(ValueError, match="v1|absolute|rejected|payload"):
        DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
            spec.method_ref,
            legacy_payload,
            path="/method_payload",
        )


def test_adaptive_epsilon_payload_requires_ratio_setpoint_basis() -> None:
    with pytest.raises(ValueError, match="setpoint_basis"):
        AdaptiveEpsilonMethodPayload(
            config={"adaptive_epsilon_curriculum": True},
            n_train_batches=1,
            chunk_batches=1,
            controller_training_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
            damage_schedule={
                "kind": "linear_ramp_then_cosine_anneal",
                "start": 1.0,
                "peak": 1.0,
                "final": 1.0,
                "ramp_batches": 0,
                "anneal_batches": 0,
            },
            lambda_update={},
            outer_adversarial_weight={},
            application_ramp_start_batch=0,
            application_ramp_end_batch=0,
            pgd_inner_maximizer={},
            checkpointing={},
        )


def test_adaptive_epsilon_ratio_update_is_loss_scale_invariant() -> None:
    config = argparse.Namespace(
        lambda_update=argparse.Namespace(
            interval_batches=1,
            ema_alpha=1.0,
            eta=1.0,
            deadband_frac=0.0,
            lambda_min=1e-12,
            lambda_max=None,
            max_log_step=10.0,
            freeze_during_application_ramp=True,
        )
    )

    base_state, base_diagnostics = _update_adaptive_epsilon_state(
        AdaptiveEpsilonState(lambda_value=2.0),
        config,
        batch_index=0,
        target_damage=0.25,
        measured_damage=50.0,
        measured_clean_loss=100.0,
    )
    scaled_state, scaled_diagnostics = _update_adaptive_epsilon_state(
        AdaptiveEpsilonState(lambda_value=2.0),
        config,
        batch_index=0,
        target_damage=0.25,
        measured_damage=500.0,
        measured_clean_loss=1000.0,
    )

    assert base_state.lambda_value == pytest.approx(scaled_state.lambda_value)
    assert base_state.lambda_value == pytest.approx(4.0)
    assert base_state.damage_ema == pytest.approx(50.0)
    assert scaled_state.damage_ema == pytest.approx(500.0)
    assert base_state.clean_loss_ema == pytest.approx(100.0)
    assert scaled_state.clean_loss_ema == pytest.approx(1000.0)
    assert bool(base_diagnostics["application_ramp_frozen"]) is False
    assert bool(base_diagnostics["ema_seeded_post_ramp"]) is False
    assert float(base_diagnostics["damage_ratio_ema"]) == pytest.approx(0.5)
    assert float(scaled_diagnostics["damage_ratio_ema"]) == pytest.approx(0.5)
    assert float(base_diagnostics["target_damage_ratio"]) == pytest.approx(0.25)
    restored = _adaptive_state_from_slot(_adaptive_state_slot(base_state))
    assert restored.clean_loss_ema == pytest.approx(100.0)


def test_adaptive_epsilon_optimizer_spec_expresses_constant_lr() -> None:
    optimizer = adaptive_epsilon_controller_optimizer_spec(
        {
            "n_train_batches": 4,
            "hps": {
                "lr_schedule": "constant",
                "learning_rate_0": 3e-5,
                "n_batches_condition": 4,
                "weight_decay": 0.0,
            },
        }
    )

    assert optimizer.type == "adamw"
    assert optimizer.lr_schedule is not None
    assert optimizer.lr_schedule.kind == "constant"
    assert optimizer.lr_schedule.learning_rate_0 == pytest.approx(3e-5)


def test_serialized_optimizer_slot_requires_custody_extension_before_deserialize() -> None:
    """Serialized payloads cannot bypass Feedbax's declared-leaf gate."""

    source = _diagnostic_optimizer_state(2)
    template = _diagnostic_optimizer_state(4)
    slot = SerializedPyTreeSlot(serialize_pytree_slot(source))

    with pytest.raises(RuntimeError, match="on disk"):
        _deserialize_optimizer_slot_value(slot, template)


def test_adaptive_epsilon_continuation_declares_segment_length(
    tmp_path: Path,
) -> None:
    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER,
    )
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, AdaptiveEpsilonMethodPayload)
    outer_weight = dict(payload.outer_adversarial_weight)
    outer_weight["ramp_batches"] = 1_000
    payload = payload.model_copy(update={"outer_adversarial_weight": outer_weight})
    spec = spec.model_copy(
        update={
            "method_payload": spec.method_payload.model_copy(
                update={"payload": payload.model_dump(mode="json", exclude_none=True)}
            )
        }
    )

    attached = attach_adaptive_epsilon_checkpoint_continuation(
        spec,
        source_completed_batches=12_000,
        target_total_batches=16_500,
    )

    request = attached.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.additional_batches == 4_500
    assert request.target_total == 16_500
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        attached.method_ref,
        attached.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, AdaptiveEpsilonMethodPayload)
    assert payload.application_ramp_origin.kind == "segment_start"
    assert payload.application_ramp_start_batch == 12_000
    assert payload.application_ramp_end_batch == 13_000


def test_adaptive_epsilon_runtime_consumes_declared_constant_lr(
    tmp_path: Path,
) -> None:
    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
    )
    constant_optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(kind="constant", learning_rate_0=3e-5),
    )
    spec = _with_controller_optimizer(spec, constant_optimizer)
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, AdaptiveEpsilonMethodPayload)
    args = _config_namespace(payload.config)
    hps = build_hps(args)

    _slots, runtime = build_adaptive_epsilon_native_initial_slots(
        run_spec=spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(0),
    )
    native = runtime.component("adaptive_epsilon")
    assert isinstance(native, AdaptiveEpsilonNativeRuntime)

    assert _scheduled_learning_rate(native.optimizer_template) == pytest.approx(3e-5)


def test_adaptive_epsilon_optimizer_spec_expresses_rewarm_schedule() -> None:
    optimizer = adaptive_epsilon_controller_optimizer_spec(
        {
            "n_train_batches": 4500,
            "optimizer": {
                "name": "adamw",
                "lr_schedule": "warmup_cosine",
                "learning_rate_0": 3e-4,
                "lr_warmup_batches": 1000,
                "lr_warmup_init_fraction": 0.1,
                "lr_cosine_alpha": 0.1,
                "total_steps": 3500,
                "weight_decay": 0.0,
            },
        }
    )

    assert optimizer.lr_schedule is not None
    assert optimizer.lr_schedule.kind == "warmup_cosine"
    assert optimizer.lr_schedule.learning_rate_0 == pytest.approx(3e-4)
    assert optimizer.lr_schedule.constant_lr_iterations == 1000
    assert optimizer.lr_schedule.total_steps == 3500
    assert optimizer.lr_schedule.cosine_annealing_alpha == pytest.approx(0.1)


def test_adaptive_epsilon_runtime_shifts_restored_count_for_restart_vs_continue(
    tmp_path: Path,
) -> None:
    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
        n_train_batches=6,
    )
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=1e-3,
            total_steps=4,
            constant_lr_iterations=2,
            warmup_init_fraction=0.01,
            cosine_annealing_alpha=0.1,
        ),
    )
    restart_lr = _realized_lr_after_restored_count(
        _with_controller_optimizer(_with_lr_continuation_mode(spec, "restart"), optimizer),
        restored_count=9,
        completed_batches=5,
    )
    continue_lr = _realized_lr_after_restored_count(
        _with_controller_optimizer(_with_lr_continuation_mode(spec, "continue"), optimizer),
        restored_count=9,
        completed_batches=5,
    )

    assert restart_lr == pytest.approx(1e-5, rel=1e-4)
    assert continue_lr == pytest.approx(1e-4, rel=1e-4)


def test_adaptive_epsilon_executor_rejects_outer_recipe_mapping(tmp_path: Path) -> None:
    """The executor must not lose typed LR continuation fields at the recipe boundary."""

    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
    )
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, AdaptiveEpsilonMethodPayload)
    args = _config_namespace(payload.config)

    with pytest.raises(TypeError, match="extracted typed TrainingRunSpec"):
        build_adaptive_epsilon_native_initial_slots(
            run_spec={"feedbax_training_run_spec": spec.model_dump(mode="json")},  # type: ignore[arg-type]
            hps=build_hps(args),
            args=args,
            key=jr.PRNGKey(3),
        )


@pytest.mark.parametrize("peak_lr", [3e-4, 3e-3])
def test_adaptive_epsilon_restart_realized_optimizer_matches_rewarm_points(
    tmp_path: Path,
    peak_lr: float,
) -> None:
    """A carried 12k optimizer count realizes the continuation-local rewarm curve."""

    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
        n_train_batches=16_500,
    )
    optimizer = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=peak_lr,
            total_steps=3_500,
            constant_lr_iterations=1_000,
            warmup_init_fraction=3e-5 / peak_lr,
            cosine_annealing_alpha=3e-5 / peak_lr,
        ),
    )
    spec = _with_controller_optimizer(_with_lr_continuation_mode(spec, "restart"), optimizer)
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, AdaptiveEpsilonMethodPayload)
    hps = build_hps(_config_namespace(payload.config))
    context = lr_resume_context_for_mode(
        mode="restart",
        completed_batches=12_000,
        optimizer_count_at_current_step=12_000,
    )

    points = adaptive_epsilon_controller_lr_points(
        spec,
        hps=hps,
        schedule_origin_step=context.schedule_origin_step,
        current_step=context.current_step,
        optimizer_count_at_current_step=context.optimizer_count_at_current_step,
        schedule_steps=[0, 500, 1_000, 2_250, 3_500],
    )
    realized = [point["lr"] for point in points]

    assert [point["optimizer_count"] for point in points] == [
        12_000,
        12_500,
        13_000,
        14_250,
        15_500,
    ]
    assert realized[0] == pytest.approx(3e-5, rel=1e-5)
    assert 3e-5 < realized[1] < peak_lr
    assert realized[2] == pytest.approx(peak_lr, rel=1e-5)
    assert 3e-5 < realized[3] < peak_lr
    assert realized[4] == pytest.approx(3e-5, rel=1e-5)


def test_adaptive_epsilon_native_executor_emits_batch_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _adaptive_epsilon_training_spec(
        tmp_path,
        controller_mode=AdaptiveEpsilonControllerMode.LOSS_BLEND,
        n_train_batches=2,
    )
    spec = _with_payload_config(spec, disable_progress=False, quiet_progress=False)

    execute_adaptive_epsilon_training_run_spec_native(
        spec,
        run_id="adaptive-epsilon-progress",
        key=jr.PRNGKey(7),
        manifest_root=tmp_path / "manifests" / "progress",
        checkpoint_root=tmp_path / "checkpoints" / "progress",
        manifest_conflict_policy="reuse-identical",
    )

    progress_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("BATCH ")
    ]
    assert progress_lines
    assert progress_lines[0].startswith("BATCH phase=adaptive_epsilon batch=0/2")
    assert "loss=" in progress_lines[0]


@pytest.mark.parametrize(
    "controller_mode",
    [
        AdaptiveEpsilonControllerMode.LOSS_BLEND,
        AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER,
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
        controller_mode=AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER,
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
    **overrides: Any,
) -> TrainingRunSpec:
    args = _adaptive_epsilon_args(
        output_dir=str(tmp_path / controller_mode / "bulk"),
        spec_dir=str(tmp_path / controller_mode / "spec"),
        adaptive_epsilon_controller_training_mode=controller_mode,
        **overrides,
    )
    payload = write_run_spec(args)["run_spec"]
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def _adaptive_epsilon_args(**overrides: Any) -> argparse.Namespace:
    args = _config_namespace(
        CsNominalGruConfig(issue="test", output_dir="_artifacts/test/runs/test")
    )
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


def _with_controller_optimizer(
    spec: TrainingRunSpec,
    optimizer: OptimizerSpec,
) -> TrainingRunSpec:
    method_payload = spec.method_payload.model_dump(mode="json", exclude_none=True)
    payload = dict(method_payload["payload"])
    payload["controller_optimizer"] = optimizer.model_dump(mode="json", exclude_none=True)
    method_payload["payload"] = payload
    return spec.model_copy(
        update={"method_payload": MethodPayloadEnvelope.model_validate(method_payload)}
    )


def _with_lr_continuation_mode(
    spec: TrainingRunSpec,
    mode: str,
) -> TrainingRunSpec:
    method_payload = spec.method_payload.model_dump(mode="json", exclude_none=True)
    payload = dict(method_payload["payload"])
    payload["lr_continuation_mode"] = mode
    method_payload["payload"] = payload
    return spec.model_copy(
        update={"method_payload": MethodPayloadEnvelope.model_validate(method_payload)}
    )


def _with_payload_config(spec: TrainingRunSpec, **updates: Any) -> TrainingRunSpec:
    method_payload = spec.method_payload.model_dump(mode="json", exclude_none=True)
    payload = dict(method_payload["payload"])
    config = dict(payload["config"])
    config.update(updates)
    payload["config"] = config
    method_payload["payload"] = payload
    return spec.model_copy(
        update={"method_payload": MethodPayloadEnvelope.model_validate(method_payload)}
    )


def _scheduled_learning_rate(opt_state: Any) -> float:
    for leaf in jt.leaves(opt_state, is_leaf=_is_injected_hyperparams_state):
        if _is_injected_hyperparams_state(leaf):
            return float(jnp.asarray(leaf.hyperparams["learning_rate"]).reshape(-1)[0])
    raise AssertionError("optimizer state does not contain injected learning rate")


def _is_injected_hyperparams_state(value: Any) -> bool:
    fields = getattr(value, "_fields", ())
    return {"hyperparams", "inner_state"}.issubset(set(fields))


def _with_injected_count(opt_state: Any, count: int) -> Any:
    def replace_learning_rate_count(hyperparams_states: Any) -> Any:
        if not isinstance(hyperparams_states, Mapping):
            return hyperparams_states
        learning_rate_state = hyperparams_states.get("learning_rate")
        if learning_rate_state is None or not hasattr(learning_rate_state, "count"):
            return hyperparams_states
        return {
            **hyperparams_states,
            "learning_rate": learning_rate_state._replace(
                count=jnp.full_like(learning_rate_state.count, count)
            ),
        }

    def replace_count(leaf: Any) -> Any:
        if _is_injected_hyperparams_state(leaf):
            return leaf._replace(
                count=jnp.full_like(leaf.count, count),
                hyperparams_states=replace_learning_rate_count(leaf.hyperparams_states),
            )
        return leaf

    return jt.map(replace_count, opt_state, is_leaf=_is_injected_hyperparams_state)


def _realized_lr_after_restored_count(
    spec: TrainingRunSpec,
    *,
    restored_count: int,
    completed_batches: int,
) -> float:
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
        key=jr.PRNGKey(3),
    )
    native = runtime.component("adaptive_epsilon")
    assert isinstance(native, AdaptiveEpsilonNativeRuntime)
    optimizer_state = _with_injected_count(
        _deserialize_slot(slots[OPTIMIZER], native.optimizer_template, slot=OPTIMIZER),
        restored_count,
    )
    assert optimizer_count_at_current_step(optimizer_state) == restored_count
    chunk_slots = dict(slots)
    chunk_slots[OPTIMIZER] = SerializedPyTreeSlot(serialize_pytree_slot(optimizer_state))
    chunk_slots[COMPLETED_BATCHES] = jnp.asarray(completed_batches, dtype=jnp.int32)

    result = _adaptive_epsilon_train_chunk(runtime, payload, chunk_slots, coordinate=None)

    assert native.trainer_resume_context is not None
    optimizer_state = _deserialize_slot(
        result[OPTIMIZER],
        native.optimizer_template,
        slot=OPTIMIZER,
    )
    return _scheduled_learning_rate(optimizer_state)


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


def _diagnostic_optimizer_state(n_batches: int) -> dict[str, object]:
    return {
        "step": jnp.asarray(2, dtype=jnp.int32),
        "gradient": GradientDiagnosticsState(
            count=jnp.asarray(2, dtype=jnp.int32),
            gradient_norm_pre_clip=jnp.arange(n_batches, dtype=jnp.float32),
            gradient_clipped=jnp.asarray(
                [index == 0 for index in range(n_batches)],
                dtype=bool,
            ),
            learning_rate=jnp.arange(n_batches, dtype=jnp.float32) + 0.5,
        ),
        "update": UpdateDiagnosticsState(
            count=jnp.asarray(2, dtype=jnp.int32),
            update_norm=jnp.arange(n_batches, dtype=jnp.float32) + 1.0,
            parameter_norm=jnp.arange(n_batches, dtype=jnp.float32) + 2.0,
            update_parameter_norm_ratio=jnp.arange(n_batches, dtype=jnp.float32) + 3.0,
        ),
    }


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
        "clean_loss_ema": (
            -1.0 if payload["clean_loss_ema"] is None else payload["clean_loss_ema"]
        ),
        "last_update_batch": (
            -1 if payload["last_update_batch"] is None else payload["last_update_batch"]
        ),
        "update_count": payload["update_count"],
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
