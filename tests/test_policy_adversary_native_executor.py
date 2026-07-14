"""Policy-adversary supervised native-executor equivalence tests."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY, TrainingRunSpec
from feedbax.training import ExecutionPreparationRequest
from rlrmp.data_products.broad_epsilon import load_pgd_radius_source

from rlrmp.train.cs_nominal_gru import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CsNominalGruConfig,
    _config_namespace,
    _latest_loss_scalars,
    _run_policy_adversary_training_chunk,
    build_hps,
    write_run_spec,
)
from rlrmp.train.executor.equivalence import assert_paired_equivalent, run_paired_equivalence
from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.execution_preparation import prepare_policy_adversary
from rlrmp.train.executor.slots import (
    ADVERSARY_LOSS,
    ADVERSARY_OPTIMIZER,
    ADVERSARY_POLICY,
    COMPLETED_BATCHES,
    MODEL,
    OPTIMIZER,
    POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
    PRNG,
    TRAIN_LOSS,
)
from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot, serialize_pytree_slot
from rlrmp.train.policy_adversary_native import (
    PolicyAdversaryMethodPayload,
    PolicyAdversaryNativeRuntime,
    SerializedPyTreeSlot,
    build_policy_adversary_native_initial_slots,
    ensure_policy_adversary_training_method_registered,
    execute_policy_adversary_training_run_spec_native,
)

HISTORICAL_020A65B_PGD_RADIUS_15CM = float(
    load_pgd_radius_source("effective_020a65b_pgd_training_radius")["l2_radius_15cm"]
)


def test_policy_adversary_run_spec_uses_native_method(tmp_path: Path) -> None:
    spec = _policy_adversary_training_spec(tmp_path)

    ensure_policy_adversary_training_method_registered()

    assert spec.method_ref.key == POLICY_ADVERSARY_SUPERVISED_METHOD_REF
    assert POLICY_ADVERSARY_SUPERVISED_METHOD_REF in (
        DEFAULT_TRAINING_METHOD_REGISTRY.available_keys()
    )
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, PolicyAdversaryMethodPayload)
    assert payload.n_train_batches == 2
    assert payload.chunk_batches == 1
    assert payload.policy["kind"] == "memoryless_mlp"
    assert "resume_context" not in spec.metadata
    assert "optimizer_build_context" not in spec.metadata
    slot_names = {slot.name for slot in spec.worker_execution.method_contract.state_slots}
    assert {ADVERSARY_POLICY, ADVERSARY_OPTIMIZER} <= slot_names


def test_policy_adversary_execution_preparation_builds_runtime_inputs(tmp_path: Path) -> None:
    spec = _policy_adversary_training_spec(tmp_path)

    prepared = prepare_policy_adversary(ExecutionPreparationRequest(run_spec=spec, resume=True))

    assert isinstance(prepared.initial_slots[MODEL], SerializedPyTreeSlot)
    assert RLRMP_RUNTIME_CONTEXT_KEY in prepared.kernel_context
    assert prepared.loss_service is not None
    assert callable(prepared.resume_slot_transform)


def test_policy_adversary_native_executor_matches_driver_chunk_loop(
    tmp_path: Path,
) -> None:
    spec = _policy_adversary_training_spec(tmp_path)
    key = jr.PRNGKey(0)
    legacy_slots, legacy_runtime = _legacy_policy_adversary_chunk_loop(spec, key=key)

    result = execute_policy_adversary_training_run_spec_native(
        spec,
        run_id="policy-adversary-native-fixed-seed",
        key=key,
        manifest_root=tmp_path / "manifests" / "executor",
        checkpoint_root=tmp_path / "checkpoints" / "executor",
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        "policy_adversary.driver",
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
    assert result.final_slots[ADVERSARY_LOSS] != 0.0


def test_policy_adversary_native_executor_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    spec = _policy_adversary_training_spec(tmp_path)
    key = jr.PRNGKey(1)
    _, template_runtime = _legacy_policy_adversary_chunk_loop(spec, key=key)
    full = execute_policy_adversary_training_run_spec_native(
        spec,
        run_id="policy-adversary-native-full",
        key=key,
        manifest_root=tmp_path / "manifests" / "full",
        checkpoint_root=tmp_path / "checkpoints" / "full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "resume"
    partial = execute_policy_adversary_training_run_spec_native(
        spec,
        run_id="policy-adversary-native-resume",
        key=key,
        manifest_root=tmp_path / "manifests" / "resume-partial",
        checkpoint_root=checkpoint_root,
        stop_after_barrier="after_policy_adversary_train_chunk",
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_policy_adversary_training_run_spec_native(
        spec,
        run_id="policy-adversary-native-resume",
        key=key,
        manifest_root=tmp_path / "manifests" / "resume-final",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        "policy_adversary.resume",
        lambda: full.final_slots,
        lambda: resumed.final_slots,
        comparable=lambda slots: _comparable_slots(slots, template_runtime),
        left_label="uninterrupted",
        right_label="resumed",
    )
    assert partial.final_coordinate.completed_barrier == "after_policy_adversary_train_chunk"
    assert_paired_equivalent(report)


def _policy_adversary_training_spec(tmp_path: Path) -> TrainingRunSpec:
    args = _policy_adversary_args(
        output_dir=str(tmp_path / "bulk"),
        spec_dir=str(tmp_path / "spec"),
    )
    payload = write_run_spec(args)["run_spec"]
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def _policy_adversary_args(**overrides: Any) -> argparse.Namespace:
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
        "lr_warmup_batches": 1,
        "lr_warmup_init_fraction": 0.1,
        "lr_cosine_alpha": 0.01,
        "log_step": 1,
        "disable_progress": True,
        "quiet_progress": True,
        "target_relative_multitarget": True,
        "force_filter_feedback": True,
        "initial_hidden_encoder": True,
        "perturbation_training": True,
        "perturbation_calibrated_timing": True,
        "perturbation_physical_level": "small",
        "policy_adversary_training": True,
        "policy_adversary_steps": 1,
        "policy_adversary_width": 4,
        "policy_adversary_radius_15cm": HISTORICAL_020A65B_PGD_RADIUS_15CM,
        "policy_adversary_radius_source": "effective_020a65b_pgd_training_radius",
        "broad_epsilon_reach_scaling": True,
        "loss_objective": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    }
    for key, value in {**defaults, **overrides}.items():
        setattr(args, key, value)
    return args


def _legacy_policy_adversary_chunk_loop(
    spec: TrainingRunSpec,
    *,
    key: Any,
) -> tuple[dict[str, Any], PolicyAdversaryNativeRuntime]:
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    assert isinstance(payload, PolicyAdversaryMethodPayload)
    args = _config_namespace(payload.config)
    hps = build_hps(args)
    slots, runtime = build_policy_adversary_native_initial_slots(
        run_spec=spec,
        hps=hps,
        args=args,
        key=key,
    )
    native = runtime.component("policy_adversary")
    assert isinstance(native, PolicyAdversaryNativeRuntime)
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
        adversary_policy = deserialize_pytree_slot(
            _slot_payload(slots[ADVERSARY_POLICY]),
            native.adversary_policy_template,
            slot=ADVERSARY_POLICY,
        )
        adversary_optimizer_state = deserialize_pytree_slot(
            _slot_payload(slots[ADVERSARY_OPTIMIZER]),
            native.adversary_optimizer_template,
            slot=ADVERSARY_OPTIMIZER,
        )
        (
            model,
            history_chunk,
            optimizer_state,
            adversary_policy,
            adversary_optimizer_state,
            diagnostics,
        ) = _run_policy_adversary_training_chunk(
            trainer=native.trainer,
            task=native.task,
            model=model,
            optimizer_state=optimizer_state,
            adversary_policy=adversary_policy,
            adversary_optimizer_state=adversary_optimizer_state,
            adversary_optimizer=native.adversary_optimizer,
            hps=native.hps,
            where_train=native.where_train,
            key=key_chunk,
            start_batch=completed,
            chunk_batches=chunk_batches,
            log_progress=False,
        )
        slots.update(
            {
                MODEL: SerializedPyTreeSlot(serialize_pytree_slot(model)),
                OPTIMIZER: SerializedPyTreeSlot(serialize_pytree_slot(optimizer_state)),
                PRNG: key_next,
                COMPLETED_BATCHES: jnp.asarray(completed + chunk_batches, dtype=jnp.int32),
                ADVERSARY_POLICY: SerializedPyTreeSlot(serialize_pytree_slot(adversary_policy)),
                ADVERSARY_OPTIMIZER: SerializedPyTreeSlot(
                    serialize_pytree_slot(adversary_optimizer_state)
                ),
                TRAIN_LOSS: float(_latest_loss_scalars(history_chunk, chunk_batches=chunk_batches)[
                    "total"
                ]),
                ADVERSARY_LOSS: float(jnp.asarray(diagnostics["adversary_objective"])),
            }
        )
    return slots, native


def _comparable_slots(
    slots: Mapping[str, Any],
    runtime: PolicyAdversaryNativeRuntime,
) -> dict[str, Any]:
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
        ADVERSARY_POLICY: _array_leaves(
            deserialize_pytree_slot(
                _slot_payload(slots[ADVERSARY_POLICY]),
                runtime.adversary_policy_template,
                slot=ADVERSARY_POLICY,
            )
        ),
        ADVERSARY_OPTIMIZER: _array_leaves(
            deserialize_pytree_slot(
                _slot_payload(slots[ADVERSARY_OPTIMIZER]),
                runtime.adversary_optimizer_template,
                slot=ADVERSARY_OPTIMIZER,
            )
        ),
        TRAIN_LOSS: slots[TRAIN_LOSS],
        ADVERSARY_LOSS: slots[ADVERSARY_LOSS],
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
