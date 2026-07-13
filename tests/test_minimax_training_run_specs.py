"""Spec-first minimax training contracts."""

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest

from feedbax.contracts.training import TrainingRunSpec
from jax_cookbook import save as fbx_save
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.model.feedbax_graph import build_rlrmp_feedbax_graph_bundle
from rlrmp.train.executor.equivalence import compare_pytrees
from rlrmp.train.minimax_native import (
    MINIMAX_METHOD_REF,
    MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
    MinimaxConfig,
    build_hps,
    build_minimax_training_run_spec,
    build_minimax_native_initial_slots,
    execute_minimax_training_run_spec_native,
    minimax_training_run_spec_from_file,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
)
from rlrmp.train.minimax_native import (
    _controller_state_from_model,
    _prepare_adversarial_batch,
    _vmapped_controller_descent,
    _vmapped_gaussian_adversary_ascent,
    _vmapped_linear_adversary_ascent,
)
from rlrmp.train.task_model import build_task_base


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _config(**overrides: Any) -> dict[str, Any]:
    return MinimaxConfig.model_validate(overrides).model_dump(mode="python")


def _payload_from_config(tmp_path: Path, config: dict) -> dict:
    args = MinimaxConfig.model_validate(config)
    hps = build_hps(args)
    graph_bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
        key=jr.PRNGKey(args.seed),
    )
    return build_minimax_training_run_spec(
        config,
        graph_spec=graph_bundle.graph_spec,
        output_dir=Path(args.output_dir),
        spec_dir=tmp_path / "results" / "54b0c2e" / "runs" / "spec",
        feedbax_graph={"graph_spec_path": "graph_spec.json", "manifest_path": "manifest.json"},
    )


def _payload(tmp_path: Path, **overrides: Any) -> dict:
    config = _config(
        **{
            "n_warmup_batches": 1,
            "n_adversary_batches": 0,
            "batch_size": 1,
            "n_replicates": 1,
            "output_dir": str(tmp_path / "_artifacts" / "54b0c2e" / "runs" / "spec"),
            **overrides,
        }
    )
    return _payload_from_config(tmp_path, config)


def _native_smoke_spec(
    tmp_path: Path,
    *,
    adversary_type: str = "gaussian_bump",
    n_adversary_batches: int = 1,
) -> TrainingRunSpec:
    config = _config(
        adversary_type=adversary_type,
        n_warmup_batches=0,
        n_adversary_batches=n_adversary_batches,
        n_adversary_steps=1,
        batch_size=1,
        adv_batch_size=1,
        n_replicates=1,
        output_dir=str(
            tmp_path / "_artifacts" / "62a658d" / "runs" / f"native_smoke_{adversary_type}"
        ),
    )
    payload = _payload_from_config(tmp_path, config)
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def _bool_leaves_as_ints(tree: Any) -> Any:
    return jt.map(
        lambda leaf: (
            leaf.astype(jnp.int8)
            if getattr(getattr(leaf, "dtype", None), "kind", None) == "b"
            else leaf
        ),
        tree,
    )


def test_warmup_only_minimax_config_round_trips_to_training_run_spec(tmp_path: Path) -> None:
    payload = _payload(tmp_path)

    validate_minimax_run_spec(payload, spec_dir=tmp_path)
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])

    assert spec.method_ref.key == MINIMAX_METHOD_REF
    assert spec.worker_execution.effective_phase.phase_program.initial_phase == "warmup"
    assert minimax_training_run_spec_to_config(spec)["n_adversary_batches"] == 0
    assert payload["schema_version"] == "rlrmp.minimax.native_run_spec.v1"


def test_gaussian_bump_minimax_config_round_trips_to_training_run_spec(tmp_path: Path) -> None:
    payload = _payload(tmp_path, n_adversary_batches=2, n_bumps=2, force_max=0.5)
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    config = minimax_training_run_spec_to_config(spec)

    assert config["adversary_type"] == "gaussian_bump"
    assert config["n_bumps"] == 2
    assert spec.method_payload.payload["config"]["force_max"] == 0.5


def test_linear_dynamics_minimax_config_authors_declarative_dynamics_node(
    tmp_path: Path,
) -> None:
    payload = _payload(
        tmp_path,
        adversary_type="linear_dynamics",
        n_adversary_batches=2,
        linear_dynamics_eta_max=0.2,
    )
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    graph = spec.graph.inline

    assert graph["nodes"][PLANT_INTERVENOR_LABEL]["type"] == "DynamicsMatrixPerturb"
    assert jnp.asarray(graph["nodes"][PLANT_INTERVENOR_LABEL]["params"]["delta_A"]).shape == (
        2,
        4,
    )
    assert spec.method_payload.payload["config"]["linear_dynamics_eta_max"] == 0.2
    projection = next(
        step
        for step in spec.worker_execution.method_contract.phase_program.update_steps
        if step.name == "adversary_projection"
    )
    assert projection.metadata["operator"] == "frobenius_ball"


def test_effective_phase_fingerprint_rejects_tampered_spec(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    spec = dict(payload["feedbax_training_run_spec"])
    spec["worker_execution"]["effective_phase"]["phase_program"]["metadata"] = {
        "phase_program_identity": "tampered"
    }
    payload["feedbax_training_run_spec"] = spec

    try:
        validate_minimax_run_spec(payload, spec_dir=tmp_path)
    except ValueError as exc:
        assert "effective-phase fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("tampered effective phase fingerprint was accepted")


def test_minimax_training_run_spec_file_round_trip_rebuilds_idempotently(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path, adversary_type="linear_dynamics", linear_dynamics_eta_max=0.2)
    spec_path = tmp_path / "run.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = minimax_training_run_spec_from_file(spec_path)
    rebuilt = _payload_from_config(tmp_path, minimax_training_run_spec_to_config(loaded))

    original_spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    rebuilt_spec = TrainingRunSpec.model_validate(rebuilt["feedbax_training_run_spec"])
    assert _canonical_json(original_spec.model_dump(mode="json", exclude_none=True)) == (
        _canonical_json(rebuilt_spec.model_dump(mode="json", exclude_none=True))
    )


def test_minimax_payload_schema_version_stays_v1() -> None:
    assert MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION == (
        "rlrmp.spec.training_method.minimax_payload.v1"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"adversary_type": "invalid"},
        {"n_warmup_batches": -1},
        {"n_replicates": 0},
        {"adversary_type": "linear_dynamics", "fused": False},
    ],
)
def test_minimax_config_validation_replaces_legacy_validator(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


@pytest.mark.parametrize(
    ("adversary_type", "step_fn_name"),
    [
        ("gaussian_bump", "_vmapped_gaussian_adversary_ascent"),
        ("linear_dynamics", "_vmapped_linear_adversary_ascent"),
    ],
)
def test_minimax_jitted_adversary_step_matches_eager_fixed_seed(
    tmp_path: Path,
    adversary_type: str,
    step_fn_name: str,
) -> None:
    spec = _native_smoke_spec(tmp_path, adversary_type=adversary_type, n_adversary_batches=1)
    config = minimax_training_run_spec_to_config(spec)
    args = MinimaxConfig.model_validate(config)
    hps = build_hps(args)
    slots, runtime_context = build_minimax_native_initial_slots(
        run_spec=spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(10),
    )
    runtime = runtime_context.component("minimax")
    prepared = _prepare_adversarial_batch(runtime, slots["rng"], batch_index=0)
    adversary = slots["adversary_population"][prepared.active_member_index]
    adv_opt_state = slots["adversary_optimizer"][prepared.active_member_index]
    step_fn = {
        "_vmapped_gaussian_adversary_ascent": _vmapped_gaussian_adversary_ascent,
        "_vmapped_linear_adversary_ascent": _vmapped_linear_adversary_ascent,
    }[step_fn_name]

    eager = step_fn.__wrapped__(
        runtime,
        slots["controller"],
        adversary,
        adv_opt_state,
        prepared.trial_specs,
        prepared.trial_keys,
    )
    jitted = step_fn(
        runtime,
        slots["controller"],
        adversary,
        adv_opt_state,
        prepared.trial_specs,
        prepared.trial_keys,
    )

    diffs = compare_pytrees(_bool_leaves_as_ints(eager), _bool_leaves_as_ints(jitted))
    assert max((diff.max_abs_diff for diff in diffs), default=0.0) <= 1e-6


def test_minimax_jitted_controller_step_matches_eager_fixed_seed(tmp_path: Path) -> None:
    spec = _native_smoke_spec(tmp_path, n_adversary_batches=1)
    config = minimax_training_run_spec_to_config(spec)
    args = MinimaxConfig.model_validate(config)
    hps = build_hps(args)
    slots, runtime_context = build_minimax_native_initial_slots(
        run_spec=spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(11),
    )
    runtime = runtime_context.component("minimax")
    prepared = _prepare_adversarial_batch(runtime, slots["rng"], batch_index=0)
    adversary = slots["adversary_population"][prepared.active_member_index]

    eager = _vmapped_controller_descent.__wrapped__(
        runtime,
        slots["controller"],
        slots["controller_optimizer"],
        adversary,
        prepared.trial_specs,
        prepared.trial_keys,
    )
    jitted = _vmapped_controller_descent(
        runtime,
        slots["controller"],
        slots["controller_optimizer"],
        adversary,
        prepared.trial_specs,
        prepared.trial_keys,
    )

    diffs = compare_pytrees(_bool_leaves_as_ints(eager), _bool_leaves_as_ints(jitted))
    assert max((diff.max_abs_diff for diff in diffs), default=0.0) <= 1e-6


def test_minimax_native_executor_emits_governed_manifest(tmp_path: Path) -> None:
    # Analysis tests intentionally enable process-global x64 in full-suite
    # workers. This training test owns the opposite precondition, so contain its
    # construction and execution in an explicit local mode.
    with jax.enable_x64(False):
        spec = _native_smoke_spec(tmp_path, n_adversary_batches=2)
        result = execute_minimax_training_run_spec_native(
            spec,
            run_id="native-minimax-post-run-protocol",
            key=jr.PRNGKey(0),
            manifest_root=tmp_path / "manifests" / "post-run",
            checkpoint_root=tmp_path / "checkpoints" / "post-run",
            manifest_conflict_policy="reuse-identical",
        )

    assert result.manifest_path.is_file()
    assert result.final_coordinate.phase == "done"
    assert result.final_slots["controller_loss"] != 0.0
    assert [write.manifest.barrier for write in result.checkpoint_writes] == [
        "after_adversarial",
        "after_adversarial",
        "after_adversarial",
    ]
    assert [
        write.manifest.metadata["barrier_visit_ordinal"]
        for write in result.checkpoint_writes
    ] == [0, 1, 2]
    assert [
        write.manifest.completed_coordinate.program_step for write in result.checkpoint_writes
    ] == [1, 2, 3]


def test_minimax_native_initial_slots_honor_explicit_warmup_model(
    tmp_path: Path,
) -> None:
    spec = _native_smoke_spec(tmp_path, n_adversary_batches=1)
    config = minimax_training_run_spec_to_config(spec)
    args = MinimaxConfig.model_validate(config)
    hps = build_hps(args)
    fresh_slots, fresh_runtime_context = build_minimax_native_initial_slots(
        run_spec=spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(0),
    )
    fresh_runtime = fresh_runtime_context.component("minimax")
    fresh_model = fresh_runtime.pair.model
    shifted_net = jt.map(
        lambda leaf: (
            leaf + 0.125
            if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)
            else leaf
        ),
        fresh_model.get_node("net"),
        is_leaf=eqx.is_array,
    )
    sentinel_model = eqx.tree_at(
        lambda model: model.get_node("net"),
        fresh_model,
        shifted_net,
    )
    warmup_model_path = tmp_path / "warmup_model.eqx"
    fbx_save(warmup_model_path, sentinel_model)

    warmup_config = {**config, "warmup_model": str(warmup_model_path)}
    warmup_spec = TrainingRunSpec.model_validate(
        _payload_from_config(tmp_path, warmup_config)["feedbax_training_run_spec"]
    )
    warmup_args = MinimaxConfig.model_validate(warmup_config)
    warmup_slots, warmup_runtime_context = build_minimax_native_initial_slots(
        run_spec=warmup_spec,
        hps=hps,
        args=warmup_args,
        key=jr.PRNGKey(0),
    )
    warmup_runtime = warmup_runtime_context.component("minimax")
    expected_controller = _controller_state_from_model(
        sentinel_model,
        warmup_runtime.controller_layout,
    )

    loaded_diffs = compare_pytrees(
        _bool_leaves_as_ints(warmup_slots["controller"].per_replicate_leaves),
        _bool_leaves_as_ints(expected_controller.per_replicate_leaves),
    )
    fresh_diffs = compare_pytrees(
        _bool_leaves_as_ints(fresh_slots["controller"].per_replicate_leaves),
        _bool_leaves_as_ints(warmup_slots["controller"].per_replicate_leaves),
    )
    assert max((diff.max_abs_diff for diff in loaded_diffs), default=0.0) <= 1e-6
    assert max((diff.max_abs_diff for diff in fresh_diffs), default=0.0) > 0.0
    assert warmup_runtime.pair.model is not fresh_runtime.pair.model
