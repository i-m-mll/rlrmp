"""Spec-first minimax training contracts."""

import json
import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest

from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.worker import ProgressCoordinate
from jax_cookbook import save as fbx_save
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.model.feedbax_graph import build_rlrmp_feedbax_graph_bundle
from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.equivalence import (
    assert_paired_equivalent,
    compare_pytrees,
    run_paired_equivalence,
)
from rlrmp.train.minimax import (
    MINIMAX_METHOD_REF,
    MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
    MinimaxConfig,
    MinimaxMethodPayload,
    _minimax_method_payload,
    build_hps,
    build_minimax_training_run_spec,
    build_minimax_native_initial_slots,
    execute_minimax_training_run_spec_native,
    legacy_cli_args_to_minimax_config,
    minimax_config_namespace,
    minimax_effective_phase_fingerprint,
    minimax_effective_phase_spec,
    minimax_method_contract,
    minimax_training_run_spec_from_file,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
)
from rlrmp.train.minimax_native import (
    INNER_ASCENT_KERNEL_REF,
    OUTER_DESCENT_KERNEL_REF,
    PROJECTION_KERNEL_REF,
    WARMUP_KERNEL_REF,
    MinimaxControllerState,
    _controller_state_from_model,
    _prepare_adversarial_batch,
    _vmapped_controller_descent,
    _vmapped_gaussian_adversary_ascent,
    _vmapped_linear_adversary_ascent,
    minimax_update_kernels,
)
from rlrmp.train.task_model import build_task_base


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _payload_from_config(tmp_path: Path, config: dict) -> dict:
    args = minimax_config_namespace(config)
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
        git={"commit": "test"},
        gpu_info={"available": False},
        feedbax_graph={"graph_spec_path": "graph_spec.json", "manifest_path": "manifest.json"},
    )


def _payload(tmp_path: Path, argv: list[str]) -> dict:
    config = legacy_cli_args_to_minimax_config(
        [
            "--n-warmup-batches",
            "1",
            "--n-adversary-batches",
            "0",
            "--batch-size",
            "1",
            "--n-replicates",
            "1",
            "--output-dir",
            str(tmp_path / "_artifacts" / "54b0c2e" / "runs" / "spec"),
            *argv,
        ]
    )
    return _payload_from_config(tmp_path, config)


def _native_smoke_spec(
    tmp_path: Path,
    *,
    adversary_type: str = "gaussian_bump",
    n_adversary_batches: int = 1,
) -> TrainingRunSpec:
    config = legacy_cli_args_to_minimax_config(
        [
            "--adversary-type",
            adversary_type,
            "--n-warmup-batches",
            "0",
            "--n-adversary-batches",
            str(n_adversary_batches),
            "--n-adversary-steps",
            "1",
            "--batch-size",
            "1",
            "--adv-batch-size",
            "1",
            "--n-replicates",
            "1",
            "--output-dir",
            str(tmp_path / "_artifacts" / "62a658d" / "runs" / f"native_smoke_{adversary_type}"),
        ]
    )
    payload = _payload_from_config(tmp_path, config)
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def _manual_minimax_kernel_loop(
    spec: TrainingRunSpec,
    *,
    key: Any,
) -> dict[str, object]:
    config = minimax_training_run_spec_to_config(spec)
    args = minimax_config_namespace(config)
    hps = build_hps(args)
    slots, runtime = build_minimax_native_initial_slots(
        run_spec=spec,
        hps=hps,
        args=args,
        key=key,
    )
    payload = MinimaxMethodPayload.model_validate(spec.method_payload.payload)
    kernels = minimax_update_kernels(payload)
    context = {
        RLRMP_RUNTIME_CONTEXT_KEY: runtime,
        "run_spec": spec,
        "method_payload": payload,
    }
    coordinate = ProgressCoordinate(run_id="manual-minimax", phase="warmup")
    slots.update(kernels[WARMUP_KERNEL_REF](slots, coordinate, context))
    coordinate = coordinate.model_copy(update={"global_step": 1, "phase": "adversarial"})
    for _ in range(int(args.n_adversary_batches)):
        for kernel_ref in (
            INNER_ASCENT_KERNEL_REF,
            PROJECTION_KERNEL_REF,
            OUTER_DESCENT_KERNEL_REF,
        ):
            slots.update(kernels[kernel_ref](slots, coordinate, context))
        coordinate = coordinate.model_copy(update={"global_step": coordinate.global_step + 1})
    return slots


def _comparable_native_slots(slots: dict[str, object]) -> dict[str, object]:
    controller = slots["controller"]
    assert isinstance(controller, MinimaxControllerState)
    comparable = {
        "controller": controller.per_replicate_leaves,
        "controller_optimizer": slots["controller_optimizer"],
        "adversary_population": slots["adversary_population"],
        "adversary_optimizer": slots["adversary_optimizer"],
        "rng": slots["rng"],
        "controller_loss": slots["controller_loss"],
        "adversary_loss": slots["adversary_loss"],
    }
    return jt.map(
        lambda leaf: (
            leaf.astype(jnp.int8)
            if getattr(getattr(leaf, "dtype", None), "kind", None) == "b"
            else leaf
        ),
        comparable,
    )


def _bool_leaves_as_ints(tree: Any) -> Any:
    return jt.map(
        lambda leaf: (
            leaf.astype(jnp.int8)
            if getattr(getattr(leaf, "dtype", None), "kind", None) == "b"
            else leaf
        ),
        tree,
    )


def test_warmup_only_minimax_cli_round_trips_to_training_run_spec(tmp_path: Path) -> None:
    payload = _payload(tmp_path, [])

    validate_minimax_run_spec(payload, spec_dir=tmp_path)
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])

    assert spec.method_ref.key == MINIMAX_METHOD_REF
    assert spec.worker_execution.effective_phase.phase_program.initial_phase == "warmup"
    assert minimax_training_run_spec_to_config(spec)["n_adversary_batches"] == 0
    assert payload["rlrmp_run_spec"]["schema_version"] == "rlrmp.run_spec.v2"


def test_gaussian_bump_minimax_cli_round_trips_to_training_run_spec(tmp_path: Path) -> None:
    payload = _payload(
        tmp_path,
        ["--n-adversary-batches", "2", "--n-bumps", "2", "--force-max", "0.5"],
    )
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    config = minimax_training_run_spec_to_config(spec)

    assert config["adversary_type"] == "gaussian_bump"
    assert config["n_bumps"] == 2
    assert spec.method_payload.payload["adversarial"]["inner_direction"] == "maximize"


def test_linear_dynamics_minimax_cli_authors_declarative_dynamics_node(
    tmp_path: Path,
) -> None:
    payload = _payload(
        tmp_path,
        [
            "--adversary-type",
            "linear_dynamics",
            "--n-adversary-batches",
            "2",
            "--linear-dynamics-eta-max",
            "0.2",
        ],
    )
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    graph = spec.graph.inline

    assert graph["nodes"][PLANT_INTERVENOR_LABEL]["type"] == "DynamicsMatrixPerturb"
    assert jnp.asarray(graph["nodes"][PLANT_INTERVENOR_LABEL]["params"]["delta_A"]).shape == (
        2,
        4,
    )
    assert spec.method_payload.payload["projection"] == {
        "target": "adversary_population[active_member].delta_A",
        "operator": "frobenius_ball",
        "radius": 0.2,
        "radius_source": "linear_dynamics_eta_max",
        "timing": "after_each_adversary_step",
        "phase_scope": "adversarial",
    }


def test_effective_phase_fingerprint_rejects_tampered_spec(tmp_path: Path) -> None:
    payload = _payload(tmp_path, [])
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


def test_minimax_method_payload_matches_pre_refactor_golden_fixtures() -> None:
    fixtures = json.loads(
        Path("tests/fixtures/minimax_method_payload_golden.json").read_text(encoding="utf-8")
    )
    graph_payload = fixtures["graph_payload"]
    contract = minimax_method_contract()
    effective_phase = minimax_effective_phase_spec(contract)

    for case in fixtures["cases"].values():
        config = legacy_cli_args_to_minimax_config(case["argv"])
        envelope = _minimax_method_payload(
            config,
            output_dir=Path("_artifacts/minimax/minimax_test"),
            spec_dir=Path("results/5e5ba8b/runs/golden"),
        )
        envelope_dump = envelope.model_dump(mode="json", exclude_none=True)

        assert _canonical_json(envelope_dump) == case["method_payload_envelope_json"]
        assert (
            minimax_effective_phase_fingerprint(
                effective_phase=effective_phase,
                graph_payload=graph_payload,
                method_payload=envelope_dump,
            )
            == case["effective_phase_fingerprint"]
        )


def test_minimax_training_run_spec_file_round_trip_rebuilds_idempotently(
    tmp_path: Path,
) -> None:
    payload = _payload(
        tmp_path,
        ["--adversary-type", "linear_dynamics", "--linear-dynamics-eta-max", "0.2"],
    )
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


def test_minimax_payload_rejects_derived_view_drift() -> None:
    envelope = _minimax_method_payload(
        MinimaxConfig().model_dump(mode="python"),
        output_dir=Path("_artifacts/minimax/minimax_test"),
        spec_dir=Path("results/5e5ba8b/runs/golden"),
    )
    payload = dict(envelope.payload)
    payload["warmup"] = {**payload["warmup"], "n_batches": 999}

    with pytest.raises(ValueError, match="derived views disagree"):
        MinimaxMethodPayload.model_validate(payload)


def test_minimax_cli_adapter_preserves_flag_forms_and_help() -> None:
    config = legacy_cli_args_to_minimax_config(
        [
            "--checkpoint",
            "--no-resume",
            "--n-bumps=2",
            "--force-max",
            "0.5",
            "--jax-explain-cache-misses=false",
        ]
    )

    assert config["checkpoint"] is True
    assert config["resume"] is False
    assert config["n_bumps"] == 2
    assert config["force_max"] == 0.5
    assert config["jax_explain_cache_misses"] is False

    with pytest.raises(ValueError, match="unknown minimax option"):
        legacy_cli_args_to_minimax_config(["--does-not-exist"])
    with pytest.raises(ValueError, match="only valid for boolean"):
        legacy_cli_args_to_minimax_config(["--no-batch-size"])

    with pytest.raises(SystemExit) as exc_info:
        legacy_cli_args_to_minimax_config(["--help"])
    help_text = str(exc_info.value)
    for name, field in MinimaxConfig.model_fields.items():
        assert f"--{name.replace('_', '-')} (default: {field.default!r})" in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["--adversary-type", "invalid"],
        ["--n-warmup-batches", "-1"],
        ["--n-replicates", "0"],
        ["--adversary-type", "linear_dynamics", "--no-fused"],
    ],
)
def test_minimax_config_validation_replaces_legacy_validator(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        legacy_cli_args_to_minimax_config(argv)


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
    args = minimax_config_namespace(config)
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
    args = minimax_config_namespace(config)
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


def test_minimax_native_executor_matches_fixed_seed_manual_kernel_loop(
    tmp_path: Path,
) -> None:
    spec = _native_smoke_spec(tmp_path, n_adversary_batches=1)
    manual_slots = _manual_minimax_kernel_loop(spec, key=jr.PRNGKey(0))

    result = execute_minimax_training_run_spec_native(
        spec,
        run_id="native-minimax-fixed-seed",
        key=jr.PRNGKey(0),
        manifest_root=tmp_path / "manifests" / "executor",
        checkpoint_root=tmp_path / "checkpoints" / "executor",
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        "minimax.gaussian_bump.driver",
        lambda: manual_slots,
        lambda: result.final_slots,
        comparable=_comparable_native_slots,
        left_label="manual_kernel_loop",
        right_label="native_executor",
    )
    assert_paired_equivalent(report)
    assert result.final_coordinate.phase == "done"
    assert result.final_slots["controller_loss"] != 0.0
    assert result.final_slots["adversary_loss"] != 0.0


def test_minimax_native_executor_emits_post_run_protocol_inputs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = _native_smoke_spec(tmp_path, n_adversary_batches=2)
    output_dir = Path(minimax_training_run_spec_to_config(spec)["output_dir"])

    with caplog.at_level(logging.INFO, logger="rlrmp.train.minimax_native"):
        result = execute_minimax_training_run_spec_native(
            spec,
            run_id="native-minimax-post-run-protocol",
            key=jr.PRNGKey(0),
            manifest_root=tmp_path / "manifests" / "post-run",
            checkpoint_root=tmp_path / "checkpoints" / "post-run",
            manifest_conflict_policy="reuse-identical",
        )

    summary_path = output_dir / "training_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["training_mode"] == "minimax"
    assert summary["n_train_batches"] == 2
    assert summary["completed_batches"] == 2
    assert summary["run_id"] == result.run_id
    assert summary["manifest_path"] == str(result.manifest_path)
    assert "final_train_loss" not in summary
    assert "final_validation_loss" not in summary

    progress_lines = [
        record.message
        for record in caplog.records
        if record.name == "rlrmp.train.minimax_native" and record.message.startswith("BATCH ")
    ]
    assert any("phase=adversarial batch=0/2" in line for line in progress_lines)
    assert any("phase=adversarial batch=1/2" in line for line in progress_lines)


def test_minimax_native_initial_slots_honor_explicit_warmup_model(
    tmp_path: Path,
) -> None:
    spec = _native_smoke_spec(tmp_path, n_adversary_batches=1)
    config = minimax_training_run_spec_to_config(spec)
    args = minimax_config_namespace(config)
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
    warmup_args = minimax_config_namespace(warmup_config)
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


def test_minimax_native_executor_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    spec = _native_smoke_spec(tmp_path, n_adversary_batches=2)
    full = execute_minimax_training_run_spec_native(
        spec,
        run_id="native-minimax-full",
        key=jr.PRNGKey(1),
        manifest_root=tmp_path / "manifests" / "full",
        checkpoint_root=tmp_path / "checkpoints" / "full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "resume"
    partial = execute_minimax_training_run_spec_native(
        spec,
        run_id="native-minimax-resume",
        key=jr.PRNGKey(1),
        manifest_root=tmp_path / "manifests" / "resume-partial",
        checkpoint_root=checkpoint_root,
        stop_after_barrier="after_adversarial",
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_minimax_training_run_spec_native(
        spec,
        run_id="native-minimax-resume",
        key=jr.PRNGKey(1),
        manifest_root=tmp_path / "manifests" / "resume-final",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        "minimax.gaussian_bump.resume",
        lambda: full.final_slots,
        lambda: resumed.final_slots,
        comparable=_comparable_native_slots,
        left_label="uninterrupted",
        right_label="resumed",
    )
    assert partial.final_coordinate.completed_barrier == "after_adversarial"
    assert_paired_equivalent(report)


def test_linear_dynamics_minimax_native_executor_matches_fixed_seed_manual_kernel_loop(
    tmp_path: Path,
) -> None:
    spec = _native_smoke_spec(tmp_path, adversary_type="linear_dynamics", n_adversary_batches=1)
    manual_slots = _manual_minimax_kernel_loop(spec, key=jr.PRNGKey(2))

    result = execute_minimax_training_run_spec_native(
        spec,
        run_id="native-minimax-linear-dynamics-fixed-seed",
        key=jr.PRNGKey(2),
        manifest_root=tmp_path / "manifests" / "linear",
        checkpoint_root=tmp_path / "checkpoints" / "linear",
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        "minimax.linear_dynamics.driver",
        lambda: manual_slots,
        lambda: result.final_slots,
        comparable=_comparable_native_slots,
        left_label="manual_kernel_loop",
        right_label="native_executor",
    )
    assert_paired_equivalent(report)
    assert result.final_coordinate.phase == "done"
    assert result.final_slots["controller_loss"] != 0.0
    assert result.final_slots["adversary_loss"] != 0.0


def test_linear_dynamics_minimax_native_executor_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    spec = _native_smoke_spec(tmp_path, adversary_type="linear_dynamics", n_adversary_batches=2)
    full = execute_minimax_training_run_spec_native(
        spec,
        run_id="native-minimax-linear-dynamics-full",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "linear-full",
        checkpoint_root=tmp_path / "checkpoints" / "linear-full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "linear-resume"
    partial = execute_minimax_training_run_spec_native(
        spec,
        run_id="native-minimax-linear-dynamics-resume",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "linear-resume-partial",
        checkpoint_root=checkpoint_root,
        stop_after_barrier="after_adversarial",
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_minimax_training_run_spec_native(
        spec,
        run_id="native-minimax-linear-dynamics-resume",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "linear-resume-final",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        "minimax.linear_dynamics.resume",
        lambda: full.final_slots,
        lambda: resumed.final_slots,
        comparable=_comparable_native_slots,
        left_label="uninterrupted",
        right_label="resumed",
    )
    assert partial.final_coordinate.completed_barrier == "after_adversarial"
    assert_paired_equivalent(report)
