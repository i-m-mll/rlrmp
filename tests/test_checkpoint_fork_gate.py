"""Tests for the RLRMP checkpoint fork parity launch gate."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    LrScheduleSpec,
    MethodPayloadEnvelope,
    LossTermSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.worker import (
    EffectivePhaseSpec,
    ProgressCoordinate,
    derive_consistency_predicate,
    toy_minimax_method_contract,
)
from feedbax.training.checkpoint_custody import write_checkpoint_transaction

from rlrmp.runtime import checkpoint_fork_gate
from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot, serialize_pytree_slot
from rlrmp.runtime.checkpoint_fork_gate import (
    PARITY_SCHEMA_VERSION,
    ForkTarget,
    ForkParityError,
    fork_checkpoints_with_parity,
    training_spec_from_row_payload,
)
from rlrmp.runtime.training_run_specs import FEEDBAX_TRAINING_RUN_SPEC_KEY
from rlrmp.train.adaptive_epsilon_native import (
    AdaptiveEpsilonMethodPayload,
    SerializedPyTreeSlot,
    build_adaptive_epsilon_native_initial_slots,
)
from rlrmp.train.cs_nominal_gru import (
    ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    _config_namespace,
    build_hps,
    build_parser,
    write_run_spec,
)
from rlrmp.train.executor.slots import COMPLETED_BATCHES, OPTIMIZER
from scripts.fork_checkpoint_gate import main as fork_gate_main


def _minimal_graph() -> dict[str, object]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _training_spec() -> TrainingRunSpec:
    contract = toy_minimax_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    program.checkpoint_barriers[0].metadata["consistency_mode"] = "population-barrier"
    method_contract = contract.model_copy(
        update={
            "method_ref": "feedbax/standard_supervised/v1",
            "method_payload_schema_version": (
                STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION
            ),
            "phase_program": program,
        }
    )
    effective_phase = EffectivePhaseSpec(
        method_ref="feedbax/standard_supervised/v1",
        axes=method_contract.axes,
        state_slots=method_contract.state_slots,
        phase_program=program,
        consistency_predicate=derive_consistency_predicate(program),
    )
    return TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=4, batch_size=3),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=method_contract,
            effective_phase=effective_phase,
        ),
    )


def _training_spec_with_optimizer_schedule() -> TrainingRunSpec:
    spec = _training_spec()
    payload = spec.method_payload.model_dump(mode="json", exclude_none=True)
    payload["payload"]["optimizer"] = OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            kind="warmup_cosine",
            learning_rate_0=3e-4,
            total_steps=10,
            constant_lr_iterations=4,
            warmup_init_fraction=0.1,
            cosine_annealing_alpha=0.1,
        ),
    ).model_dump(mode="json", exclude_none=True)
    return spec.model_copy(
        update={"method_payload": MethodPayloadEnvelope.model_validate(payload)}
    )


def _coordinate(step: int = 3) -> ProgressCoordinate:
    return ProgressCoordinate(
        run_id="source-run",
        phase="warmup",
        global_step=step,
        completed_barrier="after_warmup",
    )


def _slots() -> dict[str, object]:
    return {
        "controller": jnp.array([1.0, 2.0]),
        "controller_optimizer": {"count": jnp.array(1)},
        "adversary_population": [jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4])],
        "adversary_optimizer": {"count": jnp.array([1, 1])},
        "rng": jnp.array([11, 22], dtype=jnp.uint32),
        "loss": [0.5],
    }


def _row_payload(spec: TrainingRunSpec) -> dict[str, object]:
    return {
        "schema_version": "rlrmp.test",
        "hps": {
            "lr_schedule": "delayed_cosine",
            "learning_rate_0": 0.01,
            "constant_lr_iterations": 0,
            "n_batches_condition": 10,
            "cosine_annealing_alpha": 0.1,
        },
        FEEDBAX_TRAINING_RUN_SPEC_KEY: spec.model_dump(mode="json", exclude_none=True),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_checkpoint(tmp_path: Path, spec: TrainingRunSpec) -> Path:
    source_root = tmp_path / "source"
    write_checkpoint_transaction(
        source_root,
        run_spec=spec,
        phase_program=spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        slots=_slots(),
        population_member_ids={"adversary_population": ["adv-a", "adv-b"]},
    )
    return source_root


def _target(tmp_path: Path, spec: TrainingRunSpec, row_id: str) -> ForkTarget:
    spec_path = tmp_path / f"{row_id}.json"
    _write_json(spec_path, _row_payload(spec))
    return ForkTarget(
        row_id=row_id,
        spec_path=spec_path,
        checkpoint_root=tmp_path / f"{row_id}-checkpoint",
    )


def _run_plan(tmp_path: Path, mode: str = "continue") -> Path:
    path = tmp_path / "RUN_PLAN.md"
    path.write_text(
        f"# Launch plan\n\nLR continuation schedule: {mode}\n",
        encoding="utf-8",
    )
    return path


def _adaptive_training_spec(tmp_path: Path) -> TrainingRunSpec:
    args = build_parser().parse_args([])
    defaults = {
        "n_train_batches": 6,
        "batch_size": 1,
        "n_replicates": 1,
        "hidden_size": 4,
        "dry_run": True,
        "full_train": True,
        "resume": True,
        "checkpoint_interval_batches": 1,
        "controller_lr": 1e-3,
        "gradient_clip_norm": 5.0,
        "lr_schedule": "warmup_cosine",
        "lr_warmup_batches": 2,
        "lr_warmup_init_fraction": 0.01,
        "lr_cosine_alpha": 0.1,
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
        "adaptive_epsilon_controller_training_mode": ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
        "loss_objective": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        "output_dir": str(tmp_path / "adaptive" / "bulk"),
        "spec_dir": str(tmp_path / "adaptive" / "spec"),
    }
    for key, value in defaults.items():
        setattr(args, key, value)
    payload = write_run_spec(args)["run_spec"]
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    method_payload = spec.method_payload.model_dump(mode="json", exclude_none=True)
    method_payload["payload"]["controller_optimizer"]["lr_schedule"]["total_steps"] = 4
    return spec.model_copy(
        update={"method_payload": MethodPayloadEnvelope.model_validate(method_payload)}
    )


def _adaptive_source_checkpoint(
    tmp_path: Path,
    spec: TrainingRunSpec,
    *,
    restored_count: int,
    completed_batches: int,
) -> Path:
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
        key=jr.PRNGKey(11),
        lr_continuation_mode="restart",
    )
    native = runtime.component("adaptive_epsilon")
    optimizer_state = deserialize_pytree_slot(
        slots[OPTIMIZER].payload,
        native.optimizer_template,
        slot=OPTIMIZER,
    )
    slots = dict(slots)
    slots[OPTIMIZER] = SerializedPyTreeSlot(
        serialize_pytree_slot(_with_injected_count(optimizer_state, restored_count))
    )
    slots[COMPLETED_BATCHES] = jnp.asarray(completed_batches, dtype=jnp.int32)
    source_root = tmp_path / "adaptive-source"
    write_checkpoint_transaction(
        source_root,
        run_spec=spec,
        phase_program=spec.worker_execution.method_contract.phase_program,
        barrier_name="after_adaptive_epsilon_train_chunk",
        coordinate=ProgressCoordinate(
            run_id="adaptive-source",
            phase="adaptive_epsilon_train_chunk",
            global_step=completed_batches,
            completed_barrier="after_adaptive_epsilon_train_chunk",
        ),
        slots=slots,
    )
    return source_root


def _is_injected_hyperparams_state(value: object) -> bool:
    fields = getattr(value, "_fields", ())
    return {"count", "hyperparams", "inner_state"}.issubset(set(fields))


def _with_injected_count(opt_state: object, count: int) -> object:
    def replace_learning_rate_count(hyperparams_states: object) -> object:
        if not isinstance(hyperparams_states, dict):
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

    def replace_count(leaf: object) -> object:
        if _is_injected_hyperparams_state(leaf):
            return leaf._replace(
                count=jnp.full_like(leaf.count, count),
                hyperparams_states=replace_learning_rate_count(leaf.hyperparams_states),
            )
        return leaf

    return jt.map(replace_count, opt_state, is_leaf=_is_injected_hyperparams_state)


def test_extracts_nested_feedbax_training_spec_before_wrapper_validation(
    tmp_path: Path,
) -> None:
    spec = _training_spec()
    payload = _row_payload(spec)
    payload["unexpected_wrapper_field"] = "wrapper-only"

    extracted = training_spec_from_row_payload(payload)

    assert extracted.method_ref == spec.method_ref


def test_fork_gate_writes_parity_table_and_lr_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _training_spec()
    source_root = _source_checkpoint(tmp_path, spec)
    targets = [_target(tmp_path, spec, "row-a"), _target(tmp_path, spec, "row-b")]
    output = tmp_path / "parity.json"

    table = fork_checkpoints_with_parity(
        source_checkpoint_root=source_root,
        targets=targets,
        run_plan_path=_run_plan(tmp_path),
        parity_output_path=output,
    )

    captured = capsys.readouterr()
    assert "LR_CONTINUATION step=3 global_step=3 optimizer_count=3 lr=" in captured.out
    assert table["schema_version"] == PARITY_SCHEMA_VERSION
    assert table["ok"] is True
    assert output.is_file()
    assert {row["row_id"] for row in table["targets"]} == {"row-a", "row-b"}
    assert all(row["ok"] for row in table["targets"])


def test_fork_gate_reports_declared_feedbax_lr_schedule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _training_spec_with_optimizer_schedule()
    source_root = _source_checkpoint(tmp_path, spec)
    target = _target(tmp_path, spec, "row-a")
    payload = json.loads(target.spec_path.read_text(encoding="utf-8"))
    payload.pop("hps")
    _write_json(target.spec_path, payload)

    table = fork_checkpoints_with_parity(
        source_checkpoint_root=source_root,
        targets=[target],
        run_plan_path=_run_plan(tmp_path),
        parity_output_path=tmp_path / "parity.json",
    )

    captured = capsys.readouterr()
    assert "LR_CONTINUATION step=3 global_step=3 optimizer_count=3 lr=" in captured.out
    assert table["lr_continuation"]["lr"] == pytest.approx(0.0002325)


def test_fork_gate_reports_adaptive_lr_through_runtime_builder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _adaptive_training_spec(tmp_path)
    source_root = _adaptive_source_checkpoint(
        tmp_path,
        spec,
        restored_count=9,
        completed_batches=5,
    )
    target = _target(tmp_path, spec, "adaptive-row")

    table = fork_checkpoints_with_parity(
        source_checkpoint_root=source_root,
        targets=[target],
        run_plan_path=_run_plan(tmp_path, mode="restart"),
        parity_output_path=tmp_path / "adaptive-parity.json",
    )

    captured = capsys.readouterr()
    assert "LR_CONTINUATION step=0 " in captured.out
    assert "LR_CONTINUATION step=2 " in captured.out
    assert "LR_CONTINUATION step=4 " in captured.out
    lr_report = table["lr_continuation"]
    assert lr_report["optimizer_count_at_current_step"] == 9
    by_step = {point["step"]: point for point in lr_report["points"]}
    assert by_step[0]["optimizer_count"] == 9
    assert by_step[0]["lr"] == pytest.approx(1e-5, rel=1e-4)
    assert by_step[2]["lr"] == pytest.approx(1e-3, rel=1e-4)
    assert by_step[4]["lr"] == pytest.approx(1e-4, rel=1e-4)


def test_fork_gate_rejects_disconnected_adaptive_lr_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _adaptive_training_spec(tmp_path)
    source_root = _adaptive_source_checkpoint(
        tmp_path,
        spec,
        restored_count=9,
        completed_batches=5,
    )
    target = _target(tmp_path, spec, "adaptive-row")

    def disconnected_points(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [
            {"step": 0, "global_step": 5, "optimizer_count": 9, "lr": 1e-5},
            {"step": 2, "global_step": 7, "optimizer_count": 11, "lr": 1e-5},
            {"step": 4, "global_step": 9, "optimizer_count": 13, "lr": 1e-5},
        ]

    monkeypatch.setattr(
        checkpoint_fork_gate,
        "adaptive_epsilon_controller_lr_points",
        disconnected_points,
    )

    with pytest.raises(ForkParityError, match="does not match declared schedule"):
        fork_checkpoints_with_parity(
            source_checkpoint_root=source_root,
            targets=[target],
            run_plan_path=_run_plan(tmp_path, mode="restart"),
            parity_output_path=tmp_path / "adaptive-parity.json",
        )


def test_fork_gate_reports_tampered_slot_digest_by_row_and_slot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _training_spec()
    source_root = _source_checkpoint(tmp_path, spec)
    target = _target(tmp_path, spec, "row-a")
    output = tmp_path / "parity.json"
    run_plan = _run_plan(tmp_path)
    fork_checkpoints_with_parity(
        source_checkpoint_root=source_root,
        targets=[target],
        run_plan_path=run_plan,
        parity_output_path=output,
    )
    latest = json.loads((target.checkpoint_root / "latest.json").read_text(encoding="utf-8"))
    manifest_path = target.checkpoint_root / latest["manifest_relative_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    controller = next(slot for slot in manifest["slots"] if slot["slot"] == "controller")
    controller["content_digest"]["slot_root_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    code = fork_gate_main(
        [
            "--source-checkpoint-root",
            str(source_root),
            "--target",
            f"row-a={target.spec_path}:{target.checkpoint_root}",
            "--run-plan",
            str(run_plan),
            "--parity-output",
            str(output),
            "--skip-fork",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "row=row-a" in captured.err
    assert "slot=controller" in captured.err


def test_run_plan_must_declare_lr_continuation(tmp_path: Path) -> None:
    spec = _training_spec()
    source_root = _source_checkpoint(tmp_path, spec)
    target = _target(tmp_path, spec, "row-a")
    run_plan = tmp_path / "RUN_PLAN.md"
    run_plan.write_text("# Launch plan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="LR continuation schedule"):
        fork_checkpoints_with_parity(
            source_checkpoint_root=source_root,
            targets=[target],
            run_plan_path=run_plan,
            parity_output_path=tmp_path / "parity.json",
        )
