"""Tests for the closed-loop extLQG distillation preflight surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from rlrmp.runtime.training_run_specs import MissingTrainingRunSpecFieldError
from rlrmp.train import closed_loop_distillation


def _default_spec_args(**overrides) -> argparse.Namespace:
    values = {
        "run_id": closed_loop_distillation.RUN_ID,
        "run_spec_output": closed_loop_distillation.DEFAULT_SPEC_PATH,
        "output_dir": closed_loop_distillation.DEFAULT_OUTPUT_DIR,
        "teacher_package": closed_loop_distillation.DEFAULT_TEACHER_PACKAGE,
        "teacher_gains_key": closed_loop_distillation.DEFAULT_TEACHER_GAINS_KEY,
        "seed": 0,
        "n_replicates": 5,
        "hidden_size": 180,
        "batch_size": 64,
        "n_batches": 12000,
        "controller_lr": 3e-3,
        "lr_warmup_batches": 500,
        "lr_cosine_alpha": 0.01,
        "gradient_clip_norm": 5.0,
        "trainable_dtype": closed_loop_distillation.DEFAULT_TRAINABLE_DTYPE,
        "kinematics_trajectory_weight": 1.0,
        "velocity_weight": 1.0,
        "endpoint_weight": 0.0,
        "settling_weight": 0.0,
        "action_force_weight": 1.0,
        "perturbation_response_weight": 1.0,
        "input_output_jvp_weight": 0.25,
        "task_rollout_loss_weight": 0.0,
        "checkpoint_interval_batches": 500,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _toy_reference() -> closed_loop_distillation.ExtLQGClosedLoopReference:
    plant_a = jnp.eye(6, dtype=jnp.float32)
    plant_b = jnp.zeros((6, 2), dtype=jnp.float32)
    gains = jnp.zeros((3, 2, 6), dtype=jnp.float32)
    observation = jnp.eye(6, dtype=jnp.float32)
    return closed_loop_distillation.ExtLQGClosedLoopReference(
        plant_a=plant_a,
        plant_b=plant_b,
        controller_gains=gains,
        observation_matrix=observation,
        feedback_gains=gains,
        state_dim=6,
    )


def test_closed_loop_distillation_builds_a378b34_contract() -> None:
    spec = closed_loop_distillation.build_closed_loop_distillation_spec(_default_spec_args())

    closed_loop_distillation.validate_run_spec(spec)
    assert spec["issue"] == "a378b34"
    assert spec["run_id"] == "h0_extlqg_6d_closed_loop_distillation"
    assert spec["user_confirmed"] is False
    assert spec["artifact_output_dir"] == (
        "_artifacts/a378b34/runs/h0_extlqg_6d_closed_loop_distillation"
    )
    assert spec["training_entry"]["script"] == "scripts/train_closed_loop_distillation.py"
    assert spec["training_entry"]["full_train_status"] == (
        "implemented_no_launch_pending_user_approval"
    )
    assert "execute_distillation_training_run_spec_native" in spec["training_entry"]["trainer_path"]
    assert spec["teacher_contract"]["teacher_package"] == (
        "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz"
    )
    assert spec["teacher_contract"]["teacher_gains_key"] == "extlqg_controller_gains"
    assert spec["student_contract"]["setup_function"] == (
        "rlrmp.train.task_model.setup_task_model_pair"
    )
    assert spec["student_contract"]["controller_input_dim"] == 6
    assert spec["student_contract"]["force_filter_feedback"] is True
    assert spec["student_contract"]["initial_hidden_encoder"] is True
    assert spec["student_contract"]["hidden_size"] == 180
    assert spec["student_contract"]["n_replicates"] == 5
    assert spec["student_contract"]["batch_size"] == 64
    assert spec["student_contract"]["controller_lr"] == pytest.approx(3e-3)
    assert spec["student_contract"]["lr_cosine_alpha"] == pytest.approx(0.01)
    assert spec["student_contract"]["gradient_clip_norm"] == pytest.approx(5.0)
    assert spec["student_contract"]["broad_epsilon_pgd_training"] is False
    assert spec["student_contract"]["trainable_dtype"] == "float32"
    assert spec["closed_loop_semantics"]["student_actions_feed_future_observations"] is True
    assert spec["closed_loop_semantics"]["teacher_forced_feedback_bank_imitation"] is False
    assert spec["closed_loop_semantics"]["old_guided_trainer_is_main_path"] is False
    assert spec["loss_surface"]["weights"]["task_qr_rollout"] == 0.0
    assert spec["loss_surface"]["weights"]["endpoint"] == 0.0
    assert spec["loss_surface"]["weights"]["settling"] == 0.0
    assert spec["loss_surface"]["task_qr_rollout_loss_can_be_enabled_later"] is True
    assert spec["execution_target"]["billable_launch_authorized"] is False
    assert spec["checkpointing"]["interval_batches"] == 500
    assert spec["locked_spec_summary"]["n_batches"] == 12000
    assert spec["loss_surface"]["components"]["directional_input_output_jvp"]["basis"] == (
        "full_6d_coordinate_basis"
    )
    assert spec["loss_surface"]["components"]["directional_input_output_jvp"]["jacobian_shape"] == [
        2,
        6,
    ]
    assert (
        "full local 2x6 feedback-to-action Jacobian"
        in (spec["loss_surface"]["components"]["directional_input_output_jvp"]["implementation"])
    )


def test_closed_loop_distillation_cli_dry_run_smoke(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "run.json"

    status = closed_loop_distillation.main(
        [
            "--run-spec-output",
            str(spec_path),
            "--write-run-spec",
            "--dry-run",
            "--smoke-preflight",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    written = json.loads(spec_path.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["run_spec"]["issue"] == "a378b34"
    assert payload["smoke_preflight"]["finite"] is True
    assert payload["smoke_preflight"]["implementation"] == "full_local_jacobian_basis_jvp_vmap"
    assert payload["smoke_preflight"]["shape"] == [6, 2, 1, 2]
    assert written["run_id"] == "h0_extlqg_6d_closed_loop_distillation"


def test_full_training_artifact_writer_materializes_summary_and_pytrees(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    paths = closed_loop_distillation._save_full_training_artifacts(
        spec={"artifact_output_dir": str(output_dir), "run_id": closed_loop_distillation.RUN_ID},
        trained_model={"weight": jnp.ones((2,), dtype=jnp.float32)},
        history={"loss": jnp.array([1.0], dtype=jnp.float32)},
        summary={"mode": "full_train", "completed_batches": 1},
    )

    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert Path(paths["trained_model"]).is_file()
    assert Path(paths["training_history"]).is_file()
    assert (output_dir / "run_spec_snapshot.json").is_file()
    assert summary["mode"] == "full_train"
    assert summary["artifacts"]["training_summary"] == str(output_dir / "training_summary.json")


def test_closed_loop_distillation_smoke_train_uses_injected_executor_hook() -> None:
    spec = closed_loop_distillation.build_closed_loop_distillation_spec(_default_spec_args())
    observed = {}

    def fake_setup_pair(hps, *, key):
        observed["hps"] = hps
        observed["key_shape"] = tuple(key.shape)
        return SimpleNamespace(task=object(), model=SimpleNamespace(nodes={"net": object()}))

    def fake_train_pair(
        trainer,
        pair,
        *,
        n_batches,
        key,
        ensembled,
        loss_func,
        where_train,
        batch_size,
        **kwargs,
    ):
        del pair, key, where_train, kwargs
        observed["trainer"] = trainer
        observed["n_batches"] = n_batches
        observed["batch_size"] = batch_size
        observed["ensembled"] = ensembled
        observed["loss_type"] = type(loss_func).__name__
        return object(), object()

    result = closed_loop_distillation.run_closed_loop_distillation_training(
        spec=spec,
        n_batches=1,
        batch_size=2,
        n_replicates=1,
        hidden_size=6,
        smoke=True,
        setup_pair_fn=fake_setup_pair,
        train_pair_fn=fake_train_pair,
        loss_factory=lambda spec: closed_loop_distillation.ClosedLoopDistillationLoss(
            reference=_toy_reference(),
            weights=closed_loop_distillation.ClosedLoopLossWeights(),
        ),
    )

    assert result["mode"] == "smoke_train"
    assert result["trainer_path"] == "injected_executor_training_fn"
    assert result["completed_batches"] == 1
    assert observed["trainer"] is None
    assert observed["ensembled"] is True
    assert observed["loss_type"] == "ClosedLoopDistillationLoss"
    assert observed["hps"].model.hidden_size == 6


def test_extlqg_reference_rollout_matches_shared_shapes() -> None:
    reference = _toy_reference()
    initial_vector = jnp.asarray([[0.0, 0.0, 0.1, 0.0, 0.2, 0.0]], dtype=jnp.float32)
    target_pos = jnp.asarray([[0.15, 0.0]], dtype=jnp.float32)

    rollout = reference.rollout(
        initial_vector=initial_vector,
        target_pos=target_pos,
        n_steps=3,
    )

    assert rollout["position"].shape == (1, 3, 2)
    assert rollout["velocity"].shape == (1, 3, 2)
    assert rollout["force_filter"].shape == (1, 3, 2)
    assert rollout["action"].shape == (1, 3, 2)


def test_closed_loop_distillation_full_train_requires_approval() -> None:
    spec = closed_loop_distillation.build_closed_loop_distillation_spec(_default_spec_args())

    with pytest.raises(closed_loop_distillation.FullTrainingApprovalRequiredError) as exc:
        closed_loop_distillation.run_closed_loop_distillation_training(spec=spec)

    message = str(exc.value)
    assert "requires explicit user launch approval" in message
    assert "No Feedbax hook blocker" in message


def test_closed_loop_distillation_cli_full_train_returns_approval_guard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "run.json"
    spec = closed_loop_distillation.build_closed_loop_distillation_spec(
        _default_spec_args(run_spec_output=spec_path)
    )
    closed_loop_distillation.write_run_spec(spec_path, spec)

    status = closed_loop_distillation.main(["--run-spec", str(spec_path), "--full-train"])

    payload = json.loads(capsys.readouterr().out)
    assert status == 2
    assert payload["error"] == "full_training_requires_user_approval"
    assert payload["run_spec"]["run_id"] == "h0_extlqg_6d_closed_loop_distillation"
    assert "No Feedbax hook blocker" in payload["message"]


def test_tracked_a378b34_run_spec_fails_closed_without_horizon() -> None:
    spec_path = closed_loop_distillation.DEFAULT_SPEC_PATH
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    with pytest.raises(MissingTrainingRunSpecFieldError, match="teacher_contract.horizon"):
        closed_loop_distillation.validate_run_spec(spec)
    assert spec["expected_artifacts"]["tracked_run_spec"] == str(spec_path)
    assert spec["expected_artifacts"]["bulk_output_dir"] == (
        "_artifacts/a378b34/runs/h0_extlqg_6d_closed_loop_distillation"
    )
