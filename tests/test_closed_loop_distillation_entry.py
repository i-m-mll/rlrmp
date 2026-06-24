"""Tests for the closed-loop extLQG distillation preflight surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

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
        "trainable_dtype": "float32",
        "kinematics_trajectory_weight": 1.0,
        "velocity_weight": 1.0,
        "endpoint_weight": 1.0,
        "settling_weight": 0.5,
        "action_force_weight": 1.0,
        "perturbation_response_weight": 1.0,
        "input_output_jvp_weight": 0.25,
        "task_rollout_loss_weight": 0.0,
        "n_jvp_directions": 16,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_closed_loop_distillation_builds_a378b34_contract() -> None:
    spec = closed_loop_distillation.build_closed_loop_distillation_spec(_default_spec_args())

    closed_loop_distillation.validate_run_spec(spec)
    assert spec["issue"] == "a378b34"
    assert spec["run_id"] == "h0_extlqg_6d_closed_loop_distillation"
    assert spec["artifact_output_dir"] == (
        "_artifacts/a378b34/runs/h0_extlqg_6d_closed_loop_distillation"
    )
    assert spec["training_entry"]["script"] == "scripts/train_closed_loop_distillation.py"
    assert spec["training_entry"]["full_train_status"] == (
        "fail_closed_pending_feedbax_closed_loop_hook"
    )
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
    assert spec["loss_surface"]["task_qr_rollout_loss_can_be_enabled_later"] is True
    assert (
        "dense Jacobian materialization is forbidden"
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
    assert payload["smoke_preflight"]["implementation"] == "directional_jvp_vmap"
    assert payload["smoke_preflight"]["shape"] == [3, 2, 1, 2]
    assert written["run_id"] == "h0_extlqg_6d_closed_loop_distillation"


def test_closed_loop_distillation_full_train_fails_closed() -> None:
    spec = closed_loop_distillation.build_closed_loop_distillation_spec(_default_spec_args())

    with pytest.raises(closed_loop_distillation.ClosedLoopTrainingUnavailableError) as exc:
        closed_loop_distillation.run_closed_loop_distillation_training(spec=spec)

    message = str(exc.value)
    assert "not implemented in this preflight pass" in message
    assert "Refusing to fall back" in message
    assert "teacher-feedback-bank" in message


def test_closed_loop_distillation_cli_full_train_returns_fail_closed(
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
    assert payload["error"] == "closed_loop_training_unavailable"
    assert payload["run_spec"]["run_id"] == "h0_extlqg_6d_closed_loop_distillation"
    assert "Refusing to fall back" in payload["message"]


def test_tracked_a378b34_run_spec_parses_and_validates() -> None:
    spec_path = closed_loop_distillation.DEFAULT_SPEC_PATH
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    closed_loop_distillation.validate_run_spec(spec)
    assert spec["expected_artifacts"]["tracked_run_spec"] == str(spec_path)
    assert spec["expected_artifacts"]["bulk_output_dir"] == (
        "_artifacts/a378b34/runs/h0_extlqg_6d_closed_loop_distillation"
    )
