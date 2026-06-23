"""Tests for the guided distillation training entry surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from rlrmp.train import guided_distillation


def _write_tiny_teacher_package(path: Path) -> None:
    state_dim = 4
    action_dim = 2
    feedback_dim = 3
    horizon = 4
    plant_a = np.eye(state_dim, dtype=np.float32)
    plant_a[0, 2] = 0.05
    plant_a[1, 3] = 0.05
    plant_b = np.zeros((state_dim, action_dim), dtype=np.float32)
    plant_b[2, 0] = 0.1
    plant_b[3, 1] = 0.1
    observation = np.zeros((feedback_dim, state_dim), dtype=np.float32)
    observation[0, 0] = 1.0
    observation[1, 1] = 1.0
    observation[2, 2] = 1.0
    gains = np.zeros((horizon, action_dim, state_dim), dtype=np.float32)
    gains[:, 0, 0] = 0.5
    gains[:, 1, 1] = 0.5
    np.savez(
        path,
        plant_A=plant_a,
        plant_B=plant_b,
        x0=np.array([0.1, -0.1, 0.0, 0.0], dtype=np.float32),
        hinf_controller_gains=gains,
        observation_matrix=observation,
    )


def test_distillation_entry_builds_9727d79_run_contract() -> None:
    args = argparse.Namespace(
        run_spec_output=str(guided_distillation.DEFAULT_SPEC_PATH),
        output_dir=guided_distillation.DEFAULT_OUTPUT_DIR,
        teacher_package=guided_distillation.DEFAULT_TEACHER_PACKAGE,
        teacher_manifest=guided_distillation.DEFAULT_TEACHER_MANIFEST,
        clean_action_weight=1.0,
        perturbation_response_weight=1.0,
        input_output_jvp_weight=0.25,
        rollout_anchor_weight=0.25,
        n_jvp_directions=16,
    )

    spec = guided_distillation.build_distillation_spec(args)

    assert spec["issue"] == "9727d79"
    assert spec["training_entry"]["script"] == "scripts/train_guided_distillation.py"
    assert spec["training_entry"]["loss_function"].endswith("guided_distillation_loss")
    assert spec["training_entry"]["full_train_status"] == "implemented_no_launch"
    assert spec["model_contract"]["initial_hidden_encoder"] is True
    assert spec["model_contract"]["force_filter_feedback"] is True
    assert spec["model_contract"]["hidden_size"] == 180
    assert spec["model_contract"]["batch_size"] == 64
    assert spec["model_contract"]["n_replicates"] == 5
    assert spec["model_contract"]["broad_epsilon_pgd_training"] is False
    assert spec["optimizer"]["controller_lr"] == pytest.approx(3e-3)
    assert spec["optimizer"]["gradient_clip_norm"] == pytest.approx(5.0)
    assert spec["training_schedule"]["phases"][0]["end_batch"] == 1500
    assert spec["training_schedule"]["phases"][1]["start_batch"] == 1500
    assert spec["training_schedule"]["phases"][1]["end_batch"] == 4000
    assert spec["training_schedule"]["phases"][2]["start_batch"] == 4000
    assert spec["training_schedule"]["phases"][2]["end_batch"] == 12000
    assert spec["distillation_surface"]["components"]["input_output_jvp"]["n_directions"] == 16
    assert "approximation" in spec["teacher_bank"]


def test_distillation_entry_smoke_loss_calls_guided_surface() -> None:
    smoke = guided_distillation.smoke_distillation_loss()

    assert smoke["finite"] is True
    assert smoke["loss_total"] > 0.0
    assert set(smoke["components"]) == {
        "clean_action",
        "input_output_jvp",
        "perturbation_response",
        "student_forced_rollout_anchor",
    }


def test_distillation_cli_dry_run_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    status = guided_distillation.main(["--dry-run", "--smoke-loss"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_spec"]["issue"] == "9727d79"
    assert payload["run_spec"]["smoke_loss"]["finite"] is True


def test_distillation_schedule_consumes_staged_forcing() -> None:
    args = argparse.Namespace(
        run_spec_output=str(guided_distillation.DEFAULT_SPEC_PATH),
        output_dir=guided_distillation.DEFAULT_OUTPUT_DIR,
        teacher_package=guided_distillation.DEFAULT_TEACHER_PACKAGE,
        teacher_manifest=guided_distillation.DEFAULT_TEACHER_MANIFEST,
        clean_action_weight=1.0,
        perturbation_response_weight=1.0,
        input_output_jvp_weight=0.25,
        rollout_anchor_weight=0.25,
        n_jvp_directions=16,
    )
    spec = guided_distillation.build_distillation_spec(args)

    assert guided_distillation.forcing_fraction_for_batch(spec, 0) == pytest.approx(0.0)
    assert guided_distillation.forcing_fraction_for_batch(spec, 1500) == pytest.approx(0.5)
    assert guided_distillation.forcing_fraction_for_batch(spec, 4000) == pytest.approx(0.9)


def test_distillation_cli_smoke_train_runs_real_trainer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    teacher_package = tmp_path / "teacher.npz"
    output_dir = tmp_path / "artifacts"
    run_spec = tmp_path / "run.json"
    _write_tiny_teacher_package(teacher_package)

    status = guided_distillation.main(
        [
            "--full-train",
            "--smoke-train",
            "--teacher-package",
            str(teacher_package),
            "--run-spec-output",
            str(run_spec),
            "--output-dir",
            str(output_dir),
            "--n-batches",
            "2",
            "--batch-size",
            "2",
            "--n-replicates",
            "1",
            "--hidden-size",
            "6",
            "--horizon",
            "4",
            "--n-jvp-directions",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["n_batches"] == 2
    assert payload["n_replicates"] == 1
    assert payload["final_loss_mean"] >= 0.0
    assert (output_dir / "training_summary.json").is_file()
    assert (output_dir / "loss_history.json").is_file()
    assert (output_dir / "student_model_rep0.eqx").is_file()
