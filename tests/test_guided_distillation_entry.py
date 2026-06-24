"""Tests for the guided distillation training entry surface."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_cookbook import load_with_hyperparameters

from rlrmp.train import guided_distillation


def _setup_task_model_pair(hps, *, key):
    import rlrmp.analysis  # noqa: F401
    from rlrmp.train.task_model import setup_task_model_pair

    return setup_task_model_pair(hps, key=key)


def _write_tiny_teacher_package(path: Path) -> None:
    state_dim = 4
    action_dim = 2
    feedback_dim = 6
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
    observation[3, 3] = 1.0
    hinf_gains = np.zeros((horizon, action_dim, state_dim), dtype=np.float32)
    hinf_gains[:, 0, 0] = 0.5
    hinf_gains[:, 1, 1] = 0.5
    extlqg_gains = np.zeros((horizon, action_dim, state_dim), dtype=np.float32)
    extlqg_gains[:, 0, 0] = 0.25
    extlqg_gains[:, 1, 1] = 0.75
    np.savez(
        path,
        plant_A=plant_a,
        plant_B=plant_b,
        x0=np.array([0.1, -0.1, 0.0, 0.0], dtype=np.float32),
        extlqg_controller_gains=extlqg_gains,
        hinf_controller_gains=hinf_gains,
        observation_matrix=observation,
    )


def _default_spec_args(**overrides) -> argparse.Namespace:
    values = {
        "run_spec_output": str(guided_distillation.DEFAULT_SPEC_PATH),
        "output_dir": guided_distillation.DEFAULT_OUTPUT_DIR,
        "teacher_package": guided_distillation.DEFAULT_TEACHER_PACKAGE,
        "teacher_manifest": guided_distillation.DEFAULT_TEACHER_MANIFEST,
        "teacher_gains_key": guided_distillation.DEFAULT_TEACHER_GAINS_KEY,
        "clean_action_weight": 1.0,
        "perturbation_response_weight": 1.0,
        "input_output_jvp_weight": 0.25,
        "rollout_anchor_weight": 0.25,
        "n_jvp_directions": 16,
        "checkpoint": True,
        "checkpoint_interval_batches": 500,
        "trainable_dtype": guided_distillation.DEFAULT_TRAINABLE_DTYPE,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _tiny_hps(spec: dict, *, trainable_dtype: str | None = None):
    return guided_distillation._standard_hps_from_spec(
        spec,
        n_replicates=1,
        hidden_size=6,
        batch_size=2,
        n_batches=1,
        controller_lr=3e-3,
        lr_warmup_batches=500,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        gradient_clip_norm=5.0,
        trainable_dtype=trainable_dtype
        or spec.get("model_contract", {}).get("trainable_dtype", "float32"),
    )


def test_distillation_entry_builds_9727d79_run_contract() -> None:
    args = _default_spec_args()

    spec = guided_distillation.build_distillation_spec(args)

    assert spec["issue"] == "9727d79"
    assert spec["run_id"] == "h0_extlqg_6d_standard_graph_distillation"
    assert spec["artifact_output_dir"] == (
        "_artifacts/9727d79/runs/h0_extlqg_6d_standard_graph_distillation"
    )
    assert spec["teacher_contract"]["primary_teacher"] == "6d_output_feedback_extlqg"
    assert spec["teacher_contract"]["diagnostic_teacher"] == "6d_output_feedback_hinf"
    assert spec["teacher_contract"]["teacher_gains_key"] == "extlqg_controller_gains"
    assert spec["teacher_bank"]["teacher"] == "extlqg_controller_gains"
    assert spec["teacher_bank"]["teacher_gains_key"] == "extlqg_controller_gains"
    assert spec["training_entry"]["script"] == "scripts/train_guided_distillation.py"
    assert spec["training_entry"]["loss_function"].endswith("guided_distillation_loss")
    assert spec["training_entry"]["full_train_status"] == "implemented_no_launch"
    assert spec["training_entry"]["replicate_execution"] == (
        "vectorized with eqx.filter_vmap over the replicate axis"
    )
    assert spec["checkpointing"]["enabled"] is True
    assert spec["checkpointing"]["interval_batches"] == 500
    assert spec["checkpointing"]["resume_flag"] == "--resume"
    assert spec["model_contract"]["initial_hidden_encoder"] is True
    assert spec["model_contract"]["force_filter_feedback"] is True
    assert spec["model_contract"]["hidden_size"] == 180
    assert spec["model_contract"]["controller_input_dim"] == 6
    assert spec["model_contract"]["student_action_history_input"] is False
    assert spec["model_contract"]["batch_size"] == 64
    assert spec["model_contract"]["n_replicates"] == 5
    assert spec["model_contract"]["vectorized_replicates"] is True
    assert spec["model_contract"]["trainable_dtype"] == "float32"
    assert spec["hps"]["model"]["trainable_dtype"] == "float32"
    assert spec["model_contract"]["broad_epsilon_pgd_training"] is False
    assert spec["optimizer"]["controller_lr"] == pytest.approx(3e-3)
    assert spec["optimizer"]["gradient_clip_norm"] == pytest.approx(5.0)
    assert spec["optimizer"]["lr_cosine_alpha"] == pytest.approx(0.01)
    assert spec["hps"]["cosine_annealing_alpha"] == pytest.approx(0.01)
    assert spec["training_schedule"]["phases"][0]["end_batch"] == 1500
    assert spec["training_schedule"]["phases"][1]["start_batch"] == 1500
    assert spec["training_schedule"]["phases"][1]["end_batch"] == 4000
    assert spec["training_schedule"]["phases"][2]["start_batch"] == 4000
    assert spec["training_schedule"]["phases"][2]["end_batch"] == 12000
    assert spec["distillation_surface"]["components"]["input_output_jvp"]["n_directions"] == 16
    assert spec["distillation_surface"]["student_action_history_input"] is False
    assert spec["launch_ready_summary"]["requires_user_confirmation_before_billable_run"] is True
    assert "approximation" in spec["teacher_bank"]


def test_standard_graph_distillation_trainable_leaves_default_to_float32() -> None:
    spec = guided_distillation.build_distillation_spec(_default_spec_args())
    hps = _tiny_hps(spec)
    raw_model = guided_distillation._init_standard_model_ensemble(
        hps=hps,
        key=jax.random.PRNGKey(11),
    )
    where_train_spec = guided_distillation._where_train_spec(raw_model)

    model = guided_distillation._enforce_trainable_float_dtype(
        raw_model,
        where_train_spec,
        jnp.dtype(jnp.float32),
        context="test standard guided-distillation model",
    )

    leaves = guided_distillation._trainable_float_leaves(model, where_train_spec)
    assert leaves
    assert {leaf.dtype for leaf in leaves} == {jnp.dtype(jnp.float32)}
    assert model.nodes["net"].net.hidden.weight_ih.dtype == jnp.dtype(jnp.float32)
    assert model.nodes["net"].net.readout.weight.dtype == jnp.dtype(jnp.float32)
    assert model.nodes["net"].h0_encoder.weight.dtype == jnp.dtype(jnp.float32)


def test_standard_graph_distillation_preserves_explicit_float64_request() -> None:
    if not jax.config.jax_enable_x64:
        pytest.skip("explicit float64 trainable dtype requires jax_enable_x64")
    spec = guided_distillation.build_distillation_spec(
        _default_spec_args(trainable_dtype="float64")
    )
    hps = _tiny_hps(spec, trainable_dtype="float64")
    raw_model = guided_distillation._init_standard_model_ensemble(
        hps=hps,
        key=jax.random.PRNGKey(12),
    )
    where_train_spec = guided_distillation._where_train_spec(raw_model)

    model = guided_distillation._enforce_trainable_float_dtype(
        raw_model,
        where_train_spec,
        jnp.dtype(jnp.float64),
        context="test float64 guided-distillation model",
    )

    leaves = guided_distillation._trainable_float_leaves(model, where_train_spec)
    assert leaves
    assert {leaf.dtype for leaf in leaves} == {jnp.dtype(jnp.float64)}
    assert spec["model_contract"]["trainable_dtype"] == "float64"
    assert spec["hps"]["model"]["trainable_dtype"] == "float64"


def test_corrected_distillation_default_paths_do_not_reuse_legacy_run() -> None:
    assert guided_distillation.RUN_ID == "h0_extlqg_6d_standard_graph_distillation"
    assert (
        guided_distillation.HINF_STANDARD_GRAPH_RUN_ID == "h0_hinf_6d_standard_graph_distillation"
    )
    assert guided_distillation.LEGACY_ACTION_HISTORY_RUN_ID == "h0_hinf_6d_guided_distillation"
    assert guided_distillation.DEFAULT_SPEC_PATH == Path(
        "results/9727d79/runs/h0_extlqg_6d_standard_graph_distillation.json"
    )
    assert guided_distillation.DEFAULT_OUTPUT_DIR == (
        "_artifacts/9727d79/runs/h0_extlqg_6d_standard_graph_distillation"
    )
    assert guided_distillation.DEFAULT_TEACHER_GAINS_KEY == "extlqg_controller_gains"


def test_extlqg_teacher_selection_uses_distinct_package_key(tmp_path: Path) -> None:
    teacher_package = tmp_path / "teacher.npz"
    _write_tiny_teacher_package(teacher_package)

    extlqg = guided_distillation.load_teacher_package(
        teacher_package,
        teacher_gains_key="extlqg_controller_gains",
    )
    hinf = guided_distillation.load_teacher_package(
        teacher_package,
        teacher_gains_key="hinf_controller_gains",
    )

    np.testing.assert_allclose(
        np.asarray(extlqg["controller_gains"]),
        np.asarray(extlqg["extlqg_controller_gains"]),
    )
    np.testing.assert_allclose(
        np.asarray(hinf["controller_gains"]),
        np.asarray(hinf["hinf_controller_gains"]),
    )
    assert not np.allclose(
        np.asarray(extlqg["controller_gains"]),
        np.asarray(hinf["controller_gains"]),
    )


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
        checkpoint=True,
        checkpoint_interval_batches=500,
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
            "2",
            "--hidden-size",
            "6",
            "--horizon",
            "4",
            "--n-jvp-directions",
            "2",
            "--checkpoint-interval-batches",
            "1",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert payload["n_batches"] == 2
    assert payload["completed_batches"] == 2
    assert payload["n_replicates"] == 2
    assert payload["vectorized_replicates"] is True
    assert payload["checkpointing"] is True
    assert payload["latest_checkpoint"] is not None
    assert payload["final_loss_mean"] >= 0.0
    assert "phase=guided_distillation_vectorized" in captured.err
    assert "replicates=2" in captured.err
    assert "phase=guided_distillation_rep0" not in captured.err
    assert (output_dir / "training_summary.json").is_file()
    assert (output_dir / "loss_history.json").is_file()
    assert (output_dir / "checkpoints" / "checkpoint_latest").exists()
    assert (output_dir / "checkpoints" / "checkpoint_0000002" / "model.eqx").is_file()
    assert (output_dir / "checkpoints" / "checkpoint_0000002" / "optimizer_state.eqx").is_file()
    assert (output_dir / "checkpoints" / "checkpoint_0000002" / "batch_keys.npy").is_file()
    assert not (output_dir / "checkpoints" / "checkpoint_0000002" / "models.eqx").exists()
    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    histories = json.loads((output_dir / "loss_history.json").read_text(encoding="utf-8"))
    assert summary["n_replicates"] == 2
    assert summary["run_id"] == "h0_extlqg_6d_standard_graph_distillation"
    assert summary["completed_batches"] == 2
    assert summary["trainable_dtype"] == "float32"
    assert summary["checkpointing"]["enabled"] is True
    assert summary["vectorized_replicates"] is True
    assert len(histories) == 2
    assert {history[0]["replicate"] for history in histories} == {0, 1}
    assert all(len(history) == 2 for history in histories)
    assert (output_dir / "trained_model.eqx").is_file()
    assert not (output_dir / "student_model_rep0.eqx").exists()
    assert not (output_dir / "student_model_rep1.eqx").exists()

    spec = json.loads((output_dir / "run_spec_snapshot.json").read_text(encoding="utf-8"))
    assert spec["run_id"] == "h0_extlqg_6d_standard_graph_distillation"
    assert spec["teacher_bank"]["teacher_gains_key"] == "extlqg_controller_gains"
    assert spec["model_contract"]["controller_input_dim"] == 6
    assert spec["model_contract"]["student_action_history_input"] is False
    assert spec["model_contract"]["trainable_dtype"] == "float32"
    hps = guided_distillation._standard_hps_from_spec(
        spec,
        n_replicates=2,
        hidden_size=6,
        batch_size=2,
        n_batches=2,
        controller_lr=3e-3,
        lr_warmup_batches=500,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        gradient_clip_norm=5.0,
        trainable_dtype="float32",
    )
    model, _hyperparameters = load_with_hyperparameters(
        output_dir / "trained_model.eqx",
        setup_func=lambda key, **_kwargs: _setup_task_model_pair(hps, key=key).model,
    )
    assert model.nodes["net"].net.hidden.weight_ih.shape == (2, 18, 6)


def test_action_history_context_does_not_enter_student_forward_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        run_spec_output=str(guided_distillation.DEFAULT_SPEC_PATH),
        output_dir=guided_distillation.DEFAULT_OUTPUT_DIR,
        teacher_package=guided_distillation.DEFAULT_TEACHER_PACKAGE,
        teacher_manifest=guided_distillation.DEFAULT_TEACHER_MANIFEST,
        clean_action_weight=1.0,
        perturbation_response_weight=1.0,
        input_output_jvp_weight=0.25,
        rollout_anchor_weight=0.25,
        n_jvp_directions=2,
        checkpoint=True,
        checkpoint_interval_batches=500,
    )
    spec = guided_distillation.build_distillation_spec(args)
    hps = guided_distillation._standard_hps_from_spec(
        spec,
        n_replicates=1,
        hidden_size=6,
        batch_size=2,
        n_batches=1,
        controller_lr=3e-3,
        lr_warmup_batches=500,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        gradient_clip_norm=5.0,
    )
    model = guided_distillation._single_replicate_model(
        guided_distillation._init_standard_model_ensemble(hps=hps, key=jax.random.PRNGKey(1)),
        replicate_index=0,
        n_replicates=1,
    )
    batch = {
        "feedback_history": jnp.ones((2, 4, 6), dtype=jnp.float32),
        "teacher_actions": jnp.zeros((2, 4, 2), dtype=jnp.float32),
        "perturbation_feedback_history": jnp.ones((2, 4, 6), dtype=jnp.float32) * 1.1,
        "feedback_directions": jnp.ones((2, 2, 4, 6), dtype=jnp.float32) * 0.01,
        "action_directions": jnp.ones((2, 2, 4, 2), dtype=jnp.float32) * 0.02,
        "feedback_gains": jnp.zeros((2, 4, 2, 6), dtype=jnp.float32),
    }
    observed = {}

    def fake_guided_distillation_loss(**kwargs):
        student_policy = kwargs["student_policy"]
        feedback = kwargs["feedback_history"]
        actions_a = jnp.zeros_like(kwargs["action_history"])
        actions_b = jnp.ones_like(kwargs["action_history"])
        np.testing.assert_allclose(
            np.asarray(student_policy(feedback, actions_a)),
            np.asarray(student_policy(feedback, actions_b)),
            rtol=0.0,
            atol=0.0,
        )
        observed["checked"] = True
        return SimpleNamespace(total=jnp.array(0.0), components={"clean_action": jnp.array(0.0)})

    monkeypatch.setattr(
        guided_distillation,
        "guided_distillation_loss",
        fake_guided_distillation_loss,
    )

    loss, components = guided_distillation._loss_for_batch(
        model,
        batch,
        guided_distillation.cs_h0_distillation_config(n_jvp_directions=2),
        student_forcing_fraction=1.0,
    )

    assert float(loss) == 0.0
    assert set(components) == {"clean_action"}
    assert observed["checked"] is True


def test_distillation_cli_smoke_train_resumes_from_latest_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    teacher_package = tmp_path / "teacher.npz"
    output_dir = tmp_path / "artifacts"
    run_spec = tmp_path / "run.json"
    _write_tiny_teacher_package(teacher_package)

    common_args = [
        "--full-train",
        "--smoke-train",
        "--teacher-package",
        str(teacher_package),
        "--run-spec-output",
        str(run_spec),
        "--output-dir",
        str(output_dir),
        "--n-batches",
        "3",
        "--batch-size",
        "2",
        "--n-replicates",
        "2",
        "--hidden-size",
        "6",
        "--horizon",
        "4",
        "--n-jvp-directions",
        "2",
        "--checkpoint-interval-batches",
        "1",
    ]

    first_status = guided_distillation.main([*common_args, "--stop-after-batches", "1"])
    first_payload = json.loads(capsys.readouterr().out)

    assert first_status == 0
    assert first_payload["completed_batches"] == 1
    assert (output_dir / "checkpoints" / "checkpoint_0000001" / "metadata.json").is_file()

    second_status = guided_distillation.main([*common_args, "--resume"])
    captured = capsys.readouterr()
    second_payload = json.loads(captured.out)

    assert second_status == 0
    assert second_payload["completed_batches"] == 3
    assert second_payload["resumed_from"] is not None
    assert (output_dir / "checkpoints" / "checkpoint_0000003" / "metadata.json").is_file()
    assert (output_dir / "checkpoints" / "checkpoint_0000003" / "model.eqx").is_file()
    histories = json.loads((output_dir / "loss_history.json").read_text(encoding="utf-8"))
    assert len(histories) == 2
    assert all([entry["batch"] for entry in history] == [1, 2, 3] for history in histories)
    assert "phase=guided_distillation_vectorized" in captured.err


def test_guided_distillation_has_no_production_legacy_policy_references() -> None:
    production_roots = [Path("src"), Path("scripts")]
    forbidden = ("Guided" + "GRUPolicy", "guided" + "_policy")
    matches = []
    for root in production_roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                matches.append(path)

    assert matches == []


def test_completed_artifact_migration_writes_standard_loadable_model(tmp_path: Path) -> None:
    source_run = Path("_artifacts/9727d79/runs/h0_hinf_6d_guided_distillation")
    source_checkpoint = source_run / "checkpoints" / "checkpoint_0012000"
    if not (source_checkpoint / "models.eqx").is_file():
        pytest.skip("completed 9727d79 legacy checkpoint is not present")

    artifact_dir = tmp_path / "run"
    checkpoint_dir = artifact_dir / "checkpoints" / "checkpoint_0012000"
    checkpoint_dir.parent.mkdir(parents=True)
    shutil.copytree(source_checkpoint, checkpoint_dir)
    os.symlink("checkpoint_0012000", checkpoint_dir.parent / "checkpoint_latest")
    shutil.copy2(source_run / "training_summary.json", artifact_dir / "training_summary.json")

    module_path = Path("results/9727d79/scripts/migrate_distillation_artifacts.py")
    spec = importlib.util.spec_from_file_location("migrate_distillation_artifacts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    status = module.main(
        [
            "--run-spec",
            str(
                guided_distillation.run_spec_path_for(
                    guided_distillation.HINF_STANDARD_GRAPH_RUN_ID
                )
            ),
            "--artifact-dir",
            str(artifact_dir),
            "--no-migrate-checkpoints",
        ]
    )

    assert status == 0
    assert (artifact_dir / "trained_model.eqx").is_file()
    manifest = json.loads(
        (artifact_dir / "standard_model_migration_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["standard_loader_smoke"]["load_status"] == "ok"
    assert manifest["projection"]["status"] == "standard_loadable_lossy_projection"
    assert manifest["projection"]["dropped_legacy_action_history_columns"] == [6, 7]
