"""Tests for nominal C&S-fidelity GRU run-spec preparation."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax.random as jr
import optax
from feedbax.training.train import TaskTrainer, make_delayed_cosine_schedule, train_pair

from rlrmp.analysis.cs_game_card import OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir
from rlrmp.train.cs_nominal_gru import (
    build_graph_bundle,
    build_hps,
    build_parser,
    derive_spec_dir,
    write_run_spec,
)


def _args(**overrides) -> argparse.Namespace:
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _where_train() -> dict[int, object]:
    def where_train_fn(model):
        net = model.nodes["net"]
        return (net.hidden, net.readout)

    return {0: where_train_fn}


def test_hps_uses_canonical_cs_nominal_task() -> None:
    hps = build_hps(_args())

    assert hps.method == "nominal-cs-gru"
    assert hps.dt == 0.01
    assert hps.task.type == "simple_reach"
    assert hps.task.n_steps == 60
    assert hps.task.eval_reach_length == 0.15
    assert hps.task.hold_epochs == []
    assert hps.task.p_catch_trial == 0.0
    assert hps.model.feedback_delay_steps == 5
    assert hps.model.feedback_noise_std == 0.0
    assert hps.model.population_structure.n_input_only == 0
    assert hps.model.population_structure.n_readout_only == 0
    assert hps.model.population_structure.n_recurrent_only == 0
    assert hps.model.population_structure.n_input_readout == hps.model.hidden_size
    assert hps.loss.weights.effector_hold_pos == 0.0
    assert hps.loss.weights.effector_hold_vel == 0.0
    assert hps.loss.weights.effector_pos_running == 1.0
    assert hps.loss.effector_pos_running_schedule == "cs_eq15_power6"
    assert hps.pert.std == 0.0


def test_graph_bundle_records_nominal_provenance() -> None:
    hps = build_hps(_args(smoke=True))
    bundle = build_graph_bundle(hps)

    assert bundle.training_spec["nominal_only"] is True
    assert bundle.training_spec["adversarial_phase"] == "none"
    assert bundle.manifest["game_card_provenance"]["horizon_steps"] == 60
    assert bundle.manifest["game_card_provenance"]["target_distance_m"] == 0.15
    assert (
        bundle.manifest["game_card_provenance"]["output_feedback_certificate_gamma_factor"]
        == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    )
    assert bundle.manifest["model_structure"]["controller_kind"] == "gru"
    assert bundle.manifest["model_structure"]["population_structure"] == {
        "n_input_only": 0,
        "n_readout_only": 0,
        "n_recurrent_only": 0,
        "n_input_readout": 4,
    }
    assert bundle.graph_spec.nodes["net"].params["hidden_size"] == 4


def test_derive_spec_dir_preserves_artifact_results_mirror() -> None:
    artifact = run_artifact_dir("18ae684", "cs_nominal_gru__local_smoke")
    assert derive_spec_dir(artifact) == run_spec_dir("18ae684", "cs_nominal_gru__local_smoke")


def test_dry_run_does_not_write_files(tmp_path: Path) -> None:
    spec_dir = tmp_path / "spec"
    args = _args(output_dir=str(tmp_path / "artifacts"), spec_dir=str(spec_dir), dry_run=True)

    result = write_run_spec(args)

    assert "run_spec" in result
    assert result["run_spec"]["mode"] == "dry_run"
    assert result["run_spec"]["nominal_only"] is True
    assert not spec_dir.exists()


def test_write_run_spec_creates_only_lightweight_spec_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
    )

    result = write_run_spec(args)

    run_path = Path(result["run_spec_path"])
    graph_path = Path(result["graph_spec_path"])
    manifest_path = Path(result["graph_manifest_path"])
    payload = json.loads(run_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert run_path == spec_dir / "run.json"
    assert graph_path == spec_dir / "model.graph.json"
    assert manifest_path == spec_dir / "model.graph.manifest.json"
    assert payload["schema_version"] == "rlrmp.cs_nominal_gru.v1"
    assert payload["model_summary"]["hidden_size"] == 4
    assert payload["model_summary"]["controller_kind"] == "gru"
    assert payload["model_summary"]["population_structure"] == {
        "n_input_only": 0,
        "n_readout_only": 0,
        "n_recurrent_only": 0,
        "n_input_readout": 4,
    }
    assert payload["training_summary"]["training_mode"] == "nominal"
    assert payload["game_card"]["plant"]["bw_shape"] == [48, 8]
    assert payload["task_timing"]["type"] == "simple_reach"
    assert payload["task_timing"]["movement_window"] == {
        "kind": "full_simple_reach_trial",
        "start_transition": 0,
        "end_transition": 59,
    }
    assert "same Cartesian metre coordinates" in payload["task_timing"]["coordinate_contract"]
    assert "one position target per transition" in payload["task_timing"]["time_axis_contract"]
    assert (
        "same-coordinate target sequence"
        in payload["loss_summary"]["simple_reach_position_loss_contract"]
    )
    assert "git" in payload["provenance"]
    assert manifest["training_spec"]["nominal_only"] is True
    assert not output_dir.exists()
    assert REPO_ROOT not in output_dir.parents


def test_setup_task_model_pair_trains_tiny_nominal_simple_reach_batch() -> None:
    args = _args(
        smoke=True,
        effector_pos_running=1.0,
        effector_final_vel=1.0,
        batch_size=2,
        n_train_batches=1,
    )
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    schedule = make_delayed_cosine_schedule(
        float(hps.learning_rate_0),
        constant_steps=0,
        total_steps=1,
    )
    optimizer = optax.inject_hyperparams(partial(optax.adamw, weight_decay=0.0))(
        learning_rate=schedule
    )
    trainer = TaskTrainer(optimizer=optimizer, checkpointing=False)

    trained, _history = train_pair(
        trainer,
        pair,
        n_batches=1,
        key=jr.PRNGKey(1),
        ensembled=True,
        loss_func=pair.task.loss_func,
        where_train=_where_train(),
        batch_size=2,
        log_step=1,
        disable_progress=True,
        verbose_progress=False,
    )

    assert trained is not None
    assert hps.task.type == "simple_reach"
    assert hps.task.n_steps == 60
    assert hps.loss.effector_pos_running_schedule == "cs_eq15_power6"
    assert pair.task.loss_func.weights["effector_pos_running"] == 1.0
    assert pair.task.loss_func.weights["effector_final_vel"] == 1.0
    assert set(pair.task.loss_func.terms) >= {
        "effector_pos_running",
        "effector_final_vel",
        "nn_output",
        "nn_hidden",
    }
