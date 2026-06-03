"""Tests for stochastic C&S-fidelity GRU run-spec preparation."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from feedbax.mechanics import LinearStateSpace
from feedbax.training.train import TaskTrainer, make_delayed_cosine_schedule, train_pair
from feedbax.types import TreeNamespace

from rlrmp.analysis.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    build_canonical_game,
)
from rlrmp.analysis.cs_released_simulation import default_cs_noise_covariances
from rlrmp.analysis.output_feedback import OutputFeedbackConfig
from rlrmp.cs_lss_gru import CS_EPSILON_DIM
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CsAnalyticalQrfLoss,
)
from rlrmp.modules.training.part2 import (
    CS_LSS_PLANT_BACKEND,
    LEGACY_CAUSAL_BACKEND_WARNING,
    LEGACY_CAUSAL_PLANT_BACKEND,
    _cs_lss_process_epsilon_factor,
)
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir
from rlrmp.train.cs_nominal_gru import (
    DEFAULT_STOCHASTIC_PRESET,
    build_graph_bundle,
    build_hps,
    build_parser,
    derive_spec_dir,
    run_full_training,
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
    assert hps.model.plant_backend == CS_LSS_PLANT_BACKEND
    assert hps.dt == 0.01
    assert hps.task.type == "fixed_simple_reach"
    assert hps.task.n_steps == 61
    assert hps.task.fixed_init_pos == [0.0, 0.0]
    assert hps.task.fixed_target_pos == [0.15, 0.0]
    assert hps.task.eval_reach_length == 0.15
    assert hps.task.hold_epochs == []
    assert hps.task.p_catch_trial == 0.0
    assert hps.model.feedback_delay_steps == 5
    assert hps.model.feedback_noise_std == 0.0
    assert hps.model.stochastic_preset == DEFAULT_STOCHASTIC_PRESET
    assert hps.model.sensory_noise_std > 0.0
    assert hps.model.additive_motor_noise_std == 1e-5
    assert hps.model.signal_dependent_motor_noise_std == 0.02
    assert hps.model.plant_process_force_noise_std > 0.0
    assert hps.model.population_structure.n_input_only == 0
    assert hps.model.population_structure.n_readout_only == 0
    assert hps.model.population_structure.n_recurrent_only == 0
    assert hps.model.population_structure.n_input_readout == hps.model.hidden_size
    assert hps.loss.weights.effector_hold_pos == 0.0
    assert hps.loss.weights.effector_hold_vel == 0.0
    assert hps.loss.weights.effector_pos_running == 1e6
    assert hps.loss.weights.effector_vel_running == 1e5
    assert hps.loss.weights.effector_terminal_pos == 1e6
    assert hps.loss.weights.effector_terminal_vel == 1e5
    assert hps.loss.weights.nn_output == 1.0
    assert hps.loss.weights.nn_hidden == 0.0
    assert hps.loss.objective == CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE
    assert hps.loss.effector_pos_running_schedule == "cs_eq15_power6"
    assert hps.pert.std == 0.0


def test_full_analytical_qrf_loss_requires_cs_lss_and_no_hidden_regularizer() -> None:
    with pytest.raises(ValueError, match="requires --plant-backend cs_lss"):
        build_hps(
            _args(
                loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
                plant_backend=LEGACY_CAUSAL_PLANT_BACKEND,
            )
        )

    with pytest.raises(ValueError, match="nn_hidden is not an analytical"):
        build_hps(
            _args(
                loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
                regularized_fidelity=True,
            )
        )


def test_full_analytical_qrf_loss_scores_non_pos_vel_state_and_command() -> None:
    hps = build_hps(_args(smoke=True, loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    loss = pair.task.loss_func.terms[CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE]

    assert isinstance(loss, CsAnalyticalQrfLoss)

    zeros = jnp.zeros((1, 60, 48), dtype=jnp.float64)
    zero_command = jnp.zeros((1, 60, 2), dtype=jnp.float64)
    base_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command),
        efferent=TreeNamespace(output=zero_command),
    )
    base_value = loss.term(base_states, trial, pair.model)

    force_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros.at[:, :, 4].set(2.0)),
        net=TreeNamespace(output=zero_command),
        efferent=TreeNamespace(output=zero_command),
    )
    command_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command.at[:, :, 0].set(3.0)),
        efferent=TreeNamespace(output=zero_command.at[:, :, 0].set(3.0)),
    )
    applied_only_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command),
        efferent=TreeNamespace(output=zero_command.at[:, :, 0].set(3.0)),
    )

    assert jnp.all(loss.term(force_states, trial, pair.model) > base_value)
    assert jnp.all(loss.term(command_states, trial, pair.model) > base_value)
    assert jnp.allclose(loss.term(applied_only_states, trial, pair.model), base_value)


def test_partial_net_force_filter_ablation_scores_net_output_and_force_filter() -> None:
    hps = build_hps(
        _args(smoke=True, loss_objective=CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE)
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))

    terms = pair.task.loss_func.terms
    assert "mechanics_force_filter" in terms
    assert pair.task.loss_func.weights["mechanics_force_filter"] == pytest.approx(1 / 6)

    zeros = jnp.zeros((1, 60, 48), dtype=jnp.float64)
    zero_command = jnp.zeros((1, 60, 2), dtype=jnp.float64)
    base_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command),
        efferent=TreeNamespace(output=zero_command),
    )
    net_command_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command.at[:, :, 0].set(3.0)),
        efferent=TreeNamespace(output=zero_command),
    )
    applied_only_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command),
        efferent=TreeNamespace(output=zero_command.at[:, :, 0].set(3.0)),
    )
    force_filter_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros.at[:, :, 4].set(2.0)),
        net=TreeNamespace(output=zero_command),
        efferent=TreeNamespace(output=zero_command),
    )

    base_output = terms["nn_output"].where(base_states)
    assert jnp.any(terms["nn_output"].where(net_command_states) != base_output)
    assert jnp.allclose(terms["nn_output"].where(applied_only_states), base_output)

    base_force = terms["mechanics_force_filter"].term(base_states, trial, pair.model)
    force_value = terms["mechanics_force_filter"].term(force_filter_states, trial, pair.model)
    assert force_value.shape == (1,)
    assert jnp.all(force_value > base_force)


def test_runtime_task_executes_sixty_fixed_cs_targets() -> None:
    hps = build_hps(_args(smoke=True))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    targets = trial.targets["mechanics.effector.pos"].value

    assert isinstance(pair.model.nodes["mechanics"], LinearStateSpace)
    assert pair.model.input_ports == ("input", "epsilon")
    assert trial.timeline.n_steps == 60
    assert trial.timeline.epoch_bounds.tolist() == [0, 60]
    assert targets.shape == (60, 2)
    assert jnp.allclose(trial.inits["mechanics.vector"][:4], jnp.zeros(4))
    assert trial.inputs["input"].shape == (60,)
    assert trial.inputs["epsilon"].shape == (60, CS_EPSILON_DIM)
    assert jnp.allclose(trial.inputs["epsilon"][:, :4], 0.0)
    assert jnp.any(jnp.abs(trial.inputs["epsilon"]) > 0.0)
    assert jnp.allclose(targets, jnp.broadcast_to(jnp.array([0.15, 0.0]), (60, 2)))


def test_lss_process_epsilon_factor_matches_cs_physical_covariance() -> None:
    plant, _schedule = build_canonical_game()
    covariances = default_cs_noise_covariances(plant, OutputFeedbackConfig())
    expected = covariances.process[:CS_EPSILON_DIM, :CS_EPSILON_DIM]
    factor = _cs_lss_process_epsilon_factor()

    assert factor.shape == (CS_EPSILON_DIM, CS_EPSILON_DIM)
    assert jnp.allclose(factor @ factor.T, expected, atol=1e-14)


def test_legacy_causal_backend_is_explicit_and_warns() -> None:
    hps = build_hps(_args(smoke=True, plant_backend=LEGACY_CAUSAL_PLANT_BACKEND))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    assert hps.model.plant_backend == LEGACY_CAUSAL_PLANT_BACKEND
    assert any(LEGACY_CAUSAL_BACKEND_WARNING in str(item.message) for item in caught)
    assert pair.model.input_ports != ("input", "epsilon")


def test_graph_bundle_records_nominal_provenance() -> None:
    hps = build_hps(_args(smoke=True))
    bundle = build_graph_bundle(hps)

    assert bundle.training_spec["nominal_only"] is True
    assert bundle.training_spec["plant_backend"] == CS_LSS_PLANT_BACKEND
    assert bundle.training_spec["adversarial_phase"] == "none"
    assert bundle.training_spec["certificate_lens"] == "input_output_map_certificate"
    assert bundle.manifest["game_card_provenance"]["horizon_steps"] == 60
    assert bundle.manifest["game_card_provenance"]["feedbax_task_n_steps"] == 61
    assert bundle.manifest["game_card_provenance"]["feedbax_control_cost_stages"] == 60
    assert bundle.manifest["game_card_provenance"]["init_pos_m"] == [0.0, 0.0]
    assert bundle.manifest["game_card_provenance"]["target_distance_m"] == 0.15
    assert (
        bundle.manifest["game_card_provenance"]["cost"]["feedbax_force_filter_state_cost"]
        == "not_available"
    )
    assert (
        bundle.manifest["game_card_provenance"]["output_feedback_certificate_gamma_factor"]
        == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    )
    assert bundle.manifest["model_structure"]["controller_kind"] == "gru"
    assert bundle.manifest["model_structure"]["plant_backend"] == CS_LSS_PLANT_BACKEND
    assert bundle.manifest["model_structure"]["exact_cs_linear_state_space"] is True
    assert bundle.manifest["model_structure"]["fixed_plant_parameters"] == [
        "nodes.mechanics.A",
        "nodes.mechanics.B",
        "nodes.mechanics.B_w",
    ]
    assert (
        bundle.manifest["model_structure"]["stochastic_runtime"]["state_diffusion"]
        == "mechanics.epsilon"
    )
    assert bundle.manifest["stochastic_preset"]["name"] == DEFAULT_STOCHASTIC_PRESET
    assert bundle.manifest["stochastic_preset"]["signal_dependent_motor_noise_std"] == 0.02
    assert (
        bundle.manifest["model_structure"]["plant_process"]["noise_timing"]
        == "mechanics.epsilon_sampled_task_input"
    )
    assert bundle.manifest["model_structure"]["plant_process"]["state_diffusion"] == (
        "mechanics.epsilon"
    )
    assert "physical-process/load epsilon" in (
        bundle.manifest["model_structure"]["plant_process"]["epsilon_bridge"]
    )
    assert bundle.manifest["model_structure"]["population_structure"] == {
        "n_input_only": 0,
        "n_readout_only": 0,
        "n_recurrent_only": 0,
        "n_input_readout": 4,
    }
    assert bundle.manifest["model_structure"]["certificate_lens"] == "input_output_map_certificate"
    assert bundle.manifest["model_structure"]["analytical_delay_augmented_state_input"] is False
    assert bundle.graph_spec.nodes["net"].params["hidden_size"] == 4


def test_derive_spec_dir_preserves_artifact_results_mirror() -> None:
    artifact = run_artifact_dir("30f2313", "cs_stochastic_gru__no_hidden_penalty")
    assert derive_spec_dir(artifact) == run_spec_dir(
        "30f2313",
        "cs_stochastic_gru__no_hidden_penalty",
    )


def test_dry_run_does_not_write_files(tmp_path: Path) -> None:
    spec_dir = tmp_path / "spec"
    args = _args(output_dir=str(tmp_path / "artifacts"), spec_dir=str(spec_dir), dry_run=True)

    result = write_run_spec(args)

    assert "run_spec" in result
    assert result["run_spec"]["mode"] == "dry_run"
    assert result["run_spec"]["loss_objective"] == CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE
    assert result["run_spec"]["nominal_only"] is True
    assert result["run_spec"]["fidelity_status"]["exact_fidelity"] is False
    assert result["run_spec"]["fidelity_status"]["exact_objective_terms"] is False
    assert result["run_spec"]["fidelity_status"]["objective_fidelity"]["omitted_terms"]
    assert result["run_spec"]["fidelity_status"]["exact_stochastic_rollout"] is False
    assert result["run_spec"]["fidelity_status"]["exact_stochastic_noise_sources"] is True
    assert result["run_spec"]["fidelity_status"]["exact_plant_matrices"] is True
    assert result["run_spec"]["fidelity_status"]["plant_backend"] == CS_LSS_PLANT_BACKEND
    assert "sensory Channel" in (
        result["run_spec"]["fidelity_status"]["temporary_stochastic_bridge"]
    )
    assert "signal-dependent motor Channel" in (
        result["run_spec"]["fidelity_status"]["temporary_stochastic_bridge"]
    )
    assert result["run_spec"]["fidelity_status"]["nn_hidden"] == 0.0
    assert result["run_spec"]["optimizer"]["schedule"] == "delayed_cosine"
    assert not spec_dir.exists()


def test_regularized_run_metadata_marks_non_exact_status(tmp_path: Path) -> None:
    args = _args(
        output_dir=str(tmp_path / "artifacts"),
        spec_dir=str(tmp_path / "spec"),
        dry_run=True,
        regularized_fidelity=True,
    )

    result = write_run_spec(args)

    fidelity = result["run_spec"]["fidelity_status"]
    assert fidelity["exact_fidelity"] is False
    assert fidelity["exact_objective_terms"] is False
    assert fidelity["objective_fidelity"]["extra_terms"][0]["term"] == "nn_hidden"
    assert fidelity["regularized_pair"] is True
    assert fidelity["regularizer"] == "nn_hidden"
    assert fidelity["nn_hidden"] == 1e-5


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
    graph_path = result["graph_spec_path"]
    manifest_path = Path(result["graph_manifest_path"])
    payload = json.loads(run_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert run_path == spec_dir / "run.json"
    assert graph_path is None
    assert manifest_path == spec_dir / "model.graph.manifest.json"
    assert not (spec_dir / "model.graph.json").exists()
    assert payload["schema_version"] == "rlrmp.cs_stochastic_gru.v1"
    assert payload["issue"] == "30f2313"
    assert payload["model_summary"]["hidden_size"] == 4
    assert payload["model_summary"]["controller_kind"] == "gru"
    assert payload["model_summary"]["plant_backend"] == CS_LSS_PLANT_BACKEND
    assert payload["model_summary"]["exact_cs_linear_state_space"] is True
    assert payload["feedbax_graph"]["graph_spec_path"] is None
    assert payload["feedbax_graph"]["graph_export_status"] == "unavailable"
    assert manifest["graph_export"]["status"] == "unavailable"
    assert payload["stochastic_preset"]["name"] == DEFAULT_STOCHASTIC_PRESET
    assert payload["stochastic_preset"]["source_contract"]["contract"] == (
        "cs_released_stochastic_v1"
    )
    assert payload["model_summary"]["stochastic_runtime"]["sensory_noise_std"] > 0.0
    assert payload["model_summary"]["stochastic_runtime"]["additive_motor_noise_std"] == 1e-5
    assert (
        payload["model_summary"]["stochastic_runtime"]["signal_dependent_motor_noise_std"]
        == 0.02
    )
    assert payload["model_summary"]["stochastic_runtime"]["plant_process_force_noise_std"] > 0.0
    assert payload["model_summary"]["plant_process"]["state_diffusion"] == "mechanics.epsilon"
    assert "physical-process/load epsilon" in (
        payload["model_summary"]["plant_process"]["epsilon_bridge"]
    )
    assert payload["model_summary"]["certificate_lens"] == "input_output_map_certificate"
    assert payload["model_summary"]["analytical_delay_augmented_state_input"] is False
    assert payload["model_summary"]["population_structure"] == {
        "n_input_only": 0,
        "n_readout_only": 0,
        "n_recurrent_only": 0,
        "n_input_readout": 4,
    }
    assert payload["training_summary"]["training_mode"] == "nominal"
    assert payload["game_card"]["plant"]["bw_shape"] == [48, 8]
    assert payload["task_timing"]["type"] == "fixed_simple_reach"
    assert payload["task_timing"]["n_steps"] == 61
    assert payload["task_timing"]["control_cost_stages"] == 60
    assert payload["task_timing"]["fixed_init_pos"] == [0.0, 0.0]
    assert payload["task_timing"]["fixed_target_pos"] == [0.15, 0.0]
    assert payload["task_timing"]["extra_inputs"] == ["input", "epsilon"]
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
    assert payload["loss_summary"]["objective_profile"] == CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE
    assert payload["loss_summary"]["active_cs_terms"]["stage_position"]["scale"] == 1e6
    assert payload["loss_summary"]["active_cs_terms"]["stage_velocity"]["scale"] == 1e5
    assert payload["loss_summary"]["active_cs_terms"]["control"]["scale"] == 1.0
    assert payload["loss_summary"]["active_cs_terms"]["terminal_position"]["scale"] == 1e6
    assert payload["loss_summary"]["active_cs_terms"]["terminal_velocity"]["scale"] == 1e5
    assert payload["loss_summary"]["force_filter_state_cost"] == "not_available"
    assert payload["loss_summary"]["hidden_regularizer"]["scale"] == 0.0
    assert "git" in payload["provenance"]
    assert manifest["training_spec"]["nominal_only"] is True
    assert not output_dir.exists()
    assert REPO_ROOT not in output_dir.parents


def test_full_analytical_qrf_run_spec_records_exact_objective_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())

    assert payload["loss_objective"] == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
    assert payload["training_summary"]["loss_objective"] == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
    assert payload["loss_summary"]["objective_profile"] == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
    assert payload["loss_summary"]["matrix_shapes"] == {
        "Q": [60, 48, 48],
        "R": [60, 2, 2],
        "Q_f": [48, 48],
    }
    assert payload["loss_summary"]["force_filter_state_cost"].startswith("included")
    assert payload["loss_summary"]["disturbance_integrator_state_cost"].startswith("included")
    assert payload["fidelity_status"]["exact_objective_terms"] is True
    assert payload["fidelity_status"]["objective_fidelity"]["omitted_terms"] == []
    assert payload["fidelity_status"]["objective_fidelity"]["extra_terms"] == []


def test_partial_net_force_filter_run_spec_records_ablation_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        loss_objective=CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())

    assert payload["loss_objective"] == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE
    assert payload["loss_summary"]["objective_profile"] == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE
    assert payload["loss_summary"]["active_cs_terms"]["control"]["state_key"] == "states.net.output"
    assert payload["loss_summary"]["active_cs_terms"]["force_filter"]["scale"] == pytest.approx(1 / 6)
    assert payload["loss_summary"]["disturbance_integrator_state_cost"] == "omitted_in_this_ablation"
    fidelity = payload["fidelity_status"]["objective_fidelity"]
    assert "intended_command_quadratic_net_output" in fidelity["implemented_terms"]
    assert "running_force_filter_state_cost" in fidelity["implemented_terms"]
    assert payload["game_card"]["cost"]["feedbax_force_filter_state_cost"].startswith("included")


def test_write_run_spec_honors_issue_override(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        issue="3b2af27",
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())

    assert payload["issue"] == "3b2af27"


def test_full_training_smoke_writes_checkpoint_and_final_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        n_train_batches=4,
        batch_size=2,
        n_replicates=2,
        hidden_size=4,
        full_train=True,
        resume=True,
        checkpoint_interval_batches=2,
        controller_lr=1e-3,
        lr_warmup_batches=1,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        log_step=1,
        disable_progress=True,
        quiet_progress=True,
    )
    commits = 0

    def commit() -> None:
        nonlocal commits
        commits += 1

    result = run_full_training(args, volume_commit=commit)

    checkpoint_latest = output_dir / "checkpoints" / "checkpoint_latest"
    checkpoint_2 = output_dir / "checkpoints" / "checkpoint_0000002"
    checkpoint_4 = output_dir / "checkpoints" / "checkpoint_0000004"
    metadata = json.loads((checkpoint_latest / "metadata.json").read_text())
    summary = json.loads((output_dir / "training_summary.json").read_text())
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())

    assert result["completed_batches"] == 4
    assert Path(result["final_model_path"]) == output_dir / "trained_model.eqx"
    assert Path(result["training_history_path"]) == output_dir / "training_history.eqx"
    assert checkpoint_latest.exists()
    assert checkpoint_2.exists()
    assert checkpoint_4.exists()
    assert metadata["completed_batches"] == 4
    assert metadata["next_prng_key"]
    assert metadata["run_spec"]["mode"] == "full_train"
    assert metadata["run_spec"]["schema_version"] == "rlrmp.cs_stochastic_gru.v1"
    assert (checkpoint_latest / "model.eqx").exists()
    assert (checkpoint_latest / "optimizer_state.eqx").exists()
    assert (output_dir / "trained_model.eqx").exists()
    assert (output_dir / "training_history.eqx").exists()
    assert (output_dir / "training_diagnostics.npz").exists()
    assert (output_dir / "training_diagnostics.json").exists()
    assert (output_dir / "history_chunks" / "history_0000002.eqx").exists()
    assert (output_dir / "history_chunks" / "history_0000004.eqx").exists()
    assert summary["latest_checkpoint"] == str(checkpoint_latest)
    assert summary["training_diagnostics"]["enabled"] is True
    assert summary["training_diagnostics"]["written"] is True
    assert summary["training_diagnostics"]["sidecar_path"] == str(
        output_dir / "training_diagnostics.npz"
    )
    assert diagnostics_manifest["completed_batches"] == 4
    assert diagnostics_manifest["gradient_clip_active"] is False
    assert diagnostics_manifest["training_history_path"] == str(output_dir / "training_history.eqx")
    assert "optimizer_gradient_norm_pre_clip" in diagnostics_manifest["arrays"]
    assert "optimizer_update_parameter_norm_ratio" in diagnostics_manifest["arrays"]
    assert "optimizer_learning_rate" in diagnostics_manifest["arrays"]
    assert "train_loss__total" in diagnostics_manifest["arrays"]
    assert "validation_loss__total" in diagnostics_manifest["arrays"]
    with np.load(output_dir / "training_diagnostics.npz") as diagnostics:
        assert diagnostics["batch_index"].tolist() == [0, 1, 2, 3]
        assert diagnostics["optimizer_gradient_norm_pre_clip"].shape == (4, 2)
        assert np.isfinite(diagnostics["optimizer_gradient_norm_pre_clip"]).all()
        assert diagnostics["optimizer_gradient_clipped"].shape == (4, 2)
        assert diagnostics["optimizer_clipping_fraction"].shape == (4,)
        assert diagnostics["optimizer_update_norm"].shape == (4, 2)
        assert np.isfinite(diagnostics["optimizer_update_norm"]).all()
        assert diagnostics["optimizer_parameter_norm"].shape == (4, 2)
        assert np.isfinite(diagnostics["optimizer_parameter_norm"]).all()
        assert diagnostics["optimizer_update_parameter_norm_ratio"].shape == (4, 2)
        assert np.isfinite(diagnostics["optimizer_update_parameter_norm_ratio"]).all()
        assert diagnostics["optimizer_learning_rate"].shape == (4, 2)
        lr_trace = diagnostics["optimizer_learning_rate"][:, 0]
        assert np.isclose(lr_trace[0], 1e-4)
        assert np.isclose(lr_trace[1], 1e-3)
        assert np.all(np.diff(lr_trace[1:]) < 0)
        assert diagnostics["train_loss__total"].shape == (4, 2)
        assert np.isfinite(diagnostics["train_loss__total"]).all()
        assert diagnostics["validation_loss__total"].shape == (4, 2)
        assert np.isfinite(diagnostics["validation_loss__total"]).all()
    assert summary["training_duration_seconds"] > 0
    assert summary["training_batches_per_second"] > 0
    assert len(summary["chunks"]) == 2
    assert summary["chunks"][0]["chunk_batches"] == 2
    assert summary["chunks"][0]["duration_seconds"] > 0
    assert summary["chunks"][0]["batches_per_second"] > 0
    assert commits == 3


def test_full_training_smoke_can_disable_diagnostics(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        n_train_batches=1,
        batch_size=2,
        n_replicates=1,
        hidden_size=4,
        full_train=True,
        checkpoint_interval_batches=1,
        disable_progress=True,
        quiet_progress=True,
        training_diagnostics=False,
    )

    result = run_full_training(args)
    run_spec = json.loads((spec_dir / "run.json").read_text())
    summary = json.loads((output_dir / "training_summary.json").read_text())

    assert result["completed_batches"] == 1
    assert run_spec["training_diagnostics"]["enabled"] is False
    assert run_spec["training_summary"]["training_diagnostics"]["enabled"] is False
    assert summary["training_diagnostics"]["enabled"] is False
    assert summary["training_diagnostics"]["sidecar_path"] is None
    assert not (output_dir / "training_diagnostics.npz").exists()
    assert not (output_dir / "training_diagnostics.json").exists()


def test_setup_task_model_pair_trains_tiny_nominal_simple_reach_batch() -> None:
    args = _args(
        smoke=True,
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
    assert isinstance(pair.model.nodes["mechanics"], LinearStateSpace)
    assert pair.model.input_ports == ("input", "epsilon")
    assert hps.task.type == "fixed_simple_reach"
    assert hps.task.n_steps == 61
    assert hps.loss.effector_pos_running_schedule == "cs_eq15_power6"
    assert pair.task.loss_func.weights["effector_pos_running"] == 1e6
    assert pair.task.loss_func.weights["effector_vel_running"] == 1e5
    assert pair.task.loss_func.weights["effector_terminal_pos"] == 1e6
    assert pair.task.loss_func.weights["effector_terminal_vel"] == 1e5
    assert pair.task.loss_func.weights["nn_output"] == 1.0
    assert "nn_hidden" not in pair.task.loss_func.weights
    assert set(pair.task.loss_func.terms) >= {
        "effector_pos_running",
        "effector_vel_running",
        "effector_terminal_pos",
        "effector_terminal_vel",
        "nn_output",
    }


def test_lss_backend_excludes_fixed_plant_matrices_from_training() -> None:
    hps = build_hps(_args(smoke=True))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    where_train = _where_train()[0]
    from feedbax.train import filter_spec_leaves, get_model_parameters

    where_train_spec = filter_spec_leaves(pair.model, where_train)
    trainable = get_model_parameters(pair.model, where_train_spec)
    trainable_arrays = [leaf for leaf in jax.tree.leaves(trainable) if eqx.is_array(leaf)]

    assert trainable.nodes["mechanics"].A is None
    assert trainable.nodes["mechanics"].B is None
    assert trainable.nodes["mechanics"].B_w is None
    assert any(leaf.shape[-2:] == (12, 5) for leaf in trainable_arrays)
    assert any(leaf.shape[-2:] == (2, 4) for leaf in trainable_arrays)
