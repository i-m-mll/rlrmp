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
from rlrmp.cs_lss_gru import CS_EPSILON_DIM, TargetRelativeDelayedFeedback
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
from rlrmp.train.cs_perturbation_training import (
    GRAPH_ADAPTER_SPECS,
    MILD_COMBINED_FAMILIES,
    PERTURBATION_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
    TargetRelativeMultiTargetTrainingConfig,
    VALIDATION_BINS,
    apply_training_perturbation_mixture,
    apply_training_target_distribution,
    apply_validation_bin,
    planned_target_relative_multitarget_h0_rows,
    planned_target_relative_multitarget_rows,
    target_relative_validation_manifest,
    planned_fixed_target_perturbation_rows,
    validation_bin_manifest,
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
    assert hps.model.initial_hidden_encoder is False
    assert hps.model.initial_hidden_encoder_config.enabled is False
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


def test_full_analytical_qrf_loss_uses_trial_static_target_for_goal_centering() -> None:
    hps = build_hps(_args(smoke=True, loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    loss = pair.task.loss_func.terms[CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE]

    zeros = jnp.zeros((1, 60, 48), dtype=jnp.float64)
    zero_command = jnp.zeros((1, 60, 2), dtype=jnp.float64)
    states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros.at[:, :, 0].set(0.12)),
        net=TreeNamespace(output=zero_command),
        efferent=TreeNamespace(output=zero_command),
    )
    default_value = loss.term(states, trial, pair.model)
    target_config = TargetRelativeMultiTargetTrainingConfig(
        enabled=True,
        seen_directions_deg=(90.0,),
        held_out_directions_deg=(270.0,),
        seen_amplitudes_m=(0.12,),
        held_out_amplitudes_m=(0.12,),
        original_target_anchor_m=(0.0, 0.12),
    )
    retargeted = apply_training_target_distribution(
        trial,
        target_config,
        jr.PRNGKey(3),
    )
    target = retargeted.targets["mechanics.effector.pos"].value[..., -1, :]
    retargeted_value = loss.term(states, retargeted, pair.model)

    assert not jnp.allclose(target, jnp.array([0.15, 0.0]))
    assert not jnp.allclose(retargeted_value, default_value)


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


def test_perturbation_training_hps_preserves_fixed_target_semantics() -> None:
    hps = build_hps(
        _args(
            perturbation_training=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            batch_size=64,
            gradient_clip_norm=5.0,
            controller_lr=1e-3,
            lr_warmup_batches=500,
            lr_cosine_alpha=0.01,
        )
    )

    assert hps.task.type == "fixed_simple_reach"
    assert hps.task.fixed_target_pos == [0.15, 0.0]
    assert hps.task.eval_n_directions == 1
    assert hps.task.eval_reach_length == 0.15
    assert hps.perturbation_training.enabled is True
    assert hps.perturbation_training.nominal_fraction == pytest.approx(0.45)
    assert hps.perturbation_training.single_fraction == pytest.approx(0.45)
    assert hps.perturbation_training.combined_fraction == pytest.approx(0.10)
    semantics = hps.perturbation_training.mixture_semantics
    assert semantics.experimental_factor_note.startswith("Perturbation uncertainty level")
    assert "nominal open-loop command-replay peak delta x" in semantics.calibration_note
    assert semantics.membership.nominal_fraction == pytest.approx(0.45)
    assert semantics.membership.single_family_fraction == pytest.approx(0.45)
    assert semantics.membership.mild_combined_fraction == pytest.approx(0.10)
    assert semantics.mild_combined_families == list(MILD_COMBINED_FAMILIES)
    assert semantics.families.initial_position.randomized == [
        "axis",
        "sign",
        "amplitude_level",
    ]
    assert "not a replay" in semantics.validation_difference
    assert hps.perturbation_training.target_stream.status == "not_applicable"
    assert hps.loss.objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
    assert hps.loss.weights.nn_hidden == 0.0
    assert hps.batch_size == 64
    assert hps.gradient_clip_norm == 5.0
    assert hps.lr_schedule == "warmup_cosine"


def test_perturbation_training_setup_adds_external_adapters_without_target_input() -> None:
    hps = build_hps(_args(smoke=True, perturbation_training=True))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.validation_trials

    assert pair.model.input_ports[:2] == ("input", "epsilon")
    for spec in GRAPH_ADAPTER_SPECS.values():
        assert spec.input_key in pair.model.input_ports
        assert spec.input_key in trial.inputs
    assert not any("target" in port for port in pair.model.input_ports)
    assert trial.inputs["effector_target"].pos.shape == (1, 60, 2)
    assert jnp.allclose(trial.inputs["effector_target"].pos, 0.15 * jnp.array([1.0, 0.0]))
    assert trial.extra["perturbation_training_bin"] == "nominal"


def test_perturbation_training_validation_bins_are_separate_and_fixed_target() -> None:
    hps = build_hps(_args(smoke=True, perturbation_training=True))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    base = pair.task.task.validation_trials
    manifest = validation_bin_manifest(hps.perturbation_training)

    assert [row["bin"] for row in manifest["bins"]] == list(VALIDATION_BINS)
    for bin_name in VALIDATION_BINS:
        trial = apply_validation_bin(base, hps.perturbation_training, bin_name)
        assert trial.extra["perturbation_training_bin"] == bin_name
        assert jnp.allclose(trial.inputs["effector_target"].pos, base.inputs["effector_target"].pos)

    nominal = apply_validation_bin(base, hps.perturbation_training, "nominal")
    initial_position = apply_validation_bin(base, hps.perturbation_training, "initial_position")
    initial_velocity = apply_validation_bin(base, hps.perturbation_training, "initial_velocity")
    process = apply_validation_bin(base, hps.perturbation_training, "process_epsilon")
    command = apply_validation_bin(base, hps.perturbation_training, "command_input")

    assert jnp.allclose(nominal.inits["mechanics.vector"], base.inits["mechanics.vector"])
    assert jnp.any(initial_position.inits["mechanics.vector"] != base.inits["mechanics.vector"])
    assert jnp.any(initial_velocity.inits["mechanics.vector"] != base.inits["mechanics.vector"])
    assert jnp.any(process.inputs["epsilon"] != base.inputs["epsilon"])
    assert jnp.any(command.inputs[GRAPH_ADAPTER_SPECS["command_input"].input_key] != 0.0)
    assert tuple(manifest["bins"][-1]["families"]) == MILD_COMBINED_FAMILIES
    assert manifest["validation_role"] == "generalized_held_out_perturbation_rollout_loss"


def test_randomized_perturbation_training_uses_prng_key_and_preserves_target() -> None:
    hps = build_hps(_args(smoke=True, perturbation_training=True, batch_size=32))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    base = pair.task.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))

    first = apply_training_perturbation_mixture(
        base,
        hps.perturbation_training,
        jr.PRNGKey(10),
    )
    second = apply_training_perturbation_mixture(
        base,
        hps.perturbation_training,
        jr.PRNGKey(11),
    )

    assert jnp.allclose(first.inputs["effector_target"].pos, base.inputs["effector_target"].pos)
    assert jnp.any(first.inits["mechanics.vector"] != second.inits["mechanics.vector"])
    assert jnp.any(first.inputs["epsilon"] != second.inputs["epsilon"])

    # Training trials are built inside Feedbax's vmapped training step, so per-trial
    # metadata must stay JAX-compatible. String/list provenance lives in the config
    # and validation manifest instead of dynamic train-trial leaves.
    assert first.extra is None or "perturbation_training_bin" not in first.extra
    manifest = validation_bin_manifest(hps.perturbation_training)
    assert manifest["validation_role"] == "generalized_held_out_perturbation_rollout_loss"
    assert tuple(manifest["bins"][-1]["families"]) == MILD_COMBINED_FAMILIES
    assert hps.perturbation_training.mode == "fixed_target_perturbation_randomized"


def test_randomized_perturbation_training_has_signed_component_variation() -> None:
    hps = build_hps(
        _args(perturbation_training=True, batch_size=96, hidden_size=4, n_replicates=1)
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    base = pair.task.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))

    trials = [
        apply_training_perturbation_mixture(
            base,
            hps.perturbation_training,
            jr.PRNGKey(seed),
        )
        for seed in range(64)
    ]

    init_delta = jnp.stack(
        [trial.inits["mechanics.vector"] - base.inits["mechanics.vector"] for trial in trials]
    )
    assert jnp.any(init_delta[..., 0] > 0.0) or jnp.any(init_delta[..., 1] > 0.0)
    assert jnp.any(init_delta[..., 0] < 0.0) or jnp.any(init_delta[..., 1] < 0.0)
    assert jnp.count_nonzero(jnp.any(init_delta[..., :2] != 0.0, axis=0)) >= 2

    command = jnp.stack(
        [trial.inputs[GRAPH_ADAPTER_SPECS["command_input"].input_key] for trial in trials]
    )
    assert jnp.any(command[..., 0] != 0.0)
    assert jnp.any(command[..., 1] != 0.0)
    assert jnp.any(command > 0.0)
    assert jnp.any(command < 0.0)


def test_perturbation_training_run_spec_and_planned_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        issue="aacb9ed",
        perturbation_training=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        batch_size=64,
        gradient_clip_norm=5.0,
        controller_lr=1e-3,
        lr_warmup_batches=500,
        lr_cosine_alpha=0.01,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    rows = planned_fixed_target_perturbation_rows()

    assert payload["issue"] == "aacb9ed"
    assert payload["nominal_only"] is False
    assert payload["training_summary"]["training_mode"] == PERTURBATION_TRAINING_MODE
    assert payload["training_summary"]["validation_bins"]["bins"][0]["bin"] == "nominal"
    assert payload["training_summary"]["validation_bins"]["selection_role"].startswith(
        "aggregate rollout loss"
    )
    assert payload["model_summary"]["training_distribution"]["fixed_target_only"] is True
    assert payload["model_summary"]["training_distribution"]["checkpoint_selection_role"] == (
        "generalized_held_out_perturbation_validation"
    )
    semantics = payload["hps"]["perturbation_training"]["mixture_semantics"]
    assert semantics["experimental_factor_note"].startswith(
        "Perturbation uncertainty level"
    )
    assert "nominal open-loop command-replay peak delta x" in semantics["calibration_note"]
    assert semantics["families"]["process_epsilon"]["duration_steps"] == 5
    assert semantics["validation_difference"].startswith("Validation bins are")
    assert payload["hps"]["perturbation_training"]["controller_internal_mutation"] is False
    assert {row["controller_lr"] for row in rows} == {1e-3, 3e-3}
    assert all(row["batch_size"] == 64 for row in rows)
    assert all(row["gradient_clip_norm"] == 5.0 for row in rows)
    assert all("--perturbation-training" in row["command"] for row in rows)
    assert all("fixed_target_random_perturb" in row["run"] for row in rows)
    assert all(
        row["checkpoint_selection"] == "generalized_held_out_perturbation_validation"
        for row in rows
    )


def test_target_relative_feedback_sign_contract() -> None:
    component = TargetRelativeDelayedFeedback()
    state = jnp.zeros((48,), dtype=jnp.float32).at[40:44].set(
        jnp.array([0.02, -0.03, 0.40, -0.20], dtype=jnp.float32)
    )
    outputs, _ = component(
        {"state": state, "target": jnp.array([0.15, 0.01], dtype=jnp.float32)},
        None,
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(
        outputs["feedback"],
        jnp.array([0.13, 0.04, -0.40, 0.20], dtype=jnp.float32),
    )


def test_target_relative_feedback_batches_over_last_state_axis() -> None:
    component = TargetRelativeDelayedFeedback()
    state = jnp.zeros((2, 48), dtype=jnp.float32)
    state = state.at[:, 40:44].set(
        jnp.array(
            [
                [0.02, -0.03, 0.40, -0.20],
                [0.05, 0.04, -0.10, 0.30],
            ],
            dtype=jnp.float32,
        )
    )
    outputs, _ = component(
        {"state": state, "target": jnp.array([0.15, 0.01], dtype=jnp.float32)},
        None,
        key=jr.PRNGKey(0),
    )

    assert outputs["feedback"].shape == (2, 4)
    assert jnp.allclose(
        outputs["feedback"],
        jnp.array(
            [
                [0.13, 0.04, -0.40, 0.20],
                [0.10, -0.03, 0.10, -0.30],
            ],
            dtype=jnp.float32,
        ),
    )


def test_target_relative_multitarget_setup_uses_target_input_and_anchor() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            target_relative_multitarget=True,
            perturbation_training=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.validation_trials
    manifest = target_relative_validation_manifest(hps.target_relative_multitarget)

    assert pair.model.input_ports[:2] == ("target", "epsilon")
    assert "input" not in pair.model.input_ports
    assert "target" in trial.inputs
    assert trial.inputs["target"].shape[-2:] == (60, 2)
    assert trial.inputs["effector_target"].pos.shape[-2:] == (60, 2)
    assert hps.target_relative_multitarget.input_contract.sign_convention == [
        "target_x - delayed_x",
        "target_y - delayed_y",
        "-delayed_vx",
        "-delayed_vy",
    ]
    assert hps.target_relative_multitarget.target_distribution.original_target_anchor_m == [
        0.15,
        0.0,
    ]
    assert [row["bin"] for row in manifest["bins"][:3]] == [
        "original_target_nominal",
        "seen_multitarget_nominal",
        "held_out_multitarget_nominal",
    ]
    assert manifest["target_centered_scoring"] == "trial_static_target"
    assert any(
        row["bin"] == "command_input_diagnostic"
        and row["checkpoint_selection"] == "excluded_unless_comparator_defined"
        for row in manifest["bins"]
    )
    perturbation_bins = [
        row
        for row in manifest["bins"]
        if row["target_role"] == "seen_and_held_out_static_targets"
    ]
    assert perturbation_bins
    for row in perturbation_bins:
        assert row["targets_m"]
        assert row["targets_m"] == perturbation_bins[0]["targets_m"]
    assert perturbation_bins[0]["targets_m"] != manifest["bins"][0]["targets_m"]
    assert jnp.any(trial.inputs["target"][..., -1, :] != jnp.array([0.15, 0.0]))


def test_target_relative_multitarget_run_spec_and_planned_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        issue="ba82f3d",
        target_relative_multitarget=True,
        perturbation_training=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        batch_size=64,
        gradient_clip_norm=5.0,
        controller_lr=1e-3,
        lr_warmup_batches=500,
        lr_cosine_alpha=0.01,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    rows = planned_target_relative_multitarget_rows()

    assert payload["issue"] == "ba82f3d"
    assert payload["training_summary"]["training_mode"] == (
        TARGET_RELATIVE_MULTITARGET_TRAINING_MODE
    )
    assert payload["model_summary"]["feedback"]["basis"] == "target_relative_delayed_feedback"
    distribution = payload["model_summary"]["training_distribution"]
    assert distribution["fixed_target_only"] is False
    assert distribution["target_stream"]["status"] == "consumed_as_static_target_relative_feedback"
    assert distribution["original_target_anchor_m"] == [0.15, 0.0]
    assert payload["task_timing"]["extra_inputs"] == ["target", "epsilon"]
    assert payload["task_timing"]["target_relative_multitarget"]["enabled"] is True
    assert payload["validation_bins"]["validation_role"] == (
        "target_relative_multitarget_rollout_loss"
    )
    assert payload["validation_bins"]["input_contract"]["sign_convention"][0] == (
        "target_x - delayed_x"
    )
    assert all(
        "targets_m" in row and row["targets_m"]
        for row in payload["validation_bins"]["bins"]
        if row["target_role"] == "seen_and_held_out_static_targets"
    )
    assert payload["loss_summary"]["objective_profile"] == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
    assert {row["controller_lr"] for row in rows if row["row_kind"] == "main"} == {
        1e-3,
        3e-3,
    }
    assert all(
        "--target-relative-multitarget" in row["command"]
        for row in rows
    )
    assert all(
        row["checkpoint_selection"] == "target_relative_multitarget_rollout_validation"
        for row in rows
        if row["row_kind"] == "main"
    )


def test_initial_hidden_encoder_requires_target_relative_hps() -> None:
    with pytest.raises(ValueError, match="requires --target-relative-multitarget"):
        build_hps(_args(initial_hidden_encoder=True))


def test_target_relative_h0_run_spec_and_planned_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        issue="643f101",
        target_relative_multitarget=True,
        initial_hidden_encoder=True,
        perturbation_training=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        batch_size=64,
        gradient_clip_norm=5.0,
        controller_lr=1e-3,
        lr_warmup_batches=500,
        lr_cosine_alpha=0.01,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    rows = planned_target_relative_multitarget_h0_rows()

    assert payload["issue"] == "643f101"
    assert payload["training_summary"]["training_mode"] == (
        TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE
    )
    h0 = payload["model_summary"]["initial_hidden_encoder"]
    assert h0["enabled"] is True
    assert h0["architecture"] == "affine"
    assert h0["context_source"] == "first_controller_visible_target_relative_delayed_feedback"
    assert h0["context_shape"] == [4]
    assert h0["output_shape"] == [4]
    assert h0["initialization"] == "zero_affine"
    assert h0["teacher_or_jacobian_supervision"] is False
    assert h0["plant_live_preview"] is False
    assert h0["delayed_reach"] is False
    assert payload["model_summary"]["trainable"] == [
        "nodes.net.hidden",
        "nodes.net.readout",
        "nodes.net.h0_encoder",
    ]
    assert payload["training_summary"]["initial_hidden_encoder"]["enabled"] is True
    assert payload["training_summary"]["training_distribution"]["initial_hidden_encoder"][
        "enabled"
    ] is True
    assert payload["hps"]["where"]["0"] == [
        "nodes.net.hidden",
        "nodes.net.readout",
        "nodes.net.h0_encoder",
    ]
    main_rows = [row for row in rows if row["row_kind"] == "main"]
    assert {row["controller_lr"] for row in main_rows} == {1e-3, 3e-3}
    assert all(row["batch_size"] == 64 for row in main_rows)
    assert all(row["gradient_clip_norm"] == 5.0 for row in main_rows)
    assert all(row["n_replicates"] == 5 for row in main_rows)
    assert all(row["n_train_batches"] == 12000 for row in main_rows)
    assert all(row["training_diagnostics"] == "default_enabled" for row in main_rows)
    assert all("--initial-hidden-encoder" in row["command"] for row in rows)
    assert all("--target-relative-multitarget" in row["command"] for row in rows)
    assert all(
        row["checkpoint_selection"] == "target_relative_multitarget_rollout_validation"
        for row in main_rows
    )


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


def test_target_relative_h0_full_training_smoke_emits_diagnostics(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="643f101",
        n_train_batches=2,
        batch_size=2,
        n_replicates=1,
        hidden_size=4,
        target_relative_multitarget=True,
        initial_hidden_encoder=True,
        perturbation_training=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        full_train=True,
        checkpoint_interval_batches=1,
        controller_lr=1e-3,
        gradient_clip_norm=5.0,
        lr_warmup_batches=1,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        log_step=1,
        disable_progress=True,
        quiet_progress=True,
    )

    result = run_full_training(args)
    run_spec = json.loads((spec_dir / "run.json").read_text())
    summary = json.loads((output_dir / "training_summary.json").read_text())
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())

    assert result["completed_batches"] == 2
    assert run_spec["training_summary"]["training_mode"] == (
        TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE
    )
    assert run_spec["model_summary"]["initial_hidden_encoder"]["enabled"] is True
    assert summary["training_diagnostics"]["enabled"] is True
    assert summary["training_diagnostics"]["written"] is True
    assert diagnostics_manifest["completed_batches"] == 2
    assert "optimizer_gradient_norm_pre_clip" in diagnostics_manifest["arrays"]
    with np.load(output_dir / "training_diagnostics.npz") as diagnostics:
        assert diagnostics["batch_index"].tolist() == [0, 1]
        assert diagnostics["optimizer_gradient_norm_pre_clip"].shape == (2, 1)
        assert diagnostics["train_loss__total"].shape == (2, 1)
        assert diagnostics["validation_loss__total"].shape == (2, 1)


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
