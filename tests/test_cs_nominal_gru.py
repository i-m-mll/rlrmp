"""Tests for stochastic C&S-fidelity GRU run-spec preparation."""

from __future__ import annotations

import argparse
import json
import warnings
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from feedbax import TaskTrialSpec, TrialTimeline, WhereDict
from feedbax.loss import TargetSpec
from feedbax.mechanics import LinearStateSpace
from feedbax.training.train import TaskTrainer, make_delayed_cosine_schedule, train_pair
from feedbax.types import TreeNamespace

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    build_canonical_game,
)
from rlrmp.analysis.math.cs_released_simulation import default_cs_noise_covariances
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
from rlrmp.analysis.pipelines.gru_perturbation_calibration import (
    DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT,
)
from rlrmp.cs_lss_gru import (
    CS_EPSILON_DIM,
    CS_REDUCED_EPSILON_DIM,
    TargetRelativeDelayedFeedback,
    TargetRelativeDelayedProprioceptiveFeedback,
)
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    CsAnalyticalQrfLoss,
)
from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir
from rlrmp.train.cs_nominal_gru import (
    CS_DELAYED_REACH_TASK_TYPE,
    DEFAULT_DELAYED_P_CATCH_TRIAL,
    DEFAULT_STOCHASTIC_PRESET,
    DELAYED_REACH_TRAINING_MODE,
    build_graph_bundle,
    build_hps,
    build_parser,
    derive_spec_dir,
    run_full_training,
    write_run_spec,
)
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_TRAINING_MODE,
    BROAD_EPSILON_TRAINING_MODE,
    CALIBRATED_TIMING_PERTURBATION_TRAINING_MODE,
    GRAPH_ADAPTER_SPECS,
    MILD_COMBINED_FAMILIES,
    PERTURBATION_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
    VALIDATION_BINS,
    BroadFullStateEpsilonTrainingConfig,
    BroadFullStateEpsilonTrainingTaskAdapter,
    TargetRelativeMultiTargetTrainingConfig,
    TargetRelativeMultiTargetTrainingTaskAdapter,
    _broad_epsilon_l2_radius,
    _ensure_broad_epsilon_input,
    _epsilon_time_mask,
    _expand_bool_like,
    _expand_radius,
    _normalize_flattened_per_trial,
    _project_flattened_per_trial_l2_ball,
    _set_input,
    apply_broad_epsilon_training,
    apply_training_perturbation_mixture,
    apply_training_target_distribution,
    apply_validation_bin,
    apply_validation_target_distribution,
    config_from_broad_epsilon_pgd_hps,
    planned_fixed_target_perturbation_rows,
    planned_target_relative_multitarget_h0_rows,
    planned_target_relative_multitarget_rows,
    run_broad_epsilon_pgd_inner_maximizer,
    target_relative_validation_manifest,
    validation_bin_manifest,
)
from rlrmp.train.task_model import (
    CS_LSS_PLANT_BACKEND,
    LEGACY_CAUSAL_BACKEND_WARNING,
    LEGACY_CAUSAL_PLANT_BACKEND,
    _add_cs_lss_task_inputs,
    _CsLssTaskAdapter,
    build_task_base,
    _cs_lss_process_epsilon_factor,
    setup_task_model_pair,
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


def _delayed_cs_task(
    hps,
    *,
    target_relative: bool = True,
    go_cue_input: bool = True,
    broad_epsilon: bool = False,
):
    task = _add_cs_lss_task_inputs(
        _CsLssTaskAdapter(build_task_base(hps)),
        target_relative=target_relative,
        go_cue_input=go_cue_input,
        physical_state_dim=int(hps.model.physical_state_dim),
    )
    if target_relative:
        task = TargetRelativeMultiTargetTrainingTaskAdapter(task, hps.target_relative_multitarget)
    if broad_epsilon:
        task = BroadFullStateEpsilonTrainingTaskAdapter(task, hps.broad_epsilon_training)
    return task


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


def test_pgd_broad_epsilon_hps_declares_inner_maximizer() -> None:
    hps = build_hps(
        _args(
            target_relative_multitarget=True,
            force_filter_feedback=True,
            broad_epsilon_pgd_training=True,
            broad_epsilon_level="moderate",
            broad_epsilon_pgd_steps=4,
            broad_epsilon_pgd_step_size_fraction=0.5,
        )
    )

    cfg = config_from_broad_epsilon_pgd_hps(hps.broad_epsilon_pgd_training)

    assert cfg.enabled is True
    assert hps.broad_epsilon_pgd_training.mode == BROAD_EPSILON_PGD_TRAINING_MODE
    assert hps.broad_epsilon_pgd_training.inner_maximizer.n_steps == 4
    assert (
        hps.broad_epsilon_pgd_training.inner_maximizer.differentiated_through_outer_update is False
    )
    assert hps.broad_epsilon_training.enabled is False


def test_pgd_broad_epsilon_hps_parser_consumes_nested_and_legacy_fields() -> None:
    nested = TreeNamespace(
        enabled=True,
        level="strong",
        budget_scale=1.5,
        reach_length_scaling=False,
        inner_maximizer=TreeNamespace(
            n_steps=9,
            step_size_fraction_of_l2_radius=0.125,
            initialization="zero",
        ),
    )
    legacy = TreeNamespace(
        enabled=True,
        level="moderate",
        n_steps=7,
        step_size_fraction=0.375,
        init="zero",
    )
    nested_dict = {
        "enabled": True,
        "level": "strong",
        "inner_maximizer": {
            "n_steps": 11,
            "step_size_fraction_of_l2_radius": 0.2,
            "init": "zero",
        },
    }

    parsed_nested = config_from_broad_epsilon_pgd_hps(nested)
    parsed_legacy = config_from_broad_epsilon_pgd_hps(legacy)
    parsed_dict = config_from_broad_epsilon_pgd_hps(nested_dict)

    assert parsed_nested.level == "strong"
    assert parsed_nested.n_steps == 9
    assert parsed_nested.step_size_fraction == pytest.approx(0.125)
    assert parsed_nested.budget_scale == pytest.approx(1.5)
    assert parsed_nested.reach_length_scaling is False
    assert parsed_legacy.n_steps == 7
    assert parsed_legacy.step_size_fraction == pytest.approx(0.375)
    assert parsed_dict.n_steps == 11
    assert parsed_dict.step_size_fraction == pytest.approx(0.2)


def test_pgd_broad_epsilon_lane_requires_target_relative_and_excludes_random_lane() -> None:
    with pytest.raises(ValueError, match="requires --target-relative-multitarget"):
        build_hps(_args(broad_epsilon_pgd_training=True))

    with pytest.raises(ValueError, match="cannot be combined"):
        build_hps(
            _args(
                target_relative_multitarget=True,
                broad_epsilon_training=True,
                broad_epsilon_pgd_training=True,
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
    hps = build_hps(_args(smoke=True, loss_objective=CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE))
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
    assert (
        "physical-process/load epsilon"
        in (bundle.manifest["model_structure"]["plant_process"]["epsilon_bridge"])
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
    assert (
        "sensory Channel" in (result["run_spec"]["fidelity_status"]["temporary_stochastic_bridge"])
    )
    assert (
        "signal-dependent motor Channel"
        in (result["run_spec"]["fidelity_status"]["temporary_stochastic_bridge"])
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
        payload["model_summary"]["stochastic_runtime"]["signal_dependent_motor_noise_std"] == 0.02
    )
    assert payload["model_summary"]["stochastic_runtime"]["plant_process_force_noise_std"] > 0.0
    assert payload["model_summary"]["plant_process"]["state_diffusion"] == "mechanics.epsilon"
    assert (
        "physical-process/load epsilon"
        in (payload["model_summary"]["plant_process"]["epsilon_bridge"])
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
    assert (
        payload["loss_summary"]["objective_profile"] == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE
    )
    assert payload["loss_summary"]["active_cs_terms"]["control"]["state_key"] == "states.net.output"
    assert payload["loss_summary"]["active_cs_terms"]["force_filter"]["scale"] == pytest.approx(
        1 / 6
    )
    assert (
        payload["loss_summary"]["disturbance_integrator_state_cost"] == "omitted_in_this_ablation"
    )
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
    assert "Calibrated timing mode samples timing bins uniformly" in (semantics.calibration_note)
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
    hps = build_hps(_args(perturbation_training=True, batch_size=96, hidden_size=4, n_replicates=1))
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


def _nonzero_pulse_starts(delta: jnp.ndarray) -> set[int]:
    active = jnp.any(delta != 0.0, axis=-1)
    rows = np.asarray(active.reshape((-1, active.shape[-1])))
    return {int(np.flatnonzero(row)[0]) for row in rows if np.any(row)}


def _max_nonzero_pulse_width(delta: jnp.ndarray) -> int:
    active = np.asarray(jnp.any(delta != 0.0, axis=-1).reshape((-1, delta.shape[-2])))
    widths = [int(np.count_nonzero(row)) for row in active if np.any(row)]
    return max(widths) if widths else 0


def test_calibrated_timing_sampler_uses_family_timing_bins() -> None:
    hps = build_hps(
        _args(
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_physical_level="small",
            batch_size=256,
            hidden_size=4,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    base = pair.task.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    sampled = apply_training_perturbation_mixture(
        base,
        hps.perturbation_training,
        jr.PRNGKey(2),
    )

    process_delta = sampled.inputs["epsilon"] - base.inputs["epsilon"]
    command = sampled.inputs[GRAPH_ADAPTER_SPECS["command_input"].input_key]
    sensory = sampled.inputs[GRAPH_ADAPTER_SPECS["sensory_feedback"].input_key]
    delayed = sampled.inputs[GRAPH_ADAPTER_SPECS["delayed_observation"].input_key]

    plant_starts = {5, 15, 35}
    controller_visible_starts = {10, 20, 40}
    assert _nonzero_pulse_starts(process_delta).issubset(plant_starts)
    assert _nonzero_pulse_starts(command).issubset(plant_starts)
    assert _nonzero_pulse_starts(sensory).issubset(controller_visible_starts)
    assert _nonzero_pulse_starts(delayed).issubset(controller_visible_starts)
    assert _max_nonzero_pulse_width(sensory) <= 5
    assert _max_nonzero_pulse_width(delayed) <= 5
    assert hps.perturbation_training.mode == CALIBRATED_TIMING_PERTURBATION_TRAINING_MODE


def _unique_abs_nonzero(values: jnp.ndarray) -> np.ndarray:
    flat = np.asarray(jnp.abs(values)).reshape(-1)
    return np.unique(np.round(flat[flat > 0.0], 8))


def _assert_values_close_to_expected(values: np.ndarray, expected: set[float]) -> None:
    expected_values = np.asarray(sorted(expected), dtype=float)
    assert values.size > 0
    for value in values:
        assert np.any(np.isclose(value, expected_values, rtol=5e-5, atol=5e-7)), value


def test_calibrated_timing_sampler_consumes_calibrated_amplitudes() -> None:
    hps = build_hps(
        _args(
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_physical_level="moderate",
            batch_size=2048,
            hidden_size=4,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    base = pair.task.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    target_peak_delta_x = 0.15 * 0.10

    initial_position_bin = apply_validation_bin(
        base,
        hps.perturbation_training,
        "initial_position",
    )
    init_delta = initial_position_bin.inits["mechanics.vector"] - base.inits["mechanics.vector"]
    _assert_values_close_to_expected(
        _unique_abs_nonzero(init_delta[..., :2]),
        {target_peak_delta_x},
    )
    initial_velocity_bin = apply_validation_bin(
        base,
        hps.perturbation_training,
        "initial_velocity",
    )
    init_delta = initial_velocity_bin.inits["mechanics.vector"] - base.inits["mechanics.vector"]
    _assert_values_close_to_expected(
        _unique_abs_nonzero(init_delta[..., 2:4]),
        {
            target_peak_delta_x
            / DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT["initial_velocity_offset"][
                "initial_condition"
            ]
        },
    )

    process_bin = apply_validation_bin(base, hps.perturbation_training, "process_epsilon")
    process_delta = process_bin.inputs["epsilon"] - base.inputs["epsilon"]
    process_expected = {
        target_peak_delta_x
        / DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT["process_epsilon_force_state_xy"]["early"]
    }
    _assert_values_close_to_expected(
        _unique_abs_nonzero(process_delta),
        process_expected,
    )

    command_bin = apply_validation_bin(base, hps.perturbation_training, "command_input")
    command = command_bin.inputs[GRAPH_ADAPTER_SPECS["command_input"].input_key]
    command_full = {
        target_peak_delta_x
        / DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT["command_input_pulse"]["early"]
    }
    _assert_values_close_to_expected(
        _unique_abs_nonzero(command),
        command_full,
    )

    sensory_expected = {
        target_peak_delta_x,
    }
    sensory_bin = apply_validation_bin(base, hps.perturbation_training, "sensory_feedback")
    _assert_values_close_to_expected(
        _unique_abs_nonzero(sensory_bin.inputs[GRAPH_ADAPTER_SPECS["sensory_feedback"].input_key]),
        sensory_expected,
    )
    delayed_bin = apply_validation_bin(base, hps.perturbation_training, "delayed_observation")
    _assert_values_close_to_expected(
        _unique_abs_nonzero(
            delayed_bin.inputs[GRAPH_ADAPTER_SPECS["delayed_observation"].input_key]
        ),
        sensory_expected,
    )


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
    assert semantics["experimental_factor_note"].startswith("Perturbation uncertainty level")
    assert "Calibrated timing mode samples timing bins uniformly" in (semantics["calibration_note"])
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


def test_calibrated_timing_run_spec_exposes_family_timing_bins(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        issue="c99ad9d",
        perturbation_training=True,
        perturbation_calibrated_timing=True,
        perturbation_physical_level="moderate",
        perturbation_combined_fraction=0.10,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    training = payload["model_summary"]["training_distribution"]
    hps_config = payload["hps"]["perturbation_training"]
    timing = hps_config["timing_bins"]["family_timing_bins"]

    assert payload["training_summary"]["training_mode"] == PERTURBATION_TRAINING_MODE
    assert training["mode"] == CALIBRATED_TIMING_PERTURBATION_TRAINING_MODE
    assert training["mixture"]["calibrated_timing"] is True
    assert training["mixture"]["physical_level"] == "moderate"
    assert hps_config["physical_level_fraction_of_reach"] == 0.10
    assert hps_config["training_physical_levels"] == ["small", "moderate"]
    assert hps_config["eval_only_physical_levels"] == ["stress"]
    assert timing["process_epsilon"]["start_time_indices"] == [5, 15, 35]
    assert timing["command_input"]["start_time_indices"] == [5, 15, 35]
    assert timing["sensory_feedback"]["start_time_indices"] == [10, 20, 40]
    assert timing["delayed_observation"]["start_time_indices"] == [10, 20, 40]
    assert timing["initial_position"]["start_time_indices"] == [0]
    assert (
        "not literal extra temporal delay"
        in (hps_config["timing_bins"]["controller_visible"]["delayed_observation_semantics"])
    )
    assert (
        hps_config["mixture_semantics"]["calibrated_levels"]["amplitude_wiring_status"]
        == "wired_in_sampler_when_calibrated_timing_true"
    )
    assert hps_config["calibrated_amplitude_policy"]["artifact_dependency"] == ("none_at_runtime")


def test_target_relative_feedback_sign_contract() -> None:
    component = TargetRelativeDelayedFeedback()
    state = (
        jnp.zeros((48,), dtype=jnp.float32)
        .at[40:44]
        .set(jnp.array([0.02, -0.03, 0.40, -0.20], dtype=jnp.float32))
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
        row for row in manifest["bins"] if row["target_role"] == "seen_and_held_out_static_targets"
    ]
    assert perturbation_bins
    for row in perturbation_bins:
        assert row["targets_m"]
        assert row["targets_m"] == perturbation_bins[0]["targets_m"]
    assert perturbation_bins[0]["targets_m"] != manifest["bins"][0]["targets_m"]
    assert jnp.any(trial.inputs["target"][..., -1, :] != jnp.array([0.15, 0.0]))


def test_delayed_reach_requires_target_relative_contract() -> None:
    with pytest.raises(ValueError, match="requires --target-relative-multitarget"):
        build_hps(_args(delayed_reach=True))


def test_delayed_reach_setup_adds_go_cue_and_preserves_target_visibility() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.0,
            target_relative_multitarget=True,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    go_step = int(trial.timeline.epoch_bounds[-2])

    assert hps.task.type == CS_DELAYED_REACH_TASK_TYPE
    assert hps.task.n_steps == 90
    assert hps.task.epoch_len_ranges == [[10, 31]]
    assert hps.task.p_catch_trial == pytest.approx(0.0)
    assert hps.loss.weights.nn_output_pre_go == pytest.approx(1.0)
    assert pair.model.input_ports[:3] == ("input", "target", "epsilon")
    assert pair.model.nodes["net"].input_size == 5
    assert trial.timeline.epoch_names == ("prep", "movement")
    assert 10 <= go_step <= 30
    assert trial.inputs["input"].shape == (90,)
    assert jnp.allclose(trial.inputs["input"][:go_step], 0.0)
    assert jnp.allclose(trial.inputs["input"][go_step:], 1.0)
    assert trial.inputs["target"].shape[-2:] == (90, 2)
    assert jnp.allclose(
        trial.inputs["target"],
        jnp.broadcast_to(trial.inputs["target"][..., :1, :], trial.inputs["target"].shape),
    )
    assert trial.inputs["epsilon"].shape == (90, CS_EPSILON_DIM)

    validation = pair.task.validation_trials
    validation_targets = validation.targets["mechanics.effector.pos"].value
    assert validation.inputs["task"].effector_target.pos.shape == validation_targets.shape
    assert validation.inputs["task"].hold.shape[:2] == validation_targets.shape[:2]
    assert validation.inputs["target"].shape == validation_targets.shape
    assert validation.extra is not None
    assert validation.extra["is_catch_trial"].shape == (pair.task.n_validation_trials,)
    assert not bool(jnp.any(validation.extra["is_catch_trial"]))


def test_delayed_reach_catch_trials_keep_target_visible_without_go_cue() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=1.0,
            target_relative_multitarget=True,
            hidden_size=8,
            n_replicates=1,
        )
    )
    task = _delayed_cs_task(hps)
    trial = task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    go_step = int(trial.timeline.epoch_bounds[-2])
    visible_target = trial.inputs["target"]
    scored_target = trial.targets["mechanics.effector.pos"].value

    assert 10 <= go_step <= 30
    assert hps.task.p_catch_trial == pytest.approx(1.0)
    assert hps.delayed_reach.catch_trials.p_catch_trial == pytest.approx(1.0)
    assert trial.extra is not None
    assert bool(trial.extra["is_catch_trial"])
    assert jnp.allclose(trial.inputs["input"], 0.0)
    assert jnp.allclose(
        visible_target,
        jnp.broadcast_to(visible_target[..., :1, :], visible_target.shape),
    )
    assert jnp.any(jnp.abs(visible_target) > 0.0)
    assert jnp.allclose(scored_target, jnp.zeros_like(scored_target))


def test_delayed_reach_catch_trials_survive_target_distribution() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=1.0,
            target_relative_multitarget=True,
            broad_epsilon_training=True,
            broad_epsilon_pgd_training=False,
            hidden_size=8,
            n_replicates=1,
        )
    )
    task_base = _delayed_cs_task(hps, target_relative=False, go_cue_input=True)
    base = task_base.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    sampled = apply_training_target_distribution(
        base, hps.target_relative_multitarget, jr.PRNGKey(2)
    )
    perturbed = apply_broad_epsilon_training(sampled, hps.broad_epsilon_training, jr.PRNGKey(3))
    visible_target = perturbed.inputs["target"]
    scored_target = perturbed.targets["mechanics.effector.pos"].value
    delta = perturbed.inputs["epsilon"] - sampled.inputs["epsilon"]

    assert perturbed.extra is not None
    assert bool(perturbed.extra["is_catch_trial"])
    assert jnp.any(jnp.abs(visible_target) > 0.0)
    assert jnp.allclose(scored_target, jnp.zeros_like(scored_target))
    assert jnp.allclose(perturbed.inputs["task"].effector_target.pos, scored_target)
    assert jnp.allclose(delta, 0.0)


def test_delayed_reach_no_integrator_setup_uses_36d_state_and_6d_epsilon() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.0,
            target_relative_multitarget=True,
            force_filter_feedback=True,
            no_integrator_state=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    loss = pair.task.loss_func.terms[CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE]
    mechanics = pair.model.nodes["mechanics"]

    assert hps.model.no_integrator_state is True
    assert hps.model.state_dim == 36
    assert hps.model.physical_state_dim == 6
    assert mechanics.A.shape[-2:] == (36, 36)
    assert mechanics.B_w.shape[-2:] == (36, 6)
    assert trial.inits["mechanics.vector"].shape[-1] == 36
    assert trial.inputs["epsilon"].shape == (90, CS_REDUCED_EPSILON_DIM)
    assert loss.Q.shape[-1] == 36
    assert loss.n_phys == 6


def test_delayed_reach_movement_costs_and_broad_epsilon_are_go_cue_gated() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.0,
            target_relative_multitarget=True,
            broad_epsilon_training=True,
            broad_epsilon_pgd_training=False,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    base = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    go_step = int(base.timeline.epoch_bounds[-2])
    pos_loss = pair.task.loss_func.terms["effector_pos_running"]
    pre_go_loss = pair.task.loss_func.terms["nn_output_pre_go"]
    discount = pos_loss.spec.discount(base)
    sampled = apply_broad_epsilon_training(base, hps.broad_epsilon_training, jr.PRNGKey(2))
    delta = sampled.inputs["epsilon"] - base.inputs["epsilon"]

    assert hps.broad_epsilon_training.movement_epoch_only is True
    assert pre_go_loss.epoch_indices == (0,)
    assert jnp.allclose(discount[:go_step], 0.0)
    assert discount[go_step] == pytest.approx((1.0 / 60.0) ** 6)
    assert discount[go_step + 59] == pytest.approx(1.0)
    assert jnp.allclose(delta[..., :go_step, :], 0.0)
    assert jnp.any(delta[..., go_step:, :] != 0.0)


def test_delayed_reach_full_qrf_ignores_pre_go_commands() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.0,
            target_relative_multitarget=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    go_step = int(trial.timeline.epoch_bounds[-2])
    loss = pair.task.loss_func.terms[CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE]
    zeros = jnp.zeros((1, 90, 48), dtype=jnp.float64)
    zero_command = jnp.zeros((1, 90, 2), dtype=jnp.float64)
    base_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command),
    )
    pre_go_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command.at[:, max(go_step - 1, 0), 0].set(3.0)),
    )
    movement_states = TreeNamespace(
        mechanics=TreeNamespace(vector=zeros),
        net=TreeNamespace(output=zero_command.at[:, go_step, 0].set(3.0)),
    )

    assert isinstance(loss, CsAnalyticalQrfLoss)
    assert jnp.allclose(
        loss.term(pre_go_states, trial, pair.model), loss.term(base_states, trial, pair.model)
    )
    assert jnp.all(
        loss.term(movement_states, trial, pair.model) > loss.term(base_states, trial, pair.model)
    )


def test_delayed_reach_run_spec_declares_task_and_movement_pgd_mask(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        dry_run=True,
        issue="6c36536",
        delayed_reach=True,
        target_relative_multitarget=True,
        broad_epsilon_pgd_training=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    result = write_run_spec(args)
    payload = result["run_spec"]

    assert payload["issue"] == "6c36536"
    assert payload["delayed_reach"]["enabled"] is True
    assert payload["delayed_reach"]["mode"] == DELAYED_REACH_TRAINING_MODE
    assert payload["delayed_reach"]["catch_trials"]["p_catch_trial"] == pytest.approx(
        DEFAULT_DELAYED_P_CATCH_TRIAL
    )
    assert payload["task_timing"]["type"] == CS_DELAYED_REACH_TASK_TYPE
    assert payload["task_timing"]["preset"] == "delayed_center_out"
    assert payload["task_timing"]["n_control_stages"] == payload["task_timing"]["n_steps"] - 1
    assert payload["task_timing"]["p_catch_trial"] == pytest.approx(DEFAULT_DELAYED_P_CATCH_TRIAL)
    assert payload["task_timing"]["extra_inputs"] == ["input", "target", "epsilon"]
    assert payload["task_timing"]["movement_window"]["cost_indexing"] == (
        "movement_age_not_trial_age"
    )
    assert payload["model_summary"]["go_cue"]["enabled"] is True
    assert payload["model_summary"]["controller_input_dimension"] == 5
    assert payload["hps"]["loss"]["weights"]["nn_output_pre_go"] == 1.0
    assert payload["hps"]["broad_epsilon_pgd_training"]["movement_epoch_only"] is True
    assert payload["hps"]["broad_epsilon_pgd_training"]["time_mask"]["mode"] == (
        "movement_epoch_only"
    )
    assert payload["loss_summary"]["time_indexing"]["stage_schedule"] == (
        "movement_age_from_go_cue"
    )
    assert DELAYED_REACH_TRAINING_MODE in payload["training_summary"]["training_mode"]


def test_delayed_no_integrator_run_spec_declares_reduced_state_and_pgd_dim(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        dry_run=True,
        issue="ffff699",
        delayed_reach=True,
        target_relative_multitarget=True,
        force_filter_feedback=True,
        no_integrator_state=True,
        broad_epsilon_pgd_training=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    result = write_run_spec(args)
    payload = result["run_spec"]

    assert payload["hps"]["model"]["no_integrator_state"] is True
    assert payload["hps"]["model"]["state_dim"] == 36
    assert payload["hps"]["model"]["physical_state_dim"] == 6
    assert payload["hps"]["broad_epsilon_pgd_training"]["epsilon_dim"] == 6
    assert payload["hps"]["broad_epsilon_pgd_training"]["epsilon_channel"]["shape"] == [
        "batch",
        "time",
        6,
    ]
    assert payload["model_summary"]["state_dim"] == 36
    assert payload["model_summary"]["physical_state_dim"] == 6
    assert payload["loss_summary"]["source_module"].endswith("build_no_integrator_game")
    assert payload["loss_summary"]["state_basis"]["physical_block_size"] == 6
    assert payload["loss_summary"]["state_basis"]["dimension"] == 36


def test_target_relative_proprioceptive_feedback_extends_sign_contract() -> None:
    component = TargetRelativeDelayedProprioceptiveFeedback()
    state = jnp.zeros((48,), dtype=jnp.float32)
    state = state.at[40:46].set(
        jnp.array([0.02, -0.03, 0.40, -0.20, 0.70, -0.80], dtype=jnp.float32)
    )
    outputs, _ = component(
        {"state": state, "target": jnp.array([0.15, 0.01], dtype=jnp.float32)},
        None,
        key=jr.PRNGKey(0),
    )

    assert outputs["feedback"].shape == (6,)
    assert jnp.allclose(
        outputs["feedback"],
        jnp.array([0.13, 0.04, -0.40, 0.20, 0.70, -0.80], dtype=jnp.float32),
    )


def test_force_filter_feedback_setup_uses_six_dimensional_feedback() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            target_relative_multitarget=True,
            force_filter_feedback=True,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    assert hps.target_relative_multitarget.force_filter_feedback is True
    assert hps.target_relative_multitarget.input_contract.shape == [6]
    assert hps.model.force_filter_feedback is True
    assert pair.model.nodes["net"].input_size == 6
    assert pair.model.nodes["sensory"].input_proto.shape[-1] == 6


def test_broad_epsilon_sampler_randomized_per_trial_and_l2_budgeted() -> None:
    base_hps = build_hps(
        _args(
            smoke=True,
            target_relative_multitarget=True,
            batch_size=16,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(base_hps, key=jr.PRNGKey(0))
    base = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    base = apply_validation_target_distribution(base, base_hps.target_relative_multitarget)
    cfg = BroadFullStateEpsilonTrainingConfig(enabled=True, level="strong")
    first = apply_broad_epsilon_training(base, cfg, jr.PRNGKey(2))
    second = apply_broad_epsilon_training(base, cfg, jr.PRNGKey(3))
    delta = first.inputs["epsilon"] - base.inputs["epsilon"]
    delta_second = second.inputs["epsilon"] - base.inputs["epsilon"]
    norms = jnp.sqrt(jnp.sum(jnp.square(delta), axis=(-2, -1)))
    reach = jnp.linalg.norm(
        first.targets["mechanics.effector.pos"].value[..., -1, :]
        - first.inits["mechanics.vector"][..., :2],
        axis=-1,
    )
    expected = cfg.reference_l2_radius * reach / cfg.nominal_reach_length_m

    assert first.inputs["epsilon"].shape[-2:] == (60, CS_EPSILON_DIM)
    assert jnp.allclose(norms, expected, rtol=1e-5, atol=1e-8)
    assert not jnp.allclose(delta[0], delta[1])
    assert not jnp.allclose(delta, delta_second)
    assert first.extra == base.extra


def test_broad_epsilon_run_spec_exposes_budget_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        target_relative_multitarget=True,
        broad_epsilon_training=True,
        broad_epsilon_level="moderate",
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    broad = payload["hps"]["broad_epsilon_training"]

    assert BROAD_EPSILON_TRAINING_MODE in payload["training_summary"]["training_mode"]
    assert broad["enabled"] is True
    assert broad["budget_contract"]["gamma_factor"] == pytest.approx(1.4)
    assert broad["budget_contract"]["effective_l2_radius_15cm"] == pytest.approx(
        0.0012324305441740995
    )
    assert (
        payload["model_summary"]["training_distribution"]["training_axes"][
            "broad_full_state_epsilon_training"
        ]
        is True
    )


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
        f"{TARGET_RELATIVE_MULTITARGET_TRAINING_MODE}+{PERTURBATION_TRAINING_MODE}"
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
    assert all("--target-relative-multitarget" in row["command"] for row in rows)
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
        f"{TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE}+{PERTURBATION_TRAINING_MODE}"
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
    assert (
        payload["training_summary"]["training_distribution"]["initial_hidden_encoder"]["enabled"]
        is True
    )
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
    assert summary["training_duration_seconds"] > 0
    assert summary["training_batches_per_second"] > 0
    assert len(summary["chunks"]) == 2
    assert summary["chunks"][0]["chunk_batches"] == 2
    assert summary["chunks"][0]["duration_seconds"] > 0
    assert summary["chunks"][0]["batches_per_second"] > 0
    assert commits == 3


def test_full_training_stop_after_batches_resumes_to_full_count(tmp_path: Path) -> None:
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
        stop_after_batches=2,
        controller_lr=1e-3,
        lr_warmup_batches=1,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        log_step=1,
        disable_progress=True,
        quiet_progress=True,
    )

    partial = run_full_training(args)
    partial_summary = json.loads((output_dir / "training_summary.json").read_text())

    assert partial["completed_batches"] == 2
    assert partial_summary["completed_batches"] == 2
    assert partial_summary["n_train_batches"] == 4
    assert partial_summary["stopped_early_for_checkpoint_gate"] is True
    assert partial_summary["stop_after_batches"] == 2
    assert (output_dir / "checkpoints" / "checkpoint_0000002").exists()

    resumed_args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        n_train_batches=4,
        batch_size=2,
        n_replicates=2,
        hidden_size=4,
        full_train=True,
        resume=True,
        checkpoint_interval_batches=2,
        stop_after_batches=None,
        controller_lr=1e-3,
        lr_warmup_batches=1,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        log_step=1,
        disable_progress=True,
        quiet_progress=True,
    )
    resumed = run_full_training(resumed_args)
    resumed_summary = json.loads((output_dir / "training_summary.json").read_text())
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())

    assert resumed["completed_batches"] == 4
    assert resumed_summary["completed_batches"] == 4
    assert resumed_summary["stopped_early_for_checkpoint_gate"] is False
    assert resumed_summary["stop_after_batches"] is None
    assert (output_dir / "checkpoints" / "checkpoint_0000004").exists()
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


def test_target_relative_h0_full_training_smoke_emits_diagnostics(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="643f101",
        n_train_batches=2,
        batch_size=2,
        n_replicates=2,
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
        f"{TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE}+{PERTURBATION_TRAINING_MODE}"
    )
    assert run_spec["model_summary"]["initial_hidden_encoder"]["enabled"] is True
    assert summary["training_diagnostics"]["enabled"] is True
    assert summary["training_diagnostics"]["written"] is True
    assert diagnostics_manifest["completed_batches"] == 2
    assert "optimizer_gradient_norm_pre_clip" in diagnostics_manifest["arrays"]
    with np.load(output_dir / "training_diagnostics.npz") as diagnostics:
        assert diagnostics["batch_index"].tolist() == [0, 1]
        assert diagnostics["optimizer_gradient_norm_pre_clip"].shape == (2, 2)
        assert diagnostics["train_loss__total"].shape == (2, 2)
        assert diagnostics["validation_loss__total"].shape == (2, 2)


def test_pgd_broad_epsilon_keeps_best_seen_endpoint_for_nonmonotone_ascent() -> None:
    class EchoTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            return trial_specs.inputs["epsilon"]

    class TinyTargetLoss:
        def __call__(self, states, trial_specs, model):
            del states, model
            epsilon = trial_specs.inputs["epsilon"]
            return TreeNamespace(total=-jnp.sum((epsilon - 1e-4) ** 2))

    trial_specs = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((1, 1, 2), dtype=jnp.float32),
                )
            }
        ),
        inputs={"epsilon": jnp.zeros((1, 1, 1), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=1),
    )
    config = {
        "enabled": True,
        "level": "moderate",
        "budget_scale": 1000.0,
        "reach_length_scaling": False,
        "n_steps": 1,
        "step_size_fraction": 2.0,
        "epsilon_dim": 1,
    }

    updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        EchoTask(),
        model=None,
        trial_specs=trial_specs,
        loss_func=TinyTargetLoss(),
        keys_model=None,
        config=config,
        return_diagnostics=True,
    )

    assert jnp.allclose(updated.inputs["epsilon"], 0.0)
    assert diagnostics["inner_objective_after"] == pytest.approx(
        diagnostics["inner_objective_before"]
    )
    assert diagnostics["inner_objective_best"] == pytest.approx(
        diagnostics["inner_objective_before"]
    )
    assert diagnostics["inner_objective_final_endpoint"] < diagnostics["inner_objective_best"]
    assert diagnostics["inner_objective_final_endpoint_gap"] > 0.0


def test_pgd_broad_epsilon_value_and_grad_matches_reference_ascent() -> None:
    class EchoTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            return trial_specs.inputs["epsilon"]

    target = jnp.asarray(
        [[[0.006, -0.003], [0.001, 0.004]]],
        dtype=jnp.float32,
    )

    class ShiftedQuadraticLoss:
        def __call__(self, states, trial_specs, model):
            del states, model
            epsilon = trial_specs.inputs["epsilon"]
            return TreeNamespace(total=-jnp.sum((epsilon - target) ** 2))

    trial_specs = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((1, 2, 2), dtype=jnp.float32),
                )
            }
        ),
        inputs={"epsilon": jnp.zeros((1, 2, 2), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=2),
    )
    config = {
        "enabled": True,
        "level": "moderate",
        "budget_scale": 2.0,
        "reach_length_scaling": False,
        "n_steps": 4,
        "step_size_fraction": 0.4,
        "epsilon_dim": 2,
    }

    def reference_inner_maximizer():
        cfg = config_from_broad_epsilon_pgd_hps(config)
        specs = _ensure_broad_epsilon_input(trial_specs, epsilon_dim=cfg.epsilon_dim)
        base_epsilon = jnp.asarray(specs.inputs["epsilon"])
        radius = _broad_epsilon_l2_radius(specs, cfg).astype(base_epsilon.dtype)
        time_mask = _epsilon_time_mask(specs, base_epsilon, cfg.movement_epoch_only)
        zero_delta = jnp.zeros_like(base_epsilon)

        def objective(delta_candidate):
            candidate = _set_input(specs, "epsilon", base_epsilon + delta_candidate * time_mask)
            candidate_states = EchoTask().eval_trials(None, candidate, None)
            return ShiftedQuadraticLoss()(candidate_states, candidate, None).total

        objective_initial = objective(zero_delta)

        def body(_, state):
            delta_current, best_delta, best_objective, _last_objective = state
            grad = jax.grad(objective)(delta_current) * time_mask
            step = _normalize_flattened_per_trial(grad) * _expand_radius(
                radius * jnp.asarray(cfg.step_size_fraction, dtype=base_epsilon.dtype),
                base_epsilon.ndim,
            )
            proposal = _project_flattened_per_trial_l2_ball(
                (delta_current + step) * time_mask,
                radius,
            )
            proposal_objective = objective(proposal)
            improved = proposal_objective > best_objective
            best_delta = jnp.where(_expand_bool_like(improved, proposal), proposal, best_delta)
            best_objective = jnp.where(improved, proposal_objective, best_objective)
            return proposal, best_delta, best_objective, proposal_objective

        final_delta, best_delta, objective_best, objective_final_endpoint = jax.lax.fori_loop(
            0,
            int(cfg.n_steps),
            body,
            (zero_delta, zero_delta, objective_initial, objective_initial),
        )
        del final_delta
        delta = jax.lax.stop_gradient(best_delta * time_mask)
        updated = _set_input(specs, "epsilon", base_epsilon + delta)
        objective_selected = objective(delta)
        return updated, {
            "inner_objective_before": objective_initial,
            "inner_objective_after": objective_selected,
            "inner_objective_improvement": objective_selected - objective_initial,
            "inner_objective_best": objective_best,
            "inner_objective_final_endpoint": objective_final_endpoint,
            "inner_objective_final_endpoint_gap": objective_best - objective_final_endpoint,
        }

    updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        EchoTask(),
        model=None,
        trial_specs=trial_specs,
        loss_func=ShiftedQuadraticLoss(),
        keys_model=None,
        config=config,
        return_diagnostics=True,
    )
    reference_updated, reference_diagnostics = reference_inner_maximizer()

    np.testing.assert_allclose(updated.inputs["epsilon"], reference_updated.inputs["epsilon"])
    for key, expected in reference_diagnostics.items():
        np.testing.assert_allclose(diagnostics[key], expected, rtol=1e-6, atol=1e-8)


def test_pgd_broad_epsilon_full_training_emits_inner_diagnostics(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="020a65b",
        n_train_batches=2,
        batch_size=2,
        n_replicates=5,
        hidden_size=4,
        target_relative_multitarget=True,
        force_filter_feedback=True,
        broad_epsilon_pgd_training=True,
        broad_epsilon_pgd_steps=1,
        broad_epsilon_pgd_step_size_fraction=0.5,
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
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())

    assert result["completed_batches"] == 2
    assert BROAD_EPSILON_PGD_TRAINING_MODE in run_spec["training_summary"]["training_mode"]
    assert run_spec["hps"]["broad_epsilon_pgd_training"]["inner_maximizer"]["n_steps"] == 1
    assert "pgd_broad_epsilon_inner_objective_before" in diagnostics_manifest["arrays"]
    assert "pgd_broad_epsilon_inner_objective_after" in diagnostics_manifest["arrays"]
    assert "pgd_broad_epsilon_inner_objective_improvement" in diagnostics_manifest["arrays"]
    assert "pgd_broad_epsilon_inner_objective_best" in diagnostics_manifest["arrays"]
    assert "pgd_broad_epsilon_inner_objective_final_endpoint" in diagnostics_manifest["arrays"]
    assert "pgd_broad_epsilon_inner_objective_final_endpoint_gap" in diagnostics_manifest["arrays"]
    assert "pgd_broad_epsilon_epsilon_norm_radius_ratio_mean" in diagnostics_manifest["arrays"]
    assert "pgd_broad_epsilon_boundary_fraction" in diagnostics_manifest["arrays"]
    with np.load(output_dir / "training_diagnostics.npz") as diagnostics:
        assert diagnostics["pgd_broad_epsilon_diagnostic_sampled"].tolist() == [True, True]
        assert diagnostics["pgd_broad_epsilon_radius_mean"].shape == (2, 5)
        assert np.isfinite(diagnostics["pgd_broad_epsilon_radius_mean"]).all()
        assert np.isfinite(diagnostics["pgd_broad_epsilon_epsilon_norm_radius_ratio_mean"]).all()
        assert np.all(diagnostics["pgd_broad_epsilon_epsilon_norm_radius_ratio_mean"] <= 1.0001)
        assert np.isfinite(diagnostics["pgd_broad_epsilon_inner_objective_before"]).all()
        assert np.isfinite(diagnostics["pgd_broad_epsilon_inner_objective_after"]).all()
        assert np.isfinite(diagnostics["pgd_broad_epsilon_inner_objective_improvement"]).all()
        assert np.isfinite(diagnostics["pgd_broad_epsilon_inner_objective_best"]).all()
        assert np.isfinite(diagnostics["pgd_broad_epsilon_inner_objective_final_endpoint"]).all()
        assert np.isfinite(
            diagnostics["pgd_broad_epsilon_inner_objective_final_endpoint_gap"]
        ).all()
        assert np.all(diagnostics["pgd_broad_epsilon_inner_objective_final_endpoint_gap"] >= -1e-6)
        assert np.any(diagnostics["pgd_broad_epsilon_epsilon_norm_mean"] > 0.0)


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


def test_setup_task_model_pair_trains_tiny_nominal_simple_reach_smoke() -> None:
    n_batches = 3
    args = _args(
        smoke=True,
        batch_size=2,
        n_train_batches=n_batches,
    )
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    schedule = make_delayed_cosine_schedule(
        float(hps.learning_rate_0),
        constant_steps=0,
        total_steps=n_batches,
    )
    optimizer = optax.inject_hyperparams(partial(optax.adamw, weight_decay=0.0))(
        learning_rate=schedule
    )
    trainer = TaskTrainer(optimizer=optimizer, checkpointing=False)

    trained, _history = train_pair(
        trainer,
        pair,
        n_batches=n_batches,
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
