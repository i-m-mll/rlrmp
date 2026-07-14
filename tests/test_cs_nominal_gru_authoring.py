"""C&S nominal-GRU authoring contract tests."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from feedbax import TaskTrialSpec, TrialTimeline, WhereDict
from feedbax.contracts.training import TrainingRunSpec
from feedbax.objectives.loss import TargetSpec
from feedbax.config.namespace import TreeNamespace
from pydantic import ValidationError
from rlrmp.model.cs_lss_gru import (
    FINITE_EPSILON_POLICY_GRAPH_COMPONENT,
    FINITE_EPSILON_POLICY_NODE_LABEL,
)
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)
from rlrmp.paths import REPO_ROOT
import rlrmp.train.cs_nominal_gru as cs_nominal_gru
from rlrmp.train.cs_nominal_gru import (
    CS_DELAYED_REACH_TASK_TYPE,
    CsNominalGruConfig,
    DEFAULT_DELAYED_P_CATCH_TRIAL,
    DEFAULT_STOCHASTIC_PRESET,
    DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
    build_training_run_graph_spec,
    build_hps,
    render_run_spec_execution_dry_run,
    _initial_adaptive_epsilon_state,
    write_run_spec,
)
from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    build_feedbax_training_run_spec,
)
from rlrmp.data_products.broad_epsilon import load_pgd_radius_source
from rlrmp.train.executor.cs_supervised import build_execution_context_from_spec
from rlrmp.train.run_spec_authoring import COMPACT_RUN_SPEC_KEY
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_ADAM,
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
    BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    CLOSED_LOOP_SENSORY_CALIBRATION_REGIME,
    CLOSED_LOOP_SENSORY_COMMAND_LATERAL_CALIBRATION_REGIME,
    DEFAULT_PGD_SISU_EXACT_ZERO_MASS,
    DEFAULT_PGD_SISU_LEVELS,
    AFFINE_POLICY,
    LINEAR_NO_BIAS_POLICY,
    MILD_COMBINED_FAMILIES,
    POLICY_ADVERSARY_MEMORYLESS_MLP,
    POLICY_ADVERSARY_ENERGY_MODE,
    OPEN_LOOP_ALL_CALIBRATION_REGIME,
    TARGET_SUPPORT_CONST_REACH_M,
    TARGET_SUPPORT_DENSE_N_DIRECTIONS,
    TARGET_SUPPORT_PROFILE_020A65B,
    TARGET_SUPPORT_PROFILE_CONST_BAND8,
    TARGET_SUPPORT_PROFILE_CONST_BAND16,
    TARGET_SUPPORT_PROFILE_CONST_BAND36,
    TARGET_SUPPORT_PROFILE_CONST_DENSE_ALL,
    TARGET_SUPPORT_PROFILE_CONST_SPARSE8,
    TARGET_SUPPORT_SPARSE_N_DIRECTIONS,
    BroadFullStateEpsilonTrainingTaskAdapter,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    TargetRelativeMultiTargetTrainingConfig,
    TargetRelativeMultiTargetTrainingTaskAdapter,
    FixedTargetPerturbationTrainingConfig,
    _active_single_family_bins,
    _closed_loop_amplitudes_by_timing,
    calibration_regime_manifest,
    apply_training_target_distribution,
    run_broad_epsilon_pgd_inner_maximizer,
    target_relative_target_support_config,
)
from rlrmp.train.science_vocabulary import (
    AdaptiveEpsilonControllerMode,
    ScienceMode,
)
from rlrmp.train.executor.equivalence import assert_paired_equivalent, run_paired_equivalence
from rlrmp.train.task_model import (
    CS_LSS_PLANT_BACKEND,
    _add_cs_lss_task_inputs,
    _CsLssTaskAdapter,
    build_task_base,
)
from rlrmp.train.closed_loop_finite_adversary import (
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
)

HISTORICAL_020A65B_PGD_RADIUS_15CM = float(
    load_pgd_radius_source("effective_020a65b_pgd_training_radius")["l2_radius_15cm"]
)


def _args(**overrides) -> argparse.Namespace:
    values = CsNominalGruConfig(
        issue="test",
        output_dir="_artifacts/test/runs/test",
    ).model_dump(mode="python")
    values.update(compact_run_spec=False, verify_resume_only=False)
    values.update(overrides)
    return argparse.Namespace(**values)


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


def test_cs_nominal_gru_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CsNominalGruConfig.model_validate({"seed": 42, "unknown_field": True})


def test_run_spec_execution_requires_explicit_artifact_output_dir() -> None:
    with pytest.raises(ValueError, match="must declare a non-empty artifact_output_dir"):
        cs_nominal_gru._args_values_from_run_spec({"issue": "b2562ad", "hps": {}})


def test_build_hps_accepts_legacy_namespace_extras_and_validates_canonical_fields() -> None:
    args = _args(batch_size=3)
    args.legacy_payload_marker = {"source": "historical-run-spec"}

    hps = build_hps(args)

    assert hps.batch_size == 3
    with pytest.raises(ValidationError):
        build_hps(_args(batch_size=0))


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

    cfg = PgdFullStateEpsilonTrainingConfig.from_payload(hps.broad_epsilon_pgd_training)

    assert cfg.enabled is True
    assert hps.broad_epsilon_pgd_training.mode == ScienceMode.BROAD_EPSILON_PGD
    assert hps.broad_epsilon_pgd_training.inner_maximizer.n_steps == 4
    assert (
        hps.broad_epsilon_pgd_training.inner_maximizer.differentiated_through_outer_update is False
    )
    assert hps.broad_epsilon_training.enabled is False


def test_pgd_broad_epsilon_hps_parser_requires_canonical_config_snapshot() -> None:
    authored = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        level="strong",
        budget_scale=1.5,
        reach_length_scaling=False,
        n_steps=9,
        step_size_fraction=0.125,
    )

    parsed = PgdFullStateEpsilonTrainingConfig.from_payload(authored.to_hps_dict())

    assert parsed == authored
    with pytest.raises(ValueError, match="[Ee]xtra inputs are not permitted"):
        PgdFullStateEpsilonTrainingConfig.from_payload(
            TreeNamespace(
                enabled=True,
                level="strong",
                inner_maximizer=TreeNamespace(n_steps=9),
            )
        )


def test_pgd_sisu_budget_schedule_metadata_and_parser_round_trip() -> None:
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        budget_schedule=BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
        reach_length_scaling=False,
        sisu_condition_input="input",
        sisu_max_l2_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
        sisu_max_radius_source="effective_020a65b_pgd_training_radius",
    )
    payload = cfg.to_hps_dict()
    schedule = payload["budget_schedule"]

    assert schedule["mode"] == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE
    assert schedule["levels"] == list(DEFAULT_PGD_SISU_LEVELS)
    assert schedule["probabilities"] == pytest.approx([0.30, 0.175, 0.175, 0.175, 0.175])
    assert schedule["exact_zero_mass"] == pytest.approx(DEFAULT_PGD_SISU_EXACT_ZERO_MASS)
    assert schedule["mapping_rule"] == "epsilon_l2_radius = max_l2_radius_15cm * sqrt(SISU)"
    assert schedule["max_l2_radius_15cm"] == pytest.approx(HISTORICAL_020A65B_PGD_RADIUS_15CM)
    assert schedule["conditioning_scalar"]["input_key"] == "input"
    assert (
        schedule["max_radius_source"]["source_kind"]
        == "historical_replay_effective_pgd_training_radius"
    )
    assert schedule["max_radius_source"]["gamma_equivalent_analytical_anchor"] is False
    assert payload["budget_contract"]["active_max_l2_radius_15cm"] == pytest.approx(
        HISTORICAL_020A65B_PGD_RADIUS_15CM
    )

    parsed = PgdFullStateEpsilonTrainingConfig.from_payload(payload)
    assert parsed.budget_schedule == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE
    assert parsed.sisu_condition_input == "input"
    assert parsed.sisu_levels == DEFAULT_PGD_SISU_LEVELS
    assert parsed.sisu_exact_zero_mass == pytest.approx(DEFAULT_PGD_SISU_EXACT_ZERO_MASS)
    assert parsed.sisu_max_l2_radius == pytest.approx(HISTORICAL_020A65B_PGD_RADIUS_15CM)
    assert parsed.sisu_max_radius_source == "effective_020a65b_pgd_training_radius"


def test_pgd_fixed_radius_metadata_and_parser_round_trip() -> None:
    radius = 0.004545011406169036
    source = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        level="moderate",
        budget_scale=1.0,
        fixed_l2_radius_15cm=radius,
        fixed_radius_source=source,
    )
    payload = cfg.to_hps_dict()
    budget = payload["budget_contract"]

    assert cfg.reference_l2_radius == pytest.approx(radius)
    assert budget["effective_l2_radius_15cm"] == pytest.approx(radius)
    assert budget["active_max_l2_radius_15cm"] == pytest.approx(radius)
    assert budget["budget_source"]["key"] == source
    assert budget["budget_source"]["source_kind"] == "output_feedback_rollout_budget"
    assert budget["budget_source"]["gamma_factor"] == pytest.approx(1.4)
    assert budget["budget_source"]["epsilon_dim"] == 6

    parsed = PgdFullStateEpsilonTrainingConfig.from_payload(payload)
    assert parsed.fixed_l2_radius_15cm == pytest.approx(radius)
    assert parsed.fixed_radius_source == source
    assert parsed.reference_l2_radius == pytest.approx(radius)


def test_pgd_soft_energy_metadata_lambda_mapping_and_parser_round_trip() -> None:
    gamma_star = 9166.831285473823
    gamma_factor = 1.4
    cap_radius = 0.004545011406169036
    cap_source = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        energy_gamma_star=gamma_star,
        energy_gamma_factor=gamma_factor,
        energy_penalty_scale=1.0,
        safety_cap_l2_radius_15cm=cap_radius,
        safety_cap_source=cap_source,
    )
    payload = cfg.to_hps_dict()
    objective = payload["objective"]
    safety_cap = payload["safety_cap"]

    assert objective["kind"] == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
    assert objective["gamma_star"] == pytest.approx(gamma_star)
    assert objective["gamma_factor"] == pytest.approx(gamma_factor)
    assert objective["gamma"] == pytest.approx(gamma_star * gamma_factor)
    assert objective["penalty_scale_c"] == pytest.approx(1.0)
    assert objective["lambda"] == pytest.approx((gamma_star * gamma_factor) ** 2)
    assert objective["hard_l2_projection_is_scientific_constraint"] is False
    assert safety_cap["enabled"] is True
    assert safety_cap["l2_radius_15cm"] == pytest.approx(cap_radius)
    assert safety_cap["source"]["key"] == cap_source
    assert safety_cap["hard_budget_scientific_constraint"] is False
    assert payload["budget_contract"]["scientific_constraint"] == "soft_energy_penalty"

    parsed = PgdFullStateEpsilonTrainingConfig.from_payload(payload)
    assert parsed.objective_kind == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
    assert parsed.soft_energy_gamma == pytest.approx(gamma_star * gamma_factor)
    assert parsed.soft_energy_lambda == pytest.approx((gamma_star * gamma_factor) ** 2)
    assert parsed.safety_cap_l2_radius == pytest.approx(cap_radius)
    assert parsed.safety_cap_source == cap_source


def test_adaptive_epsilon_curriculum_hps_contract() -> None:
    hps = build_hps(
        _args(
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            broad_epsilon_pgd_energy_lambda=2.5,
            adaptive_epsilon_curriculum=True,
            target_relative_multitarget=True,
        )
    )

    cfg = hps.adaptive_epsilon_curriculum
    assert cfg.enabled is True
    assert cfg.controller_training_mode == AdaptiveEpsilonControllerMode.LOSS_BLEND
    assert cfg.damage_schedule.peak == pytest.approx(3500.0)
    assert cfg.damage_schedule.final == pytest.approx(1000.0)
    assert cfg.damage_schedule.ramp_batches == 2500
    assert cfg.damage_schedule.anneal_batches == 5000
    assert cfg.lambda_update.interval_batches == 50
    assert cfg.lambda_update.eta == pytest.approx(0.1)
    assert cfg.lambda_update.deadband_frac == pytest.approx(0.10)
    assert cfg.lambda_update.freeze_during_application_ramp is False
    assert cfg.lambda_update.gain_normalization is False
    assert cfg.lambda_update.gain_ema_alpha == pytest.approx(0.2)
    assert cfg.lambda_update.gain_min == pytest.approx(0.25)
    assert cfg.lambda_update.gain_max == pytest.approx(8.0)
    assert cfg.lambda_update.lambda_min == pytest.approx(2.5e-3)
    assert cfg.outer_adversarial_weight.ramp_batches == 2500
    assert cfg.outer_adversarial_weight.applies_to == "optimized_direct_epsilon_loss_only"
    adaptive_state = _initial_adaptive_epsilon_state(hps)
    assert adaptive_state is not None
    assert adaptive_state.lambda_value == pytest.approx(2.5)


def test_adaptive_epsilon_run_spec_replay_preserves_curriculum(tmp_path: Path) -> None:
    output_dir = tmp_path / "_artifacts" / "91a090c" / "runs" / "smoke"
    spec_dir = tmp_path / "results" / "91a090c" / "runs" / "smoke"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="91a090c",
        target_relative_multitarget=True,
        broad_epsilon_pgd_training=True,
        broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        broad_epsilon_pgd_energy_lambda=2.5,
        adaptive_epsilon_curriculum=True,
        adaptive_epsilon_controller_training_mode=(
            AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER
        ),
        adaptive_epsilon_damage_peak=3500.0,
        adaptive_epsilon_damage_final=1000.0,
        adaptive_epsilon_damage_ramp_batches=1,
        adaptive_epsilon_damage_anneal_batches=2,
        adaptive_epsilon_update_interval_batches=3,
        adaptive_epsilon_ema_alpha=0.2,
        adaptive_epsilon_eta=0.3,
        adaptive_epsilon_deadband_frac=0.4,
        adaptive_epsilon_hysteresis_frac=0.45,
        adaptive_epsilon_freeze_during_application_ramp=True,
        adaptive_epsilon_gain_normalization=True,
        adaptive_epsilon_gain_ema_alpha=0.25,
        adaptive_epsilon_gain_min=0.5,
        adaptive_epsilon_gain_max=4.0,
        adaptive_epsilon_lambda_min=1e-9,
        adaptive_epsilon_max_log_step=0.5,
        adaptive_epsilon_outer_weight_ramp_batches=6,
    )

    result = write_run_spec(args)
    replay_args = build_execution_context_from_spec(result["run_spec_path"]).args

    assert replay_args.adaptive_epsilon_curriculum is True
    assert (
        replay_args.adaptive_epsilon_controller_training_mode
        == AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER
    )
    assert replay_args.adaptive_epsilon_damage_peak == pytest.approx(3500.0)
    assert replay_args.adaptive_epsilon_damage_final == pytest.approx(1000.0)
    assert replay_args.adaptive_epsilon_damage_ramp_batches == 1
    assert replay_args.adaptive_epsilon_damage_anneal_batches == 2
    assert replay_args.adaptive_epsilon_update_interval_batches == 3
    assert replay_args.adaptive_epsilon_ema_alpha == pytest.approx(0.2)
    assert replay_args.adaptive_epsilon_eta == pytest.approx(0.3)
    assert replay_args.adaptive_epsilon_deadband_frac == pytest.approx(0.4)
    assert replay_args.adaptive_epsilon_hysteresis_frac == pytest.approx(0.45)
    assert replay_args.adaptive_epsilon_freeze_during_application_ramp is True
    assert replay_args.adaptive_epsilon_gain_normalization is True
    assert replay_args.adaptive_epsilon_gain_ema_alpha == pytest.approx(0.25)
    assert replay_args.adaptive_epsilon_gain_min == pytest.approx(0.5)
    assert replay_args.adaptive_epsilon_gain_max == pytest.approx(4.0)
    assert replay_args.adaptive_epsilon_lambda_min == pytest.approx(1e-9)
    assert replay_args.adaptive_epsilon_max_log_step == pytest.approx(0.5)
    assert replay_args.adaptive_epsilon_outer_weight_ramp_batches == 6

    hps = build_hps(replay_args)
    assert hps.adaptive_epsilon_curriculum.enabled is True


def test_adaptive_epsilon_scaled_outer_training_hps_contract() -> None:
    hps = build_hps(
        _args(
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            broad_epsilon_pgd_energy_lambda=2.5,
            adaptive_epsilon_curriculum=True,
            adaptive_epsilon_controller_training_mode=(
                AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER
            ),
            target_relative_multitarget=True,
        )
    )

    cfg = hps.adaptive_epsilon_curriculum
    assert cfg.controller_training_mode == AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER
    assert (
        cfg.outer_adversarial_weight.applies_to
        == "optimized_direct_epsilon_channel_scale_for_controller_rollout"
    )


def test_pgd_inner_optimizer_metadata_and_parser_round_trip() -> None:
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=LINEAR_NO_BIAS_POLICY,
        objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        energy_lambda=3.0,
        safety_cap_l2_radius_15cm=1.0,
        safety_cap_source="effective_020a65b_pgd_training_radius",
        inner_optimizer_method=BROAD_EPSILON_PGD_ADAM,
        adam_learning_rate=1e-3,
        adam_b1=0.8,
        adam_b2=0.95,
        adam_eps=1e-6,
    )
    payload = cfg.to_hps_dict()
    optimizer = payload["inner_maximizer"]

    assert optimizer["method"] == BROAD_EPSILON_PGD_ADAM
    assert optimizer["learning_rate"] == pytest.approx(1e-3)
    assert optimizer["adam"]["b1"] == pytest.approx(0.8)
    assert optimizer["adam"]["b2"] == pytest.approx(0.95)
    assert optimizer["adam"]["eps"] == pytest.approx(1e-6)

    parsed = PgdFullStateEpsilonTrainingConfig.from_payload(payload)
    assert parsed.inner_optimizer_method == BROAD_EPSILON_PGD_ADAM
    assert parsed.adam_learning_rate == pytest.approx(1e-3)
    assert parsed.adam_b1 == pytest.approx(0.8)
    assert parsed.adam_b2 == pytest.approx(0.95)
    assert parsed.adam_eps == pytest.approx(1e-6)


def test_pgd_hard_l2_default_metadata_preserves_existing_projection_contract() -> None:
    cfg = PgdFullStateEpsilonTrainingConfig(enabled=True)
    payload = cfg.to_hps_dict()

    assert cfg.adversary_mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    assert payload["adversary_mechanism"] == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    assert payload["mechanism"]["name"] == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    assert payload["mechanism"]["implementation_status"] == "implemented"
    assert payload["mechanism"]["matches_legacy_default"] is True
    assert cfg.objective_kind == BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE
    assert payload["objective"]["kind"] == BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE
    assert payload["objective"]["hard_l2_projection_is_scientific_constraint"] is True
    assert payload["safety_cap"]["enabled"] is False
    assert payload["inner_maximizer"]["method"] == BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT
    assert payload["inner_maximizer"]["projection"] == "per_trial_flattened_time_component_l2_ball"
    assert payload["budget_contract"]["scientific_constraint"] == "hard_l2_projection"


def test_pgd_soft_energy_cli_and_run_spec_metadata(tmp_path: Path) -> None:
    gamma_star = 9166.831285473823
    gamma_factor = 1.05
    cap_radius = 0.004545011406169036
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="c92ebd8",
        target_relative_multitarget=True,
        force_filter_feedback=True,
        broad_epsilon_pgd_training=True,
        no_integrator_state=True,
        broad_epsilon_pgd_steps=10,
        broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        broad_epsilon_pgd_energy_gamma_star=gamma_star,
        broad_epsilon_pgd_energy_gamma_factor=gamma_factor,
        broad_epsilon_pgd_energy_penalty_scale=1.0,
        broad_epsilon_pgd_safety_cap_15cm=cap_radius,
        broad_epsilon_pgd_safety_cap_source="ofb_6d_no_integrator_gamma_1p4_rollout_radius",
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    pgd = payload["hps"]["broad_epsilon_pgd_training"]
    objective = pgd["objective"]

    assert objective["kind"] == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
    assert objective["gamma"] == pytest.approx(gamma_star * gamma_factor)
    assert objective["lambda"] == pytest.approx((gamma_star * gamma_factor) ** 2)
    assert pgd["safety_cap"]["l2_radius_15cm"] == pytest.approx(cap_radius)
    assert pgd["adversary_mechanism"] == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    assert pgd["mechanism"]["implementation_status"] == "implemented"
    assert payload["adversarial_phase"] == "broad_epsilon_pgd_direct_epsilon"
    assert payload["task_timing"]["extra_inputs"] == ["target", "epsilon"]
    replay_args = build_execution_context_from_spec(result["run_spec_path"]).args
    assert replay_args.broad_epsilon_pgd_objective == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
    assert replay_args.broad_epsilon_pgd_mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    assert replay_args.broad_epsilon_pgd_energy_lambda == pytest.approx(
        (gamma_star * gamma_factor) ** 2
    )
    assert replay_args.broad_epsilon_pgd_safety_cap_15cm == pytest.approx(cap_radius)


def test_pgd_mechanism_cli_run_spec_and_replay_for_linear_no_bias(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="ae9f30f",
        target_relative_multitarget=True,
        force_filter_feedback=True,
        broad_epsilon_pgd_training=True,
        broad_epsilon_pgd_mechanism=LINEAR_NO_BIAS_POLICY,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    pgd = payload["hps"]["broad_epsilon_pgd_training"]

    assert pgd["adversary_mechanism"] == LINEAR_NO_BIAS_POLICY
    assert pgd["mechanism"]["implementation_status"] == "implemented"
    assert pgd["mechanism"]["graph_component"] == FINITE_EPSILON_POLICY_GRAPH_COMPONENT
    assert pgd["mechanism"]["graph_component_label"] == FINITE_EPSILON_POLICY_NODE_LABEL
    assert pgd["mechanism"]["live_evaluation"]["implementation"] == "graph_component"
    assert pgd["mechanism"]["live_evaluation"]["hook"] is None
    assert pgd["mechanism"]["live_evaluation"]["input_keys"] == [
        "epsilon",
        FINITE_POLICY_GAINS_INPUT,
    ]
    assert pgd["mechanism"]["runtime_inputs"]["base_epsilon"] == "TaskTrialSpec.inputs['epsilon']"
    assert payload["adversarial_phase"] == "broad_epsilon_pgd_live_finite_policy_linear_no_bias"
    assert payload["training_summary"]["adversarial_phase"] == (
        "broad_epsilon_pgd_live_finite_policy_linear_no_bias"
    )
    assert payload["training_summary"]["training_distribution"]["adversarial_phase"] == (
        "broad_epsilon_pgd_live_finite_policy_linear_no_bias"
    )
    assert payload["task_timing"]["extra_inputs"][-1] == FINITE_POLICY_GAINS_INPUT
    replay_args = build_execution_context_from_spec(result["run_spec_path"]).args
    assert replay_args.broad_epsilon_pgd_mechanism == LINEAR_NO_BIAS_POLICY


def test_pgd_mechanism_run_spec_for_affine_lists_bias_input(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="ae9f30f",
        target_relative_multitarget=True,
        force_filter_feedback=True,
        broad_epsilon_pgd_training=True,
        broad_epsilon_pgd_mechanism=AFFINE_POLICY,
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    pgd = payload["hps"]["broad_epsilon_pgd_training"]

    assert pgd["mechanism"]["live_evaluation"]["input_keys"] == [
        "epsilon",
        FINITE_POLICY_GAINS_INPUT,
        FINITE_POLICY_BIAS_INPUT,
    ]
    assert payload["adversarial_phase"] == "broad_epsilon_pgd_live_finite_policy_affine"
    assert payload["task_timing"]["extra_inputs"][-2:] == [
        FINITE_POLICY_GAINS_INPUT,
        FINITE_POLICY_BIAS_INPUT,
    ]


@pytest.mark.parametrize("policy_class", [LINEAR_NO_BIAS_POLICY, AFFINE_POLICY])
def test_live_finite_pgd_run_spec_replays_adam_inner_optimizer(
    policy_class: str,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="9bb676f",
        target_relative_multitarget=True,
        force_filter_feedback=True,
        broad_epsilon_pgd_training=True,
        broad_epsilon_pgd_mechanism=policy_class,
        broad_epsilon_pgd_inner_optimizer_method=BROAD_EPSILON_PGD_ADAM,
        broad_epsilon_pgd_adam_lr=2e-3,
        broad_epsilon_pgd_adam_b1=0.85,
        broad_epsilon_pgd_adam_b2=0.97,
        broad_epsilon_pgd_adam_eps=1e-6,
        broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        broad_epsilon_pgd_energy_lambda=10.0,
        broad_epsilon_pgd_safety_cap_15cm=1.0,
        broad_epsilon_pgd_safety_cap_source="effective_020a65b_pgd_training_radius",
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    pgd = payload["hps"]["broad_epsilon_pgd_training"]
    optimizer = pgd["inner_maximizer"]
    replay_args = build_execution_context_from_spec(result["run_spec_path"]).args

    assert pgd["adversary_mechanism"] == policy_class
    assert pgd["mechanism"]["live_evaluation"]["implementation"] == "graph_component"
    assert payload["hps"]["policy_adversary_training"]["enabled"] is False
    assert optimizer["method"] == BROAD_EPSILON_PGD_ADAM
    assert optimizer["learning_rate"] == pytest.approx(2e-3)
    assert optimizer["adam"]["b1"] == pytest.approx(0.85)
    assert optimizer["adam"]["b2"] == pytest.approx(0.97)
    assert optimizer["adam"]["eps"] == pytest.approx(1e-6)
    assert (
        payload["training_distribution"]["broad_epsilon_pgd_training"]["inner_maximizer"]["method"]
        == BROAD_EPSILON_PGD_ADAM
    )
    assert replay_args.broad_epsilon_pgd_mechanism == policy_class
    assert replay_args.broad_epsilon_pgd_inner_optimizer_method == BROAD_EPSILON_PGD_ADAM
    assert replay_args.broad_epsilon_pgd_adam_lr == pytest.approx(2e-3)
    assert replay_args.broad_epsilon_pgd_adam_b1 == pytest.approx(0.85)
    assert replay_args.broad_epsilon_pgd_adam_b2 == pytest.approx(0.97)
    assert replay_args.broad_epsilon_pgd_adam_eps == pytest.approx(1e-6)


def test_direct_epsilon_pgd_dict_and_config_routing_are_equivalent() -> None:
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

    def run(config_value):
        updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
            EchoTask(),
            model=None,
            trial_specs=trial_specs,
            loss_func=ShiftedQuadraticLoss(),
            keys_model=None,
            config=config_value,
            return_diagnostics=True,
        )
        return {
            "epsilon": updated.inputs["epsilon"],
            "inner_objective_after": diagnostics["inner_objective_after"],
            "inner_objective_best": diagnostics["inner_objective_best"],
            "inner_objective_final_endpoint": diagnostics["inner_objective_final_endpoint"],
        }

    report = run_paired_equivalence(
        "pgd_direct_epsilon_inner_maximizer",
        lambda: run(config),
        lambda: run(PgdFullStateEpsilonTrainingConfig.from_payload(config)),
        left_label="dict_config",
        right_label="model_config",
    )

    assert_paired_equivalent(report)


def test_pgd_sisu_budget_cli_and_run_spec_metadata(tmp_path: Path) -> None:
    args = _args(
        output_dir=str(tmp_path / "artifacts"),
        spec_dir=str(tmp_path / "spec"),
        dry_run=True,
        broad_epsilon_pgd_training=True,
        target_relative_multitarget=True,
        broad_epsilon_reach_scaling=False,
        broad_epsilon_pgd_budget_schedule=BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
        broad_epsilon_pgd_sisu_condition_input="input",
        broad_epsilon_pgd_sisu_max_radius=HISTORICAL_020A65B_PGD_RADIUS_15CM,
        broad_epsilon_pgd_sisu_max_radius_source="effective_020a65b_pgd_training_radius",
    )

    payload = write_run_spec(args)["run_spec"]
    pgd = payload["hps"]["broad_epsilon_pgd_training"]
    distribution = payload["training_summary"]["training_distribution"][
        "broad_epsilon_pgd_training"
    ]

    assert pgd["enabled"] is True
    assert pgd["reach_length_scaling"] is False
    assert pgd["budget_schedule"]["mode"] == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE
    assert pgd["budget_schedule"]["probabilities"] == pytest.approx(
        [0.30, 0.175, 0.175, 0.175, 0.175]
    )
    assert pgd["budget_schedule"]["max_l2_radius_15cm"] == pytest.approx(
        HISTORICAL_020A65B_PGD_RADIUS_15CM
    )
    assert pgd["budget_schedule"]["max_radius_source"]["source_kind"] == (
        "historical_replay_effective_pgd_training_radius"
    )
    assert (
        pgd["budget_schedule"]["max_radius_source"]["gamma_equivalent_analytical_anchor"] is False
    )
    assert distribution["budget_schedule"]["mode"] == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE
    assert payload["task_timing"]["extra_inputs"] == ["input", "target", "epsilon"]


def test_policy_adversary_hps_declares_memoryless_policy_and_excludes_pgd() -> None:
    hps = build_hps(
        _args(
            target_relative_multitarget=True,
            force_filter_feedback=True,
            initial_hidden_encoder=True,
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_physical_level="small",
            policy_adversary_training=True,
            policy_adversary_mode=POLICY_ADVERSARY_ENERGY_MODE,
            policy_adversary_steps=5,
            policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
            policy_adversary_radius_source="effective_020a65b_pgd_training_radius",
            broad_epsilon_reach_scaling=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        )
    )
    cfg = PolicyFullStateEpsilonTrainingConfig.from_payload(hps.policy_adversary_training)

    assert cfg.enabled is True
    assert cfg.mode == POLICY_ADVERSARY_ENERGY_MODE
    assert cfg.n_steps == 5
    assert cfg.width == 64
    assert cfg.epsilon_dim == 8
    assert cfg.state_feature_dim == 48
    assert cfg.reference_l2_radius == pytest.approx(HISTORICAL_020A65B_PGD_RADIUS_15CM)
    assert hps.policy_adversary_training.mode == ScienceMode.POLICY_ADVERSARY
    assert hps.policy_adversary_training.policy_class == POLICY_ADVERSARY_MEMORYLESS_MLP
    assert hps.policy_adversary_training.policy.kind == POLICY_ADVERSARY_MEMORYLESS_MLP
    assert hps.policy_adversary_training.objective.formal_certificate is False
    assert hps.broad_epsilon_pgd_training.enabled is False
    assert hps.broad_epsilon_training.enabled is False

    with pytest.raises(ValueError, match="separate broad-epsilon lanes"):
        build_hps(
            _args(
                target_relative_multitarget=True,
                policy_adversary_training=True,
                policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
                policy_adversary_radius_source="effective_020a65b_pgd_training_radius",
                broad_epsilon_pgd_training=True,
            )
        )


def test_policy_adversary_training_requires_explicit_radius_and_source() -> None:
    required_args = dict(
        target_relative_multitarget=True,
        force_filter_feedback=True,
        initial_hidden_encoder=True,
        perturbation_training=True,
        perturbation_calibrated_timing=True,
        perturbation_physical_level="small",
        policy_adversary_training=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    with pytest.raises(ValueError, match="explicit reference_l2_radius_15cm"):
        build_hps(_args(**required_args))

    with pytest.raises(ValueError, match="explicit budget_source"):
        build_hps(
            _args(
                **required_args,
                policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
            )
        )


def test_policy_adversary_run_spec_replay_preserves_payload_fields(tmp_path: Path) -> None:
    output_dir = tmp_path / "_artifacts" / "8f7c282" / "runs" / "policy"
    spec_dir = tmp_path / "results" / "8f7c282" / "runs" / "policy"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="8f7c282",
        target_relative_multitarget=True,
        force_filter_feedback=True,
        initial_hidden_encoder=True,
        perturbation_training=True,
        perturbation_calibrated_timing=True,
        perturbation_physical_level="small",
        policy_adversary_training=True,
        policy_adversary_policy_class=POLICY_ADVERSARY_MEMORYLESS_MLP,
        policy_adversary_mode=POLICY_ADVERSARY_ENERGY_MODE,
        policy_adversary_width=11,
        policy_adversary_depth=3,
        policy_adversary_steps=7,
        policy_adversary_lr=1e-3,
        policy_adversary_energy_gamma=1.7,
        policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
        policy_adversary_radius_source="effective_020a65b_pgd_training_radius",
        broad_epsilon_reach_scaling=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    result = write_run_spec(args)
    replay_args = build_execution_context_from_spec(result["run_spec_path"]).args

    assert replay_args.policy_adversary_training is True
    assert replay_args.policy_adversary_policy_class == POLICY_ADVERSARY_MEMORYLESS_MLP
    assert replay_args.policy_adversary_mode == POLICY_ADVERSARY_ENERGY_MODE
    assert replay_args.policy_adversary_width == 11
    assert replay_args.policy_adversary_depth == 3
    assert replay_args.policy_adversary_steps == 7
    assert replay_args.policy_adversary_lr == pytest.approx(1e-3)
    assert replay_args.policy_adversary_energy_gamma == pytest.approx(1.7)
    assert replay_args.policy_adversary_radius_15cm == pytest.approx(
        HISTORICAL_020A65B_PGD_RADIUS_15CM
    )
    assert replay_args.policy_adversary_radius_source == "effective_020a65b_pgd_training_radius"

    hps = build_hps(replay_args)
    cfg = PolicyFullStateEpsilonTrainingConfig.from_payload(hps.policy_adversary_training)
    assert cfg.enabled is True
    assert cfg.policy_class == POLICY_ADVERSARY_MEMORYLESS_MLP
    assert cfg.mode == POLICY_ADVERSARY_ENERGY_MODE


@pytest.mark.parametrize("policy_class", [LINEAR_NO_BIAS_POLICY, AFFINE_POLICY])
def test_finite_policy_adversary_hps_declares_active_adam_and_excludes_pgd(
    policy_class: str,
    tmp_path: Path,
) -> None:
    args = _args(
        output_dir=str(tmp_path / "artifacts"),
        spec_dir=str(tmp_path / "spec"),
        dry_run=True,
        issue="9bb676f",
        target_relative_multitarget=True,
        force_filter_feedback=True,
        initial_hidden_encoder=True,
        perturbation_training=True,
        perturbation_calibrated_timing=True,
        perturbation_physical_level="small",
        policy_adversary_training=True,
        policy_adversary_policy_class=policy_class,
        policy_adversary_mode=POLICY_ADVERSARY_ENERGY_MODE,
        policy_adversary_steps=7,
        policy_adversary_lr=1e-3,
        policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
        policy_adversary_radius_source="effective_020a65b_pgd_training_radius",
        broad_epsilon_reach_scaling=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    hps = build_hps(args)
    cfg = PolicyFullStateEpsilonTrainingConfig.from_payload(hps.policy_adversary_training)
    payload = write_run_spec(args)["run_spec"]
    policy = payload["hps"]["policy_adversary_training"]["policy"]
    optimizer = payload["hps"]["policy_adversary_training"]["inner_optimizer"]
    distribution = payload["training_distribution"]

    assert cfg.enabled is True
    assert cfg.policy_class == policy_class
    assert cfg.mode == POLICY_ADVERSARY_ENERGY_MODE
    assert hps.broad_epsilon_pgd_training.enabled is False
    assert hps.broad_epsilon_training.enabled is False
    assert payload["adversarial_phase"] == f"learned_finite_{policy_class}_policy_adversary"
    assert policy["kind"] == policy_class
    assert policy["parameterization"] == "shared_time_varying_finite_policy"
    assert (
        policy["evaluation_semantics"] == "static_epsilon_materialized_from_clean_rollout_pre_step"
    )
    assert policy["closed_loop_semantics_status"] == "not_live_rollout_hook"
    assert policy["has_bias"] is (policy_class == AFFINE_POLICY)
    assert optimizer["method"] == "adam"
    assert optimizer["n_ascent_steps_per_controller_step"] == 7
    assert optimizer["learning_rate"] == pytest.approx(1e-3)
    assert optimizer["weights_persist_across_batches"] is True
    assert distribution["training_axes"]["policy_adversary_training"] is True
    assert distribution["broad_epsilon_pgd_training"]["enabled"] is False
    assert distribution["policy_adversary_training"]["policy"]["kind"] == policy_class


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

    spec_path = spec_dir.with_suffix(".json")
    run_path = Path(result["run_spec_path"])
    graph_path = result["graph_spec_path"]
    manifest_path = Path(result["graph_manifest_path"])
    run_spec_text = run_path.read_text()
    payload = json.loads(run_spec_text)
    manifest = json.loads(manifest_path.read_text())

    assert run_path == spec_path
    assert len(run_spec_text.splitlines()) == 1
    assert "\n  " not in run_spec_text
    assert not (spec_dir / "run.json").exists()
    assert graph_path is None
    assert manifest_path == spec_dir / "model.graph.manifest.json"
    assert not (spec_dir / "model.graph.json").exists()
    assert payload["schema_version"] == "rlrmp.cs_stochastic_gru.v1"
    assert COMPACT_RUN_SPEC_KEY not in payload
    assert payload["issue"] == args.issue
    assert payload["model_summary"]["hidden_size"] == 4
    assert payload["model_summary"]["controller_kind"] == "gru"
    assert payload["model_summary"]["plant_backend"] == CS_LSS_PLANT_BACKEND
    assert payload["model_summary"]["exact_cs_linear_state_space"] is True
    assert payload["feedbax_graph"]["graph_spec_path"] is None
    assert payload["feedbax_graph"]["graph_export_status"] == "unavailable"
    assert payload["model_summary"]["training_distribution"]["mode"] == "nominal"
    assert payload["model_summary"]["training_distribution"]["fixed_target_only"] is True
    assert payload["training_summary"]["training_distribution"]["mode"] == "nominal"
    assert payload["training_summary"]["training_distribution"]["fixed_target_only"] is True
    assert "validation_bins" not in payload["training_summary"]
    assert payload["provenance_refs"]["training_summary.validation_bins"] == "$.validation_bins"
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


def test_feedbax_training_run_spec_rejects_cs_fields(tmp_path: Path) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "bulk"),
            spec_dir=str(tmp_path / "spec"),
            smoke=True,
            dry_run=True,
            gradient_clip_norm=5.0,
        )
    )
    feedbax_spec = result["run_spec"][FEEDBAX_TRAINING_RUN_SPEC_KEY]

    for field_name in (
        "game_card",
        "loss_objective",
        "training_mode",
        "CS_LSS_FEEDBACK_COMPONENT_TYPES",
    ):
        with pytest.raises(ValidationError):
            TrainingRunSpec.model_validate({**feedbax_spec, field_name: "not allowed"})


@pytest.mark.parametrize(
    ("variant", "overrides"),
    [
        ("nominal", {}),
        (
            "delayed_reach",
            {"delayed_reach": True, "target_relative_multitarget": True},
        ),
        ("perturbation_training", {"perturbation_training": True}),
        (
            "broad_epsilon",
            {"target_relative_multitarget": True, "broad_epsilon_training": True},
        ),
        ("target_relative", {"target_relative_multitarget": True}),
        (
            "target_relative_h0",
            {"target_relative_multitarget": True, "initial_hidden_encoder": True},
        ),
    ],
)
def test_cs_gru_hps_adapter_matches_expected_training_run_spec(
    tmp_path: Path,
    variant: str,
    overrides: dict[str, object],
) -> None:
    args = _args(
        output_dir=str(tmp_path / variant / "bulk"),
        spec_dir=str(tmp_path / variant / "spec"),
        smoke=True,
        dry_run=True,
        gradient_clip_norm=5.0,
        **overrides,
    )
    result = write_run_spec(args)
    payload = result["run_spec"]
    actual = TrainingRunSpec.model_validate(payload[FEEDBAX_TRAINING_RUN_SPEC_KEY])
    hps = build_hps(args)
    expected = build_feedbax_training_run_spec(
        payload,
        graph_spec=build_training_run_graph_spec(hps, seed=int(args.seed)),
        output_dir=Path(args.output_dir),
        spec_dir=Path(args.spec_dir),
    )

    assert actual == expected


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


def test_mixed_calibration_regime_loads_closed_loop_table_and_manifest() -> None:
    table_path = "results/c92ebd8/notes/closed_loop_calibration_table.json"
    open_cfg = FixedTargetPerturbationTrainingConfig(
        enabled=True,
        calibrated_timing=True,
        physical_level="moderate",
        calibration_regime=OPEN_LOOP_ALL_CALIBRATION_REGIME,
    )
    sensory_cfg = FixedTargetPerturbationTrainingConfig(
        enabled=True,
        calibrated_timing=True,
        physical_level="moderate",
        calibration_regime=CLOSED_LOOP_SENSORY_CALIBRATION_REGIME,
        closed_loop_calibration_table_path=table_path,
    )
    full_cfg = FixedTargetPerturbationTrainingConfig(
        enabled=True,
        calibrated_timing=True,
        physical_level="moderate",
        calibration_regime=CLOSED_LOOP_SENSORY_COMMAND_LATERAL_CALIBRATION_REGIME,
        closed_loop_calibration_table_path=table_path,
    )

    sensory_manifest = calibration_regime_manifest(sensory_cfg)
    full_manifest = calibration_regime_manifest(full_cfg)
    command_amplitudes = _closed_loop_amplitudes_by_timing(
        full_cfg,
        family="command_input_pulse",
        timing_labels=("early", "mid", "late"),
        component="random_force_pulse_cardinal_basis",
        reducer="mean",
    )
    lateral_amplitudes = _closed_loop_amplitudes_by_timing(
        full_cfg,
        family="target_aligned_lateral_command_load_pulse",
        timing_labels=("early", "mid", "late"),
        component="target_aligned_lateral_load",
        axis="y",
    )

    assert "target_aligned_lateral_load" not in _active_single_family_bins(open_cfg)
    assert "target_aligned_lateral_load" not in _active_single_family_bins(sensory_cfg)
    assert "target_aligned_lateral_load" in _active_single_family_bins(full_cfg)
    assert sensory_manifest["closed_loop_families"] == ["sensory_feedback_offset"]
    assert full_manifest["closed_loop_families"] == [
        "sensory_feedback_offset",
        "command_input_pulse",
        "target_aligned_lateral_command_load_pulse",
    ]
    assert np.all(np.asarray(command_amplitudes) > 0.0)
    assert np.all(np.asarray(lateral_amplitudes) > 0.0)


def test_mixed_calibration_regime_requires_closed_loop_table() -> None:
    with pytest.raises(ValueError, match="closed_loop_calibration_table_path"):
        FixedTargetPerturbationTrainingConfig(
            enabled=True,
            calibrated_timing=True,
            calibration_regime=CLOSED_LOOP_SENSORY_CALIBRATION_REGIME,
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
    training = payload["training_distribution"]
    hps_config = payload["hps"]["perturbation_training"]
    timing = hps_config["timing_bins"]["family_timing_bins"]

    assert payload["training_summary"]["training_mode"] == ScienceMode.PERTURBATION
    assert training["mode"] == ScienceMode.PERTURBATION_CALIBRATED
    assert training["mixture"]["calibrated_timing"] is True
    assert training["mixture"]["physical_level"] == "moderate"
    assert hps_config["physical_level_fraction_of_reach"] == 0.10
    assert hps_config["training_physical_levels"] == ["small", "moderate"]
    assert hps_config["eval_only_physical_levels"] == ["stress"]
    assert timing["process_epsilon"]["start_time_indices"] == [5, 15, 35]
    assert timing["command_input"]["start_time_indices"] == [5, 15, 35]
    assert timing["sensory_feedback"]["start_time_indices"] == [10, 20, 40]
    assert timing["initial_position"]["start_time_indices"] == [0]
    assert "delayed_observation" not in timing
    assert "delayed_observation" not in hps_config["validation_bins"]
    assert "delayed_observation" not in hps_config["families"]
    assert hps_config["inactive_legacy_bins"]["bins"] == ["delayed_observation"]
    assert (
        hps_config["mixture_semantics"]["calibrated_levels"]["amplitude_wiring_status"]
        == "wired_in_sampler_when_calibrated_timing_true"
    )
    # Fail-closed identity: calibration IS consumed at runtime in calibrated-timing
    # mode, so the emitted policy must carry a data-product identity block with a
    # non-null hash and must not claim "none_at_runtime" (issue ea6ccb4).
    calibrated_policy = hps_config["calibrated_amplitude_policy"]
    data_product = calibrated_policy["data_product"]
    assert data_product["role"] == "perturbation_open_loop_calibration"
    assert data_product["product_schema_version"] == "rlrmp.perturbation_open_loop_calibration.v2"
    assert data_product["product_identity_hash"]
    assert calibrated_policy["artifact_dependency"] != "none_at_runtime"
    assert calibrated_policy["artifact_dependency"] == data_product["product_path"]
    consumed = payload["consumed_data_identities"]
    calibration_ids = [
        entry for entry in consumed if entry["role"] == "perturbation_open_loop_calibration"
    ]
    assert len(calibration_ids) == 1
    assert calibration_ids[0]["hash"] == data_product["product_identity_hash"]
    assert calibration_ids[0]["schema"] == "rlrmp.perturbation_open_loop_calibration.v2"


def test_movement_age_timing_run_spec_distinguishes_timing_basis(tmp_path: Path) -> None:
    absolute_result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "absolute_bulk"),
            spec_dir=str(tmp_path / "absolute_spec"),
            smoke=True,
            issue="020a65b",
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_physical_level="small",
        )
    )
    movement_result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "movement_bulk"),
            spec_dir=str(tmp_path / "movement_spec"),
            smoke=True,
            issue="6c36536",
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_movement_age_timing=True,
            perturbation_physical_level="small",
            target_relative_multitarget=True,
            delayed_reach=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        )
    )
    absolute = json.loads(Path(absolute_result["run_spec_path"]).read_text())
    movement = json.loads(Path(movement_result["run_spec_path"]).read_text())

    absolute_perturbation = absolute["hps"]["perturbation_training"]
    movement_perturbation = movement["hps"]["perturbation_training"]
    assert absolute_perturbation["movement_age_timing"] is False
    assert absolute_perturbation["timing_basis"]["mode"] == "absolute_trial_time"
    assert absolute_perturbation["timing_bins"]["start_time_indices_are"] == (
        "absolute_trial_indices"
    )
    assert movement_perturbation["movement_age_timing"] is True
    assert movement_perturbation["timing_basis"]["mode"] == "movement_age"
    assert movement_perturbation["timing_basis"]["epoch_source"] == (
        "trial_specs.timeline.epoch_bounds[-2]"
    )
    assert movement_perturbation["timing_bins"]["start_time_indices_are"] == (
        "movement_start_relative_offsets"
    )
    assert movement["training_distribution"]["perturbation_training"]["movement_age_timing"] is True

    replay_args = build_execution_context_from_spec(movement_result["run_spec_path"]).args
    assert replay_args.perturbation_movement_age_timing is True
    assert replay_args.output_dir == str(tmp_path / "movement_bulk")
    assert replay_args.spec_dir == str(tmp_path / "movement_spec")


def test_delayed_reach_task_hold_preserves_catch_target_without_extra_metadata() -> None:
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
    task_base = _delayed_cs_task(hps, target_relative=False, go_cue_input=False)
    base = task_base.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    base = eqx.tree_at(lambda trial: trial.extra, base, None)

    sampled = apply_training_target_distribution(
        base,
        hps.target_relative_multitarget,
        jr.PRNGKey(2),
    )

    assert sampled.extra is None
    assert jnp.all(sampled.inputs["task"].hold > 0.5)
    assert jnp.any(jnp.abs(sampled.inputs["target"]) > 0.0)
    assert jnp.allclose(
        sampled.targets["mechanics.effector.pos"].value,
        jnp.zeros_like(sampled.targets["mechanics.effector.pos"].value),
    )


def test_delayed_reach_pre_go_hold_penalty_args_build_hps_and_run_spec(
    tmp_path: Path,
) -> None:
    args = _args(
        dry_run=True,
        smoke=True,
        output_dir=str(tmp_path / "bulk"),
        spec_dir=str(tmp_path / "spec"),
        delayed_reach=True,
        target_relative_multitarget=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        nn_output_pre_go=0.0,
        delayed_pre_go_force_filter_hold=11.0,
        delayed_pre_go_start_pos_hold=22.0,
        delayed_pre_go_start_pos_hold_norm="l1",
        delayed_pre_go_zero_vel_hold=33.0,
    )
    hps = build_hps(args)
    payload = write_run_spec(args)["run_spec"]
    aux = payload["loss_summary"]["delayed_pre_go_auxiliary_terms"]

    assert hps.loss.weights.nn_output_pre_go == pytest.approx(0.0)
    assert hps.loss.weights.delayed_pre_go_force_filter_hold == pytest.approx(11.0)
    assert hps.loss.weights.delayed_pre_go_start_pos_hold == pytest.approx(22.0)
    assert hps.loss.weights.delayed_pre_go_zero_vel_hold == pytest.approx(33.0)
    assert hps.loss.delayed_pre_go_start_pos_hold_norm == "l1"
    assert payload["hps"]["loss"]["weights"]["delayed_pre_go_force_filter_hold"] == 11.0
    assert payload["hps"]["loss"]["weights"]["delayed_pre_go_start_pos_hold"] == 22.0
    assert payload["hps"]["loss"]["delayed_pre_go_start_pos_hold_norm"] == "l1"
    assert payload["hps"]["loss"]["weights"]["delayed_pre_go_zero_vel_hold"] == 33.0
    assert aux["scope"] == "prep_epoch_only"
    assert aux["movement_window_qrf_comparator"] == "unchanged"
    assert aux["terms"]["delayed_pre_go_start_pos_hold"]["norm"] == "l1"
    assert set(aux["active_terms"]) == {
        "delayed_pre_go_force_filter_hold",
        "delayed_pre_go_start_pos_hold",
        "delayed_pre_go_zero_vel_hold",
    }


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
    assert payload["delayed_reach"]["mode"] == ScienceMode.DELAYED_REACH
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
    assert payload["model_summary"]["controller_input_dimension"] == 7
    assert payload["hps"]["loss"]["weights"]["nn_output_pre_go"] == 1.0
    assert payload["hps"]["broad_epsilon_pgd_training"]["movement_epoch_only"] is True
    assert payload["hps"]["broad_epsilon_pgd_training"]["time_mask"]["mode"] == (
        "movement_epoch_only"
    )
    assert payload["loss_summary"]["time_indexing"]["stage_schedule"] == (
        "movement_age_from_go_cue"
    )
    assert ScienceMode.DELAYED_REACH in payload["training_summary"]["training_mode"]


def test_delayed_reach_trial_type_normalized_run_spec_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
        dry_run=True,
        issue="c7c27dd",
        delayed_reach=True,
        delayed_reach_trial_type_normalized_loss=True,
        delayed_reach_no_catch_qrf_weight=2.0,
        delayed_reach_catch_qrf_weight=3.0,
        target_relative_multitarget=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    result = write_run_spec(args)
    payload = result["run_spec"]

    normalization = payload["hps"]["loss"]["delayed_trial_type_normalization"]
    assert normalization["enabled"] is True
    assert normalization["no_catch_weight"] == pytest.approx(2.0)
    assert normalization["catch_weight"] == pytest.approx(3.0)
    assert "Feedbax grouped reductions" in normalization["semantics"]
    assert payload["loss_summary"]["delayed_trial_type_normalization"] == normalization
    assert payload["loss_summary"]["grouped_reduction_implementation"] == (
        "rlrmp_bridge_pending_feedbax_69d8d76"
    )


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

    assert ScienceMode.BROAD_EPSILON in payload["training_summary"]["training_mode"]
    assert broad["enabled"] is True
    assert broad["budget_contract"]["gamma_factor"] == pytest.approx(1.4)
    assert broad["budget_contract"]["effective_l2_radius_15cm"] == pytest.approx(
        0.0012324305441740995
    )
    assert (
        payload["training_distribution"]["training_axes"]["broad_full_state_epsilon_training"]
        is True
    )


def test_modern_run_spec_replays_to_current_training_args(tmp_path: Path) -> None:
    spec_dir = tmp_path / "historical_spec"
    output_dir = tmp_path / "historical_artifacts"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="020a65b",
        full_train=True,
        target_relative_multitarget=True,
        perturbation_training=True,
        perturbation_calibrated_timing=True,
        perturbation_physical_level="small",
        force_filter_feedback=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        batch_size=64,
        controller_lr=3e-3,
        gradient_clip_norm=5.0,
        lr_warmup_batches=1000,
        lr_cosine_alpha=0.1,
    )
    result = write_run_spec(args)

    replay_args = build_execution_context_from_spec(
        result["run_spec_path"], stop_after_batches=1000
    ).args

    assert replay_args.issue == "020a65b"
    assert replay_args.output_dir == str(output_dir)
    assert replay_args.spec_dir == str(spec_dir)
    assert replay_args.full_train is True
    assert replay_args.stop_after_batches == 1000
    assert replay_args.n_train_batches == 12000
    assert replay_args.batch_size == 64
    assert replay_args.controller_lr == pytest.approx(3e-3)
    assert replay_args.gradient_clip_norm == pytest.approx(5.0)
    assert replay_args.lr_warmup_batches == 1000
    assert replay_args.lr_cosine_alpha == pytest.approx(0.1)
    assert replay_args.loss_objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
    assert replay_args.target_relative_multitarget is True
    assert replay_args.perturbation_training is True
    assert replay_args.perturbation_calibrated_timing is True
    assert replay_args.perturbation_physical_level == "small"
    assert replay_args.force_filter_feedback is True
    assert replay_args.broad_epsilon_training is False
    assert replay_args.broad_epsilon_pgd_training is False


def test_flat_run_spec_replay_does_not_require_adjacent_graph_manifest(
    tmp_path: Path,
) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "historical_artifacts"),
            spec_dir=str(tmp_path / "historical_spec"),
            issue="6c36536",
            full_train=True,
            target_relative_multitarget=True,
            delayed_reach=True,
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_movement_age_timing=True,
            perturbation_physical_level="small",
            force_filter_feedback=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        )
    )
    flat_spec_dir = tmp_path / "flat_specs"
    flat_spec_dir.mkdir()
    flat_run_spec = flat_spec_dir / "delayed_movement_bank.json"
    flat_run_spec.write_text(Path(result["run_spec_path"]).read_text(), encoding="utf-8")

    replay_args = build_execution_context_from_spec(flat_run_spec).args

    assert replay_args.output_dir == str(tmp_path / "historical_artifacts")
    assert replay_args.spec_dir == str(tmp_path / "historical_spec")


def test_full_train_run_spec_replay_dry_run_stays_on_spec_path(
    tmp_path: Path,
) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "historical_artifacts"),
            spec_dir=str(tmp_path / "historical_spec"),
            issue="246182c",
            smoke=True,
            full_train=True,
            target_relative_multitarget=True,
            delayed_reach=True,
            force_filter_feedback=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            delayed_movement_cost_tail_mode=DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
        )
    )
    flat_run_spec = tmp_path / "flat_full_train.json"
    flat_run_spec.write_text(Path(result["run_spec_path"]).read_text(), encoding="utf-8")

    context = build_execution_context_from_spec(flat_run_spec, dry_run=True)
    payload = render_run_spec_execution_dry_run(context)

    assert "would_write" in payload
    assert payload["would_execute"]["entrypoint"].endswith("_run_full_training_from_context")
    assert payload["run_spec"]["full_training_launch"] == "requested"
    assert payload["run_spec"]["hps"]["loss"]["delayed_movement_cost_tail_mode"] == (
        DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON
    )


def test_run_spec_dry_run_renderer_does_not_write(tmp_path: Path) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "artifacts"),
            spec_dir=str(tmp_path / "spec"),
            smoke=True,
            full_train=True,
        )
    )
    context = build_execution_context_from_spec(result["run_spec_path"], dry_run=True)

    rendered = render_run_spec_execution_dry_run(context)

    assert rendered["validated"] is True
    assert rendered["would_write"] == []
    assert rendered["would_execute"]["output_dir"] == str(tmp_path / "artifacts")
    assert rendered["run_spec"]["mode"] == "full_train"


def test_initial_hidden_encoder_requires_target_relative_hps() -> None:
    with pytest.raises(ValueError, match="requires --target-relative-multitarget"):
        build_hps(_args(initial_hidden_encoder=True))


def test_33b0dcb_target_support_profiles() -> None:
    default = target_relative_target_support_config(
        enabled=True,
        force_filter_feedback=True,
    )
    old = target_relative_target_support_config(
        profile=TARGET_SUPPORT_PROFILE_020A65B,
        enabled=True,
        force_filter_feedback=True,
    )
    dense = target_relative_target_support_config(
        profile=TARGET_SUPPORT_PROFILE_CONST_DENSE_ALL,
        enabled=True,
        force_filter_feedback=True,
    )
    sparse = target_relative_target_support_config(
        profile=TARGET_SUPPORT_PROFILE_CONST_SPARSE8,
        enabled=True,
        force_filter_feedback=True,
    )
    band8 = target_relative_target_support_config(
        profile=TARGET_SUPPORT_PROFILE_CONST_BAND8,
        enabled=True,
        force_filter_feedback=True,
    )
    band16 = target_relative_target_support_config(
        profile=TARGET_SUPPORT_PROFILE_CONST_BAND16,
        enabled=True,
        force_filter_feedback=True,
    )
    band36 = target_relative_target_support_config(
        profile=TARGET_SUPPORT_PROFILE_CONST_BAND36,
        enabled=True,
        force_filter_feedback=True,
    )

    assert default.target_support_profile == TARGET_SUPPORT_PROFILE_CONST_BAND16
    assert len(default.seen_targets_m) == 56
    assert len(default.held_out_targets_m) == 16
    assert old.target_support_profile == TARGET_SUPPORT_PROFILE_020A65B
    assert len(old.seen_targets_m) == 12
    assert len(old.held_out_targets_m) == 8
    assert len(dense.seen_targets_m) == TARGET_SUPPORT_DENSE_N_DIRECTIONS
    assert len(dense.held_out_targets_m) == 0
    assert len(dense.validation_targets_m) == TARGET_SUPPORT_DENSE_N_DIRECTIONS
    assert len(sparse.seen_targets_m) == TARGET_SUPPORT_SPARSE_N_DIRECTIONS
    assert len(sparse.held_out_targets_m) == (
        TARGET_SUPPORT_DENSE_N_DIRECTIONS - TARGET_SUPPORT_SPARSE_N_DIRECTIONS
    )
    assert len(band8.seen_targets_m) == 64
    assert len(band8.held_out_targets_m) == 8
    assert len(band16.seen_targets_m) == 56
    assert len(band16.held_out_targets_m) == 16
    assert len(band36.seen_targets_m) == 36
    assert len(band36.held_out_targets_m) == 36
    assert len(band8.seen_targets_m) + len(band8.held_out_targets_m) == (
        TARGET_SUPPORT_DENSE_N_DIRECTIONS
    )
    assert len(band16.seen_targets_m) + len(band16.held_out_targets_m) == (
        TARGET_SUPPORT_DENSE_N_DIRECTIONS
    )
    assert len(band36.seen_targets_m) + len(band36.held_out_targets_m) == (
        TARGET_SUPPORT_DENSE_N_DIRECTIONS
    )
    assert len(band36.held_out_targets_m) > len(band16.held_out_targets_m)
    assert len(band16.held_out_targets_m) > len(band8.held_out_targets_m)

    for config in (default, dense, sparse, band8, band16, band36):
        all_targets = [*config.seen_targets_m, *config.held_out_targets_m]
        radii = np.linalg.norm(np.asarray(all_targets, dtype=np.float64), axis=1)
        assert np.allclose(radii, TARGET_SUPPORT_CONST_REACH_M)
        assert not set(config.seen_targets_m).intersection(set(config.held_out_targets_m))


def test_target_hps_without_profile_uses_governed_class_preset() -> None:
    config = TargetRelativeMultiTargetTrainingConfig.from_payload(
        {"config": {"enabled": True, "force_filter_feedback": True}}
    )

    assert config.target_support_profile == TARGET_SUPPORT_PROFILE_020A65B
    assert config.seen_amplitudes_m == (0.10, 0.15)
    assert config.held_out_amplitudes_m == (0.12, 0.18)


def test_target_hps_rejects_retired_nested_distribution_alias() -> None:
    with pytest.raises(ValueError, match="target_distribution"):
        TargetRelativeMultiTargetTrainingConfig.from_payload(
            TreeNamespace(
                enabled=True,
                force_filter_feedback=True,
                target_distribution=TreeNamespace(),
            )
        )
