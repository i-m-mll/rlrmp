"""Tests for stochastic C&S-fidelity GRU run-spec preparation."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax
import pytest
from feedbax import TaskTrialSpec, TrialTimeline, WhereDict
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY, TrainingRunSpec
from feedbax.objectives.loss import AbstractLoss, TargetSpec
from feedbax.mechanics import LinearStateSpace
from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.state_feedback import StateFeedbackSelector
from feedbax.config.namespace import TreeNamespace
from pydantic import ValidationError

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    build_canonical_game,
)
from rlrmp.analysis.math.cs_released_simulation import default_cs_noise_covariances
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
from rlrmp.data_products.calibration import load_open_loop_calibration
from rlrmp.model.cs_lss_gru import (
    FINITE_EPSILON_POLICY_GRAPH_COMPONENT,
    FINITE_EPSILON_POLICY_NODE_LABEL,
    CS_EPSILON_DIM,
    CS_REDUCED_EPSILON_DIM,
    CsLssFiniteEpsilonPolicy,
    build_cs_lss_gru_graph_spec,
)
from rlrmp.model.trainable import staged_network_trainable_parts
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    CsAnalyticalQrfLoss,
    DelayedReachTrialTypeNormalizedLoss,
    get_reach_loss,
)
from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir
import rlrmp.train.cs_nominal_gru as cs_nominal_gru
import rlrmp.train.cs_perturbation_training as cs_perturbation_training
import rlrmp.train.executor.cs_supervised as cs_supervised_executor
from rlrmp.train.cs_nominal_gru import (
    ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
    ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
    AdaptiveEpsilonState,
    CS_DELAYED_REACH_TASK_TYPE,
    CsNominalGruConfig,
    DEFAULT_DELAYED_P_CATCH_TRIAL,
    DEFAULT_STOCHASTIC_PRESET,
    DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
    DELAYED_REACH_TRAINING_MODE,
    GradientDiagnosticsState,
    SCHEMA_VERSION,
    TrainingState,
    UpdateDiagnosticsState,
    build_graph_bundle,
    build_training_run_graph_spec,
    build_hps,
    build_parser,
    build_run_spec_execution_context,
    cs_nominal_gru_config_from_args,
    derive_spec_dir,
    derive_spec_path,
    _adaptive_epsilon_damage_target,
    _adaptive_epsilon_outer_weight,
    _adaptive_epsilon_schedule_batch,
    _adaptive_epsilon_zero_guard_from_state,
    _initial_adaptive_epsilon_zero_guard,
    _update_adaptive_epsilon_zero_guard,
    render_run_spec_execution_dry_run,
    _emit_checkpoint_progress,
    _initial_adaptive_epsilon_state,
    _prepend_existing_training_diagnostics,
    _resize_optimizer_diagnostics_for_batches,
    _sample_adaptive_epsilon_damage_eval_batch,
    _sample_adaptive_epsilon_training_batch,
    _scale_direct_epsilon_trial_specs,
    _update_adaptive_epsilon_state,
    load_latest_checkpoint,
    resolve_run_spec_args,
    run_full_training,
    save_training_checkpoint,
    write_run_spec,
)
from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    RLRMP_RUN_SPEC_PAYLOAD_KEY,
    assert_runtime_graph_matches_training_spec,
    build_feedbax_training_run_spec,
    feedbax_training_run_spec_from_payload,
    hydrate_compact_run_spec_envelope,
)
from rlrmp.runtime.run_specs import validate_nominal_gru_run_spec_file
from rlrmp.train.executor.slots import (
    ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    CS_SUPERVISED_METHOD_REF,
    POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
)
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_ADAM,
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
    BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    BROAD_EPSILON_PGD_TRAINING_MODE,
    BROAD_EPSILON_TRAINING_MODE,
    CALIBRATED_TIMING_PERTURBATION_TRAINING_MODE,
    CLOSED_LOOP_SENSORY_CALIBRATION_REGIME,
    CLOSED_LOOP_SENSORY_COMMAND_LATERAL_CALIBRATION_REGIME,
    DEFAULT_PGD_SISU_EXACT_ZERO_MASS,
    DEFAULT_PGD_SISU_LEVELS,
    DEFAULT_TARGET_SUPPORT_PROFILE,
    HISTORICAL_020A65B_PGD_RADIUS_15CM,
    GRAPH_ADAPTER_SPECS,
    AFFINE_POLICY,
    LINEAR_NO_BIAS_POLICY,
    MILD_COMBINED_FAMILIES,
    PERTURBATION_TRAINING_MODE,
    POLICY_ADVERSARY_MEMORYLESS_MLP,
    POLICY_ADVERSARY_ENERGY_MODE,
    POLICY_ADVERSARY_PLAIN_MODE,
    POLICY_ADVERSARY_TRAINING_MODE,
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
    TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
    VALIDATION_BINS,
    BroadFullStateEpsilonTrainingConfig,
    BroadFullStateEpsilonTrainingTaskAdapter,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    TargetRelativeMultiTargetTrainingConfig,
    TargetRelativeMultiTargetTrainingTaskAdapter,
    FixedTargetPerturbationTrainingConfig,
    _broad_epsilon_l2_radius,
    _active_single_family_bins,
    _closed_loop_amplitudes_by_timing,
    _command_input_direction_pulse,
    _ensure_broad_epsilon_input,
    _epsilon_time_mask,
    _expand_bool_like,
    _expand_radius,
    _flattened_per_trial_norm,
    _normalize_flattened_per_trial,
    _project_flattened_per_trial_l2_ball,
    _run_finite_broad_epsilon_pgd_inner_maximizer,
    _set_input,
    _target_aligned_lateral_direction_pulse,
    calibration_regime_manifest,
    apply_broad_epsilon_training,
    apply_training_perturbation_mixture,
    apply_training_target_distribution,
    apply_validation_bin,
    apply_validation_target_distribution,
    graph_adapter_specs,
    make_broad_epsilon_pgd_pre_step,
    make_memoryless_policy_adversary,
    policy_adversary_trial_specs,
    policy_adversary_projection_diagnostics,
    run_broad_epsilon_pgd_inner_maximizer,
    target_relative_target_support_config,
    target_relative_validation_manifest,
    validation_bin_manifest,
)
from rlrmp.train.executor.equivalence import assert_paired_equivalent, run_paired_equivalence
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
from rlrmp.train.closed_loop_finite_adversary import (
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
)


def _args(**overrides) -> argparse.Namespace:
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _remove_training_method_registration(
    monkeypatch: pytest.MonkeyPatch,
    method_ref: str,
) -> None:
    registrations = dict(DEFAULT_TRAINING_METHOD_REGISTRY._registrations)
    registrations.pop(method_ref, None)
    monkeypatch.setattr(DEFAULT_TRAINING_METHOD_REGISTRY, "_registrations", registrations)
    assert method_ref not in DEFAULT_TRAINING_METHOD_REGISTRY.available_keys()


def _absolute_string_leaves(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)] if Path(value).is_absolute() else []
    if isinstance(value, dict):
        leaves: list[tuple[str, str]] = []
        for key, child in value.items():
            leaves.extend(_absolute_string_leaves(child, path=f"{path}.{key}"))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, child in enumerate(value):
            leaves.extend(_absolute_string_leaves(child, path=f"{path}[{index}]"))
        return leaves
    return []


def _assert_no_absolute_string_leaves(value: Any) -> None:
    absolute = _absolute_string_leaves(value)
    assert absolute == []


def _cs_nominal_gru_golden_fixture() -> dict:
    return _normalize_golden_fixture_paths(
        json.loads(
            Path("tests/fixtures/cs_nominal_gru_config_golden.json").read_text(encoding="utf-8")
        )
    )


def _normalize_golden_fixture_paths(value, *, key: str | None = None):
    if key == "rlrmp_branch":
        return "<current-branch>"
    if isinstance(value, dict):
        return {
            child_key: _normalize_golden_fixture_paths(item, key=child_key)
            for child_key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_golden_fixture_paths(item) for item in value]
    if isinstance(value, str):
        marker = "/worktrees/"
        if marker in value:
            prefix, suffix = value.split(marker, 1)
            worktree_parts = suffix.split("/", 1)
            if len(worktree_parts) == 2 and prefix.endswith("/rlrmp"):
                return str(REPO_ROOT / worktree_parts[1])
    return value


def _stable_golden_run_spec_payload(payload: dict) -> dict:
    fixture = _cs_nominal_gru_golden_fixture()
    stable = {key: payload[key] for key in fixture["stable_run_spec_keys"]}
    canonical = json.loads(
        json.dumps(
            stable,
            sort_keys=True,
        )
    )
    canonical["rlrmp_run_spec"]["provenance"]["git"]["rlrmp_commit"] = "<current-commit>"
    canonical["rlrmp_run_spec"]["provenance"]["git"]["rlrmp_branch"] = "<current-branch>"
    return _normalize_stochastic_float_precision(canonical)


def _normalize_stochastic_float_precision(value, *, key: str | None = None):
    precision_by_key = {
        "diag_first_block": 6,
        "initial_diag_first_block": 6,
        "noise_std": 7,
        "sensory_noise_std": 7,
        "sensory_covariance_diag": 7,
    }
    if isinstance(value, dict):
        return {
            child_key: _normalize_stochastic_float_precision(item, key=child_key)
            for child_key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_stochastic_float_precision(item, key=key) for item in value]
    if isinstance(value, float) and key in precision_by_key:
        return float(f"{value:.{precision_by_key[key]}g}")
    return value


def _cs_stochastic_gru_run_spec_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted(Path("results").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            continue
        if {"hps", "feedbax_graph", "training_script"}.issubset(payload):
            paths.append(path)
    return paths


class _ScalarLoss:
    def __init__(
        self, value: np.ndarray | None = None, children: dict[str, "_ScalarLoss"] | None = None
    ):
        self.value = value
        self.weight = 1.0
        self._children = children or {}
        self.children = tuple(self._children.values())

    def flatten(self) -> dict[str, np.ndarray]:
        if self.value is not None:
            return {"self": self.value}
        return {
            name: child.value for name, child in self._children.items() if child.value is not None
        }


def test_progress_defaults_enabled_unless_disabled() -> None:
    parser = build_parser()

    assert parser.parse_args([]).disable_progress is False
    assert parser.parse_args(["--disable-progress"]).disable_progress is True


def test_checkpoint_progress_includes_loss_terms_and_pgd_penalty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    history = argparse.Namespace(
        loss=_ScalarLoss(
            children={
                "control": _ScalarLoss(np.array([1.0, 2.0], dtype=np.float32)),
                "effector_pos_running": _ScalarLoss(np.array([3.0, 4.0], dtype=np.float32)),
            }
        )
    )
    pgd_diagnostics = {
        "pgd_broad_epsilon_energy_penalty_term_selected": np.array([np.nan, 0.25]),
        "pgd_broad_epsilon_penalized_objective_selected": np.array([np.nan, 1.75]),
        "pgd_broad_epsilon_epsilon_energy_mean": np.array([np.nan, 0.5]),
    }

    _emit_checkpoint_progress(
        history,
        pgd_diagnostics,
        chunk_batches=2,
        completed_batches=2,
        total_batches=1000,
        elapsed_seconds=12.3,
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith("BATCH phase=checkpoint batch=1/1000")
    assert "loss=6" in line
    assert "loss_control=2" in line
    assert "loss_effector_pos_running=4" in line
    assert "adv_penalty=0.25" in line
    assert "adv_energy=0.5" in line
    assert "adv_objective=1.75" in line


def test_delayed_reach_resolves_force_filter_and_perturbation_defaults() -> None:
    hps = build_hps(
        _args(
            delayed_reach=True,
            target_relative_multitarget=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        )
    )

    assert hps.model.force_filter_feedback is True
    assert hps.target_relative_multitarget.force_filter_feedback is True
    assert hps.perturbation_training.enabled is True
    assert hps.perturbation_training.calibrated_timing is True
    assert hps.perturbation_training.timing_basis.mode == "movement_age"
    assert hps.perturbation_training.physical_level == "small"


def test_non_delayed_rows_keep_force_filter_and_perturbation_defaults_off() -> None:
    hps = build_hps(_args(target_relative_multitarget=True))

    assert hps.model.force_filter_feedback is False
    assert hps.target_relative_multitarget.force_filter_feedback is False
    assert hps.perturbation_training.enabled is False
    assert hps.perturbation_training.calibrated_timing is False
    assert hps.perturbation_training.timing_basis.mode == "absolute_trial_time"
    assert hps.perturbation_training.physical_level == "moderate"


def test_target_support_cli_default_is_band16_fixed_reach() -> None:
    args = build_parser().parse_args([])

    assert DEFAULT_TARGET_SUPPORT_PROFILE == TARGET_SUPPORT_PROFILE_CONST_BAND16
    assert args.target_support_profile == TARGET_SUPPORT_PROFILE_CONST_BAND16


def test_cs_nominal_gru_config_defaults_match_pre_refactor_fixture() -> None:
    fixture = _cs_nominal_gru_golden_fixture()
    expected = dict(fixture["cases"]["default"]["parsed_args"])
    expected["dry_run"] = False

    assert CsNominalGruConfig().model_dump(mode="python") == expected
    assert vars(build_parser().parse_args([])) == expected


def test_stochastic_preset_metadata_is_independent_of_jax_x64() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    code = """
import json
import sys

import jax

jax.config.update("jax_enable_x64", sys.argv[1] == "true")
from rlrmp.train.cs_nominal_gru import stochastic_preset

print(json.dumps(stochastic_preset("cs2019-rollout").summary(), sort_keys=True))
"""
    summaries = []
    for flag in ("false", "true"):
        completed = subprocess.run(
            [sys.executable, "-c", code, flag],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        summaries.append(json.loads(completed.stdout))

    assert summaries[0] == summaries[1]


def test_cs_nominal_gru_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CsNominalGruConfig.model_validate({"seed": 42, "unknown_field": True})


def test_build_hps_accepts_legacy_namespace_extras_and_validates_canonical_fields() -> None:
    args = _args(batch_size=3)
    args.legacy_payload_marker = {"source": "historical-run-spec"}

    hps = build_hps(args)

    assert hps.batch_size == 3
    with pytest.raises(ValidationError):
        build_hps(_args(batch_size=0))


def test_cs_nominal_gru_argparse_defaults_and_choices_derive_from_config() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    assert parser.parse_args(["--seed", "7"]).seed == 7
    assert parser.parse_args(
        ["--target-support-profile", "const_sparse8"]
    ).target_support_profile == (TARGET_SUPPORT_PROFILE_CONST_SPARSE8)
    assert parser.parse_args(
        ["--broad-epsilon-pgd-inner-optimizer-method", "adam"]
    ).broad_epsilon_pgd_inner_optimizer_method == (BROAD_EPSILON_PGD_ADAM)
    assert "--seed SEED" in help_text
    assert "(default: 42)" in help_text
    assert "{old_020a65b,const_dense_all,const_sparse8,const_band8,const_band16,const_band36}" in (
        help_text
    )
    assert "{projected_gradient_ascent,adam}" in help_text


def test_cs_nominal_gru_pre_refactor_golden_payloads_stay_stable() -> None:
    fixture = _cs_nominal_gru_golden_fixture()
    parser = build_parser()

    for case in fixture["cases"].values():
        args = parser.parse_args(case["argv"])
        with jax.enable_x64(False):
            config = cs_nominal_gru_config_from_args(args)
            payload = write_run_spec(args)["run_spec"]

        assert config.model_dump(mode="python") == case["parsed_args"]
        assert _stable_golden_run_spec_payload(payload) == _normalize_stochastic_float_precision(
            case["stable_run_spec_payload"]
        )


def test_cs_nominal_gru_config_validates_tracked_cs_stochastic_gru_corpus() -> None:
    paths = _cs_stochastic_gru_run_spec_paths()
    clean_paths = []
    fail_closed: set[Path] = set()

    assert len(paths) == 149
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            CsNominalGruConfig.model_validate(cs_nominal_gru._args_values_from_run_spec(payload))
        except ValidationError:
            fail_closed.add(path)
        else:
            clean_paths.append(path)

    assert len(clean_paths) == 149
    assert fail_closed == set()


def _where_train() -> dict[int, object]:
    def where_train_fn(model):
        net = model.nodes["net"]
        return staged_network_trainable_parts(net)

    return {0: where_train_fn}


def _native_recurrent_input_size(model) -> int:
    return int(model.nodes["net"].nodes["cell"].input_size)


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


class _StaticPerTrialLoss(AbstractLoss):
    label: str
    values: jax.Array

    def __init__(self, values) -> None:
        self.label = "static_per_trial"
        self.values = jnp.asarray(values)

    def term(self, states, trial_specs, model):
        del states, trial_specs, model
        return self.values


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

    cfg = PgdFullStateEpsilonTrainingConfig.from_payload(hps.broad_epsilon_pgd_training)

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

    parsed_nested = PgdFullStateEpsilonTrainingConfig.from_payload(nested)
    parsed_legacy = PgdFullStateEpsilonTrainingConfig.from_payload(legacy)
    parsed_dict = PgdFullStateEpsilonTrainingConfig.from_payload(nested_dict)

    assert parsed_nested.level == "strong"
    assert parsed_nested.n_steps == 9
    assert parsed_nested.step_size_fraction == pytest.approx(0.125)
    assert parsed_nested.budget_scale == pytest.approx(1.5)
    assert parsed_nested.reach_length_scaling is False
    assert parsed_legacy.n_steps == 7
    assert parsed_legacy.step_size_fraction == pytest.approx(0.375)
    assert parsed_dict.n_steps == 11
    assert parsed_dict.step_size_fraction == pytest.approx(0.2)


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


def test_pgd_explicit_radius_and_safety_cap_require_provenance() -> None:
    with pytest.raises(ValueError, match="fixed PGD L2 radius requires explicit provenance"):
        PgdFullStateEpsilonTrainingConfig(
            enabled=True,
            fixed_l2_radius_15cm=1.0,
        )

    with pytest.raises(ValueError, match="SISU PGD max L2 radius requires explicit provenance"):
        PgdFullStateEpsilonTrainingConfig(
            enabled=True,
            budget_schedule=BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
            sisu_max_l2_radius_15cm=1.0,
        )

    with pytest.raises(
        ValueError,
        match="PGD soft-energy safety cap radius requires explicit provenance",
    ):
        PgdFullStateEpsilonTrainingConfig(
            enabled=True,
            objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            energy_lambda=1.0,
            safety_cap_l2_radius_15cm=1.0,
        )

    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        energy_lambda=1.0,
    )
    payload = cfg.to_hps_dict()

    assert cfg.adversary_mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    assert cfg.safety_cap_l2_radius_15cm is None
    assert payload["inner_maximizer"]["projection"] == "none_cap_free_direct_soft_energy"
    assert payload["inner_maximizer"]["step_size_reference"] == (
        "absolute_normalized_gradient_step"
    )
    assert payload["safety_cap"]["enabled"] is False
    assert payload["safety_cap"]["role"] == "cap_free_soft_energy_no_trust_region"
    assert payload["budget_contract"]["effective_l2_radius_15cm"] is None
    assert payload["budget_contract"]["active_max_l2_radius_15cm"] is None
    assert payload["budget_contract"]["radius_bound_mode"] is False
    assert payload["budget_contract"]["budget_source"] is None
    assert payload["budget_contract"]["scientific_constraint"] == "soft_energy_penalty_cap_free"

    parsed = PgdFullStateEpsilonTrainingConfig.from_payload(payload)
    assert parsed.objective_kind == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
    assert parsed.soft_energy_lambda == pytest.approx(1.0)
    assert parsed.safety_cap_l2_radius_15cm is None

    with pytest.raises(ValueError, match="Finite-policy PGD soft-energy objectives require"):
        PgdFullStateEpsilonTrainingConfig(
            enabled=True,
            adversary_mechanism=LINEAR_NO_BIAS_POLICY,
            objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            energy_lambda=1.0,
        )


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
    assert cfg.controller_training_mode == ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND
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
            ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
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
    replay_args = resolve_run_spec_args(_args(run_spec=result["run_spec_path"]))

    assert replay_args.adaptive_epsilon_curriculum is True
    assert (
        replay_args.adaptive_epsilon_controller_training_mode
        == ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
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
                ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
            ),
            target_relative_multitarget=True,
        )
    )

    cfg = hps.adaptive_epsilon_curriculum
    assert cfg.controller_training_mode == ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
    assert (
        cfg.outer_adversarial_weight.applies_to
        == "optimized_direct_epsilon_channel_scale_for_controller_rollout"
    )


def test_adaptive_epsilon_curriculum_requires_soft_direct_pgd() -> None:
    with pytest.raises(ValueError, match="requires --broad-epsilon-pgd-training"):
        build_hps(_args(adaptive_epsilon_curriculum=True))

    with pytest.raises(ValueError, match="applies only to direct_epsilon"):
        build_hps(
            _args(
                broad_epsilon_pgd_training=True,
                broad_epsilon_pgd_mechanism=LINEAR_NO_BIAS_POLICY,
                broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                broad_epsilon_pgd_energy_lambda=1.0,
                broad_epsilon_pgd_safety_cap_15cm=1.0,
                broad_epsilon_pgd_safety_cap_source="unit_test_cap",
                adaptive_epsilon_curriculum=True,
                target_relative_multitarget=True,
            )
        )

    with pytest.raises(ValueError, match="requires --broad-epsilon-pgd-objective soft_energy"):
        build_hps(
            _args(
                broad_epsilon_pgd_training=True,
                adaptive_epsilon_curriculum=True,
                target_relative_multitarget=True,
            )
        )


def test_adaptive_epsilon_schedules_and_lambda_update_are_conservative() -> None:
    hps = build_hps(
        _args(
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            broad_epsilon_pgd_energy_lambda=10.0,
            adaptive_epsilon_curriculum=True,
            target_relative_multitarget=True,
        )
    )
    cfg = hps.adaptive_epsilon_curriculum

    assert _adaptive_epsilon_damage_target(cfg, 0) == pytest.approx(0.0)
    assert _adaptive_epsilon_damage_target(cfg, 1250) == pytest.approx(1750.0)
    assert _adaptive_epsilon_damage_target(cfg, 2500) == pytest.approx(3500.0)
    assert _adaptive_epsilon_damage_target(cfg, 7500) == pytest.approx(1000.0)
    assert _adaptive_epsilon_outer_weight(cfg, 0) == pytest.approx(0.0)
    assert _adaptive_epsilon_outer_weight(cfg, 1250) == pytest.approx(0.5)
    assert _adaptive_epsilon_outer_weight(cfg, 2500) == pytest.approx(1.0)

    state = _initial_adaptive_epsilon_state(hps)
    assert state is not None
    state, diagnostics = _update_adaptive_epsilon_state(
        state,
        cfg,
        batch_index=48,
        target_damage=1000.0,
        measured_damage=1200.0,
        measured_clean_loss=1.0,
    )
    assert diagnostics["update_due"] == np.asarray(False)
    assert diagnostics["lambda_updated"] == np.asarray(False)
    assert state.lambda_value == pytest.approx(10.0)

    state, diagnostics = _update_adaptive_epsilon_state(
        state,
        cfg,
        batch_index=49,
        target_damage=1000.0,
        measured_damage=1200.0,
        measured_clean_loss=1.0,
    )
    assert diagnostics["update_due"] == np.asarray(True)
    assert diagnostics["lambda_updated"] == np.asarray(True)
    assert state.lambda_value > 10.0
    assert state.update_count == 1


def test_adaptive_epsilon_continuation_schedule_is_relative_to_resume_start() -> None:
    hps = build_hps(
        _args(
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            broad_epsilon_pgd_energy_lambda=10.0,
            adaptive_epsilon_curriculum=True,
            target_relative_multitarget=True,
        )
    )
    cfg = hps.adaptive_epsilon_curriculum
    resumed_state = _initial_adaptive_epsilon_state(hps, schedule_start_batch=12000)
    scratch_state = _initial_adaptive_epsilon_state(hps)
    assert resumed_state is not None
    assert scratch_state is not None

    assert _adaptive_epsilon_schedule_batch(resumed_state, 12000) == 0
    assert _adaptive_epsilon_schedule_batch(resumed_state, 13250) == 1250
    assert _adaptive_epsilon_schedule_batch(resumed_state, 14500) == 2500
    assert _adaptive_epsilon_schedule_batch(resumed_state, 19499) == 7499
    assert _adaptive_epsilon_damage_target(
        cfg,
        _adaptive_epsilon_schedule_batch(resumed_state, 13250),
    ) == pytest.approx(1750.0)
    assert _adaptive_epsilon_outer_weight(
        cfg,
        _adaptive_epsilon_schedule_batch(resumed_state, 13250),
    ) == pytest.approx(0.5)

    assert _adaptive_epsilon_schedule_batch(scratch_state, 0) == 0
    assert _adaptive_epsilon_schedule_batch(scratch_state, 2500) == 2500


def test_scale_direct_epsilon_trial_specs_scales_only_epsilon_channel() -> None:
    clean = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((2, 1, 1), dtype=jnp.float32),
                )
            }
        ),
        inputs={
            "epsilon": jnp.asarray([[[1.0]], [[2.0]]], dtype=jnp.float32),
            "cue": jnp.asarray([[[5.0]], [[6.0]]], dtype=jnp.float32),
        },
        timeline=TrialTimeline(n_steps=1),
    )
    adversarial = TaskTrialSpec(
        inits=clean.inits,
        targets=clean.targets,
        inputs={
            "epsilon": jnp.asarray([[[5.0]], [[10.0]]], dtype=jnp.float32),
            "cue": jnp.asarray([[[50.0]], [[60.0]]], dtype=jnp.float32),
        },
        timeline=clean.timeline,
    )

    scaled = _scale_direct_epsilon_trial_specs(
        clean_specs=clean,
        adv_specs=adversarial,
        epsilon_scale=jnp.asarray(0.25, dtype=jnp.float32),
    )

    np.testing.assert_allclose(
        np.asarray(scaled.inputs["epsilon"]),
        np.asarray([[[2.0]], [[4.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(scaled.inputs["cue"]),
        np.asarray(adversarial.inputs["cue"]),
    )


def test_resume_optimizer_diagnostics_resize_pads_cross_length_buffers() -> None:
    adam_state = optax.ScaleByAdamState(
        count=jnp.asarray(7, dtype=jnp.int32),
        mu={"w": jnp.asarray([1.0, 2.0], dtype=jnp.float32)},
        nu={"w": jnp.asarray([3.0, 4.0], dtype=jnp.float32)},
    )
    optimizer_state = {
        "adam": adam_state,
        "gradient": GradientDiagnosticsState(
            count=jnp.asarray(2, dtype=jnp.int32),
            gradient_norm_pre_clip=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            gradient_clipped=jnp.asarray([True, False], dtype=bool),
            learning_rate=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        ),
        "update": UpdateDiagnosticsState(
            count=jnp.asarray(2, dtype=jnp.int32),
            update_norm=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
            parameter_norm=jnp.asarray([5.0, 6.0], dtype=jnp.float32),
            update_parameter_norm_ratio=jnp.asarray([0.3, 0.4], dtype=jnp.float32),
        ),
    }

    resized = _resize_optimizer_diagnostics_for_batches(optimizer_state, 4)

    np.testing.assert_allclose(resized["gradient"].gradient_norm_pre_clip[:2], [1.0, 2.0])
    assert np.isnan(np.asarray(resized["gradient"].gradient_norm_pre_clip[2:])).all()
    assert resized["gradient"].gradient_clipped.tolist() == [True, False, False, False]
    np.testing.assert_allclose(resized["update"].update_norm[:2], [3.0, 4.0])
    assert np.isnan(np.asarray(resized["update"].update_norm[2:])).all()
    assert int(resized["gradient"].count) == 2
    assert int(resized["update"].count) == 2
    assert int(resized["adam"].count) == 7
    np.testing.assert_allclose(resized["adam"].mu["w"], [1.0, 2.0])
    np.testing.assert_allclose(resized["adam"].nu["w"], [3.0, 4.0])

    shrunk = _resize_optimizer_diagnostics_for_batches(resized, 1)
    assert shrunk["gradient"].gradient_norm_pre_clip.shape == (1,)
    np.testing.assert_allclose(shrunk["gradient"].gradient_norm_pre_clip, [1.0])

    vmapped = {
        "gradient": GradientDiagnosticsState(
            count=jnp.asarray([2, 2], dtype=jnp.int32),
            gradient_norm_pre_clip=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
            gradient_clipped=jnp.asarray([[True, False], [False, True]], dtype=bool),
            learning_rate=jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
        )
    }
    resized_vmapped = _resize_optimizer_diagnostics_for_batches(vmapped, 4)
    assert resized_vmapped["gradient"].gradient_norm_pre_clip.shape == (2, 4)
    np.testing.assert_allclose(
        resized_vmapped["gradient"].gradient_norm_pre_clip[:, :2],
        [[1.0, 2.0], [3.0, 4.0]],
    )
    assert np.isnan(np.asarray(resized_vmapped["gradient"].gradient_norm_pre_clip[:, 2:])).all()
    assert resized_vmapped["gradient"].gradient_clipped.tolist() == [
        [True, False, False, False],
        [False, True, False, False],
    ]


def test_adaptive_epsilon_zero_adversary_guard_stops_after_two_active_checkpoints() -> None:
    guard = _initial_adaptive_epsilon_zero_guard(enabled=True)
    inactive_zero = {
        "adaptive_epsilon_adaptive_update_inner_selected_objective_gain_over_zero": np.array([0.0]),
        "adaptive_epsilon_target_damage": np.array([0.0]),
        "adaptive_epsilon_outer_weight": np.array([0.0]),
    }
    active_zero = {
        "adaptive_epsilon_adaptive_update_inner_selected_objective_gain_over_zero": np.array([0.0]),
        "adaptive_epsilon_target_damage": np.array([100.0]),
        "adaptive_epsilon_outer_weight": np.array([1.0]),
    }
    active_nonzero = {
        "adaptive_epsilon_adaptive_update_inner_selected_objective_gain_over_zero": np.array(
            [1.0e-3]
        ),
        "adaptive_epsilon_target_damage": np.array([100.0]),
        "adaptive_epsilon_outer_weight": np.array([1.0]),
    }

    guard = _update_adaptive_epsilon_zero_guard(guard, inactive_zero)
    assert guard["last_checkpoint"]["active"] is False
    assert guard["consecutive_active_zero_adversary_checkpoints"] == 0
    assert guard["should_stop"] is False

    guard = _update_adaptive_epsilon_zero_guard(guard, active_zero)
    assert guard["last_checkpoint"]["active"] is True
    assert guard["last_checkpoint"]["zero_adversary"] is True
    assert guard["consecutive_active_zero_adversary_checkpoints"] == 1
    assert guard["should_stop"] is False

    guard = _update_adaptive_epsilon_zero_guard(guard, active_nonzero)
    assert guard["last_checkpoint"]["zero_adversary"] is False
    assert guard["consecutive_active_zero_adversary_checkpoints"] == 0
    assert guard["should_stop"] is False

    guard = _update_adaptive_epsilon_zero_guard(guard, active_zero)
    guard = _update_adaptive_epsilon_zero_guard(guard, active_zero)
    assert guard["consecutive_active_zero_adversary_checkpoints"] == 2
    assert guard["should_stop"] is True


def test_adaptive_epsilon_zero_guard_survives_checkpoint_resume(tmp_path: Path) -> None:
    active_zero = {
        "adaptive_epsilon_adaptive_update_inner_selected_objective_gain_over_zero": np.array([0.0]),
        "adaptive_epsilon_target_damage": np.array([100.0]),
        "adaptive_epsilon_outer_weight": np.array([1.0]),
    }
    guard = _update_adaptive_epsilon_zero_guard(
        _initial_adaptive_epsilon_zero_guard(enabled=True),
        active_zero,
    )
    checkpoint_root = tmp_path / "checkpoints"
    model_template = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    optimizer_state_template = jnp.asarray([3.0], dtype=jnp.float32)
    state = TrainingState(
        model=model_template,
        optimizer_state=optimizer_state_template,
        completed_batches=4,
        key=jnp.asarray([0, 1], dtype=jnp.uint32),
        history=None,
        adaptive_epsilon_state=AdaptiveEpsilonState(
            lambda_value=0.5,
            zero_adversary_guard=guard,
        ),
    )
    save_training_checkpoint(
        checkpoint_root,
        state,
        args=_args(n_train_batches=8, checkpoint_interval_batches=4),
        run_spec={"schema_version": "test"},
    )

    loaded = load_latest_checkpoint(
        checkpoint_root,
        model_template=model_template,
        optimizer_state_template=optimizer_state_template,
    )
    assert loaded.adaptive_epsilon_state is not None
    restored_guard = _adaptive_epsilon_zero_guard_from_state(
        loaded.adaptive_epsilon_state,
        enabled=True,
    )

    assert restored_guard["checkpoints_seen"] == 1
    assert restored_guard["consecutive_active_zero_adversary_checkpoints"] == 1
    assert restored_guard["should_stop"] is False

    resumed_guard = _update_adaptive_epsilon_zero_guard(restored_guard, active_zero)
    assert resumed_guard["checkpoints_seen"] == 2
    assert resumed_guard["consecutive_active_zero_adversary_checkpoints"] == 2
    assert resumed_guard["should_stop"] is True


def test_adaptive_epsilon_zero_guard_legacy_checkpoint_defaults_to_zero(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    model_template = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    optimizer_state_template = jnp.asarray([3.0], dtype=jnp.float32)
    state = TrainingState(
        model=model_template,
        optimizer_state=optimizer_state_template,
        completed_batches=4,
        key=jnp.asarray([0, 1], dtype=jnp.uint32),
        history=None,
        adaptive_epsilon_state=AdaptiveEpsilonState(lambda_value=0.5),
    )
    checkpoint_path = save_training_checkpoint(
        checkpoint_root,
        state,
        args=_args(n_train_batches=8, checkpoint_interval_batches=4),
        run_spec={"schema_version": "test"},
    )
    metadata_path = checkpoint_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["adaptive_epsilon_state"].pop("zero_adversary_guard", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_latest_checkpoint(
        checkpoint_root,
        model_template=model_template,
        optimizer_state_template=optimizer_state_template,
    )
    assert loaded.adaptive_epsilon_state is not None
    restored_guard = _adaptive_epsilon_zero_guard_from_state(
        loaded.adaptive_epsilon_state,
        enabled=True,
    )

    assert restored_guard["checkpoints_seen"] == 0
    assert restored_guard["consecutive_active_zero_adversary_checkpoints"] == 0
    assert restored_guard["should_stop"] is False


def test_adaptive_epsilon_lambda_update_uses_clipped_log_ratio() -> None:
    hps = build_hps(
        _args(
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            broad_epsilon_pgd_energy_lambda=10.0,
            adaptive_epsilon_curriculum=True,
            target_relative_multitarget=True,
            adaptive_epsilon_update_interval_batches=1,
            adaptive_epsilon_ema_alpha=1.0,
        )
    )
    cfg = hps.adaptive_epsilon_curriculum
    base_state = _initial_adaptive_epsilon_state(hps)
    assert base_state is not None

    high_state, high_diagnostics = _update_adaptive_epsilon_state(
        base_state,
        cfg,
        batch_index=0,
        target_damage=100.0,
        measured_damage=200.0,
        measured_clean_loss=1.0,
    )
    low_state, low_diagnostics = _update_adaptive_epsilon_state(
        base_state,
        cfg,
        batch_index=0,
        target_damage=100.0,
        measured_damage=50.0,
        measured_clean_loss=1.0,
    )

    assert high_diagnostics["lambda_log_step"] == pytest.approx(
        -float(low_diagnostics["lambda_log_step"])
    )
    assert high_diagnostics["lambda_log_step"] == pytest.approx(0.1 * math.log(2.0))
    assert high_state.lambda_value / base_state.lambda_value == pytest.approx(
        base_state.lambda_value / low_state.lambda_value
    )

    clipped_state, clipped_diagnostics = _update_adaptive_epsilon_state(
        base_state,
        cfg,
        batch_index=0,
        target_damage=100.0,
        measured_damage=1.0e9,
        measured_clean_loss=1.0,
    )
    assert clipped_diagnostics["lambda_log_step"] == pytest.approx(cfg.lambda_update.max_log_step)
    assert clipped_state.lambda_value == pytest.approx(
        base_state.lambda_value * math.exp(cfg.lambda_update.max_log_step)
    )

    zero_target_state, zero_target_diagnostics = _update_adaptive_epsilon_state(
        base_state,
        cfg,
        batch_index=0,
        target_damage=0.0,
        measured_damage=1.0e9,
        measured_clean_loss=1.0,
    )
    assert zero_target_diagnostics["update_due"] == np.asarray(True)
    assert zero_target_diagnostics["lambda_updated"] == np.asarray(False)
    assert zero_target_diagnostics["lambda_log_step"] == pytest.approx(0.0)
    assert zero_target_state.lambda_value == pytest.approx(base_state.lambda_value)


def test_adaptive_epsilon_application_ramp_freeze_holds_then_seeds_ema() -> None:
    hps = build_hps(
        _args(
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            broad_epsilon_pgd_energy_lambda=10.0,
            adaptive_epsilon_curriculum=True,
            target_relative_multitarget=True,
            adaptive_epsilon_update_interval_batches=1,
            adaptive_epsilon_ema_alpha=0.1,
            adaptive_epsilon_outer_weight_ramp_batches=2,
            adaptive_epsilon_freeze_during_application_ramp=True,
        )
    )
    cfg = hps.adaptive_epsilon_curriculum
    state = _initial_adaptive_epsilon_state(hps)
    assert state is not None

    held_state, held_diagnostics = _update_adaptive_epsilon_state(
        state,
        cfg,
        batch_index=0,
        target_damage=1.0,
        measured_damage=2.0,
        measured_clean_loss=1.0,
    )
    assert held_state == state
    assert held_diagnostics["application_ramp_frozen"] == np.asarray(True)
    assert held_diagnostics["lambda_updated"] == np.asarray(False)
    assert np.isnan(held_diagnostics["damage_ema"])

    seeded_state, seeded_diagnostics = _update_adaptive_epsilon_state(
        held_state,
        cfg,
        batch_index=2,
        target_damage=1.0,
        measured_damage=2.0,
        measured_clean_loss=1.0,
    )
    assert seeded_diagnostics["application_ramp_frozen"] == np.asarray(False)
    assert seeded_diagnostics["ema_seeded_post_ramp"] == np.asarray(True)
    assert seeded_diagnostics["lambda_updated"] == np.asarray(False)
    assert seeded_state.damage_ema == pytest.approx(2.0)
    assert seeded_state.clean_loss_ema == pytest.approx(1.0)
    assert seeded_state.lambda_value == pytest.approx(state.lambda_value)

    updated_state, updated_diagnostics = _update_adaptive_epsilon_state(
        seeded_state,
        cfg,
        batch_index=3,
        target_damage=1.0,
        measured_damage=2.0,
        measured_clean_loss=1.0,
    )
    assert updated_diagnostics["ema_seeded_post_ramp"] == np.asarray(False)
    assert updated_diagnostics["lambda_updated"] == np.asarray(True)
    assert updated_state.lambda_value > seeded_state.lambda_value


def test_adaptive_epsilon_application_ramp_freeze_default_off_and_zero_target_is_not_special() -> (
    None
):
    hps = build_hps(
        _args(
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            broad_epsilon_pgd_energy_lambda=10.0,
            adaptive_epsilon_curriculum=True,
            target_relative_multitarget=True,
            adaptive_epsilon_update_interval_batches=1,
            adaptive_epsilon_outer_weight_ramp_batches=2,
        )
    )
    cfg = hps.adaptive_epsilon_curriculum
    assert cfg.lambda_update.freeze_during_application_ramp is False
    state = _initial_adaptive_epsilon_state(hps)
    assert state is not None

    live_state, live_diagnostics = _update_adaptive_epsilon_state(
        state,
        cfg,
        batch_index=0,
        target_damage=1.0,
        measured_damage=2.0,
        measured_clean_loss=1.0,
    )
    assert live_diagnostics["application_ramp_frozen"] == np.asarray(False)
    assert live_state.damage_ema == pytest.approx(2.0)
    assert live_diagnostics["lambda_updated"] == np.asarray(True)

    zero_target_state, zero_target_diagnostics = _update_adaptive_epsilon_state(
        state,
        cfg,
        batch_index=0,
        target_damage=0.0,
        measured_damage=2.0,
        measured_clean_loss=1.0,
    )
    assert zero_target_diagnostics["application_ramp_frozen"] == np.asarray(False)
    assert zero_target_state.damage_ema == pytest.approx(2.0)


def test_adaptive_epsilon_probe_estimator_recovers_synthetic_gain() -> None:
    hps = build_hps(
        _args(
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            broad_epsilon_pgd_energy_lambda=10.0,
            adaptive_epsilon_curriculum=True,
            target_relative_multitarget=True,
            adaptive_epsilon_update_interval_batches=1,
            adaptive_epsilon_ema_alpha=1.0,
        )
    )
    cfg = hps.adaptive_epsilon_curriculum
    expected_gain = 2.3
    base_lambda = 10.0

    def damage(lambda_value: float) -> float:
        return 1000.0 * math.exp(-expected_gain * math.log(lambda_value / base_lambda))

    state = AdaptiveEpsilonState(
        lambda_value=base_lambda,
        damage_ema=damage(base_lambda),
        clean_loss_ema=1.0,
        last_log_damage_ema=math.log(damage(base_lambda)),
    )
    for batch_index, log_step in enumerate([0.02, -0.03, 0.04, -0.02]):
        probed_lambda = state.lambda_value * math.exp(log_step)
        state = replace(
            state,
            lambda_value=probed_lambda,
            pending_lambda_log_step=log_step,
            pending_log_damage_ema=math.log(damage(state.lambda_value)),
        )
        state, diagnostics = _update_adaptive_epsilon_state(
            state,
            cfg,
            batch_index=batch_index,
            target_damage=damage(probed_lambda),
            measured_damage=damage(probed_lambda),
            measured_clean_loss=1.0,
        )

        assert diagnostics["gain_probe_raw"] == pytest.approx(expected_gain)

    assert state.gain_estimate == pytest.approx(expected_gain)
    assert state.gain_samples == 4


def test_adaptive_epsilon_gain_normalization_stabilizes_small_margin_staircase() -> None:
    true_gain = 4.0
    target = 1.0
    initial_lambda = 0.8

    def damage(lambda_value: float) -> float:
        return target * math.exp(-true_gain * math.log(lambda_value))

    def run_staircase(*, normalized: bool) -> list[float]:
        hps = build_hps(
            _args(
                broad_epsilon_pgd_training=True,
                broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                broad_epsilon_pgd_energy_lambda=initial_lambda,
                adaptive_epsilon_curriculum=True,
                target_relative_multitarget=True,
                adaptive_epsilon_update_interval_batches=1,
                adaptive_epsilon_ema_alpha=1.0,
                adaptive_epsilon_eta=0.7,
                adaptive_epsilon_deadband_frac=0.0,
                adaptive_epsilon_gain_normalization=normalized,
                adaptive_epsilon_gain_min=0.1,
                adaptive_epsilon_gain_max=10.0,
                adaptive_epsilon_max_log_step=10.0,
            )
        )
        cfg = hps.adaptive_epsilon_curriculum
        state = AdaptiveEpsilonState(
            lambda_value=initial_lambda,
            damage_ema=damage(initial_lambda),
            clean_loss_ema=1.0,
            gain_estimate=true_gain if normalized else None,
        )
        errors = []
        for batch_index in range(8):
            measured = damage(state.lambda_value)
            errors.append(abs(math.log(measured / target)))
            state, _diagnostics = _update_adaptive_epsilon_state(
                state,
                cfg,
                batch_index=batch_index,
                target_damage=target,
                measured_damage=measured,
                measured_clean_loss=1.0,
            )
        return errors

    fixed_gain_errors = run_staircase(normalized=False)
    normalized_errors = run_staircase(normalized=True)

    assert normalized_errors[-1] < normalized_errors[0] * 0.01
    assert fixed_gain_errors[-1] > fixed_gain_errors[0] * 10.0


def test_adaptive_epsilon_damage_eval_batch_is_nominal_when_training_batch_is_perturbed() -> None:
    class ContaminatingPerturbationTask:
        seed_validation = 123

        @staticmethod
        def _trial(marker: float, *, contaminated: bool) -> TaskTrialSpec:
            intervene = (
                {
                    "perturbation_bank": TreeNamespace(
                        active=jnp.asarray(True),
                        marker=jnp.asarray(marker, dtype=jnp.float32),
                    )
                }
                if contaminated
                else {}
            )
            return TaskTrialSpec(
                inits=WhereDict({}),
                targets=WhereDict(
                    {
                        "mechanics.effector.pos": TargetSpec(
                            value=jnp.zeros((1, 1), dtype=jnp.float32),
                        )
                    }
                ),
                inputs={
                    "epsilon": jnp.zeros((1, 1), dtype=jnp.float32),
                    "perturbation_marker": jnp.asarray([[marker]], dtype=jnp.float32),
                },
                intervene=intervene,
                timeline=TrialTimeline(n_steps=1),
            )

        def get_train_trial(self, key, batch_info=None):
            del key, batch_info
            return self._trial(0.0, contaminated=False)

        def get_train_trial_with_intervenor_params(self, key, batch_info=None):
            del key, batch_info
            return self._trial(1.0, contaminated=True)

    task = ContaminatingPerturbationTask()
    batch_info = BatchInfo(
        size=4,
        start=jnp.asarray(0),
        current=jnp.asarray(17),
        total=jnp.asarray(100),
    )
    keys_trials = jr.split(jr.PRNGKey(1), 4)

    training_specs = _sample_adaptive_epsilon_training_batch(
        task,
        batch_info=batch_info,
        keys_trials=keys_trials,
    )
    eval_specs, first_keys_init, first_keys_model = _sample_adaptive_epsilon_damage_eval_batch(
        task,
        jr.PRNGKey(2),
        batch_info=batch_info,
        batch_size=4,
        include_graph_adapter_inputs=True,
        force_filter_feedback=True,
    )
    eval_specs_again, second_keys_init, second_keys_model = (
        _sample_adaptive_epsilon_damage_eval_batch(
            task,
            jr.PRNGKey(2),
            batch_info=batch_info,
            batch_size=4,
            include_graph_adapter_inputs=True,
            force_filter_feedback=True,
        )
    )

    np.testing.assert_allclose(training_specs.inputs["perturbation_marker"], 1.0)
    assert "perturbation_bank" in training_specs.intervene
    np.testing.assert_allclose(eval_specs.inputs["perturbation_marker"], 0.0)
    assert "perturbation_bank" in eval_specs.intervene
    np.testing.assert_allclose(eval_specs.intervene["perturbation_bank"].active, False)
    for spec in graph_adapter_specs(force_filter_feedback=True).values():
        assert spec.input_key in eval_specs.inputs
        np.testing.assert_allclose(eval_specs.inputs[spec.input_key], 0.0)
        assert eval_specs.inputs[spec.input_key].shape[-1] == spec.payload_shape[-1]
    np.testing.assert_allclose(
        eval_specs.inputs["perturbation_marker"],
        eval_specs_again.inputs["perturbation_marker"],
    )
    np.testing.assert_array_equal(first_keys_init, second_keys_init)
    np.testing.assert_array_equal(first_keys_model, second_keys_model)


def test_pgd_inner_optimizer_metadata_and_parser_round_trip() -> None:
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=LINEAR_NO_BIAS_POLICY,
        objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        energy_lambda=3.0,
        safety_cap_l2_radius_15cm=1.0,
        safety_cap_source="unit_test_cap",
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


def test_direct_epsilon_accepts_adam_inner_optimizer() -> None:
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
        inner_optimizer_method=BROAD_EPSILON_PGD_ADAM,
    )

    assert cfg.inner_optimizer_method == BROAD_EPSILON_PGD_ADAM


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


def test_pgd_finite_mechanism_serializes_live_graph_contract() -> None:
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=LINEAR_NO_BIAS_POLICY,
    )
    payload = cfg.to_hps_dict()

    assert payload["adversary_mechanism"] == LINEAR_NO_BIAS_POLICY
    assert payload["mechanism"]["implementation_status"] == "implemented"
    assert payload["mechanism"]["required_policy_contract"]["live_feature_source"] == (
        "live_perturbed_rollout_state"
    )
    assert payload["mechanism"]["required_policy_contract"]["feature_source_detail"] == (
        "pre_mechanics_state"
    )
    assert payload["mechanism"]["live_evaluation"]["implementation"] == "graph_component"
    assert payload["mechanism"]["live_evaluation"]["component"] == (
        FINITE_EPSILON_POLICY_GRAPH_COMPONENT
    )
    assert payload["mechanism"]["live_evaluation"]["component_label"] == (
        FINITE_EPSILON_POLICY_NODE_LABEL
    )
    assert payload["mechanism"]["live_evaluation"]["static_clean_rollout_materialization"] is False
    assert payload["mechanism"]["no_fake_open_loop_replay"] is True
    assert payload["mechanism"]["runtime_inputs"]["gains"] == (
        f"TaskTrialSpec.inputs[{FINITE_POLICY_GAINS_INPUT!r}]"
    )
    parsed = PgdFullStateEpsilonTrainingConfig.from_payload(payload)
    assert parsed.adversary_mechanism == LINEAR_NO_BIAS_POLICY

    assert make_broad_epsilon_pgd_pre_step(payload) is not None


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
    replay_args = resolve_run_spec_args(_args(run_spec=result["run_spec_path"]))
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
    replay_args = resolve_run_spec_args(_args(run_spec=result["run_spec_path"]))
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
        broad_epsilon_pgd_safety_cap_source="unit_test_cap",
    )

    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text())
    pgd = payload["hps"]["broad_epsilon_pgd_training"]
    optimizer = pgd["inner_maximizer"]
    replay_args = resolve_run_spec_args(_args(run_spec=result["run_spec_path"]))

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


def test_finite_epsilon_component_uses_live_6d_target_centered_state() -> None:
    component = CsLssFiniteEpsilonPolicy(
        policy_class=AFFINE_POLICY,
        physical_block_size=6,
    )
    state = (
        jnp.zeros((36,), dtype=jnp.float32)
        .at[0:6]
        .set(jnp.array([0.20, -0.05, 0.3, -0.4, 0.01, -0.02], dtype=jnp.float32))
    )
    gains = jnp.zeros((6, 36), dtype=jnp.float32)
    gains = gains.at[0, 0].set(2.0).at[1, 2].set(-3.0)
    bias = jnp.arange(6, dtype=jnp.float32) * 0.1

    outputs, _ = component(
        {
            "base_epsilon": jnp.ones((6,), dtype=jnp.float32),
            "state": state,
            "target": jnp.array([0.15, 0.05], dtype=jnp.float32),
            "gains": gains,
            "bias": bias,
        },
        None,
        key=jr.PRNGKey(0),
    )

    assert outputs["epsilon"].shape == (6,)
    assert outputs["epsilon"][0] == pytest.approx(1.0 + 2.0 * 0.05)
    assert outputs["epsilon"][1] == pytest.approx(1.0 - 3.0 * 0.3 + 0.1)


def test_finite_pgd_graph_wires_policy_inputs_to_mechanics_epsilon() -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=4,
        target_relative_feedback=True,
        bind_epsilon_input=True,
        finite_epsilon_policy=LINEAR_NO_BIAS_POLICY,
        no_integrator_state=True,
        key=jr.PRNGKey(0),
    )

    assert spec.nodes["finite_epsilon_policy"].type == "AffineValueComposer"
    assert spec.nodes["finite_epsilon_policy"].input_ports == [
        "base",
        "state",
        "target",
        "gain",
        "bias",
    ]
    assert spec.nodes["finite_epsilon_policy"].output_ports == ["value"]
    assert spec.nodes["finite_epsilon_policy"].params["use_bias"] is False
    assert spec.nodes["finite_epsilon_policy"].params["output_block_size"] == CS_REDUCED_EPSILON_DIM
    assert spec.input_bindings["epsilon"] == ("finite_epsilon_policy", "base")
    assert spec.input_bindings[FINITE_POLICY_GAINS_INPUT] == (
        "finite_epsilon_policy",
        "gain",
    )
    assert FINITE_POLICY_BIAS_INPUT not in spec.input_bindings
    assert any(
        wire.source_node == "mechanics"
        and wire.source_port == "state"
        and wire.target_node == "finite_epsilon_policy"
        and wire.target_port == "state"
        and wire.temporality == "recurrent"
        for wire in spec.wires
    )
    assert any(
        wire.source_node == "target_source"
        and wire.source_port == "output"
        and wire.target_node == "feedback"
        and wire.target_port == "target"
        for wire in spec.wires
    )
    assert any(
        wire.source_node == "target_source"
        and wire.source_port == "output"
        and wire.target_node == "finite_epsilon_policy"
        and wire.target_port == "target"
        for wire in spec.wires
    )
    assert any(
        wire.source_node == "finite_epsilon_policy"
        and wire.source_port == "value"
        and wire.target_node == "mechanics"
        and wire.target_port == "epsilon"
        for wire in spec.wires
    )


@pytest.mark.parametrize("policy_class", [LINEAR_NO_BIAS_POLICY, AFFINE_POLICY])
def test_pgd_mechanism_contract_points_at_native_finite_policy_component(
    policy_class: str,
) -> None:
    """The live-eval descriptor must name the native node actually emitted.

    Structural equivalence check for the retired-ID retirement (b20e0ea): the
    ``pgd_adversary_mechanism_contract`` descriptor no longer stamps the retired
    ``RLRMPCsLssFiniteEpsilonPolicy`` ID; it must instead resolve to the exact
    Feedbax-native ``AffineValueComposer`` node (type + label) that
    ``build_cs_lss_gru_graph_spec`` emits for the same policy class.
    """

    cfg = PgdFullStateEpsilonTrainingConfig(enabled=True, adversary_mechanism=policy_class)
    contract = cs_perturbation_training.pgd_adversary_mechanism_contract(cfg)

    spec = build_cs_lss_gru_graph_spec(
        hidden_size=4,
        target_relative_feedback=True,
        bind_epsilon_input=True,
        finite_epsilon_policy=policy_class,
        no_integrator_state=True,
        key=jr.PRNGKey(0),
    )

    # Descriptor names must be native, not the retired branded ID.
    assert contract["graph_component"] == FINITE_EPSILON_POLICY_GRAPH_COMPONENT
    assert contract["graph_component_label"] == FINITE_EPSILON_POLICY_NODE_LABEL
    assert contract["live_evaluation"]["component"] == FINITE_EPSILON_POLICY_GRAPH_COMPONENT
    assert contract["live_evaluation"]["component_label"] == FINITE_EPSILON_POLICY_NODE_LABEL
    assert "RLRMPCsLss" not in contract["graph_component"]

    # The descriptor must resolve to a node that actually exists in the emitted
    # native GraphSpec, with the matching component type.
    node = spec.nodes[contract["graph_component_label"]]
    assert node.type == contract["graph_component"]
    assert node.params["label"] == contract["graph_component_label"]
    assert node.params["use_bias"] is (policy_class == AFFINE_POLICY)


def test_finite_pgd_inner_maximizer_installs_policy_inputs_before_rollout() -> None:
    class FiniteInputOnlyTask:
        def __init__(self) -> None:
            self.n_eval_calls = 0

        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            self.n_eval_calls += 1
            assert FINITE_POLICY_GAINS_INPUT in trial_specs.inputs
            np.testing.assert_allclose(np.asarray(trial_specs.inputs["epsilon"]), 0.0)
            gains = jnp.asarray(trial_specs.inputs[FINITE_POLICY_GAINS_INPUT])
            return TreeNamespace(
                mechanics=TreeNamespace(
                    vector=jnp.sum(gains, axis=-2),
                )
            )

    class SumVectorLoss:
        def __call__(self, states, trial_specs, model):
            del trial_specs, model
            return TreeNamespace(total=jnp.sum(states.mechanics.vector))

    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=LINEAR_NO_BIAS_POLICY,
        reach_length_scaling=False,
        n_steps=1,
        epsilon_dim=2,
        objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        energy_lambda=1.0,
        safety_cap_l2_radius_15cm=1.0,
        safety_cap_source="unit_test_cap",
    )
    trial_specs = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((1, 4), dtype=jnp.float32)}),
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
    task = FiniteInputOnlyTask()

    updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        task,
        model=None,
        trial_specs=trial_specs,
        loss_func=SumVectorLoss(),
        keys_model=None,
        config=cfg,
        return_diagnostics=True,
    )

    assert task.n_eval_calls > 0
    assert FINITE_POLICY_GAINS_INPUT in updated.inputs
    assert "finite_policy_delta_zero_energy_mean" in diagnostics
    np.testing.assert_allclose(np.asarray(updated.inputs["epsilon"]), 0.0)


def test_finite_adam_inner_maximizer_uses_live_policy_inputs() -> None:
    class FiniteInputOnlyTask:
        def __init__(self) -> None:
            self.n_eval_calls = 0

        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            self.n_eval_calls += 1
            assert FINITE_POLICY_GAINS_INPUT in trial_specs.inputs
            assert FINITE_POLICY_BIAS_INPUT not in trial_specs.inputs
            np.testing.assert_allclose(np.asarray(trial_specs.inputs["epsilon"]), 0.0)
            gains = jnp.asarray(trial_specs.inputs[FINITE_POLICY_GAINS_INPUT])
            return TreeNamespace(
                mechanics=TreeNamespace(
                    vector=jnp.sum(gains, axis=-2),
                )
            )

    class SumVectorLoss:
        def __call__(self, states, trial_specs, model):
            del trial_specs, model
            return TreeNamespace(total=jnp.sum(states.mechanics.vector))

    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=LINEAR_NO_BIAS_POLICY,
        reach_length_scaling=False,
        n_steps=2,
        epsilon_dim=2,
        objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        energy_lambda=1e-6,
        safety_cap_l2_radius_15cm=1.0,
        safety_cap_source="unit_test_cap",
        inner_optimizer_method=BROAD_EPSILON_PGD_ADAM,
        adam_learning_rate=1e-2,
    )
    trial_specs = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((1, 4), dtype=jnp.float32)}),
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
    task = FiniteInputOnlyTask()

    updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        task,
        model=None,
        trial_specs=trial_specs,
        loss_func=SumVectorLoss(),
        keys_model=None,
        config=cfg,
        return_diagnostics=True,
    )

    assert task.n_eval_calls > 0
    assert diagnostics["inner_optimizer_method_is_adam"].tolist() is True
    assert diagnostics["adam_learning_rate"] == pytest.approx(1e-2)
    assert diagnostics["inner_objective_after"] > diagnostics["inner_objective_before"]
    assert FINITE_POLICY_GAINS_INPUT in updated.inputs
    assert FINITE_POLICY_BIAS_INPUT not in updated.inputs
    assert np.linalg.norm(np.asarray(updated.inputs[FINITE_POLICY_GAINS_INPUT])) > 0.0
    np.testing.assert_allclose(np.asarray(updated.inputs["epsilon"]), 0.0)


@pytest.mark.parametrize(
    ("inner_optimizer_method", "expected_adam"),
    [
        (BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT, False),
        (BROAD_EPSILON_PGD_ADAM, True),
    ],
)
def test_direct_epsilon_pgd_does_not_install_finite_policy_inputs(
    inner_optimizer_method: str,
    expected_adam: bool,
) -> None:
    class DirectEpsilonTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            assert FINITE_POLICY_GAINS_INPUT not in trial_specs.inputs
            assert FINITE_POLICY_BIAS_INPUT not in trial_specs.inputs
            epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
            return TreeNamespace(
                mechanics=TreeNamespace(
                    vector=epsilon,
                )
            )

    class SumVectorLoss:
        def __call__(self, states, trial_specs, model):
            del trial_specs, model
            return TreeNamespace(total=jnp.sum(states.mechanics.vector))

    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
        inner_optimizer_method=inner_optimizer_method,
        reach_length_scaling=False,
        n_steps=1,
        epsilon_dim=2,
    )
    trial_specs = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((1, 2), dtype=jnp.float32)}),
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

    updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        DirectEpsilonTask(),
        model=None,
        trial_specs=trial_specs,
        loss_func=SumVectorLoss(),
        keys_model=None,
        config=cfg,
        return_diagnostics=True,
    )

    assert FINITE_POLICY_GAINS_INPUT not in updated.inputs
    assert FINITE_POLICY_BIAS_INPUT not in updated.inputs
    assert "finite_policy_delta_zero_energy_mean" not in diagnostics
    assert diagnostics["inner_optimizer_method_is_adam"].tolist() is expected_adam
    assert updated.inputs["epsilon"].shape == (1, 2, 2)


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


def test_finite_pgd_public_routing_matches_finite_core_equivalence() -> None:
    class FiniteInputOnlyTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            gains = jnp.asarray(trial_specs.inputs[FINITE_POLICY_GAINS_INPUT])
            return TreeNamespace(
                mechanics=TreeNamespace(
                    vector=jnp.sum(gains, axis=-2),
                )
            )

    class SumVectorLoss:
        def __call__(self, states, trial_specs, model):
            del trial_specs, model
            return TreeNamespace(total=jnp.sum(states.mechanics.vector))

    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=LINEAR_NO_BIAS_POLICY,
        reach_length_scaling=False,
        n_steps=2,
        epsilon_dim=2,
        objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        energy_lambda=1e-6,
        safety_cap_l2_radius_15cm=1.0,
        safety_cap_source="unit_test_cap",
        inner_optimizer_method=BROAD_EPSILON_PGD_ADAM,
        adam_learning_rate=1e-2,
    )
    trial_specs = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((1, 4), dtype=jnp.float32)}),
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

    def comparable(result):
        updated, diagnostics = result
        return {
            "gains": updated.inputs[FINITE_POLICY_GAINS_INPUT],
            "epsilon": updated.inputs["epsilon"],
            "inner_objective_after": diagnostics["inner_objective_after"],
            "inner_objective_best": diagnostics["inner_objective_best"],
            "inner_objective_final_endpoint": diagnostics["inner_objective_final_endpoint"],
            "finite_policy_final_endpoint_energy_mean": diagnostics[
                "finite_policy_final_endpoint_energy_mean"
            ],
        }

    report = run_paired_equivalence(
        "pgd_finite_policy_inner_maximizer",
        lambda: run_broad_epsilon_pgd_inner_maximizer(
            FiniteInputOnlyTask(),
            model=None,
            trial_specs=trial_specs,
            loss_func=SumVectorLoss(),
            keys_model=None,
            config=cfg,
            return_diagnostics=True,
        ),
        lambda: _run_finite_broad_epsilon_pgd_inner_maximizer(
            FiniteInputOnlyTask(),
            model=None,
            trial_specs=trial_specs,
            loss_func=SumVectorLoss(),
            keys_model=None,
            cfg=cfg,
            return_diagnostics=True,
        ),
        comparable=comparable,
        left_label="public_router",
        right_label="finite_core",
    )

    assert_paired_equivalent(report)


def test_pgd_sisu_budget_radius_uses_sqrt_energy_fraction() -> None:
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        budget_schedule=BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
        reach_length_scaling=False,
        sisu_condition_input="input",
        sisu_max_l2_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
        sisu_max_radius_source="effective_020a65b_pgd_training_radius",
        epsilon_dim=1,
    )
    trial_specs = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((3, 8), dtype=jnp.float32)}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((3, 2, 2), dtype=jnp.float32),
                )
            }
        ),
        inputs={
            "input": jnp.asarray(
                [[0.0, 0.0], [0.25, 0.25], [1.0, 1.0]],
                dtype=jnp.float32,
            ),
            "epsilon": jnp.zeros((3, 2, 1), dtype=jnp.float32),
        },
        timeline=TrialTimeline(n_steps=2),
    )

    radius = _broad_epsilon_l2_radius(trial_specs, cfg)

    np.testing.assert_allclose(
        radius,
        np.asarray(
            [
                0.0,
                0.5 * HISTORICAL_020A65B_PGD_RADIUS_15CM,
                HISTORICAL_020A65B_PGD_RADIUS_15CM,
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-10,
    )


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


def test_pgd_broad_epsilon_lane_requires_target_relative_and_excludes_random_lane() -> None:
    with pytest.raises(ValueError, match="Reach-scaled broad-epsilon"):
        build_hps(_args(broad_epsilon_pgd_training=True))

    with pytest.raises(ValueError, match="cannot be combined"):
        build_hps(
            _args(
                target_relative_multitarget=True,
                broad_epsilon_training=True,
                broad_epsilon_pgd_training=True,
            )
        )


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
    assert hps.policy_adversary_training.mode == POLICY_ADVERSARY_TRAINING_MODE
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


def test_policy_adversary_defaults_do_not_inherit_historical_radius() -> None:
    defaults = build_parser().parse_args([])
    assert defaults.policy_adversary_radius_15cm is None
    assert defaults.policy_adversary_radius_source is None

    hps = build_hps(defaults)
    policy = hps.policy_adversary_training
    assert policy.enabled is False
    assert policy.budget_contract.effective_l2_radius_15cm is None
    assert policy.budget_contract.active_max_l2_radius_15cm is None
    assert policy.budget_contract.budget_source is None


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


def test_policy_adversary_historical_spec_with_explicit_radius_and_source_parses() -> None:
    parsed = PolicyFullStateEpsilonTrainingConfig.from_payload(
        {
            "enabled": True,
            "budget_contract": {
                "effective_l2_radius_15cm": HISTORICAL_020A65B_PGD_RADIUS_15CM,
                "budget_source": {"key": "effective_020a65b_pgd_training_radius"},
            },
        }
    )

    assert parsed.reference_l2_radius == pytest.approx(HISTORICAL_020A65B_PGD_RADIUS_15CM)
    assert parsed.budget_source == "effective_020a65b_pgd_training_radius"


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
    replay_args = resolve_run_spec_args(_args(run_spec=result["run_spec_path"]))

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


def test_policy_adversary_projection_reports_radius_energy_and_boundary() -> None:
    cfg = PolicyFullStateEpsilonTrainingConfig(
        enabled=True,
        epsilon_dim=2,
        state_feature_dim=4,
        reference_l2_radius_15cm=2.0,
        budget_source="unit_test_radius",
        reach_length_scaling=False,
    )
    trial_specs = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((2, 4), dtype=jnp.float32)}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((2, 2, 2), dtype=jnp.float32),
                )
            }
        ),
        inputs={"epsilon": jnp.zeros((2, 2, 2), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=2),
    )
    raw = jnp.asarray(
        [
            [[3.0, 0.0], [4.0, 0.0]],
            [[0.3, 0.4], [0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    radius = _broad_epsilon_l2_radius(trial_specs, cfg)
    projected = _project_flattened_per_trial_l2_ball(raw, radius)
    diagnostics = policy_adversary_projection_diagnostics(
        projected,
        radius,
        mode=POLICY_ADVERSARY_PLAIN_MODE,
    )

    np.testing.assert_allclose(_flattened_per_trial_norm(projected), np.asarray([2.0, 0.5]))
    assert diagnostics["epsilon_norm_radius_ratio_max"] == pytest.approx(1.0)
    assert diagnostics["epsilon_energy_mean"] == pytest.approx((4.0 + 0.25) / 2.0)
    assert diagnostics["boundary_fraction"] == pytest.approx(0.5)


def test_policy_projection_has_finite_gradient_at_zero_start() -> None:
    raw = jnp.zeros((2, 3, 1), dtype=jnp.float32)
    radius = jnp.asarray([1.0, 1.0], dtype=jnp.float32)

    grad = jax.grad(lambda value: jnp.sum(_project_flattened_per_trial_l2_ball(value, radius)))(raw)

    assert jnp.all(jnp.isfinite(grad))


def test_policy_adversary_controller_prestep_detaches_projected_epsilon() -> None:
    class EchoTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            return TreeNamespace(
                mechanics=TreeNamespace(
                    vector=jnp.ones((1, 2, 1), dtype=jnp.float32),
                )
            )

    cfg = PolicyFullStateEpsilonTrainingConfig(
        enabled=True,
        epsilon_dim=1,
        state_feature_dim=1,
        width=2,
        depth=0,
        reference_l2_radius_15cm=10.0,
        budget_source="unit_test_radius",
        reach_length_scaling=False,
    )
    policy = make_memoryless_policy_adversary(cfg, key=jr.PRNGKey(0))
    trial_specs = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((1, 2, 2), dtype=jnp.float32),
                )
            }
        ),
        inputs={"epsilon": jnp.zeros((1, 2, 1), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=2),
    )

    def epsilon_sum(candidate_policy, *, stop_gradient_epsilon: bool):
        updated, _diagnostics = policy_adversary_trial_specs(
            candidate_policy,
            EchoTask(),
            model=None,
            trial_specs=trial_specs,
            keys_model=None,
            config=cfg,
            stop_gradient_epsilon=stop_gradient_epsilon,
        )
        return jnp.sum(updated.inputs["epsilon"])

    attached_grads = eqx.filter_grad(
        lambda candidate_policy: epsilon_sum(
            candidate_policy,
            stop_gradient_epsilon=False,
        )
    )(policy)
    detached_grads = eqx.filter_grad(
        lambda candidate_policy: epsilon_sum(
            candidate_policy,
            stop_gradient_epsilon=True,
        )
    )(policy)

    attached_norm = sum(
        float(jnp.sum(jnp.abs(leaf)))
        for leaf in jt.leaves(eqx.filter(attached_grads, eqx.is_array))
    )
    detached_norm = sum(
        float(jnp.sum(jnp.abs(leaf)))
        for leaf in jt.leaves(eqx.filter(detached_grads, eqx.is_array))
    )

    assert attached_norm > 0.0
    assert detached_norm == pytest.approx(0.0)


def test_delayed_sisu_rejects_overloaded_input_condition_key() -> None:
    with pytest.raises(ValueError, match="go cue and SISU budget key are distinct"):
        build_hps(
            _args(
                delayed_reach=True,
                target_relative_multitarget=True,
                broad_epsilon_pgd_training=True,
                broad_epsilon_pgd_budget_schedule=BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
                broad_epsilon_pgd_sisu_condition_input="input",
                broad_epsilon_pgd_sisu_max_radius=HISTORICAL_020A65B_PGD_RADIUS_15CM,
                broad_epsilon_pgd_sisu_max_radius_source=("effective_020a65b_pgd_training_radius"),
                broad_epsilon_reach_scaling=False,
            )
        )


def test_delayed_sisu_uses_separate_budget_key_and_composite_controller_input() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            batch_size=5,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.0,
            target_relative_multitarget=True,
            force_filter_feedback=True,
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_budget_schedule=BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
            broad_epsilon_pgd_sisu_condition_input="sisu",
            broad_epsilon_pgd_sisu_max_radius=HISTORICAL_020A65B_PGD_RADIUS_15CM,
            broad_epsilon_pgd_sisu_max_radius_source="effective_020a65b_pgd_training_radius",
            broad_epsilon_reach_scaling=False,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    trial = pair.task.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    go_step = int(trial.timeline.epoch_bounds[-2])

    assert pair.model.input_ports[:3] == ("input", "target", "epsilon")
    assert _native_recurrent_input_size(pair.model) == 8
    assert sorted(trial.inputs) == ["epsilon", "input", "sisu", "target", "task"]
    assert trial.inputs["input"].shape[-1] == 2
    assert jnp.allclose(trial.inputs["input"][:go_step, 0], 0.0)
    assert jnp.allclose(trial.inputs["input"][go_step:, 0], 1.0)
    assert jnp.allclose(trial.inputs["input"][..., 1], trial.inputs["sisu"][..., :-1])

    radius = _broad_epsilon_l2_radius(
        trial,
        PgdFullStateEpsilonTrainingConfig.from_payload(hps.broad_epsilon_pgd_training),
    )
    expected_radius = HISTORICAL_020A65B_PGD_RADIUS_15CM * jnp.sqrt(jnp.mean(trial.inputs["sisu"]))
    assert radius == pytest.approx(float(expected_radius))

    spec = build_graph_bundle(hps).task_spec
    summary = build_graph_bundle(hps).manifest["model_structure"]
    assert spec["extra_inputs"] == ["input", "sisu", "target", "epsilon"]
    assert summary["go_cue"]["controller_input_index"] == 0
    assert summary["sisu_conditioning"]["input_key"] == "sisu"
    assert summary["sisu_conditioning"]["controller_input_index"] == 1


def test_delayed_sisu_catch_trials_preserve_hold_targets_with_sisu_present() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=1.0,
            target_relative_multitarget=True,
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_budget_schedule=BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
            broad_epsilon_pgd_sisu_condition_input="sisu",
            broad_epsilon_pgd_sisu_max_radius=HISTORICAL_020A65B_PGD_RADIUS_15CM,
            broad_epsilon_pgd_sisu_max_radius_source="effective_020a65b_pgd_training_radius",
            broad_epsilon_reach_scaling=False,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    base = pair.task.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    sampled = apply_training_target_distribution(
        base,
        hps.target_relative_multitarget,
        jr.PRNGKey(2),
    )

    assert sampled.extra is not None
    assert bool(sampled.extra["is_catch_trial"])
    assert "sisu" in sampled.inputs
    assert jnp.allclose(sampled.inputs["input"][..., 0], 0.0)
    assert jnp.allclose(sampled.inputs["input"][..., 1], sampled.inputs["sisu"][..., :-1])
    assert jnp.any(jnp.abs(sampled.inputs["target"]) > 0.0)
    assert jnp.allclose(
        sampled.targets["mechanics.effector.pos"].value,
        jnp.zeros_like(sampled.targets["mechanics.effector.pos"].value),
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
    assert trial.inputs["input"].dtype == jnp.dtype(jnp.float32)
    assert trial.inputs["epsilon"].shape == (60, CS_EPSILON_DIM)
    assert trial.inputs["epsilon"].dtype == jnp.dtype(jnp.float32)
    assert jnp.allclose(trial.inputs["epsilon"][:, :4], 0.0)
    assert jnp.any(jnp.abs(trial.inputs["epsilon"]) > 0.0)
    assert jnp.allclose(targets, jnp.broadcast_to(jnp.array([0.15, 0.0]), (60, 2)))


def test_lss_process_epsilon_factor_matches_cs_physical_covariance() -> None:
    plant, _schedule = build_canonical_game()
    covariances = default_cs_noise_covariances(plant, OutputFeedbackConfig())
    expected = covariances.process[:CS_EPSILON_DIM, :CS_EPSILON_DIM]
    factor = _cs_lss_process_epsilon_factor(dtype=jnp.float64)

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
    assert derive_spec_path(artifact) == (
        REPO_ROOT / "results" / "30f2313" / "runs" / "cs_stochastic_gru__no_hidden_penalty.json"
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
    assert payload["issue"] == "30f2313"
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


def test_feedbax_training_run_spec_authoring_uses_portable_binding_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError):
        _assert_no_absolute_string_leaves(
            {"method_payload": {"payload": {"config": {"spec_dir": str(tmp_path / "spec")}}}}
        )

    def authored_payload(root: Path) -> dict[str, Any]:
        output_dir = root / "_artifacts" / "81e3d8d" / "runs" / "adaptive"
        spec_dir = root / "results" / "81e3d8d" / "runs" / "adaptive"
        result = write_run_spec(
            _args(
                output_dir=str(output_dir),
                spec_dir=str(spec_dir),
                issue="81e3d8d",
                smoke=True,
                dry_run=True,
                gradient_clip_norm=5.0,
                target_relative_multitarget=True,
                broad_epsilon_pgd_training=True,
                broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                broad_epsilon_pgd_energy_lambda=2.5,
                adaptive_epsilon_curriculum=True,
            )
        )
        return result["run_spec"][FEEDBAX_TRAINING_RUN_SPEC_KEY]

    first = authored_payload(tmp_path / "worktree_a")
    second = authored_payload(tmp_path / "worktree_b")
    method_payload = first["method_payload"]

    _assert_no_absolute_string_leaves(method_payload)
    assert first["artifacts"]["artifact_root"] == "_artifacts/81e3d8d/runs/adaptive"
    assert first["artifacts"]["metadata"]["tracked_spec_dir"] == ("results/81e3d8d/runs/adaptive")
    assert method_payload["payload"]["config"]["output_dir"] == ("_artifacts/81e3d8d/runs/adaptive")
    assert method_payload["payload"]["config"]["spec_dir"] == "results/81e3d8d/runs/adaptive"
    assert first["method_payload"] == second["method_payload"]


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


def test_training_run_spec_graph_guard_rejects_diverging_hps(tmp_path: Path) -> None:
    args = _args(
        output_dir=str(tmp_path / "bulk"),
        spec_dir=str(tmp_path / "spec"),
        smoke=True,
        dry_run=True,
        gradient_clip_norm=5.0,
    )
    payload = write_run_spec(args)["run_spec"]
    diverging_hps = build_hps(_args(hidden_size=5, n_replicates=1))

    with pytest.raises(ValueError, match="Serialized TrainingRunSpec graph"):
        assert_runtime_graph_matches_training_spec(
            payload,
            graph_spec=build_training_run_graph_spec(diverging_hps, seed=int(args.seed)),
        )


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
        assert spec.target.kind == "edge"
        assert spec.payload_shape in ([2], [4])
        adapter_node = f"{spec.label}_additive"
        assert adapter_node in pair.model.nodes
        assert pair.model.input_bindings[spec.input_key] == (adapter_node, "b")
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
    active_input_keys = (
        "epsilon",
        GRAPH_ADAPTER_SPECS["command_input"].input_key,
        GRAPH_ADAPTER_SPECS["sensory_feedback"].input_key,
    )
    input_differs = any(
        bool(jnp.any(first.inputs[key] != second.inputs[key])) for key in active_input_keys
    )
    init_differs = bool(
        jnp.any(first.inits["mechanics.vector"] != second.inits["mechanics.vector"])
    )
    assert input_differs or init_differs
    assert jnp.all(first.inputs[GRAPH_ADAPTER_SPECS["delayed_observation"].input_key] == 0.0)
    assert jnp.all(second.inputs[GRAPH_ADAPTER_SPECS["delayed_observation"].input_key] == 0.0)

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


def _nonzero_pulse_start_offsets(delta: jnp.ndarray, movement_start: jnp.ndarray) -> set[int]:
    active = np.asarray(jnp.any(delta != 0.0, axis=-1).reshape((-1, delta.shape[-2])))
    starts = np.asarray(movement_start).reshape((-1,))
    offsets = set()
    for row, start in zip(active, starts, strict=True):
        if np.any(row):
            offsets.add(int(np.flatnonzero(row)[0]) - int(start))
    return offsets


def _assert_no_prep_pulse_support(delta: jnp.ndarray, movement_start: jnp.ndarray) -> None:
    active = np.asarray(jnp.any(delta != 0.0, axis=-1).reshape((-1, delta.shape[-2])))
    starts = np.asarray(movement_start).reshape((-1,))
    for row, start in zip(active, starts, strict=True):
        assert not np.any(row[: int(start)])


def _manual_movement_age_trial(go_steps: jnp.ndarray, *, n_steps: int = 90) -> TaskTrialSpec:
    go_steps = jnp.asarray(go_steps, dtype=jnp.int32)
    batch = int(go_steps.shape[0])
    target = jnp.broadcast_to(
        jnp.asarray([0.15, 0.0], dtype=jnp.float32),
        (batch, n_steps, 2),
    )
    epoch_bounds = jnp.stack(
        [
            jnp.zeros_like(go_steps),
            go_steps,
            jnp.full_like(go_steps, n_steps),
        ],
        axis=-1,
    )
    return TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((batch, 8), dtype=jnp.float32)}),
        targets=WhereDict({"mechanics.effector.pos": TargetSpec(value=target)}),
        inputs={"epsilon": jnp.zeros((batch, n_steps, 8), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=n_steps, epoch_bounds=epoch_bounds),
    )


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
    assert not _nonzero_pulse_starts(delayed)
    assert _max_nonzero_pulse_width(sensory) <= 5
    assert hps.perturbation_training.mode == CALIBRATED_TIMING_PERTURBATION_TRAINING_MODE


def test_calibrated_movement_age_timing_preserves_undelayed_starts() -> None:
    hps = build_hps(
        _args(
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_movement_age_timing=True,
            perturbation_physical_level="small",
            batch_size=512,
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

    command = sampled.inputs[GRAPH_ADAPTER_SPECS["command_input"].input_key]
    sensory = sampled.inputs[GRAPH_ADAPTER_SPECS["sensory_feedback"].input_key]
    delayed = sampled.inputs[GRAPH_ADAPTER_SPECS["delayed_observation"].input_key]

    assert hps.perturbation_training.timing_basis.mode == "movement_age"
    assert _nonzero_pulse_starts(command).issubset({5, 15, 35})
    assert _nonzero_pulse_starts(sensory).issubset({10, 20, 40})
    assert not _nonzero_pulse_starts(delayed)


def test_calibrated_movement_age_timing_shifts_by_delayed_go_cue() -> None:
    go_steps = jnp.tile(jnp.arange(10, 31, dtype=jnp.int32), 32)
    base = _manual_movement_age_trial(go_steps)
    hps = build_hps(
        _args(
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_movement_age_timing=True,
            perturbation_physical_level="small",
            target_relative_multitarget=True,
            delayed_reach=True,
            batch_size=int(go_steps.shape[0]),
            hidden_size=4,
            n_replicates=1,
        )
    )
    sampled = apply_training_perturbation_mixture(
        base,
        hps.perturbation_training,
        jr.PRNGKey(2),
    )

    process_delta = sampled.inputs["epsilon"] - base.inputs["epsilon"]
    command = sampled.inputs[GRAPH_ADAPTER_SPECS["command_input"].input_key]
    sensory = sampled.inputs[GRAPH_ADAPTER_SPECS["sensory_feedback"].input_key]
    delayed = sampled.inputs[GRAPH_ADAPTER_SPECS["delayed_observation"].input_key]

    assert _nonzero_pulse_start_offsets(process_delta, go_steps).issubset({0, 5, 15, 35})
    assert _nonzero_pulse_start_offsets(command, go_steps).issubset({5, 15, 35})
    assert _nonzero_pulse_start_offsets(sensory, go_steps).issubset({10, 20, 40})
    assert not _nonzero_pulse_start_offsets(delayed, go_steps)
    _assert_no_prep_pulse_support(process_delta, go_steps)
    _assert_no_prep_pulse_support(command, go_steps)
    _assert_no_prep_pulse_support(sensory, go_steps)
    _assert_no_prep_pulse_support(delayed, go_steps)


def test_movement_age_initial_offsets_are_movement_onset_process_impulses() -> None:
    go_steps = jnp.asarray([10, 20, 30], dtype=jnp.int32)
    base = _manual_movement_age_trial(go_steps)
    hps = build_hps(
        _args(
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_movement_age_timing=True,
            perturbation_physical_level="small",
            target_relative_multitarget=True,
            delayed_reach=True,
            batch_size=int(go_steps.shape[0]),
            hidden_size=4,
            n_replicates=1,
        )
    )

    shifted = apply_validation_bin(base, hps.perturbation_training, "initial_position")
    delta = shifted.inputs["epsilon"] - base.inputs["epsilon"]

    assert jnp.allclose(shifted.inits["mechanics.vector"], base.inits["mechanics.vector"])
    assert _nonzero_pulse_start_offsets(delta, go_steps) == {0}
    _assert_no_prep_pulse_support(delta, go_steps)


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
            / load_open_loop_calibration()["initial_velocity_offset"]["initial_condition"]
        },
    )

    process_bin = apply_validation_bin(base, hps.perturbation_training, "process_epsilon")
    process_delta = process_bin.inputs["epsilon"] - base.inputs["epsilon"]
    process_expected = {
        target_peak_delta_x
        / load_open_loop_calibration()["process_epsilon_force_state_xy"]["early"]
    }
    _assert_values_close_to_expected(
        _unique_abs_nonzero(process_delta),
        process_expected,
    )

    command_bin = apply_validation_bin(base, hps.perturbation_training, "command_input")
    command = command_bin.inputs[GRAPH_ADAPTER_SPECS["command_input"].input_key]
    command_full = {
        target_peak_delta_x / load_open_loop_calibration()["command_input_pulse"]["early"]
    }
    command_xy = np.asarray(command[..., 5, :2])
    command_norm = np.linalg.norm(command_xy, axis=-1)
    nonzero_norm = command_norm[command_norm > 1e-7]
    _assert_values_close_to_expected(np.unique(np.round(nonzero_norm, 8)), command_full)

    sensory_expected = {
        target_peak_delta_x,
    }
    sensory_bin = apply_validation_bin(base, hps.perturbation_training, "sensory_feedback")
    _assert_values_close_to_expected(
        _unique_abs_nonzero(sensory_bin.inputs[GRAPH_ADAPTER_SPECS["sensory_feedback"].input_key]),
        sensory_expected,
    )
    assert "delayed_observation" not in VALIDATION_BINS
    assert "delayed_observation" not in hps.perturbation_training.single_family_bins
    delayed_bin = apply_validation_bin(base, hps.perturbation_training, "delayed_observation")
    _assert_values_close_to_expected(
        _unique_abs_nonzero(
            delayed_bin.inputs[GRAPH_ADAPTER_SPECS["delayed_observation"].input_key]
        ),
        sensory_expected,
    )


def test_command_input_training_sampler_uses_random_2d_vector_norm() -> None:
    pulse = _command_input_direction_pulse(
        batch_shape=(256,),
        n_steps=20,
        width=2,
        amount=jnp.ones((256,), dtype=jnp.float32),
        duration=5,
        start=5,
        key=jr.PRNGKey(2),
        dtype=jnp.float32,
    )
    pulse_xy = np.asarray(pulse[:, 5, :2])
    norms = np.linalg.norm(pulse_xy, axis=-1)

    assert np.all(np.count_nonzero(np.abs(pulse_xy) > 1e-7, axis=-1) == 2)
    assert np.allclose(norms, 1.0, rtol=5e-5, atol=5e-7)
    assert np.allclose(np.asarray(pulse[:, :5, :]), 0.0)
    assert np.allclose(np.asarray(pulse[:, 10:, :]), 0.0)


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


def test_target_aligned_lateral_load_uses_trial_reach_direction() -> None:
    trial = _manual_movement_age_trial(jnp.asarray([5, 5]), n_steps=20)
    target = jnp.asarray(
        [
            [[0.15, 0.0]] * 20,
            [[0.0, 0.15]] * 20,
        ],
        dtype=jnp.float32,
    )
    trial = eqx.tree_at(
        lambda ts: ts.targets["mechanics.effector.pos"].value,
        trial,
        target,
    )
    pulse = _target_aligned_lateral_direction_pulse(
        trial,
        batch_shape=(2,),
        n_steps=20,
        width=2,
        amount=jnp.ones((2,), dtype=jnp.float32),
        duration=5,
        start=5,
        dtype=jnp.float32,
    )

    assert np.allclose(np.asarray(pulse[0, 5, :2]), [0.0, 1.0])
    assert np.allclose(np.asarray(pulse[1, 5, :2]), [-1.0, 0.0])
    assert np.allclose(np.asarray(pulse[:, :5, :]), 0.0)
    assert np.allclose(np.asarray(pulse[:, 10:, :]), 0.0)


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

    parser = build_parser()
    replay_args = resolve_run_spec_args(
        parser.parse_args(["--run-spec", movement_result["run_spec_path"]]),
        parser=parser,
    )
    assert replay_args.perturbation_movement_age_timing is True
    assert replay_args.output_dir == str(tmp_path / "movement_bulk")
    assert replay_args.spec_dir == str(tmp_path / "movement_spec")

    with pytest.raises(ValueError, match="artifact route"):
        resolve_run_spec_args(
            parser.parse_args(
                [
                    "--run-spec",
                    movement_result["run_spec_path"],
                    "--output-dir",
                    str(tmp_path / "replay_bulk"),
                    "--spec-dir",
                    str(tmp_path / "replay_spec"),
                ]
            ),
            parser=parser,
        )


def test_target_relative_feedback_sign_contract() -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=4,
        target_relative_feedback=True,
        bind_epsilon_input=True,
        key=jr.PRNGKey(0),
    )
    component = StateFeedbackSelector(**spec.nodes["feedback"].params)
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
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=4,
        target_relative_feedback=True,
        bind_epsilon_input=True,
        key=jr.PRNGKey(0),
    )
    component = StateFeedbackSelector(**spec.nodes["feedback"].params)
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
    assert (
        hps.target_relative_multitarget.target_distribution.target_support_profile
        == TARGET_SUPPORT_PROFILE_CONST_BAND16
    )
    assert len(hps.target_relative_multitarget.target_distribution.seen_targets_m) == 56
    assert len(hps.target_relative_multitarget.target_distribution.held_out_targets_m) == 16
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
    assert all(row["bin"] != "delayed_observation_offsets" for row in manifest["bins"])
    assert all("delayed_observation" not in row.get("families", ()) for row in manifest["bins"])
    perturbation_bins = [
        row for row in manifest["bins"] if row["target_role"] == "seen_and_held_out_static_targets"
    ]
    assert perturbation_bins
    for row in perturbation_bins:
        assert row["targets_m"]
        assert row["targets_m"] == perturbation_bins[0]["targets_m"]
    assert perturbation_bins[0]["targets_m"] != manifest["bins"][0]["targets_m"]
    assert jnp.any(trial.inputs["target"][..., -1, :] != jnp.array([0.15, 0.0]))


def test_sisu_conditioned_input_does_not_trigger_catch_target_rewrite() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            batch_size=5,
            target_relative_multitarget=True,
            force_filter_feedback=True,
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_budget_schedule=BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
            broad_epsilon_pgd_sisu_condition_input="input",
            broad_epsilon_pgd_sisu_max_radius=HISTORICAL_020A65B_PGD_RADIUS_15CM,
            broad_epsilon_pgd_sisu_max_radius_source="effective_020a65b_pgd_training_radius",
            broad_epsilon_reach_scaling=False,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    base = pair.task.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))

    assert hps.task.p_catch_trial == pytest.approx(0.0)
    assert pair.model.input_ports[:3] == ("input", "target", "epsilon")
    assert sorted(base.inputs) == ["effector_target", "epsilon", "input"]

    for i, sisu in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        trial = _set_input(base, "input", jnp.full_like(base.inputs["input"], sisu))
        retargeted = apply_training_target_distribution(
            trial,
            hps.target_relative_multitarget,
            jr.PRNGKey(10 + i),
        )
        scored_target = retargeted.targets["mechanics.effector.pos"].value
        visible_target = retargeted.inputs["target"]

        assert jnp.allclose(scored_target, visible_target)
        assert jnp.any(jnp.abs(scored_target) > 0.0)


def test_generic_input_without_go_cue_role_does_not_preserve_catch_target() -> None:
    trial = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((8,), dtype=jnp.float32)}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((3, 2), dtype=jnp.float32),
                )
            }
        ),
        inputs={"input": jnp.zeros((3,), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=3),
    )
    config = TargetRelativeMultiTargetTrainingConfig(
        enabled=True,
        seen_directions_deg=(0.0,),
        held_out_directions_deg=(90.0,),
        seen_amplitudes_m=(0.15,),
        held_out_amplitudes_m=(0.12,),
        original_target_anchor_m=(0.15, 0.0),
    )

    retargeted = apply_training_target_distribution(trial, config, jr.PRNGKey(1))

    assert jnp.allclose(
        retargeted.targets["mechanics.effector.pos"].value,
        retargeted.inputs["target"],
    )
    assert jnp.any(jnp.abs(retargeted.targets["mechanics.effector.pos"].value) > 0.0)


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
    assert _native_recurrent_input_size(pair.model) == 7
    assert trial.timeline.epoch_names == ("prep", "movement")
    assert 10 <= go_step <= 30
    assert trial.inputs["input"].shape == (trial.timeline.n_steps - 1,)
    assert jnp.allclose(trial.inputs["input"][:go_step], 0.0)
    assert jnp.allclose(trial.inputs["input"][go_step:], 1.0)
    assert trial.inputs["target"].shape[-2:] == (90, 2)
    assert jnp.allclose(
        trial.inputs["target"],
        jnp.broadcast_to(trial.inputs["target"][..., :1, :], trial.inputs["target"].shape),
    )
    assert trial.inputs["epsilon"].shape == (trial.timeline.n_steps - 1, CS_EPSILON_DIM)

    validation = pair.task.validation_trials
    validation_targets = validation.targets["mechanics.effector.pos"].value
    assert validation.inputs["task"].effector_target.pos.shape == validation_targets.shape
    assert validation.inputs["task"].hold.shape[:2] == (
        validation_targets.shape[0],
        validation.timeline.n_steps - 1,
    )
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
    assert trial.inputs["epsilon"].shape == (trial.timeline.n_steps - 1, CS_REDUCED_EPSILON_DIM)
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


def test_delayed_reach_full_qrf_default_keeps_unsplit_objective() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.5,
            target_relative_multitarget=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            nn_output_pre_go=1.0,
            hidden_size=8,
            n_replicates=1,
        )
    )
    loss_func = get_reach_loss(hps)

    assert hps.loss.delayed_trial_type_normalization.enabled is False
    assert set(loss_func.terms) == {
        CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        "nn_output_pre_go",
    }
    assert isinstance(
        loss_func.terms[CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE],
        CsAnalyticalQrfLoss,
    )


def test_delayed_reach_full_qrf_can_split_trial_type_normalized_terms() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.5,
            delayed_reach_trial_type_normalized_loss=True,
            delayed_reach_no_catch_qrf_weight=2.0,
            delayed_reach_catch_qrf_weight=3.0,
            target_relative_multitarget=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            nn_output_pre_go=1.0,
            hidden_size=8,
            n_replicates=1,
        )
    )
    loss_func = get_reach_loss(hps)

    assert set(loss_func.terms) == {
        f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_no_catch",
        f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_catch",
        "nn_output_pre_go",
    }
    assert isinstance(
        loss_func.terms[f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_no_catch"],
        DelayedReachTrialTypeNormalizedLoss,
    )
    assert isinstance(
        loss_func.terms[f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_catch"],
        DelayedReachTrialTypeNormalizedLoss,
    )
    assert loss_func.weights == {
        f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_no_catch": 2.0,
        f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_catch": 3.0,
        "nn_output_pre_go": 1.0,
    }
    assert hps.loss.delayed_trial_type_normalization.enabled is True
    assert hps.loss.delayed_trial_type_normalization.no_catch_weight == pytest.approx(2.0)
    assert hps.loss.delayed_trial_type_normalization.catch_weight == pytest.approx(3.0)


def test_delayed_reach_pre_go_hold_penalty_defaults_are_zero() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.5,
            target_relative_multitarget=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            hidden_size=8,
            n_replicates=1,
        )
    )
    loss_func = get_reach_loss(hps)

    assert hps.loss.weights.delayed_pre_go_force_filter_hold == pytest.approx(0.0)
    assert hps.loss.weights.delayed_pre_go_start_pos_hold == pytest.approx(0.0)
    assert hps.loss.weights.delayed_pre_go_zero_vel_hold == pytest.approx(0.0)
    assert "delayed_pre_go_force_filter_hold" not in loss_func.terms
    assert "delayed_pre_go_start_pos_hold" not in loss_func.terms
    assert "delayed_pre_go_zero_vel_hold" not in loss_func.terms


def test_delayed_reach_pre_go_hold_penalty_args_build_hps_and_run_spec(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--smoke",
            "--output-dir",
            str(tmp_path / "bulk"),
            "--spec-dir",
            str(tmp_path / "spec"),
            "--delayed-reach",
            "--target-relative-multitarget",
            "--loss-objective",
            CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            "--nn-output-pre-go",
            "0",
            "--delayed-pre-go-force-filter-hold",
            "11",
            "--delayed-pre-go-start-pos-hold",
            "22",
            "--delayed-pre-go-start-pos-hold-norm",
            "l1",
            "--delayed-pre-go-zero-vel-hold",
            "33",
        ]
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


def test_delayed_reach_trial_type_normalized_loss_reduces_over_extra_support() -> None:
    trial = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict({}),
        inputs={},
        timeline=TrialTimeline(1),
        extra={"is_catch_trial": jnp.array([False, False, True, True])},
    )
    values = jnp.array([10.0, 20.0, 100.0, 300.0])
    no_catch = DelayedReachTrialTypeNormalizedLoss(
        base_loss=_StaticPerTrialLoss(values),
        trial_type="no_catch",
        label="no_catch",
    )
    catch = DelayedReachTrialTypeNormalizedLoss(
        base_loss=_StaticPerTrialLoss(values),
        trial_type="catch",
        label="catch",
    )

    no_catch_values = no_catch.term(None, trial, None)
    catch_values = catch.term(None, trial, None)

    assert jnp.mean(no_catch_values) == pytest.approx(15.0)
    assert jnp.mean(catch_values) == pytest.approx(200.0)
    assert jnp.allclose(no_catch_values, jnp.asarray([20.0, 40.0, 0.0, 0.0]))
    assert jnp.allclose(catch_values, jnp.asarray([0.0, 0.0, 200.0, 600.0]))


def test_delayed_reach_trial_type_normalized_loss_falls_back_to_hold_support() -> None:
    hold = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    trial = TaskTrialSpec(
        inits={},
        inputs={"task": TreeNamespace(hold=hold)},
        targets={},
        intervene={},
        timeline=TrialTimeline.from_epochs_events(
            n_steps=4,
            epoch_bounds=jnp.asarray([[0, 2, 4]] * 4),
            epoch_names=("prep", "movement"),
        ),
    )
    catch = DelayedReachTrialTypeNormalizedLoss(
        base_loss=_StaticPerTrialLoss([2.0, 4.0, 10.0, 20.0]),
        trial_type="catch",
        label="catch",
    )
    no_catch = DelayedReachTrialTypeNormalizedLoss(
        base_loss=_StaticPerTrialLoss([2.0, 4.0, 10.0, 20.0]),
        trial_type="no_catch",
        label="no_catch",
    )

    assert jnp.mean(catch.term(None, trial, None)) == pytest.approx(3.0)
    assert jnp.mean(no_catch.term(None, trial, None)) == pytest.approx(15.0)


def test_delayed_reach_full_qrf_pre_go_auxiliary_masks_only_prep_epoch() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.5,
            target_relative_multitarget=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            nn_output_pre_go=1.0,
            hidden_size=8,
            n_replicates=1,
        )
    )
    loss_func = get_reach_loss(hps)
    go_step = 12
    trial = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict({}),
        inputs={},
        timeline=TrialTimeline.from_epochs_events(
            n_steps=90,
            epoch_bounds=jnp.asarray([0, go_step, go_step + 60]),
            epoch_names=("prep", "movement"),
        ),
        extra={"is_catch_trial": jnp.asarray([False])},
    )
    pre_go_loss = loss_func.terms["nn_output_pre_go"]
    zeros = jnp.zeros((1, 90, 48), dtype=jnp.float64)
    zero_command = jnp.zeros((1, 90, 2), dtype=jnp.float64)

    def states_with_efferent(command):
        return TreeNamespace(
            mechanics=TreeNamespace(vector=zeros),
            net=TreeNamespace(output=zero_command),
            efferent=TreeNamespace(output=command),
        )

    base_states = states_with_efferent(zero_command)
    prep_states = states_with_efferent(zero_command.at[:, max(go_step - 1, 1), 0].set(3.0))
    movement_states = states_with_efferent(zero_command.at[:, go_step, 0].set(3.0))
    base_value = pre_go_loss(base_states, trial, None).total
    prep_value = pre_go_loss(prep_states, trial, None).total
    movement_value = pre_go_loss(movement_states, trial, None).total

    assert loss_func.weights["nn_output_pre_go"] == pytest.approx(1.0)
    assert pre_go_loss.epoch_indices == (0,)
    assert trial.timeline.epoch_names == ("prep", "movement")
    assert prep_value > base_value
    assert jnp.allclose(movement_value, base_value)


def test_delayed_reach_full_qrf_pre_go_hold_auxiliaries_mask_only_prep_epoch() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.5,
            target_relative_multitarget=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            nn_output_pre_go=0.0,
            delayed_pre_go_force_filter_hold=1.0,
            delayed_pre_go_start_pos_hold=1.0,
            delayed_pre_go_zero_vel_hold=1.0,
            hidden_size=8,
            n_replicates=1,
        )
    )
    loss_func = get_reach_loss(hps)
    go_step = 12
    initial_vector = (
        jnp.zeros((1, 48), dtype=jnp.float64)
        .at[:, :2]
        .set(jnp.asarray([[1.0, -2.0]], dtype=jnp.float64))
    )
    trial = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": initial_vector}),
        targets=WhereDict({}),
        inputs={},
        timeline=TrialTimeline.from_epochs_events(
            n_steps=90,
            epoch_bounds=jnp.asarray([0, go_step, go_step + 60]),
            epoch_names=("prep", "movement"),
        ),
        extra={"is_catch_trial": jnp.asarray([False])},
    )
    base_vector = jnp.zeros((1, 90, 48), dtype=jnp.float64)
    base_pos = jnp.broadcast_to(initial_vector[:, None, :2], (1, 90, 2))
    base_vel = jnp.zeros((1, 90, 2), dtype=jnp.float64)

    def states_with(*, vector=base_vector, pos=base_pos, vel=base_vel):
        return TreeNamespace(
            mechanics=TreeNamespace(
                vector=vector,
                effector=TreeNamespace(pos=pos, vel=vel),
            ),
            net=TreeNamespace(output=jnp.zeros((1, 90, 2), dtype=jnp.float64)),
        )

    cases = {
        "delayed_pre_go_force_filter_hold": (
            states_with(vector=base_vector.at[:, max(go_step - 1, 1), 4].set(3.0)),
            states_with(vector=base_vector.at[:, go_step, 4].set(3.0)),
        ),
        "delayed_pre_go_start_pos_hold": (
            states_with(pos=base_pos.at[:, max(go_step - 1, 1), 0].add(3.0)),
            states_with(pos=base_pos.at[:, go_step, 0].add(3.0)),
        ),
        "delayed_pre_go_zero_vel_hold": (
            states_with(vel=base_vel.at[:, max(go_step - 1, 1), 0].set(3.0)),
            states_with(vel=base_vel.at[:, go_step, 0].set(3.0)),
        ),
    }

    assert loss_func.weights["delayed_pre_go_force_filter_hold"] == pytest.approx(1.0)
    assert loss_func.weights["delayed_pre_go_start_pos_hold"] == pytest.approx(1.0)
    assert loss_func.weights["delayed_pre_go_zero_vel_hold"] == pytest.approx(1.0)
    for name, (prep_states, movement_states) in cases.items():
        term = loss_func.terms[name]
        base_value = term.term(states_with(), trial, None)
        prep_value = term.term(prep_states, trial, None)
        movement_value = term.term(movement_states, trial, None)
        assert term.epoch_indices == (0,)
        assert prep_value > base_value
        assert jnp.allclose(movement_value, base_value)


def test_delayed_reach_start_pos_hold_supports_l1_norm() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            delayed_reach=True,
            delayed_reach_p_catch_trial=0.5,
            target_relative_multitarget=True,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            nn_output_pre_go=0.0,
            delayed_pre_go_start_pos_hold=1.0,
            delayed_pre_go_start_pos_hold_norm="l1",
            hidden_size=8,
            n_replicates=1,
        )
    )
    loss_func = get_reach_loss(hps)
    initial_vector = jnp.zeros((1, 48), dtype=jnp.float64)
    trial = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": initial_vector}),
        targets=WhereDict({}),
        inputs={},
        timeline=TrialTimeline.from_epochs_events(
            n_steps=4,
            epoch_bounds=jnp.asarray([0, 2, 4]),
            epoch_names=("prep", "movement"),
        ),
        extra={"is_catch_trial": jnp.asarray([False])},
    )
    pos = (
        jnp.zeros((1, 4, 2), dtype=jnp.float64)
        .at[:, 1, :]
        .set(jnp.asarray([[3.0, -4.0]], dtype=jnp.float64))
    )
    states = TreeNamespace(
        mechanics=TreeNamespace(
            vector=jnp.zeros((1, 4, 48), dtype=jnp.float64),
            effector=TreeNamespace(pos=pos, vel=jnp.zeros((1, 4, 2), dtype=jnp.float64)),
        ),
        net=TreeNamespace(output=jnp.zeros((1, 4, 2), dtype=jnp.float64)),
    )
    term = loss_func.terms["delayed_pre_go_start_pos_hold"]

    assert term.norm == "l1"
    assert term.term(states, trial, None) == pytest.approx(jnp.asarray([7.0]))


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
    assert payload["model_summary"]["controller_input_dimension"] == 7
    assert payload["hps"]["loss"]["weights"]["nn_output_pre_go"] == 1.0
    assert payload["hps"]["broad_epsilon_pgd_training"]["movement_epoch_only"] is True
    assert payload["hps"]["broad_epsilon_pgd_training"]["time_mask"]["mode"] == (
        "movement_epoch_only"
    )
    assert payload["loss_summary"]["time_indexing"]["stage_schedule"] == (
        "movement_age_from_go_cue"
    )
    assert DELAYED_REACH_TRAINING_MODE in payload["training_summary"]["training_mode"]


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


def test_target_relative_proprioceptive_feedback_extends_sign_contract() -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=4,
        target_relative_feedback=True,
        force_filter_feedback=True,
        bind_epsilon_input=True,
        key=jr.PRNGKey(0),
    )
    component = StateFeedbackSelector(**spec.nodes["feedback"].params)
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
    assert _native_recurrent_input_size(pair.model) == 6
    assert pair.model.nodes["sensory"].input_proto.shape[-1] == 6


def test_force_filter_perturbation_adapters_use_six_dimensional_feedback_payloads() -> None:
    hps = build_hps(
        _args(
            smoke=True,
            target_relative_multitarget=True,
            force_filter_feedback=True,
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_physical_level="small",
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            hidden_size=8,
            n_replicates=1,
        )
    )
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    specs = graph_adapter_specs(force_filter_feedback=True)
    trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(1))
    validation = pair.task.validation_trials

    assert hps.perturbation_training.force_filter_feedback is True
    for bin_name in ("sensory_feedback", "delayed_observation"):
        spec = specs[bin_name]
        adapter_node = f"{spec.label}_additive"
        assert spec.input_key in pair.model.input_ports
        assert pair.model.input_bindings[spec.input_key] == (adapter_node, "b")
        assert pair.model.nodes[adapter_node].__class__.__name__ == "Sum"
        assert trial.inputs[spec.input_key].shape[-1] == 6
        assert jnp.all(trial.inputs[spec.input_key][..., 4:] == 0.0)
        assert validation.inputs[spec.input_key].shape[-1] == 6


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

    parser = build_parser()
    replay_args = resolve_run_spec_args(
        parser.parse_args(
            [
                "--run-spec",
                result["run_spec_path"],
                "--stop-after-batches",
                "1000",
            ]
        ),
        parser=parser,
    )

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


def _native_method_run_spec_payload(tmp_path: Path, method_ref: str) -> dict[str, Any]:
    common = {
        "output_dir": str(tmp_path / method_ref.replace("/", "_") / "artifacts"),
        "spec_dir": str(tmp_path / method_ref.replace("/", "_") / "spec"),
        "dry_run": True,
    }
    if method_ref == CS_SUPERVISED_METHOD_REF:
        return write_run_spec(
            _args(
                **common,
                target_relative_multitarget=True,
                perturbation_training=True,
                perturbation_calibrated_timing=True,
                perturbation_physical_level="small",
            )
        )["run_spec"]
    if method_ref == ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF:
        return write_run_spec(
            _args(
                **common,
                target_relative_multitarget=True,
                broad_epsilon_pgd_training=True,
                broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                broad_epsilon_pgd_energy_lambda=2.5,
                adaptive_epsilon_curriculum=True,
                adaptive_epsilon_controller_training_mode=(
                    ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
                ),
                adaptive_epsilon_damage_peak=3500.0,
                adaptive_epsilon_damage_final=1000.0,
                adaptive_epsilon_damage_ramp_batches=1,
                adaptive_epsilon_damage_anneal_batches=2,
                adaptive_epsilon_update_interval_batches=3,
                adaptive_epsilon_ema_alpha=0.2,
                adaptive_epsilon_eta=0.3,
                adaptive_epsilon_deadband_frac=0.4,
                adaptive_epsilon_lambda_min=1e-9,
                adaptive_epsilon_max_log_step=0.5,
                adaptive_epsilon_outer_weight_ramp_batches=6,
            )
        )["run_spec"]
    if method_ref == POLICY_ADVERSARY_SUPERVISED_METHOD_REF:
        return write_run_spec(
            _args(
                **common,
                target_relative_multitarget=True,
                force_filter_feedback=True,
                initial_hidden_encoder=True,
                perturbation_training=True,
                perturbation_calibrated_timing=True,
                perturbation_physical_level="small",
                policy_adversary_training=True,
                policy_adversary_mode=POLICY_ADVERSARY_PLAIN_MODE,
                policy_adversary_steps=5,
                policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
                policy_adversary_radius_source="effective_020a65b_pgd_training_radius",
                n_train_batches=12000,
                stop_after_batches=1000,
                loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            )
        )["run_spec"]
    raise AssertionError(f"unsupported native method reference: {method_ref}")


@pytest.mark.parametrize(
    "method_ref",
    (
        CS_SUPERVISED_METHOD_REF,
        ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
        POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
    ),
)
def test_generic_run_spec_loader_registers_native_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_ref: str,
) -> None:
    payload = _native_method_run_spec_payload(tmp_path, method_ref)
    recipe_path = tmp_path / "runs" / f"{method_ref.split('/')[1]}.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    _remove_training_method_registration(monkeypatch, method_ref)

    _path, loaded = cs_supervised_executor.load_validated_run_spec(recipe_path)

    assert loaded[FEEDBAX_TRAINING_RUN_SPEC_KEY]["method_ref"] == {
        "package": "rlrmp",
        "name": method_ref.split("/")[1],
        "version": "v1",
    }
    assert method_ref in DEFAULT_TRAINING_METHOD_REGISTRY.available_keys()


def _compact_run_spec(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "compact_run_spec": True,
        "game_card": payload["game_card"],
        "training_distribution": {
            "perturbation_training": payload["training_distribution"]["perturbation_training"],
        },
        "feedbax_graph": payload["feedbax_graph"],
        FEEDBAX_TRAINING_RUN_SPEC_KEY: payload[FEEDBAX_TRAINING_RUN_SPEC_KEY],
        RLRMP_RUN_SPEC_PAYLOAD_KEY: payload[RLRMP_RUN_SPEC_PAYLOAD_KEY],
    }


def _write_flat_recipe_sidecars(recipe_path: Path, payload: dict[str, Any]) -> None:
    sidecar_dir = recipe_path.with_suffix("")
    sidecar_dir.mkdir(parents=True)
    graph = payload["feedbax_graph"]
    for key in ("graph_spec_path", "manifest_path"):
        pointer = graph[key]
        if pointer is None:
            continue
        sidecar_path = sidecar_dir / pointer
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_payload = (
            {
                "nodes": {
                    "mechanics": {"type": "LinearStateSpace"},
                    "feedback": {"type": "StateFeedbackSelector"},
                }
            }
            if key == "graph_spec_path"
            else {}
        )
        sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")


def test_generic_loader_resolves_flat_recipe_sibling_sidecars(tmp_path: Path) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    recipe_path = tmp_path / "runs" / "baseline.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_flat_recipe_sidecars(recipe_path, payload)

    _path, loaded = cs_supervised_executor.load_validated_run_spec(
        recipe_path,
        require_graph_sidecars=True,
    )

    assert loaded["feedbax_graph"] == payload["feedbax_graph"]


def test_public_flat_recipe_validator_hydrates_compact_envelope(tmp_path: Path) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    recipe_path = tmp_path / "runs" / "baseline.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(_compact_run_spec(payload)), encoding="utf-8")
    _write_flat_recipe_sidecars(recipe_path, payload)

    validate_nominal_gru_run_spec_file(recipe_path)


def test_compact_run_spec_loader_hydrates_authoritative_extension(tmp_path: Path) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    compact = _compact_run_spec(payload)
    recipe_path = tmp_path / "runs" / "baseline.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(compact), encoding="utf-8")

    _path, hydrated = cs_supervised_executor.load_validated_run_spec(recipe_path)

    assert "compact_run_spec" not in hydrated
    assert hydrated["hps"] == payload["hps"]
    assert hydrated["game_card"] == payload["game_card"]
    assert hydrated["training_distribution"] == payload["training_distribution"]
    assert hydrated["feedbax_graph"] == payload["feedbax_graph"]
    assert hydrated[FEEDBAX_TRAINING_RUN_SPEC_KEY] == payload[FEEDBAX_TRAINING_RUN_SPEC_KEY]


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda payload: payload.__setitem__("compact_run_spec", False), "boolean true"),
        (lambda payload: payload.pop(RLRMP_RUN_SPEC_PAYLOAD_KEY), "rlrmp_run_spec"),
        (lambda payload: payload.__setitem__(RLRMP_RUN_SPEC_PAYLOAD_KEY, []), "rlrmp_run_spec"),
        (
            lambda payload: payload.__setitem__("game_card", {"issue_id": "mismatched-identity"}),
            "game_card",
        ),
    ),
)
def test_compact_run_spec_loader_rejects_missing_or_mismatched_extension(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    compact = _compact_run_spec(payload)
    mutation(compact)

    with pytest.raises(ValueError, match=match):
        hydrate_compact_run_spec_envelope(compact)


def test_run_spec_replay_rejects_artifact_route_override(tmp_path: Path) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "historical_artifacts"),
            spec_dir=str(tmp_path / "historical_spec"),
            full_train=True,
        )
    )
    parser = build_parser()

    with pytest.raises(ValueError, match="artifact route"):
        resolve_run_spec_args(
            parser.parse_args(
                [
                    "--run-spec",
                    result["run_spec_path"],
                    "--output-dir",
                    str(tmp_path / "current_artifacts"),
                ]
            ),
            parser=parser,
        )


def test_run_spec_replay_rejects_scientific_payload_override(tmp_path: Path) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "historical_artifacts"),
            spec_dir=str(tmp_path / "historical_spec"),
            full_train=True,
        )
    )
    parser = build_parser()

    with pytest.raises(ValueError, match="scientific payload"):
        resolve_run_spec_args(
            parser.parse_args(
                [
                    "--run-spec",
                    result["run_spec_path"],
                    "--n-train-batches",
                    "8",
                ]
            ),
            parser=parser,
        )


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

    parser = build_parser()
    replay_args = resolve_run_spec_args(
        parser.parse_args(["--run-spec", str(flat_run_spec)]),
        parser=parser,
    )

    assert replay_args.output_dir == str(tmp_path / "historical_artifacts")
    assert replay_args.spec_dir == str(tmp_path / "historical_spec")

    with pytest.raises(ValueError, match="scientific payload"):
        resolve_run_spec_args(
            parser.parse_args(
                [
                    "--run-spec",
                    str(flat_run_spec),
                    "--broad-epsilon-pgd-training",
                    "--broad-epsilon-budget-scale",
                    "3.688240371719434",
                    "--broad-epsilon-pgd-steps",
                    "10",
                ]
            ),
            parser=parser,
        )


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

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["JAX_ENABLE_X64"] = "False"

    replay_code = "\n".join(
        [
            "from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods",
            "from rlrmp.train.cs_nominal_gru import main",
            "register_rlrmp_training_methods()",
            (
                "raise SystemExit(main(['--run-spec', "
                f"{json.dumps(str(flat_run_spec))}, '--dry-run']))"
            ),
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", replay_code],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)

    assert "would_write" in payload
    assert payload["would_execute"]["entrypoint"].endswith("_run_full_training_from_context")
    assert payload["run_spec"]["full_training_launch"] == "requested"
    assert payload["run_spec"]["hps"]["loss"]["delayed_movement_cost_tail_mode"] == (
        DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON
    )


def test_spec_file_execution_context_uses_validated_payload(tmp_path: Path) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "artifacts"),
            spec_dir=str(tmp_path / "spec"),
            smoke=True,
            full_train=True,
        )
    )
    parser = build_parser()
    context = build_run_spec_execution_context(
        parser.parse_args(
            [
                "--run-spec",
                result["run_spec_path"],
                "--resume",
                "--stop-after-batches",
                "1",
            ]
        ),
        parser=parser,
    )

    assert context.run_spec_path == Path(result["run_spec_path"])
    assert context.run_spec["full_training_launch"] == "requested"
    assert context.args.resume is True
    assert context.args.stop_after_batches == 1
    assert context.hps.hidden_type is eqx.nn.GRUCell
    assert context.hps.model.hidden_size == 4


def test_legacy_full_train_flags_emit_spec_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_executor(context, *, volume_commit=None):  # noqa: ANN001, ANN202
        captured["context"] = context
        captured["volume_commit"] = volume_commit
        return {"run_spec_path": str(context.run_spec_path), "completed_batches": 0}

    monkeypatch.setattr(cs_supervised_executor, "_run_full_training_from_context", fake_executor)

    result = run_full_training(
        _args(
            output_dir=str(tmp_path / "artifacts"),
            spec_dir=str(tmp_path / "spec"),
            smoke=True,
            full_train=True,
        )
    )
    context = captured["context"]

    assert Path(result["run_spec_path"]).is_file()
    assert context.run_spec_path == Path(result["run_spec_path"])
    assert context.args.smoke is False
    assert context.run_spec["mode"] == "full_train"
    assert context.run_spec["model_summary"]["hidden_size"] == 4
    assert context.hps.model.hidden_size == 4


def test_run_spec_dry_run_renderer_does_not_write(tmp_path: Path) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "artifacts"),
            spec_dir=str(tmp_path / "spec"),
            smoke=True,
            full_train=True,
        )
    )
    context = build_run_spec_execution_context(
        build_parser().parse_args(["--run-spec", result["run_spec_path"], "--dry-run"])
    )

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


def test_target_hps_without_profile_normalizes_to_band16_default() -> None:
    config = TargetRelativeMultiTargetTrainingConfig.from_payload(
        TreeNamespace(
            enabled=True,
            force_filter_feedback=True,
            target_distribution=TreeNamespace(),
        )
    )

    assert config.target_support_profile == TARGET_SUPPORT_PROFILE_CONST_BAND16
    assert len(config.seen_targets_m) == 56
    assert len(config.held_out_targets_m) == 16
    assert config.seen_amplitudes_m == (TARGET_SUPPORT_CONST_REACH_M,)
    assert config.held_out_amplitudes_m == (TARGET_SUPPORT_CONST_REACH_M,)


def test_target_hps_normalization_matches_frozen_override_outputs() -> None:
    cases = {
        "nested_old": TreeNamespace(
            enabled=True,
            force_filter_feedback=TreeNamespace(enabled=True),
            target_distribution=TreeNamespace(
                target_support_profile=TARGET_SUPPORT_PROFILE_020A65B,
                seen_directions_deg=(0.0, 90.0),
                held_out_directions_deg=(180.0,),
                seen_amplitudes_m=(0.15,),
                held_out_amplitudes_m=(0.2,),
                original_target_anchor_m=(0.15, 0.0),
                support_metadata={"source": "nested"},
            ),
        ),
        "top_level_overrides_nested": TreeNamespace(
            enabled=True,
            force_filter_feedback=False,
            target_support_profile=TARGET_SUPPORT_PROFILE_CONST_SPARSE8,
            seen_directions_deg=(0.0, 180.0),
            held_out_directions_deg=(90.0, 270.0),
            seen_amplitudes_m=(0.15,),
            held_out_amplitudes_m=(0.12,),
            original_target_anchor_m=(0.15, 0.0),
            support_metadata={"source": "top"},
            target_distribution=TreeNamespace(
                target_support_profile=TARGET_SUPPORT_PROFILE_020A65B,
                seen_directions_deg=(45.0,),
                support_metadata={"source": "nested"},
            ),
        ),
    }

    frozen_outputs = {
        "nested_old": {
            "enabled": True,
            "force_filter_feedback": True,
            "target_support_profile": TARGET_SUPPORT_PROFILE_020A65B,
            "seen_directions_deg": (0.0, 90.0),
            "held_out_directions_deg": (180.0,),
            "seen_amplitudes_m": (0.15,),
            "held_out_amplitudes_m": (0.2,),
            "original_target_anchor_m": (0.15, 0.0),
            "support_metadata": (("source", "nested"),),
        },
        "top_level_overrides_nested": {
            "enabled": True,
            "force_filter_feedback": False,
            "target_support_profile": TARGET_SUPPORT_PROFILE_CONST_SPARSE8,
            "seen_directions_deg": (0.0, 180.0),
            "held_out_directions_deg": (90.0, 270.0),
            "seen_amplitudes_m": (0.15,),
            "held_out_amplitudes_m": (0.12,),
            "original_target_anchor_m": (0.15, 0.0),
            "support_metadata": (("source", "top"),),
        },
    }

    assert {
        name: TargetRelativeMultiTargetTrainingConfig.from_payload(case).model_dump(mode="python")
        for name, case in cases.items()
    } == frozen_outputs


def test_resume_training_diagnostics_stitches_replicate_major_current_chunk(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "training_diagnostics.npz"
    np.savez_compressed(
        npz_path,
        batch_index=np.arange(1000),
        history_learning_rate=np.ones((1000, 5), dtype=np.float32),
        train_loss__total=np.arange(12000, dtype=np.float32),
    )

    stitched = _prepend_existing_training_diagnostics(
        npz_path,
        {
            "batch_index": np.arange(12000),
            "history_learning_rate": np.full((5, 11000), 2.0, dtype=np.float32),
        },
        completed_batches=12000,
    )

    assert stitched["history_learning_rate"].shape == (12000, 5)
    assert np.all(stitched["history_learning_rate"][:1000] == 1.0)
    assert np.all(stitched["history_learning_rate"][1000:] == 2.0)
    assert stitched["train_loss__total"].shape == (12000,)


def test_resume_training_diagnostics_stitches_replicate_major_prior_and_current(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "training_diagnostics.npz"
    np.savez_compressed(
        npz_path,
        batch_index=np.arange(500),
        history_learning_rate=np.ones((5, 500), dtype=np.float32),
    )

    stitched = _prepend_existing_training_diagnostics(
        npz_path,
        {
            "batch_index": np.arange(7500),
            "history_learning_rate": np.full((5, 7000), 2.0, dtype=np.float32),
        },
        completed_batches=7500,
    )

    assert stitched["history_learning_rate"].shape == (5, 7500)
    assert np.all(stitched["history_learning_rate"][:, :500] == 1.0)
    assert np.all(stitched["history_learning_rate"][:, 500:] == 2.0)


def test_resume_training_diagnostics_stitch_is_idempotent_for_checkpoint_sidecars(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "training_diagnostics.npz"
    prior = np.concatenate(
        [
            np.ones((5, 500), dtype=np.float32),
            np.full((5, 500), 2.0, dtype=np.float32),
        ],
        axis=1,
    )
    np.savez_compressed(
        npz_path,
        batch_index=np.arange(1000),
        history_learning_rate=prior,
    )
    continuation = np.concatenate(
        [
            np.full((5, 500), 2.0, dtype=np.float32),
            np.full((5, 500), 3.0, dtype=np.float32),
        ],
        axis=1,
    )

    stitched = _prepend_existing_training_diagnostics(
        npz_path,
        {
            "batch_index": np.arange(1500),
            "history_learning_rate": continuation,
        },
        completed_batches=1500,
    )

    assert stitched["history_learning_rate"].shape == (5, 1500)
    assert np.all(stitched["history_learning_rate"][:, :500] == 1.0)
    assert np.all(stitched["history_learning_rate"][:, 500:1000] == 2.0)
    assert np.all(stitched["history_learning_rate"][:, 1000:] == 3.0)


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
        allow_fresh_start=True,
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
    assert diagnostics_manifest["gradient_clip_norm"] is None
    assert diagnostics_manifest["training_history_path"] == str(output_dir / "training_history.eqx")
    assert "optimizer_gradient_norm_pre_clip" in diagnostics_manifest["arrays"]
    assert summary["training_duration_seconds"] > 0
    assert summary["training_batches_per_second"] > 0
    assert len(summary["chunks"]) == 2
    assert summary["chunks"][0]["chunk_batches"] == 2
    assert summary["chunks"][0]["duration_seconds"] > 0
    assert summary["chunks"][0]["batches_per_second"] > 0
    assert commits == 3


def test_policy_adversary_full_training_uses_checkpoint_sized_chunks(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        n_train_batches=2,
        batch_size=1,
        n_replicates=1,
        hidden_size=4,
        full_train=True,
        resume=True,
        allow_fresh_start=True,
        checkpoint_interval_batches=2,
        controller_lr=1e-3,
        lr_warmup_batches=1,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        log_step=1,
        disable_progress=True,
        quiet_progress=True,
        target_relative_multitarget=True,
        force_filter_feedback=True,
        initial_hidden_encoder=True,
        perturbation_training=True,
        perturbation_calibrated_timing=True,
        perturbation_physical_level="small",
        policy_adversary_training=True,
        policy_adversary_steps=1,
        policy_adversary_width=4,
        policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
        policy_adversary_radius_source="effective_020a65b_pgd_training_radius",
        broad_epsilon_reach_scaling=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    result = run_full_training(args)
    summary = json.loads((output_dir / "training_summary.json").read_text())
    checkpoint_latest = output_dir / "checkpoints" / "checkpoint_latest"
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())

    assert result["completed_batches"] == 2
    assert len(summary["chunks"]) == 1
    assert summary["chunks"][0]["chunk_batches"] == 2
    assert (output_dir / "history_chunks" / "history_0000002.eqx").exists()
    assert (checkpoint_latest / "adversary_policy.eqx").exists()
    assert (checkpoint_latest / "adversary_optimizer_state.eqx").exists()
    assert (output_dir / "trained_policy_adversary.eqx").exists()
    assert diagnostics_manifest["arrays"]["policy_adversary_diagnostic_sampled"]["shape"] == [2]
    with np.load(output_dir / "training_diagnostics.npz") as diagnostics:
        assert diagnostics["policy_adversary_diagnostic_sampled"].tolist() == [False, True]


def test_finite_affine_policy_adversary_full_training_persists_adam_state(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        n_train_batches=2,
        batch_size=1,
        n_replicates=1,
        hidden_size=4,
        full_train=True,
        resume=True,
        allow_fresh_start=True,
        checkpoint_interval_batches=2,
        controller_lr=1e-3,
        lr_warmup_batches=1,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        log_step=1,
        disable_progress=True,
        quiet_progress=True,
        target_relative_multitarget=True,
        force_filter_feedback=True,
        initial_hidden_encoder=True,
        perturbation_training=True,
        perturbation_calibrated_timing=True,
        perturbation_physical_level="small",
        policy_adversary_training=True,
        policy_adversary_policy_class=AFFINE_POLICY,
        policy_adversary_steps=1,
        policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
        policy_adversary_radius_source="effective_020a65b_pgd_training_radius",
        broad_epsilon_reach_scaling=True,
        loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    )

    result = run_full_training(args)
    run_spec = json.loads(Path(result["run_spec_path"]).read_text())
    checkpoint_latest = output_dir / "checkpoints" / "checkpoint_latest"
    metadata = json.loads((checkpoint_latest / "metadata.json").read_text())
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())

    assert result["completed_batches"] == 2
    assert run_spec["adversarial_phase"] == "learned_finite_affine_policy_adversary"
    assert run_spec["hps"]["policy_adversary_training"]["policy"]["kind"] == AFFINE_POLICY
    assert run_spec["hps"]["policy_adversary_training"]["inner_optimizer"]["method"] == "adam"
    assert metadata["run_spec"]["hps"]["policy_adversary_training"]["policy"]["kind"] == (
        AFFINE_POLICY
    )
    assert (checkpoint_latest / "adversary_policy.eqx").exists()
    assert (checkpoint_latest / "adversary_optimizer_state.eqx").exists()
    assert (output_dir / "trained_policy_adversary.eqx").exists()
    assert diagnostics_manifest["arrays"]["policy_adversary_diagnostic_sampled"]["shape"] == [2]


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
        allow_fresh_start=True,
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
    partial_diagnostics_manifest = json.loads(
        (output_dir / "training_diagnostics.json").read_text()
    )
    assert partial_diagnostics_manifest["completed_batches"] == 2
    assert (output_dir / "training_diagnostics.npz").exists()
    with np.load(output_dir / "training_diagnostics.npz") as partial_diagnostics:
        assert partial_diagnostics["batch_index"].tolist() == [0, 1]
        assert partial_diagnostics["optimizer_learning_rate"].shape == (2, 2)

    resumed_args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        n_train_batches=4,
        batch_size=2,
        n_replicates=2,
        hidden_size=4,
        full_train=True,
        resume=True,
        allow_fresh_start=True,
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


def test_cs_supervised_full_training_uses_native_executor(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    common = dict(
        n_train_batches=2,
        batch_size=2,
        n_replicates=1,
        hidden_size=4,
        full_train=True,
        resume=True,
        allow_fresh_start=True,
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
    native_args = _args(
        output_dir=str(native_dir / "bulk"),
        spec_dir=str(native_dir / "spec"),
        **common,
    )

    native_result = run_full_training(native_args)
    run_spec = json.loads(Path(native_result["run_spec_path"]).read_text(encoding="utf-8"))
    training_spec = feedbax_training_run_spec_from_payload(run_spec)
    summary = json.loads(
        (Path(native_args.output_dir) / "training_summary.json").read_text(encoding="utf-8")
    )

    assert native_result["completed_batches"] == 2
    assert training_spec.method_ref.key == "rlrmp/cs_supervised/v1"
    assert (
        training_spec.worker_execution.metadata["native_executor"]
        == "feedbax.training.executor.execute_training_run_spec"
    )
    assert summary["completed_batches"] == 2
    assert Path(native_result["training_manifest_path"]).exists()


def test_cs_supervised_native_same_length_resume_equivalence(tmp_path: Path) -> None:
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    common = dict(
        n_train_batches=4,
        batch_size=2,
        n_replicates=1,
        hidden_size=4,
        full_train=True,
        resume=True,
        allow_fresh_start=True,
        checkpoint_interval_batches=2,
        controller_lr=1e-3,
        gradient_clip_norm=5.0,
        lr_warmup_batches=1,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        log_step=1,
        disable_progress=True,
        quiet_progress=True,
    )
    full_args = _args(
        output_dir=str(full_dir / "bulk"),
        spec_dir=str(full_dir / "spec"),
        **common,
    )
    partial_args = _args(
        output_dir=str(resumed_dir / "bulk"),
        spec_dir=str(resumed_dir / "spec"),
        stop_after_batches=2,
        **common,
    )
    resume_args = _args(
        output_dir=str(resumed_dir / "bulk"),
        spec_dir=str(resumed_dir / "spec"),
        stop_after_batches=None,
        **common,
    )

    full = run_full_training(full_args)
    partial = run_full_training(partial_args)
    resumed = run_full_training(resume_args)

    full_state = _load_materialized_training_state(full_args)
    resumed_state = _load_materialized_training_state(resume_args)
    assert partial["completed_batches"] == 2
    assert full["completed_batches"] == resumed["completed_batches"] == 4
    _assert_pytree_close(full_state.model, resumed_state.model)
    _assert_pytree_close(full_state.optimizer_state, resumed_state.optimizer_state)
    np.testing.assert_allclose(
        _loss_series(Path(full_args.output_dir)),
        _loss_series(Path(resume_args.output_dir)),
        rtol=0,
        atol=1e-7,
    )


def _load_materialized_training_state(args: argparse.Namespace) -> TrainingState:
    hps = build_hps(args)
    key_init, key_train, _key_adversary = jr.split(jr.PRNGKey(int(args.seed)), 3)
    pair = setup_task_model_pair(hps, key=key_init)
    trainer = cs_nominal_gru._build_trainer(hps)
    template_state = cs_nominal_gru._initial_training_state(
        model=pair.model,
        trainer=trainer,
        where_train=cs_nominal_gru._where_train()[0],
        key=key_train,
    )
    return load_latest_checkpoint(
        Path(args.output_dir) / "checkpoints",
        model_template=pair.model,
        optimizer_state_template=template_state.optimizer_state,
    )


def _assert_pytree_close(left: object, right: object, *, atol: float = 1e-7) -> None:
    left_leaves = tuple(jt.leaves(eqx.filter(left, eqx.is_array)))
    right_leaves = tuple(jt.leaves(eqx.filter(right, eqx.is_array)))
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        assert left_array.shape == right_array.shape
        if left_array.dtype == jnp.bool_:
            np.testing.assert_array_equal(np.asarray(left_array), np.asarray(right_array))
        else:
            np.testing.assert_allclose(
                np.asarray(left_array),
                np.asarray(right_array),
                rtol=0,
                atol=atol,
            )


def _loss_series(output_dir: Path) -> np.ndarray:
    with np.load(output_dir / "training_diagnostics.npz") as diagnostics:
        return np.asarray(diagnostics["train_loss__total"])


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
    run_spec_path = Path(result["run_spec_path"])
    run_spec = json.loads(run_spec_path.read_text())
    summary = json.loads((output_dir / "training_summary.json").read_text())
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())

    assert result["completed_batches"] == 2
    assert run_spec_path == spec_dir.with_suffix(".json")
    assert not (spec_dir / "run.json").exists()
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
        cfg = PgdFullStateEpsilonTrainingConfig.from_payload(config)
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


def test_pgd_soft_energy_objective_penalizes_epsilon_energy() -> None:
    class EchoTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            return trial_specs.inputs["epsilon"]

    class LinearLoss:
        def __call__(self, states, trial_specs, model):
            del trial_specs, model
            return TreeNamespace(total=jnp.sum(states))

    trial_specs = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((1, 1, 1), dtype=jnp.float32),
                )
            }
        ),
        inputs={"epsilon": jnp.zeros((1, 1, 1), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=1),
    )

    hard_updated, hard_diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        EchoTask(),
        model=None,
        trial_specs=trial_specs,
        loss_func=LinearLoss(),
        keys_model=None,
        config={
            "enabled": True,
            "reach_length_scaling": False,
            "fixed_l2_radius_15cm": 1.0,
            "fixed_radius_source": "unit_test_fixed_radius",
            "n_steps": 1,
            "step_size_fraction": 1.0,
            "epsilon_dim": 1,
        },
        return_diagnostics=True,
    )
    soft_updated, soft_diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        EchoTask(),
        model=None,
        trial_specs=trial_specs,
        loss_func=LinearLoss(),
        keys_model=None,
        config={
            "enabled": True,
            "reach_length_scaling": False,
            "objective": {
                "kind": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                "lambda": 10.0,
            },
            "safety_cap": {
                "l2_radius_15cm": 1.0,
                "source": {"key": "unit_test_cap"},
            },
            "n_steps": 1,
            "step_size_fraction": 1.0,
            "epsilon_dim": 1,
        },
        return_diagnostics=True,
    )

    assert hard_updated.inputs["epsilon"][0, 0, 0] == pytest.approx(1.0)
    assert soft_updated.inputs["epsilon"][0, 0, 0] == pytest.approx(0.0)
    assert bool(hard_diagnostics["objective_kind_is_soft_energy"]) is False
    assert bool(soft_diagnostics["objective_kind_is_soft_energy"]) is True
    assert soft_diagnostics["energy_lambda"] == pytest.approx(10.0)
    assert hard_diagnostics["energy_penalty_term_selected"] == pytest.approx(0.0)
    assert hard_diagnostics["penalized_objective_selected"] == pytest.approx(
        hard_diagnostics["raw_task_loss_selected"]
    )
    assert hard_diagnostics["selected_objective_gain_over_zero"] == pytest.approx(
        hard_diagnostics["inner_objective_improvement"]
    )
    assert soft_diagnostics["raw_task_loss_final_endpoint"] == pytest.approx(1.0)
    assert soft_diagnostics["energy_penalty_term_final_endpoint"] == pytest.approx(10.0)
    assert soft_diagnostics["selected_vs_final_objective_gap"] == pytest.approx(9.0)
    assert soft_diagnostics["cap_boundary_fraction"] == pytest.approx(0.0)
    assert bool(soft_diagnostics["inner_objective_nonfinite_seen"]) is False


def test_pgd_cap_free_soft_energy_direct_epsilon_does_not_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EchoTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            return trial_specs.inputs["epsilon"]

    class LinearLoss:
        def __call__(self, states, trial_specs, model):
            del trial_specs, model
            return TreeNamespace(total=jnp.sum(states))

    def fail_projection(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cap-free direct-epsilon soft-energy must not project")

    monkeypatch.setattr(
        cs_perturbation_training,
        "_project_flattened_per_trial_l2_ball",
        fail_projection,
    )

    trial_specs = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((1, 1, 1), dtype=jnp.float32),
                )
            }
        ),
        inputs={"epsilon": jnp.zeros((1, 1, 1), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=1),
    )

    updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        EchoTask(),
        model=None,
        trial_specs=trial_specs,
        loss_func=LinearLoss(),
        keys_model=None,
        config={
            "enabled": True,
            "reach_length_scaling": False,
            "objective": {
                "kind": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                "lambda": 0.1,
            },
            "n_steps": 1,
            "step_size_fraction": 1.0,
            "epsilon_dim": 1,
        },
        return_diagnostics=True,
    )

    assert updated.inputs["epsilon"][0, 0, 0] == pytest.approx(1.0)
    assert bool(diagnostics["cap_free_soft_energy"]) is True
    assert bool(diagnostics["projection_active"]) is False
    assert bool(diagnostics["radius_bound_mode"]) is False
    assert bool(diagnostics["safety_cap_enabled"]) is False
    assert bool(diagnostics["step_size_uses_radius"]) is False
    assert bool(diagnostics["energy_lambda_override_active"]) is False
    assert np.isnan(diagnostics["radius_mean"])
    assert np.isnan(diagnostics["epsilon_norm_radius_ratio_mean"])
    assert diagnostics["cap_boundary_fraction"] == pytest.approx(0.0)
    assert diagnostics["energy_lambda"] == pytest.approx(0.1)


def test_pgd_cap_free_soft_energy_lambda_override_is_jittable() -> None:
    class EchoTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            return trial_specs.inputs["epsilon"]

    class LinearLoss:
        def __call__(self, states, trial_specs, model):
            del trial_specs, model
            return TreeNamespace(total=jnp.sum(states))

    trial_specs = TaskTrialSpec(
        inits=WhereDict({}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((1, 1, 1), dtype=jnp.float32),
                )
            }
        ),
        inputs={"epsilon": jnp.zeros((1, 1, 1), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=1),
    )
    config = {
        "enabled": True,
        "reach_length_scaling": False,
        "objective": {
            "kind": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            "lambda": 10.0,
        },
        "n_steps": 1,
        "step_size_fraction": 1.0,
        "epsilon_dim": 1,
    }

    def run_with_override(lambda_value):
        updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
            EchoTask(),
            model=None,
            trial_specs=trial_specs,
            loss_func=LinearLoss(),
            keys_model=None,
            config=config,
            soft_energy_lambda_override=lambda_value,
            return_diagnostics=True,
        )
        return (
            updated.inputs["epsilon"][0, 0, 0],
            diagnostics["energy_lambda"],
            diagnostics["energy_penalty_term_final_endpoint"],
            diagnostics["energy_lambda_override_active"],
        )

    jitted_run = jax.jit(run_with_override)

    low_epsilon, low_lambda, low_final_penalty, low_override = jitted_run(
        jnp.asarray(0.1, dtype=jnp.float32)
    )
    high_epsilon, high_lambda, high_final_penalty, high_override = jitted_run(
        jnp.asarray(10.0, dtype=jnp.float32)
    )

    assert low_epsilon == pytest.approx(1.0)
    assert low_lambda == pytest.approx(0.1)
    assert low_final_penalty == pytest.approx(0.1)
    assert bool(low_override) is True
    assert high_epsilon == pytest.approx(0.0)
    assert high_lambda == pytest.approx(10.0)
    assert high_final_penalty == pytest.approx(10.0)
    assert bool(high_override) is True


def test_pgd_soft_energy_lambda_override_rejects_non_direct_soft_modes() -> None:
    trial_specs = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((1, 4), dtype=jnp.float32)}),
        targets=WhereDict(
            {
                "mechanics.effector.pos": TargetSpec(
                    value=jnp.zeros((1, 1, 1), dtype=jnp.float32),
                )
            }
        ),
        inputs={"epsilon": jnp.zeros((1, 1, 1), dtype=jnp.float32)},
        timeline=TrialTimeline(n_steps=1),
    )

    with pytest.raises(ValueError, match="only valid for soft-energy PGD"):
        run_broad_epsilon_pgd_inner_maximizer(
            task=None,
            model=None,
            trial_specs=trial_specs,
            loss_func=None,
            keys_model=None,
            config={
                "enabled": True,
                "reach_length_scaling": False,
                "fixed_l2_radius_15cm": 1.0,
                "fixed_radius_source": "unit_test_fixed_radius",
                "epsilon_dim": 1,
            },
            soft_energy_lambda_override=jnp.asarray(1.0, dtype=jnp.float32),
        )

    with pytest.raises(ValueError, match="only supported for the direct_epsilon"):
        run_broad_epsilon_pgd_inner_maximizer(
            task=None,
            model=None,
            trial_specs=trial_specs,
            loss_func=None,
            keys_model=None,
            config={
                "enabled": True,
                "adversary_mechanism": LINEAR_NO_BIAS_POLICY,
                "reach_length_scaling": False,
                "objective": {
                    "kind": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                    "lambda": 1.0,
                },
                "safety_cap": {
                    "l2_radius_15cm": 1.0,
                    "source": {"key": "unit_test_cap"},
                },
                "epsilon_dim": 1,
            },
            soft_energy_lambda_override=jnp.asarray(1.0, dtype=jnp.float32),
        )


def test_pgd_soft_energy_objective_is_batch_size_invariant() -> None:
    class EchoTask:
        def eval_trials(self, model, trial_specs, keys_model):
            del model, keys_model
            return trial_specs.inputs["epsilon"]

    class MeanLinearLoss:
        def __call__(self, states, trial_specs, model):
            del trial_specs, model
            per_trial = jnp.sum(states, axis=tuple(range(1, states.ndim)))
            return TreeNamespace(total=jnp.mean(per_trial))

    def run(batch_size: int):
        trial_specs = TaskTrialSpec(
            inits=WhereDict({}),
            targets=WhereDict(
                {
                    "mechanics.effector.pos": TargetSpec(
                        value=jnp.zeros((batch_size, 1, 1), dtype=jnp.float32),
                    )
                }
            ),
            inputs={"epsilon": jnp.zeros((batch_size, 1, 1), dtype=jnp.float32)},
            timeline=TrialTimeline(n_steps=1),
        )
        return run_broad_epsilon_pgd_inner_maximizer(
            EchoTask(),
            model=None,
            trial_specs=trial_specs,
            loss_func=MeanLinearLoss(),
            keys_model=None,
            config={
                "enabled": True,
                "reach_length_scaling": False,
                "objective": {
                    "kind": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                    "lambda": 0.1,
                },
                "safety_cap": {
                    "l2_radius_15cm": 1.0,
                    "source": {"key": "unit_test_cap"},
                },
                "n_steps": 1,
                "step_size_fraction": 1.0,
                "epsilon_dim": 1,
            },
            return_diagnostics=True,
        )

    single_updated, single_diagnostics = run(batch_size=1)
    batch_updated, batch_diagnostics = run(batch_size=4)

    np.testing.assert_allclose(single_updated.inputs["epsilon"], 1.0)
    np.testing.assert_allclose(batch_updated.inputs["epsilon"], 1.0)
    for key in (
        "raw_task_loss_selected",
        "epsilon_energy_mean",
        "epsilon_energy_max",
        "energy_penalty_term_selected",
        "penalized_objective_selected",
        "selected_objective_gain_over_zero",
        "inner_objective_after",
        "inner_objective_improvement",
    ):
        assert batch_diagnostics[key] == pytest.approx(single_diagnostics[key])

    assert single_diagnostics["raw_task_loss_selected"] == pytest.approx(1.0)
    assert single_diagnostics["epsilon_energy_mean"] == pytest.approx(1.0)
    assert single_diagnostics["energy_penalty_term_selected"] == pytest.approx(0.1)
    assert single_diagnostics["penalized_objective_selected"] == pytest.approx(0.9)


def test_pgd_broad_epsilon_full_training_emits_inner_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
        quiet_progress=True,
    )

    result = run_full_training(args)
    progress = capsys.readouterr().out
    run_spec_path = Path(result["run_spec_path"])
    run_spec = json.loads(run_spec_path.read_text())
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())
    checkpoint_lines = [
        line for line in progress.splitlines() if line.startswith("BATCH phase=checkpoint")
    ]

    assert result["completed_batches"] == 2
    assert checkpoint_lines
    assert any("adv_penalty=" in line for line in checkpoint_lines)
    assert any("adv_energy=" in line for line in checkpoint_lines)
    assert any("adv_objective=" in line for line in checkpoint_lines)
    assert run_spec_path == spec_dir.with_suffix(".json")
    assert not (spec_dir / "run.json").exists()
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


def test_adaptive_epsilon_scaled_outer_full_training_emits_explicit_diagnostics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        issue="1ab1fef",
        n_train_batches=2,
        batch_size=1,
        n_replicates=5,
        hidden_size=4,
        target_relative_multitarget=True,
        force_filter_feedback=True,
        broad_epsilon_pgd_training=True,
        broad_epsilon_pgd_steps=1,
        broad_epsilon_pgd_step_size_fraction=0.5,
        broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        broad_epsilon_pgd_energy_lambda=1.0,
        adaptive_epsilon_curriculum=True,
        adaptive_epsilon_controller_training_mode=(
            ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
        ),
        adaptive_epsilon_update_interval_batches=1,
        adaptive_epsilon_outer_weight_start=0.25,
        adaptive_epsilon_outer_weight_final=0.25,
        adaptive_epsilon_outer_weight_ramp_batches=0,
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
    run_spec = json.loads(Path(result["run_spec_path"]).read_text())
    diagnostics_manifest = json.loads((output_dir / "training_diagnostics.json").read_text())

    cfg = run_spec["hps"]["adaptive_epsilon_curriculum"]
    assert cfg["controller_training_mode"] == ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
    assert (
        cfg["outer_adversarial_weight"]["applies_to"]
        == "optimized_direct_epsilon_channel_scale_for_controller_rollout"
    )
    assert "adaptive_epsilon_target_damage" in diagnostics_manifest["arrays"]
    assert "adaptive_epsilon_lambda_value" in diagnostics_manifest["arrays"]
    assert "adaptive_epsilon_gain_hat" in diagnostics_manifest["arrays"]
    assert "adaptive_epsilon_lambda_update_eta_eff" in diagnostics_manifest["arrays"]
    assert "adaptive_epsilon_sign_alternation_fraction" in diagnostics_manifest["arrays"]
    assert "adaptive_epsilon_ema_noise_floor" in diagnostics_manifest["arrays"]
    assert "adaptive_epsilon_outer_weight" in diagnostics_manifest["arrays"]
    assert (
        "adaptive_epsilon_training_batch_full_strength_damage_raw" in diagnostics_manifest["arrays"]
    )
    assert (
        "adaptive_epsilon_training_batch_applied_scaled_damage_raw"
        in diagnostics_manifest["arrays"]
    )
    assert (
        "adaptive_epsilon_adaptive_update_full_strength_damage_raw"
        in diagnostics_manifest["arrays"]
    )
    assert (
        "adaptive_epsilon_adaptive_update_applied_scaled_damage_raw"
        in diagnostics_manifest["arrays"]
    )
    with np.load(output_dir / "training_diagnostics.npz") as diagnostics:
        assert diagnostics["adaptive_epsilon_outer_weight"].tolist() == [
            pytest.approx(0.25),
            pytest.approx(0.25),
        ]
        assert diagnostics["adaptive_epsilon_epsilon_scale_used"].shape == (2, 5)
        assert diagnostics[
            "adaptive_epsilon_controller_training_mode_is_epsilon_scaled_outer"
        ].tolist() == [[True, True, True, True, True], [True, True, True, True, True]]
        np.testing.assert_allclose(
            diagnostics["adaptive_epsilon_training_batch_weighted_loss_total"],
            diagnostics["adaptive_epsilon_training_batch_applied_scaled_loss_total"],
        )


def test_full_training_smoke_can_disable_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    progress = capsys.readouterr().out
    run_spec_path = Path(result["run_spec_path"])
    run_spec = json.loads(run_spec_path.read_text())
    summary = json.loads((output_dir / "training_summary.json").read_text())

    assert result["completed_batches"] == 1
    assert "BATCH phase=checkpoint" not in progress
    assert run_spec_path == spec_dir.with_suffix(".json")
    assert not (spec_dir / "run.json").exists()
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
    optimizer = cs_nominal_gru._build_optimizer(hps)
    initial = cs_nominal_gru._initial_training_state(
        model=pair.model,
        trainer=optimizer,
        where_train=_where_train()[0],
        key=jr.PRNGKey(1),
    )
    trained, _history, _optimizer_state = cs_nominal_gru._run_cs_supervised_training_chunk(
        optimizer=optimizer,
        task=pair.task,
        model=pair.model,
        optimizer_state=initial.optimizer_state,
        hps=hps,
        where_train=_where_train()[0],
        key=jr.PRNGKey(2),
        start_batch=0,
        chunk_batches=n_batches,
        log_progress=False,
        log_every=1,
        pre_step_fn=None,
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
    from jax_cookbook.tree import filter_spec_leaves

    where_train_spec = filter_spec_leaves(pair.model, where_train)
    trainable = cs_nominal_gru.get_model_parameters(pair.model, where_train_spec)
    trainable_arrays = [leaf for leaf in jax.tree.leaves(trainable) if eqx.is_array(leaf)]

    assert trainable.nodes["mechanics"].A is None
    assert trainable.nodes["mechanics"].B is None
    assert trainable.nodes["mechanics"].B_w is None
    assert any(leaf.shape[-2:] == (12, 5) for leaf in trainable_arrays)
    assert any(leaf.shape[-2:] == (2, 4) for leaf in trainable_arrays)
