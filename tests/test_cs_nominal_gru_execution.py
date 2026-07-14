"""C&S nominal-GRU execution contract tests."""

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import pytest
from feedbax import TaskTrialSpec, TrialTimeline, WhereDict
from feedbax.objectives.loss import AbstractLoss, TargetSpec
from feedbax.mechanics import LinearStateSpace
from feedbax.runtime.state_feedback import StateFeedbackSelector
from feedbax.config.namespace import TreeNamespace
from rlrmp.analysis.math.cs_game_card import build_canonical_game
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
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    CsAnalyticalQrfLoss,
    DelayedReachTrialTypeNormalizedLoss,
    get_reach_loss,
)
import rlrmp.train.cs_nominal_gru as cs_nominal_gru
import rlrmp.train.cs_perturbation_training as cs_perturbation_training
from rlrmp.train.cs_nominal_gru import (
    CS_DELAYED_REACH_TASK_TYPE,
    CsNominalGruConfig,
    TrainingState,
    build_graph_bundle,
    build_hps,
    _adaptive_epsilon_outer_weight,
    _adaptive_epsilon_schedule_batch,
    _scale_direct_epsilon_trial_specs,
    load_latest_checkpoint,
    run_full_training,
    write_run_spec,
)
from rlrmp.runtime.training_run_specs import feedbax_training_run_spec_from_payload
from rlrmp.data_products.broad_epsilon import load_pgd_radius_source
import rlrmp.train.executor.cs_supervised as cs_supervised_executor
from rlrmp.train.executor.cs_supervised import build_execution_context_from_spec
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_ADAM,
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    GRAPH_ADAPTER_SPECS,
    AFFINE_POLICY,
    LINEAR_NO_BIAS_POLICY,
    MILD_COMBINED_FAMILIES,
    TARGET_SUPPORT_PROFILE_CONST_BAND16,
    VALIDATION_BINS,
    BroadFullStateEpsilonTrainingConfig,
    BroadFullStateEpsilonTrainingTaskAdapter,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    TargetRelativeMultiTargetTrainingConfig,
    TargetRelativeMultiTargetTrainingTaskAdapter,
    _broad_epsilon_l2_radius,
    _command_input_direction_pulse,
    _ensure_broad_epsilon_input,
    _epsilon_time_mask,
    _expand_bool_like,
    _expand_radius,
    _normalize_flattened_per_trial,
    _project_flattened_per_trial_l2_ball,
    _run_finite_broad_epsilon_pgd_inner_maximizer,
    _set_input,
    _target_aligned_lateral_direction_pulse,
    apply_broad_epsilon_training,
    apply_training_perturbation_mixture,
    apply_training_target_distribution,
    apply_validation_bin,
    apply_validation_target_distribution,
    graph_adapter_specs,
    make_broad_epsilon_pgd_pre_step,
    make_memoryless_policy_adversary,
    policy_adversary_trial_specs,
    run_broad_epsilon_pgd_inner_maximizer,
    target_relative_validation_manifest,
    validation_bin_manifest,
)
from rlrmp.train.science_vocabulary import ScienceMode
from rlrmp.train.executor.equivalence import assert_paired_equivalent, run_paired_equivalence
from rlrmp.train.task_model import (
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

HISTORICAL_020A65B_PGD_RADIUS_15CM = float(
    load_pgd_radius_source("effective_020a65b_pgd_training_radius")["l2_radius_15cm"]
)


def _canonical_pgd_payload(**overrides: object) -> dict[str, object]:
    """Build the canonical config envelope consumed by PGD runtime boundaries."""

    config = PgdFullStateEpsilonTrainingConfig.model_validate(overrides)
    return {"config": config.model_dump(mode="python")}


def _args(**overrides) -> argparse.Namespace:
    values = CsNominalGruConfig(
        issue="test",
        output_dir="_artifacts/test/runs/test",
    ).model_dump(mode="python")
    values.update(compact_run_spec=False, verify_resume_only=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_feedbax_manifest_root_resolves_shared_artifacts_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_artifacts = tmp_path / "shared-artifacts"
    shared_artifacts.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "_artifacts").symlink_to(shared_artifacts, target_is_directory=True)
    monkeypatch.setattr(cs_supervised_executor, "REPO_ROOT", worktree)
    monkeypatch.setenv(
        "FEEDBAX_RUNS_DIR",
        str(worktree / "_artifacts" / "feedbax_runs"),
    )

    assert cs_supervised_executor._feedbax_manifest_root() == (shared_artifacts / "feedbax_runs")


@pytest.fixture
def isolated_feedbax_manifest_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    manifest_root = tmp_path / "feedbax-runs"
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(manifest_root))
    return manifest_root


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


def _unique_abs_nonzero(values: jnp.ndarray) -> np.ndarray:
    flat = np.asarray(jnp.abs(values)).reshape(-1)
    return np.unique(np.round(flat[flat > 0.0], 8))


def _assert_values_close_to_expected(values: np.ndarray, expected: set[float]) -> None:
    expected_values = np.asarray(sorted(expected), dtype=float)
    assert values.size > 0
    for value in values:
        assert np.any(np.isclose(value, expected_values, rtol=5e-5, atol=5e-7)), value


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
                broad_epsilon_pgd_safety_cap_source="effective_020a65b_pgd_training_radius",
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


def test_adaptive_epsilon_continuation_schedule_is_relative_to_resume_start() -> None:
    config = TreeNamespace(
        outer_adversarial_weight=TreeNamespace(start=0.0, final=1.0, ramp_batches=1_000)
    )
    scales = [
        _adaptive_epsilon_outer_weight(
            config,
            _adaptive_epsilon_schedule_batch(batch, ramp_start_batch=12_000),
        )
        for batch in (12_000, 12_500, 13_000)
    ]

    assert scales == pytest.approx([0.0, 0.5, 1.0])


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


def test_direct_epsilon_accepts_adam_inner_optimizer() -> None:
    cfg = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        adversary_mechanism=BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
        inner_optimizer_method=BROAD_EPSILON_PGD_ADAM,
    )

    assert cfg.inner_optimizer_method == BROAD_EPSILON_PGD_ADAM


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
        safety_cap_source="effective_020a65b_pgd_training_radius",
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
        safety_cap_source="effective_020a65b_pgd_training_radius",
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
        safety_cap_source="effective_020a65b_pgd_training_radius",
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


def test_policy_adversary_defaults_do_not_inherit_historical_radius() -> None:
    defaults = _args()
    assert defaults.policy_adversary_radius_15cm is None
    assert defaults.policy_adversary_radius_source is None

    hps = build_hps(defaults)
    policy = hps.policy_adversary_training
    assert policy.enabled is False
    assert policy.budget_contract.effective_l2_radius_15cm is None
    assert policy.budget_contract.active_max_l2_radius_15cm is None
    assert policy.budget_contract.budget_source is None


def test_policy_adversary_historical_spec_with_explicit_radius_and_source_parses() -> None:
    parsed = PolicyFullStateEpsilonTrainingConfig.from_payload(
        {
            "config": {
                "enabled": True,
                "reference_l2_radius_15cm": HISTORICAL_020A65B_PGD_RADIUS_15CM,
                "budget_source": "effective_020a65b_pgd_training_radius",
            },
        }
    )

    assert parsed.reference_l2_radius == pytest.approx(HISTORICAL_020A65B_PGD_RADIUS_15CM)
    assert parsed.budget_source == "effective_020a65b_pgd_training_radius"


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
        budget_source="effective_020a65b_pgd_training_radius",
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
    assert hps.perturbation_training.mode == ScienceMode.PERTURBATION_CALIBRATED


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


def test_spec_file_execution_context_uses_validated_payload(tmp_path: Path) -> None:
    result = write_run_spec(
        _args(
            output_dir=str(tmp_path / "artifacts"),
            spec_dir=str(tmp_path / "spec"),
            smoke=True,
            full_train=True,
        )
    )
    context = build_execution_context_from_spec(
        result["run_spec_path"],
        resume=True,
        stop_after_batches=1,
    )

    assert context.run_spec_path == Path(result["run_spec_path"])
    assert context.run_spec["full_training_launch"] == "requested"
    assert context.args.resume is True
    assert context.args.stop_after_batches == 1
    assert context.hps.hidden_type is eqx.nn.GRUCell
    assert context.hps.model.hidden_size == 4


def test_full_training_smoke_writes_checkpoint_and_final_artifacts(
    tmp_path: Path,
    isolated_feedbax_manifest_root: Path,
) -> None:
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


def test_policy_adversary_full_training_uses_checkpoint_sized_chunks(
    tmp_path: Path,
    isolated_feedbax_manifest_root: Path,
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
    isolated_feedbax_manifest_root: Path,
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


def test_full_training_stop_after_batches_resumes_to_full_count(
    tmp_path: Path,
    isolated_feedbax_manifest_root: Path,
) -> None:
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


def test_cs_supervised_full_training_uses_native_executor(
    tmp_path: Path,
    isolated_feedbax_manifest_root: Path,
) -> None:
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


def test_cs_supervised_native_same_length_resume_equivalence(
    tmp_path: Path,
    isolated_feedbax_manifest_root: Path,
) -> None:
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
        config=_canonical_pgd_payload(
            enabled=True,
            reach_length_scaling=False,
            fixed_l2_radius_15cm=1.0,
            fixed_radius_source="unit_test_fixed_radius",
            n_steps=1,
            step_size_fraction=1.0,
            epsilon_dim=1,
        ),
        return_diagnostics=True,
    )
    soft_updated, soft_diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        EchoTask(),
        model=None,
        trial_specs=trial_specs,
        loss_func=LinearLoss(),
        keys_model=None,
        config=_canonical_pgd_payload(
            enabled=True,
            reach_length_scaling=False,
            objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            energy_lambda=10.0,
            safety_cap_l2_radius_15cm=1.0,
            safety_cap_source="effective_020a65b_pgd_training_radius",
            n_steps=1,
            step_size_fraction=1.0,
            epsilon_dim=1,
        ),
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
        config=_canonical_pgd_payload(
            enabled=True,
            reach_length_scaling=False,
            objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            energy_lambda=0.1,
            n_steps=1,
            step_size_fraction=1.0,
            epsilon_dim=1,
        ),
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
    config = _canonical_pgd_payload(
        enabled=True,
        reach_length_scaling=False,
        objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        energy_lambda=10.0,
        n_steps=1,
        step_size_fraction=1.0,
        epsilon_dim=1,
    )

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
            config=_canonical_pgd_payload(
                enabled=True,
                reach_length_scaling=False,
                fixed_l2_radius_15cm=1.0,
                fixed_radius_source="unit_test_fixed_radius",
                epsilon_dim=1,
            ),
            soft_energy_lambda_override=jnp.asarray(1.0, dtype=jnp.float32),
        )

    with pytest.raises(ValueError, match="only supported for the direct_epsilon"):
        run_broad_epsilon_pgd_inner_maximizer(
            task=None,
            model=None,
            trial_specs=trial_specs,
            loss_func=None,
            keys_model=None,
            config=_canonical_pgd_payload(
                enabled=True,
                adversary_mechanism=LINEAR_NO_BIAS_POLICY,
                reach_length_scaling=False,
                objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                energy_lambda=1.0,
                safety_cap_l2_radius_15cm=1.0,
                safety_cap_source="effective_020a65b_pgd_training_radius",
                epsilon_dim=1,
            ),
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
            config=_canonical_pgd_payload(
                enabled=True,
                reach_length_scaling=False,
                objective_kind=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                energy_lambda=0.1,
                safety_cap_l2_radius_15cm=1.0,
                safety_cap_source="effective_020a65b_pgd_training_radius",
                n_steps=1,
                step_size_fraction=1.0,
                epsilon_dim=1,
            ),
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
