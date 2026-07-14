"""C&S nominal-GRU diagnostics contract tests."""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from feedbax import TaskTrialSpec, TrialTimeline, WhereDict
from feedbax.objectives.loss import TargetSpec
from feedbax.runtime.batch import BatchInfo
from feedbax.config.namespace import TreeNamespace
from rlrmp.loss import CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
from rlrmp.train.cs_nominal_gru import (
    AdaptiveEpsilonState,
    CsNominalGruConfig,
    GradientDiagnosticsState,
    TrainingState,
    UpdateDiagnosticsState,
    build_hps,
    _adaptive_epsilon_damage_target,
    _adaptive_epsilon_outer_weight,
    _adaptive_epsilon_zero_guard_from_state,
    _initial_adaptive_epsilon_zero_guard,
    _update_adaptive_epsilon_zero_guard,
    _emit_checkpoint_progress,
    _initial_adaptive_epsilon_state,
    _prepend_existing_training_diagnostics,
    _resize_optimizer_diagnostics_for_batches,
    _sample_adaptive_epsilon_damage_eval_batch,
    _sample_adaptive_epsilon_training_batch,
    _update_adaptive_epsilon_state,
    load_latest_checkpoint,
    run_full_training,
    save_training_checkpoint,
)
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    POLICY_ADVERSARY_PLAIN_MODE,
    PolicyFullStateEpsilonTrainingConfig,
    _broad_epsilon_l2_radius,
    _flattened_per_trial_norm,
    _project_flattened_per_trial_l2_ball,
    graph_adapter_specs,
    policy_adversary_projection_diagnostics,
)
from rlrmp.train.science_vocabulary import (
    AdaptiveEpsilonControllerMode,
    ScienceMode,
)


def _args(**overrides) -> argparse.Namespace:
    values = CsNominalGruConfig(
        issue="test",
        output_dir="_artifacts/test/runs/test",
    ).model_dump(mode="python")
    values.update(compact_run_spec=False, verify_resume_only=False)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def isolated_feedbax_manifest_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    manifest_root = tmp_path / "feedbax-runs"
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(manifest_root))
    return manifest_root


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


def test_policy_adversary_projection_reports_radius_energy_and_boundary() -> None:
    cfg = PolicyFullStateEpsilonTrainingConfig(
        enabled=True,
        epsilon_dim=2,
        state_feature_dim=4,
        reference_l2_radius_15cm=2.0,
        budget_source="effective_020a65b_pgd_training_radius",
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


def test_target_relative_h0_full_training_smoke_emits_diagnostics(
    tmp_path: Path,
    isolated_feedbax_manifest_root: Path,
) -> None:
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
        f"{ScienceMode.TARGET_RELATIVE_H0}+{ScienceMode.PERTURBATION}"
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


def test_pgd_broad_epsilon_full_training_emits_inner_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    isolated_feedbax_manifest_root: Path,
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
    assert ScienceMode.BROAD_EPSILON_PGD in run_spec["training_summary"]["training_mode"]
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
    isolated_feedbax_manifest_root: Path,
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
            AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER
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
    assert (
        cfg["controller_training_mode"]
        == AdaptiveEpsilonControllerMode.EPSILON_SCALED_OUTER
    )
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
    assert "adaptive_epsilon_epsilon_scale_used" in diagnostics_manifest["arrays"]
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
        assert diagnostics["adaptive_epsilon_outer_weight"].shape == (2,)
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
    isolated_feedbax_manifest_root: Path,
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
