"""Tests for the temporary GRU pilot figure materializer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from feedbax.loss import TargetSpec
from feedbax.state import CartesianState
from feedbax.task import DelayedReachTaskInputs, TaskTrialSpec, TrialTimeline

import rlrmp.analysis.gru_pilot_figures as gpf
from rlrmp.analysis.delayed_reach_eval_bank import make_delayed_reach_eval_bank
from rlrmp.analysis.gru_pilot_figures import (
    REFERENCE_4D_LABEL,
    REFERENCE_LABEL,
    ReferenceProfile,
    RunFigureInputs,
    VelocityProfile,
    active_loss_term_labels,
    build_figure_summary,
    evaluate_stochastic_forward_velocity_profile,
    initial_effector_velocity,
    load_gru_training_history,
    repeat_single_validation_trial,
    write_velocity_figure,
    write_velocity_by_replicate_figure,
    _go_aligned_forward_velocity_profile,
)


def _run_spec(*, hidden_weight: float = 0.0) -> dict[str, object]:
    return {
        "hps": {
            "loss": {
                "weights": {
                    "effector_pos_running": 1e6,
                    "effector_terminal_pos": 1e6,
                    "effector_terminal_vel": 1e5,
                    "effector_vel_running": 1e5,
                    "nn_hidden": hidden_weight,
                    "nn_output": 1.0,
                }
            }
        }
    }


def _write_history(path, labels: tuple[str, ...]) -> None:
    with path.open("wb") as stream:
        stream.write(b"null\n")
        for context_offset in (0.0, 100.0):
            for idx, _label in enumerate(labels):
                np.save(stream, np.full((3, 2), context_offset + idx, dtype=np.float64))
                np.save(stream, np.asarray(float(idx + 1), dtype=np.float64))
            np.save(stream, np.asarray(1.0, dtype=np.float64))
        np.save(stream, np.full((3, 2), 0.01, dtype=np.float64))


def test_active_loss_term_labels_follow_gru_feedbax_order() -> None:
    assert active_loss_term_labels(_run_spec(hidden_weight=0.0)) == (
        "effector_pos_running",
        "effector_terminal_pos",
        "effector_terminal_vel",
        "effector_vel_running",
        "nn_output",
    )
    assert active_loss_term_labels(_run_spec(hidden_weight=1e-5)) == (
        "effector_pos_running",
        "effector_terminal_pos",
        "effector_terminal_vel",
        "effector_vel_running",
        "nn_hidden",
        "nn_output",
    )


def test_active_loss_term_labels_use_full_qrf_objective() -> None:
    run_spec = _run_spec(hidden_weight=1e-5)
    run_spec["loss_objective"] = "full_analytical_qrf"

    assert active_loss_term_labels(run_spec) == ("full_analytical_qrf",)


def test_active_loss_term_labels_include_full_qrf_pre_go_auxiliary() -> None:
    run_spec = _run_spec(hidden_weight=1e-5)
    run_spec["loss_objective"] = "full_analytical_qrf"
    run_spec["hps"]["loss"]["weights"]["nn_output_pre_go"] = 1.0

    assert active_loss_term_labels(run_spec) == (
        "full_analytical_qrf",
        "nn_output_pre_go",
    )


def test_active_loss_term_labels_include_force_filter_ablation_term() -> None:
    run_spec = _run_spec(hidden_weight=0.0)
    run_spec["loss_objective"] = "partial_net_output_force_filter"
    run_spec["hps"]["loss"]["weights"]["mechanics_force_filter"] = 1.0 / 6.0

    assert active_loss_term_labels(run_spec) == (
        "effector_pos_running",
        "effector_terminal_pos",
        "effector_terminal_vel",
        "effector_vel_running",
        "mechanics_force_filter",
        "nn_output",
    )


def test_load_gru_training_history_rebuilds_feedbax_loss_tree(tmp_path) -> None:
    labels = active_loss_term_labels(_run_spec(hidden_weight=1e-5))
    path = tmp_path / "training_history.eqx"
    _write_history(path, labels)

    history = load_gru_training_history(_run_spec(hidden_weight=1e-5), path)

    assert history.loss.names == labels
    assert history.loss_validation.names == labels
    assert history.loss.children[4].label == "nn_hidden"
    assert np.asarray(history.loss.children[4].value).shape == (3, 2)
    assert float(history.loss.children[4].weight) == 5.0
    assert np.asarray(history.learning_rate).shape == (3, 2)


def test_repeat_single_validation_trial_preserves_initial_velocity() -> None:
    trial_specs = TaskTrialSpec(
        inits={
            lambda state: state.mechanics.effector: CartesianState(
                pos=np.asarray([[0.0, 0.0]]),
                vel=np.asarray([[0.0, 0.0]]),
                force=np.asarray([[0.0, 0.0]]),
            )
        },
        targets={},
        inputs=np.zeros((1, 2), dtype=np.float64),
    )
    repeated = repeat_single_validation_trial(trial_specs, 4)

    np.testing.assert_allclose(
        initial_effector_velocity(repeated),
        np.zeros((4, 2), dtype=np.float64),
    )


def test_repeat_single_validation_trial_slices_multitarget_bank() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.arange(20 * 48, dtype=np.float64).reshape(20, 48)},
        targets={
            "mechanics.effector.pos": TargetSpec(
                value=np.arange(20 * 60 * 2, dtype=np.float64).reshape(20, 60, 2)
            )
        },
        inputs={"target": np.arange(20 * 60 * 2, dtype=np.float64).reshape(20, 60, 2)},
    )

    repeated = repeat_single_validation_trial(trial_specs, 4)

    assert repeated.inits["mechanics.vector"].shape == (4, 48)
    assert repeated.targets["mechanics.effector.pos"].value.shape == (4, 60, 2)
    assert repeated.inputs["target"].shape == (4, 60, 2)
    np.testing.assert_allclose(
        repeated.inits["mechanics.vector"],
        np.repeat(trial_specs.inits["mechanics.vector"][:1], 4, axis=0),
    )


def test_initial_effector_velocity_reads_cs_lss_vector_init() -> None:
    lss_vector = np.zeros((2, 48), dtype=np.float64)
    lss_vector[:, 2:4] = np.asarray([[0.1, -0.2], [0.3, -0.4]])
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": lss_vector},
        targets={},
        inputs=np.zeros((2, 2), dtype=np.float64),
    )

    np.testing.assert_allclose(
        initial_effector_velocity(trial_specs),
        np.asarray([[0.1, -0.2], [0.3, -0.4]]),
    )


def test_go_aligned_velocity_profile_includes_full_support_pre_go_context_by_default() -> None:
    forward = np.broadcast_to(
        np.arange(90, dtype=np.float64)[None, None, :],
        (2, 3, 90),
    ).copy()
    go_index = np.asarray([24, 24, 24], dtype=np.int64)

    time_s, mean, std, replicate_mean, replicate_std, alignment = (
        _go_aligned_forward_velocity_profile(
            forward,
            go_index,
            dt=0.01,
            movement_horizon_steps=60,
        )
    )

    assert time_s.shape == (84,)
    assert mean.shape == (84,)
    assert std.shape == (84,)
    assert replicate_mean.shape == (2, 84)
    assert replicate_std.shape == (2, 84)
    np.testing.assert_allclose(time_s[0], -0.24)
    np.testing.assert_allclose(time_s[-1], 0.59)
    assert alignment["pre_go_context_mode"] == "full_support"
    assert alignment["requested_pre_go_context_steps"] is None
    assert alignment["full_support_pre_go_steps"] == 24
    assert alignment["plot_left_context_steps"] == 24
    assert alignment["plot_post_go_steps"] == 60
    assert alignment["full_support_slice"] == [0, 90]
    assert alignment["requested_window_slice"] == [0, 84]
    assert alignment["trim_slice"] == [0, 84]


def test_go_aligned_velocity_profile_respects_explicit_pre_go_context() -> None:
    forward = np.broadcast_to(
        np.arange(90, dtype=np.float64)[None, None, :],
        (2, 3, 90),
    ).copy()
    go_index = np.asarray([24, 24, 24], dtype=np.int64)

    time_s, mean, std, replicate_mean, replicate_std, alignment = (
        _go_aligned_forward_velocity_profile(
            forward,
            go_index,
            dt=0.01,
            movement_horizon_steps=60,
            pre_go_context_steps=10,
        )
    )

    assert time_s.shape == (70,)
    assert mean.shape == (70,)
    assert std.shape == (70,)
    assert replicate_mean.shape == (2, 70)
    assert replicate_std.shape == (2, 70)
    np.testing.assert_allclose(time_s[0], -0.10)
    np.testing.assert_allclose(time_s[-1], 0.59)
    assert alignment["pre_go_context_mode"] == "explicit_steps"
    assert alignment["requested_pre_go_context_steps"] == 10
    assert alignment["full_support_pre_go_steps"] == 24
    assert alignment["plot_left_context_steps"] == 10
    assert alignment["plot_post_go_steps"] == 60


def test_velocity_evaluator_reports_fixed_delayed_eval_bank_metadata(monkeypatch) -> None:
    base = _adapted_delayed_trial_spec_template()
    bank = make_delayed_reach_eval_bank(
        base,
        catch=False,
        direction_count=2,
        movement_horizon_steps=60,
    )

    class FakeTask:
        validation_trials = base

        def eval_trials(self, model, trial_specs, keys):
            del model, keys
            n_trials = trial_specs.targets["mechanics.effector.pos"].value.shape[0]
            n_steps = trial_specs.timeline.n_steps
            velocity = jnp.zeros((n_trials, n_steps, 2), dtype=jnp.float32)
            velocity = velocity.at[..., 0].set(1.0)
            return SimpleNamespace(
                mechanics=SimpleNamespace(
                    effector=SimpleNamespace(vel=velocity),
                ),
            )

    monkeypatch.setattr(
        gpf,
        "setup_task_model_pair",
        lambda hps, key: SimpleNamespace(task=FakeTask(), model=object()),
    )
    monkeypatch.setattr(
        gpf,
        "load_with_hyperparameters",
        lambda path, setup_func: (object(), {}),
    )
    monkeypatch.setattr(gpf.eqx, "partition", lambda model, pred: (jnp.zeros((1,)), None))
    monkeypatch.setattr(gpf.eqx, "combine", lambda leaves, other: leaves)
    monkeypatch.setattr(
        gpf.eqx,
        "filter_vmap",
        lambda fn, in_axes=None: (
            lambda model_arrays, keys: jnp.stack(
                [fn(model_arrays[index], keys[index]) for index in range(model_arrays.shape[0])]
            )
        ),
    )

    profile = evaluate_stochastic_forward_velocity_profile(
        RunFigureInputs(
            run_id="run",
            label="Run",
            run_spec_path=Path(__file__),
            artifact_dir=Path(__file__).parent,
            run_spec={
                "hps": {"model": {"n_replicates": 1}, "dt": 0.01},
                "game_card": {"dt": 0.01},
                "task_timing": {"movement_window": {"cs_horizon_steps": 60}},
            },
        ),
        n_rollout_trials=1,
        evaluation_bank=bank,
        pre_go_context_steps=10,
    )

    assert profile.alignment["go_index_min"] == 10
    assert profile.alignment["go_index_max"] == 30
    assert profile.alignment["evaluation_bank"]["kind"] == "no_catch"
    assert profile.alignment["evaluation_bank"]["trial_count"] == 42
    assert profile.alignment["plot_left_context_steps"] == 10
    assert profile.alignment["plot_post_go_steps"] == 60


def _adapted_delayed_trial_spec_template() -> TaskTrialSpec:
    n_trials = 2
    n_steps = 90
    target = np.zeros((n_trials, n_steps, 2), dtype=np.float32)
    target[0, :, :] = np.asarray([0.15, 0.0], dtype=np.float32)
    target[1, :, :] = np.asarray([0.0, 0.15], dtype=np.float32)
    hold = np.ones((n_trials, n_steps), dtype=np.float32)
    hold[:, 29:] = 0.0
    return TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((n_trials, 48), dtype=np.float32)},
        inputs={
            "task": DelayedReachTaskInputs(
                CartesianState(pos=target),
                hold,
                np.ones_like(hold),
            ),
            "input": 1.0 - hold,
            "target": target,
            "effector_target": CartesianState(pos=target),
            "epsilon": np.zeros((n_trials, n_steps, 8), dtype=np.float32),
        },
        targets={"mechanics.effector.pos": TargetSpec(target)},
        timeline=TrialTimeline.from_epochs_events(
            n_steps,
            epoch_bounds=np.asarray([[0, 29, n_steps], [0, 29, n_steps]]),
            epoch_names=("prep", "movement"),
        ),
    )


def test_build_figure_summary_records_8d_and_4d_reference_metadata(tmp_path) -> None:
    run = RunFigureInputs(
        run_id="cs_stochastic_gru__no_hidden_penalty",
        label="nn_hidden = 0",
        run_spec_path=tmp_path / "run.json",
        artifact_dir=tmp_path,
        run_spec={},
    )
    profile = VelocityProfile(
        run_id=run.run_id,
        label=run.label,
        time_s=np.asarray([0.0, 0.01]),
        mean=np.asarray([0.0, 1.0]),
        std=np.asarray([0.0, 0.1]),
        n_replicates=2,
        n_rollout_trials_per_replicate=3,
    )
    references = (
        ReferenceProfile(
            label=REFERENCE_LABEL,
            observation_channel="oldest_delayed_physical_block_full_8d",
            observation_dim=8,
            observed_physical_indices=tuple(range(8)),
            time_s=np.asarray([0.0, 0.01]),
            forward_velocity=np.asarray([0.0, 0.9]),
            forward_velocity_std=np.asarray([0.0, 0.05]),
            n_samples=10,
            peak_forward_velocity_m_s=0.9,
            time_of_peak_forward_velocity_s=0.01,
            terminal_position_error_m=0.0,
            gamma_factor=1.05,
            parity_status="fixed_point",
            line_color="#111827",
            line_dash="dash",
        ),
        ReferenceProfile(
            label=REFERENCE_4D_LABEL,
            observation_channel="oldest_delayed_position_velocity_4d",
            observation_dim=4,
            observed_physical_indices=(0, 1, 2, 3),
            time_s=np.asarray([0.0, 0.01]),
            forward_velocity=np.asarray([0.0, 0.8]),
            forward_velocity_std=np.asarray([0.0, 0.04]),
            n_samples=10,
            peak_forward_velocity_m_s=0.8,
            time_of_peak_forward_velocity_s=0.01,
            terminal_position_error_m=0.0,
            gamma_factor=1.05,
            parity_status="fixed_point",
            line_color="#f97316",
            line_dash="dot",
        ),
    )

    summary = build_figure_summary(
        experiment="30f2313",
        runs=(run,),
        loss_files=(tmp_path / "loss_training.html",),
        velocity_file=tmp_path / "forward_velocity_profiles_stochastic.html",
        alias_file=tmp_path / "forward_velocity_profiles_stochastic_with_extlqg.html",
        velocity_profiles=(profile,),
        references=references,
    )

    metadata = summary["velocity_profiles"]["references"]
    assert set(metadata) == {REFERENCE_LABEL, REFERENCE_4D_LABEL}
    assert metadata[REFERENCE_LABEL]["observation_dim"] == 8
    assert metadata[REFERENCE_4D_LABEL]["observation_dim"] == 4
    assert metadata[REFERENCE_4D_LABEL]["observed_physical_indices"] == [0, 1, 2, 3]
    assert metadata[REFERENCE_4D_LABEL]["n_stochastic_samples"] == 10


def test_velocity_figure_accepts_twelve_profile_rows(tmp_path, monkeypatch) -> None:
    profiles = tuple(
        VelocityProfile(
            run_id=f"run_{idx}",
            label=f"row {idx}",
            time_s=np.asarray([0.0, 0.01]),
            mean=np.asarray([0.0, 1.0 + 0.01 * idx]),
            std=np.asarray([0.0, 0.1]),
            n_replicates=2,
            n_rollout_trials_per_replicate=3,
        )
        for idx in range(12)
    )
    captured: dict[str, go.Figure] = {}

    def capture_write_html(self, file, *_args, **_kwargs) -> None:
        captured["fig"] = self
        file.write_text("", encoding="utf-8")

    monkeypatch.setattr(go.Figure, "write_html", capture_write_html)

    path = write_velocity_figure(profiles, output_dir=tmp_path)

    assert path == tmp_path / "forward_velocity_profiles_stochastic.html"
    fig = captured["fig"]
    assert "yaxis12" in fig.layout
    assert fig.layout.height == 420 * len(profiles)
    assert np.isclose(fig.layout.yaxis.domain[0] - fig.layout.yaxis2.domain[1], 0.02)
    assert fig.layout.yaxis.domain[1] - fig.layout.yaxis.domain[0] > 0.06


def test_velocity_by_replicate_legend_groups_cross_subplot_toggles(
    tmp_path, monkeypatch
) -> None:
    profile_a = VelocityProfile(
        run_id="run_a",
        label="no hidden",
        time_s=np.asarray([0.0, 0.01]),
        mean=np.asarray([0.0, 1.0]),
        std=np.asarray([0.0, 0.1]),
        n_replicates=2,
        n_rollout_trials_per_replicate=3,
        replicate_mean=np.asarray([[0.0, 1.0], [0.0, 0.8]]),
        replicate_std=np.asarray([[0.0, 0.1], [0.0, 0.2]]),
    )
    profile_b = VelocityProfile(
        run_id="run_b",
        label="hidden",
        time_s=np.asarray([0.0, 0.01]),
        mean=np.asarray([0.0, 0.9]),
        std=np.asarray([0.0, 0.1]),
        n_replicates=2,
        n_rollout_trials_per_replicate=3,
        replicate_mean=np.asarray([[0.0, 0.9], [0.0, 0.7]]),
        replicate_std=np.asarray([[0.0, 0.1], [0.0, 0.2]]),
    )
    reference = ReferenceProfile(
        label=REFERENCE_LABEL,
        observation_channel="oldest_delayed_physical_block_full_8d",
        observation_dim=8,
        observed_physical_indices=tuple(range(8)),
        time_s=np.asarray([0.0, 0.01]),
        forward_velocity=np.asarray([0.0, 0.95]),
        forward_velocity_std=np.asarray([0.0, 0.05]),
        n_samples=10,
        peak_forward_velocity_m_s=0.95,
        time_of_peak_forward_velocity_s=0.01,
        terminal_position_error_m=0.0,
        gamma_factor=1.05,
        parity_status="fixed_point",
        line_color="#111827",
        line_dash="dash",
    )
    captured: dict[str, go.Figure] = {}

    def capture_write_html(self, file, *_args, **_kwargs) -> None:
        captured["fig"] = self
        file.write_text("", encoding="utf-8")

    monkeypatch.setattr(go.Figure, "write_html", capture_write_html)

    write_velocity_by_replicate_figure(
        (profile_a, profile_b),
        output_dir=tmp_path,
        references=(reference,),
    )

    fig = captured["fig"]
    assert fig.layout.height == 440 * 2
    assert np.isclose(fig.layout.yaxis.domain[0] - fig.layout.yaxis2.domain[1], 0.02)
    assert fig.layout.legend.groupclick == "togglegroup"

    legend_items = [trace for trace in fig.data if trace.showlegend]
    assert [(trace.name, trace.legendgroup) for trace in legend_items] == [
        ("replicate 0", "replicate-0"),
        ("replicate 1", "replicate-1"),
        (REFERENCE_LABEL, "reference-oldest_delayed_physical_block_full_8d"),
    ]

    grouped = {}
    for trace in fig.data:
        grouped.setdefault(trace.legendgroup, []).append(trace)
    assert len(grouped["replicate-0"]) == 4
    assert len(grouped["replicate-1"]) == 4
    assert len(grouped["reference-oldest_delayed_physical_block_full_8d"]) == 4
    assert all(not trace.showlegend for trace in fig.data if trace.fill == "toself")
