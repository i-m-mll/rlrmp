"""Tests for the temporary GRU pilot figure materializer."""

from __future__ import annotations

import numpy as np
from feedbax.state import CartesianState
from feedbax.task import TaskTrialSpec

from rlrmp.analysis.gru_pilot_figures import (
    REFERENCE_4D_LABEL,
    REFERENCE_LABEL,
    ReferenceProfile,
    RunFigureInputs,
    VelocityProfile,
    active_loss_term_labels,
    build_figure_summary,
    initial_effector_velocity,
    load_gru_training_history,
    repeat_single_validation_trial,
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
