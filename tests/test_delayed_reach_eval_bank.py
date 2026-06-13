"""Tests for fixed delayed-reach evaluation banks."""

from __future__ import annotations

import numpy as np
import pytest
from feedbax.loss import TargetSpec
from feedbax.state import CartesianState
from feedbax.task import DelayedReachTaskInputs, TaskTrialSpec, TrialTimeline

from rlrmp.analysis.delayed_reach_eval_bank import (
    DEFAULT_GO_CUE_STEPS,
    DEFAULT_UNIFORM_REACH_LENGTH_M,
    make_delayed_reach_eval_bank,
)
from rlrmp.analysis.trial_alignment import trial_timing_from_specs


def test_delayed_eval_bank_spans_go_cues_and_centerout_directions() -> None:
    base = _base_delayed_trials()

    bank = make_delayed_reach_eval_bank(
        base,
        catch=False,
        direction_count=4,
        movement_horizon_steps=60,
    )
    timing = trial_timing_from_specs(bank.trial_specs, movement_horizon_steps=60)
    target = np.asarray(bank.trial_specs.targets["mechanics.effector.pos"].value)
    visible_target = np.asarray(bank.trial_specs.inputs.effector_target.pos)

    assert bank.metadata["go_cue_min"] == 10
    assert bank.metadata["go_cue_max"] == 30
    assert bank.metadata["go_cue_steps"] == list(DEFAULT_GO_CUE_STEPS)
    assert bank.metadata["direction_source"] == "uniform_grid"
    assert bank.metadata["direction_count"] == 4
    assert bank.metadata["trial_count"] == 84
    assert bank.metadata["reach_length_m"] == DEFAULT_UNIFORM_REACH_LENGTH_M
    assert bank.metadata["reach_length_source"] == "uniform_default"
    assert bank.metadata["direction_source_inferred_from_validation_targets"] is False
    assert bank.metadata["duplicate_direction_count"] == 0
    np.testing.assert_allclose(
        bank.metadata["target_radii_m"],
        np.full(4, DEFAULT_UNIFORM_REACH_LENGTH_M),
    )
    np.testing.assert_allclose(
        bank.metadata["target_angles_rad"],
        [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
    )
    assert timing.to_json()["go_index_min"] == 10
    assert timing.to_json()["go_index_max"] == 30
    assert sorted(set(timing.go_index.tolist())) == list(DEFAULT_GO_CUE_STEPS)
    assert target.shape == (84, 90, 2)
    assert visible_target.shape == (84, 90, 2)
    assert np.linalg.matrix_rank(visible_target[:4, -1, :]) == 2


def test_uniform_bank_does_not_infer_reach_length_from_validation_targets() -> None:
    base = _nonuniform_delayed_trials()

    bank = make_delayed_reach_eval_bank(
        base,
        catch=False,
        go_cue_steps=(10,),
        direction_count=4,
        movement_horizon_steps=60,
    )
    visible_target = np.asarray(bank.trial_specs.inputs.effector_target.pos)

    assert bank.metadata["direction_source"] == "uniform_grid"
    assert bank.metadata["reach_length_source"] == "uniform_default"
    assert bank.metadata["reach_length_m_explicit"] is False
    assert bank.metadata["direction_source_inferred_from_validation_targets"] is False
    assert bank.metadata["duplicate_direction_count"] == 0
    np.testing.assert_allclose(
        bank.metadata["source_target_radii_m"],
        [0.11, 0.20, 0.12, 0.13],
    )
    np.testing.assert_allclose(
        bank.metadata["target_radii_m"],
        np.full(4, DEFAULT_UNIFORM_REACH_LENGTH_M),
    )
    np.testing.assert_allclose(visible_target[0, -1, :], [0.15, 0.0], atol=1e-7)
    np.testing.assert_allclose(visible_target[1, -1, :], [0.0, 0.15], atol=1e-7)
    np.testing.assert_allclose(visible_target[2, -1, :], [-0.15, 0.0], atol=1e-7)
    np.testing.assert_allclose(visible_target[3, -1, :], [0.0, -0.15], atol=1e-7)


def test_validation_target_direction_source_preserves_target_list_metadata() -> None:
    base = _nonuniform_delayed_trials()

    bank = make_delayed_reach_eval_bank(
        base,
        catch=False,
        go_cue_steps=(10,),
        direction_count=4,
        direction_source="validation_targets",
        movement_horizon_steps=60,
    )
    visible_target = np.asarray(bank.trial_specs.inputs.effector_target.pos)

    assert bank.metadata["direction_source"] == "validation_targets"
    assert bank.metadata["reach_length_source"] == "validation_targets_median"
    assert bank.metadata["reach_length_m_explicit"] is False
    assert bank.metadata["direction_source_inferred_from_validation_targets"] is True
    assert bank.metadata["duplicate_direction_count"] == 1
    assert bank.metadata["source_trial_indices"] == [0, 1, 2, 3]
    np.testing.assert_allclose(
        bank.metadata["target_radii_m"],
        [0.11, 0.20, 0.12, 0.13],
    )
    np.testing.assert_allclose(bank.metadata["reach_length_m"], 0.125)
    np.testing.assert_allclose(
        bank.metadata["target_angles_rad"],
        [0.0, 0.0, 0.5 * np.pi, np.pi],
    )
    np.testing.assert_allclose(visible_target[:, -1, :], _nonuniform_endpoints())


def test_validation_target_direction_source_requires_enough_targets() -> None:
    base = _nonuniform_delayed_trials()

    with pytest.raises(ValueError, match="requires at least direction_count"):
        make_delayed_reach_eval_bank(
            base,
            catch=False,
            go_cue_steps=(10,),
            direction_count=5,
            direction_source="validation_targets",
            movement_horizon_steps=60,
        )


def test_no_catch_bank_holds_until_go_then_moves_to_visible_target() -> None:
    base = _base_delayed_trials()
    bank = make_delayed_reach_eval_bank(
        base,
        catch=False,
        go_cue_steps=(10,),
        direction_count=4,
        movement_horizon_steps=60,
    )
    hold = np.asarray(bank.trial_specs.inputs.hold)
    scored_target = np.asarray(bank.trial_specs.targets["mechanics.effector.pos"].value)
    visible_target = np.asarray(bank.trial_specs.inputs.effector_target.pos)

    np.testing.assert_allclose(hold[:, :10], 1.0)
    np.testing.assert_allclose(hold[:, 10:], 0.0)
    np.testing.assert_allclose(scored_target[:, :10, :], 0.0)
    np.testing.assert_allclose(scored_target[:, 10:, :], visible_target[:, 10:, :])
    assert np.any(np.abs(scored_target[:, 10:, :]) > 0.0)


def test_catch_bank_keeps_hold_and_scored_target_at_initial_position() -> None:
    base = _adapted_delayed_trial_spec_template()
    bank = make_delayed_reach_eval_bank(
        base,
        catch=True,
        go_cue_steps=(10, 30),
        direction_count=3,
        movement_horizon_steps=60,
    )
    hold = np.asarray(bank.trial_specs.inputs["task"].hold)
    go = np.asarray(bank.trial_specs.inputs["input"])
    scored_target = np.asarray(bank.trial_specs.targets["mechanics.effector.pos"].value)
    visible_target = np.asarray(bank.trial_specs.inputs["target"])
    task_visible_target = np.asarray(bank.trial_specs.inputs["task"].effector_target.pos)
    effector_target = np.asarray(bank.trial_specs.inputs["effector_target"].pos)

    assert bank.metadata["catch"] is True
    np.testing.assert_allclose(hold, 1.0)
    np.testing.assert_allclose(go, 0.0)
    np.testing.assert_allclose(scored_target, 0.0)
    np.testing.assert_allclose(task_visible_target, visible_target)
    np.testing.assert_allclose(effector_target, scored_target)
    assert np.any(np.abs(visible_target) > 0.0)


def _base_delayed_trials() -> TaskTrialSpec:
    n_trials = 4
    n_steps = 90
    angles = np.linspace(0.0, 2.0 * np.pi, n_trials, endpoint=False)
    endpoints = 0.15 * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    target = np.broadcast_to(endpoints[:, None, :], (n_trials, n_steps, 2)).astype(
        np.float32
    )
    hold = np.ones((n_trials, n_steps, 1), dtype=np.float32)
    hold[:, 29:] = 0.0
    return TaskTrialSpec(
        inits={"mechanics.effector": CartesianState(pos=np.zeros((n_trials, 2)))},
        inputs=DelayedReachTaskInputs(
            CartesianState(pos=target),
            hold,
            np.ones_like(hold),
        ),
        targets={"mechanics.effector.pos": TargetSpec(target)},
        timeline=TrialTimeline.from_epochs_events(
            n_steps,
            epoch_bounds=np.broadcast_to(
                np.asarray([[0, 29, n_steps]], dtype=np.int32),
                (n_trials, 3),
            ),
            epoch_names=("prep", "movement"),
        ),
    )


def _nonuniform_endpoints() -> np.ndarray:
    return np.asarray(
        [
            [0.11, 0.0],
            [0.20, 0.0],
            [0.0, 0.12],
            [-0.13, 0.0],
        ],
        dtype=np.float32,
    )


def _nonuniform_delayed_trials() -> TaskTrialSpec:
    n_trials = 4
    n_steps = 90
    endpoints = _nonuniform_endpoints()
    target = np.broadcast_to(endpoints[:, None, :], (n_trials, n_steps, 2)).astype(
        np.float32
    )
    hold = np.ones((n_trials, n_steps, 1), dtype=np.float32)
    hold[:, 29:] = 0.0
    return TaskTrialSpec(
        inits={"mechanics.effector": CartesianState(pos=np.zeros((n_trials, 2)))},
        inputs=DelayedReachTaskInputs(
            CartesianState(pos=target),
            hold,
            np.ones_like(hold),
        ),
        targets={"mechanics.effector.pos": TargetSpec(target)},
        timeline=TrialTimeline.from_epochs_events(
            n_steps,
            epoch_bounds=np.broadcast_to(
                np.asarray([[0, 29, n_steps]], dtype=np.int32),
                (n_trials, 3),
            ),
            epoch_names=("prep", "movement"),
        ),
    )


def _adapted_delayed_trial_spec_template() -> TaskTrialSpec:
    n_trials = 2
    n_steps = 90
    init = np.zeros((n_trials, 48), dtype=np.float32)
    target = np.zeros((n_trials, n_steps, 2), dtype=np.float32)
    target[0, :, :] = np.asarray([0.15, 0.0], dtype=np.float32)
    target[1, :, :] = np.asarray([0.0, 0.15], dtype=np.float32)
    hold = np.ones((n_trials, n_steps, 1), dtype=np.float32)
    hold[:, 29:] = 0.0
    task_inputs = DelayedReachTaskInputs(
        CartesianState(pos=target),
        hold,
        np.ones_like(hold),
    )
    return TaskTrialSpec(
        inits={"mechanics.vector": init},
        inputs={
            "epsilon": np.zeros((n_trials, n_steps, 8), dtype=np.float32),
            "task": task_inputs,
            "input": 1.0 - hold,
            "target": target,
            "effector_target": CartesianState(pos=target),
        },
        targets={"mechanics.effector.pos": TargetSpec(target)},
        timeline=TrialTimeline.from_epochs_events(
            n_steps,
            epoch_bounds=np.asarray([[0, 29, n_steps], [0, 29, n_steps]]),
            epoch_names=("prep", "movement"),
        ),
    )
