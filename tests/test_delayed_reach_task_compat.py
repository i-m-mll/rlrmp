"""Delayed-reach task-name compatibility tests."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from feedbax import DelayedReaches
from feedbax.objectives.loss import AbstractLoss

from rlrmp.task import TASK_TYPES


class _ToyLoss(AbstractLoss):
    label: str = "toy"

    def term(self, states, trial_specs, model):
        del states, trial_specs, model
        return jnp.asarray(0.0)


def test_legacy_center_out_delayed_reach_uses_feedbax_center_out_mode() -> None:
    task = TASK_TYPES["center_out_delayed_reach"](
        loss_func=_ToyLoss(),
        n_steps=20,
        workspace=[[-1.0, -1.0], [1.0, 1.0]],
        eval_reach_length=0.5,
        epoch_len_ranges=[[1, 2], [1, 2]],
        p_catch_trial=0.0,
    )

    trial = task.get_train_trial(jr.PRNGKey(0))

    assert isinstance(task, DelayedReaches)
    assert task.train_endpoint_mode == "center_out"
    assert task.preset is None
    assert jnp.allclose(trial.inits["mechanics.effector"].pos, jnp.zeros(2))


def test_legacy_cs_delayed_center_out_reach_uses_feedbax_preset() -> None:
    task = TASK_TYPES["cs_delayed_center_out_reach"](
        loss_func=_ToyLoss(),
        n_steps=20,
        workspace=[[-1.0, -1.0], [1.0, 1.0]],
        eval_reach_length=0.5,
        epoch_len_ranges=[[3, 3]],
        p_catch_trial=1.0,
    )

    trial = task.get_train_trial(jr.PRNGKey(0))

    assert isinstance(task, DelayedReaches)
    assert task.preset == "delayed_center_out"
    assert task.n_steps == 20
    assert trial.timeline.epoch_names == ("prep", "movement")
    assert trial.timeline.event_names == ("go_cue",)
    assert trial.extra is not None
    assert bool(trial.extra["is_catch_trial"])
    assert jnp.all(trial.inputs.target_on == 1.0)
    assert jnp.all(trial.inputs.hold == 1.0)


def test_feedbax_delayed_center_out_preset_uses_control_stage_alias() -> None:
    task = TASK_TYPES["delayed_reach"](
        loss_func=_ToyLoss(),
        preset="delayed_center_out",
        n_control_stages=8,
        workspace=[[-1.0, -1.0], [1.0, 1.0]],
        epoch_len_ranges=[[2, 2]],
        p_catch_trial=0.0,
    )

    assert isinstance(task, DelayedReaches)
    assert task.preset == "delayed_center_out"
    assert task.n_steps == 9
