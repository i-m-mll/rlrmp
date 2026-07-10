"""Regression tests for the minimax trainer's public Feedbax trial API use."""

from __future__ import annotations
from rlrmp.io import load_named_python_module

import argparse
import importlib.util

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import StateIndex
from feedbax import TaskTrialSpec, TrialTimeline, WhereDict
from feedbax.runtime.graph import Component, Graph, init_state_from_component
from feedbax.objectives.loss import CompositeLoss, ModelLoss

from rlrmp.model.feedbax_graph import POINT_MASS_TARGET_POSITION_INPUT
from rlrmp.paths import REPO_ROOT
from rlrmp.train.task_model import setup_task_model_pair


def _load_train_minimax_module():
    return load_named_python_module('train_minimax_under_test', REPO_ROOT / 'scripts' / 'train_minimax.py')


class _NodeState(eqx.Module):
    hidden: jnp.ndarray


class _Passthrough(Component):
    input_ports = ("input",)
    output_ports = ("output",)

    state_index: StateIndex

    def __init__(self):
        self.state_index = StateIndex(_NodeState(hidden=jnp.zeros((2,), dtype=jnp.float32)))

    def __call__(self, inputs, state, *, key):
        return {"output": inputs["input"]}, state


def _graph() -> Graph:
    return Graph(
        nodes={"net": _Passthrough()},
        wires=(),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("net", "input")},
        output_bindings={"output": ("net", "output")},
    )


def test_streaming_minimax_eval_uses_public_prepared_trial() -> None:
    """The streaming loop should run through Feedbax's public ``prepare_trial``."""
    train_minimax = _load_train_minimax_module()
    trial_specs = TaskTrialSpec(
        inits=WhereDict(
            {
                "net.hidden": jnp.asarray(
                    [[1.0, 2.0], [3.0, 4.0]],
                    dtype=jnp.float32,
                )
            }
        ),
        targets=WhereDict(),
        inputs=jnp.zeros((2, 3, 2), dtype=jnp.float32),
        intervene={},
        timeline=TrialTimeline(n_steps=3),
    )
    loss = CompositeLoss(
        label="zero",
        terms={
            "zero": ModelLoss(
                "zero",
                lambda model: jnp.asarray(0.0, dtype=jnp.float32),
            )
        },
    )

    value = train_minimax._eval_trials_streaming(
        object(),
        _graph(),
        trial_specs,
        jr.split(jr.PRNGKey(0), 2),
        loss,
    )

    assert value == jnp.asarray(0.0, dtype=jnp.float32)


def test_multiplicative_minimax_adversarial_selector_includes_sisu_alpha() -> None:
    train_minimax = _load_train_minimax_module()
    args = argparse.Namespace(
        n_warmup_batches=10,
        n_adversary_batches=20,
        controller_lr=0.01,
        loss_update_enabled=False,
        loss_update_ratio=0.3,
        hidden_type="gru",
        sisu_gating="multiplicative",
        n_replicates=1,
    )
    hps = train_minimax.build_hps(args)
    if hps.pert.type == "gusts":
        hps = hps | {"pert": hps.pert | {"type": "constant"}}
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    trainable = train_minimax._get_trainable(pair.model)
    where_trainable = train_minimax._trainable_where(pair.model)(pair.model)

    assert trainable[-1].shape[-1] == hps.model.hidden_size
    assert where_trainable[-1].shape == trainable[-1].shape
    assert jnp.all(where_trainable[-1] == pair.model.nodes["net"].nodes["sisu_modulator"].gain)


def test_linear_tracker_minimax_selector_uses_affine_gain_and_feedforward() -> None:
    train_minimax = _load_train_minimax_module()
    args = argparse.Namespace(
        n_warmup_batches=10,
        n_adversary_batches=20,
        controller_lr=0.01,
        loss_update_enabled=False,
        loss_update_ratio=0.3,
        hidden_type="linear_tracker",
        sisu_gating="additive",
        n_replicates=1,
    )
    hps = train_minimax.build_hps(args)
    if hps.pert.type == "gusts":
        hps = hps | {"pert": hps.pert | {"type": "constant"}}
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    trainable = train_minimax._get_trainable(pair.model)
    where_trainable = train_minimax._trainable_where(pair.model)(pair.model)

    assert pair.model.nodes["net"].__class__.__name__ == "AffineFeedbackController"
    assert POINT_MASS_TARGET_POSITION_INPUT in pair.model.input_ports
    assert "net_state" not in pair.model.nodes
    assert hps.where["0"] == ["nodes.net.gain", "nodes.net.feedforward"]
    assert len(trainable) == len(where_trainable) == 2
    assert all(
        jnp.all(selected == expected)
        for selected, expected in zip(where_trainable, trainable, strict=True)
    )
    assert trainable[0] is pair.model.nodes["net"].gain
    assert trainable[1] is pair.model.nodes["net"].feedforward
    state_view = pair.model.state_view(init_state_from_component(pair.model))
    assert state_view.net.output.shape[-1] == 2
    assert state_view.net.hidden.shape[-1] == 2
