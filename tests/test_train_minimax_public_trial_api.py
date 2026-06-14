"""Regression tests for the minimax trainer's public Feedbax trial API use."""

from __future__ import annotations

import argparse
import importlib.util

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import StateIndex
from feedbax import TaskTrialSpec, TrialTimeline, WhereDict
from feedbax.runtime.graph import Component, Graph
from feedbax.objectives.loss import CompositeLoss, ModelLoss

from rlrmp.paths import REPO_ROOT
from rlrmp.train.task_model import setup_task_model_pair


def _load_train_minimax_module():
    spec = importlib.util.spec_from_file_location(
        "train_minimax_under_test",
        REPO_ROOT / "scripts" / "train_minimax.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    )
    hps = train_minimax.build_hps(args)
    if hps.pert.type == "gusts":
        hps = hps | {"pert": hps.pert | {"type": "constant"}}
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    trainable = train_minimax._get_trainable(pair.model)
    where_trainable = train_minimax._trainable_where(pair.model)(pair.model)

    assert trainable[-1].shape[-1] == hps.model.hidden_size
    assert where_trainable[-1].shape == trainable[-1].shape
    assert jnp.all(where_trainable[-1] == pair.model.nodes["net"].sisu_alpha)
