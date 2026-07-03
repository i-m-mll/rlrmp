"""Feedbax-backed controller adapters for RLRMP analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.analysis import GraphControllerAdapter, graph_controller
from feedbax.runtime.channel import Channel
from feedbax.runtime.graph import Graph, Wire
from feedbax.mechanics import MechanicsState
from feedbax.runtime.state import CartesianState
from jaxtyping import Array, PRNGKeyArray


@dataclass(frozen=True)
class SimpleFeedbackInducedGainController:
    """Use Feedbax graph execution for a SimpleFeedback controller subgraph.

    The induced-gain analyzer owns the plant state and disturbances, so this
    adapter wraps only the original ``feedback -> net`` controller path. The
    Feedbax ``GraphControllerAdapter`` owns graph/state packing and executes
    the original feedback channel, preserving delay queues and network state.
    """

    adapter: GraphControllerAdapter
    mechanics_template: MechanicsState
    task_input: Array
    target_pos: Array
    delay: int
    n_obs: int = 4

    @property
    def h0_flat(self) -> Array:
        """Initial flat state exposed for legacy runner metadata."""
        return self.adapter.h0_flat

    def initial_state(self) -> Array:
        """Return the Feedbax adapter's initial controller state."""
        return self.adapter.initial_state()

    def _mechanics_state_from_observation(self, sensory_obs: Array) -> MechanicsState:
        """Build a point-mass MechanicsState from analyzer-frame observation."""
        pos_goal_centered = jnp.asarray(sensory_obs[:2])
        vel = jnp.asarray(sensory_obs[2:4])
        pos_abs = pos_goal_centered + self.target_pos
        zero_force = jnp.zeros_like(vel)
        skeleton = CartesianState(pos=pos_abs, vel=vel, force=zero_force)
        effector = CartesianState(pos=pos_abs, vel=vel, force=zero_force)
        return eqx.tree_at(
            lambda state: (state.plant.skeleton, state.effector),
            self.mechanics_template,
            (skeleton, effector),
        )

    def step(self, h: Array, sensory_obs: Array, t: int) -> tuple[Array, Array]:
        """Advance the controller subgraph one analysis timestep."""
        inputs = {
            "task": self.task_input,
            "mechanics": self._mechanics_state_from_observation(sensory_obs),
        }
        return self.adapter.step(h, inputs, t)


def _task_input(target_pos: Array, *, sisu: float, dtype: Any) -> Array:
    target_pos_arr = jnp.asarray(target_pos, dtype=dtype)
    return jnp.concatenate(
        [
            target_pos_arr,
            jnp.zeros((2,), dtype=dtype),
            jnp.zeros((1,), dtype=dtype),
            jnp.ones((1,), dtype=dtype),
            jnp.array([float(sisu)], dtype=dtype),
        ],
        axis=0,
    )


def _feedback_delay(feedback_node: Any) -> int:
    channels = feedback_node.channels
    if hasattr(channels, "delay"):
        return int(channels.delay)
    for channel in jt.leaves(channels, is_leaf=lambda x: hasattr(x, "delay")):
        if hasattr(channel, "delay"):
            return int(channel.delay)
    return 0


def _disable_feedback_noise(feedback_node: Any) -> Any:
    channels = jt.map(
        lambda channel: eqx.tree_at(lambda c: c.add_noise, channel, False),
        feedback_node.channels,
        is_leaf=lambda x: isinstance(x, Channel),
    )
    return eqx.tree_at(lambda node: node.channels, feedback_node, channels)


def simple_feedback_induced_gain_controller(
    model: Graph,
    *,
    target_pos: Array,
    sisu: float = 0.5,
    key: PRNGKeyArray = jr.PRNGKey(0),
    dtype: Any = jnp.float64,
) -> GraphControllerAdapter:
    """Build a Feedbax-backed induced-gain controller for SimpleFeedback models.

    Args:
        model: A Feedbax ``SimpleFeedback``-shaped graph with ``feedback`` and
            ``net`` nodes.
        target_pos: Absolute target position, shape ``(2,)``.
        sisu: Held SISU scalar appended to the task input.
        key: Base key for deterministic per-step graph execution.
        dtype: Flat controller-state dtype.

    Returns:
        A controller structurally compatible with Feedbax's
        ``GraphControllerAdapter`` interface (``initial_state`` / ``step``).
    """
    feedback_node = _disable_feedback_noise(model.nodes["feedback"])
    net = model.nodes["net"]
    task_input = _task_input(target_pos, sisu=sisu, dtype=dtype)
    expected_task_size = int(net.input_size) - 4
    if int(task_input.shape[0]) != expected_task_size:
        raise ValueError(
            f"Constructed task_input has dim {task_input.shape[0]} but "
            f"net expects {expected_task_size} (= {net.input_size} - 4 feedback)."
        )

    controller_graph = Graph(
        nodes={"feedback": feedback_node, "net": net},
        wires=(Wire("feedback", "feedback", "net", "feedback"),),
        input_ports=("task", "mechanics"),
        output_ports=("output",),
        input_bindings={
            "task": ("net", "input"),
            "mechanics": ("feedback", "mechanics"),
        },
        output_bindings={"output": ("net", "output")},
    )
    adapter = graph_controller(
        controller_graph,
        key=key,
        input_port="task",
        output_port="output",
        dtype=dtype,
    )
    full_state = model.init_state(key=jr.PRNGKey(0))
    mechanics_template = full_state.get(model.nodes["mechanics"].state_index)
    target_pos_arr = jnp.asarray(target_pos, dtype=dtype)
    return SimpleFeedbackInducedGainController(
        adapter=adapter,
        mechanics_template=mechanics_template,
        task_input=task_input,
        target_pos=target_pos_arr,
        delay=_feedback_delay(feedback_node),
    )


__all__ = [
    "SimpleFeedbackInducedGainController",
    "simple_feedback_induced_gain_controller",
]
