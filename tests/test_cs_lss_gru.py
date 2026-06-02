"""Tests for the C&S LinearStateSpace GRU graph path."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from feedbax.graph import init_state_from_component
from feedbax.train import filter_spec_leaves, get_model_parameters

from rlrmp.analysis.cs_game_card import build_canonical_game
from rlrmp.cs_lss_gru import (
    CS_DELAYED_POS_VEL_INDICES,
    CS_EPSILON_DIM,
    DelayedPositionVelocityFeedback,
    build_cs_lss_gru_graph,
    cs_lss_gru_where_train,
    is_canonical_cs_lss_mechanics,
)


def test_feedback_selector_uses_oldest_delayed_position_velocity_block() -> None:
    selector = DelayedPositionVelocityFeedback()
    vector = jnp.arange(48, dtype=jnp.float64)
    state = init_state_from_component(selector)

    outputs, _ = selector({"state": vector}, state, key=jax.random.PRNGKey(0))

    assert outputs["feedback"].shape == (4,)
    assert tuple(outputs["feedback"].tolist()) == CS_DELAYED_POS_VEL_INDICES


def test_graph_first_step_feedback_is_seeded_from_initial_lss_state() -> None:
    initial_state = jnp.arange(48, dtype=jnp.float64)
    graph = build_cs_lss_gru_graph(
        hidden_size=5,
        initial_state=initial_state,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(1),
    )
    state = init_state_from_component(graph)
    epsilon = jnp.zeros((CS_EPSILON_DIM,), dtype=jnp.float64)

    outputs, _state, _cycle = graph.step(
        {"epsilon": epsilon},
        state,
        cycle_port_values=None,
        key=jax.random.PRNGKey(2),
    )

    expected_feedback = initial_state[jnp.array(CS_DELAYED_POS_VEL_INDICES)]
    assert jnp.allclose(outputs["feedback"], expected_feedback)


def test_lss_command_updates_force_filter_without_same_step_velocity_jump() -> None:
    plant, _schedule = build_canonical_game()
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(3),
    )
    mechanics = graph.nodes["mechanics"]
    command = jnp.array([1.25, -0.5], dtype=plant.A.dtype)
    state = init_state_from_component(mechanics)

    outputs, _ = mechanics(
        {
            "force": command,
            "epsilon": jnp.zeros((plant.m_w,), dtype=plant.A.dtype),
        },
        state,
        key=jax.random.PRNGKey(4),
    )

    assert is_canonical_cs_lss_mechanics(mechanics)
    assert jnp.allclose(outputs["state"][2:4], 0.0, atol=0.0)
    assert jnp.allclose(outputs["state"][4:6], plant.B[4:6] @ command, atol=1e-14)


def test_graph_runs_multiple_steps_with_zero_epsilon() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=6,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(5),
    )
    state = init_state_from_component(graph)
    epsilon = jnp.zeros((3, CS_EPSILON_DIM), dtype=jnp.float64)

    outputs, final_state, state_history = graph(
        {"epsilon": epsilon},
        state,
        key=jax.random.PRNGKey(6),
        return_state_history=True,
    )

    assert outputs["state"].shape == (3, 48)
    assert outputs["feedback"].shape == (3, 4)
    assert outputs["force"].shape == (3, 2)
    assert state_history.mechanics.vector.shape == (4, 48)
    assert state_history.mechanics.effector.pos.shape == (4, 2)
    assert state_history.mechanics.effector.vel.shape == (4, 2)
    assert jnp.allclose(state_history.mechanics.effector.pos, state_history.mechanics.vector[:, :2])
    assert jnp.allclose(state_history.mechanics.effector.vel, state_history.mechanics.vector[:, 2:4])
    assert state_history.net.hidden.shape == (4, 6)
    assert final_state is not None


def test_graph_omits_epsilon_binding_for_deterministic_default() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        bind_epsilon_input=False,
        key=jax.random.PRNGKey(8),
    )
    state = init_state_from_component(graph)

    outputs, _final_state = graph(
        {},
        state,
        key=jax.random.PRNGKey(9),
        n_steps=2,
    )

    assert graph.input_ports == ()
    assert outputs["state"].shape == (2, 48)
    assert outputs["feedback"].shape == (2, 4)


def test_train_filter_excludes_canonical_lss_matrices() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=6,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(7),
    )
    where_train = cs_lss_gru_where_train()[0]
    where_train_spec = filter_spec_leaves(graph, where_train)
    trainable = get_model_parameters(graph, where_train_spec)
    trainable_arrays = [leaf for leaf in jax.tree.leaves(trainable) if eqx.is_array(leaf)]
    mechanics = graph.nodes["mechanics"]

    assert trainable.nodes["mechanics"].A is None
    assert trainable.nodes["mechanics"].B is None
    assert trainable.nodes["mechanics"].B_w is None
    assert mechanics.A is graph.nodes["mechanics"].A
    assert any(leaf.shape == (18, 4) for leaf in trainable_arrays)
