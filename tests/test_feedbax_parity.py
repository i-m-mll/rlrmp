"""Phase 2 Feedbax parity tests for the canonical C&S game card."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from feedbax.graph import init_state_from_component
from feedbax.mechanics import LinearStateSpace
from feedbax.web.serialization import spec_to_graph

from rlrmp.analysis.cs_game_card import build_canonical_game
from rlrmp.analysis.feedbax_parity import (
    DEFAULT_NODE_ID,
    PHYSICAL_STATE_LABELS,
    build_cs2019_feedbax_graph_spec,
    build_cs2019_feedbax_mechanics,
    canonical_state_map,
)


def _step_component(
    component: LinearStateSpace,
    *,
    force: jax.Array | None = None,
    epsilon: jax.Array | None = None,
) -> jax.Array:
    plant, _ = build_canonical_game()
    if force is None:
        force = jnp.zeros(plant.m_u, dtype=plant.A.dtype)
    if epsilon is None:
        epsilon = jnp.zeros(plant.m_w, dtype=plant.A.dtype)
    state = init_state_from_component(component)
    outputs, _ = component(
        {"force": force, "epsilon": epsilon},
        state,
        key=jax.random.PRNGKey(0),
    )
    return outputs["state"]


def test_feedbax_mechanics_materializes_canonical_game_card() -> None:
    plant, _ = build_canonical_game()
    component = build_cs2019_feedbax_mechanics()

    assert isinstance(component, LinearStateSpace)
    assert component.A.shape == (48, 48)
    assert component.B.shape == (48, 2)
    assert component.B_w.shape == (48, 8)
    assert component.dt == plant.dt
    assert component.pos_slice == plant.pos_slice
    assert component.vel_slice == plant.vel_slice
    assert jnp.allclose(component.A, plant.A, atol=0.0)
    assert jnp.allclose(component.B, plant.B, atol=0.0)
    assert jnp.allclose(component.B_w, plant.Bw, atol=0.0)


def test_feedbax_mechanics_recovers_bw_columns_from_epsilon_basis() -> None:
    plant, _ = build_canonical_game()

    for column in range(plant.m_w):
        epsilon = jnp.eye(plant.m_w, dtype=plant.A.dtype)[column]
        component = build_cs2019_feedbax_mechanics()
        next_state = _step_component(component, epsilon=epsilon)

        assert jnp.allclose(next_state, plant.Bw[:, column], atol=1e-14)
        assert jnp.allclose(next_state[:8], plant.Bw[:8, column], atol=1e-14)
        assert jnp.allclose(next_state[8:], 0.0, atol=1e-14)


def test_feedbax_mechanics_recovers_a_columns_from_state_basis() -> None:
    plant, _ = build_canonical_game()

    for column in range(plant.n):
        initial_state = jnp.eye(plant.n, dtype=plant.A.dtype)[column]
        component = build_cs2019_feedbax_mechanics(initial_state=initial_state)
        next_state = _step_component(component)

        assert jnp.allclose(next_state, plant.A[:, column], atol=1e-14)


def test_feedbax_graph_spec_round_trip_preserves_canonical_update() -> None:
    plant, _ = build_canonical_game()
    spec = build_cs2019_feedbax_graph_spec()

    node = spec.nodes[DEFAULT_NODE_ID]
    assert node.type == "LinearStateSpace"
    assert node.params["pos_slice"] == [0, 2]
    assert node.params["vel_slice"] == [2, 4]
    assert node.params["B_w"][:8] == jnp.eye(8, dtype=plant.A.dtype).tolist()

    graph = spec_to_graph(spec, {})
    state = init_state_from_component(graph)
    force = jnp.zeros(plant.m_u, dtype=plant.A.dtype)
    epsilon = jnp.arange(1, plant.m_w + 1, dtype=plant.A.dtype)
    outputs, _ = graph(
        {"force": force, "epsilon": epsilon},
        state,
        key=jax.random.PRNGKey(1),
    )

    assert isinstance(graph.nodes[DEFAULT_NODE_ID], LinearStateSpace)
    assert jnp.allclose(outputs["state"], plant.Bw @ epsilon, atol=1e-14)
    assert jnp.allclose(outputs["effector"].pos, outputs["state"][:2], atol=1e-14)
    assert jnp.allclose(outputs["effector"].vel, outputs["state"][2:4], atol=1e-14)


def test_canonical_state_map_covers_48_delay_augmented_coordinates() -> None:
    state_map = canonical_state_map()

    assert len(state_map) == 48
    assert tuple(row.index for row in state_map) == tuple(range(48))
    assert tuple(row.label for row in state_map[:8]) == PHYSICAL_STATE_LABELS
    assert tuple(row.label for row in state_map[40:]) == PHYSICAL_STATE_LABELS
    assert tuple(row.delay_steps for row in state_map[:8]) == (0,) * 8
    assert tuple(row.delay_steps for row in state_map[40:]) == (5,) * 8
