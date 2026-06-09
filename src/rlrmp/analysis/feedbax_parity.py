"""Feedbax materialization of the canonical C&S game-card plant.

This module is the Phase 2 bridge for issue ``020a65b`` under umbrella
``43e8728``. It keeps the C&S plant definition in ``cs_game_card`` as the
source of truth, then exposes that exact discrete linear system as a Feedbax
mechanics component and GraphSpec node.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from feedbax.mechanics import LinearStateSpace
from feedbax.web.models.graph import ComponentSpec, GraphSpec
from jaxtyping import Array, Float

from rlrmp.analysis.cs_game_card import (
    assert_physical_selector_bw,
    build_canonical_game,
    build_no_integrator_game,
)
from rlrmp.analysis.hinf_riccati import PlantLinearization


PHYSICAL_STATE_LABELS: tuple[str, ...] = (
    "px",
    "py",
    "vx",
    "vy",
    "fx",
    "fy",
    "eps_x_int",
    "eps_y_int",
)
DEFAULT_NODE_ID = "cs2019_mechanics"


@dataclass(frozen=True)
class StateMapEntry:
    """One coordinate in the 48D delay-augmented C&S plant state."""

    index: int
    block: int
    delay_steps: int
    label: str


def canonical_state_map() -> tuple[StateMapEntry, ...]:
    """Return the fixed 48D C&S state-coordinate map.

    The order is the Phase 0 contract:
    ``[x_t, x_(t-1), ..., x_(t-5)]``, with each 8D block using
    ``PHYSICAL_STATE_LABELS``.
    """

    rows: list[StateMapEntry] = []
    for block in range(6):
        for offset, label in enumerate(PHYSICAL_STATE_LABELS):
            rows.append(
                StateMapEntry(
                    index=block * len(PHYSICAL_STATE_LABELS) + offset,
                    block=block,
                    delay_steps=block,
                    label=label,
                )
            )
    return tuple(rows)


def build_cs2019_feedbax_mechanics(
    *,
    initial_state: Float[Array, "48"] | None = None,
    no_integrator_state: bool = False,
) -> LinearStateSpace:
    """Build the canonical C&S plant as a Feedbax linear state-space mechanics.

    The returned component is a direct materialization of
    ``build_canonical_game()[0]``. The disturbance input is the 8D epsilon
    channel from the game card, not a physical force perturbation adapter.
    """

    if no_integrator_state:
        plant, _ = build_no_integrator_game()
    else:
        plant, _ = build_canonical_game()
        assert_physical_selector_bw(plant)

    if initial_state is None:
        initial_state = jnp.zeros(plant.n, dtype=plant.A.dtype)

    return LinearStateSpace(
        A=plant.A,
        B=plant.B,
        B_w=plant.Bw,
        dt=plant.dt,
        initial_state=initial_state,
        pos_slice=plant.pos_slice,
        vel_slice=plant.vel_slice,
    )


def build_cs2019_feedbax_graph_spec(
    *,
    node_id: str = DEFAULT_NODE_ID,
    initial_state: Float[Array, "48"] | None = None,
) -> GraphSpec:
    """Build a Feedbax GraphSpec containing the canonical C&S mechanics node."""

    plant, _ = build_canonical_game()
    assert_physical_selector_bw(plant)

    if initial_state is None:
        initial_state = jnp.zeros(plant.n, dtype=plant.A.dtype)

    params = _plant_to_linear_state_space_params(plant, initial_state)
    return GraphSpec(
        nodes={
            node_id: ComponentSpec(
                type="LinearStateSpace",
                params=params,
                input_ports=["force", "epsilon"],
                output_ports=["effector", "state"],
            )
        },
        input_ports=["force", "epsilon"],
        output_ports=["effector", "state"],
        input_bindings={
            "force": (node_id, "force"),
            "epsilon": (node_id, "epsilon"),
        },
        output_bindings={
            "effector": (node_id, "effector"),
            "state": (node_id, "state"),
        },
    )


def _plant_to_linear_state_space_params(
    plant: PlantLinearization,
    initial_state: Float[Array, "48"],
) -> dict[str, object]:
    return {
        "A": _array_param(plant.A),
        "B": _array_param(plant.B),
        "B_w": _array_param(plant.Bw),
        "dt": float(plant.dt),
        "initial_state": _array_param(initial_state),
        "pos_slice": list(plant.pos_slice),
        "vel_slice": list(plant.vel_slice),
    }


def _array_param(array: Array) -> list[object]:
    return np.asarray(array, dtype=np.float64).tolist()
