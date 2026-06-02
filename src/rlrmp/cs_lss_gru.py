"""C&S LinearStateSpace GRU graph path.

This module provides the first scoped runtime path for issue ``3b2af27``:
the controller is a Feedbax ``SimpleStagedNetwork`` GRU, but the mechanics are
the exact canonical C&S 2019 discrete ``LinearStateSpace`` plant rather than
the legacy point-mass plus first-order force-filter ``SimpleFeedback`` plant.
The GRU observes only the 4D delayed position/velocity channel used by the
input/output certificate; it never receives the full 48D delay-augmented state.

The graph is deterministic by default. ``LinearStateSpace.epsilon`` is left
unbound unless the caller supplies an external epsilon input, so Feedbax's
component-level zero-epsilon default applies. Mapping the full stochastic C&S
training covariance contract into this runtime remains future work.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
from equinox import Module, field
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.channel import Channel, ChannelState
from feedbax.graph import Component, Graph, Wire
from feedbax.mechanics import LinearStateSpace
from feedbax.mechanics.linear_state_space import LinearStateSpaceState
from feedbax.noise import Multiplicative, Normal
from feedbax.nn import NetworkState, PopulationStructure, SimpleStagedNetwork
from feedbax.state import CartesianState

from rlrmp.analysis.cs_game_card import build_canonical_game
from rlrmp.analysis.feedbax_parity import build_cs2019_feedbax_mechanics


CS_DELAYED_POS_VEL_INDICES = (40, 41, 42, 43)
CS_EPSILON_DIM = 8
CS_FORCE_DIM = 2
CS_FEEDBACK_DIM = 4


class DelayedPositionVelocityFeedback(Component):
    """Select delayed physical position/velocity from the 48D C&S state.

    Inputs:
        state: C&S delay-augmented state vector, shape ``[48]``.

    Outputs:
        feedback: Delayed ``[px, py, vx, vy]`` vector, shape ``[4]``.
    """

    input_ports = ("state",)
    output_ports = ("feedback",)

    indices: tuple[int, ...] = field(static=True)

    def __init__(self, indices: tuple[int, ...] = CS_DELAYED_POS_VEL_INDICES):
        self.indices = tuple(int(index) for index in indices)
        if self.indices != CS_DELAYED_POS_VEL_INDICES:
            raise ValueError(
                "This first C&S GRU path is fixed to oldest delayed pos/vel indices "
                f"{CS_DELAYED_POS_VEL_INDICES}; got {self.indices}."
            )

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        vector = jnp.asarray(inputs["state"])
        if vector.shape[-1] != 48:
            raise ValueError(f"Expected a 48D C&S state vector; got shape {vector.shape}.")
        feedback = vector[jnp.asarray(self.indices, dtype=jnp.int32)]
        return {"feedback": feedback}, state


class CsLssMechanicsView(Module):
    """Mechanics view exposing semantic effector observables over the 48D state."""

    vector: Array
    effector: CartesianState


class CsLssGruState(Module):
    """SimpleFeedback-like state view for the C&S LSS GRU graph."""

    mechanics: CsLssMechanicsView
    sensory: ChannelState
    net: NetworkState
    efferent: ChannelState


def build_cs_lss_gru_graph(
    *,
    hidden_size: int,
    input_size: int = 0,
    encoding_size: int | None = None,
    hidden_type: Callable[..., Module] = eqx.nn.GRUCell,
    population_structure: PopulationStructure | None = None,
    sisu_gating: str = "additive",
    initial_state: Array | None = None,
    sensory_noise_std: float = 0.0,
    additive_motor_noise_std: float = 0.0,
    signal_dependent_motor_noise_std: float = 0.0,
    bind_epsilon_input: bool = False,
    key: PRNGKeyArray,
) -> Graph:
    """Build the C&S LinearStateSpace GRU feedback graph.

    Args:
        hidden_size: GRU hidden width.
        input_size: Extra task-input width supplied through the external
            ``"input"`` port. The recurrent feedback channel contributes four
            additional inputs internally.
        encoding_size: Optional network encoder width.
        hidden_type: Recurrent cell type; defaults to ``eqx.nn.GRUCell``.
        population_structure: Optional Feedbax population mask.
        sisu_gating: Feedbax SISU gating mode for ``SimpleStagedNetwork``.
        initial_state: Optional 48D C&S plant initial state.
        sensory_noise_std: Additive Gaussian noise standard deviation on the
            4D delayed position/velocity observation sent to the GRU.
        additive_motor_noise_std: Additive Gaussian command-channel motor
            noise standard deviation before the LSS plant input.
        signal_dependent_motor_noise_std: Multiplicative command-channel motor
            noise scale before the LSS plant input.
        bind_epsilon_input: If true, expose external graph port ``"epsilon"``
            and bind it to mechanics. If false, mechanics uses its zero-epsilon
            default.
        key: PRNG key for network construction.

    Returns:
        A Feedbax ``Graph`` with nodes ``feedback``, ``net``, ``efferent``, and
        ``mechanics``.
    """

    if input_size < 0:
        raise ValueError("input_size must be non-negative.")
    if sensory_noise_std < 0:
        raise ValueError("sensory_noise_std must be non-negative.")
    if additive_motor_noise_std < 0:
        raise ValueError("additive_motor_noise_std must be non-negative.")
    if signal_dependent_motor_noise_std < 0:
        raise ValueError("signal_dependent_motor_noise_std must be non-negative.")

    mechanics = build_cs2019_feedbax_mechanics(initial_state=initial_state)
    key_net = jr.fold_in(key, 0)
    net = SimpleStagedNetwork(
        input_size=input_size + CS_FEEDBACK_DIM,
        hidden_size=hidden_size,
        out_size=CS_FORCE_DIM,
        encoding_size=encoding_size,
        hidden_type=hidden_type,
        population_structure=population_structure,
        sisu_gating=sisu_gating,
        key=key_net,
    )
    sensory = Channel(
        delay=0,
        noise_func=Normal(std=float(sensory_noise_std)),
        add_noise=float(sensory_noise_std) != 0.0,
        input_proto=jnp.zeros((CS_FEEDBACK_DIM,), dtype=mechanics.A.dtype),
        init_value=0.0,
    )
    motor_noise_func = (
        Multiplicative(Normal(std=float(signal_dependent_motor_noise_std)))
        + Normal(std=float(additive_motor_noise_std))
    )
    efferent = Channel(
        delay=0,
        noise_func=motor_noise_func,
        add_noise=(
            float(additive_motor_noise_std) != 0.0
            or float(signal_dependent_motor_noise_std) != 0.0
        ),
        input_proto=jnp.zeros((CS_FORCE_DIM,), dtype=mechanics.A.dtype),
        init_value=0.0,
    )
    feedback = DelayedPositionVelocityFeedback()

    nodes = {
        "feedback": feedback,
        "sensory": sensory,
        "net": net,
        "efferent": efferent,
        "mechanics": mechanics,
    }
    wires = [
        Wire("feedback", "feedback", "sensory", "input"),
        Wire("sensory", "output", "net", "feedback"),
        Wire("net", "output", "efferent", "input"),
        Wire("efferent", "output", "mechanics", "force"),
        Wire("mechanics", "state", "feedback", "state", temporality="recurrent"),
    ]

    input_ports = ("input",) if input_size > 0 else ()
    input_bindings = {"input": ("net", "input")} if input_size > 0 else {}
    if bind_epsilon_input:
        input_ports = (*input_ports, "epsilon")
        input_bindings["epsilon"] = ("mechanics", "epsilon")

    def _state_view(node_states: dict[str, PyTree]) -> CsLssGruState:
        mechanics_state = node_states["mechanics"]
        mechanics_view = CsLssMechanicsView(
            vector=mechanics_state.vector,
            effector=mechanics._effector(
                mechanics_state.vector,
                jnp.zeros((CS_FORCE_DIM,), dtype=mechanics.A.dtype),
            ),
        )
        return CsLssGruState(
            mechanics=mechanics_view,
            sensory=node_states["sensory"],
            net=node_states["net"],
            efferent=node_states["efferent"],
        )

    return Graph(
        nodes=nodes,
        wires=tuple(wires),
        input_ports=input_ports,
        output_ports=("effector", "state", "feedback", "clean_feedback", "force"),
        input_bindings=input_bindings,
        output_bindings={
            "effector": ("mechanics", "effector"),
            "state": ("mechanics", "state"),
            "feedback": ("sensory", "output"),
            "clean_feedback": ("feedback", "feedback"),
            "force": ("efferent", "output"),
        },
        state_view_fn=_state_view,
    )


def cs_lss_gru_where_train() -> dict[int, Callable[[Graph], tuple[Module, Module | None]]]:
    """Return a train filter that excludes the fixed C&S plant matrices."""

    def where_train_fn(model: Graph) -> tuple[Module, Module | None]:
        net = model.nodes["net"]
        return (net.hidden, net.readout)

    return {0: where_train_fn}


def is_canonical_cs_lss_mechanics(mechanics: LinearStateSpace) -> bool:
    """Return whether mechanics matrices exactly match the canonical C&S plant."""

    plant, _schedule = build_canonical_game()
    return bool(
        jnp.allclose(mechanics.A, plant.A, atol=0.0)
        and jnp.allclose(mechanics.B, plant.B, atol=0.0)
        and jnp.allclose(mechanics.B_w, plant.Bw, atol=0.0)
    )
