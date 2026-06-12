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

from rlrmp.analysis.math.cs_game_card import build_canonical_game
from rlrmp.analysis.pipelines.feedbax_parity import build_cs2019_feedbax_mechanics


CS_PHYSICAL_STATE_DIM = 8
CS_REDUCED_PHYSICAL_STATE_DIM = 6
CS_DELAY_BLOCKS = 6
CS_DELAYED_POS_VEL_INDICES = (40, 41, 42, 43)
CS_DELAYED_POS_VEL_FORCE_INDICES = (40, 41, 42, 43, 44, 45)
CS_EPSILON_DIM = 8
CS_REDUCED_EPSILON_DIM = 6
CS_FORCE_DIM = 2
CS_FEEDBACK_DIM = 4
CS_PROPRIOCEPTIVE_FEEDBACK_DIM = 6
CS_TARGET_DIM = 2
CS_H0_CONTEXT_DIM = CS_FEEDBACK_DIM
CS_H0_ENCODER_INIT = "zero_affine"


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
    expected_state_dim: int = field(static=True)

    def __init__(
        self,
        indices: tuple[int, ...] = CS_DELAYED_POS_VEL_INDICES,
        *,
        expected_state_dim: int = CS_PHYSICAL_STATE_DIM * CS_DELAY_BLOCKS,
    ):
        self.indices = tuple(int(index) for index in indices)
        self.expected_state_dim = int(expected_state_dim)
        if len(self.indices) != CS_FEEDBACK_DIM:
            raise ValueError(f"Delayed pos/vel feedback needs 4 indices; got {self.indices}.")

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        vector = jnp.asarray(inputs["state"])
        if vector.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"Expected a {self.expected_state_dim}D C&S state vector; got shape "
                f"{vector.shape}."
            )
        feedback = vector[jnp.asarray(self.indices, dtype=jnp.int32)]
        return {"feedback": feedback}, state


class TargetRelativeDelayedFeedback(Component):
    """Return static-target-relative delayed feedback for C&S GRU controllers.

    The sign convention is ``[target_x - delayed_x, target_y - delayed_y,
    -delayed_vx, -delayed_vy]``. Static targets are task inputs and are not
    delayed; only the plant feedback is delayed by the C&S 48D state.
    """

    input_ports = ("state", "target")
    output_ports = ("feedback",)

    indices: tuple[int, ...] = field(static=True)
    expected_state_dim: int = field(static=True)

    def __init__(
        self,
        indices: tuple[int, ...] = CS_DELAYED_POS_VEL_INDICES,
        *,
        expected_state_dim: int = CS_PHYSICAL_STATE_DIM * CS_DELAY_BLOCKS,
    ):
        self.indices = tuple(int(index) for index in indices)
        self.expected_state_dim = int(expected_state_dim)
        if len(self.indices) != CS_FEEDBACK_DIM:
            raise ValueError(f"Target-relative delayed feedback needs 4 indices; got {self.indices}.")

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        vector = jnp.asarray(inputs["state"])
        target = jnp.asarray(inputs["target"])
        if vector.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"Expected a {self.expected_state_dim}D C&S state vector; got shape "
                f"{vector.shape}."
            )
        if target.shape[-1] != CS_TARGET_DIM:
            raise ValueError(f"Expected a 2D target vector; got shape {target.shape}.")
        delayed = jnp.take(vector, jnp.asarray(self.indices, dtype=jnp.int32), axis=-1)
        target = jnp.broadcast_to(target, delayed[..., :2].shape)
        feedback = jnp.concatenate([target - delayed[..., :2], -delayed[..., 2:4]], axis=-1)
        return {"feedback": feedback}, state


class TargetRelativeDelayedProprioceptiveFeedback(Component):
    """Return target-relative delayed pos/vel plus delayed force/filter state.

    The sign convention extends the 4D target-relative contract:
    ``[target_x - delayed_x, target_y - delayed_y, -delayed_vx, -delayed_vy,
    delayed_fx, delayed_fy]``. The final two coordinates expose the C&S
    force/filter state as a proprioceptive analogue without exposing the
    disturbance-integrator coordinates.
    """

    input_ports = ("state", "target")
    output_ports = ("feedback",)

    indices: tuple[int, ...] = field(static=True)
    expected_state_dim: int = field(static=True)

    def __init__(
        self,
        indices: tuple[int, ...] = CS_DELAYED_POS_VEL_FORCE_INDICES,
        *,
        expected_state_dim: int = CS_PHYSICAL_STATE_DIM * CS_DELAY_BLOCKS,
    ):
        self.indices = tuple(int(index) for index in indices)
        self.expected_state_dim = int(expected_state_dim)
        if len(self.indices) != CS_PROPRIOCEPTIVE_FEEDBACK_DIM:
            raise ValueError(
                "Target-relative proprioceptive delayed feedback needs 6 indices; "
                f"got {self.indices}."
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
        target = jnp.asarray(inputs["target"])
        if vector.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"Expected a {self.expected_state_dim}D C&S state vector; got shape "
                f"{vector.shape}."
            )
        if target.shape[-1] != CS_TARGET_DIM:
            raise ValueError(f"Expected a 2D target vector; got shape {target.shape}.")
        delayed = jnp.take(vector, jnp.asarray(self.indices, dtype=jnp.int32), axis=-1)
        target = jnp.broadcast_to(target, delayed[..., :2].shape)
        feedback = jnp.concatenate(
            [target - delayed[..., :2], -delayed[..., 2:4], delayed[..., 4:6]],
            axis=-1,
        )
        return {"feedback": feedback}, state


class InitialHiddenEncoder(Module):
    """Minimal affine map from trial-start feedback context to GRU hidden state."""

    weight: Array
    bias: Array

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        dtype: jnp.dtype = jnp.float32,
        init: str = CS_H0_ENCODER_INIT,
    ):
        if input_size <= 0:
            raise ValueError("H0 encoder input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("H0 encoder hidden_size must be positive.")
        if init != CS_H0_ENCODER_INIT:
            raise ValueError(
                f"Unknown H0 encoder init {init!r}; expected {CS_H0_ENCODER_INIT!r}."
            )
        self.weight = jnp.zeros((int(hidden_size), int(input_size)), dtype=dtype)
        self.bias = jnp.zeros((int(hidden_size),), dtype=dtype)

    def __call__(self, context: Array) -> Array:
        context = jnp.asarray(context, dtype=self.weight.dtype)
        return self.weight @ context + self.bias


class InitialHiddenStagedNetwork(Component):
    """Simple staged network with first-step H0 conditioning.

    The H0 encoder consumes the same controller-visible feedback vector that is
    wired into the GRU. It only replaces the recurrent hidden state on the first
    graph step; later steps use the GRU's own recurrent state.
    """

    input_ports = SimpleStagedNetwork.input_ports
    output_ports = SimpleStagedNetwork.output_ports

    net: SimpleStagedNetwork
    h0_encoder: InitialHiddenEncoder
    h0_state_index: eqx.nn.StateIndex
    h0_context_source: str = field(static=True)
    h0_initialization: str = field(static=True)

    def __init__(
        self,
        *,
        net: SimpleStagedNetwork,
        h0_encoder: InitialHiddenEncoder,
        h0_context_source: str = "target_relative_delayed_feedback",
        h0_initialization: str = CS_H0_ENCODER_INIT,
    ):
        self.net = net
        self.h0_encoder = h0_encoder
        self.h0_state_index = eqx.nn.StateIndex(jnp.asarray(False))
        self.h0_context_source = str(h0_context_source)
        self.h0_initialization = str(h0_initialization)

    @property
    def hidden(self) -> Module:
        return self.net.hidden

    @property
    def readout(self) -> Module | None:
        return self.net.readout

    @property
    def hidden_size(self) -> int:
        return self.net.hidden_size

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        applied = state.get(self.h0_state_index)
        feedback = jnp.asarray(inputs["feedback"])
        h0 = self.h0_encoder(feedback)
        net_state = state.get(self.net.state_index)
        initial_net_state = NetworkState(
            input=net_state.input,
            hidden=jnp.where(applied, net_state.hidden, h0),
            output=net_state.output,
            encoding=net_state.encoding,
        )
        state = state.set(self.net.state_index, initial_net_state)
        outputs, state = self.net(inputs, state, key=key)
        state = state.set(self.h0_state_index, jnp.asarray(True))
        return outputs, state

    def state_view(self, state: eqx.nn.State) -> NetworkState:
        return self.net.state_view(state)


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
    target_relative_feedback: bool = False,
    force_filter_feedback: bool = False,
    initial_hidden_encoder: bool = False,
    no_integrator_state: bool = False,
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
        target_relative_feedback: If true, replace raw delayed feedback with
            ``[target_x - delayed_x, target_y - delayed_y, -delayed_vx, -delayed_vy]``
            and expose external graph port ``"target"``.
        force_filter_feedback: If true with target-relative feedback, append
            delayed force/filter state ``[delayed_fx, delayed_fy]`` to the
            controller-visible feedback vector.
        initial_hidden_encoder: If true, initialize the GRU hidden state on the
            first graph step from the first controller-visible feedback vector.
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
    if initial_hidden_encoder and not target_relative_feedback:
        raise ValueError(
            "initial_hidden_encoder currently requires target_relative_feedback so "
            "the trial-start context is the controller-visible target-relative vector."
        )
    if force_filter_feedback and not target_relative_feedback:
        raise ValueError(
            "force_filter_feedback currently requires target_relative_feedback so "
            "the added force/filter coordinates are appended to a documented "
            "controller-visible basis."
        )

    physical_state_dim = (
        CS_REDUCED_PHYSICAL_STATE_DIM
        if bool(no_integrator_state)
        else CS_PHYSICAL_STATE_DIM
    )
    mechanics = build_cs2019_feedbax_mechanics(
        initial_state=initial_state,
        no_integrator_state=bool(no_integrator_state),
    )
    state_dim = int(mechanics.A.shape[0])
    delayed_pos_vel_indices = _delayed_feedback_indices(
        state_dim=state_dim,
        physical_state_dim=physical_state_dim,
        include_force=False,
    )
    delayed_pos_vel_force_indices = _delayed_feedback_indices(
        state_dim=state_dim,
        physical_state_dim=physical_state_dim,
        include_force=True,
    )
    feedback_dim = (
        CS_PROPRIOCEPTIVE_FEEDBACK_DIM
        if bool(force_filter_feedback)
        else CS_FEEDBACK_DIM
    )
    key_net = jr.fold_in(key, 0)
    net = SimpleStagedNetwork(
        input_size=input_size + feedback_dim,
        hidden_size=hidden_size,
        out_size=CS_FORCE_DIM,
        encoding_size=encoding_size,
        hidden_type=hidden_type,
        population_structure=population_structure,
        sisu_gating=sisu_gating,
        key=key_net,
    )
    if initial_hidden_encoder:
        net = InitialHiddenStagedNetwork(
            net=net,
            h0_encoder=InitialHiddenEncoder(
                input_size=feedback_dim,
                hidden_size=hidden_size,
                dtype=mechanics.A.dtype,
            ),
        )
    sensory = Channel(
        delay=0,
        noise_func=Normal(std=float(sensory_noise_std)),
        add_noise=float(sensory_noise_std) != 0.0,
        input_proto=jnp.zeros((feedback_dim,), dtype=mechanics.A.dtype),
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
    if bool(force_filter_feedback):
        feedback = TargetRelativeDelayedProprioceptiveFeedback(
            delayed_pos_vel_force_indices,
            expected_state_dim=state_dim,
        )
    elif bool(target_relative_feedback):
        feedback = TargetRelativeDelayedFeedback(
            delayed_pos_vel_indices,
            expected_state_dim=state_dim,
        )
    else:
        feedback = DelayedPositionVelocityFeedback(
            delayed_pos_vel_indices,
            expected_state_dim=state_dim,
        )

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
    if target_relative_feedback:
        input_ports = (*input_ports, "target")
        input_bindings["target"] = ("feedback", "target")
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


def _delayed_feedback_indices(
    *,
    state_dim: int,
    physical_state_dim: int,
    include_force: bool,
) -> tuple[int, ...]:
    if physical_state_dim < CS_REDUCED_PHYSICAL_STATE_DIM:
        raise ValueError(f"physical_state_dim must be >= 6; got {physical_state_dim}.")
    if state_dim % physical_state_dim != 0:
        raise ValueError(
            f"state_dim={state_dim} is not divisible by physical_state_dim={physical_state_dim}."
        )
    delayed_start = state_dim - physical_state_dim
    width = CS_PROPRIOCEPTIVE_FEEDBACK_DIM if include_force else CS_FEEDBACK_DIM
    return tuple(range(delayed_start, delayed_start + width))


def cs_lss_gru_where_train() -> dict[int, Callable[[Graph], tuple[Module, Module | None]]]:
    """Return a train filter that excludes the fixed C&S plant matrices."""

    def where_train_fn(model: Graph) -> tuple[Module, ...]:
        net = model.nodes["net"]
        if hasattr(net, "h0_encoder"):
            return (net.hidden, net.readout, net.h0_encoder)
        return (net.hidden, net.readout)

    return {0: where_train_fn}


def is_canonical_cs_lss_mechanics(mechanics: LinearStateSpace) -> bool:
    """Return whether mechanics matrices exactly match the canonical C&S plant."""

    plant, _schedule = build_canonical_game()
    if (
        mechanics.A.shape != plant.A.shape
        or mechanics.B.shape != plant.B.shape
        or mechanics.B_w.shape != plant.Bw.shape
    ):
        return False
    return bool(
        jnp.allclose(mechanics.A, plant.A, atol=0.0)
        and jnp.allclose(mechanics.B, plant.B, atol=0.0)
        and jnp.allclose(mechanics.B_w, plant.Bw, atol=0.0)
    )
