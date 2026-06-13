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
from typing import Any

import equinox as eqx
from equinox import Module, field
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.channel import ChannelState
from feedbax.component_registry import get_component_registry
from feedbax.contracts.graph import ComponentSpec, GraphMetadata, GraphSpec, WireSpec
from feedbax.graph import Component, Graph
from feedbax.mechanics import LinearStateSpace
from feedbax.nn import NetworkState, PopulationStructure, SimpleStagedNetwork
from feedbax.serialization import spec_to_graph
from feedbax.state import CartesianState

from rlrmp.analysis.math.cs_game_card import build_canonical_game
from rlrmp.analysis.pipelines.feedbax_parity import build_cs2019_feedbax_mechanics
from rlrmp.feedbax_graph import register_rlrmp_graph_components


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
CS_LSS_GRAPH_SPEC_VERSION = "1.0.0"
CS_LSS_DELAYED_FEEDBACK_COMPONENT = "RLRMPCsLssDelayedPositionVelocityFeedback"
CS_LSS_TARGET_FEEDBACK_COMPONENT = "RLRMPCsLssTargetRelativeDelayedFeedback"
CS_LSS_TARGET_PROPRIOCEPTIVE_FEEDBACK_COMPONENT = (
    "RLRMPCsLssTargetRelativeDelayedProprioceptiveFeedback"
)
CS_LSS_INITIAL_HIDDEN_NET_COMPONENT = "RLRMPCsLssInitialHiddenStagedNetwork"
FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT = "StateFeedbackSelector"


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
                f"Expected a {self.expected_state_dim}D C&S state vector; got shape {vector.shape}."
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
            raise ValueError(
                f"Target-relative delayed feedback needs 4 indices; got {self.indices}."
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
                f"Expected a {self.expected_state_dim}D C&S state vector; got shape {vector.shape}."
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
                f"Expected a {self.expected_state_dim}D C&S state vector; got shape {vector.shape}."
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
            raise ValueError(f"Unknown H0 encoder init {init!r}; expected {CS_H0_ENCODER_INIT!r}.")
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


def register_cs_lss_graph_components(component_registry: Any | None = None) -> Any:
    """Register executable CS-LSS component builders for GraphSpec materialization."""

    registry = register_rlrmp_graph_components(component_registry or get_component_registry())
    registry.register_component_type(
        CS_LSS_INITIAL_HIDDEN_NET_COMPONENT,
        _build_initial_hidden_staged_network,
        category="RLRMP",
        description="C&S GRU controller with first-step target-relative H0 conditioning.",
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        output_prototype_fn=_staged_network_output_prototype,
        provenance="rlrmp",
    )
    _install_cs_lss_registry_output_prototypes(registry)
    return registry


def build_cs_lss_gru_graph_spec(
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
) -> GraphSpec:
    """Build the durable GraphSpec for the C&S LinearStateSpace GRU graph."""

    _validate_cs_lss_graph_args(
        input_size=input_size,
        sensory_noise_std=sensory_noise_std,
        additive_motor_noise_std=additive_motor_noise_std,
        signal_dependent_motor_noise_std=signal_dependent_motor_noise_std,
        target_relative_feedback=target_relative_feedback,
        force_filter_feedback=force_filter_feedback,
        initial_hidden_encoder=initial_hidden_encoder,
    )
    physical_state_dim = (
        CS_REDUCED_PHYSICAL_STATE_DIM if bool(no_integrator_state) else CS_PHYSICAL_STATE_DIM
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
        CS_PROPRIOCEPTIVE_FEEDBACK_DIM if bool(force_filter_feedback) else CS_FEEDBACK_DIM
    )
    feedback_type, feedback_params, feedback_ports = _feedback_component_contract(
        target_relative_feedback=target_relative_feedback,
        force_filter_feedback=force_filter_feedback,
        delayed_pos_vel_indices=delayed_pos_vel_indices,
        delayed_pos_vel_force_indices=delayed_pos_vel_force_indices,
        state_dim=state_dim,
        feedback_dim=feedback_dim,
    )
    key_param = [int(value) for value in jnp.asarray(jr.fold_in(key, 0)).tolist()]
    net_type = (
        CS_LSS_INITIAL_HIDDEN_NET_COMPONENT
        if bool(initial_hidden_encoder)
        else "RLRMPSimpleStagedNetwork"
    )
    net_params = {
        "controller_kind": "gru",
        "input_size": int(input_size) + feedback_dim,
        "hidden_size": int(hidden_size),
        "out_size": CS_FORCE_DIM,
        "encoding_size": encoding_size,
        "hidden_type": _hidden_type_name(hidden_type),
        "sisu_gating": str(sisu_gating),
        "population_structure": _population_structure_params(population_structure, hidden_size),
        "key": key_param,
    }
    if initial_hidden_encoder:
        net_params.update(
            {
                "h0_input_size": feedback_dim,
                "h0_context_source": "target_relative_delayed_feedback",
                "h0_initialization": CS_H0_ENCODER_INIT,
                "h0_dtype": jnp.dtype(mechanics.A.dtype).name,
            }
        )

    nodes = {
        "feedback": ComponentSpec(
            type=feedback_type,
            params=feedback_params,
            input_ports=feedback_ports,
            output_ports=["feedback"],
        ),
        "sensory": ComponentSpec(
            type="Channel",
            params={
                "delay": 0,
                "noise_std": float(sensory_noise_std),
                "add_noise": float(sensory_noise_std) != 0.0,
                "input_shape": [feedback_dim],
            },
            input_ports=["input"],
            output_ports=["output"],
        ),
        "net": ComponentSpec(
            type=net_type,
            params=net_params,
            input_ports=["input", "feedback"],
            output_ports=["output", "hidden"],
        ),
        "efferent": ComponentSpec(
            type="Channel",
            params={
                "delay": 0,
                "additive_noise_std": float(additive_motor_noise_std),
                "signal_dependent_noise_std": float(signal_dependent_motor_noise_std),
                "add_noise": (
                    float(additive_motor_noise_std) != 0.0
                    or float(signal_dependent_motor_noise_std) != 0.0
                ),
                "noise_model": "signal_dependent_plus_additive",
                "noise_role": "motor_command",
                "noise_timing": "pre_lss_mechanics",
                "input_shape": [CS_FORCE_DIM],
            },
            input_ports=["input"],
            output_ports=["output"],
        ),
        "mechanics": ComponentSpec(
            type="LinearStateSpace",
            params={
                "A": mechanics.A.tolist(),
                "B": mechanics.B.tolist(),
                "B_w": mechanics.B_w.tolist(),
                "dt": mechanics.dt,
                "initial_state": list(mechanics.initial_state),
                "pos_slice": list(mechanics.pos_slice),
                "vel_slice": list(mechanics.vel_slice),
            },
            input_ports=["force", "epsilon"],
            output_ports=["effector", "state"],
        ),
    }
    wires = [
        WireSpec(
            source_node="feedback",
            source_port="feedback",
            target_node="sensory",
            target_port="input",
        ),
        WireSpec(
            source_node="sensory",
            source_port="output",
            target_node="net",
            target_port="feedback",
        ),
        WireSpec(
            source_node="net",
            source_port="output",
            target_node="efferent",
            target_port="input",
        ),
        WireSpec(
            source_node="efferent",
            source_port="output",
            target_node="mechanics",
            target_port="force",
        ),
        WireSpec(
            source_node="mechanics",
            source_port="state",
            target_node="feedback",
            target_port="state",
            temporality="recurrent",
        ),
    ]
    input_ports = ["input"] if input_size > 0 else []
    input_bindings = {"input": ("net", "input")} if input_size > 0 else {}
    if target_relative_feedback:
        input_ports.append("target")
        input_bindings["target"] = ("feedback", "target")
    if bind_epsilon_input:
        input_ports.append("epsilon")
        input_bindings["epsilon"] = ("mechanics", "epsilon")

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=input_ports,
        output_ports=["effector", "state", "feedback", "clean_feedback", "force"],
        input_bindings=input_bindings,
        output_bindings={
            "effector": ("mechanics", "effector"),
            "state": ("mechanics", "state"),
            "feedback": ("sensory", "output"),
            "clean_feedback": ("feedback", "feedback"),
            "force": ("efferent", "output"),
        },
        metadata=GraphMetadata(
            name="RLRMP CS-LSS GRU loop",
            description="Executable GraphSpec contract for the C&S LinearStateSpace GRU path.",
            created_at="1970-01-01T00:00:00",
            updated_at="1970-01-01T00:00:00",
            version=CS_LSS_GRAPH_SPEC_VERSION,
            tags=["rlrmp", "feedbax", "graphspec", "cs_lss", "gru"],
        ),
    )


def materialize_cs_lss_gru_graph_spec(
    graph_spec: GraphSpec,
    component_registry: Any | None = None,
) -> Graph:
    """Materialize a CS-LSS GraphSpec and install the CS-LSS runtime hooks."""

    registry = register_cs_lss_graph_components(component_registry)
    graph = spec_to_graph(_migrate_legacy_cs_lss_graph_spec(graph_spec), registry)
    return install_cs_lss_gru_runtime_hooks(graph)


def install_cs_lss_gru_runtime_hooks(graph: Graph) -> Graph:
    """Install CS-LSS state-view hooks on a materialized graph."""

    mechanics = graph.nodes.get("mechanics")
    if not isinstance(mechanics, LinearStateSpace):
        return graph

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

    return graph.with_state_view(_state_view)


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

    graph_spec = build_cs_lss_gru_graph_spec(
        hidden_size=hidden_size,
        input_size=input_size,
        encoding_size=encoding_size,
        hidden_type=hidden_type,
        population_structure=population_structure,
        sisu_gating=sisu_gating,
        initial_state=initial_state,
        sensory_noise_std=sensory_noise_std,
        additive_motor_noise_std=additive_motor_noise_std,
        signal_dependent_motor_noise_std=signal_dependent_motor_noise_std,
        bind_epsilon_input=bind_epsilon_input,
        target_relative_feedback=target_relative_feedback,
        force_filter_feedback=force_filter_feedback,
        initial_hidden_encoder=initial_hidden_encoder,
        no_integrator_state=no_integrator_state,
        key=key,
    )
    return materialize_cs_lss_gru_graph_spec(graph_spec)


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


def _validate_cs_lss_graph_args(
    *,
    input_size: int,
    sensory_noise_std: float,
    additive_motor_noise_std: float,
    signal_dependent_motor_noise_std: float,
    target_relative_feedback: bool,
    force_filter_feedback: bool,
    initial_hidden_encoder: bool,
) -> None:
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


def _feedback_component_contract(
    *,
    target_relative_feedback: bool,
    force_filter_feedback: bool,
    delayed_pos_vel_indices: tuple[int, ...],
    delayed_pos_vel_force_indices: tuple[int, ...],
    state_dim: int,
    feedback_dim: int,
) -> tuple[str, dict[str, Any], list[str]]:
    if force_filter_feedback:
        return (
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            _state_feedback_selector_params(
                indices=delayed_pos_vel_force_indices,
                expected_state_dim=state_dim,
                target_relative=True,
            ),
            ["state", "target"],
        )
    if target_relative_feedback:
        return (
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            _state_feedback_selector_params(
                indices=delayed_pos_vel_indices,
                expected_state_dim=state_dim,
                target_relative=True,
            ),
            ["state", "target"],
        )
    return (
        FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
        _state_feedback_selector_params(
            indices=delayed_pos_vel_indices,
            expected_state_dim=state_dim,
            target_relative=False,
        ),
        ["state"],
    )


def _state_feedback_selector_params(
    *,
    indices: tuple[int, ...],
    expected_state_dim: int,
    target_relative: bool,
) -> dict[str, Any]:
    if len(indices) not in {CS_FEEDBACK_DIM, CS_PROPRIOCEPTIVE_FEEDBACK_DIM}:
        raise ValueError(f"CS-LSS feedback selectors require 4 or 6 indices; got {indices}.")
    state_slices: dict[str, dict[str, list[int]]] = {
        "delayed_position": {"indices": [int(index) for index in indices[:2]]},
        "delayed_velocity": {"indices": [int(index) for index in indices[2:4]]},
    }
    channels: list[dict[str, Any]]
    if target_relative:
        channels = [
            {
                "slice": "delayed_position",
                "transform": "target_minus",
                "target_slice": [0, 2],
            },
            {"slice": "delayed_velocity", "transform": "negate"},
        ]
    else:
        channels = [
            {"slice": "delayed_position", "transform": "identity"},
            {"slice": "delayed_velocity", "transform": "identity"},
        ]
    if len(indices) == CS_PROPRIOCEPTIVE_FEEDBACK_DIM:
        state_slices["delayed_force"] = {"indices": [int(index) for index in indices[4:6]]}
        channels.append({"slice": "delayed_force", "transform": "identity"})
    return {
        "state_slices": state_slices,
        "channels": channels,
        "expected_state_dim": int(expected_state_dim),
        "output_size": len(indices),
    }


def _migrate_legacy_cs_lss_graph_spec(graph_spec: GraphSpec) -> GraphSpec:
    nodes: dict[str, ComponentSpec] = {}
    for node_id, node_spec in graph_spec.nodes.items():
        if node_spec.type in {
            CS_LSS_DELAYED_FEEDBACK_COMPONENT,
            CS_LSS_TARGET_FEEDBACK_COMPONENT,
            CS_LSS_TARGET_PROPRIOCEPTIVE_FEEDBACK_COMPONENT,
        }:
            params = dict(node_spec.params)
            indices = tuple(int(index) for index in params["indices"])
            nodes[node_id] = node_spec.model_copy(
                update={
                    "type": FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
                    "params": _state_feedback_selector_params(
                        indices=indices,
                        expected_state_dim=int(params["expected_state_dim"]),
                        target_relative=node_spec.type != CS_LSS_DELAYED_FEEDBACK_COMPONENT,
                    ),
                }
            )
        else:
            nodes[node_id] = node_spec
    return graph_spec.model_copy(update={"nodes": nodes})


def _build_initial_hidden_staged_network(
    params: dict[str, Any],
) -> InitialHiddenStagedNetwork:
    net = _build_simple_staged_network(params)
    return InitialHiddenStagedNetwork(
        net=net,
        h0_encoder=InitialHiddenEncoder(
            input_size=int(params["h0_input_size"]),
            hidden_size=int(params["hidden_size"]),
            dtype=jnp.dtype(params.get("h0_dtype", jnp.float32)),
        ),
        h0_context_source=str(params.get("h0_context_source", "target_relative_delayed_feedback")),
        h0_initialization=str(params.get("h0_initialization", CS_H0_ENCODER_INIT)),
    )


def _build_simple_staged_network(params: dict[str, Any]) -> SimpleStagedNetwork:
    return SimpleStagedNetwork(
        input_size=int(params["input_size"]),
        hidden_size=int(params["hidden_size"]),
        out_size=int(params.get("out_size", CS_FORCE_DIM)),
        encoding_size=params.get("encoding_size"),
        hidden_type=_hidden_type_from_name(str(params.get("hidden_type", "GRUCell"))),
        population_structure=_population_structure_from_params(params.get("population_structure")),
        sisu_gating=str(params.get("sisu_gating", "additive")),
        key=_key_from_params(params),
    )


def _hidden_type_from_name(name: str) -> Callable[..., Module]:
    if name == "VanillaRNNCell":
        from rlrmp.models import VanillaRNNCell

        return VanillaRNNCell
    if name in {"GRU", "GRUCell", "gru"}:
        return eqx.nn.GRUCell
    raise ValueError(f"Unsupported CS-LSS hidden_type {name!r}")


def _hidden_type_name(hidden_type: Any | None) -> str:
    if hidden_type is None:
        return "GRUCell"
    if isinstance(hidden_type, str):
        return hidden_type
    name = getattr(hidden_type, "__name__", None)
    if name is None and hasattr(hidden_type, "func"):
        name = getattr(hidden_type.func, "__name__", None)
    return str(name or "GRUCell")


def _key_from_params(params: dict[str, Any]) -> Array:
    key_data = params.get("key")
    if key_data is not None:
        return jnp.asarray(key_data, dtype=jnp.uint32)
    return jr.PRNGKey(int(params.get("seed", 0)))


def _population_structure_params(
    population_structure: PopulationStructure | None,
    hidden_size: int,
) -> dict[str, int]:
    if population_structure is None:
        return {}
    return {
        "hidden_size": int(hidden_size),
        "n_input_only": int(getattr(population_structure, "n_input_only", 0) or 0),
        "n_readout_only": int(getattr(population_structure, "n_readout_only", 0) or 0),
        "n_recurrent_only": int(getattr(population_structure, "n_recurrent_only", 0) or 0),
        "n_input_readout": int(getattr(population_structure, "n_input_readout", 0) or 0),
    }


def _population_structure_from_params(params: Any) -> PopulationStructure | None:
    if not params:
        return None
    return PopulationStructure.create(
        hidden_size=int(params["hidden_size"]),
        n_input_only=int(params.get("n_input_only", 0) or 0),
        n_readout_only=int(params.get("n_readout_only", 0) or 0),
        n_recurrent_only=int(params.get("n_recurrent_only", 0) or 0),
        n_input_readout=int(params.get("n_input_readout", 0) or 0),
        assignment_fn=None,
        key=jr.PRNGKey(int(params.get("seed", 0))),
    )


def _install_cs_lss_registry_output_prototypes(registry: Any) -> None:
    for name, output_prototype_fn in {
        "Channel": _motor_channel_output_prototype,
        "RLRMPSimpleStagedNetwork": _staged_network_output_prototype,
        "LinearStateSpace": _linear_state_space_output_prototype,
    }.items():
        meta = registry.get(name)
        if meta is None:
            raise ValueError(f"CS-LSS GraphSpec materialization requires registered {name!r}.")
        meta.output_prototype_fn = output_prototype_fn


def _feedback_output_prototype(
    params: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    del inputs
    return {"feedback": jnp.zeros((int(params["feedback_dim"]),))}


def _staged_network_output_prototype(
    params: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    del inputs
    hidden = jnp.zeros((int(params["hidden_size"]),))
    return {
        "output": jnp.zeros((int(params.get("out_size", CS_FORCE_DIM)),)),
        "hidden": hidden,
    }


def _motor_channel_output_prototype(
    params: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    del inputs
    return {"output": jnp.zeros(tuple(int(dim) for dim in params.get("input_shape", [2])))}


def _linear_state_space_output_prototype(
    params: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    del inputs
    state_dim = len(params["A"])
    dtype = jnp.asarray(params["A"]).dtype
    return {
        "effector": CartesianState(
            pos=jnp.zeros((CS_TARGET_DIM,), dtype=dtype),
            vel=jnp.zeros((CS_TARGET_DIM,), dtype=dtype),
            force=jnp.zeros((CS_FORCE_DIM,), dtype=dtype),
        ),
        "state": jnp.zeros((state_dim,), dtype=dtype),
    }


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
