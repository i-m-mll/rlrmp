"""RLRMP Feedbax GraphSpec builders and materializers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import feedbax.serialization_prototypes as _fbx_prototypes
from feedbax.bodies import FeedbackChannels, SimpleFeedback, SimpleFeedbackState
from feedbax.channel import Channel, ChannelSpec
from feedbax.component_registry import get_component_registry
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphMetadata,
    GraphSpec,
    RetainedObservableSpec,
    RetainedObservableTargetSpec,
    RetentionPolicySpec,
    WireSpec,
)
from feedbax.filters import FilterState
from feedbax.graph import Graph
from feedbax.intervene import (
    CurlField,
    CurlFieldParams,
    DynamicsMatrixPerturb,
    DynamicsMatrixPerturbParams,
    FixedField,
    FixedFieldParams,
)
from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.nn import PopulationStructure, SimpleStagedNetwork
from feedbax.noise import CompositeNoise, Multiplicative, Normal
from feedbax.serialization import spec_to_graph
from equinox.nn import StateIndex

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.stochastic_runtime import (
    PLANT_PROCESS_FORCE_NOISE_LABEL,
    PlantProcessForceNoise,
    command_motor_noise_func,
    graphspec_noise_contract,
    stochastic_runtime_config_from_model,
)


SCHEMA_VERSION = "rlrmp.feedbax_graph.v1"
EXECUTION_BACKEND = "feedbax.serialization.spec_to_graph"
GRAPH_PLANT_INTERVENOR_NODE = PLANT_INTERVENOR_LABEL


def _point_mass_feedback(state: MechanicsState):
    return (
        state.plant.skeleton.pos,
        state.plant.skeleton.vel,
    )


@dataclass(frozen=True)
class RLRMPFeedbaxGraphBundle:
    """Serializable graph contract plus adjacent training metadata."""

    graph_spec: GraphSpec
    task_spec: dict[str, Any]
    loss_spec: dict[str, Any]
    training_spec: dict[str, Any]
    manifest: dict[str, Any]

    def to_run_metadata(
        self,
        *,
        graph_spec_path: str | None = "model.graph.json",
        manifest_path: str = "model.graph.manifest.json",
    ) -> dict[str, Any]:
        """Return the compact metadata embedded into ``run.json``."""

        return {
            "schema_version": SCHEMA_VERSION,
            "graph_spec_path": graph_spec_path,
            "manifest_path": manifest_path,
            "graph_export_status": "unavailable" if graph_spec_path is None else "available",
            "execution_backend": EXECUTION_BACKEND,
            "component_policy": self.manifest["component_policy"],
            "legacy_loader": self.manifest["legacy_loader"],
        }


def register_rlrmp_graph_components(component_registry: Any | None = None) -> Any:
    """Register executable RLRMP component builders for GraphSpec materialization."""

    registry = component_registry or get_component_registry()
    registry.register_component_type(
        "RLRMPPointMass",
        _build_point_mass_mechanics,
        category="RLRMP",
        description="Point-mass mechanics preserving RLRMP mass and damping.",
        input_ports=["force"],
        output_ports=["effector", "state"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "RLRMPFeedbackChannels",
        _build_feedback_channels,
        category="RLRMP",
        description="Point-mass position/velocity feedback channels.",
        input_ports=["mechanics"],
        output_ports=["feedback"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "RLRMPMotorChannel",
        _build_motor_channel,
        category="RLRMP",
        description="RLRMP command channel with signal-dependent and additive motor noise.",
        input_ports=["input"],
        output_ports=["output"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "RLRMPSimpleStagedNetwork",
        _build_simple_staged_network,
        category="RLRMP",
        description="RLRMP SimpleStagedNetwork controller.",
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "RLRMPLinearController",
        _build_linear_controller,
        category="RLRMP",
        description="RLRMP linear regulator controller.",
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "RLRMPLinearTrackerController",
        _build_linear_tracker_controller,
        category="RLRMP",
        description="RLRMP linear tracker controller.",
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "FixedField",
        _build_fixed_field,
        category="Intervention",
        description="Fixed force-field intervention preserving GraphSpec label.",
        input_ports=["force", "params_override"],
        output_ports=["force"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "CurlField",
        _build_curl_field,
        category="Intervention",
        description="Curl force-field intervention preserving GraphSpec label.",
        input_ports=["effector", "force", "params_override"],
        output_ports=["force"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "DynamicsMatrixPerturb",
        _build_dynamics_matrix_perturb,
        category="RLRMP",
        description="State-feedback dynamics matrix perturbation.",
        input_ports=["effector", "force", "params_override"],
        output_ports=["force"],
        provenance="rlrmp",
    )
    registry.register_component_type(
        "RLRMPPlantProcessForceNoise",
        _build_plant_process_force_noise,
        category="RLRMP",
        description="Additive plant/load force noise before mechanics.",
        input_ports=["force"],
        output_ports=["force"],
        provenance="rlrmp",
    )
    return registry


def materialize_rlrmp_graph_spec(
    graph_spec: GraphSpec,
    component_registry: Any | None = None,
    *,
    install_runtime_hooks: bool = True,
) -> Graph:
    """Materialize an RLRMP GraphSpec through Feedbax and install runtime hooks."""

    registry = register_rlrmp_graph_components(component_registry)
    graph = _spec_to_graph_with_rlrmp_prototypes(graph_spec, registry)
    if install_runtime_hooks:
        graph = install_simple_feedback_runtime_hooks(graph)
    return graph


def _spec_to_graph_with_rlrmp_prototypes(graph_spec: GraphSpec, registry: Any) -> Graph:
    original = _fbx_prototypes.output_prototypes_for_node

    def _output_prototypes_for_node(
        node_name,
        node_spec,
        input_prototypes,
        subgraphs,
        *,
        strict=True,
    ):
        node_type = node_spec.type
        params = node_spec.params
        if node_type == "RLRMPPointMass":
            mechanics = _build_point_mass_mechanics(params)
            plant_state = mechanics.plant.init(key=jr.PRNGKey(0))
            effector = mechanics.plant.skeleton.effector(plant_state.skeleton)
            state = MechanicsState(plant=plant_state, effector=effector, solver=None)
            return {"effector": effector, "state": state}
        if node_type == "RLRMPFeedbackChannels":
            return {"feedback": (jnp.zeros(2), jnp.zeros(2))}
        if node_type in {"RLRMPMotorChannel", "RLRMPPlantProcessForceNoise"}:
            return {"output" if node_type == "RLRMPMotorChannel" else "force": jnp.zeros(2)}
        if node_type == "RLRMPSimpleStagedNetwork":
            hidden = jnp.zeros(int(params.get("hidden_size", 1)))
            output = jnp.zeros(int(params.get("out_size", 2)))
            return {"output": output, "hidden": hidden}
        if node_type in {"RLRMPLinearController", "RLRMPLinearTrackerController"}:
            controls = int(params.get("n_controls", 2))
            return {"output": jnp.zeros(controls), "hidden": jnp.zeros(1)}
        if node_type in {"FixedField", "CurlField", "DynamicsMatrixPerturb"}:
            proto = input_prototypes.get((node_name, "force"))
            return {"force": jnp.zeros(2) if proto is None else proto}
        return original(
            node_name,
            node_spec,
            input_prototypes,
            subgraphs,
            strict=strict,
        )

    _fbx_prototypes.output_prototypes_for_node = _output_prototypes_for_node
    try:
        return spec_to_graph(graph_spec, registry)
    finally:
        _fbx_prototypes.output_prototypes_for_node = original


def install_simple_feedback_runtime_hooks(graph: Graph) -> Graph:
    """Install SimpleFeedback-compatible state view and consistency hooks."""

    mechanics = graph.nodes.get("mechanics")
    feedback = graph.nodes.get("feedback")
    if not isinstance(mechanics, Mechanics) or not isinstance(feedback, FeedbackChannels):
        return graph

    def _state_view(node_states):
        force_filter_state = node_states.get("force_filter", FilterState(output=None, solver=None))
        return SimpleFeedbackState(
            mechanics=node_states["mechanics"],
            net=node_states["net"],
            feedback=node_states["feedback"],
            efferent=node_states["efferent"],
            force_filter=force_filter_state,
        )

    def _consistency_update(state):
        mechanics_state: MechanicsState = state.get(mechanics.state_index)
        new_skeleton = mechanics.plant.skeleton.inverse_kinematics(mechanics_state.effector)
        mechanics_state = eqx.tree_at(
            lambda s: s.plant.skeleton,
            mechanics_state,
            new_skeleton,
        )
        state = state.set(mechanics.state_index, mechanics_state)
        return feedback.fill_queues(state, mechanics_state)

    object.__setattr__(graph, "state_view_fn", _state_view)
    object.__setattr__(graph, "state_consistency_fn", _consistency_update)
    return graph


def _build_feedback_channels(params: dict[str, Any]) -> FeedbackChannels:
    delay = int(params.get("delay", 0))
    noise_std = float(params.get("noise_std", 0.0) or 0.0)
    add_noise = bool(params.get("add_noise", noise_std != 0.0))
    noise_func = Normal(std=noise_std) if add_noise and noise_std != 0.0 else None
    specs = ChannelSpec(
        where=_point_mass_feedback,
        delay=delay,
        noise_func=noise_func,
    )
    channel = Channel(
        delay=delay,
        noise_func=noise_func,
        add_noise=add_noise,
        input_proto=(jnp.zeros(2), jnp.zeros(2)),
        init_value=float(params.get("init_value", 0.0)),
    )
    return FeedbackChannels(channel, specs)


def _build_point_mass_mechanics(params: dict[str, Any]) -> Mechanics:
    return Mechanics(
        plant=DirectForceInput(
            PointMass(
                mass=float(params.get("mass", 1.0)),
                damping=float(params.get("damping", 0.0)),
            )
        ),
        dt=float(params.get("dt", 0.01)),
    )


def _build_motor_channel(params: dict[str, Any]) -> Channel:
    from rlrmp.stochastic_runtime import StochasticRuntimeConfig

    config = StochasticRuntimeConfig(
        additive_motor_noise_std=float(params.get("additive_noise_std", 0.0) or 0.0),
        signal_dependent_motor_noise_std=float(
            params.get("signal_dependent_noise_std", 0.0) or 0.0
        ),
    )
    return Channel(
        delay=int(params.get("delay", 0)),
        noise_func=command_motor_noise_func(config),
        add_noise=config.has_command_noise,
        input_proto=jnp.zeros(tuple(int(dim) for dim in params.get("input_shape", [2]))),
        init_value=float(params.get("init_value", 0.0)),
    )


def _build_simple_staged_network(params: dict[str, Any]) -> SimpleStagedNetwork:
    hidden_type_name = str(params.get("hidden_type", "GRUCell"))
    if hidden_type_name == "VanillaRNNCell":
        from rlrmp.models import VanillaRNNCell

        hidden_type = VanillaRNNCell
    elif hidden_type_name in {"GRU", "GRUCell", "gru"}:
        hidden_type = eqx.nn.GRUCell
    else:
        raise ValueError(f"Unsupported RLRMP hidden_type {hidden_type_name!r}")
    population_structure = _population_structure_from_params(params.get("population_structure"))
    key = _key_from_params(params)
    return SimpleStagedNetwork(
        input_size=int(params["input_size"]),
        hidden_size=int(params["hidden_size"]),
        out_size=int(params.get("out_size", 2)),
        encoding_size=params.get("encoding_size"),
        hidden_type=hidden_type,
        population_structure=population_structure,
        sisu_gating=str(params.get("sisu_gating", "additive")),
        key=key,
    )


def _build_linear_controller(params: dict[str, Any]):
    from rlrmp.networks.linear_controllers import LinearController

    return LinearController(
        n_steps=int(params["n_steps"]),
        n_controls=int(params.get("n_controls", 2)),
        n_states=int(params.get("n_states", 4)),
        K_init_scale=float(params.get("K_init_scale", 0.0) or 0.0),
        key=_key_from_params(params),
    )


def _build_linear_tracker_controller(params: dict[str, Any]):
    from rlrmp.networks.linear_controllers import LinearTrackerController

    return LinearTrackerController(
        n_steps=int(params["n_steps"]),
        n_controls=int(params.get("n_controls", 2)),
        n_states=int(params.get("n_states", 4)),
        K_init_scale=float(params.get("K_init_scale", 0.0) or 0.0),
        u_ff_init_scale=float(params.get("u_ff_init_scale", 0.0) or 0.0),
        key=_key_from_params(params),
    )


def _build_dynamics_matrix_perturb(params: dict[str, Any]) -> DynamicsMatrixPerturb:
    shape = params.get("delta_A_shape", [2, 4])
    return DynamicsMatrixPerturb(
        params=DynamicsMatrixPerturbParams(
            scale=float(params.get("scale", 1.0)),
            active=bool(params.get("active", False)),
            delta_A=jnp.zeros(tuple(int(dim) for dim in shape), dtype=jnp.float32),
        ),
        label=str(params.get("label", GRAPH_PLANT_INTERVENOR_NODE)),
        mass=float(params.get("mass", 1.0)),
    )


def _build_fixed_field(params: dict[str, Any]) -> FixedField:
    return FixedField(
        params=FixedFieldParams(
            scale=float(params.get("scale", 1.0)),
            amplitude=float(params.get("amplitude", 1.0)),
            field=jnp.asarray(params.get("field", [0.0, 0.0])),
            active=bool(params.get("active", False)),
        ),
        label=str(params.get("label", GRAPH_PLANT_INTERVENOR_NODE)),
    )


def _build_curl_field(params: dict[str, Any]) -> CurlField:
    return CurlField(
        params=CurlFieldParams(
            scale=float(params.get("scale", 1.0)),
            amplitude=float(params.get("amplitude", 1.0)),
            active=bool(params.get("active", False)),
        ),
        label=str(params.get("label", GRAPH_PLANT_INTERVENOR_NODE)),
    )


def _build_plant_process_force_noise(params: dict[str, Any]) -> PlantProcessForceNoise:
    return PlantProcessForceNoise(
        std=float(params.get("noise_std", 0.0)),
        label=str(params.get("label", PLANT_PROCESS_FORCE_NOISE_LABEL)),
    )


def _key_from_params(params: dict[str, Any]):
    key_data = params.get("key")
    if key_data is not None:
        return jnp.asarray(key_data, dtype=jnp.uint32)
    return jr.PRNGKey(int(params.get("seed", 0)))


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


def build_rlrmp_feedbax_graph_bundle(
    hps: Any,
    *,
    task: Any | None = None,
    n_extra_inputs: int = 0,
    hidden_type: Any | None = None,
    sisu_gating: str | None = None,
    key: Any | None = None,
) -> RLRMPFeedbaxGraphBundle:
    """Build the GraphSpec/manifest bundle for the current RLRMP model setup."""

    controller_kind = _controller_kind(hps)
    graph_spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        task=task,
        n_extra_inputs=n_extra_inputs,
        hidden_type=hidden_type,
        sisu_gating=sisu_gating,
        key=key,
        controller_kind=controller_kind,
        intervention_type="FixedField",
    )
    task_spec, loss_spec, training_spec, manifest = _graph_bundle_metadata(
        hps,
        controller_kind=controller_kind,
    )
    return RLRMPFeedbaxGraphBundle(
        graph_spec=graph_spec,
        task_spec=task_spec,
        loss_spec=loss_spec,
        training_spec=training_spec,
        manifest=manifest,
    )


def build_runtime_rlrmp_feedbax_graph_bundle(
    hps: Any,
    model: Graph,
) -> RLRMPFeedbaxGraphBundle:
    """Build a GraphSpec/manifest bundle from the constructed runtime model."""

    controller_kind = _controller_kind(hps)
    task_spec, loss_spec, training_spec, manifest = _graph_bundle_metadata(
        hps,
        controller_kind=controller_kind,
    )
    return RLRMPFeedbaxGraphBundle(
        graph_spec=graph_spec_from_model(
            model,
            n_replicates=int(getattr(hps.model, "n_replicates", 1)),
        ),
        task_spec=task_spec,
        loss_spec=loss_spec,
        training_spec=training_spec,
        manifest=manifest,
    )


def _graph_bundle_metadata(
    hps: Any,
    *,
    controller_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    task_spec = _task_spec(hps)
    loss_spec = _loss_spec(hps)
    training_spec = _training_spec(hps, controller_kind=controller_kind)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "execution_backend": EXECUTION_BACKEND,
        "component_policy": {
            "rlrmp_component_types": [
                "RLRMPPointMass",
                "RLRMPFeedbackChannels",
                "RLRMPSimpleStagedNetwork",
                "RLRMPLinearController",
                "RLRMPLinearTrackerController",
                "DynamicsMatrixPerturb",
                "RLRMPMotorChannel",
                "RLRMPPlantProcessForceNoise",
            ],
            "note": (
                "RLRMP component types are registered locally and materialized "
                "through feedbax.serialization.spec_to_graph."
            ),
        },
        "legacy_loader": {
            "setup_function": "rlrmp.modules.training.part2.setup_task_model_pair",
            "checkpoint_format": "feedbax._io.save/load_with_hyperparameters",
        },
        "task_spec": task_spec,
        "loss_spec": loss_spec,
        "training_spec": training_spec,
        "stochastic_runtime": graphspec_noise_contract(
            stochastic_runtime_config_from_model(hps.model)
        ),
    }
    return task_spec, loss_spec, training_spec, manifest


def build_point_mass_sensorimotor_graph_spec(
    hps: Any,
    *,
    task: Any | None = None,
    n_extra_inputs: int = 1,
    population_structure: PopulationStructure | None = None,
    hidden_type: Any | None = None,
    sisu_gating: str | None = None,
    key: Any | None = None,
    controller_kind: str | None = None,
    intervention_type: str | None = "FixedField",
) -> GraphSpec:
    """Serialize the point-mass feedback loop used by RLRMP training.

    The shape mirrors Feedbax ``SimpleFeedback``: feedback channel, controller,
    efferent channel, optional force filter, mechanics, and optional plant
    intervenor inserted before ``mechanics.force``.
    """

    if controller_kind is None:
        controller_kind = _controller_kind(hps)
    noise_config = stochastic_runtime_config_from_model(hps.model)
    if hidden_type is None:
        hidden_type = getattr(hps, "hidden_type", None)
    if sisu_gating is None:
        sisu_gating = str(getattr(hps, "sisu_gating", "additive"))
    key_param = None if key is None else [int(value) for value in jnp.asarray(key).tolist()]

    mechanics_params = {
        "dt": float(hps.dt),
        "mass": float(hps.model.effector_mass),
        "damping": float(hps.model.damping),
    }
    nodes: dict[str, ComponentSpec] = {
        "feedback": ComponentSpec(
            type="RLRMPFeedbackChannels",
            params={
                "delay": int(hps.model.feedback_delay_steps),
                "noise_std": noise_config.sensory_noise_std,
                "noise_role": "sensory_feedback",
                "add_noise": noise_config.sensory_noise_std != 0.0,
            },
            input_ports=["mechanics"],
            output_ports=["feedback"],
        ),
        "net": _controller_component_spec(
            hps,
            controller_kind,
            task=task,
            n_extra_inputs=n_extra_inputs,
            population_structure=population_structure,
            hidden_type=hidden_type,
            sisu_gating=sisu_gating,
            key=key_param,
        ),
        "efferent": ComponentSpec(
            type="RLRMPMotorChannel",
            params={
                "delay": 0,
                "additive_noise_std": noise_config.additive_motor_noise_std,
                "signal_dependent_noise_std": (noise_config.signal_dependent_motor_noise_std),
                "add_noise": noise_config.has_command_noise,
                "noise_model": "signal_dependent_plus_additive",
                "noise_role": "motor_command",
                "noise_timing": "pre_force_filter",
                "input_shape": [2],
            },
            input_ports=["input"],
            output_ports=["output"],
        ),
        "mechanics": ComponentSpec(
            type="RLRMPPointMass",
            params=mechanics_params,
            input_ports=["force"],
            output_ports=["effector", "state"],
        ),
    }
    wires: list[WireSpec] = [
        WireSpec(
            source_node="feedback",
            source_port="feedback",
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
            source_node="mechanics",
            source_port="state",
            target_node="feedback",
            target_port="mechanics",
            temporality="recurrent",
            recurrent_initializer={
                "kind": "state_output",
                "scope": "trial",
                "source": "state_initializer",
                "state_slot": "mechanics",
            },
        ),
    ]

    force_source = ("efferent", "output")
    tau_rise = float(getattr(hps.model, "tau_rise", 0.0) or 0.0)
    tau_decay = float(getattr(hps.model, "tau_decay", tau_rise) or 0.0)
    if tau_rise != 0.0 or tau_decay != 0.0:
        nodes["force_filter"] = ComponentSpec(
            type="FirstOrderFilter",
            params={
                "tau_rise": tau_rise,
                "tau_decay": tau_decay,
                "dt": float(hps.dt),
                "init_value": 0.0,
                "input_shape": [2],
            },
            input_ports=["input"],
            output_ports=["output"],
        )
        wires.append(
            WireSpec(
                source_node="efferent",
                source_port="output",
                target_node="force_filter",
                target_port="input",
            )
        )
        force_source = ("force_filter", "output")

    input_bindings: dict[str, tuple[str, str]] = {"input": ("net", "input")}
    if intervention_type is None:
        wires.append(
            WireSpec(
                source_node=force_source[0],
                source_port=force_source[1],
                target_node="mechanics",
                target_port="force",
            )
        )
    else:
        nodes[GRAPH_PLANT_INTERVENOR_NODE] = _intervention_component_spec(intervention_type, hps)
        wires.append(
            WireSpec(
                source_node=force_source[0],
                source_port=force_source[1],
                target_node=GRAPH_PLANT_INTERVENOR_NODE,
                target_port="force",
            )
        )
        if intervention_type in {"CurlField", "DynamicsMatrixPerturb"}:
            wires.append(
                WireSpec(
                    source_node="mechanics",
                    source_port="effector",
                    target_node=GRAPH_PLANT_INTERVENOR_NODE,
                    target_port="effector",
                    temporality="recurrent",
                    recurrent_initializer={
                        "kind": "state_output",
                        "scope": "trial",
                        "source": "state_initializer",
                        "state_slot": "effector",
                    },
                )
            )
        wires.append(
            WireSpec(
                source_node=GRAPH_PLANT_INTERVENOR_NODE,
                source_port="force",
                target_node="mechanics",
                target_port="force",
            )
        )
        input_bindings[f"intervene:{GRAPH_PLANT_INTERVENOR_NODE}"] = (
            GRAPH_PLANT_INTERVENOR_NODE,
            "params_override",
        )

    if noise_config.has_plant_process_force_noise:
        force_wires = [
            wire
            for wire in wires
            if wire.target_node == "mechanics" and wire.target_port == "force"
        ]
        if len(force_wires) != 1:
            raise ValueError(
                "Expected exactly one GraphSpec force wire into mechanics before "
                f"inserting {PLANT_PROCESS_FORCE_NOISE_LABEL!r}; found {len(force_wires)}"
            )
        force_wire = force_wires[0]
        wires.remove(force_wire)
        nodes[PLANT_PROCESS_FORCE_NOISE_LABEL] = ComponentSpec(
            type="RLRMPPlantProcessForceNoise",
            params={
                "noise_std": noise_config.plant_process_force_noise_std,
                "noise_role": "plant_process_load",
                "noise_timing": "post_force_filter_pre_mechanics",
                "state_diffusion": False,
            },
            input_ports=["force"],
            output_ports=["force"],
        )
        wires.extend(
            [
                WireSpec(
                    source_node=force_wire.source_node,
                    source_port=force_wire.source_port,
                    target_node=PLANT_PROCESS_FORCE_NOISE_LABEL,
                    target_port="force",
                ),
                WireSpec(
                    source_node=PLANT_PROCESS_FORCE_NOISE_LABEL,
                    source_port="force",
                    target_node="mechanics",
                    target_port="force",
                ),
            ]
        )

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=list(input_bindings),
        output_ports=["effector"],
        input_bindings=input_bindings,
        output_bindings={"effector": ("mechanics", "effector")},
        retained_observables=_retained_observables(
            include_plant_process_force_noise=noise_config.has_plant_process_force_noise,
        ),
        metadata=GraphMetadata(
            name="RLRMP point-mass sensorimotor loop",
            description=("Executable GraphSpec contract for RLRMP minimax training."),
            created_at="1970-01-01T00:00:00",
            updated_at="1970-01-01T00:00:00",
            version="1.0.0",
            tags=["rlrmp", "feedbax", "graphspec", "minimax"],
        ),
    )


def create_point_mass_graph_model(
    hps: Any,
    task: Any,
    *,
    n_extra_inputs: int = 1,
    population_structure: PopulationStructure | None = None,
    hidden_type: Any | None = None,
    sisu_gating: str = "additive",
    controller_kind: str | None = None,
    intervention_type: str = "FixedField",
    key: Any,
) -> Graph:
    """Build one executable point-mass model from an RLRMP GraphSpec."""

    spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        task=task,
        n_extra_inputs=n_extra_inputs,
        population_structure=population_structure,
        hidden_type=hidden_type,
        sisu_gating=sisu_gating,
        controller_kind=controller_kind,
        intervention_type=intervention_type,
        key=key,
    )
    return materialize_rlrmp_graph_spec(spec)


def create_point_mass_graph_ensemble(
    hps: Any,
    task: Any,
    *,
    n: int,
    key: Any,
    n_extra_inputs: int = 1,
    population_structure: PopulationStructure | None = None,
    hidden_type: Any | None = None,
    sisu_gating: str = "additive",
    controller_kind: str | None = None,
    intervention_type: str = "FixedField",
) -> Graph:
    """Build a batched ensemble of executable GraphSpec-materialized models."""

    keys = jr.split(key, int(n))
    models = [
        materialize_rlrmp_graph_spec(
            build_point_mass_sensorimotor_graph_spec(
                hps,
                task=task,
                n_extra_inputs=n_extra_inputs,
                population_structure=population_structure,
                hidden_type=hidden_type,
                sisu_gating=sisu_gating,
                controller_kind=controller_kind,
                intervention_type=intervention_type,
                key=key_i,
            ),
            install_runtime_hooks=False,
        )
        for key_i in keys
    ]
    template = models[0]
    models = [template, *[_align_state_index_markers(template, model) for model in models[1:]]]
    dynamic_models = [eqx.filter(model, eqx.is_array) for model in models]
    static_model = eqx.filter(template, lambda leaf: not eqx.is_array(leaf))
    stacked_dynamic = jt.map(lambda *leaves: jnp.stack(leaves), *dynamic_models)
    return install_simple_feedback_runtime_hooks(eqx.combine(stacked_dynamic, static_model))


def _align_state_index_markers(template: Graph, model: Graph) -> Graph:
    def _align(template_leaf, model_leaf):
        if isinstance(template_leaf, StateIndex) and isinstance(model_leaf, StateIndex):
            aligned = StateIndex(model_leaf.init)
            object.__setattr__(aligned, "marker", template_leaf.marker)
            return aligned
        return model_leaf

    return jt.map(
        _align,
        template,
        model,
        is_leaf=lambda leaf: isinstance(leaf, StateIndex),
    )


def graph_spec_from_model(
    model: Graph,
    *,
    n_replicates: int | None = None,
    replicate_index: int = 0,
) -> GraphSpec:
    """Return a GraphSpec for a materialized RLRMP runtime graph."""

    from feedbax.serialization import graph_to_spec

    if n_replicates is not None and n_replicates > 1:
        model = _representative_runtime_graph(
            model,
            n_replicates=n_replicates,
            replicate_index=replicate_index,
        )
    graph_spec = graph_to_spec(model)
    nodes = dict(graph_spec.nodes)
    subgraphs = dict(graph_spec.subgraphs or {})
    drop_subgraphs: set[str] = set()
    cs_lss_graph = _is_cs_lss_graph(model)
    for name, component in model.nodes.items():
        node_spec = nodes.get(name)
        if node_spec is None:
            continue
        cs_lss_spec = _cs_lss_runtime_component_spec(component) if cs_lss_graph else None
        if cs_lss_spec is not None:
            nodes[name] = cs_lss_spec
            drop_subgraphs.add(name)
        elif isinstance(component, Mechanics) and isinstance(component.plant, DirectForceInput):
            skeleton = component.plant.skeleton
            if isinstance(skeleton, PointMass):
                nodes[name] = ComponentSpec(
                    type="RLRMPPointMass",
                    params={
                        "dt": float(component.dt),
                        "mass": float(skeleton.mass),
                        "damping": float(skeleton.damping),
                    },
                    input_ports=list(component.input_ports),
                    output_ports=list(component.output_ports),
                )
        elif isinstance(component, FeedbackChannels):
            channel = component.channels
            noise_std = 0.0
            if isinstance(channel.noise_func, Normal):
                noise_std = float(channel.noise_func.std)
            nodes[name] = ComponentSpec(
                type="RLRMPFeedbackChannels",
                params={
                    "delay": int(channel.delay),
                    "noise_std": noise_std,
                    "noise_role": "sensory_feedback",
                    "add_noise": bool(channel.add_noise),
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
        elif isinstance(component, Channel) and name == "efferent":
            motor_noise = _runtime_motor_noise_params(component)
            nodes[name] = ComponentSpec(
                type="RLRMPMotorChannel",
                params={
                    "delay": int(component.delay),
                    "additive_noise_std": motor_noise["additive_noise_std"],
                    "signal_dependent_noise_std": motor_noise["signal_dependent_noise_std"],
                    "add_noise": bool(component.add_noise),
                    "noise_model": "signal_dependent_plus_additive",
                    "noise_role": "motor_command",
                    "noise_timing": "pre_force_filter",
                    "input_shape": _runtime_channel_input_shape(component),
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
        elif isinstance(component, DynamicsMatrixPerturb):
            nodes[name] = ComponentSpec(
                type="DynamicsMatrixPerturb",
                params={
                    "active": bool(component._initial_state.active),
                    "scale": float(component._initial_state.scale),
                    "delta_A_shape": [int(dim) for dim in component._initial_state.delta_A.shape],
                    "mass": float(component.mass),
                    "label": component.label,
                },
                input_ports=list(component.input_ports),
                output_ports=list(component.output_ports),
            )
        else:
            linear_params = _runtime_linear_controller_params(component)
            if linear_params is not None:
                nodes[name] = ComponentSpec(
                    type=linear_params.pop("component_type"),
                    params=linear_params,
                    input_ports=list(component.input_ports),
                    output_ports=list(component.output_ports),
                )
    for name in drop_subgraphs:
        subgraphs.pop(name, None)
    return graph_spec.model_copy(update={"nodes": nodes, "subgraphs": subgraphs or None})


def _is_cs_lss_graph(model: Graph) -> bool:
    from feedbax.mechanics import LinearStateSpace

    return isinstance(model.nodes.get("mechanics"), LinearStateSpace)


def _cs_lss_runtime_component_spec(component: Any) -> ComponentSpec | None:
    from feedbax.mechanics import LinearStateSpace
    from rlrmp.cs_lss_gru import (
        CS_FORCE_DIM,
        CS_LSS_DELAYED_FEEDBACK_COMPONENT,
        CS_LSS_INITIAL_HIDDEN_NET_COMPONENT,
        CS_LSS_TARGET_FEEDBACK_COMPONENT,
        CS_LSS_TARGET_PROPRIOCEPTIVE_FEEDBACK_COMPONENT,
        DelayedPositionVelocityFeedback,
        InitialHiddenStagedNetwork,
        TargetRelativeDelayedFeedback,
        TargetRelativeDelayedProprioceptiveFeedback,
    )

    if isinstance(component, LinearStateSpace):
        return ComponentSpec(
            type="LinearStateSpace",
            params={
                "A": component.A.tolist(),
                "B": component.B.tolist(),
                "B_w": component.B_w.tolist(),
                "dt": component.dt,
                "initial_state": list(component.initial_state),
                "pos_slice": list(component.pos_slice),
                "vel_slice": list(component.vel_slice),
            },
            input_ports=list(component.input_ports),
            output_ports=list(component.output_ports),
        )
    if isinstance(component, TargetRelativeDelayedProprioceptiveFeedback):
        return _cs_lss_feedback_component_spec(
            component,
            component_type=CS_LSS_TARGET_PROPRIOCEPTIVE_FEEDBACK_COMPONENT,
        )
    if isinstance(component, TargetRelativeDelayedFeedback):
        return _cs_lss_feedback_component_spec(
            component,
            component_type=CS_LSS_TARGET_FEEDBACK_COMPONENT,
        )
    if isinstance(component, DelayedPositionVelocityFeedback):
        return _cs_lss_feedback_component_spec(
            component,
            component_type=CS_LSS_DELAYED_FEEDBACK_COMPONENT,
        )
    if isinstance(component, InitialHiddenStagedNetwork):
        params = _cs_lss_simple_staged_network_params(component.net)
        params.update(
            {
                "h0_input_size": int(component.h0_encoder.weight.shape[1]),
                "h0_context_source": component.h0_context_source,
                "h0_initialization": component.h0_initialization,
            }
        )
        return ComponentSpec(
            type=CS_LSS_INITIAL_HIDDEN_NET_COMPONENT,
            params=params,
            input_ports=list(component.input_ports),
            output_ports=list(component.output_ports),
        )
    if isinstance(component, SimpleStagedNetwork):
        params = _cs_lss_simple_staged_network_params(component)
        params["out_size"] = int(params.get("out_size", CS_FORCE_DIM))
        return ComponentSpec(
            type="RLRMPSimpleStagedNetwork",
            params=params,
            input_ports=list(component.input_ports),
            output_ports=list(component.output_ports),
        )
    return None


def _cs_lss_feedback_component_spec(component: Any, *, component_type: str) -> ComponentSpec:
    return ComponentSpec(
        type=component_type,
        params={
            "indices": [int(index) for index in component.indices],
            "expected_state_dim": int(component.expected_state_dim),
            "feedback_dim": len(component.indices),
        },
        input_ports=list(component.input_ports),
        output_ports=list(component.output_ports),
    )


def _cs_lss_simple_staged_network_params(component: SimpleStagedNetwork) -> dict[str, Any]:
    return {
        "controller_kind": "gru",
        "input_size": int(component.input_size),
        "hidden_size": int(component.hidden_size),
        "out_size": int(component.out_size),
        "encoding_size": component.encoding_size,
        "hidden_type": type(component.hidden).__name__,
        "sisu_gating": component.sisu_gating,
        "population_structure": _runtime_population_structure_params(
            component.population_structure,
            hidden_size=int(component.hidden_size),
        ),
    }


def _runtime_population_structure_params(
    population_structure: PopulationStructure | None,
    *,
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


def _representative_runtime_graph(
    model: Graph,
    *,
    n_replicates: int,
    replicate_index: int,
) -> Graph:
    dynamic, static = eqx.partition(
        model,
        lambda leaf: (
            eqx.is_array(leaf)
            and leaf.ndim > 0
            and int(getattr(leaf, "shape", (0,))[0]) == int(n_replicates)
        ),
    )
    representative_dynamic = jt.map(
        lambda leaf: None if leaf is None else leaf[replicate_index],
        dynamic,
        is_leaf=lambda leaf: leaf is None,
    )
    representative = eqx.combine(representative_dynamic, static)
    for node_name, node in representative.nodes.items():
        state_index = getattr(node, "state_index", None)
        if not isinstance(state_index, StateIndex):
            continue
        changed = False

        def _unbatch_init_leaf(leaf: Any) -> Any:
            nonlocal changed
            if (
                eqx.is_array(leaf)
                and leaf.ndim > 0
                and int(getattr(leaf, "shape", (0,))[0]) == int(n_replicates)
            ):
                changed = True
                return leaf[replicate_index]
            return leaf

        init = jt.map(_unbatch_init_leaf, state_index.init)
        if changed:
            representative = eqx.tree_at(
                lambda graph, name=node_name: graph.nodes[name].state_index.init,
                representative,
                init,
            )
    return representative


def _runtime_motor_noise_params(channel: Channel) -> dict[str, float]:
    terms = ()
    if isinstance(channel.noise_func, CompositeNoise):
        terms = channel.noise_func.terms
    elif channel.noise_func is not None:
        terms = (channel.noise_func,)

    additive_noise_std = 0.0
    signal_dependent_noise_std = 0.0
    for term in terms:
        if isinstance(term, Multiplicative) and isinstance(term.noise_func, Normal):
            signal_dependent_noise_std = float(term.noise_func.std)
        elif isinstance(term, Normal):
            additive_noise_std = float(term.std)
    return {
        "additive_noise_std": additive_noise_std,
        "signal_dependent_noise_std": signal_dependent_noise_std,
    }


def _runtime_channel_input_shape(channel: Channel) -> list[int]:
    leaves = jt.leaves(channel._initial_state.output)
    if not leaves or not hasattr(leaves[0], "shape"):
        return [2]
    return [int(dim) for dim in leaves[0].shape]


def _runtime_linear_controller_params(component: Any) -> dict[str, Any] | None:
    from rlrmp.networks.linear_controllers import LinearController, LinearTrackerController

    if isinstance(component, LinearTrackerController):
        component_type = "RLRMPLinearTrackerController"
    elif isinstance(component, LinearController):
        component_type = "RLRMPLinearController"
    else:
        return None
    return {
        "component_type": component_type,
        "n_steps": int(component.n_steps),
        "n_controls": int(component.n_controls),
        "n_states": int(component.n_states),
        "target_source": "input.task.effector_target.pos",
        "feedback_order": ["pos_x", "pos_y", "vel_x", "vel_y"],
    }


def graph_spec_payload(graph_spec: GraphSpec) -> dict[str, Any]:
    """Return a JSON-serializable GraphSpec payload."""

    return graph_spec.model_dump(mode="json", exclude_none=True)


def write_graph_spec_bundle(bundle: RLRMPFeedbaxGraphBundle, spec_dir: Path) -> Path:
    """Write ``model.graph.json`` beside a run spec and return its path."""

    graph_path = spec_dir / "model.graph.json"
    graph_path.write_text(
        _json_dumps(graph_spec_payload(bundle.graph_spec)),
        encoding="utf-8",
    )
    manifest_path = spec_dir / "model.graph.manifest.json"
    manifest_path.write_text(_json_dumps(bundle.manifest), encoding="utf-8")
    return graph_path


def _controller_kind(hps: Any) -> str:
    hidden_type = getattr(hps, "hidden_type", None)
    if isinstance(hidden_type, str) and hidden_type in {"linear", "linear_tracker"}:
        return hidden_type
    name = getattr(hidden_type, "__name__", None)
    if name is None and hasattr(hidden_type, "func"):
        name = getattr(hidden_type.func, "__name__", None)
    return "vanilla_rnn" if name == "VanillaRNNCell" else "gru"


def _controller_component_spec(
    hps: Any,
    controller_kind: str,
    *,
    task: Any | None = None,
    n_extra_inputs: int = 1,
    population_structure: PopulationStructure | None = None,
    hidden_type: Any | None = None,
    sisu_gating: str = "additive",
    key: list[int] | None = None,
) -> ComponentSpec:
    if controller_kind == "linear":
        return ComponentSpec(
            type="RLRMPLinearController",
            params={**_linear_controller_params(hps), "key": key},
            input_ports=["input", "feedback"],
            output_ports=["output", "hidden"],
        )
    if controller_kind == "linear_tracker":
        return ComponentSpec(
            type="RLRMPLinearTrackerController",
            params={**_linear_controller_params(hps), "key": key},
            input_ports=["input", "feedback"],
            output_ports=["output", "hidden"],
        )
    input_size = None
    if task is not None:
        input_size = _point_mass_network_input_size(
            hps,
            task=task,
            n_extra_inputs=n_extra_inputs,
        )
    return ComponentSpec(
        type="RLRMPSimpleStagedNetwork",
        params={
            "controller_kind": controller_kind,
            "input_size": input_size,
            "input_size_source": "task-derived" if input_size is not None else "unresolved",
            "hidden_size": int(hps.model.hidden_size),
            "out_size": 2,
            "encoding_size": None,
            "hidden_type": _hidden_type_name(hidden_type),
            "sisu_gating": sisu_gating,
            "n_extra_inputs": int(n_extra_inputs),
            "population_structure": _population_structure_params(
                hps,
                population_structure,
            ),
            "key": key,
        },
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
    )


def _point_mass_network_input_size(
    hps: Any,
    *,
    task: Any,
    n_extra_inputs: int,
) -> int:
    mechanics = Mechanics(
        DirectForceInput(
            PointMass(
                mass=float(hps.model.effector_mass),
                damping=float(hps.model.damping),
            )
        ),
        float(hps.dt),
    )
    feedback_spec = {
        "where": _point_mass_feedback,
        "delay": int(hps.model.feedback_delay_steps),
        "noise_func": Normal(std=stochastic_runtime_config_from_model(hps.model).sensory_noise_std),
    }
    return SimpleFeedback.get_nn_input_size(task, mechanics, feedback_spec=feedback_spec) + int(
        n_extra_inputs
    )


def _hidden_type_name(hidden_type: Any | None) -> str:
    if hidden_type is None:
        return "GRUCell"
    if isinstance(hidden_type, str):
        return hidden_type
    name = getattr(hidden_type, "__name__", None)
    if name is None and hasattr(hidden_type, "func"):
        name = getattr(hidden_type.func, "__name__", None)
    return str(name or "GRUCell")


def _linear_controller_params(hps: Any) -> dict[str, Any]:
    return {
        "n_steps": int(hps.task.n_steps) - 1,
        "n_controls": 2,
        "n_states": 4,
        "target_source": "input.task.effector_target.pos",
        "feedback_order": ["pos_x", "pos_y", "vel_x", "vel_y"],
    }


def _intervention_component_spec(intervention_type: str, hps: Any) -> ComponentSpec:
    input_ports = ["force", "params_override"]
    if intervention_type in {"CurlField", "DynamicsMatrixPerturb"}:
        input_ports = ["effector", *input_ports]
    params: dict[str, Any] = {
        "active": False,
        "scale": 1.0,
        "label": GRAPH_PLANT_INTERVENOR_NODE,
    }
    if intervention_type == "FixedField":
        params.update({"amplitude": 1.0, "field": [0.0, 0.0]})
    elif intervention_type == "DynamicsMatrixPerturb":
        params.update(
            {
                "delta_A_shape": [2, 4],
                "mass": float(hps.model.effector_mass),
            }
        )
    return ComponentSpec(
        type=intervention_type,
        params=params,
        input_ports=input_ports,
        output_ports=["force"],
    )


def _retained_observables(
    *,
    include_plant_process_force_noise: bool = False,
) -> list[RetainedObservableSpec]:
    observables = [
        _port_observable("mechanics.effector", "mechanics", "effector"),
        _port_observable("mechanics.state", "mechanics", "state"),
        _port_observable("net.output", "net", "output"),
        _port_observable("net.hidden", "net", "hidden"),
        _port_observable("efferent.output", "efferent", "output"),
        _port_observable(
            f"{GRAPH_PLANT_INTERVENOR_NODE}.force",
            GRAPH_PLANT_INTERVENOR_NODE,
            "force",
        ),
    ]
    if include_plant_process_force_noise:
        observables.append(
            _port_observable(
                f"{PLANT_PROCESS_FORCE_NOISE_LABEL}.force",
                PLANT_PROCESS_FORCE_NOISE_LABEL,
                "force",
            )
        )
    return observables


def _port_observable(selector: str, node_id: str, port: str) -> RetainedObservableSpec:
    return RetainedObservableSpec(
        id=f"observable:{selector}",
        label=selector,
        selector=selector,
        target=RetainedObservableTargetSpec(
            kind="port",
            selector=selector,
            node_id=node_id,
            port=port,
            timing="output",
        ),
        retention=RetentionPolicySpec(mode="trajectory"),
    )


def _task_spec(hps: Any) -> dict[str, Any]:
    return {
        "type": str(hps.task.type),
        "n_steps": int(hps.task.n_steps),
        "workspace": _plain(hps.task.workspace),
        "eval_grid_n": int(hps.task.eval_grid_n),
        "eval_n_directions": int(hps.task.eval_n_directions),
        "eval_reach_length": float(hps.task.eval_reach_length),
        "epoch_len_ranges": _plain(hps.task.epoch_len_ranges),
        "target_on_epochs": _plain(hps.task.target_on_epochs),
        "hold_epochs": _plain(hps.task.hold_epochs),
        "move_epochs": _plain(hps.task.move_epochs),
        "p_catch_trial": float(hps.task.p_catch_trial),
        "extra_inputs": ["sisu", f"intervene:{GRAPH_PLANT_INTERVENOR_NODE}"],
    }


def _loss_spec(hps: Any) -> dict[str, Any]:
    return {
        "weights": _plain(hps.loss.weights),
        "effector_pos_late": _plain(hps.loss.effector_pos_late),
        "effector_vel_late": _plain(hps.loss.effector_vel_late),
        "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
        "effector_hold_pos_schedule": str(hps.loss.effector_hold_pos_schedule),
        "position_powerlaw_power": float(hps.loss.position_powerlaw_power),
        "movement_ramp_shape": str(hps.loss.movement_ramp_shape),
        "movement_ramp_duration_steps": int(hps.loss.movement_ramp_duration_steps),
        "movement_ramp_power": float(hps.loss.movement_ramp_power),
    }


def _training_spec(hps: Any, *, controller_kind: str) -> dict[str, Any]:
    trainable = (
        ["nodes.net.K"] if controller_kind == "linear" else ["nodes.net.K", "nodes.net.u_ff"]
    )
    if controller_kind not in {"linear", "linear_tracker"}:
        trainable = ["nodes.net.hidden", "nodes.net.readout"]
        if str(getattr(hps, "sisu_gating", "additive")) == "multiplicative":
            trainable.append("nodes.net.sisu_alpha")
    return {
        "dt": float(hps.dt),
        "batch_size": int(hps.batch_size),
        "n_replicates": int(hps.model.n_replicates),
        "controller_kind": controller_kind,
        "trainable": trainable,
        "method": str(hps.method),
        "loss_update": _plain(hps.loss_update),
        "stochastic_runtime": graphspec_noise_contract(
            stochastic_runtime_config_from_model(hps.model)
        ),
    }


def _population_structure_params(
    hps: Any,
    population_structure: PopulationStructure | None,
) -> dict[str, int]:
    pop = population_structure or getattr(hps.model, "population_structure", None)
    if pop is None:
        return {}
    return {
        "hidden_size": int(hps.model.hidden_size),
        "n_input_only": int(getattr(pop, "n_input_only", 0) or 0),
        "n_readout_only": int(getattr(pop, "n_readout_only", 0) or 0),
        "n_recurrent_only": int(getattr(pop, "n_recurrent_only", 0) or 0),
        "n_input_readout": int(getattr(pop, "n_input_readout", 0) or 0),
    }


def _plain(value: Any) -> Any:
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
