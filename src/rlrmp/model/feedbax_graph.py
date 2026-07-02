"""RLRMP Feedbax GraphSpec builders and materializers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.models.feedback import FeedbackChannels, SimpleFeedbackState
from feedbax.component_registry import (
    ComponentMigration,
    ComponentMigrationPack,
    get_component_registry,
)
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphMetadata,
    GraphSpec,
    RetainedObservableSpec,
    RetainedObservableTargetSpec,
    RetentionPolicySpec,
    StudioTaskBindingSpec,
    WireSpec,
)
from feedbax.contracts.graphs.templates import (
    recurrent_controller_template_graph,
    recurrent_graph_input_initializer,
)
from feedbax.runtime.filters import FilterState
from feedbax.runtime.graph import Graph, GraphState
from feedbax.runtime.components import ElementwiseAffineModulator as RuntimeElementwiseAffineModulator
from feedbax.runtime.components import GRU as RuntimeGRU
from feedbax.runtime.components import Linear as RuntimeLinear
from feedbax.intervene import DynamicsMatrixPerturb
from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.models.networks import (
    NetworkState,
    PopulationStructure,
    SimpleStagedNetwork,
    population_structure_from_spec,
)
from feedbax.runtime.noise import Normal
from feedbax.contracts.graphs.serialization import spec_to_graph
from equinox.nn import StateIndex

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.io import compact_json_dumps
from rlrmp.model.stochastic_runtime import (
    PLANT_PROCESS_FORCE_NOISE_LABEL,
    graphspec_noise_contract,
    stochastic_runtime_config_from_model,
)
from rlrmp.model.trainable import staged_network_trainable_paths


SCHEMA_VERSION = "rlrmp.feedbax_graph.v1"
SUPPORTED_GRAPH_SPEC_VERSIONS = ("1.0.0",)
EXECUTION_BACKEND = "feedbax.contracts.graphs.serialization.spec_to_graph"


def tree_sum_n_features(tree) -> int:
    return jt.reduce(lambda x, y: x + y, jt.map(lambda x: x.shape[-1], tree))


def _identity_activation(x: Any) -> Any:
    return x


GRAPH_PLANT_INTERVENOR_NODE = PLANT_INTERVENOR_LABEL
NATIVE_POINT_MASS_COMPONENT = "PointMass"
NATIVE_FEEDBACK_CHANNELS_COMPONENT = "FeedbackChannels"
NATIVE_CHANNEL_COMPONENT = "Channel"
NATIVE_SUBGRAPH_COMPONENT = "Subgraph"
NATIVE_CONSTANT_COMPONENT = "Constant"
NATIVE_RAVEL_COMPONENT = "Ravel"
NATIVE_MUX_COMPONENT = "Mux"
NATIVE_DEMUX_COMPONENT = "Demux"
NATIVE_ELEMENTWISE_AFFINE_MODULATOR_COMPONENT = "ElementwiseAffineModulator"
NATIVE_AFFINE_FEEDBACK_CONTROLLER_COMPONENT = "AffineFeedbackController"
POINT_MASS_TARGET_POSITION_INPUT = "task:target_position->reference_mux.in_0"
RLRMP_MIGRATION_PACK_OWNER = "rlrmp"
RLRMP_COMPONENT_MIGRATION_PACK_VERSION = "1"


@dataclass(frozen=True)
class _NetworkInputSizes:
    external: int
    feedback: int

    @property
    def total(self) -> int:
        return self.external + self.feedback


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
        "RLRMPSimpleStagedNetwork",
        _build_simple_staged_network,
        category="RLRMP",
        description="RLRMP SimpleStagedNetwork controller.",
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        output_prototype_fn=_simple_staged_network_output_prototype,
        provenance="rlrmp",
    )
    registry.register_component_type(
        "RLRMPLinearController",
        _build_linear_controller,
        category="RLRMP",
        description="RLRMP linear regulator controller.",
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        output_prototype_fn=_linear_controller_output_prototype,
        provenance="rlrmp",
    )
    registry.register_component_type(
        "RLRMPLinearTrackerController",
        _build_linear_tracker_controller,
        category="RLRMP",
        description="RLRMP linear tracker controller.",
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
        output_prototype_fn=_linear_controller_output_prototype,
        provenance="rlrmp",
    )
    _install_feedbax_intervention_output_prototypes(registry)
    _install_feedbax_demux_output_prototype(registry)
    _install_parameter_aware_recurrent_builders(registry)
    register_rlrmp_graph_migration_pack(registry)
    return registry


def _install_feedbax_intervention_output_prototypes(registry: Any) -> None:
    for component_type in ("FixedField", "CurlField", "DynamicsMatrixPerturb"):
        meta = registry.get(component_type)
        if meta is not None and meta.output_prototype_fn is None:
            meta.output_prototype_fn = _force_passthrough_output_prototype


def _install_feedbax_demux_output_prototype(registry: Any) -> None:
    meta = registry.get(NATIVE_DEMUX_COMPONENT)
    if meta is not None and meta.output_prototype_fn is None:
        meta.output_prototype_fn = _demux_output_prototype


def _install_parameter_aware_recurrent_builders(registry: Any) -> None:
    if callable(getattr(registry, "register_builder", None)):
        registry.register_builder("GRU", _build_seeded_gru, provenance="rlrmp")
        registry.register_builder("Linear", _build_seeded_linear, provenance="rlrmp")
        registry.register_builder(
            "ElementwiseAffineModulator",
            _build_typed_elementwise_affine_modulator,
            provenance="rlrmp",
        )


def _build_seeded_gru(params: dict[str, Any]) -> RuntimeGRU:
    return RuntimeGRU(
        input_size=int(params.get("input_size", 1)),
        hidden_size=int(params.get("hidden_size", 1)),
        dtype=params.get("dtype", None),
        key=_key_from_params(params),
    )


def _build_seeded_linear(params: dict[str, Any]) -> RuntimeLinear:
    activation_name = str(params.get("activation", "identity"))
    component = RuntimeLinear(
        input_size=int(params.get("input_size", 1)),
        output_size=int(params.get("output_size", 1)),
        use_bias=bool(params.get("use_bias", True)),
        activation=activation_name,
        dtype=params.get("dtype", None),
        key=_key_from_params(params),
    )
    if activation_name == "identity":
        object.__setattr__(component, "activation", _identity_activation)
    if params.get("zero_bias", False) and component.layer.bias is not None:
        component = eqx.tree_at(
            lambda node: node.layer.bias,
            component,
            jnp.zeros_like(component.layer.bias),
        )
    return component


def _build_typed_elementwise_affine_modulator(
    params: dict[str, Any],
) -> RuntimeElementwiseAffineModulator:
    signal_shape = params.get("signal_shape")
    if not isinstance(signal_shape, (list, tuple)):
        raise ValueError("ElementwiseAffineModulator requires array parameter 'signal_shape'")
    dtype = params.get("dtype", None)

    def typed(value: Any) -> Any:
        if dtype is None:
            return value
        return jnp.asarray(value, dtype=jnp.dtype(dtype))

    return RuntimeElementwiseAffineModulator(
        signal_shape=signal_shape,
        baseline=typed(params.get("baseline", 1.0)),
        gain_init=typed(params.get("gain_init", 0.0)),
        bias_init=typed(params.get("bias_init", 0.0)),
    )


def register_rlrmp_graph_migration_pack(component_registry: Any | None = None) -> Any:
    """Register RLRMP-owned historical component migrations with Feedbax."""

    registry = component_registry or get_component_registry()
    pack = ComponentMigrationPack(
        owner=RLRMP_MIGRATION_PACK_OWNER,
        package="rlrmp",
        version=RLRMP_COMPONENT_MIGRATION_PACK_VERSION,
        description="RLRMP historical GraphSpec component IDs.",
        migrations=(
            ComponentMigration(
                source_type="RLRMPPointMass",
                target_type=NATIVE_POINT_MASS_COMPONENT,
                owner=RLRMP_MIGRATION_PACK_OWNER,
                migration_id="rlrmp.component.RLRMPPointMass-to-PointMass.v1",
                description="RLRMP historical point-mass alias now materializes via Feedbax.",
            ),
            ComponentMigration(
                source_type="RLRMPFeedbackChannels",
                target_type=NATIVE_FEEDBACK_CHANNELS_COMPONENT,
                owner=RLRMP_MIGRATION_PACK_OWNER,
                migration_id="rlrmp.component.RLRMPFeedbackChannels-to-FeedbackChannels.v1",
                migrate_params=_migrate_legacy_feedback_channels_params,
                description="RLRMP historical feedback-channel params to Feedbax selector params.",
            ),
            ComponentMigration(
                source_type="RLRMPMotorChannel",
                target_type=NATIVE_CHANNEL_COMPONENT,
                owner=RLRMP_MIGRATION_PACK_OWNER,
                migration_id="rlrmp.component.RLRMPMotorChannel-to-Channel.v1",
                description="RLRMP historical motor channel alias now materializes via Feedbax.",
            ),
            ComponentMigration(
                source_type="RLRMPPlantProcessForceNoise",
                target_type=NATIVE_CHANNEL_COMPONENT,
                owner=RLRMP_MIGRATION_PACK_OWNER,
                migration_id="rlrmp.component.RLRMPPlantProcessForceNoise-to-Channel.v1",
                migrate_params=_migrate_legacy_plant_process_force_noise_params,
                description="RLRMP historical plant-process force noise channel.",
            ),
            ComponentMigration(
                source_type="rlrmp.RLRMPFeedbackChannels",
                target_type=NATIVE_FEEDBACK_CHANNELS_COMPONENT,
                owner=RLRMP_MIGRATION_PACK_OWNER,
                migration_id="rlrmp.component.qualified-RLRMPFeedbackChannels-to-FeedbackChannels.v1",
                migrate_params=_migrate_legacy_feedback_channels_params,
                description="Owner-qualified RLRMP feedback-channel legacy alias.",
            ),
        ),
    )
    _register_migration_pack_idempotent(registry, pack)
    return registry


def _register_migration_pack_idempotent(registry: Any, pack: ComponentMigrationPack) -> None:
    try:
        registry.register_migration_pack(pack)
    except ValueError as exc:
        if "Component migration already registered" not in str(exc):
            raise


def _migrate_legacy_feedback_channels_params(params: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(params)
    legacy_where = migrated.pop("where", None)
    if legacy_where is not None and "paths" not in migrated:
        migrated["selector"] = "paths"
        migrated["paths"] = list(legacy_where)
    migrated.setdefault("selector", "point_mass_pos_vel")
    migrated.setdefault("noise_model", "additive_gaussian")
    migrated.setdefault("noise_timing", "pre_controller")
    migrated.setdefault("input_shape", [[2], [2]])
    return migrated


def _migrate_legacy_plant_process_force_noise_params(params: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(params)
    noise_std = float(migrated.get("noise_std", 0.0) or 0.0)
    return {
        "delay": 0,
        "noise_model": "additive_gaussian",
        "noise_std": noise_std,
        "add_noise": noise_std != 0.0,
        "noise_role": migrated.get("noise_role", "plant_process_load"),
        "noise_timing": migrated.get("noise_timing", "post_force_filter_pre_mechanics"),
        "input_shape": migrated.get("input_shape", [2]),
    }


def _normalize_legacy_rlrmp_graph_topology(graph_spec: GraphSpec) -> GraphSpec:
    """Normalize legacy topology details that component migrations cannot rewrite."""

    nodes = dict(graph_spec.nodes)
    legacy_plant_process_nodes: set[str] = set()
    for node_id, node_spec in graph_spec.nodes.items():
        params = dict(node_spec.params)
        if node_spec.type == "RLRMPPlantProcessForceNoise":
            legacy_plant_process_nodes.add(node_id)
            nodes[node_id] = node_spec.model_copy(
                update={
                    "input_ports": ["input"],
                    "output_ports": ["output"],
                }
            )
        elif node_spec.type == "DynamicsMatrixPerturb" and "delta_A_shape" in params:
            shape = params.pop("delta_A_shape")
            params["delta_A"] = [[0.0 for _ in range(int(shape[1]))] for _ in range(int(shape[0]))]
            nodes[node_id] = node_spec.model_copy(update={"params": params})
        elif node_spec.type == "CurlField" and "amplitude" not in params:
            params["amplitude"] = 1.0
            nodes[node_id] = node_spec.model_copy(update={"params": params})
        else:
            nodes[node_id] = node_spec

    if not legacy_plant_process_nodes:
        return graph_spec.model_copy(update={"nodes": nodes})

    def _rename_source_port(node: str, port: str) -> str:
        if node in legacy_plant_process_nodes and port == "force":
            return "output"
        return port

    def _rename_target_port(node: str, port: str) -> str:
        if node in legacy_plant_process_nodes and port == "force":
            return "input"
        return port

    wires = [
        WireSpec(
            source_node=wire.source_node,
            source_port=_rename_source_port(wire.source_node, wire.source_port),
            target_node=wire.target_node,
            target_port=_rename_target_port(wire.target_node, wire.target_port),
            temporality=wire.temporality,
            recurrent_initializer=wire.recurrent_initializer,
        )
        for wire in graph_spec.wires
    ]
    return graph_spec.model_copy(update={"nodes": nodes, "wires": wires})


def resolve_registered_graph_component_migrations(
    graph_spec: GraphSpec, registry: Any
) -> GraphSpec:
    """Apply registered Feedbax component migrations before prototype inference."""

    registry_names = set(registry.names()) if callable(getattr(registry, "names", None)) else set()
    nodes: dict[str, ComponentSpec] = {}
    changed = False
    for node_id, node_spec in graph_spec.nodes.items():
        should_try = (
            node_spec.type not in registry_names or node_spec.param_schema_version is not None
        )
        if not should_try:
            nodes[node_id] = node_spec
            continue
        resolution = registry.resolve_component_spec(
            node_spec.type,
            node_spec.params,
            param_schema_version=node_spec.param_schema_version,
        )
        nodes[node_id] = node_spec.model_copy(
            update={
                "type": resolution.type_id,
                "params": resolution.params,
                "param_schema_version": resolution.param_schema_version,
            }
        )
        changed = True
    if not changed:
        return graph_spec
    return graph_spec.model_copy(update={"nodes": nodes})


def materialize_rlrmp_graph_spec(
    graph_spec: GraphSpec,
    component_registry: Any | None = None,
    *,
    install_runtime_hooks: bool = True,
) -> Graph:
    """Materialize an RLRMP GraphSpec through Feedbax and install runtime hooks."""

    registry = register_rlrmp_graph_components(component_registry)
    graph_spec = _normalize_legacy_rlrmp_graph_topology(graph_spec)
    graph_spec = resolve_registered_graph_component_migrations(graph_spec, registry)
    graph = spec_to_graph(graph_spec, registry)
    graph = normalize_native_recurrent_runtime_initializers(graph)
    if install_runtime_hooks:
        graph = install_simple_feedback_runtime_hooks(graph)
    return graph


def normalize_native_recurrent_runtime_initializers(graph: Graph) -> Graph:
    """Make executable native recurrent graph initializers match component dtypes."""

    changed = False
    nodes = dict(graph.nodes)
    for name, node in list(nodes.items()):
        if isinstance(node, Graph):
            normalized = normalize_native_recurrent_runtime_initializers(node)
            if normalized is not node:
                nodes[name] = normalized
                changed = True
    graph = eqx.tree_at(lambda g: g.nodes, graph, nodes) if changed else graph

    cell = graph.nodes.get("cell")
    modulator = graph.nodes.get("sisu_modulator")
    if cell is None or modulator is None:
        return graph
    hidden_dtype = getattr(getattr(cell, "cell", None), "weight_ih", None)
    if hidden_dtype is None:
        return graph
    dtype = jnp.asarray(hidden_dtype).dtype
    wires = []
    wire_changed = False
    hidden_size = int(getattr(cell, "hidden_size", jnp.asarray(hidden_dtype).shape[0] // 3))
    for wire in graph.wires:
        if (
            wire.temporality == "recurrent"
            and wire.source_node == "sisu_modulator"
            and wire.source_port == "output"
            and wire.target_node == "cell"
            and wire.target_port == "hidden"
        ):
            initializer = {
                "kind": "constant",
                "scope": "trial",
                "source": "state_initializer",
                "state_slot": "hidden",
                "value": jnp.zeros((hidden_size,), dtype=dtype),
            }
            wire = replace(wire, recurrent_initializer=initializer)
            wire_changed = True
        wires.append(wire)
    if wire_changed:
        graph = eqx.tree_at(lambda g: g.wires, graph, tuple(wires))
    return graph


def _simple_staged_network_output_prototype(
    params: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    del inputs
    hidden = jnp.zeros(int(params.get("hidden_size", 1)))
    output = jnp.zeros(int(params.get("out_size", 2)))
    return {"output": output, "hidden": hidden}


def _linear_controller_output_prototype(
    params: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    del inputs
    controls = int(params.get("n_controls", 2))
    return {"output": jnp.zeros(controls), "hidden": jnp.zeros(1)}


def _force_passthrough_output_prototype(
    params: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    del params
    return {"force": inputs.get("force", jnp.zeros(2))}


def _demux_output_prototype(
    params: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    sizes = tuple(int(size) for size in params.get("sizes", ()))
    if not sizes:
        source = jnp.asarray(inputs.get("input", jnp.zeros(1)))
        sizes = (int(source.shape[-1]),)
    return {
        f"out_{index}": jnp.zeros((size,), dtype=jnp.asarray(inputs.get("input", 0.0)).dtype)
        for index, size in enumerate(sizes)
    }


def install_simple_feedback_runtime_hooks(graph: Graph) -> Graph:
    """Install SimpleFeedback-compatible state view and consistency hooks."""

    mechanics = graph.nodes.get("mechanics")
    feedback = graph.nodes.get("feedback")
    if not isinstance(mechanics, Mechanics) or not isinstance(feedback, FeedbackChannels):
        return graph

    def _state_view(node_states):
        force_filter_state = node_states.get("force_filter", FilterState(output=None, solver=None))
        net_state = node_states["net"]
        if not hasattr(net_state, "output"):
            net_state = recurrent_graph_state_to_network_state(
                net_state,
                command=jnp.asarray(node_states["efferent"].output),
            )
        return SimpleFeedbackState(
            mechanics=node_states["mechanics"],
            net=net_state,
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


def _build_simple_staged_network(params: dict[str, Any]) -> SimpleStagedNetwork:
    hidden_type_name = str(params.get("hidden_type", "GRUCell"))
    if hidden_type_name == "VanillaRNNCell":
        from rlrmp.model import VanillaRNNCell

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
        dtype=jnp.dtype(params.get("trainable_dtype", jnp.float32)),
        key=key,
    )


def _build_linear_controller(params: dict[str, Any]):
    from rlrmp.controllers.linear import LinearController

    return LinearController(
        n_steps=int(params["n_steps"]),
        n_controls=int(params.get("n_controls", 2)),
        n_states=int(params.get("n_states", 4)),
        K_init_scale=float(params.get("K_init_scale", 0.0) or 0.0),
        key=_key_from_params(params),
    )


def _build_linear_tracker_controller(params: dict[str, Any]):
    from rlrmp.controllers.linear import LinearTrackerController

    return LinearTrackerController(
        n_steps=int(params["n_steps"]),
        n_controls=int(params.get("n_controls", 2)),
        n_states=int(params.get("n_states", 4)),
        K_init_scale=float(params.get("K_init_scale", 0.0) or 0.0),
        u_ff_init_scale=float(params.get("u_ff_init_scale", 0.0) or 0.0),
        key=_key_from_params(params),
    )


def _key_from_params(params: dict[str, Any]):
    key_data = params.get("key")
    if key_data is not None:
        return jnp.asarray(key_data, dtype=jnp.uint32)
    return jr.PRNGKey(int(params.get("seed", 0)))


def _population_structure_from_params(params: Any) -> PopulationStructure | None:
    if not params:
        return None
    if isinstance(params, PopulationStructure):
        return params
    if isinstance(params, dict) and params.get("assignment") == "explicit":
        return population_structure_from_spec(int(params["hidden_size"]), params)
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
    if controller_kind in {"linear", "linear_tracker"}:
        task_spec["task_binding_spec"] = _point_mass_reference_task_binding_spec().model_dump(
            mode="json"
        )
    loss_spec = _loss_spec(hps)
    training_spec = _training_spec(hps, controller_kind=controller_kind)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "execution_backend": EXECUTION_BACKEND,
        "component_policy": {
            "feedbax_component_types": [
                NATIVE_SUBGRAPH_COMPONENT,
                NATIVE_CONSTANT_COMPONENT,
                NATIVE_RAVEL_COMPONENT,
                NATIVE_MUX_COMPONENT,
                NATIVE_DEMUX_COMPONENT,
                NATIVE_ELEMENTWISE_AFFINE_MODULATOR_COMPONENT,
                NATIVE_AFFINE_FEEDBACK_CONTROLLER_COMPONENT,
                "GRU",
                "Linear",
            ],
            "rlrmp_component_types": [
                "DynamicsMatrixPerturb",
            ],
            "legacy_component_aliases": [
                "RLRMPLinearController",
                "RLRMPLinearTrackerController",
                "RLRMPPointMass",
                "RLRMPFeedbackChannels",
                "RLRMPMotorChannel",
                "RLRMPPlantProcessForceNoise",
            ],
            "note": (
                "Generic point-mass mechanics, feedback, stochastic channel, and "
                "recurrent controller nodes use Feedbax-native component types. GRU "
                "controllers emit explicit Subgraph wiring around Feedbax Mux, GRU, "
                "Linear, ElementwiseAffineModulator, and population-constraint "
                "primitives. Linear and linear-tracker controllers emit "
                "Feedbax AffineFeedbackController nodes fed by native feedback-vector "
                "and task-data-bound reference-vector plumbing. The runtime state view "
                "adapts native affine/efferent state for legacy RLRMP losses without "
                "adding graph nodes. SISU is an RLRMP task input, not a Feedbax "
                "network port."
            ),
        },
        "legacy_loader": {
            "setup_function": "rlrmp.train.task_model.setup_task_model_pair",
            "checkpoint_format": "jax_cookbook.save/load_with_hyperparameters",
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
    _legacy_recurrent_controller: bool = False,
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
    if task is None and controller_kind == "gru":
        from rlrmp.train.task_model import build_task_base

        task = build_task_base(hps)
    key_param = None if key is None else [int(value) for value in jnp.asarray(key).tolist()]

    mechanics_params = {
        "dt": float(hps.dt),
        "mass": float(hps.model.effector_mass),
        "damping": float(hps.model.damping),
    }
    net_spec, net_subgraph = _controller_component_spec(
        hps,
        controller_kind,
        task=task,
        n_extra_inputs=n_extra_inputs,
        population_structure=population_structure,
        hidden_type=hidden_type,
        sisu_gating=sisu_gating,
        key=key_param,
        legacy_recurrent_controller=_legacy_recurrent_controller,
    )
    subgraphs = {"net": net_subgraph} if net_subgraph is not None else None
    nodes: dict[str, ComponentSpec] = {
        "feedback": ComponentSpec(
            type=NATIVE_FEEDBACK_CHANNELS_COMPONENT,
            params={
                "delay": int(hps.model.feedback_delay_steps),
                "selector": "point_mass_pos_vel",
                "noise_model": "additive_gaussian",
                "noise_std": noise_config.sensory_noise_std,
                "noise_role": "sensory_feedback",
                "noise_timing": "pre_controller",
                "add_noise": noise_config.sensory_noise_std != 0.0,
                "input_shape": [[2], [2]],
            },
            input_ports=["mechanics"],
            output_ports=["feedback"],
        ),
        "net": net_spec,
        "efferent": ComponentSpec(
            type=NATIVE_CHANNEL_COMPONENT,
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
            type=NATIVE_POINT_MASS_COMPONENT,
            params=mechanics_params,
            input_ports=["force"],
            output_ports=["effector", "state"],
        ),
    }
    if controller_kind in {"linear", "linear_tracker"}:
        nodes["feedback_vector"] = ComponentSpec(
            type=NATIVE_RAVEL_COMPONENT,
            params={},
            input_ports=["input"],
            output_ports=["output"],
        )
        nodes["zero_velocity"] = ComponentSpec(
            type=NATIVE_CONSTANT_COMPONENT,
            params={"value": [0.0, 0.0]},
            input_ports=[],
            output_ports=["output"],
        )
        nodes["reference_mux"] = ComponentSpec(
            type=NATIVE_MUX_COMPONENT,
            params={"n_inputs": 2},
            input_ports=["in_0", "in_1"],
            output_ports=["output"],
        )
        controller_output_source = ("net", "command")
        wires: list[WireSpec] = [
            WireSpec(
                source_node="feedback",
                source_port="feedback",
                target_node="feedback_vector",
                target_port="input",
            ),
            WireSpec(
                source_node="feedback_vector",
                source_port="output",
                target_node="net",
                target_port="feedback",
            ),
            WireSpec(
                source_node="zero_velocity",
                source_port="output",
                target_node="reference_mux",
                target_port="in_1",
            ),
            WireSpec(
                source_node="reference_mux",
                source_port="output",
                target_node="net",
                target_port="reference",
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
        input_bindings: dict[str, tuple[str, str]] = {
            POINT_MASS_TARGET_POSITION_INPUT: ("reference_mux", "in_0")
        }
    else:
        controller_output_source = ("net", "output")
        wires = [
            WireSpec(
                source_node="feedback",
                source_port="feedback",
                target_node="net",
                target_port="feedback",
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
        input_bindings = {"input": ("net", "input")}
    wires.append(
        WireSpec(
            source_node=controller_output_source[0],
            source_port=controller_output_source[1],
            target_node="efferent",
            target_port="input",
        )
    )

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
            type=NATIVE_CHANNEL_COMPONENT,
            params={
                "delay": 0,
                "noise_std": noise_config.plant_process_force_noise_std,
                "noise_model": "additive_gaussian",
                "add_noise": True,
                "noise_role": "plant_process_load",
                "noise_timing": "post_force_filter_pre_mechanics",
                "state_diffusion": False,
                "input_shape": [2],
            },
            input_ports=["input"],
            output_ports=["output"],
        )
        wires.extend(
            [
                WireSpec(
                    source_node=force_wire.source_node,
                    source_port=force_wire.source_port,
                    target_node=PLANT_PROCESS_FORCE_NOISE_LABEL,
                    target_port="input",
                ),
                WireSpec(
                    source_node=PLANT_PROCESS_FORCE_NOISE_LABEL,
                    source_port="output",
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
            controller_kind=controller_kind,
        ),
        subgraphs=subgraphs,
        metadata=GraphMetadata(
            name="RLRMP point-mass sensorimotor loop",
            description=("Executable GraphSpec contract for RLRMP minimax training."),
            created_at="1970-01-01T00:00:00",
            updated_at="1970-01-01T00:00:00",
            version="1.0.0",
            tags=["rlrmp", "feedbax", "graphspec", "minimax"],
        ),
    )


def _point_mass_reference_task_binding_spec() -> StudioTaskBindingSpec:
    return StudioTaskBindingSpec.model_validate(
        {
            "schema_version": "feedbax.spec.studio.task_bindings.v2",
            "exposed_data": [
                {
                    "id": "target_position",
                    "label": "Target position",
                    "kind": "signal",
                    "role": "model_input",
                    "path": "inputs.effector_target.pos",
                    "bindable": True,
                    "expected_shape": ["time", 2],
                    "dtype": "vector",
                    "metadata": {"temporal_support": "trajectory"},
                }
            ],
            "bindings": [
                {
                    "id": POINT_MASS_TARGET_POSITION_INPUT,
                    "source_data_id": "target_position",
                    "target_node_id": "reference_mux",
                    "target_port": "in_0",
                    "role": "model_input",
                    "metadata": {},
                }
            ],
            "metadata": {},
        }
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


def create_legacy_point_mass_graph_ensemble(
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
    """Build the historical staged-network graph shape for checkpoint materialization."""

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
                _legacy_recurrent_controller=True,
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

    from feedbax.contracts.graphs.serialization import graph_to_spec

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
        elif isinstance(component, DynamicsMatrixPerturb):
            nodes[name] = ComponentSpec(
                type="DynamicsMatrixPerturb",
                params={
                    "active": bool(component._initial_state.active),
                    "scale": float(component._initial_state.scale),
                    "delta_A": jnp.asarray(component._initial_state.delta_A).tolist(),
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
    from rlrmp.model.cs_lss_gru import (
        CS_LSS_INITIAL_HIDDEN_NET_COMPONENT,
        InitialHiddenStagedNetwork,
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
    return None


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


def _runtime_linear_controller_params(component: Any) -> dict[str, Any] | None:
    from rlrmp.controllers.linear import LinearController, LinearTrackerController

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
    legacy_recurrent_controller: bool = False,
) -> tuple[ComponentSpec, GraphSpec | None]:
    if controller_kind == "linear":
        return (
            ComponentSpec(
                type=NATIVE_AFFINE_FEEDBACK_CONTROLLER_COMPONENT,
                params=_affine_linear_controller_params(hps, include_feedforward=False),
                input_ports=["feedback", "reference"],
                output_ports=["command"],
            ),
            None,
        )
    if controller_kind == "linear_tracker":
        return (
            ComponentSpec(
                type=NATIVE_AFFINE_FEEDBACK_CONTROLLER_COMPONENT,
                params=_affine_linear_controller_params(hps, include_feedforward=True),
                input_ports=["feedback", "reference"],
                output_ports=["command"],
            ),
            None,
        )
    input_size = None
    external_input_size = None
    feedback_size = None
    if task is not None:
        input_sizes = _point_mass_network_input_sizes(
            hps,
            task=task,
            n_extra_inputs=n_extra_inputs,
        )
        input_size = input_sizes.total
        external_input_size = input_sizes.external
        feedback_size = input_sizes.feedback
    hidden_type_name = _hidden_type_name(hidden_type)
    population_params = _population_structure_params(
        hps,
        population_structure,
    )
    if legacy_recurrent_controller:
        return (
            ComponentSpec(
                type="RLRMPSimpleStagedNetwork",
                params={
                    "controller_kind": controller_kind,
                    "input_size": input_size,
                    "input_size_source": "task-derived" if input_size is not None else "unresolved",
                    "hidden_size": int(hps.model.hidden_size),
                    "out_size": 2,
                    "encoding_size": None,
                    "hidden_type": hidden_type_name,
                    "sisu_gating": str(sisu_gating),
                    "n_extra_inputs": int(n_extra_inputs),
                    "population_structure": population_params,
                    "key": key,
                },
                input_ports=["input", "feedback"],
                output_ports=["output", "hidden"],
            ),
            None,
        )
    if _requires_native_recurrent_controller_gap(
        controller_kind=controller_kind,
        input_size=input_size,
        hidden_type_name=hidden_type_name,
    ):
        raise ValueError(
            "Feedbax-native recurrent controller emission requires a resolved GRU input "
            f"size and GRU hidden type; got controller_kind={controller_kind!r}, "
            f"input_size={input_size!r}, hidden_type={hidden_type_name!r}."
        )
    return (
        ComponentSpec(
            type=NATIVE_SUBGRAPH_COMPONENT,
            params={
                "controller_kind": controller_kind,
                "input_size": input_size,
                "external_input_size": external_input_size,
                "feedback_size": feedback_size,
                "hidden_size": int(hps.model.hidden_size),
                "out_size": 2,
                "hidden_type": hidden_type_name,
                "sisu_gating": str(sisu_gating),
                "population_structure": population_params,
            },
            input_ports=["input", "feedback"],
            output_ports=["output", "hidden"],
        ),
        native_recurrent_controller_subgraph(
            input_size=int(input_size),
            external_input_size=int(external_input_size),
            hidden_size=int(hps.model.hidden_size),
            out_size=2,
            hidden_type_name=hidden_type_name,
            sisu_gating=str(sisu_gating),
            population_params=population_params,
            key=key,
        ),
    )


def native_recurrent_controller_subgraph(
    *,
    input_size: int,
    external_input_size: int | None = None,
    hidden_size: int,
    out_size: int,
    hidden_type_name: str = "GRUCell",
    sisu_gating: str = "additive",
    population_params: dict[str, int] | None = None,
    h0_initializer_source: str | None = None,
    key: list[int] | None = None,
    dtype: str | None = None,
) -> GraphSpec:
    """Build ordinary Feedbax graph wiring for an RLRMP recurrent controller."""

    if sisu_gating not in {"additive", "multiplicative"}:
        raise ValueError(f"Unsupported sisu_gating {sisu_gating!r}.")
    if hidden_type_name not in {"GRU", "GRUCell", "gru"}:
        raise ValueError(f"Unsupported Feedbax-native hidden_type {hidden_type_name!r}.")
    population_structure = _population_structure_from_params(population_params or {})
    network_input_size = int(input_size) - (1 if sisu_gating == "multiplicative" else 0)
    if network_input_size <= 0:
        raise ValueError(f"Native recurrent controller input_size must be positive; got {input_size}.")
    graph = recurrent_controller_template_graph(
        input_size=network_input_size,
        hidden_size=int(hidden_size),
        out_size=int(out_size),
        cell_type="GRU",
        out_nonlinearity="identity",
        population_structure=population_structure,
        name="RLRMP recurrent controller",
        description="Explicit graph wiring around ordinary Feedbax recurrent primitives.",
    )
    graph = graph.model_copy(
        update={
            "nodes": _with_recurrent_leaf_params(
                graph.nodes,
                key=key,
                dtype=dtype,
            ),
            "metadata": GraphMetadata(
                name="RLRMP recurrent controller",
                description=(
                    "Explicit graph wiring around ordinary Feedbax Mux, GRU, Linear, "
                    "ElementwiseAffineModulator, and parameter-constraint primitives."
                ),
                created_at="1970-01-01T00:00:00",
                updated_at="1970-01-01T00:00:00",
                version="1.0.0",
                tags=["rlrmp", "feedbax", "gru"],
            )
        }
    )
    if h0_initializer_source is not None:
        graph = _with_graph_input_hidden_initializer(
            graph,
            source=h0_initializer_source,
        )
    if sisu_gating == "multiplicative":
        graph = _with_multiplicative_sisu_modulation(
            graph,
            input_size=int(input_size),
            external_input_size=external_input_size,
            hidden_size=int(hidden_size),
            dtype=dtype,
        )
    return graph


def _with_recurrent_leaf_params(
    nodes: dict[str, ComponentSpec],
    *,
    key: list[int] | None,
    dtype: str | None,
) -> dict[str, ComponentSpec]:
    key1, _key2, key3 = jr.split(_key_from_params({"key": key}), 3)
    updates = {
        "cell": {"key": [int(value) for value in jnp.asarray(key1).tolist()]},
        "readout": {
            "key": [int(value) for value in jnp.asarray(key3).tolist()],
            "zero_bias": True,
        },
    }
    if dtype is not None:
        updates["cell"]["dtype"] = str(dtype)
        updates["readout"]["dtype"] = str(dtype)
    out = dict(nodes)
    for name, params_update in updates.items():
        node = out[name]
        out[name] = node.model_copy(
            update={"params": {**dict(node.params), **params_update}}
        )
    return out


def _requires_native_recurrent_controller_gap(
    *,
    controller_kind: str,
    input_size: int | None,
    hidden_type_name: str,
) -> bool:
    if controller_kind != "gru":
        return True
    if input_size is None:
        return True
    if hidden_type_name not in {"GRU", "GRUCell", "gru"}:
        return True
    return False


def _with_graph_input_hidden_initializer(graph: GraphSpec, *, source: str) -> GraphSpec:
    wires = []
    for wire in graph.wires:
        if (
            wire.source_node == "cell"
            and wire.source_port == "hidden"
            and wire.target_node == "cell"
            and wire.target_port == "hidden"
            and wire.temporality == "recurrent"
        ):
            wire = wire.model_copy(
                update={
                    "recurrent_initializer": recurrent_graph_input_initializer(
                        source,
                        state_slot="hidden",
                    )
                }
            )
        wires.append(wire)
    input_ports = list(graph.input_ports)
    if source not in input_ports:
        input_ports.append(source)
    input_bindings = dict(graph.input_bindings)
    input_bindings[source] = ("cell", "hidden")
    return graph.model_copy(
        update={
            "wires": wires,
            "input_ports": input_ports,
            "input_bindings": input_bindings,
        }
    )


def _with_multiplicative_sisu_modulation(
    graph: GraphSpec,
    *,
    input_size: int,
    external_input_size: int | None,
    hidden_size: int,
    dtype: str | None,
) -> GraphSpec:
    if external_input_size is None or external_input_size <= 0:
        raise ValueError(
            "Multiplicative SISU native graph emission requires a positive external "
            "input size so the SISU channel can be routed as a graph signal."
        )
    nodes = dict(graph.nodes)
    wires = []
    input_bindings = dict(graph.input_bindings)
    output_bindings = dict(graph.output_bindings)
    non_sisu_external_size = int(external_input_size) - 1
    if non_sisu_external_size > 0:
        nodes["sisu_split"] = ComponentSpec(
            type=NATIVE_DEMUX_COMPONENT,
            params={"sizes": [non_sisu_external_size, 1]},
            input_ports=["input"],
            output_ports=["out_0", "out_1"],
        )
        input_bindings["input"] = ("sisu_split", "input")
        sisu_source = ("sisu_split", "out_1")
        external_source = ("sisu_split", "out_0")
    else:
        sisu_source = ("sisu_modulator", "modulator")
        external_source = None
        input_bindings["input"] = sisu_source
        nodes["input_mux"] = nodes["input_mux"].model_copy(
            update={
                "params": {"n_inputs": 1},
                "input_ports": ["in_0"],
            }
        )
        input_bindings["feedback"] = ("input_mux", "in_0")
    nodes["sisu_modulator"] = ComponentSpec(
        type=NATIVE_ELEMENTWISE_AFFINE_MODULATOR_COMPONENT,
        params={
            "signal_shape": [int(hidden_size)],
            "baseline": 1.0,
            "gain_init": 0.0,
            "bias_init": 0.0,
            "dtype": dtype,
        },
        input_ports=["signal", "modulator", "scale", "bias"],
        output_ports=["output"],
    )
    for wire in graph.wires:
        if external_source is not None and (
            wire.source_node == "input_mux"
            and wire.source_port == "output"
            and wire.target_node == "cell"
            and wire.target_port == "input"
        ):
            wires.append(wire)
            continue
        if (
            wire.source_node == "cell"
            and wire.source_port == "hidden"
            and wire.target_node == "readout"
            and wire.target_port == "input"
        ):
            wires.append(
                WireSpec(
                    source_node="sisu_modulator",
                    source_port="output",
                    target_node="readout",
                    target_port="input",
                )
            )
            continue
        if (
            wire.source_node == "cell"
            and wire.source_port == "hidden"
            and wire.target_node == "cell"
            and wire.target_port == "hidden"
            and wire.temporality == "recurrent"
        ):
            wires.append(
                wire.model_copy(
                    update={
                        "source_node": "sisu_modulator",
                        "source_port": "output",
                    }
                )
            )
            continue
        wires.append(wire)
    if external_source is not None:
        wires.append(
            WireSpec(
                source_node=external_source[0],
                source_port=external_source[1],
                target_node="input_mux",
                target_port="in_0",
            )
        )
    if sisu_source != ("sisu_modulator", "modulator"):
        wires.append(
            WireSpec(
                source_node=sisu_source[0],
                source_port=sisu_source[1],
                target_node="sisu_modulator",
                target_port="modulator",
            )
        )
    wires.append(
        WireSpec(
            source_node="cell",
            source_port="hidden",
            target_node="sisu_modulator",
            target_port="signal",
        )
    )
    output_bindings["hidden"] = ("sisu_modulator", "output")
    if non_sisu_external_size > 0:
        expected_input_size = non_sisu_external_size + 1 + int(input_size - external_input_size)
        if expected_input_size != int(input_size):
            raise ValueError("Inconsistent SISU input-size accounting.")
    return graph.model_copy(
        update={
            "nodes": nodes,
            "wires": wires,
            "input_bindings": input_bindings,
            "output_bindings": output_bindings,
        }
    )


def recurrent_graph_state_to_network_state(net_state: Any, *, command: Any | None = None) -> NetworkState:
    """Return the legacy NetworkState view for native recurrent-controller graphs."""

    if hasattr(net_state, "output"):
        return net_state
    if isinstance(net_state, GraphState):
        cell_state = net_state.nodes.get("cell")
        hidden = getattr(cell_state, "hidden", None)
        if hidden is None:
            hidden = getattr(cell_state, "output", None)
        if hidden is not None:
            output = command if command is not None else hidden
            return NetworkState(
                input=jnp.zeros((0,), dtype=jnp.asarray(hidden).dtype),
                hidden=jnp.asarray(hidden),
                output=jnp.asarray(output),
                encoding=None,
            )
    command_array = jnp.asarray(command if command is not None else net_state)
    return NetworkState(
        input=jnp.zeros((0,), dtype=command_array.dtype),
        hidden=jnp.atleast_1d(command_array),
        output=command_array,
        encoding=None,
    )


def _point_mass_network_input_size(
    hps: Any,
    *,
    task: Any,
    n_extra_inputs: int,
) -> int:
    return _point_mass_network_input_sizes(
        hps,
        task=task,
        n_extra_inputs=n_extra_inputs,
    ).total


def _point_mass_network_input_sizes(
    hps: Any,
    *,
    task: Any,
    n_extra_inputs: int,
) -> _NetworkInputSizes:
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
    plant_state = mechanics.plant.init(key=jr.PRNGKey(0))
    example_feedback = feedback_spec["where"](
        MechanicsState(
            plant=plant_state,
            effector=mechanics.plant.skeleton.effector(plant_state.skeleton),
            solver=None,
        )
    )
    example_trial_spec = task.get_train_trial_with_intervenor_params(key=jr.PRNGKey(0))
    return _NetworkInputSizes(
        external=tree_sum_n_features(example_trial_spec.inputs) + int(n_extra_inputs),
        feedback=tree_sum_n_features(example_feedback),
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


def _affine_linear_controller_params(
    hps: Any,
    *,
    include_feedforward: bool,
) -> dict[str, Any]:
    n_steps = int(hps.task.n_steps) - 1
    n_controls = 2
    n_states = 4
    params: dict[str, Any] = {
        "gain": jnp.zeros((n_steps, n_controls, n_states)).tolist(),
        "schedule_policy": "hold",
        "target_source": "input.task.effector_target.pos",
        "feedback_order": ["pos_x", "pos_y", "vel_x", "vel_y"],
        "reference_order": ["target_pos_x", "target_pos_y", "zero_vel_x", "zero_vel_y"],
    }
    if include_feedforward:
        params["feedforward"] = jnp.zeros((n_steps, n_controls)).tolist()
    return params


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
    elif intervention_type == "CurlField":
        params.update({"amplitude": 1.0})
    elif intervention_type == "DynamicsMatrixPerturb":
        params.update(
            {
                "delta_A": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
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
    controller_kind: str = "gru",
) -> list[RetainedObservableSpec]:
    observables = [
        _port_observable("mechanics.effector", "mechanics", "effector"),
        _port_observable("mechanics.state", "mechanics", "state"),
        _port_observable("efferent.output", "efferent", "output"),
        _port_observable(
            f"{GRAPH_PLANT_INTERVENOR_NODE}.force",
            GRAPH_PLANT_INTERVENOR_NODE,
            "force",
        ),
    ]
    if controller_kind in {"linear", "linear_tracker"}:
        observables.append(_port_observable("net.output", "net", "command"))
    else:
        observables.extend(
            [
                _port_observable("net.output", "net", "output"),
                _port_observable("net.hidden", "net", "hidden"),
            ]
        )
    if include_plant_process_force_noise:
        observables.append(
            _port_observable(
                f"{PLANT_PROCESS_FORCE_NOISE_LABEL}.force",
                PLANT_PROCESS_FORCE_NOISE_LABEL,
                "output",
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
        ["nodes.net.gain"]
        if controller_kind == "linear"
        else ["nodes.net.gain", "nodes.net.feedforward"]
    )
    if controller_kind not in {"linear", "linear_tracker"}:
        trainable = staged_network_trainable_paths(
            sisu_gating=str(getattr(hps, "sisu_gating", "additive"))
        )
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
) -> dict[str, Any]:
    pop = population_structure or getattr(hps.model, "population_structure", None)
    if pop is None:
        return {}
    if hasattr(pop, "to_spec"):
        spec = dict(pop.to_spec())
        spec["hidden_size"] = int(hps.model.hidden_size)
        return spec
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
    return compact_json_dumps(payload)
