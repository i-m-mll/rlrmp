"""RLRMP GraphSpec migration tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.control import AffineFeedbackController
from feedbax.models.networks import PopulationStructure
from feedbax.runtime.graph import Graph
from feedbax.runtime.graph import init_state_from_component
from feedbax.intervene import CurlField, DynamicsMatrixPerturb, FixedField
from feedbax.contracts.manifest import SCHEMA_VERSION as FEEDBAX_MANIFEST_SCHEMA_VERSION

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.model.feedbax_graph import (
    EXECUTION_BACKEND,
    GRAPH_PLANT_INTERVENOR_NODE,
    POINT_MASS_TARGET_POSITION_INPUT,
    SCHEMA_VERSION,
    SUPPORTED_GRAPH_SPEC_VERSIONS,
    _build_simple_staged_network,
    build_point_mass_sensorimotor_graph_spec,
    build_rlrmp_feedbax_graph_bundle,
    build_runtime_rlrmp_feedbax_graph_bundle,
    graph_spec_from_model,
    graph_spec_payload,
    materialize_rlrmp_graph_spec,
    native_recurrent_controller_subgraph,
    register_rlrmp_graph_components,
    resolve_registered_graph_component_migrations,
    write_graph_spec_bundle,
)
from rlrmp.intervention_compat import swap_plant_intervenor_to_dynamics_matrix
from rlrmp.controllers.linear import LinearController, LinearTrackerController
from rlrmp.train.task_model import build_task_base, setup_task_model_pair
from rlrmp.model.stochastic_runtime import PLANT_PROCESS_FORCE_NOISE_LABEL
from rlrmp.train.minimax import build_hps


pytestmark = pytest.mark.feedbax_contract


def _args(**overrides):
    base = {
        "n_warmup_batches": 10,
        "n_adversary_batches": 20,
        "controller_lr": 0.01,
        "loss_update_enabled": False,
        "loss_update_ratio": 0.3,
        "hidden_type": "gru",
        "sisu_gating": "additive",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _hps(**overrides):
    hps = build_hps(_args(**overrides))
    if hps.pert.type == "gusts":
        hps = hps | {"pert": hps.pert | {"type": "constant"}}
    return hps


def _plain_network_hps(**overrides):
    return _hps(**overrides)


def _unmasked_population() -> argparse.Namespace:
    return argparse.Namespace(
        n_input_only=0,
        n_readout_only=0,
        n_recurrent_only=0,
        n_input_readout=0,
    )


@pytest.mark.parametrize(
    "hidden_type",
    ["gru", "linear", "linear_tracker"],
)
def test_available_rlrmp_graph_specs_round_trip_through_feedbax_contract(
    hidden_type: str,
) -> None:
    hps = _hps(hidden_type=hidden_type)
    task = build_task_base(hps)
    bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=task,
        n_extra_inputs=0,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
    )
    payload = graph_spec_payload(bundle.graph_spec)

    round_tripped = GraphSpec.model_validate_json(json.dumps(payload))

    assert graph_spec_payload(round_tripped) == payload
    assert isinstance(materialize_rlrmp_graph_spec(round_tripped), Graph)
    assert round_tripped.metadata is not None
    assert round_tripped.metadata.version == "1.0.0"
    assert bundle.to_run_metadata()["schema_version"] == SCHEMA_VERSION
    assert bundle.manifest["schema_version"] == SCHEMA_VERSION


def test_rlrmp_graph_contract_versions_pin_feedbax_manifest_schema() -> None:
    hps = _hps()
    bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
    )

    assert SCHEMA_VERSION == "rlrmp.feedbax_graph.v1"
    assert SUPPORTED_GRAPH_SPEC_VERSIONS == ("1.0.0",)
    assert FEEDBAX_MANIFEST_SCHEMA_VERSION == "feedbax.manifest.v1"
    assert bundle.graph_spec.metadata is not None
    assert bundle.graph_spec.metadata.version in SUPPORTED_GRAPH_SPEC_VERSIONS
    assert bundle.graph_spec.metadata.version == "1.0.0"
    assert bundle.manifest["schema_version"] == SCHEMA_VERSION
    assert bundle.to_run_metadata()["schema_version"] == SCHEMA_VERSION


def test_minimax_graph_bundle_materializes_runtime_graph() -> None:
    hps = build_hps(_args())
    task = build_task_base(hps)
    bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=task,
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
    )
    spec = bundle.graph_spec
    graph = materialize_rlrmp_graph_spec(spec)

    assert bundle.manifest["execution_backend"] == EXECUTION_BACKEND
    assert isinstance(graph, Graph)
    assert graph.nodes["net"].__class__.__name__ == "Graph"
    assert spec.nodes["feedback"].type == "FeedbackChannels"
    assert spec.nodes["net"].type == "Subgraph"
    assert spec.nodes["net"].params["sisu_gating"] == "additive"
    assert spec.nodes["mechanics"].type == "PointMass"
    assert spec.nodes["mechanics"].params["damping"] == 10.0
    assert spec.nodes["force_filter"].type == "FirstOrderFilter"
    assert spec.nodes[PLANT_INTERVENOR_LABEL].type == "FixedField"
    assert isinstance(graph.nodes[PLANT_INTERVENOR_LABEL], FixedField)
    assert graph.nodes[PLANT_INTERVENOR_LABEL].label == PLANT_INTERVENOR_LABEL
    assert PLANT_INTERVENOR_LABEL in graph.intervention_state_indices()
    assert spec.input_bindings["input"] == ("net", "input")
    assert spec.input_bindings[f"intervene:{PLANT_INTERVENOR_LABEL}"] == (
        PLANT_INTERVENOR_LABEL,
        "params_override",
    )

    recurrent_edges = {
        (wire.source_node, wire.target_node)
        for wire in spec.wires
        if wire.temporality == "recurrent"
    }
    assert ("mechanics", "feedback") in recurrent_edges


def test_multiplicative_point_mass_training_metadata_includes_sisu_alpha() -> None:
    hps = _hps(sisu_gating="multiplicative")
    task = build_task_base(hps)
    bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=task,
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
    )

    assert bundle.training_spec["trainable"] == [
        "nodes.net.hidden",
        "nodes.net.readout",
        "nodes.net.sisu_alpha",
    ]
    assert hps.where["0"] == [
        "nodes.net.hidden",
        "nodes.net.readout",
        "nodes.net.sisu_alpha",
    ]


def test_plain_additive_gru_graph_spec_uses_explicit_feedbax_primitives() -> None:
    hps = _plain_network_hps()
    task = build_task_base(hps)

    spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        task=task,
        n_extra_inputs=1,
        population_structure=_unmasked_population(),
        hidden_type=hps.hidden_type,
        sisu_gating="additive",
    )
    graph = materialize_rlrmp_graph_spec(spec)

    net = spec.nodes["net"]
    assert net.type == "Subgraph"
    assert net.input_ports == ["input", "feedback"]
    assert net.params["sisu_gating"] == "additive"
    assert (
        net.params["input_size"]
        == net.params["external_input_size"] + net.params["feedback_size"]
    )
    assert spec.input_bindings["input"] == ("net", "input")
    assert spec.subgraphs is not None
    net_graph = spec.subgraphs["net"]
    assert net_graph.nodes["input_mux"].type == "Mux"
    assert net_graph.nodes["cell"].type == "GRU"
    assert net_graph.nodes["cell"].params["input_size"] == net.params["input_size"]
    assert net_graph.nodes["readout"].type == "Linear"
    assert net_graph.input_ports == ["input", "feedback"]
    assert net_graph.input_bindings == {"input": ("input_mux", "in_0"), "feedback": ("input_mux", "in_1")}
    assert isinstance(graph.nodes["net"], Graph)


@pytest.mark.parametrize(
    "hps",
    [
        _plain_network_hps(sisu_gating="multiplicative"),
        _hps(),
    ],
)
def test_sisu_and_population_cases_use_native_recurrent_subgraph(hps) -> None:
    spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
    )

    assert spec.nodes["net"].type == "Subgraph"
    assert spec.nodes["net"].params["sisu_gating"] == hps.sisu_gating
    assert spec.subgraphs is not None
    net_graph = spec.subgraphs["net"]
    assert net_graph.nodes["cell"].type == "GRU"
    assert net_graph.nodes["readout"].type == "Linear"
    if hps.sisu_gating == "multiplicative":
        assert net_graph.nodes["sisu_modulator"].type == "ElementwiseAffineModulator"
    if net_graph.parameter_constraints:
        assert {constraint.node for constraint in net_graph.parameter_constraints} == {
            "cell",
            "readout",
        }


@pytest.mark.parametrize(
    ("sisu_gating", "population_structure", "atol"),
    [
        ("additive", None, 0.0),
        ("multiplicative", None, 0.0),
        (
            "additive",
            PopulationStructure.create(
                hidden_size=7,
                n_input_only=1,
                n_readout_only=1,
                n_recurrent_only=1,
                n_input_readout=1,
                key=jr.PRNGKey(19),
            ),
            1e-7,
        ),
    ],
)
def test_native_recurrent_subgraph_matches_legacy_staged_network_fixed_seed(
    sisu_gating: str,
    population_structure: PopulationStructure | None,
    atol: float,
) -> None:
    hidden_size = 7
    out_size = 2
    input_size = 6
    trainable_dtype = "float64" if population_structure is not None else "float32"
    array_dtype = jnp.dtype(trainable_dtype)
    params = {
        "input_size": input_size,
        "external_input_size": 2,
        "hidden_size": hidden_size,
        "out_size": out_size,
        "hidden_type": "GRUCell",
        "sisu_gating": sisu_gating,
        "population_structure": (
            None
            if population_structure is None
            else {**population_structure.to_spec(), "hidden_size": hidden_size}
        ),
        "key": [0, 123],
        "trainable_dtype": trainable_dtype,
    }
    legacy = _build_simple_staged_network(params)
    native_spec = native_recurrent_controller_subgraph(
        input_size=input_size,
        external_input_size=2,
        hidden_size=hidden_size,
        out_size=out_size,
        hidden_type_name="GRUCell",
        sisu_gating=sisu_gating,
        population_params=params["population_structure"],
        key=params["key"],
        dtype=params["trainable_dtype"],
    )
    registry = register_rlrmp_graph_components(
        ComponentRegistry(load_user_components=False, discover_plugins=False)
    )
    native = materialize_rlrmp_graph_spec(
        GraphSpec(
            nodes={
                "net": ComponentSpec(
                    type="Subgraph",
                    params=params,
                    input_ports=list(native_spec.input_ports),
                    output_ports=["output", "hidden"],
                )
            },
            wires=[],
            input_ports=list(native_spec.input_ports),
            output_ports=["output", "hidden"],
            input_bindings={name: ("net", name) for name in native_spec.input_ports},
            output_bindings={"output": ("net", "output"), "hidden": ("net", "hidden")},
            subgraphs={"net": native_spec},
        ),
        registry,
        install_runtime_hooks=False,
    )
    x = jnp.asarray([0.2, -0.1], dtype=array_dtype)
    feedback = jnp.asarray([0.3, -0.4, 0.5, -0.6], dtype=array_dtype)
    inputs = {"input": x, "feedback": feedback}

    legacy_outputs, legacy_state = legacy(
        inputs,
        init_state_from_component(legacy),
        key=jr.PRNGKey(5),
    )
    native_outputs, native_state, _cycle = native.step(
        inputs,
        init_state_from_component(native),
        cycle_port_values=None,
        key=jr.PRNGKey(5),
    )

    assert jnp.allclose(native_outputs["output"], legacy_outputs["output"], rtol=0.0, atol=atol)
    assert jnp.allclose(native_outputs["hidden"], legacy_outputs["hidden"], rtol=0.0, atol=atol)
    del legacy_state, native_state


@pytest.mark.parametrize(
    ("tau_rise", "tau_decay", "expected_present"),
    [
        (0.0, 0.0, False),
        (0.0, 0.07, True),
        (0.05, 0.0, True),
        (0.05, 0.07, True),
    ],
)
def test_force_filter_spec_uses_independent_tau_decay(
    tau_rise: float,
    tau_decay: float,
    expected_present: bool,
) -> None:
    hps = _hps()
    hps = hps | {
        "model": hps.model | {
            "tau_rise": tau_rise,
            "tau_decay": tau_decay,
        }
    }

    spec = build_point_mass_sensorimotor_graph_spec(hps)

    assert ("force_filter" in spec.nodes) is expected_present
    if not expected_present:
        force_edges = [
            (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
            for wire in spec.wires
            if wire.target_port == "force"
        ]
        assert all(edge[0] != "force_filter" and edge[2] != "force_filter" for edge in force_edges)
        return

    force_filter = spec.nodes["force_filter"]
    assert force_filter.type == "FirstOrderFilter"
    assert force_filter.params["tau_rise"] == pytest.approx(tau_rise)
    assert force_filter.params["tau_decay"] == pytest.approx(tau_decay)


def test_graph_spec_serializes_explicit_stochastic_runtime_contract() -> None:
    hps = _hps(
        sensory_noise_std=0.02,
        additive_motor_noise_std=0.03,
        signal_dependent_motor_noise_std=0.04,
        plant_process_force_noise_std=0.05,
    )

    bundle = build_rlrmp_feedbax_graph_bundle(hps)
    spec = bundle.graph_spec

    assert spec.nodes["feedback"].params["noise_std"] == 0.02
    assert spec.nodes["feedback"].params["noise_role"] == "sensory_feedback"
    assert spec.nodes["efferent"].params["additive_noise_std"] == 0.03
    assert spec.nodes["efferent"].params["signal_dependent_noise_std"] == 0.04
    assert spec.nodes["efferent"].params["noise_timing"] == "pre_force_filter"
    assert spec.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL].type == "Channel"
    assert spec.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL].params["noise_std"] == 0.05
    assert spec.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL].params["state_diffusion"] is False
    assert bundle.manifest["stochastic_runtime"]["state_diffusion"] == "not_used"

    force_edges = [
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in spec.wires
        if wire.source_port == "force" or wire.target_port == "force"
    ]
    assert ("force_filter", "output", GRAPH_PLANT_INTERVENOR_NODE, "force") in force_edges
    assert (
        GRAPH_PLANT_INTERVENOR_NODE,
        "force",
        PLANT_PROCESS_FORCE_NOISE_LABEL,
        "input",
    ) in force_edges
    assert (PLANT_PROCESS_FORCE_NOISE_LABEL, "output", "mechanics", "force") in force_edges


def _linear_controller_fixed_inputs():
    gain = jnp.asarray(
        [
            [
                [1.5, -0.5, 0.25, 2.0],
                [-1.0, 0.75, -0.5, 1.25],
            ]
        ],
        dtype=jnp.float32,
    )
    feedforward = jnp.asarray([[0.2, -0.1]], dtype=jnp.float32)
    target = jnp.asarray([0.4, -0.2], dtype=jnp.float32)
    feedback = (
        jnp.asarray([0.1, 0.3], dtype=jnp.float32),
        jnp.asarray([0.5, -0.4], dtype=jnp.float32),
    )
    input_value = {
        "task": SimpleNamespace(
            effector_target=SimpleNamespace(pos=target),
        )
    }
    reference = jnp.concatenate([target, jnp.zeros(2, dtype=target.dtype)])
    feedback_vector = jnp.concatenate(feedback)
    return gain, feedforward, input_value, feedback, reference, feedback_vector


def _with_float32_network_state(controller):
    init_state = type(controller._initial_state)(
        input=jnp.zeros(0),
        hidden=jnp.zeros((1,), dtype=jnp.float32),
        output=jnp.zeros(controller.n_controls, dtype=jnp.float32),
        encoding=None,
    )
    object.__setattr__(controller, "_initial_state", init_state)
    object.__setattr__(controller.state_index, "init", init_state)
    return controller


def test_affine_linear_controller_sign_convention_matches_rlrmp_forms() -> None:
    gain, feedforward, input_value, feedback, reference, feedback_vector = (
        _linear_controller_fixed_inputs()
    )

    regulator = _with_float32_network_state(LinearController(n_steps=1, key=jr.PRNGKey(0)))
    tracker = _with_float32_network_state(LinearTrackerController(n_steps=1, key=jr.PRNGKey(0)))
    regulator = eqx.tree_at(lambda controller: controller.K, regulator, gain)
    tracker = eqx.tree_at(
        lambda controller: (controller.K, controller.u_ff),
        tracker,
        (gain, feedforward),
    )
    affine_regulator = AffineFeedbackController(gain=gain)
    affine_tracker = AffineFeedbackController(gain=gain, feedforward=feedforward)

    legacy_inputs = {"input": input_value, "feedback": feedback}
    affine_inputs = {"feedback": feedback_vector, "reference": reference}

    legacy_regulator_output, _ = regulator(
        legacy_inputs,
        init_state_from_component(regulator),
        key=jr.PRNGKey(1),
    )
    affine_regulator_output, _ = affine_regulator(
        affine_inputs,
        init_state_from_component(affine_regulator),
        key=jr.PRNGKey(1),
    )
    legacy_tracker_output, _ = tracker(
        legacy_inputs,
        init_state_from_component(tracker),
        key=jr.PRNGKey(1),
    )
    affine_tracker_output, _ = affine_tracker(
        affine_inputs,
        init_state_from_component(affine_tracker),
        key=jr.PRNGKey(1),
    )

    assert jnp.allclose(
        legacy_regulator_output["output"],
        jnp.asarray([1.375, 0.075], dtype=jnp.float32),
        rtol=0.0,
        atol=1e-6,
    )
    assert jnp.allclose(
        affine_regulator_output["command"],
        legacy_regulator_output["output"],
        rtol=0.0,
        atol=1e-6,
    )
    assert jnp.allclose(
        legacy_tracker_output["output"],
        jnp.asarray([1.575, -0.025], dtype=jnp.float32),
        rtol=0.0,
        atol=1e-6,
    )
    assert jnp.allclose(
        affine_tracker_output["command"],
        legacy_tracker_output["output"],
        rtol=0.0,
        atol=1e-6,
    )


def test_linear_tracker_uses_native_affine_controller_graph_structure() -> None:
    hps = _hps(hidden_type="linear_tracker")
    task = build_task_base(hps)
    bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=task,
        n_extra_inputs=0,
        hidden_type=hps.hidden_type,
    )
    graph = materialize_rlrmp_graph_spec(bundle.graph_spec)

    net = bundle.graph_spec.nodes["net"]
    assert net.type == "AffineFeedbackController"
    assert jnp.asarray(net.params["gain"]).shape == (hps.task.n_steps - 1, 2, 4)
    assert jnp.asarray(net.params["feedforward"]).shape == (hps.task.n_steps - 1, 2)
    assert bundle.graph_spec.nodes["feedback_vector"].type == "Ravel"
    assert bundle.graph_spec.nodes["zero_velocity"].type == "Constant"
    assert bundle.graph_spec.nodes["zero_velocity"].params["value"] == [0.0, 0.0]
    assert bundle.graph_spec.nodes["reference_mux"].type == "Mux"
    task_binding_spec = bundle.task_spec["task_binding_spec"]
    binding = task_binding_spec["bindings"][0]
    data = task_binding_spec["exposed_data"][0]
    assert binding["id"] == POINT_MASS_TARGET_POSITION_INPUT
    assert binding["target_node_id"] == "reference_mux"
    assert binding["target_port"] == "in_0"
    assert data["path"] == "inputs.effector_target.pos"
    assert bundle.graph_spec.input_bindings[POINT_MASS_TARGET_POSITION_INPUT] == (
        "reference_mux",
        "in_0",
    )
    assert graph.nodes["net"].__class__.__name__ == "AffineFeedbackController"
    assert "net_state" not in bundle.graph_spec.nodes
    assert "net_state" not in graph.nodes
    assert graph.input_bindings[POINT_MASS_TARGET_POSITION_INPUT] == ("reference_mux", "in_0")
    assert bundle.training_spec["trainable"] == [
        "nodes.net.gain",
        "nodes.net.feedforward",
    ]
    assert ("zero_velocity", "output", "reference_mux", "in_1") in {
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in bundle.graph_spec.wires
    }
    assert ("reference_mux", "output", "net", "reference") in {
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in bundle.graph_spec.wires
    }
    assert ("net", "command", "efferent", "input") in {
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in bundle.graph_spec.wires
    }


def test_legacy_generic_rlrmp_graph_component_ids_materialize_through_migration() -> None:
    hps = _hps(
        hidden_type="linear",
        sensory_noise_std=0.02,
        additive_motor_noise_std=0.03,
        signal_dependent_motor_noise_std=0.04,
        plant_process_force_noise_std=0.05,
    )
    spec = build_point_mass_sensorimotor_graph_spec(hps)

    nodes = dict(spec.nodes)
    nodes["feedback"] = nodes["feedback"].model_copy(update={"type": "RLRMPFeedbackChannels"})
    nodes["efferent"] = nodes["efferent"].model_copy(update={"type": "RLRMPMotorChannel"})
    nodes["mechanics"] = nodes["mechanics"].model_copy(update={"type": "RLRMPPointMass"})
    nodes[PLANT_PROCESS_FORCE_NOISE_LABEL] = nodes[PLANT_PROCESS_FORCE_NOISE_LABEL].model_copy(
        update={
            "type": "RLRMPPlantProcessForceNoise",
            "input_ports": ["force"],
            "output_ports": ["force"],
        }
    )
    wires = [
        wire.model_copy(update={"target_port": "force"})
        if wire.target_node == PLANT_PROCESS_FORCE_NOISE_LABEL
        else wire.model_copy(update={"source_port": "force"})
        if wire.source_node == PLANT_PROCESS_FORCE_NOISE_LABEL
        else wire
        for wire in spec.wires
    ]
    legacy_spec = spec.model_copy(update={"nodes": nodes, "wires": wires})

    registry = register_rlrmp_graph_components(
        ComponentRegistry(load_user_components=False, discover_plugins=False)
    )
    graph = materialize_rlrmp_graph_spec(legacy_spec, registry)

    assert isinstance(graph, Graph)
    assert graph.nodes["feedback"].__class__.__name__ == "FeedbackChannels"
    assert graph.nodes["efferent"].__class__.__name__ == "Channel"
    assert graph.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL].__class__.__name__ == "Channel"
    feedback_definition = next(item for item in registry.list_all() if item.name == "FeedbackChannels")
    feedback_migration = next(
        item for item in feedback_definition.migrations if item.source_type == "RLRMPFeedbackChannels"
    )
    assert feedback_migration.owner == "rlrmp"
    assert feedback_migration.target_type == "FeedbackChannels"


def test_absent_rlrmp_component_migration_pack_fails_with_owner_context() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    spec = build_point_mass_sensorimotor_graph_spec(_hps())
    nodes = dict(spec.nodes)
    nodes["feedback"] = nodes["feedback"].model_copy(
        update={"type": "rlrmp.RLRMPFeedbackChannels"}
    )
    legacy_spec = spec.model_copy(update={"nodes": nodes})

    with pytest.raises(ValueError) as exc_info:
        resolve_registered_graph_component_migrations(legacy_spec, registry)

    message = str(exc_info.value)
    assert "owner='rlrmp'" in message
    assert "migration pack" in message
    assert "rlrmp.RLRMPFeedbackChannels" in message


def test_b41c940_manifest_inline_graph_validates_through_rlrmp_pack() -> None:
    manifest_path = (
        Path("results")
        / "b41c940"
        / "migrated"
        / "efc4d68"
        / "baseline_gru__smooth"
        / "model.artifact.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph_payload = manifest["graph_spec"]["inline"]
    assert graph_payload["nodes"]["feedback"]["type"] == "RLRMPFeedbackChannels"
    assert manifest["migration_records"] == []

    registry = register_rlrmp_graph_components(
        ComponentRegistry(load_user_components=False, discover_plugins=False)
    )
    migrated = resolve_registered_graph_component_migrations(
        GraphSpec.model_validate(graph_payload),
        registry,
    )

    assert migrated.nodes["feedback"].type == "FeedbackChannels"
    assert migrated.nodes["feedback"].params["selector"] == "paths"
    assert migrated.nodes["feedback"].params["paths"] == [
        "plant.skeleton.pos",
        "plant.skeleton.vel",
    ]


def test_rlrmp_registry_uses_feedbax_intervention_builders() -> None:
    registry = register_rlrmp_graph_components(
        ComponentRegistry(load_user_components=False, discover_plugins=False)
    )

    for component_type in ("FixedField", "CurlField", "DynamicsMatrixPerturb"):
        meta = registry.get(component_type)
        assert meta is not None
        assert meta.builder is not None
        assert meta.provenance == "feedbax"
        assert meta.output_prototype_fn is not None


def test_dynamics_matrix_perturb_spec_preserves_delta_a_contract() -> None:
    hps = _hps(hidden_type="linear")
    spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=0,
        hidden_type=hps.hidden_type,
        intervention_type="DynamicsMatrixPerturb",
    )
    graph = materialize_rlrmp_graph_spec(spec)

    intervenor = spec.nodes[GRAPH_PLANT_INTERVENOR_NODE]
    assert intervenor.type == "DynamicsMatrixPerturb"
    assert jnp.asarray(intervenor.params["delta_A"]).shape == (2, 4)
    assert "delta_A_shape" not in intervenor.params
    assert "effector" in intervenor.input_ports
    assert isinstance(graph.nodes[PLANT_INTERVENOR_LABEL], DynamicsMatrixPerturb)
    assert graph.nodes[PLANT_INTERVENOR_LABEL].label == PLANT_INTERVENOR_LABEL
    assert spec.input_bindings[f"intervene:{GRAPH_PLANT_INTERVENOR_NODE}"] == (
        GRAPH_PLANT_INTERVENOR_NODE,
        "params_override",
    )

    recurrent_edges = {
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in spec.wires
        if wire.temporality == "recurrent"
    }
    assert (
        "mechanics",
        "effector",
        GRAPH_PLANT_INTERVENOR_NODE,
        "effector",
    ) in recurrent_edges


def test_legacy_dynamics_matrix_delta_a_shape_materializes_through_migration() -> None:
    hps = _hps(hidden_type="linear")
    spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=0,
        hidden_type=hps.hidden_type,
        intervention_type="DynamicsMatrixPerturb",
    )
    nodes = dict(spec.nodes)
    intervenor = nodes[GRAPH_PLANT_INTERVENOR_NODE]
    params = dict(intervenor.params)
    params.pop("delta_A")
    params["delta_A_shape"] = [2, 4]
    nodes[GRAPH_PLANT_INTERVENOR_NODE] = intervenor.model_copy(update={"params": params})
    legacy_spec = spec.model_copy(update={"nodes": nodes})

    graph = materialize_rlrmp_graph_spec(legacy_spec)

    materialized = graph.nodes[PLANT_INTERVENOR_LABEL]
    assert isinstance(materialized, DynamicsMatrixPerturb)
    assert materialized.label == PLANT_INTERVENOR_LABEL
    assert materialized.input_ports == ("effector", "force", "params_override")
    assert materialized.output_ports == ("force",)
    assert materialized._initial_state.delta_A.shape == (2, 4)


def test_legacy_curl_field_missing_amplitude_materializes_through_migration() -> None:
    hps = _hps(hidden_type="linear")
    spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=0,
        hidden_type=hps.hidden_type,
        intervention_type="CurlField",
    )
    nodes = dict(spec.nodes)
    intervenor = nodes[GRAPH_PLANT_INTERVENOR_NODE]
    params = dict(intervenor.params)
    params.pop("amplitude")
    nodes[GRAPH_PLANT_INTERVENOR_NODE] = intervenor.model_copy(update={"params": params})
    legacy_spec = spec.model_copy(update={"nodes": nodes})

    graph = materialize_rlrmp_graph_spec(legacy_spec)

    materialized = graph.nodes[PLANT_INTERVENOR_LABEL]
    assert isinstance(materialized, CurlField)
    assert materialized.label == PLANT_INTERVENOR_LABEL
    assert materialized._initial_state.amplitude == pytest.approx(1.0)


def test_write_graph_spec_bundle_creates_companion_manifest(tmp_path) -> None:
    hps = build_hps(_args())
    bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
    )

    graph_path = write_graph_spec_bundle(bundle, tmp_path)

    graph_payload = json.loads(graph_path.read_text())
    manifest_text = (tmp_path / "model.graph.manifest.json").read_text()
    manifest_payload = json.loads(manifest_text)
    assert len(manifest_text.splitlines()) == 1
    assert "\n  " not in manifest_text
    assert graph_payload["nodes"]["net"]["type"] == "Subgraph"
    assert graph_payload == graph_spec_payload(bundle.graph_spec)
    round_tripped = GraphSpec.model_validate(graph_payload)
    assert graph_spec_payload(round_tripped) == graph_payload
    assert isinstance(materialize_rlrmp_graph_spec(round_tripped), Graph)
    assert manifest_payload["schema_version"] == "rlrmp.feedbax_graph.v1"
    assert bundle.to_run_metadata()["graph_spec_path"] == "model.graph.json"
    assert bundle.to_run_metadata()["graph_export_status"] == "available"
    assert bundle.to_run_metadata()["manifest_path"] == "model.graph.manifest.json"


def test_setup_task_model_pair_materializes_gru_and_linear_graph_paths() -> None:
    import jax.random as jr

    for hidden_type, expected_net in [
        ("gru", "Graph"),
        ("linear", "AffineFeedbackController"),
    ]:
        hps = build_hps(_args(hidden_type=hidden_type, n_replicates=2))
        pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

        assert isinstance(pair.model, Graph)
        assert pair.model.nodes["net"].__class__.__name__ == expected_net
        assert pair.model.nodes[PLANT_INTERVENOR_LABEL].label == PLANT_INTERVENOR_LABEL
        assert PLANT_INTERVENOR_LABEL in pair.model.intervention_state_indices()


@pytest.mark.parametrize(
    ("hidden_type", "expected_net"),
    [
        ("gru", "Subgraph"),
        ("linear", "AffineFeedbackController"),
    ],
)
@pytest.mark.parametrize(
    ("pert_type", "expected_intervenor", "expected_class"),
    [
        ("constant", "FixedField", FixedField),
        ("curl", "CurlField", CurlField),
    ],
)
def test_runtime_graph_bundle_exports_constructed_model_intervenor(
    hidden_type: str,
    expected_net: str,
    pert_type: str,
    expected_intervenor: str,
    expected_class: type,
) -> None:
    hps = _hps(hidden_type=hidden_type, n_replicates=2)
    hps = hps | {"pert": hps.pert | {"type": pert_type}}
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    bundle = build_runtime_rlrmp_feedbax_graph_bundle(hps, pair.model)
    graph = materialize_rlrmp_graph_spec(bundle.graph_spec)

    assert bundle.graph_spec.nodes[PLANT_INTERVENOR_LABEL].type == expected_intervenor
    assert isinstance(pair.model.nodes[PLANT_INTERVENOR_LABEL], expected_class)
    assert isinstance(graph.nodes[PLANT_INTERVENOR_LABEL], expected_class)
    if hidden_type == "gru":
        assert bundle.graph_spec.nodes["net"].type == "Subgraph"
        assert bundle.graph_spec.subgraphs is not None
        net_graph = bundle.graph_spec.subgraphs["net"]
        assert net_graph.nodes["input_mux"].type == "Mux"
        assert net_graph.nodes["cell"].type == "GRU"
        assert net_graph.nodes["readout"].type == "Linear"
        assert graph.nodes["net"].nodes["cell"].__class__.__name__ == "GRU"
    else:
        assert bundle.graph_spec.nodes["net"].type == expected_net
        assert bundle.graph_spec.nodes["reference_mux"].type == "Mux"
        assert bundle.graph_spec.nodes["zero_velocity"].type == "Constant"
        assert "net_state" not in bundle.graph_spec.nodes
    assert bundle.graph_spec.nodes["mechanics"].params["damping"] == hps.model.damping
    assert bundle.graph_spec.nodes["feedback"].params["delay"] == hps.model.feedback_delay_steps
    assert bundle.graph_spec.nodes["efferent"].type == "Channel"
    assert bundle.graph_spec.nodes["efferent"].params["input_shape"] == [2]


def test_runtime_graph_bundle_catches_old_fixedfield_default_mismatch() -> None:
    hps = _hps(hidden_type="gru", n_replicates=2)
    hps = hps | {"pert": hps.pert | {"type": "curl"}}
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    runtime_bundle = build_runtime_rlrmp_feedbax_graph_bundle(hps, pair.model)
    stale_hps_bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
        key=jr.PRNGKey(0),
    )

    assert runtime_bundle.graph_spec.nodes[PLANT_INTERVENOR_LABEL].type == "CurlField"
    assert stale_hps_bundle.graph_spec.nodes[PLANT_INTERVENOR_LABEL].type == "FixedField"
    assert graph_spec_payload(runtime_bundle.graph_spec) != graph_spec_payload(
        stale_hps_bundle.graph_spec
    )


def test_runtime_graph_spec_preserves_dynamics_matrix_intervenor() -> None:
    hps = _hps(hidden_type="gru", n_replicates=2)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    runtime_model = swap_plant_intervenor_to_dynamics_matrix(
        pair.model,
        PLANT_INTERVENOR_LABEL,
        mass=hps.model.effector_mass,
    )

    graph_spec = graph_spec_from_model(
        runtime_model,
        n_replicates=int(hps.model.n_replicates),
    )
    graph = materialize_rlrmp_graph_spec(graph_spec)

    assert graph_spec.nodes[PLANT_INTERVENOR_LABEL].type == "DynamicsMatrixPerturb"
    assert jnp.asarray(graph_spec.nodes[PLANT_INTERVENOR_LABEL].params["delta_A"]).shape == (2, 4)
    assert "delta_A_shape" not in graph_spec.nodes[PLANT_INTERVENOR_LABEL].params
    assert isinstance(graph.nodes[PLANT_INTERVENOR_LABEL], DynamicsMatrixPerturb)
