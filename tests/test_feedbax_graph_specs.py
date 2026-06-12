"""RLRMP GraphSpec migration tests."""

from __future__ import annotations

import argparse
import json

import jax.random as jr
import pytest
from feedbax.contracts.graph import GraphSpec
from feedbax.graph import Graph
from feedbax.intervene import CurlField, DynamicsMatrixPerturb, FixedField
from feedbax.manifest import SCHEMA_VERSION as FEEDBAX_MANIFEST_SCHEMA_VERSION

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.feedbax_graph import (
    EXECUTION_BACKEND,
    GRAPH_PLANT_INTERVENOR_NODE,
    SCHEMA_VERSION,
    build_point_mass_sensorimotor_graph_spec,
    build_rlrmp_feedbax_graph_bundle,
    build_runtime_rlrmp_feedbax_graph_bundle,
    graph_spec_from_model,
    graph_spec_payload,
    materialize_rlrmp_graph_spec,
    write_graph_spec_bundle,
)
from rlrmp.intervention_compat import swap_plant_intervenor_to_dynamics_matrix
from rlrmp.modules.training.part2 import build_task_base, setup_task_model_pair
from rlrmp.stochastic_runtime import PLANT_PROCESS_FORCE_NOISE_LABEL
from rlrmp.train.minimax import build_hps


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
    assert FEEDBAX_MANIFEST_SCHEMA_VERSION == "feedbax.manifest.v1"
    assert bundle.graph_spec.metadata is not None
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
    assert graph.nodes["net"].__class__.__name__ == "SimpleStagedNetwork"
    assert spec.nodes["feedback"].type == "RLRMPFeedbackChannels"
    assert spec.nodes["net"].type == "RLRMPSimpleStagedNetwork"
    assert spec.nodes["net"].params["sisu_gating"] == "additive"
    assert spec.nodes["mechanics"].type == "RLRMPPointMass"
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
    assert spec.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL].type == "RLRMPPlantProcessForceNoise"
    assert spec.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL].params["noise_std"] == 0.05
    assert spec.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL].params["state_diffusion"] is False
    assert bundle.manifest["stochastic_runtime"]["state_diffusion"] == "not_used"

    force_edges = [
        (wire.source_node, wire.target_node) for wire in spec.wires if wire.target_port == "force"
    ]
    assert ("force_filter", GRAPH_PLANT_INTERVENOR_NODE) in force_edges
    assert (GRAPH_PLANT_INTERVENOR_NODE, PLANT_PROCESS_FORCE_NOISE_LABEL) in force_edges
    assert (PLANT_PROCESS_FORCE_NOISE_LABEL, "mechanics") in force_edges


def test_linear_tracker_is_graphspec_addressable_as_rlrmp_component() -> None:
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
    assert net.type == "RLRMPLinearTrackerController"
    assert net.params["n_steps"] == hps.task.n_steps - 1
    assert net.params["target_source"] == "input.task.effector_target.pos"
    assert graph.nodes["net"].__class__.__name__ == "LinearTrackerController"
    assert bundle.training_spec["trainable"] == ["nodes.net.K", "nodes.net.u_ff"]


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
    assert intervenor.params["delta_A_shape"] == [2, 4]
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
    manifest_payload = json.loads((tmp_path / "model.graph.manifest.json").read_text())
    assert graph_payload["nodes"]["net"]["type"] == "RLRMPSimpleStagedNetwork"
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
        ("gru", "SimpleStagedNetwork"),
        ("linear", "LinearController"),
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
        ("gru", "Network"),
        ("linear", "RLRMPLinearController"),
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
    assert bundle.graph_spec.nodes["net"].type == expected_net
    assert bundle.graph_spec.nodes["mechanics"].params["damping"] == hps.model.damping
    assert bundle.graph_spec.nodes["feedback"].params["delay"] == hps.model.feedback_delay_steps
    assert bundle.graph_spec.nodes["efferent"].type == "RLRMPMotorChannel"
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
    assert graph_spec.nodes[PLANT_INTERVENOR_LABEL].params["delta_A_shape"] == [2, 4]
    assert isinstance(graph.nodes[PLANT_INTERVENOR_LABEL], DynamicsMatrixPerturb)
