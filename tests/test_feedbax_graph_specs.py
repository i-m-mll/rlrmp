"""RLRMP GraphSpec migration tests."""

from __future__ import annotations

import argparse
import json

from rlrmp.feedbax_graph import (
    EXECUTION_BACKEND,
    PLANT_INTERVENOR_LABEL,
    build_point_mass_sensorimotor_graph_spec,
    build_rlrmp_feedbax_graph_bundle,
    write_graph_spec_bundle,
)
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


def test_minimax_graph_bundle_serializes_legacy_feedback_contract() -> None:
    hps = build_hps(_args())
    bundle = build_rlrmp_feedbax_graph_bundle(hps)
    spec = bundle.graph_spec

    assert bundle.manifest["execution_backend"] == EXECUTION_BACKEND
    assert spec.nodes["feedback"].type == "RLRMPFeedbackChannels"
    assert spec.nodes["net"].type == "RLRMPSimpleStagedNetwork"
    assert spec.nodes["net"].params["sisu_gating"] == "additive"
    assert spec.nodes["mechanics"].type == "PointMass"
    assert spec.nodes["mechanics"].params["damping"] == 10.0
    assert spec.nodes["force_filter"].type == "FirstOrderFilter"
    assert spec.nodes[PLANT_INTERVENOR_LABEL].type == "FixedField"
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


def test_linear_tracker_is_graphspec_addressable_as_rlrmp_component() -> None:
    hps = build_hps(_args(hidden_type="linear_tracker"))
    bundle = build_rlrmp_feedbax_graph_bundle(hps)

    net = bundle.graph_spec.nodes["net"]
    assert net.type == "RLRMPLinearTrackerController"
    assert net.params["n_steps"] == hps.task.n_steps - 1
    assert net.params["target_source"] == "input.task.effector_target.pos"
    assert bundle.training_spec["trainable"] == ["nodes.net.K", "nodes.net.u_ff"]


def test_dynamics_matrix_perturb_spec_preserves_delta_a_contract() -> None:
    hps = build_hps(_args(hidden_type="linear"))
    spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        intervention_type="DynamicsMatrixPerturb",
    )

    intervenor = spec.nodes[PLANT_INTERVENOR_LABEL]
    assert intervenor.type == "DynamicsMatrixPerturb"
    assert intervenor.params["delta_A_shape"] == [2, 4]
    assert "effector" in intervenor.input_ports
    assert spec.input_bindings[f"intervene:{PLANT_INTERVENOR_LABEL}"] == (
        PLANT_INTERVENOR_LABEL,
        "params_override",
    )

    recurrent_edges = {
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in spec.wires
        if wire.temporality == "recurrent"
    }
    assert ("mechanics", "effector", PLANT_INTERVENOR_LABEL, "effector") in recurrent_edges


def test_write_graph_spec_bundle_creates_companion_manifest(tmp_path) -> None:
    hps = build_hps(_args())
    bundle = build_rlrmp_feedbax_graph_bundle(hps)

    graph_path = write_graph_spec_bundle(bundle, tmp_path)

    graph_payload = json.loads(graph_path.read_text())
    manifest_payload = json.loads((tmp_path / "model.graph.manifest.json").read_text())
    assert graph_payload["nodes"]["net"]["type"] == "RLRMPSimpleStagedNetwork"
    assert manifest_payload["schema_version"] == "rlrmp.feedbax_graph.v1"
    assert bundle.to_run_metadata()["graph_spec_path"] == "model.graph.json"
    assert bundle.to_run_metadata()["manifest_path"] == "model.graph.manifest.json"
