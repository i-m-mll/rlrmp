"""Regression tests for issue ae15851 archived graph recipe conversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.graphs.serialization import spec_to_graph
from feedbax.runtime.graph import Graph

from rlrmp.runtime.run_specs import (
    CS_LSS_FEEDBACK_COMPONENT_TYPES,
    LEGACY_POINT_MASS_GRAPH_TYPES,
)
from rlrmp.runtime.graph_spec_migrations import migrate_feedbax_graph_payload


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_MANIFEST_PATH = REPO_ROOT / "results/e9fc384/notes/graph_sidecar_audit_manifest.json"
CONVERSION_MANIFEST_PATH = REPO_ROOT / "results/ae15851/converted/conversion_manifest.json"
CONVERTED_DIR = REPO_ROOT / "results/ae15851/converted"
RETIRED_TYPES = frozenset(
    {
        "RLRMPSimpleStagedNetwork",
        "RLRMPLinearController",
        "RLRMPLinearTrackerController",
        "RLRMPCsLssInitialHiddenStagedNetwork",
        "RLRMPFeedbackChannels",
        "RLRMPMotorChannel",
        "RLRMPPlantProcessForceNoise",
        "RLRMPPointMass",
        "rlrmp.RLRMPFeedbackChannels",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _conversion_manifest() -> dict[str, Any]:
    assert CONVERSION_MANIFEST_PATH.is_file(), "ae15851 conversion manifest is missing"
    return _load_json(CONVERSION_MANIFEST_PATH)


def _retired_type_hits(payload: Any) -> list[str]:
    hits: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            type_value = value.get("type")
            if type_value in RETIRED_TYPES:
                hits.append(str(type_value))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return hits


def test_conversion_manifest_is_exactly_audit_manifest_driven() -> None:
    audit = _load_json(AUDIT_MANIFEST_PATH)
    conversion = _conversion_manifest()

    assert audit["expected_count"] == 38
    assert audit["audited_count"] == 36
    assert conversion["source_manifest"] == "results/e9fc384/notes/graph_sidecar_audit_manifest.json"
    assert conversion["audit_manifest_sha256"] == _sha256(AUDIT_MANIFEST_PATH)
    assert conversion["expected_count"] == audit["expected_count"]
    assert conversion["audited_count"] == audit["audited_count"]
    assert len(conversion["entries"]) == len(audit["files"])
    assert conversion["converted_count"] == 36
    assert conversion["excluded_count"] == 0

    audit_by_path = {entry["path"]: entry for entry in audit["files"]}
    conversion_by_path = {entry["original_path"]: entry for entry in conversion["entries"]}
    assert set(conversion_by_path) == set(audit_by_path)

    for path, entry in conversion_by_path.items():
        audit_entry = audit_by_path[path]
        assert entry["original_sha256"] == audit_entry["sha256"]
        assert entry["conversion_candidate_key"] == audit_entry["conversion_candidate_key"]
        if audit_entry["classification"] == "known_wrong":
            assert entry["disposition"] == "excluded_known_wrong"
            assert entry["converted_path"] is None
            assert entry["exclusion_reason"] == audit_entry["classification_reason"]
        else:
            assert audit_entry["classification"] == "clean"
            assert entry["disposition"] == "converted"
            assert entry["converted_path"].startswith("results/ae15851/converted/")


def test_converted_recipe_set_is_hash_pinned_by_conversion_manifest() -> None:
    conversion = _conversion_manifest()
    converted_entries = [
        entry for entry in conversion["entries"] if entry["disposition"] == "converted"
    ]
    converted_paths = {REPO_ROOT / entry["converted_path"] for entry in converted_entries}
    live_paths = set(CONVERTED_DIR.glob("*.graph.json"))

    assert len(converted_entries) == 36
    assert live_paths == converted_paths
    for entry in converted_entries:
        path = REPO_ROOT / entry["converted_path"]
        assert path.is_file(), entry["converted_path"]
        assert _sha256(path) == entry["converted_sha256"], entry["converted_path"]


def test_every_converted_recipe_loads_with_plain_feedbax_spec_to_graph() -> None:
    conversion = _conversion_manifest()
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    loaded = 0

    for entry in conversion["entries"]:
        if entry["disposition"] != "converted":
            continue
        payload = _load_json(REPO_ROOT / entry["converted_path"])
        original = _load_json(REPO_ROOT / entry["original_path"])

        assert _retired_type_hits(payload) == [], entry["converted_path"]
        assert "input_size_source" not in json.dumps(payload), entry["converted_path"]
        assert payload["metadata"]["version"] == "1.0.0"
        assert payload["nodes"].keys() == original["nodes"].keys()
        assert payload["wires"] == original["wires"]
        assert payload["input_bindings"] == original["input_bindings"]
        assert payload["output_bindings"] == original["output_bindings"]
        assert payload["retained_observables"] == original["retained_observables"]

        nodes = payload["nodes"]
        assert nodes["feedback"]["type"] == "FeedbackChannels"
        assert nodes["feedback"]["params"]["selector"] == "paths"
        assert nodes["feedback"]["params"]["paths"] == [
            "plant.skeleton.pos",
            "plant.skeleton.vel",
        ]
        assert nodes["efferent"]["type"] == "Channel"
        assert nodes["efferent"]["params"]["noise_model"] == "signal_dependent_plus_additive"
        assert nodes["efferent"]["params"]["signal_dependent_noise_std"] == 0.01
        assert nodes["efferent"]["params"]["additive_noise_std"] == pytest.approx(0.018)
        assert nodes["mechanics"]["type"] == "PointMass"
        assert nodes["force_filter"]["type"] == "FirstOrderFilter"
        assert nodes["plant_intervenor"]["type"] == "FixedField"

        net = nodes["net"]
        assert net["type"] == "Subgraph"
        assert net["params"]["input_size"] == 11
        assert net["params"]["external_input_size"] == 7
        assert net["params"]["feedback_size"] == 4
        assert net["params"]["population_structure"] == {
            "hidden_size": 180,
            "n_input_only": 60,
            "n_readout_only": 60,
            "n_recurrent_only": 60,
            "n_input_readout": 0,
        }

        expected_cell = "VanillaRNN" if entry["controller_kind"] == "vanilla_rnn" else "GRU"
        assert payload["subgraphs"]["net"]["nodes"]["cell"]["type"] == expected_cell
        graph_payload = migrate_feedbax_graph_payload(payload)
        graph = spec_to_graph(GraphSpec.model_validate(graph_payload), component_registry=registry)
        assert isinstance(graph.nodes["net"], Graph)
        assert graph.nodes["net"].nodes["cell"].__class__.__name__ == expected_cell
        loaded += 1

    assert loaded == conversion["converted_count"] == 36


def test_run_specs_allowlists_are_legacy_archive_confinement_lists() -> None:
    assert CS_LSS_FEEDBACK_COMPONENT_TYPES == frozenset(
        {
            "StateFeedbackSelector",
            "RLRMPCsLssDelayedPositionVelocityFeedback",
            "RLRMPCsLssTargetRelativeDelayedFeedback",
            "RLRMPCsLssTargetRelativeDelayedProprioceptiveFeedback",
        }
    )
    assert LEGACY_POINT_MASS_GRAPH_TYPES == frozenset(
        {
            "FirstOrderFilter",
            "PointMass",
            "RLRMPFeedbackChannels",
            "RLRMPPointMass",
        }
    )

    conversion = _conversion_manifest()
    for entry in conversion["entries"]:
        if entry["disposition"] != "converted":
            continue
        payload = _load_json(REPO_ROOT / entry["converted_path"])
        node_types = {node["type"] for node in payload["nodes"].values()}
        assert "RLRMPFeedbackChannels" not in node_types
        assert "RLRMPPointMass" not in node_types
        assert "StateFeedbackSelector" not in node_types
