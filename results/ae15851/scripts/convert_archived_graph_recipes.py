"""Convert audited archived graph recipes to plain Feedbax GraphSpecs.

This one-time driver consumes the issue e9fc384 audit manifest. It does not
glob for conversion targets: the manifest is the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from feedbax.contracts.graph import ComponentSpec, GraphMetadata, GraphSpec
from feedbax.contracts.graphs.templates import recurrent_controller_template_graph
from feedbax.models.networks import population_structure_from_spec


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_MANIFEST_PATH = (
    REPO_ROOT / "results" / "e9fc384" / "notes" / "graph_sidecar_audit_manifest.json"
)
OUTPUT_DIR = REPO_ROOT / "results" / "ae15851" / "converted"
CONVERSION_MANIFEST_PATH = OUTPUT_DIR / "conversion_manifest.json"

SCHEMA_VERSION = 1
FEEDBAX_GRAPH_VERSION = "1.0.0"
ARCHIVE_GRAPH_VERSION = "rlrmp.feedbax_graph.v1"
POINT_MASS_INPUT_SIZE = 11
POINT_MASS_EXTERNAL_INPUT_SIZE = 7
POINT_MASS_FEEDBACK_SIZE = 4
RETIRED_COMPONENT_TYPES = frozenset(
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


def main() -> None:
    manifest = _load_json(AUDIT_MANIFEST_PATH)
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 38:
        raise ValueError("audit manifest must contain exactly 38 files")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT_DIR.glob("*.graph.json"):
        stale.unlink()

    entries: list[dict[str, Any]] = []
    for audit_entry in files:
        original_relpath = str(audit_entry["path"])
        original_path = REPO_ROOT / original_relpath
        original_payload = _load_json(original_path)
        live_hash = _sha256(original_path)
        if live_hash != audit_entry["sha256"]:
            raise ValueError(
                f"{original_relpath}: live hash {live_hash} differs from audit "
                f"hash {audit_entry['sha256']}"
            )

        manifest_entry = _base_manifest_entry(audit_entry)
        if audit_entry["classification"] == "known_wrong":
            manifest_entry.update(
                {
                    "disposition": "excluded_known_wrong",
                    "converted_path": None,
                    "exclusion_reason": audit_entry["classification_reason"],
                }
            )
            entries.append(manifest_entry)
            continue

        converted = convert_clean_point_mass_recipe(original_payload, audit_entry)
        converted_relpath = f"results/ae15851/converted/{_original_path_slug(original_relpath)}.graph.json"
        converted_path = REPO_ROOT / converted_relpath
        converted_path.write_text(_compact_json_dumps(converted), encoding="utf-8")
        manifest_entry.update(
            {
                "disposition": "converted",
                "converted_path": converted_relpath,
                "converted_sha256": _sha256(converted_path),
                "exclusion_reason": None,
            }
        )
        entries.append(manifest_entry)

    conversion_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(AUDIT_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "audit_manifest_sha256": _sha256(AUDIT_MANIFEST_PATH),
        "expected_count": manifest["expected_count"],
        "audited_count": manifest["audited_count"],
        "converted_count": sum(1 for entry in entries if entry["disposition"] == "converted"),
        "excluded_count": sum(
            1 for entry in entries if entry["disposition"] == "excluded_known_wrong"
        ),
        "conversion_policy": {
            "known_wrong": "excluded_with_provenance",
            "dimension_source": "issue ae15851 archived point-mass task contract",
            "input_size": POINT_MASS_INPUT_SIZE,
            "external_input_size": POINT_MASS_EXTERNAL_INPUT_SIZE,
            "feedback_size": POINT_MASS_FEEDBACK_SIZE,
            "motor_noise_mapping": (
                "multiplicative_plus_constant -> signal_dependent_plus_additive; "
                "signal_dependent_noise_std=noise_std; "
                "additive_noise_std=constant_noise_scale*noise_std"
            ),
        },
        "entries": entries,
    }
    CONVERSION_MANIFEST_PATH.write_text(_pretty_json_dumps(conversion_manifest), encoding="utf-8")


def convert_clean_point_mass_recipe(
    original_payload: dict[str, Any],
    audit_entry: dict[str, Any],
) -> dict[str, Any]:
    """Return a standard-vocabulary GraphSpec payload for one clean archive row."""

    _assert_clean_point_mass_entry(audit_entry)
    nodes = dict(original_payload["nodes"])
    original_net = nodes["net"]
    net_params = dict(original_net["params"])
    controller_kind = str(audit_entry["controller_kind"])
    hidden_type_name = _hidden_type_name(controller_kind)
    population_params = _population_params(net_params)
    hidden_size = int(net_params["hidden_size"])
    out_size = int(net_params["out_size"])

    converted_nodes: dict[str, dict[str, Any]] = {}
    for node_id, node in nodes.items():
        node_type = node["type"]
        params = dict(node.get("params", {}))
        if node_id == "net":
            converted_nodes[node_id] = _component_payload(
                "Subgraph",
                {
                    "controller_kind": controller_kind,
                    "input_size": POINT_MASS_INPUT_SIZE,
                    "external_input_size": POINT_MASS_EXTERNAL_INPUT_SIZE,
                    "feedback_size": POINT_MASS_FEEDBACK_SIZE,
                    "hidden_size": hidden_size,
                    "out_size": out_size,
                    "hidden_type": hidden_type_name,
                    "sisu_gating": str(net_params.get("sisu_gating", "additive")),
                    "population_structure": population_params,
                },
                input_ports=node["input_ports"],
                output_ports=node["output_ports"],
            )
        elif node_type in {"RLRMPFeedbackChannels", "rlrmp.RLRMPFeedbackChannels"}:
            converted_nodes[node_id] = _component_payload(
                "FeedbackChannels",
                _feedback_channel_params(params),
                input_ports=node["input_ports"],
                output_ports=node["output_ports"],
            )
        elif node_id == "efferent" and node_type == "Channel":
            converted_nodes[node_id] = _component_payload(
                "Channel",
                _motor_channel_params(params),
                input_ports=node["input_ports"],
                output_ports=node["output_ports"],
            )
        elif node_type in {"RLRMPPointMass", "PointMass"}:
            converted_nodes[node_id] = _component_payload(
                "PointMass",
                params,
                input_ports=node["input_ports"],
                output_ports=node["output_ports"],
            )
        else:
            converted_nodes[node_id] = _component_payload(
                node_type,
                params,
                input_ports=node["input_ports"],
                output_ports=node["output_ports"],
            )

    subgraph = _recurrent_subgraph_payload(
        controller_kind=controller_kind,
        hidden_type_name=hidden_type_name,
        hidden_size=hidden_size,
        out_size=out_size,
        population_params=population_params,
    )
    payload = {
        "metadata": {
            "name": "Converted archived RLRMP point-mass graph recipe",
            "description": (
                "Standard-vocabulary conversion of an audited archived RLRMP "
                "point-mass graph sidecar."
            ),
            "created_at": "1970-01-01T00:00:00",
            "updated_at": "1970-01-01T00:00:00",
            "version": FEEDBAX_GRAPH_VERSION,
            "tags": ["rlrmp", "feedbax", "archive-conversion", "ae15851"],
        },
        "nodes": converted_nodes,
        "wires": original_payload["wires"],
        "input_ports": original_payload["input_ports"],
        "output_ports": original_payload["output_ports"],
        "input_bindings": original_payload["input_bindings"],
        "output_bindings": original_payload["output_bindings"],
        "retained_observables": original_payload.get("retained_observables", []),
        "subgraphs": {"net": subgraph},
    }
    GraphSpec.model_validate(payload)
    _assert_no_retired_types(payload)
    return payload


def _base_manifest_entry(audit_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_path": audit_entry["path"],
        "original_sha256": audit_entry["sha256"],
        "classification": audit_entry["classification"],
        "classification_reason": audit_entry["classification_reason"],
        "controller_kind": audit_entry["controller_kind"],
        "structural_family_actual": audit_entry["structural_family_actual"],
        "structural_subfamily_actual": audit_entry["structural_subfamily_actual"],
        "expected_conversion_family": audit_entry["expected_conversion_family"],
        "conversion_candidate_key": audit_entry["conversion_candidate_key"],
    }


def _assert_clean_point_mass_entry(audit_entry: dict[str, Any]) -> None:
    if audit_entry["classification"] != "clean":
        raise ValueError(f"{audit_entry['path']}: only clean entries can be converted")
    if audit_entry["metadata_version"] != ARCHIVE_GRAPH_VERSION:
        raise ValueError(f"{audit_entry['path']}: unexpected metadata version")
    if audit_entry["structural_family_actual"] != "point_mass":
        raise ValueError(f"{audit_entry['path']}: expected point-mass structural family")
    if audit_entry["controller_kind"] not in {"gru", "vanilla_rnn"}:
        raise ValueError(f"{audit_entry['path']}: unsupported controller kind")


def _component_payload(
    type_id: str,
    params: dict[str, Any],
    *,
    input_ports: list[str],
    output_ports: list[str],
) -> dict[str, Any]:
    return ComponentSpec(
        type=type_id,
        params=params,
        input_ports=input_ports,
        output_ports=output_ports,
    ).model_dump(mode="json", exclude_none=True)


def _feedback_channel_params(params: dict[str, Any]) -> dict[str, Any]:
    paths = list(params.get("where") or ["plant.skeleton.pos", "plant.skeleton.vel"])
    noise_std = float(params.get("noise_std", 0.0) or 0.0)
    return {
        "delay": int(params.get("delay", 0)),
        "selector": "paths",
        "paths": paths,
        "noise_model": "additive_gaussian",
        "noise_std": noise_std,
        "add_noise": noise_std != 0.0,
        "noise_role": "sensory_feedback",
        "noise_timing": "pre_controller",
        "input_shape": [[2], [2]],
    }


def _motor_channel_params(params: dict[str, Any]) -> dict[str, Any]:
    model = str(params.get("noise_model", "additive_gaussian"))
    noise_std = float(params.get("noise_std", 0.0) or 0.0)
    if model == "multiplicative_plus_constant":
        constant_scale = float(params.get("constant_noise_scale", 0.0) or 0.0)
        return {
            "delay": int(params.get("delay", 0)),
            "signal_dependent_noise_std": noise_std,
            "additive_noise_std": constant_scale * noise_std,
            "add_noise": bool(params.get("add_noise", True)),
            "noise_model": "signal_dependent_plus_additive",
            "noise_role": "motor_command",
            "noise_timing": "pre_force_filter",
            "input_shape": list(params.get("input_shape", [2])),
        }
    return {
        **params,
        "noise_role": params.get("noise_role", "motor_command"),
        "noise_timing": params.get("noise_timing", "pre_force_filter"),
    }


def _recurrent_subgraph_payload(
    *,
    controller_kind: str,
    hidden_type_name: str,
    hidden_size: int,
    out_size: int,
    population_params: dict[str, int],
) -> dict[str, Any]:
    population_structure = population_structure_from_spec(hidden_size, population_params)
    cell_type = "VanillaRNN" if controller_kind == "vanilla_rnn" else "GRU"
    graph = recurrent_controller_template_graph(
        input_size=POINT_MASS_INPUT_SIZE,
        hidden_size=hidden_size,
        out_size=out_size,
        cell_type=cell_type,
        out_nonlinearity="identity",
        population_structure=population_structure,
        name=f"Converted archived RLRMP {hidden_type_name} controller",
        description="Explicit Feedbax recurrent controller subgraph for ae15851 fixtures.",
    )
    graph = graph.model_copy(
        update={
            "metadata": GraphMetadata(
                name=f"Converted archived RLRMP {hidden_type_name} controller",
                description="Explicit Feedbax recurrent controller subgraph for ae15851 fixtures.",
                created_at="1970-01-01T00:00:00",
                updated_at="1970-01-01T00:00:00",
                version=FEEDBAX_GRAPH_VERSION,
                tags=["rlrmp", "feedbax", controller_kind],
            )
        }
    )
    return graph.model_dump(mode="json", exclude_none=True)


def _population_params(net_params: dict[str, Any]) -> dict[str, int]:
    raw = dict(net_params.get("population_structure") or {})
    return {
        "hidden_size": int(net_params["hidden_size"]),
        "n_input_only": int(raw.get("n_input_only", 0) or 0),
        "n_readout_only": int(raw.get("n_readout_only", 0) or 0),
        "n_recurrent_only": int(raw.get("n_recurrent_only", 0) or 0),
        "n_input_readout": int(raw.get("n_input_readout", 0) or 0),
    }


def _hidden_type_name(controller_kind: str) -> str:
    if controller_kind == "vanilla_rnn":
        return "VanillaRNN"
    if controller_kind == "gru":
        return "GRU"
    raise ValueError(f"unsupported controller kind {controller_kind!r}")


def _assert_no_retired_types(payload: Any) -> None:
    retired: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            type_value = value.get("type")
            if type_value in RETIRED_COMPONENT_TYPES:
                retired.append(str(type_value))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    if retired:
        raise ValueError(f"converted payload still contains retired component types: {retired}")


def _original_path_slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _pretty_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _compact_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


if __name__ == "__main__":
    main()
