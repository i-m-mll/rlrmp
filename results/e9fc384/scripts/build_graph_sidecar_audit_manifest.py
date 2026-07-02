"""Build the graph-sidecar fidelity audit manifest for issue e9fc384.

This is a one-time audit script tied to `results/e9fc384/`. It inspects every
tracked archived Feedbax graph sidecar under `results/` (the
`model.graph.json` / `*.graph.json` pattern that issue ae15851 will later
convert into loadability fixtures) and records a structural classification —
clean vs. known-wrong — keyed by inspected graph structure, not by production
date. See `results/e9fc384/notes/graph_sidecar_audit.md` for the narrative and
`results/e9fc384/notes/graph_sidecar_audit_manifest.json` for the committed
machine-readable output this script produces.

Re-run with:
    PYTHONPATH="<worktree>/src" uv run --no-sync python \
        results/e9fc384/scripts/build_graph_sidecar_audit_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "results" / "e9fc384" / "notes" / "graph_sidecar_audit_manifest.json"

SIDECAR_PATTERNS = ("results/**/model.graph.json", "results/**/*.graph.json")
EXPECTED_COUNT = 38
MANIFEST_SCHEMA_VERSION = 1

# Structural families this audit knows how to recognize. Each entry is the
# set of node "type" values that must ALL be present for the family to match.
POINT_MASS_CORE_TYPES = frozenset(
    {"FirstOrderFilter", "PointMass", "RLRMPFeedbackChannels", "RLRMPSimpleStagedNetwork"}
)
CS_LSS_MARKER_TYPE = "LinearStateSpace"

# Explicit per-path expected-family assignments, established by human audit
# (not derived from a free glob or filename heuristic). Paths not listed here
# default to "point_mass" (the historically dominant family for this corpus).
# The two 30f2313 CS-stochastic runs are audited as CS-LSS candidates: their
# own run.json records an analytical 48-state delay-augmented C&S plant
# (game_card.plant) and an explicit fidelity gap
# (execution_backend=rlrmp.legacy_simple_feedback_compat,
# analytical_delay_augmented_state_input=false, certificate_lens=
# input_output_map_certificate, exact_fidelity=false) — i.e. the run itself
# documents that it did NOT execute the analytical LinearStateSpace plant.
EXPECTED_FAMILY_OVERRIDES: dict[str, str] = {
    "results/30f2313/runs/cs_stochastic_gru__hidden_penalty/model.graph.json": "cs_lss",
    "results/30f2313/runs/cs_stochastic_gru__no_hidden_penalty/model.graph.json": "cs_lss",
}
DEFAULT_EXPECTED_FAMILY = "point_mass"


def _git_ls_files(*patterns: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted({line for line in result.stdout.splitlines() if line})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _audit_file(relpath: str) -> dict[str, Any]:
    path = REPO_ROOT / relpath
    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)

    top_level_keys = sorted(payload.keys())
    has_schema_version = "schema_version" in payload
    metadata_version = payload.get("metadata", {}).get("version")

    nodes: dict[str, Any] = payload.get("nodes", {})
    retired_component_ids = sorted(nodes.keys())
    node_types = {node_id: node.get("type") for node_id, node in nodes.items()}
    structural_types = sorted({t for t in node_types.values() if t is not None})

    net_params = nodes.get("net", {}).get("params", {})
    controller_kind = net_params.get("controller_kind")

    predicates: dict[str, bool] = {
        "metadata_version_is_v1": metadata_version == "rlrmp.feedbax_graph.v1",
        "no_top_level_schema_version": not has_schema_version,
        "point_mass_filter_channel_family_present": POINT_MASS_CORE_TYPES.issubset(
            structural_types
        ),
        "linear_state_space_absent": CS_LSS_MARKER_TYPE not in structural_types,
        "controller_kind_recorded": controller_kind in {"gru", "vanilla_rnn"},
    }

    # Actual structural family, purely from inspected node types.
    if CS_LSS_MARKER_TYPE in structural_types:
        actual_family = "cs_lss"
    elif POINT_MASS_CORE_TYPES.issubset(structural_types):
        actual_family = "point_mass"
    else:
        actual_family = "unrecognized"

    expected_family = EXPECTED_FAMILY_OVERRIDES.get(relpath, DEFAULT_EXPECTED_FAMILY)

    if expected_family == actual_family and actual_family != "unrecognized":
        classification = "clean"
        classification_reason = (
            f"Inspected structure matches the expected '{expected_family}' family."
        )
    else:
        classification = "known_wrong"
        classification_reason = (
            f"Expected structural family '{expected_family}' but inspected structure "
            f"encodes '{actual_family}' (node types: {structural_types})."
        )

    # Recurrent-controller sub-family, informational only (does not affect
    # classification): vanilla_rnn vs gru within the point_mass family.
    if actual_family == "point_mass" and controller_kind is not None:
        structural_subfamily = f"point_mass_{controller_kind}"
    else:
        structural_subfamily = actual_family

    return {
        "path": relpath,
        "sha256": content_hash,
        "metadata_version": metadata_version,
        "top_level_keys": top_level_keys,
        "has_top_level_schema_version": has_schema_version,
        "retired_component_ids": retired_component_ids,
        "node_types": node_types,
        "structural_types": structural_types,
        "controller_kind": controller_kind,
        "structural_family_actual": actual_family,
        "structural_subfamily_actual": structural_subfamily,
        "expected_conversion_family": expected_family,
        "classification": classification,
        "classification_reason": classification_reason,
        "structural_predicates_checked": predicates,
        "conversion_candidate_key": {
            "retired_type": structural_types,
            "structural_predicate": "point_mass_filter_channel_family_present"
            if actual_family == "point_mass"
            else ("cs_lss_family_present" if actual_family == "cs_lss" else "unrecognized"),
            "metadata_version": metadata_version,
            "audited_fixture_hash": content_hash,
        },
    }


def build_manifest() -> dict[str, Any]:
    live_paths = _git_ls_files(*SIDECAR_PATTERNS)
    files = [_audit_file(relpath) for relpath in live_paths]

    clean = [f for f in files if f["classification"] == "clean"]
    known_wrong = [f for f in files if f["classification"] == "known_wrong"]

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_patterns": list(SIDECAR_PATTERNS),
        "expected_count": EXPECTED_COUNT,
        "audited_count": len(files),
        "classification_summary": {
            "clean": len(clean),
            "known_wrong": len(known_wrong),
        },
        "files": files,
    }


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"audited_count={manifest['audited_count']} expected_count={manifest['expected_count']}")
    print(f"classification_summary={manifest['classification_summary']}")


if __name__ == "__main__":
    main()
