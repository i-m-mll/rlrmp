from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "dedupe_reconciliation.jsonl"
BINDING_PATH = (
    REPO_ROOT / "_artifacts" / "31aaa31" / "verification" / "dedupe_closure_manifest.jsonl"
)
EXPECTED_STATE_COUNTS = {
    "canonical_survivor": 4,
    "cross_repo_resolved": 3,
    "excluded_cross_repo": 2,
    "removed": 159,
    "thin_adapter": 108,
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _definition(path: Path, qualname: str) -> tuple[ast.AST | None, str]:
    if not path.exists():
        return None, ""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    name = qualname.rsplit(".", 1)[-1]
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
    ]
    assert len(nodes) <= 1, (path, qualname, len(nodes))
    return (nodes[0] if nodes else None), source


def _calls(node: ast.AST) -> set[str]:
    calls = set()
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if isinstance(call.func, ast.Name):
            calls.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            calls.add(call.func.attr)
    return calls


def _assert_member_state(cluster_id: str, member: Mapping[str, Any]) -> None:
    state = member["state"]
    repo = member["repo"]
    relative_path = str(member["module_relpath"])
    if state in {"cross_repo_resolved", "excluded_cross_repo"}:
        assert repo != "rlrmp", (cluster_id, member["id"])
        return
    if state == "excluded_pipeline_lane":
        assert repo == "rlrmp"
        assert relative_path.startswith("src/rlrmp/analysis/pipelines/")
        return

    assert repo == "rlrmp", (cluster_id, member["id"])
    node, source = _definition(REPO_ROOT / relative_path, str(member["qualname"]))
    if state == "removed":
        assert node is None, (cluster_id, member["id"], "definition reaccreted")
        return
    assert node is not None, (cluster_id, member["id"], "definition missing")

    if state == "preserved_distinct":
        segment = ast.get_source_segment(source, node) or ""
        for marker in member["required_markers"]:
            assert marker in segment, (cluster_id, member["id"], marker)
        return

    assert state in {"thin_adapter", "canonical_survivor"}, (cluster_id, member["id"], state)
    assert node.end_lineno is not None
    loc = node.end_lineno - node.lineno + 1
    assert loc <= member["max_loc"], (cluster_id, member["id"], loc, member["max_loc"])
    if state == "canonical_survivor":
        return
    calls = _calls(node)
    assert member["canonical_call"] in calls, (
        cluster_id,
        member["id"],
        member["canonical_call"],
        calls,
    )
    forbidden = set(member.get("forbidden_calls", ()))
    assert calls.isdisjoint(forbidden), (cluster_id, member["id"], calls & forbidden)


def test_all_confirm_clusters_have_enforced_final_reconciliation() -> None:
    binding = [row for row in _jsonl(BINDING_PATH) if row["verdict"] == "CONFIRM"]
    reconciliation = _jsonl(FIXTURE_PATH)
    assert len(binding) == len(reconciliation) == 42

    binding_by_id = {row["cluster_id"]: row for row in binding}
    reconciliation_by_id = {row["cluster_id"]: row for row in reconciliation}
    assert len(binding_by_id) == len(reconciliation_by_id) == 42
    assert reconciliation_by_id.keys() == binding_by_id.keys()

    state_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    for cluster_id, row in reconciliation_by_id.items():
        original = binding_by_id[cluster_id]
        disposition_counts[row["final_disposition"]] += 1
        assert row["survivor"]
        assert row["evidence"]["target_module"] == original["target_module"]
        assert row["evidence"]["expected_loc_reduction"] == original["expected_loc_reduction"]
        original_ids = [member["id"] for member in original["members"]]
        member_ids = [member["id"] for member in row["members"]]
        assert member_ids == original_ids, cluster_id
        assert len(member_ids) == len(set(member_ids)), cluster_id
        assert [
            (member["repo"], member["module_relpath"], member["qualname"])
            for member in row["members"]
        ] == [
            (member["repo"], member["module_relpath"], member["qualname"])
            for member in original["members"]
        ], cluster_id
        for member in row["members"]:
            state_counts[member["state"]] += 1
            _assert_member_state(cluster_id, member)

    assert dict(state_counts) == EXPECTED_STATE_COUNTS
    assert disposition_counts == {
        "resolved": 40,
        "resolved_destination_deviation": 1,
        "resolved_cross_repo": 1,
    }
    assert reconciliation_by_id["dup_0041"]["final_disposition"] == (
        "resolved_destination_deviation"
    )
    feedbax = reconciliation_by_id["dup_0063"]
    assert feedbax["final_disposition"] == "resolved_cross_repo"
    assert feedbax["evidence"]["feedbax_commit"] == "8dc210ad"
    assert feedbax["evidence"]["focused_tests"] == "28 passed"
