"""Contract tests for tracked materializations and experiment-cost recording."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_KPI_SPEC = importlib.util.spec_from_file_location(
    "experiment_kpi", REPO_ROOT / "scripts" / "experiment_kpi.py"
)
assert EXPERIMENT_KPI_SPEC is not None
assert EXPERIMENT_KPI_SPEC.loader is not None
experiment_kpi = importlib.util.module_from_spec(EXPERIMENT_KPI_SPEC)
sys.modules[EXPERIMENT_KPI_SPEC.name] = experiment_kpi
EXPERIMENT_KPI_SPEC.loader.exec_module(experiment_kpi)

MATERIALIZATION_GATE_SPEC = importlib.util.spec_from_file_location(
    "tracked_materialization_gate", REPO_ROOT / "scripts" / "tracked_materialization_gate.py"
)
assert MATERIALIZATION_GATE_SPEC is not None
assert MATERIALIZATION_GATE_SPEC.loader is not None
tracked_materialization_gate = importlib.util.module_from_spec(MATERIALIZATION_GATE_SPEC)
sys.modules[MATERIALIZATION_GATE_SPEC.name] = tracked_materialization_gate
MATERIALIZATION_GATE_SPEC.loader.exec_module(tracked_materialization_gate)

measure = experiment_kpi.measure
materialization_reasons = tracked_materialization_gate.materialization_reasons
scan_tracked = tracked_materialization_gate.scan_tracked


pytestmark = pytest.mark.feedbax_contract
ALLOWLIST = REPO_ROOT / "ci" / "tracked-materialization-allowlist.toml"


def test_tracked_materializations_are_explicitly_grandfathered() -> None:
    allowed = {
        entry["path"]
        for entry in tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))["violations"]
    }
    found = scan_tracked()
    assert "results/c6c5997/runs/matrix.json" not in found
    assert "results/ef9c882/runs/matrix.json" not in found
    assert (
        "results/3cd018b/runs/ramp3500_to1000/feedbax_training_run_spec.json"
        not in found
    )
    assert allowed == set()
    assert not set(found) - allowed, (
        "New tracked expanded/resolved spec materializations: "
        f"{sorted(set(found) - allowed)}. Store a compact authored ref/spec instead; "
        "grandfather only historical violations with an owning migration issue."
    )


def test_materialization_gate_negative_canary_rejects_semantic_patterns() -> None:
    expanded = {"schema_id": "feedbax.spec.training_run_matrix", "base": {"inline": {}}}
    expanded["base"]["inline"] = {f"field_{i}": i for i in range(100)}
    assert materialization_reasons(expanded) == ["large_run_matrix_base_inline"]

    resolved = {
        "schema_id": "feedbax.spec.training_run",
        "graph": {"inline": {f"node_{i}": {"type": "Node"} for i in range(50)}},
    }
    assert materialization_reasons(resolved) == ["resolved_training_run_graph_inline"]


def test_materialization_allowlist_has_owner_and_rationale() -> None:
    entries = tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))["violations"]
    for entry in entries:
        assert re.fullmatch(r"[0-9a-f]{7}", entry["owner"])
        assert entry["reason"].strip()


def test_experiment_kpi_is_revision_pinned_and_deterministic(tmp_path: Path) -> None:
    spec_path = "results/abcdef0/runs/matrix.json"
    script_path = "results/abcdef0/scripts/author.py"
    (tmp_path / spec_path).parent.mkdir(parents=True)
    (tmp_path / script_path).parent.mkdir(parents=True)
    (tmp_path / spec_path).write_text('{\n  "base": {"ref": "x"},\n  "rows": []\n}\n')
    (tmp_path / script_path).write_text("first = 1\nsecond = 2\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    manifest = {
        "experiment_issue": "abcdef0",
        "authored_production_paths": [spec_path, script_path],
        "authored_spec_paths": [spec_path],
        "generated_materialization_paths": [],
        "c1_extra_concepts": ["scientific_axis"],
        "concepts": {
            "c2_new_registry_entries": 1,
            "c3_authored_callbacks": 0,
            "c4_escape_hatch_invocations": 0,
            "c5_non_boilerplate_control_flow": 0,
        },
    }
    report = measure(manifest, "HEAD", tmp_path)
    assert report["authored_production_loc"] == 6
    assert report["authored_spec_loc"] == 4
    assert report["generated_materialization_loc"] == 0
    assert report["c1_distinct_authored_keys"] == 4
    assert report["c2_new_registry_entries"] == 1


def test_experiment_kpi_rejects_spec_outside_authored_inventory() -> None:
    manifest = {
        "experiment_issue": "abcdef0",
        "authored_production_paths": [],
        "authored_spec_paths": ["results/abcdef0/runs/matrix.json"],
        "concepts": {},
    }
    with pytest.raises(ValueError, match="subset"):
        measure(manifest, "HEAD")
