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
MAX_TRACKED_JSON_BYTES = tracked_materialization_gate.MAX_TRACKED_JSON_BYTES


pytestmark = pytest.mark.feedbax_contract
ALLOWLIST = REPO_ROOT / "ci" / "tracked-materialization-allowlist.toml"


def test_materialization_inventory_is_exactly_grandfathered() -> None:
    allowed = {
        entry["path"]
        for entry in tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))["violations"]
    }
    found = scan_tracked()
    assert "results/3cd018b/runs/const1000.json" in found
    assert not any(path.startswith("results/ef9c882/") for path in found)
    assert not any(
        entry["owner"] == "dd7234e"
        for entry in tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))["violations"]
    )
    assert allowed == set(found), (
        "Tracked materialization allowlist drifted. Remove stale entries when files are "
        "remediated, and add no new entry without an owning migration issue or keep rationale. "
        f"Only detected: {sorted(set(found) - allowed)}; only allowed: "
        f"{sorted(allowed - set(found))}."
    )


def test_materialization_gate_negative_canary_rejects_semantic_patterns() -> None:
    expanded = {
        "wrapper": {
            "rlrmp_run_spec": {
                "schema_id": "rlrmp.spec.training_run",
                "payload": "x" * 20_000,
            },
            "nested": {"graph": {"inline": {"payload": "x" * 20_000}}},
        }
    }
    assert materialization_reasons(expanded) == [
        "expanded_run_envelope",
        "expanded_inline_envelope",
    ]


def test_materialization_gate_recurses_into_inline_envelopes() -> None:
    payload = {
        "outer": [{"inner": {"graph": {"inline": {"payload": "x" * 20_000}}}}]
    }
    assert materialization_reasons(payload) == ["expanded_inline_envelope"]


def test_materialization_gate_uses_bytes_not_newlines() -> None:
    payload = ["single-line payload"]
    assert materialization_reasons(payload, byte_size=MAX_TRACKED_JSON_BYTES + 1) == [
        "oversized_json_payload"
    ]


@pytest.mark.parametrize("path", ["/Users/example/run", "/tmp/run", r"C:\\runs\\example"])
def test_materialization_gate_rejects_absolute_spec_override_values(path: str) -> None:
    payload = {
        "schema_id": "feedbax.spec.training_run_matrix",
        "rows": [{"overrides": [{"path": "checkpoint_root", "value": path}]}],
    }
    assert materialization_reasons(payload) == ["absolute_filesystem_path_in_spec"]


def test_materialization_gate_accepts_governed_worker_execution_json_pointers() -> None:
    payload = {
        "schema_id": "feedbax.spec.training_run",
        "worker_execution": {
            "effective_phase": {
                "consistency_predicate": {
                    "rules": [{"path": "/phase_program/phases/0"}],
                },
            },
            "method_contract": {
                "objective_reducers": [
                    {"path": "/objective/payload/loss_summary"},
                ],
                "worker_reducers": [{"path": "/risk_aggregation/replicate"}],
            },
        },
    }

    assert materialization_reasons(payload) == []


def test_materialization_gate_rejects_posix_path_outside_pointer_context() -> None:
    payload = {
        "schema_id": "feedbax.spec.training_run",
        "worker_execution": {
            "metadata": {"path": "/tmp/not-a-json-pointer-field"},
        },
    }

    assert materialization_reasons(payload) == ["absolute_filesystem_path_in_spec"]


def test_materialization_scan_covers_nested_tracked_json_and_skips_invalid_json(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "results" / "abcdef0" / "runs" / "nested" / "run.json"
    nested.parent.mkdir(parents=True)
    nested.write_text('{"schema_id":"feedbax.spec.run","root":"/tmp/run"}')
    invalid = nested.with_name("invalid.json")
    invalid.write_text("not-json")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "results"], cwd=tmp_path, check=True)

    assert scan_tracked(tmp_path) == {
        "results/abcdef0/runs/nested/run.json": ["absolute_filesystem_path_in_spec"]
    }


def test_materialization_allowlist_has_owner_and_rationale() -> None:
    entries = tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))["violations"]
    assert len({entry["path"] for entry in entries}) == len(entries)
    retained_roles = {
        "retained analysis/evidence document",
        "retained governed analysis data product",
        "retained historical analysis/evidence document",
        "retained legacy authored/execution run document",
    }
    for entry in entries:
        assert re.fullmatch(r"[0-9a-f]{7}", entry["owner"])
        assert entry["reason"].strip()
        if entry["owner"] not in {"dd7234e", "ee7a6f4"}:
            assert entry["reason"].startswith("Keep rationale:")
            assert any(role in entry["reason"] for role in retained_roles)
            assert "no current sibling migration owns removal" in entry["reason"]


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
