"""Tests for diagnostic regeneration provenance helpers."""

from __future__ import annotations

import json
from pathlib import Path

from rlrmp.analysis.pipelines.diagnostic_provenance import (
    path_ref,
    repo_relative,
    sha256_file,
    write_regeneration_spec,
)


def test_path_ref_hashes_files_and_uses_repo_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path
    source = repo_root / "results" / "abc1234" / "notes" / "manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"ok": true}\n', encoding="utf-8")

    ref = path_ref(source, role="manifest", repo_root=repo_root)

    assert ref["path"] == "results/abc1234/notes/manifest.json"
    assert ref["role"] == "manifest"
    assert ref["kind"] == "file"
    assert ref["exists"] is True
    assert ref["sha256"] == sha256_file(source)
    assert repo_relative(source, repo_root=repo_root) == ref["path"]


def test_write_regeneration_spec_records_inputs_outputs_and_source_files(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    input_path = repo_root / "results" / "abc1234" / "runs" / "run_a" / "run.json"
    output_path = repo_root / "results" / "abc1234" / "notes" / "diag.json"
    source_path = repo_root / "src" / "rlrmp" / "analysis" / "diag.py"
    input_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    source_path.parent.mkdir(parents=True)
    input_path.write_text('{"run": "a"}\n', encoding="utf-8")
    output_path.write_text('{"diag": true}\n', encoding="utf-8")
    source_path.write_text("# source\n", encoding="utf-8")

    spec = write_regeneration_spec(
        spec_path=repo_root / "results" / "abc1234" / "notes" / "diag_regeneration_spec.json",
        diagnostic_name="unit_test_diagnostic",
        materializer="tests.fake.materializer",
        command=["uv", "run", "python", "scripts/materialize_fake.py"],
        parameters={"run_ids": ("run_a",)},
        inputs=[{"role": "run_spec", "path": input_path}],
        outputs=[{"role": "diagnostic_manifest", "path": output_path}],
        source_files=[source_path],
        notes=["test note"],
        repo_root=repo_root,
    )

    on_disk = json.loads(
        (repo_root / "results" / "abc1234" / "notes" / "diag_regeneration_spec.json")
        .read_text(encoding="utf-8")
    )
    assert on_disk == spec
    assert spec["schema_version"] == "rlrmp.diagnostic_regeneration_spec.v1"
    assert spec["parameters"]["run_ids"] == ["run_a"]
    assert spec["inputs"][0]["path"] == "results/abc1234/runs/run_a/run.json"
    assert spec["outputs"][0]["path"] == "results/abc1234/notes/diag.json"
    assert spec["source_files"][0]["path"] == "src/rlrmp/analysis/diag.py"
    assert spec["future_graphspec"]["status"] == "temporary_rlrmp_bridge"
