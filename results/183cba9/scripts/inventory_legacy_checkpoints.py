"""Inventory legacy checkpoint trees for issue 183cba9."""

from __future__ import annotations
from rlrmp.paths import portable_repo_path as _rel

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "_artifacts"
INDEX_PATH = REPO / "results/183cba9/notes/legacy_checkpoint_manifests.md"
BASELINE_INDEX_PATH = REPO / "results/08483d5/notes/legacy_checkpoint_manifests.md"


@dataclass(frozen=True)
class Row:
    run: str
    checkpoint: str
    commit: str
    spec_path: str
    status: str


def main() -> int:
    rows = list(_inventory())
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(_render_index(rows), encoding="utf-8")
    baseline_rows = [
        row
        for row in rows
        if row.run == "08483d5/runs/h0_6d_no_pgd_const_band16_cpu"
        and row.checkpoint == "checkpoint_0012000"
    ]
    BASELINE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_INDEX_PATH.write_text(_render_baseline_index(baseline_rows), encoding="utf-8")
    return 0


def _inventory() -> list[Row]:
    rows: list[Row] = []
    for checkpoint_dir in sorted(ARTIFACTS.glob("*/runs/*/checkpoints*/checkpoint_*")):
        if not checkpoint_dir.is_dir() or checkpoint_dir.name == "checkpoint_latest":
            continue
        if not (checkpoint_dir / "model.eqx").is_file():
            continue
        rel = checkpoint_dir.relative_to(ARTIFACTS)
        run = f"{rel.parts[0]}/runs/{rel.parts[2]}"
        checkpoint = checkpoint_dir.name
        spec_path = REPO / "results" / rel.parts[0] / "runs" / f"{rel.parts[2]}.json"
        metadata_path = checkpoint_dir / "metadata.json"
        metadata = _read_json(metadata_path)
        spec = _read_json(spec_path)
        commit = _resolve_commit(metadata, spec)
        spec_text = _rel(spec_path) if spec_path.exists() else "missing"
        status_bits = []
        if commit == "missing":
            status_bits.append("missing_commit")
        else:
            status_bits.append("commit_known")
        if not spec_path.exists():
            status_bits.append("missing_spec")
        if (checkpoint_dir / "leaf_manifest.json").is_file():
            status_bits.append("leaf_manifest_present")
        rows.append(
            Row(
                run=run,
                checkpoint=checkpoint,
                commit=commit,
                spec_path=spec_text,
                status="; ".join(status_bits),
            )
        )
    return rows


def _resolve_commit(metadata: dict[str, Any], spec: dict[str, Any]) -> str:
    candidates = [
        metadata.get("run_spec", {}).get("provenance", {}).get("git", {}).get("rlrmp_commit"),
        metadata.get("run_spec", {}).get("provenance", {}).get("git", {}).get("commit"),
        metadata.get("provenance", {}).get("git", {}).get("rlrmp_commit"),
        metadata.get("provenance", {}).get("git", {}).get("commit"),
        spec.get("provenance", {}).get("git", {}).get("rlrmp_commit"),
        spec.get("provenance", {}).get("git", {}).get("commit"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return "missing"


def _render_index(rows: list[Row]) -> str:
    lines = [
        "# Legacy Checkpoint LeafManifest Inventory",
        "",
        "Issue 183cba9 inventory of `_artifacts/*/runs/*/checkpoints*` checkpoint trees.",
        "Rows marked `missing_commit` require adjudication before a LeafManifest can be dumped.",
        "",
        f"Total checkpoint rows: {len(rows)}",
        "",
        "| run | checkpoint | commit | spec path | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.run}` | `{row.checkpoint}` | `{row.commit}` | "
            f"`{row.spec_path}` | {row.status} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_baseline_index(rows: list[Row]) -> str:
    lines = [
        "# 08483d5 Legacy Baseline LeafManifest",
        "",
        "Baseline target for issue 3cd018b adoption.",
        "",
        "| run | checkpoint | commit | spec path | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.run}` | `{row.checkpoint}` | `{row.commit}` | "
            f"`{row.spec_path}` | {row.status} |"
        )
    if not rows:
        lines.append("| `08483d5/runs/h0_6d_no_pgd_const_band16_cpu` | `checkpoint_0012000` | `missing` | `missing` | missing |")
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}




if __name__ == "__main__":
    raise SystemExit(main())
