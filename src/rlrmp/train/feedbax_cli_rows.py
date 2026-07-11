"""Canonical row-manifest builder for Feedbax CLI training launches."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from rlrmp.runtime.training_run_specs import feedbax_training_run_spec_from_payload


def build_feedbax_cli_rows_manifest(
    source_path: Path,
    output_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Materialize nested specs and supported Feedbax CLI commands for rows."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != 1 or not isinstance(source.get("rows"), list):
        raise ValueError("row source must have schema_version=1 and a rows list")

    spec_output_dir = output_path.parent / "feedbax_training_run_specs"
    spec_output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for raw_row in source["rows"]:
        if not isinstance(raw_row, Mapping):
            raise TypeError("each row source entry must be an object")
        row_id = str(raw_row["id"])
        workdir = str(raw_row.get("workdir", "/workspace/rlrmp"))
        outer_spec_rel = Path(str(raw_row["run_spec"]))
        outer_spec = json.loads((repo_root / outer_spec_rel).read_text(encoding="utf-8"))
        training_spec = feedbax_training_run_spec_from_payload(outer_spec)

        nested_spec = spec_output_dir / f"{row_id}.json"
        nested_spec.write_text(
            json.dumps(
                training_spec.model_dump(mode="json", exclude_none=True),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        nested_spec_rel = nested_spec.relative_to(repo_root)
        checkpoint_root = _checkpoint_root(training_spec)
        run_id = _native_run_id(
            row_id=row_id,
            workdir=workdir,
            artifact_root=training_spec.artifacts.artifact_root,
        )
        command_parts = [
            "PYTHONPATH=src",
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "feedbax",
            "execute-training-run-spec",
            str(nested_spec_rel),
            "--run-id",
            run_id,
            "--checkpoint-root",
            checkpoint_root,
            "--resume",
        ]
        rows.append(
            {
                "id": row_id,
                "workdir": workdir,
                "command": " ".join(shlex.quote(part) for part in command_parts),
            }
        )

    manifest = {"schema_version": 1, "rows": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _checkpoint_root(training_spec: Any) -> str:
    metadata = training_spec.checkpoint_progress.metadata
    value = metadata.get("checkpoint_dir")
    if not isinstance(value, str) or not value:
        return str(PurePosixPath(training_spec.artifacts.artifact_root) / "checkpoints")
    return value


def _native_run_id(*, row_id: str, workdir: str, artifact_root: str) -> str:
    output_path = str(PurePosixPath(workdir) / artifact_root)
    output_hash = hashlib.sha256(output_path.encode()).hexdigest()[:8]
    return f"{row_id}-{output_hash}"
