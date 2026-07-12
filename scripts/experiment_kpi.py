#!/usr/bin/env python3
"""Measure the marginal authoring cost of one experiment.

The input manifest deliberately classifies files instead of guessing whether a
file was authored.  Counts are reproducible from Git blobs, while the c2--c5
concept counts remain explicit assertions by the experiment owner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_blob(path: str, revision: str, repo_root: Path = REPO_ROOT) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _line_count(blob: bytes) -> int:
    return len(blob.splitlines())


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def measure(manifest: dict[str, Any], revision: str, repo_root: Path = REPO_ROOT) -> dict:
    authored_paths = manifest["authored_production_paths"]
    spec_paths = manifest["authored_spec_paths"]
    generated_paths = manifest.get("generated_materialization_paths", [])
    if not set(spec_paths) <= set(authored_paths):
        raise ValueError("authored_spec_paths must be a subset of authored_production_paths")

    blobs = {path: _git_blob(path, revision, repo_root) for path in set(authored_paths + generated_paths)}
    keys: set[str] = set()
    for path in spec_paths:
        try:
            keys |= _keys(json.loads(blobs[path]))
        except json.JSONDecodeError:
            # YAML/TOML keys are not safely countable without format-specific
            # semantic parsing; list them explicitly in c1_extra_concepts.
            pass
    c1_extra = set(manifest.get("c1_extra_concepts", []))
    concepts = manifest["concepts"]
    return {
        "schema_version": 1,
        "experiment_issue": manifest["experiment_issue"],
        "revision": revision,
        "authored_production_loc": sum(_line_count(blobs[path]) for path in authored_paths),
        "authored_spec_loc": sum(_line_count(blobs[path]) for path in spec_paths),
        "generated_materialization_loc": sum(
            _line_count(blobs[path]) for path in generated_paths
        ),
        "c1_distinct_authored_keys": len(keys | c1_extra),
        "c2_new_registry_entries": concepts["c2_new_registry_entries"],
        "c3_authored_callbacks": concepts["c3_authored_callbacks"],
        "c4_escape_hatch_invocations": concepts["c4_escape_hatch_invocations"],
        "c5_non_boilerplate_control_flow": concepts["c5_non_boilerplate_control_flow"],
        "paths": {
            "authored_production": authored_paths,
            "authored_specs": spec_paths,
            "generated_materializations": generated_paths,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = measure(json.loads(args.manifest.read_text()), args.revision)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
