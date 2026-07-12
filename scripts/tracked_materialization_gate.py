#!/usr/bin/env python3
"""Find tracked expanded/resolved experiment-spec materializations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_EXPANDED_NODES = 100


def _node_count(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(_node_count(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_node_count(item) for item in value)
    return 1


def materialization_reasons(payload: Any) -> list[str]:
    """Return semantic reasons that ``payload`` is an expanded spec."""
    if not isinstance(payload, dict):
        return []
    reasons: list[str] = []
    base = payload.get("base")
    if isinstance(base, dict) and "inline" in base and _node_count(base["inline"]) >= MIN_EXPANDED_NODES:
        reasons.append("large_run_matrix_base_inline")

    schema = str(payload.get("schema_id", "")) + " " + str(payload.get("schema_version", ""))
    graph = payload.get("graph")
    if (
        "training_run" in schema
        and isinstance(graph, dict)
        and "inline" in graph
        and _node_count(graph["inline"]) >= MIN_EXPANDED_NODES
    ):
        reasons.append("resolved_training_run_graph_inline")
    return reasons


def scan_tracked(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    listed = subprocess.run(
        ["git", "ls-files", "results/**/*.json"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    found: dict[str, list[str]] = {}
    for relpath in listed:
        try:
            payload = json.loads((repo_root / relpath).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        reasons = materialization_reasons(payload)
        if reasons:
            found[relpath] = reasons
    return found
