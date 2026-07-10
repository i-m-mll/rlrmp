#!/usr/bin/env python3
"""Combine per-family data-in-code resolutions into a bijective closing manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--fragment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _starting_baseline(repo_root: Path, source_commit: str) -> tuple[list[str], str]:
    result = subprocess.run(
        ["git", "show", f"{source_commit}:ci/data_in_code_baseline.json"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    payload = result.stdout
    entries = json.loads(payload)
    if not isinstance(entries, list) or not all(isinstance(key, str) for key in entries):
        raise ValueError("source baseline must be a JSON list of string keys")
    return sorted(entries), hashlib.sha256(payload).hexdigest()


def _load_resolutions(
    paths: list[Path],
    starting: list[str],
) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        resolutions = dict(payload.get("resolutions", {}))
        groups = payload.get("resolution_groups", [])
        if not isinstance(resolutions, dict) or not isinstance(groups, list):
            raise ValueError(f"{path} must contain resolution objects or groups")
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(f"invalid resolution group in {path}: {group!r}")
            keys = group.get("keys", [])
            prefixes = group.get("key_prefixes", [])
            exclude_keys = group.get("exclude_keys", [])
            if (
                not isinstance(keys, list)
                or not isinstance(prefixes, list)
                or not isinstance(exclude_keys, list)
            ):
                raise ValueError(f"resolution group keys/prefixes must be lists in {path}")
            selected = set(keys)
            selected.update(
                key for key in starting if any(key.startswith(prefix) for prefix in prefixes)
            )
            selected.difference_update(exclude_keys)
            resolution = {
                name: value
                for name, value in group.items()
                if name not in {"exclude_keys", "keys", "key_prefixes"}
            }
            for key in selected:
                if key in resolutions:
                    raise ValueError(f"duplicate resolution key inside {path}: {key}")
                resolutions[key] = resolution
        overlap = set(combined).intersection(resolutions)
        if overlap:
            raise ValueError(f"duplicate resolution keys in {path}: {sorted(overlap)}")
        for key, resolution in resolutions.items():
            if not isinstance(key, str) or not isinstance(resolution, dict):
                raise ValueError(f"invalid resolution entry in {path}: {key!r}")
            required = {"route", "destination", "rationale"}
            missing = required.difference(resolution)
            if missing:
                raise ValueError(f"resolution {key} in {path} is missing {sorted(missing)}")
            combined[key] = resolution
    return combined


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    starting, baseline_sha256 = _starting_baseline(repo_root, args.source_commit)
    resolutions = _load_resolutions(
        [path.resolve() for path in args.fragment],
        starting,
    )
    missing = sorted(set(starting).difference(resolutions))
    unexpected = sorted(set(resolutions).difference(starting))
    if missing or unexpected:
        raise ValueError(
            f"closing manifest is not bijective: missing={missing}, unexpected={unexpected}"
        )

    output = {
        "schema_id": "rlrmp.data_in_code_closing_manifest",
        "schema_version": "rlrmp.data_in_code_closing_manifest.v1",
        "source_commit": args.source_commit,
        "source_baseline_sha256": baseline_sha256,
        "starting_entry_count": len(starting),
        "resolutions": {key: resolutions[key] for key in starting},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(starting)} resolutions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
