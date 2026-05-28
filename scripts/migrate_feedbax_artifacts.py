#!/usr/bin/env python3
"""Migrate selected RLRMP Feedbax artifacts to schema-versioned array stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.artifact_migration import (
    DEFAULT_OUTPUT_ISSUE,
    discover_b_set_runs,
    migrate_legacy_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--issue",
        action="append",
        default=None,
        help="Restrict to one or more source issue IDs.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Restrict to one or more run labels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Migrate only the first N selected runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected runs without loading or writing artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root
    runs = discover_b_set_runs(repo_root)
    if args.issue:
        allowed_issues = set(args.issue)
        runs = [run for run in runs if run.issue_id in allowed_issues]
    if args.run:
        allowed_runs = set(args.run)
        runs = [run for run in runs if run.run_label in allowed_runs]
    if args.limit is not None:
        runs = runs[: args.limit]

    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "issue_id": run.issue_id,
                        "run_label": run.run_label,
                        "run_spec": str(run.run_spec_path),
                        "artifact_dir": str(run.artifact_dir),
                        "model_path": str(run.model_path),
                    }
                    for run in runs
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return

    migrated = []
    for run in runs:
        result = migrate_legacy_run(run, repo_root=repo_root)
        migrated.append(
            {
                "issue_id": run.issue_id,
                "run_label": run.run_label,
                "manifest": str(result.manifest_path),
                "array_store": str(result.array_store_path),
                "array_count": result.array_count,
                "total_array_nbytes": result.total_nbytes,
                "validation_status": result.validation_status,
            }
        )
        print(
            f"{run.issue_id}/{run.run_label}: {result.validation_status}, "
            f"{result.array_count} arrays -> {result.array_store_path}"
        )

    index_path = repo_root / "results" / DEFAULT_OUTPUT_ISSUE / "migration_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "rlrmp.feedbax_artifact_migration.v1",
                "migrated": migrated,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
