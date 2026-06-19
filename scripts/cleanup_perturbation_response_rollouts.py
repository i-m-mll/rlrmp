#!/usr/bin/env python
"""Dry-run or apply cleanup for raw perturbation-response rollout arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.rollout_cleanup import (
    CleanupPreconditionError,
    cleanup_raw_perturbation_rollouts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete raw .npz rollout arrays from one perturbation-response bulk "
            "directory after tracked summaries and regeneration specs are present. "
            "Dry-run is the default; pass --apply to delete files."
        )
    )
    parser.add_argument("--manifest", required=True, help="Tracked response manifest JSON.")
    parser.add_argument("--bulk-dir", default=None, help="Bulk perturbation-response directory.")
    parser.add_argument(
        "--summary",
        action="append",
        default=None,
        help="Required tracked summary/note path. May be repeated.",
    )
    parser.add_argument(
        "--regeneration-spec",
        default=None,
        help="Required regeneration spec path. Defaults to manifest metadata.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help=(
            "Cleanup manifest destination. Apply mode defaults to "
            "results/<issue>/notes/raw_rollout_cleanup_<bulk>.json."
        ),
    )
    parser.add_argument("--repo-root", default=".", help="Repository root (default: cwd).")
    parser.add_argument("--apply", action="store_true", help="Delete the raw rollout arrays.")
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Allow replacing an existing cleanup manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    try:
        manifest = cleanup_raw_perturbation_rollouts(
            tracked_manifest_path=Path(args.manifest),
            repo_root=repo_root,
            bulk_dir=None if args.bulk_dir is None else Path(args.bulk_dir),
            tracked_summary_paths=None
            if args.summary is None
            else [Path(path) for path in args.summary],
            regeneration_spec_path=None
            if args.regeneration_spec is None
            else Path(args.regeneration_spec),
            manifest_out_path=None if args.manifest_out is None else Path(args.manifest_out),
            apply=args.apply,
            overwrite_manifest=args.overwrite_manifest,
        )
    except CleanupPreconditionError as exc:
        raise SystemExit(f"error: {exc}") from exc

    verb = "deleted" if args.apply else "would delete"
    print(
        f"{manifest['mode']}: {verb} {manifest['candidate_file_count']} "
        f"raw rollout file(s), {manifest['candidate_bytes']} byte(s)"
    )
    if args.apply:
        print(f"deleted bytes: {manifest['deleted_bytes']}")
    if "cleanup_manifest_path" in manifest:
        print(f"cleanup manifest: {manifest['cleanup_manifest_path']}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
