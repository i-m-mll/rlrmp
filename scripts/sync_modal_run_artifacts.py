"""Pull Modal Volume artifacts for nominal GRU runs into the repo layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlrmp.modal_artifact_sync import ModalArtifactSyncError, sync_modal_run_artifacts
from rlrmp.modal_runner import MODAL_VOLUME_NAME, shell_join
from rlrmp.paths import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sync tracked run specs/GraphSpec sidecars and bulk Modal artifacts "
            "from a Modal Volume into results/ and _artifacts/."
        )
    )
    parser.add_argument("--issue", required=True, help="7-character issue/experiment id.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run id to sync. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--volume-name",
        default=MODAL_VOLUME_NAME,
        help=f"Modal Volume name. Defaults to {MODAL_VOLUME_NAME}.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root receiving results/ and _artifacts/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned modal volume get commands without executing or validating.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = sync_modal_run_artifacts(
            issue=args.issue,
            runs=args.run,
            repo_root=args.repo_root,
            volume_name=args.volume_name,
            dry_run=args.dry_run,
        )
    except ModalArtifactSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for result in results:
        for command in result.commands:
            if result.dry_run:
                print(shell_join(command))
        if result.validated:
            print(f"synced and validated {result.issue}/{result.run}")
        else:
            print(f"planned sync for {result.issue}/{result.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
