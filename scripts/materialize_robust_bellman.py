"""Write the 583d764 robust Bellman diagnostic artifacts."""

from __future__ import annotations

import argparse

from rlrmp.analysis.robust_bellman import ISSUE_ID, write_outputs
from rlrmp.analysis.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    DISCRETIZATION_CHOICES,
    LANE_CHOICES,
    metadata_cli_help,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discretization",
        choices=DISCRETIZATION_CHOICES,
        default=DEFAULT_DISCRETIZATION,
        help=metadata_cli_help(),
    )
    parser.add_argument(
        "--lane",
        choices=LANE_CHOICES,
        default=DEFAULT_LANE,
        help=metadata_cli_help(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_outputs(
        issue_id=ISSUE_ID,
        discretization=args.discretization,
        lane=args.lane,
    )
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['tracked_manifest']}")


if __name__ == "__main__":
    main()
