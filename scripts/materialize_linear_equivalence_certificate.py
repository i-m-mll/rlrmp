"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Write the d01c35a linear equivalence certificate artifacts."""

from __future__ import annotations

import argparse

from rlrmp.analysis.math.linear_equivalence_certificate import ISSUE_ID, write_outputs
from rlrmp.analysis.math.rerun_metadata import (
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
    print(f"Wrote {manifest['artifact_npz']}")


if __name__ == "__main__":
    main()
