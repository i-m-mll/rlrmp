"""Write the Phase 1 released stochastic C&S evaluation artifacts."""

from __future__ import annotations

import argparse

from rlrmp.analysis.cs_stochastic_phase1 import DEFAULT_SEEDS, ISSUE_ID, write_outputs
from rlrmp.analysis.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DISCRETIZATION_CHOICES,
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
        "--seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_SEEDS),
        help="Monte Carlo seeds to materialize; defaults to 12 deterministic seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_outputs(
        issue_id=ISSUE_ID,
        discretization=args.discretization,
        seeds=tuple(args.seeds),
    )
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['tracked_manifest']}")
    print(f"Wrote {manifest['artifact_npz']}")


if __name__ == "__main__":
    main()
