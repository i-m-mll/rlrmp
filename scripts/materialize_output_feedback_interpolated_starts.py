"""Write the 7cea1b7 output-feedback interpolated-start artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.output_feedback_interpolated_starts import (
    DEFAULT_SOURCE_ARTIFACT,
    ISSUE_ID,
    write_outputs,
)
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
        "--source-artifact",
        type=Path,
        default=DEFAULT_SOURCE_ARTIFACT,
        help=(
            "Rollout-recovery NPZ containing strong_optimizer_whitened__scratch_K "
            "and lqr_reference_K."
        ),
    )
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
        source_artifact=args.source_artifact,
        discretization=args.discretization,
        lane=args.lane,
    )
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['tracked_manifest']}")
    print(f"Wrote {manifest['artifact_npz']}")


if __name__ == "__main__":
    main()
