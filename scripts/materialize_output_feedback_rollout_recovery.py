"""Write the 7a459bb output-feedback rollout-recovery artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from feedbax.analysis.specs import execute_analysis_run_spec
from rlrmp.analysis.output_feedback_rollout_recovery import ISSUE_ID
from rlrmp.analysis.declarative_materialization import (
    output_feedback_rollout_recovery_spec,
    register_certificate_analysis_recipes,
)
from rlrmp.analysis.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    DISCRETIZATION_CHOICES,
    LANE_CHOICES,
    metadata_cli_help,
)


MATERIALIZER_ISSUE_ID = "c4416c5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-id", default=ISSUE_ID)
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
    parser.add_argument(
        "--note-output",
        type=Path,
        default=None,
        help="Tracked markdown note path. Defaults to results/7a459bb/notes/...",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Tracked JSON manifest path. Defaults to results/7a459bb/notes/...",
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=None,
        help="Bulk NPZ artifact path. Defaults to _artifacts/7a459bb/...",
    )
    parser.add_argument(
        "--feedbax-runs-root",
        type=Path,
        default=None,
        help="Feedbax manifest/artifact root. Defaults to Feedbax's manifest root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issue_id = args.issue_id
    note_output = args.note_output or (
        Path("results") / issue_id / "notes" / "output_feedback_rollout_recovery.md"
    )
    manifest_output = args.manifest_output or (
        Path("results")
        / issue_id
        / "notes"
        / "output_feedback_rollout_recovery_manifest.json"
    )
    artifact_output = args.artifact_output or (
        Path("_artifacts")
        / issue_id
        / "output_feedback_rollout_recovery"
        / "output_feedback_rollout_recovery.npz"
    )
    register_certificate_analysis_recipes(replace=True)
    spec = output_feedback_rollout_recovery_spec(
        issue_id=issue_id,
        discretization=args.discretization,
        lane=args.lane,
        note_output=note_output,
        manifest_output=manifest_output,
        artifact_output=artifact_output,
    )
    manifest, feedbax_manifest_path = execute_analysis_run_spec(
        spec,
        root=args.feedbax_runs_root,
        issues=[MATERIALIZER_ISSUE_ID, issue_id],
    )
    print(f"Wrote {note_output}")
    print(f"Wrote {manifest_output}")
    print(f"Wrote {artifact_output}")
    print(f"Wrote {feedbax_manifest_path}")
    print(f"Feedbax analysis manifest: {manifest.id}")


if __name__ == "__main__":
    main()
