"""Materialize the d6d25d6 phase-modulated recurrent bridge rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.output_feedback_phase_modulated_recurrent import (
    ARTIFACT_PATH,
    MANIFEST_PATH,
    NOTE_PATH,
    materialize,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    """Parse materializer arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note-output", type=Path, default=NOTE_PATH)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--artifact-output", type=Path, default=ARTIFACT_PATH)
    parser.add_argument(
        "--skip-reward",
        action="store_true",
        help="Materialize exact-oracle, projected-oracle, and supervised rows without reward rows.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the materializer and write outputs."""

    args = parse_args()
    summary, arrays = materialize(include_reward=not args.skip_reward)
    write_outputs(
        summary,
        arrays,
        note_path=args.note_output,
        manifest_path=args.manifest_output,
        artifact_path=args.artifact_output,
    )
    print(f"Wrote {args.note_output}")
    print(f"Wrote {args.manifest_output}")
    print(f"Wrote {args.artifact_output}")


if __name__ == "__main__":
    main()
