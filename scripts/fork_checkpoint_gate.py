"""Fork training checkpoints and fail closed on manifest digest drift."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rlrmp.runtime.checkpoint_fork_gate import (
    ForkParityError,
    format_ratio_setpoint_report,
    fork_checkpoints_with_parity,
    parse_target,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        required=True,
        type=Path,
        help="TrainingRunMatrixSpec JSON document.",
    )
    parser.add_argument(
        "--source-checkpoint-root",
        required=True,
        type=Path,
        help="Checkpoint root containing the source latest.json.",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        required=True,
        help="Target row as ROW=CHECKPOINT_ROOT; may be repeated.",
    )
    parser.add_argument(
        "--parity-output",
        required=True,
        type=Path,
        help="JSON parity table output path.",
    )
    parser.add_argument(
        "--skip-fork",
        action="store_true",
        help="Read existing target fork manifests without writing new forks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        table = fork_checkpoints_with_parity(
            matrix_path=args.matrix,
            source_checkpoint_root=args.source_checkpoint_root,
            targets=args.target,
            parity_output_path=args.parity_output,
            skip_fork=args.skip_fork,
        )
    except ForkParityError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ratio_setpoint = table.get("ratio_setpoint")
    if isinstance(ratio_setpoint, dict):
        print(format_ratio_setpoint_report(ratio_setpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
