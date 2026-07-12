#!/usr/bin/env python3
"""Thin CLI for validating and executing authored training matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rlrmp.train.launch import (
    LaunchRuntimeControls,
    TransitionalFeedbaxBackend,
    launch_evidence,
    load_authored_training_intent,
    verify_resume_authored_training_intent,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the operational launch CLI (never a scientific config parser)."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("document", type=Path)
    validate.add_argument("--repo-root", type=Path, default=Path.cwd())

    execute = commands.add_parser("execute")
    execute.add_argument("document", type=Path)
    execute.add_argument("--repo-root", type=Path, default=Path.cwd())
    execute.add_argument("--row")
    execute.add_argument("--resume", action="store_true")
    execute.add_argument("--allow-fresh-start", action="store_true")
    execute.add_argument("--stop-after-batches", type=int)
    execute.add_argument("--disable-progress", action="store_true")
    execute.add_argument("--quiet-progress", action="store_true")
    execute.add_argument("--log-step", type=int, default=1)
    execute.add_argument("--manifest-root", type=Path)
    execute.add_argument("--checkpoint-root", type=Path)

    dry_run = commands.add_parser("dry-run")
    dry_run.add_argument("document", type=Path)
    dry_run.add_argument("--repo-root", type=Path, default=Path.cwd())
    dry_run.add_argument("--row")
    verify = commands.add_parser("verify-resume")
    verify.add_argument("document", type=Path)
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify.add_argument("--row")
    verify.add_argument("--checkpoint-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the authored launch command."""

    args = build_parser().parse_args(argv)
    launch = load_authored_training_intent(args.document, repo_root=args.repo_root)
    if args.command == "validate":
        print(f"valid TrainingRunMatrixSpec: {args.document}")
        return 0
    if args.command == "dry-run":
        from rlrmp.train.launch import compile_authored_training_intent, select_launch_rows

        rows = select_launch_rows(compile_authored_training_intent(launch), args.row)
        print(json.dumps([{"row_id": row.row_id, "run_id": row.planned_run_id} for row in rows]))
        return 0
    if args.command == "verify-resume":
        evidence = verify_resume_authored_training_intent(
            launch, row=args.row, checkpoint_root=args.checkpoint_root
        )
        print(json.dumps(evidence))
        return 0
    controls = LaunchRuntimeControls(
        resume=args.resume,
        allow_fresh_start=args.allow_fresh_start,
        stop_after_batches=args.stop_after_batches,
        disable_progress=args.disable_progress,
        quiet_progress=args.quiet_progress,
        log_step=args.log_step,
        manifest_root=args.manifest_root,
        checkpoint_root=args.checkpoint_root,
    )
    from rlrmp.train.launch import compile_authored_training_intent, select_launch_rows

    rows = select_launch_rows(compile_authored_training_intent(launch), args.row)
    print(json.dumps(launch_evidence(rows, controls), sort_keys=True))
    backend = TransitionalFeedbaxBackend()
    for row in rows:
        backend.execute(row, controls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
