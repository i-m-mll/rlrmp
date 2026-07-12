#!/usr/bin/env python3
"""Thin CLI for validating and executing authored training matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rlrmp.train.launch import (
    LaunchRuntimeControls,
    execute_authored_training_intent,
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
    execute.add_argument("--driver", choices=("local", "runpod"), default="local")
    execute.add_argument("--runpod-profile", type=Path)

    dry_run = commands.add_parser("dry-run")
    dry_run.add_argument("document", type=Path)
    dry_run.add_argument("--repo-root", type=Path, default=Path.cwd())
    dry_run.add_argument("--row")
    verify = commands.add_parser("verify-resume")
    verify.add_argument("document", type=Path)
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify.add_argument("--row")
    verify.add_argument("--checkpoint-root", type=Path)
    post_run = commands.add_parser("map-post-run")
    post_run.add_argument("run_set_dir", type=Path)
    post_run.add_argument("--repo-root", type=Path, default=Path.cwd())
    post_run.add_argument("--issue", required=True)
    post_run.add_argument("--run-prefix", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the authored launch command."""

    args = build_parser().parse_args(argv)
    if args.command == "map-post-run":
        from rlrmp.train.orchestrated_post_run import map_registered_run_set

        outputs = map_registered_run_set(
            args.run_set_dir,
            repo_root=args.repo_root,
            issue=args.issue,
            run_prefix=args.run_prefix,
        )
        print(json.dumps([str(path) for path in outputs]))
        return 0
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
    evidence = launch_evidence(rows, controls)
    evidence["driver"] = args.driver
    print(json.dumps(evidence, sort_keys=True))
    execute_authored_training_intent(
        launch,
        row=args.row,
        controls=controls,
        driver=args.driver,
        runpod_profile=args.runpod_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
