#!/usr/bin/env python
"""Emit a training-run matrix through Feedbax three-layer spec storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from rlrmp.runtime.spec_storage import emit_rlrmp_training_run_spec_storage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Authored matrix JSON to emit.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--custody-root", type=Path, default=Path("_artifacts"))
    parser.add_argument("--dependency-lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--materializer-commit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    source = args.source if args.source.is_absolute() else repo_root / args.source
    output = args.output if args.output.is_absolute() else repo_root / args.output
    custody_root = (
        args.custody_root if args.custody_root.is_absolute() else repo_root / args.custody_root
    )
    dependency_lock = (
        args.dependency_lock
        if args.dependency_lock.is_absolute()
        else repo_root / args.dependency_lock
    )
    commit = (
        args.materializer_commit
        or subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    result = emit_rlrmp_training_run_spec_storage(
        json.loads(source.read_text(encoding="utf-8")),
        repo_root=repo_root,
        authored_path=output,
        custody_root=custody_root,
        materializer_commit=commit,
        dependency_lock_path=dependency_lock,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
