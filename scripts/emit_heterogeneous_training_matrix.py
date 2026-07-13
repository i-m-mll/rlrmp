#!/usr/bin/env python
"""Emit the compact heterogeneous C&S matrix through governed storage."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
from typing import Any

from rlrmp.runtime.spec_storage import emit_rlrmp_training_run_spec_storage
from rlrmp.train.heterogeneous_training_matrix import ARCHITECTURES, author_training_run_matrix
from rlrmp.train.training_configs import CsNominalGruConfig


REQUIRED_ARCHITECTURE_FIELD = "controller_architecture"
REQUIRED_LOWERING_CONTRACT = "rlrmp.heterogeneous_cs_architecture.v1"
REQUIRED_CONTRACT_SYMBOL = "RLRMP_TRAINING_ARCHITECTURE_CONTRACT"
REQUIRED_ARCHITECTURES_SYMBOL = "RLRMP_TRAINING_ARCHITECTURES"


def require_heterogeneous_row_lowering_contract() -> None:
    """Fail closed until generic row lowering owns architecture selection."""

    try:
        lowering = importlib.import_module("rlrmp.train.matrix_lowering")
    except ModuleNotFoundError:
        lowering = None
    declared_contract = (
        None if lowering is None else getattr(lowering, REQUIRED_CONTRACT_SYMBOL, None)
    )
    declared_architectures = (
        () if lowering is None else getattr(lowering, REQUIRED_ARCHITECTURES_SYMBOL, ())
    )
    if (
        REQUIRED_ARCHITECTURE_FIELD not in CsNominalGruConfig.model_fields
        or declared_contract != REQUIRED_LOWERING_CONTRACT
        or set(declared_architectures) != set(ARCHITECTURES)
    ):
        raise RuntimeError(
            "blocked by 5816bf0 row lowering: compact heterogeneous emission requires "
            "CsNominalGruConfig.controller_architecture with values "
            "gru|time_constrained_free_gain|linear_recurrence, plus registered lowering "
            f"contract {REQUIRED_LOWERING_CONTRACT} declared as "
            f"matrix_lowering.{REQUIRED_CONTRACT_SYMBOL}, with the exact supported tuple in "
            f"matrix_lowering.{REQUIRED_ARCHITECTURES_SYMBOL}, dispatching each value to its "
            "registered provider without accepting compiled graph/task/method/worker payloads"
        )


def emit_heterogeneous_training_matrix(
    *,
    base_intent_path: Path,
    output_path: Path,
    issue: str,
    repo_root: Path,
    custody_root: Path,
    dependency_lock_path: Path,
    materializer_commit: str,
) -> Any:
    """Route compact authoring through RLRMP's three-layer storage entry point."""

    require_heterogeneous_row_lowering_contract()
    base_intent = json.loads(base_intent_path.read_text(encoding="utf-8"))
    matrix = author_training_run_matrix(
        base_intent,
        issue=issue,
        base_ref=base_intent_path,
        repo_root=repo_root,
    )
    return emit_rlrmp_training_run_spec_storage(
        matrix,
        repo_root=repo_root,
        authored_path=output_path,
        custody_root=custody_root,
        materializer_commit=materializer_commit,
        dependency_lock_path=dependency_lock_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-intent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--custody-root", type=Path, default=Path("_artifacts"))
    parser.add_argument("--dependency-lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--materializer-commit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    base_intent = (
        args.base_intent if args.base_intent.is_absolute() else repo_root / args.base_intent
    )
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
    result = emit_heterogeneous_training_matrix(
        base_intent_path=base_intent,
        output_path=output,
        issue=args.issue,
        repo_root=repo_root,
        custody_root=custody_root,
        dependency_lock_path=dependency_lock,
        materializer_commit=commit,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
