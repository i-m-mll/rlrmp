#!/usr/bin/env python
"""Materialize paired broad-epsilon attribution diagnostics for C&S GRU rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.gru_broad_epsilon_attribution import (
    DEFAULT_EXPERIMENT,
    DEFAULT_OUTPUT_TAG,
    materialize_broad_epsilon_attribution,
)
from rlrmp.paths import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument("--n-rollout-trials", type=int, default=8)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-gradient-replicates", type=int, default=1)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--final-checkpoints", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = materialize_broad_epsilon_attribution(
        experiment=args.experiment,
        run_ids=args.run_ids,
        output_tag=args.output_tag,
        n_rollout_trials=args.n_rollout_trials,
        max_runs=args.max_runs,
        include_smoke=args.include_smoke,
        use_validation_selected_checkpoints=not args.final_checkpoints,
        max_gradient_replicates=args.max_gradient_replicates,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
