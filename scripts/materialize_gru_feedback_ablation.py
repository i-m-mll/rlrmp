#!/usr/bin/env python
"""Materialize feedback-ablation diagnostics for C&S GRU checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.gru_feedback_ablation import (
    DEFAULT_RESULT_EXPERIMENT,
    DEFAULT_RUN_IDS,
    DEFAULT_SCOPE,
    DEFAULT_SOURCE_EXPERIMENT,
    materialize_gru_feedback_ablation,
)
from rlrmp.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--result-experiment", default=DEFAULT_RESULT_EXPERIMENT)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--n-rollout-trials", type=int, default=4)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--note-path", type=Path)
    args = parser.parse_args()

    manifest = materialize_gru_feedback_ablation(
        source_experiment=args.source_experiment,
        result_experiment=args.result_experiment,
        scope=args.scope,
        run_ids=tuple(args.run_ids or DEFAULT_RUN_IDS),
        labels=None if args.labels is None else tuple(args.labels),
        n_rollout_trials=args.n_rollout_trials,
        output_path=args.output_path,
        note_path=args.note_path,
        repo_root=REPO_ROOT,
    )
    print(f"Wrote feedback-ablation diagnostic for {len(manifest['runs'])} run(s).")


if __name__ == "__main__":
    main()
