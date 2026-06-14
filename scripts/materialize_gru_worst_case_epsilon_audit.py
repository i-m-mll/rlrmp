#!/usr/bin/env python
"""Materialize the b8aa38e GRU worst-case full-state epsilon audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.gru_worst_case_epsilon_audit import (
    DEFAULT_RESULT_EXPERIMENT,
    DEFAULT_RUN_IDS,
    DEFAULT_SOURCE_EXPERIMENT,
    materialize_gru_worst_case_epsilon_audit,
)
from rlrmp.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--result-experiment", default=DEFAULT_RESULT_EXPERIMENT)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--n-rollout-trials", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--n-random-baselines", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--budget-level-override",
        choices=("moderate", "strong"),
        help=(
            "Use a named broad-epsilon budget instead of requiring the row to "
            "declare hps.broad_epsilon_training. Useful for auditing nominal or "
            "proprioceptive-feedback rows on the same 8D epsilon channel."
        ),
    )
    parser.add_argument(
        "--budget-scale-override",
        type=float,
        help=(
            "Optional multiplier for --budget-level-override. Use this when "
            "auditing rows against a corrected/shared budget radius."
        ),
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--note-path", type=Path)
    parser.add_argument("--bulk-dir", type=Path)
    parser.add_argument("--preferred-checkpoint-manifest-path", type=Path)
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=("sparse_history", "fixed_bank_manifest"),
        default="sparse_history",
    )
    args = parser.parse_args()

    manifest = materialize_gru_worst_case_epsilon_audit(
        source_experiment=args.source_experiment,
        result_experiment=args.result_experiment,
        run_ids=tuple(args.run_ids or DEFAULT_RUN_IDS),
        labels=args.labels,
        n_rollout_trials=args.n_rollout_trials,
        n_steps=args.n_steps,
        n_restarts=args.n_restarts,
        step_size=args.step_size,
        n_random_baselines=args.n_random_baselines,
        seed=args.seed,
        budget_level_override=args.budget_level_override,
        budget_scale_override=args.budget_scale_override,
        output_path=args.output_path,
        note_path=args.note_path,
        bulk_dir=args.bulk_dir,
        preferred_checkpoint_manifest_path=args.preferred_checkpoint_manifest_path,
        checkpoint_selection_mode=args.checkpoint_selection_mode,
        repo_root=REPO_ROOT,
    )
    print(
        "Wrote worst-case epsilon audit for "
        f"{len(manifest['runs'])} run(s) to "
        f"{args.output_path or REPO_ROOT / 'results' / args.result_experiment / 'notes'}."
    )


if __name__ == "__main__":
    main()
