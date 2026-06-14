"""Materialize temporary standard figures for GRU pilot runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.pipelines.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    materialize_gru_pilot_figures,
)


DEFAULT_EXPERIMENT = "30f2313"
DEFAULT_RUN_IDS = (
    "cs_stochastic_gru__no_hidden_penalty",
    "cs_stochastic_gru__hidden_penalty",
)
DEFAULT_LABELS = ("nn_hidden = 0", "nn_hidden = 1e-5")


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    run_ids = tuple(args.run_id or DEFAULT_RUN_IDS)
    labels = tuple((args.label or DEFAULT_LABELS) if args.run_id is None else (args.label or run_ids))
    summary = materialize_gru_pilot_figures(
        experiment=args.experiment,
        run_ids=run_ids,
        labels=labels,
        output_dir=args.output_dir,
        n_rollout_trials=args.n_rollout_trials,
        include_reference=not args.no_reference,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--label", action="append")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-rollout-trials", type=int, default=DEFAULT_N_ROLLOUT_TRIALS)
    parser.add_argument("--no-reference", action="store_true")
    return parser


if __name__ == "__main__":
    main()
