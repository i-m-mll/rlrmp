"""Materialize the standard post-run bundle for GRU runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.gru_postrun_materialization import (
    DEFAULT_OUTPUT_TAG,
    materialize_gru_postrun_analysis,
)
from rlrmp.analysis.gru_pilot_figures import DEFAULT_N_ROLLOUT_TRIALS


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    manifest = materialize_gru_postrun_analysis(
        experiment=args.experiment,
        run_ids=tuple(args.run_id),
        labels=None if args.label is None else tuple(args.label),
        output_tag=args.output_tag,
        use_validation_selected_checkpoints=not args.final_checkpoints,
        fixed_bank_rescore_manifest_path=args.fixed_bank_rescore_manifest,
        include_reference=not args.no_reference,
        n_rollout_trials=args.n_rollout_trials,
        include_objective_comparator=not args.no_objective_comparator,
        include_map_decomposition=not args.no_map_decomposition,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment/issue ID under results/.")
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="Run identifier to materialize. May be passed more than once.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional display label. Pass once per --run-id when provided.",
    )
    parser.add_argument(
        "--output-tag",
        default=DEFAULT_OUTPUT_TAG,
        help=(
            "Filename/path suffix for generated summaries. Defaults to "
            f"{DEFAULT_OUTPUT_TAG!r}."
        ),
    )
    parser.add_argument(
        "--final-checkpoints",
        action="store_true",
        help=(
            "Override the default validation-selected per-replicate checkpoint "
            "policy and use final checkpoints."
        ),
    )
    parser.add_argument(
        "--fixed-bank-rescore-manifest",
        type=Path,
        help=(
            "Optional fixed-bank checkpoint rescore manifest to prefer for "
            "validation-selected materialization. Non-materialized manifests are "
            "recorded as provenance and fall back to sparse logged validation."
        ),
    )
    parser.add_argument("--n-rollout-trials", type=int, default=DEFAULT_N_ROLLOUT_TRIALS)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument(
        "--no-objective-comparator",
        action="store_true",
        help="Skip the optional objective-comparator hook.",
    )
    parser.add_argument(
        "--no-map-decomposition",
        action="store_true",
        help="Skip the optional map-error decomposition hook.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


if __name__ == "__main__":
    main()
