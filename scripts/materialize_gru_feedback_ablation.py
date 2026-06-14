#!/usr/bin/env python
"""Materialize feedback-ablation diagnostics for C&S GRU checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.gru_feedback_ablation import (
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
    parser.add_argument(
        "--bank-mode",
        choices=("raw", "calibrated"),
        default="raw",
        help="Perturbation-bank mode used for the representative feedback bins.",
    )
    parser.add_argument(
        "--calibration-level",
        action="append",
        help=(
            "Calibrated-bank level to include. May be passed more than once; "
            "defaults to all levels when omitted."
        ),
    )
    parser.add_argument(
        "--calibration-reach",
        help="Calibrated-bank reach selector, e.g. canonical_15cm or a float in meters.",
    )
    parser.add_argument(
        "--feedback-selection-level",
        default="small",
        help="Calibrated severity level used for feedback checkpoint rescoring.",
    )
    parser.add_argument(
        "--preferred-checkpoint-manifest",
        type=Path,
        help="Optional fixed-bank-style manifest selecting checkpoints to load.",
    )
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
        bank_mode=args.bank_mode,
        calibration_level=args.calibration_level,
        calibration_reach=args.calibration_reach,
        feedback_selection_level=args.feedback_selection_level,
        preferred_checkpoint_manifest_path=args.preferred_checkpoint_manifest,
        output_path=args.output_path,
        note_path=args.note_path,
        repo_root=REPO_ROOT,
    )
    print(f"Wrote feedback-ablation diagnostic for {len(manifest['runs'])} run(s).")


if __name__ == "__main__":
    main()
