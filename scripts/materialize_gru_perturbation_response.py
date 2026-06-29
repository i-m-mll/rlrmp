#!/usr/bin/env python
"""Materialize the C&S GRU perturbation-response diagnostic bank."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    DEFAULT_RESULT_EXPERIMENT,
    DEFAULT_RUN_IDS,
    DEFAULT_SOURCE_EXPERIMENT,
    materialize_gru_perturbation_response,
)
from rlrmp.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--result-experiment", default=DEFAULT_RESULT_EXPERIMENT)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--n-rollout-trials", type=int, default=8)
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--no-bulk-arrays", action="store_true")
    parser.add_argument("--bank-mode", choices=("raw", "calibrated"), default="raw")
    parser.add_argument(
        "--calibration-level",
        action="append",
        dest="calibration_levels",
        help="Calibrated-bank severity level to include; repeat for multiple levels.",
    )
    parser.add_argument(
        "--calibration-reach",
        help="Calibrated-bank reach label or fixed reach length in meters.",
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--note-path", type=Path)
    parser.add_argument("--bulk-dir", type=Path)
    parser.add_argument(
        "--feedback-scale-manifest",
        type=Path,
        help=(
            "Evaluation-diagnostics manifest containing controller_feedback_scales. "
            "Required for calibrated force/filter feedback rows."
        ),
    )
    parser.add_argument(
        "--extlqg-physical-dim",
        type=int,
        choices=(6, 8),
        default=8,
        help="Analytical extLQG physical-state dimension to use for comparator rows.",
    )
    args = parser.parse_args()

    manifest = materialize_gru_perturbation_response(
        source_experiment=args.source_experiment,
        result_experiment=args.result_experiment,
        run_ids=tuple(args.run_ids or DEFAULT_RUN_IDS),
        labels=args.labels,
        n_rollout_trials=args.n_rollout_trials,
        bank_mode=args.bank_mode,
        calibration_level=args.calibration_levels,
        calibration_reach=args.calibration_reach,
        evaluate=not args.no_evaluate,
        write_bulk_arrays=not args.no_bulk_arrays,
        output_path=args.output_path,
        note_path=args.note_path,
        bulk_dir=args.bulk_dir,
        feedback_scale_manifest_path=args.feedback_scale_manifest,
        extlqg_physical_dim=args.extlqg_physical_dim,
        repo_root=REPO_ROOT,
    )
    print(
        f"Wrote perturbation-response manifest for {len(manifest['runs'])} run(s) "
        f"with {len(manifest['bank']['perturbations'])} bank row(s)."
    )


if __name__ == "__main__":
    main()
