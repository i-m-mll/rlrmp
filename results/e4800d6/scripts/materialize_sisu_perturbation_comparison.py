"""Materialize SISU=1 vs SISU=0 perturbation-class comparisons for e4800d6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.pipelines.sisu_perturbation_comparison import (
    DEFAULT_N_ROLLOUT_TRIALS,
    DEFAULT_OUTPUT_STEM,
    materialize_sisu_perturbation_comparison,
)
from rlrmp.paths import REPO_ROOT


DEFAULT_RUN_IDS = (
    "cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64",
    "cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64",
)
DEFAULT_LABELS = (
    "raw strong gamma-1.05 targetfix",
    "effective 020a65b PGD targetfix",
)
DEFAULT_FEEDBACK_SCALE_MANIFEST = (
    REPO_ROOT
    / "results"
    / "e4800d6"
    / "notes"
    / "gru_evaluation_diagnostics_h0_sisu_spectrum_targetfix_two_rows_validation_selected.json"
)


def main() -> None:
    """Run the targetfix SISU perturbation-class materializer."""

    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = materialize_sisu_perturbation_comparison(
        source_experiment=args.source_experiment,
        result_experiment=args.result_experiment,
        run_ids=tuple(args.run_ids),
        labels=tuple(args.labels),
        n_rollout_trials=args.n_rollout_trials,
        output_stem=args.output_stem,
        bank_mode=args.bank_mode,
        calibration_level=args.calibration_level,
        calibration_reach=args.calibration_reach,
        feedback_scale_manifest_path=args.feedback_scale_manifest_path,
        preferred_checkpoint_manifest_path=args.preferred_checkpoint_manifest_path,
        checkpoint_selection_mode=args.checkpoint_selection_mode,
        repo_root=repo_root,
    )
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-experiment", default="e4800d6")
    parser.add_argument("--result-experiment", default="e4800d6")
    parser.add_argument("--run-id", dest="run_ids", action="append", default=None)
    parser.add_argument("--label", dest="labels", action="append", default=None)
    parser.add_argument("--n-rollout-trials", type=int, default=DEFAULT_N_ROLLOUT_TRIALS)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--bank-mode", choices=("raw", "calibrated"), default="calibrated")
    parser.add_argument("--calibration-level", default=None)
    parser.add_argument("--calibration-reach", default=None)
    parser.add_argument(
        "--feedback-scale-manifest-path",
        type=Path,
        default=DEFAULT_FEEDBACK_SCALE_MANIFEST,
    )
    parser.add_argument("--preferred-checkpoint-manifest-path", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=("sparse_history", "fixed_bank_manifest"),
        default="sparse_history",
    )
    args = parser.parse_args()
    args.run_ids = args.run_ids or list(DEFAULT_RUN_IDS)
    args.labels = args.labels or list(DEFAULT_LABELS)
    if len(args.run_ids) != len(args.labels):
        raise SystemExit("--run-id and --label must be passed the same number of times")
    return args


if __name__ == "__main__":
    main()
