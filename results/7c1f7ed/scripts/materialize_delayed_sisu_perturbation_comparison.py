"""Materialize delayed SISU=1 vs SISU=0 perturbation-class comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.pipelines.sisu_perturbation_comparison import (
    DEFAULT_N_ROLLOUT_TRIALS,
    materialize_sisu_perturbation_comparison,
)
from rlrmp.paths import REPO_ROOT


RUN_IDS = (
    "delayed_sisu_spectrum__raw_strong_gamma_1p05_radius_lr1e-2_clip5_b64",
    "delayed_sisu_spectrum__effective_020a65b_pgd_radius_lr1e-2_clip5_b64",
)
LABELS = (
    "raw strong gamma-1.05 delayed SISU",
    "effective 020a65b PGD delayed SISU",
)
DEFAULT_OUTPUT_STEM = "delayed_sisu_perturbation_class_comparison"
DEFAULT_FEEDBACK_SCALE_MANIFEST = (
    REPO_ROOT
    / "results"
    / "7c1f7ed"
    / "notes"
    / "gru_evaluation_diagnostics_delayed_sisu_final.json"
)


def main() -> None:
    """Run the delayed SISU perturbation comparison materializer."""

    args = parse_args()
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
        repo_root=args.repo_root.resolve(),
    )
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-experiment", default="7c1f7ed")
    parser.add_argument("--result-experiment", default="7c1f7ed")
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
    args.run_ids = args.run_ids or list(RUN_IDS)
    args.labels = args.labels or list(LABELS)
    if len(args.run_ids) != len(args.labels):
        raise SystemExit("--run-id and --label must be passed the same number of times")
    return args


if __name__ == "__main__":
    main()
