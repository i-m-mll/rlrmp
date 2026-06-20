"""Materialize issue 87424a4 steady-state GRU feedback perturbation banks."""

from __future__ import annotations

from pathlib import Path
import argparse

from rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank import (
    ComparisonSpec,
    identity_condition,
    materialize_steady_state_comparisons,
    sisu_condition,
)
from rlrmp.paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    """Parse materializer parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--pulse-duration-steps", type=int, default=5)
    parser.add_argument("--n-rollout-trials", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Run the requested local steady-state perturbation comparisons."""

    args = parse_args()
    comparisons = (
        ComparisonSpec(
            comparison_id="delayed_sisu_effective_020a65b",
            title="Delayed SISU effective 020a65b PGD-radius row",
            kind="sisu",
            source_experiment="7c1f7ed",
            run_id="delayed_sisu_spectrum__effective_020a65b_pgd_radius_lr1e-2_clip5_b64",
            conditions=(
                sisu_condition(0.0, label="SISU=0"),
                sisu_condition(1.0, label="SISU=1"),
            ),
            delayed=True,
        ),
        ComparisonSpec(
            comparison_id="undelayed_targetfix_sisu_effective_020a65b",
            title="Undelayed target-fix SISU effective 020a65b PGD-radius row",
            kind="sisu",
            source_experiment="e4800d6",
            run_id=(
                "cs_gru_h0_sisu_spectrum_targetfix__"
                "effective_020a65b_pgd_radius_lr3e-3_clip5_b64"
            ),
            conditions=(
                sisu_condition(0.0, label="SISU=0"),
                sisu_condition(1.0, label="SISU=1"),
            ),
            delayed=False,
        ),
        ComparisonSpec(
            comparison_id="matched_020a65b_no_pgd_vs_pgd",
            title="Matched 020a65b no-PGD vs PGD feedback rows",
            kind="pgd",
            source_experiment="020a65b",
            run_id="target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64",
            conditions=(
                identity_condition("no_pgd", "No PGD"),
                identity_condition(
                    "pgd",
                    "PGD",
                    run_id=(
                        "target_relative_multitarget_fullqrf_warmcos__"
                        "proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64"
                    ),
                ),
            ),
            delayed=False,
        ),
    )
    materialize_steady_state_comparisons(
        comparisons=comparisons,
        result_experiment="87424a4",
        n_rollout_trials=args.n_rollout_trials,
        pulse_duration_steps=args.pulse_duration_steps,
        repo_root=Path(REPO_ROOT),
    )


if __name__ == "__main__":
    main()
