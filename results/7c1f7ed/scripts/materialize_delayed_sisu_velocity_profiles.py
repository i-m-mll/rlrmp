"""Materialize delayed SISU=1 vs SISU=0 fixed-bank velocity profiles."""

from __future__ import annotations

import runpy
import sys

from rlrmp.paths import REPO_ROOT


RUN_REFS = (
    (
        "7c1f7ed/"
        "delayed_sisu_spectrum__raw_strong_gamma_1p05_radius_lr1e-2_clip5_b64="
        "raw strong gamma-1.05 delayed SISU"
    ),
    (
        "7c1f7ed/"
        "delayed_sisu_spectrum__effective_020a65b_pgd_radius_lr1e-2_clip5_b64="
        "effective 020a65b PGD delayed SISU"
    ),
)


def main() -> None:
    """Run the delayed fixed-bank velocity materializer with 7c1f7ed defaults."""

    script = (
        REPO_ROOT
        / "results"
        / "40e1911"
        / "scripts"
        / "materialize_delayed_timing_hold_lane_velocity_profiles.py"
    )
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--result-experiment",
                "7c1f7ed",
                "--topic",
                "delayed_sisu_velocity_profiles",
                "--sisu-level",
                "0.0",
                "--sisu-level",
                "1.0",
            ]
        )
        for run_ref in RUN_REFS:
            sys.argv.extend(["--run-ref", run_ref])
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
