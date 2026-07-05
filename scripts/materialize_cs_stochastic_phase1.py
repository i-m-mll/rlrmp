"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Write the Phase 1 released stochastic C&S evaluation artifacts."""

from __future__ import annotations

import argparse

from rlrmp.analysis.math.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    CSReleasedStochasticNoiseConfig,
)
from rlrmp.analysis.pipelines.cs_stochastic_phase1 import DEFAULT_SEEDS, ISSUE_ID, write_outputs
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DISCRETIZATION_CHOICES,
    metadata_cli_help,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discretization",
        choices=DISCRETIZATION_CHOICES,
        default=DEFAULT_DISCRETIZATION,
        help=metadata_cli_help(),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_SEEDS),
        help="Monte Carlo seeds to materialize; defaults to 12 deterministic seeds.",
    )
    parser.add_argument(
        "--motor-covariance-scale",
        type=float,
        default=DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.motor_covariance_scale,
    )
    parser.add_argument(
        "--process-covariance-scale",
        type=float,
        default=DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.process_covariance_scale,
    )
    parser.add_argument(
        "--signal-dependent-scale",
        type=float,
        default=DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.signal_dependent_scale,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_outputs(
        issue_id=ISSUE_ID,
        discretization=args.discretization,
        seeds=tuple(args.seeds),
        noise_config=CSReleasedStochasticNoiseConfig(
            motor_covariance_scale=args.motor_covariance_scale,
            process_covariance_scale=args.process_covariance_scale,
            signal_dependent_scale=args.signal_dependent_scale,
        ),
    )
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['tracked_manifest']}")
    print(f"Wrote {manifest['artifact_npz']}")


if __name__ == "__main__":
    main()
