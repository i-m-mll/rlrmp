"""Write the dd232cd Phase 3 released-stochastic evaluation artifacts."""

from __future__ import annotations

import argparse

from rlrmp.analysis.cs_stochastic_phase3 import Phase3StochasticConfig, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2323)
    parser.add_argument("--motor-covariance-scale", type=float, default=1e-8)
    parser.add_argument("--process-covariance-scale", type=float, default=None)
    parser.add_argument("--signal-dependent-scale", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Phase3StochasticConfig(
        n_trials=args.n_trials,
        seed=args.seed,
        motor_covariance_scale=args.motor_covariance_scale,
        process_covariance_scale=args.process_covariance_scale,
        signal_dependent_scale=args.signal_dependent_scale,
    )
    manifest = write_outputs(config=config)
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['tracked_manifest']}")
    print(f"Wrote {manifest['artifact_npz']}")


if __name__ == "__main__":
    main()
