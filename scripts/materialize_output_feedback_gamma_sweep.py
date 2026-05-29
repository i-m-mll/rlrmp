"""Write the 97604a8 output-feedback robust gamma-sweep artifacts."""

from __future__ import annotations

import argparse

from rlrmp.analysis.output_feedback import (
    GAMMA_FEASIBILITY_SWEEP_ISSUE_ID,
    GAMMA_SWEEP_FACTORS,
    write_gamma_sweep_outputs,
)


def _parse_factors(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize gamma-penalized output-feedback robust sweep artifacts."
    )
    parser.add_argument(
        "--gamma-factors",
        default=",".join(str(factor) for factor in GAMMA_SWEEP_FACTORS),
        help="Comma-separated gamma/gamma_star ratios to sweep.",
    )
    args = parser.parse_args()
    manifest = write_gamma_sweep_outputs(
        issue_id=GAMMA_FEASIBILITY_SWEEP_ISSUE_ID,
        gamma_factors=_parse_factors(args.gamma_factors),
    )
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['tracked_manifest']}")
    print(f"Wrote {manifest['artifact_npz']}")


if __name__ == "__main__":
    main()
