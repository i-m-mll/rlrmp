#!/usr/bin/env python
"""Materialize C&S perturbation open-loop calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.gru_perturbation_calibration import (
    DEFAULT_AMPLITUDE_FACTORS,
    DEFAULT_RESULT_EXPERIMENT,
    materialize_perturbation_open_loop_calibration,
)
from rlrmp.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-experiment", default=DEFAULT_RESULT_EXPERIMENT)
    parser.add_argument("--amplitude-factor", action="append", type=float, dest="factors")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--note-path", type=Path)
    parser.add_argument("--regeneration-spec-path", type=Path)
    args = parser.parse_args()
    manifest = materialize_perturbation_open_loop_calibration(
        amplitude_factors=tuple(args.factors or DEFAULT_AMPLITUDE_FACTORS),
        result_experiment=args.result_experiment,
        output_path=args.output_path,
        note_path=args.note_path,
        regeneration_spec_path=args.regeneration_spec_path,
        repo_root=REPO_ROOT,
    )
    print(json.dumps({"rows": len(manifest["rows"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
