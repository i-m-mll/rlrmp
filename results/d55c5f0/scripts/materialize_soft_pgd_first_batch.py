"""Materialize the first soft-constraint PGD run specs."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from rlrmp.train.cs_nominal_gru import (
    _args_values_from_run_spec,
    build_parser,
    write_run_spec,
)
from rlrmp.train.cs_perturbation_training import BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE

ISSUE = "d55c5f0"
BASE_ISSUE = "c92ebd8"
BASE_RUN = "moderate_pgd_ofb1p4"
BASE_SPEC = Path("results") / BASE_ISSUE / "runs" / f"{BASE_RUN}.json"
GAMMA_STAR = 9166.831285473823
PENALTY_SCALE_C = 1.0
SAFETY_CAP_RADIUS_15CM = 0.004545011406169036
SAFETY_CAP_SOURCE = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"

ROWS: tuple[tuple[str, float], ...] = (
    ("soft_pgd_ofb1p05", 1.05),
    ("soft_pgd_ofb1p4", 1.4),
    ("soft_pgd_ofb1p8", 1.8),
)


def _base_values() -> dict[str, Any]:
    parser = build_parser()
    values = vars(parser.parse_args([])).copy()
    payload = json.loads(BASE_SPEC.read_text(encoding="utf-8"))
    values.update(_args_values_from_run_spec(payload))
    return values


def _row_args(base_values: dict[str, Any], run: str, gamma_factor: float) -> Namespace:
    gamma = GAMMA_STAR * float(gamma_factor)
    values = dict(base_values)
    values.update(
        {
            "issue": ISSUE,
            "output_dir": str(Path("_artifacts") / ISSUE / "runs" / run),
            "spec_dir": str(Path("results") / ISSUE / "runs" / run),
            "full_train": False,
            "dry_run": False,
            "smoke": False,
            "broad_epsilon_pgd_training": True,
            "broad_epsilon_pgd_objective": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            "broad_epsilon_pgd_energy_gamma_star": GAMMA_STAR,
            "broad_epsilon_pgd_energy_gamma_factor": float(gamma_factor),
            "broad_epsilon_pgd_energy_gamma": gamma,
            "broad_epsilon_pgd_energy_penalty_scale": PENALTY_SCALE_C,
            "broad_epsilon_pgd_energy_lambda": PENALTY_SCALE_C * gamma**2,
            "broad_epsilon_pgd_safety_cap_15cm": SAFETY_CAP_RADIUS_15CM,
            "broad_epsilon_pgd_safety_cap_source": SAFETY_CAP_SOURCE,
        }
    )
    return Namespace(**values)


def main() -> int:
    base_values = _base_values()
    materialized = []
    for run, gamma_factor in ROWS:
        result = write_run_spec(_row_args(base_values, run, gamma_factor))
        materialized.append(
            {
                "run": run,
                "gamma_factor": gamma_factor,
                "run_spec_path": result["run_spec_path"],
                "graph_manifest_path": result["graph_manifest_path"],
            }
        )
    print(json.dumps({"materialized": materialized}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
