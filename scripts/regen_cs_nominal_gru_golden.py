#!/usr/bin/env python
"""Regenerate the compact C&S nominal-GRU configuration identity fixture."""

from __future__ import annotations

import json
from pathlib import Path

import jax

from rlrmp.run_spec_identity import (
    run_spec_payload_identity_sha256,
    run_spec_semantic_checks,
)
from rlrmp.train.cs_nominal_gru import (
    build_parser,
    cs_nominal_gru_config_from_args,
    write_run_spec,
)


FIXTURE_PATH = Path("tests/fixtures/cs_nominal_gru_config_identity.json")
CASE_ARGV = {
    "default": ["--dry-run"],
    "delayed_full_qrf": [
        "--dry-run",
        "--target-relative-multitarget",
        "--delayed-reach",
        "--loss-objective",
        "full_analytical_qrf",
        "--controller-lr",
        "0.003",
        "--batch-size",
        "64",
        "--gradient-clip-norm",
        "5",
    ],
    "target_h0_pgd_sisu": [
        "--dry-run",
        "--target-relative-multitarget",
        "--initial-hidden-encoder",
        "--broad-epsilon-pgd-training",
        "--no-broad-epsilon-reach-scaling",
        "--broad-epsilon-pgd-budget-schedule",
        "sisu_energy_fraction",
        "--broad-epsilon-pgd-sisu-max-radius",
        "0.004545500088363065",
        "--broad-epsilon-pgd-sisu-max-radius-source",
        "effective_020a65b_pgd_training_radius",
        "--n-train-batches",
        "120",
        "--seed",
        "43",
    ],
}


def main() -> None:
    parser = build_parser()
    cases = {}
    for name, argv in CASE_ARGV.items():
        args = parser.parse_args(argv)
        with jax.enable_x64(False):
            config = cs_nominal_gru_config_from_args(args)
            payload = write_run_spec(args)["run_spec"]
        cases[name] = {
            "argv": argv,
            "parsed_args": config.model_dump(mode="python"),
            "stable_run_spec_identity_sha256": run_spec_payload_identity_sha256(payload),
            "semantic_checks": run_spec_semantic_checks(payload),
        }
    FIXTURE_PATH.write_text(
        json.dumps({"cases": cases}, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {FIXTURE_PATH} ({len(FIXTURE_PATH.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
