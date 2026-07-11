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


def main() -> None:
    """Recompute parsed_args, semantic checks, and identity hashes for all cases.

    The per-case ``argv`` lists are read from the existing fixture; they are the
    stable case identities. To add or change a case, edit the ``argv`` field in
    the fixture, then rerun this script to fill in the derived fields.
    """
    parser = build_parser()
    existing = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = {}
    for name, case in existing["cases"].items():
        argv = list(case["argv"])
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
